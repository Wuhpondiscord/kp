import importlib.util,math,unittest
import numpy as np
HAS_TORCH=importlib.util.find_spec('torch') is not None
if HAS_TORCH:
    from helper.joint_research import SmallJoint,NeuralPredictor,MarketOffset,features,sequence,torch
    from helper.observation_calibration import calibrate
    from test_forecast_archive import fixture

@unittest.skipUnless(HAS_TORCH,'Optional PyTorch research dependency not installed')
class JointResearchTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2)
    def row(self):
        return dict(series='KXHIGHNY',ticker='KXHIGHNY-26APR04',at='2026-04-04T15:00:00Z',remaining_forecast_max=75,
            forecast_complete=True,remaining_hours=14,gfs_remaining_max=74,ecmwf_remaining_max=76,
            model_disagreement_f=2,observation_innovation_f=-1,observed=True,observed_trend=1,
            observed_temperature=70,observed_max=71,observation_age_hours=1,p=.3,spread=.02,
            event='KXHIGHNY-26APR04',group='2026-04-04',lower=72,upper=78,y=1)
    def calibration(self):
        pairs=[dict(event=str(i),station='NYC',settled_max=75+i%2,observed_max=75,
            settled_at='2025-01-02T12:00:00Z',complete_day=True,target_source='kalshi_expiration_value') for i in range(80)]
        return calibrate(pairs,{'test':'synthetic'})
    def test_features_exclude_market_and_labels(self):
        r=self.row();changed=dict(r,p=.9,spread=.7,y=0,settled_max=-100)
        np.testing.assert_array_equal(features([r]),features([changed]))
    def test_sequence_uses_only_issued_forecasts(self):
        r=self.row();s=sequence(fixture(),r)
        self.assertEqual(s.shape,(2,16));self.assertTrue(np.isfinite(s).all())
        f=fixture();f['run_initialized_at']='2026-04-04T12:00:00Z'
        with self.assertRaises(ValueError):sequence(f,r)
        f=fixture();f['hourly']={} # An unused top-level field cannot supply data.
        np.testing.assert_array_equal(sequence(f,r),s)
    def test_market_only_does_not_read_weather(self):
        m=MarketOffset();m.coef=np.zeros(6);r=self.row()
        self.assertAlmostEqual(m.predict([r])[0],r['p'])
        self.assertEqual(m.predict([r])[0],m.predict([dict(r,remaining_forecast_max=100,observed_max=0)])[0])
    def test_all_parameters_registered_and_weather_market_separation(self):
        m=NeuralPredictor(self.calibration(),temporal=True,fusion=True);m.mean=np.zeros(12);m.scale=np.ones(12)
        rows=[self.row(),dict(self.row(),p=.8)];paths=np.zeros((2,2,16));before=sum(p.numel() for p in m.network.parameters())
        weather,p=m.network(*m.batch(rows,paths))
        self.assertEqual(before,sum(p.numel() for p in m.network.parameters()))
        self.assertAlmostEqual(float(weather[0].detach()),float(weather[1].detach()))
        np.testing.assert_allclose(p.detach().numpy(),[.3,.8])
    def test_infinite_bracket_gradients_and_mass(self):
        m=NeuralPredictor(self.calibration(),fusion=True);m.mean=np.zeros(12);m.scale=np.ones(12)
        rows=[dict(self.row(),lower=a,upper=b) for a,b in [(-math.inf,72),(72,78),(78,math.inf)]]
        weather,p=m.network(*m.batch(rows,np.zeros((3,2,16))))
        self.assertAlmostEqual(float(weather.sum().detach()),1,places=8)
        (-torch.log(weather).mean()+p.mean()).backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in m.network.parameters()))
    def test_training_is_finite_and_reproducible(self):
        rows=[dict(self.row(),event=str(i),group=f'2026-04-{i+1:02}',y=1) for i in range(4)]
        paths=np.zeros((4,2,16));results=[]
        for _ in range(2):
            m=NeuralPredictor(self.calibration(),fusion=True,seed=10).fit(rows,paths,epochs=5)
            results.append(m.predict(rows,paths));self.assertTrue(np.isfinite(results[-1]).all())
        np.testing.assert_array_equal(*results)
    def test_residual_penalty_reduces_departure_from_market(self):
        rows=[dict(self.row(),event=str(i),group=f'2026-04-{i+1:02}',y=1) for i in range(4)]
        paths=np.zeros((4,2,16));departures=[]
        for penalty in (0.,10.):
            m=NeuralPredictor(self.calibration(),fusion=True,seed=10,weather_loss_weight=.1,residual_penalty=penalty).fit(rows,paths,epochs=60)
            p=m.predict(rows,paths)
            self.assertTrue(np.isfinite(p).all())
            departures.append(float(np.mean(abs(p-.3))))
        self.assertLess(departures[1],departures[0])
    def test_invalid_loss_configuration_rejected(self):
        for kwargs in ({'weather_loss_weight':0},{'weather_loss_weight':math.nan},{'residual_penalty':-1},{'residual_penalty':math.inf},{'penalty_deadzone':-.1},{'penalty_deadzone':math.nan},{'average_last':-1}):
            with self.assertRaises(ValueError):NeuralPredictor(self.calibration(),fusion=True,**kwargs)
    def test_last_checkpoint_average_matches_final_weights(self):
        rows=[self.row()];paths=np.zeros((1,2,16));models=[]
        for window in (0,1):
            models.append(NeuralPredictor(self.calibration(),fusion=True,seed=11,average_last=window).fit(rows,paths,epochs=4))
        np.testing.assert_allclose(models[0].predict(rows,paths),models[1].predict(rows,paths),rtol=0,atol=1e-12)
        self.assertEqual(models[1].averaged_checkpoints,1)
        with self.assertRaises(ValueError):NeuralPredictor(self.calibration(),average_last=5).fit(rows,paths,epochs=4)
    def test_deadzone_and_averaging_training_remain_finite(self):
        rows=[self.row()];paths=np.zeros((1,2,16))
        m=NeuralPredictor(self.calibration(),fusion=True,residual_penalty=1,penalty_deadzone=.1,average_last=3).fit(rows,paths,epochs=5)
        self.assertEqual(m.averaged_checkpoints,3)
        self.assertTrue(np.isfinite(m.predict(rows,paths)).all())
    def test_warm_start_frozen_backbone_preserves_encoder_and_normalization(self):
        rows=[self.row()];paths=np.zeros((1,2,16))
        m=NeuralPredictor(self.calibration(),fusion=True).fit(rows,paths,epochs=3)
        before={n:p.detach().clone() for n,p in m.network.named_parameters()};mean=m.mean.copy()
        m.fit(rows,paths,epochs=4,warm_start=True,freeze_backbone=True,learning_rate=.001)
        np.testing.assert_array_equal(m.mean,mean)
        for name,p in m.network.named_parameters():
            if not name.startswith('market.'):self.assertTrue(torch.equal(before[name],p))
        self.assertTrue(any(not torch.equal(before[n],p) for n,p in m.network.named_parameters() if n.startswith('market.')))
