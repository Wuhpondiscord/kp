import importlib.util,math,pickle,unittest
import numpy as np
HAS_TORCH=importlib.util.find_spec('torch') is not None
if HAS_TORCH:
    from helper.joint_research import EventNeuralPredictor,LogisticMarketOffset,MarketOffset,torch
    from helper.observation_calibration import calibrate

@unittest.skipUnless(HAS_TORCH,'Optional PyTorch not installed')
class EventJointTests(unittest.TestCase):
    def setup_model(self):
        torch.set_num_threads(2)
        pairs=[dict(event=str(i),station='NYC',settled_max=75+i%2,observed_max=75,settled_at='2025-01-02T12:00:00Z',complete_day=True,target_source='kalshi_expiration_value') for i in range(80)]
        m=EventNeuralPredictor(calibrate(pairs,{'test':'synthetic'}),residual_penalty=.1)
        m.mean=np.zeros(12);m.scale=np.ones(12)
        rows=[]
        for e in range(2):
            for i,(lo,hi,p) in enumerate([(-math.inf,72,.15),(72,78,.4),(78,math.inf,.65)]):
                rows.append(dict(series='KXHIGHNY',ticker=f'e{e}-{i}',event=f'e{e}',at='2026-04-04T15:00:00Z',group='2026-04-04',remaining_forecast_max=75,remaining_hours=14,gfs_remaining_max=74,ecmwf_remaining_max=76,model_disagreement_f=2,observation_innovation_f=-1,observed=True,observed_trend=1,observed_temperature=70,observed_max=71,observation_age_hours=1,p=p,spread=.02,lower=lo,upper=hi,y=int(i==1)))
        return m,rows,np.zeros((len(rows),2,16))

    def test_initial_reference_is_normalized_price_not_normalized_odds(self):
        m,r,x=self.setup_model();p=m.predict(r,x)
        np.testing.assert_allclose(p,np.tile(np.array([.15,.4,.65])/1.2,2),atol=1e-12)
        self.assertIs(MarketOffset,LogisticMarketOffset)

    def test_categorical_training_and_roundtrip_preserve_event_mass(self):
        m,r,x=self.setup_model();m.fit(r,x,epochs=25)
        p=m.predict(r,x)
        self.assertTrue(all(np.isfinite(t['loss']) for t in m.trace))
        self.assertLess(m.trace[-1]['loss'],m.trace[0]['loss'])
        for a in [p[:3],p[3:]]:self.assertAlmostEqual(float(sum(a)),1,places=12)
        order=[4,0,5,2,1,3]
        np.testing.assert_allclose(m.predict([r[i] for i in order],x[order]),p[order],atol=1e-12)
        np.testing.assert_array_equal(p,pickle.loads(pickle.dumps(m)).predict(r,x))

    def test_missing_brackets_bad_targets_and_inconsistent_weather_rejected(self):
        m,r,x=self.setup_model()
        with self.assertRaises(ValueError):m.predict(r[1:],x[1:])
        bad=[dict(a,y=1) for a in r]
        with self.assertRaises(ValueError):m.fit(bad,x,epochs=1)
        for key,value in [('observed_max',80),('remaining_forecast_max',80),('observation_age_hours',2)]:
            bad=[dict(a) for a in r];bad[0][key]=value
            with self.assertRaises(ValueError):m.predict(bad,x)
        altered=x.copy();altered[0,0,0]=1
        with self.assertRaises(ValueError):m.predict(r,altered)

    def test_softmax_gradient_matches_finite_difference(self):
        m,r,x=self.setup_model();batch=m.batch(r,x)
        weight=m.network.market[-1].weight
        with torch.no_grad():weight.fill_(.1)
        def loss():return -torch.log(m.network(*batch)[1][1])
        loss().backward();analytic=float(weight.grad[0,0]);initial=float(weight[0,0].detach());eps=1e-5
        with torch.no_grad():
            weight[0,0]=initial+eps;plus=float(loss())
            weight[0,0]=initial-eps;minus=float(loss())
            weight[0,0]=initial
        self.assertAlmostEqual(analytic,(plus-minus)/(2*eps),places=7)

    def test_other_brackets_affect_probability_but_other_events_do_not(self):
        m,r,x=self.setup_model();p=m.predict(r,x)
        changed=[dict(a) for a in r];changed[0]['p']=.8
        new=m.predict(changed,x)
        self.assertNotEqual(new[1],p[1]);np.testing.assert_array_equal(new[3:],p[3:])
        # Inference cannot consult settlement labels.
        np.testing.assert_array_equal(m.predict([dict(a,y=1-a['y']) for a in r],x),p)
