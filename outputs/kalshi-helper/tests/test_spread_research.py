import json,math,tempfile,unittest
from unittest.mock import patch
import numpy as np
from scipy.special import ndtr
from helper.event_weather import EventMaximum
from helper.weather_research import RemainingMaximum
from helper.observation_calibration import calibrate,block_uncertainty
from helper.spread_research import compare,calibration_metrics

class SpreadTests(unittest.TestCase):
    def calibration(self):
        pairs=[dict(event=str(i),station='NYC' if i<80 else 'DEN',settled_max=75+(i%2)*(1 if i<80 else 3),
            observed_max=75,settled_at='2025-01-02T12:00:00Z',complete_day=True,target_source='kalshi_expiration_value') for i in range(160)]
        return dict(calibrate(pairs,{'fixture':'synthetic'}),pooling='station')

    def base(self):return dict(remaining_forecast_max=75,remaining_hours=18,forecast_complete=True,series='KXHIGHNY',observed=False,model_disagreement_f=0)

    def test_bias_center_and_station_dispatch(self):
        c=self.calibration();m=EventMaximum(c);m.theta=np.array([-10,0,0,0,0,0,math.log(.5)])
        for series,station in [('KXHIGHNY','NYC'),('KXHIGHDEN','DEN')]:
            s=c['by_station'][station]
            row=dict(gfs=50,ecmwf=50,hours_left=12,series=series,observed=True,observed_max=75,lower=-math.inf,upper=75)
            expected=ndtr(-s['bias_f']/s['sigma_f'])
            self.assertAlmostEqual(m.predict([row])[0],expected)
            self.assertLess(expected,.5)

    def test_spread_monotone_and_legacy_zero_slope(self):
        rows=[dict(self.base(),model_disagreement_f=d,lower=72,upper=78) for d in (0,5,10,100)]
        m=RemainingMaximum(remaining_spread='disagreement');n=m.inputs(rows)[1].shape[1]
        m.theta=np.r_[np.zeros(n),math.log(2),.5]
        mu,sigma=m.distribution(rows)
        np.testing.assert_allclose(mu,75)
        np.testing.assert_allclose(sigma,2*np.exp([0,.5,1,2]))
        m.theta[-1]=0
        old=RemainingMaximum();old.theta=m.theta[:-1]
        np.testing.assert_array_equal(m.predict(rows),old.predict(rows))
        del old.remaining_spread
        np.testing.assert_array_equal(m.predict(rows),old.predict(rows))
        complete=[dict(rows[0],lower=a,upper=b) for a,b in [(-math.inf,72),(72,78),(78,math.inf)]]
        self.assertAlmostEqual(sum(m.predict(complete)),1)

    def test_invalid_disagreement_is_not_zero(self):
        m=RemainingMaximum(remaining_spread='disagreement')
        for v in (None,-1,float('nan'),float('inf')):
            with self.assertRaises(ValueError):m.spread_feature([dict(self.base(),model_disagreement_f=v)])

    def test_synthetic_refit_learns_known_spread_signal(self):
        rng=np.random.default_rng(81);training=[];validation=[]
        for i in range(600):
            d=float(rng.choice([0,5,10]));y=float(rng.normal(75,1.5*math.exp(.5*d/5)))
            base=dict(self.base(),event=str(i),ticker=str(i),group=str(i),model_disagreement_f=d)
            if i<400:training.append(dict(base,lower=math.floor(y),upper=math.floor(y)+1,y=1))
            else:
                for low,high in [(-math.inf,72),(72,78),(78,math.inf)]:
                    validation.append(dict(base,lower=low,upper=high,y=int(low<=y<high)))
        a=RemainingMaximum().fit(training);b=RemainingMaximum(remaining_spread='disagreement').fit(training)
        self.assertGreater(b.theta[-1],.15)
        ma=calibration_metrics(validation,a.predict(validation));mb=calibration_metrics(validation,b.predict(validation))
        self.assertLess(mb['log_loss'],ma['log_loss'])
        self.assertLess(mb['brier'],ma['brier'])
        # Exercise the complete reporting path; the reserved test is unusable
        # on purpose and must never be consumed by this validation-only runner.
        with tempfile.TemporaryDirectory() as d,patch('helper.spread_research.build_dataset',return_value=(training+validation,{'synthetic':True})),patch('helper.spread_research.partitions',return_value=(training, [dict(r,p=.5) for r in validation],None)):
            report=compare(d,d+'/comparison')
            self.assertEqual(report['status'],'research_only')
            self.assertFalse(report['test_evaluated'])
            self.assertIn('by_disagreement',report['validation']['disagreement'])

    def test_station_month_resampling_is_deterministic(self):
        rows=[dict(event=f'KXHIGHNY-25{month}{day:02}',station='NYC',settled_max=75+value,observed_max=75)
            for month,value in [('JAN',0),('FEB',2)] for day in range(1,11)]
        r=block_uncertainty(rows,100,12)
        self.assertEqual(r,block_uncertainty(rows,100,12))
        self.assertEqual(r['station_month_blocks'],2)
        self.assertEqual(r['pooled']['sigma_interval_f'][0],0)

    def test_real_data_shortage_does_not_trigger_fit(self):
        with tempfile.TemporaryDirectory() as d,patch('helper.spread_research.RemainingMaximum.fit') as fit:
            r=compare(d,d+'/report');self.assertEqual(r['status'],'insufficient_data');fit.assert_not_called()
            self.assertFalse(r['test_evaluated'])
