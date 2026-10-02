from copy import deepcopy
from datetime import timedelta
import pickle
import threading
import unittest
import numpy as np
from helper.adaptive import OffsetCalibrator,SelectedPredictor,select_adaptive,basis
from helper.weather_experiment import freshest_feature,WeatherDistribution
from helper.weather_model import event_date


def rows(n=120):
    return [dict(p=.2,spread=.02,hours_left=12,series='KXHIGHNY',group=str(i//6),
                 ticker=str(i),event=str(i),y=int(i%10<7)) for i in range(n)]


class AdaptiveTests(unittest.TestCase):
    def test_calibrator_learns_bias_and_is_serializable(self):
        data=rows();model=OffsetCalibrator().fit(data)
        self.assertGreater(float(model.predict(data).mean()),.6)
        np.testing.assert_array_equal(model.predict(data),pickle.loads(pickle.dumps(model)).predict(data))
        model.weight=0;np.testing.assert_array_equal(model.predict(data),[r['p'] for r in data])

    def test_features_ignore_labels_and_scaler_only_sees_train(self):
        data=rows();model=OffsetCalibrator('spline').fit(data);changed=deepcopy(data)
        for r in changed:r['y']=1-r['y']
        np.testing.assert_array_equal(basis(data,'spline'),basis(changed,'spline'))
        np.testing.assert_allclose(model.scaler.mean_,basis(data,'spline').mean(axis=0))
        before=model.scaler.mean_.copy();model.predict([dict(data[0],hours_left=100,spread=.8)])
        np.testing.assert_array_equal(before,model.scaler.mean_)

    def test_validation_rejects_opposite_training_bias(self):
        train=rows();val=[dict(r,y=0) for r in rows()]
        neural=OffsetCalibrator().fit(train)
        model,candidates=select_adaptive(train,val,neural)
        self.assertEqual(model.weight,0)
        np.testing.assert_array_equal(model.predict(val),[r['p'] for r in val])
        self.assertTrue(all('brier' in c and 'eligible' in c for c in candidates))

    def test_cancel_selection(self):
        cancel=threading.Event();cancel.set()
        with self.assertRaises(InterruptedError):select_adaptive(rows(),rows(),SelectedPredictor(None,0,'baseline'),cancel)

    def test_newest_forecast_requires_entire_day_available(self):
        ticker='KXHIGHNY-26SEP01-B70.5';begin=event_date(ticker)+timedelta(hours=5)
        archive={lead:{(begin+timedelta(hours=i)).strftime('%Y-%m-%dT%H:%M'):70+lead for i in range(24)} for lead in (1,2)}
        old=freshest_feature('KXHIGHNY',ticker,archive,'2026-09-01T09:59:00Z')
        new=freshest_feature('KXHIGHNY',ticker,archive,'2026-09-01T10:00:00Z')
        self.assertEqual(old['weather_lead_days'],2);self.assertEqual(new['weather_lead_days'],1)
        archive[1].pop(next(iter(archive[1])))
        self.assertEqual(freshest_feature('KXHIGHNY',ticker,archive,'2026-09-01T12:00Z')['weather_lead_days'],2)
        with self.assertRaisesRegex(ValueError,'No complete forecast'):freshest_feature('KXHIGHNY',ticker,archive,'2026-08-29T00:00Z')

    def test_distribution_fitting_does_not_count_duplicate_quotes_as_outcomes(self):
        data=[dict(event=str(i),y=1,series='KXHIGHNY',weather_day='2026-09-01',gfs=70+i%5,
                   gfs_old=70+i%5,ecmwf=71+i%5,gfs_lead=1,ecmwf_lead=1,lower=69+i%5,upper=72+i%5) for i in range(90)]
        a=WeatherDistribution('multi').fit(data);b=WeatherDistribution('multi').fit(data*3)
        self.assertEqual(a.events,90);self.assertEqual(b.events,90)
        np.testing.assert_allclose(a.predict(data),b.predict(data))
        base=data[0];partitions=[dict(base,lower=-float('inf'),upper=70),dict(base,lower=70,upper=72),dict(base,lower=72,upper=float('inf'))]
        self.assertAlmostEqual(float(a.predict(partitions).sum()),1,places=5)
