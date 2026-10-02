import math,unittest,pickle
import numpy as np
from helper.observation_calibration import calibrate,validate_training_scope
from helper.event_weather import EventMaximum
from helper.weather_research import RemainingMaximum
from helper.station_observations import parse_observations,ObservationIndex

def pairs():
    return [dict(event=str(i),station='NYC',settled_max=75+(i%2),observed_max=75,
        settled_at='2025-01-02T12:00:00Z',complete_day=True,target_source='kalshi_expiration_value') for i in range(80)]

class ObservationCalibrationTests(unittest.TestCase):
    def test_location_scale_and_no_duplicate_weighting(self):
        c=calibrate(pairs(),{'fixture':'synthetic'})
        self.assertEqual(c['bias_f'],.5);self.assertAlmostEqual(c['sigma_f'],np.std([0,1]*40,ddof=1))
        with self.assertRaises(ValueError):calibrate(pairs()+pairs()[:1],{'fixture':'synthetic'})
    def test_censored_targets_and_incomplete_days_rejected(self):
        for key,value in [('target_source','winning_bracket'),('complete_day',False),('settled_max',float('nan'))]:
            rows=pairs();rows[0][key]=value
            with self.assertRaises(ValueError):calibrate(rows,{'fixture':'synthetic'})
    def test_calibration_cannot_consult_future_holdout(self):
        c=calibrate(pairs(),{'fixture':'synthetic'})
        with self.assertRaises(ValueError):validate_training_scope(c,[dict(event='new',at='2025-01-01T00:00:00Z')])
        validate_training_scope(c,[dict(event='new',at='2025-02-01T00:00:00Z')])
    def test_mass_legacy_pickle_and_caller_mutation(self):
        c=calibrate(pairs(),{'fixture':'synthetic'});m=EventMaximum(c)
        c['sigma_f']=0
        m.theta=np.array([0,0,0,0,0,0,math.log(4)])
        rows=[dict(gfs=75,ecmwf=75,hours_left=12,series='KXHIGHNY',observed=True,observed_max=75,lower=a,upper=b) for a,b in [(-math.inf,74),(74,76),(76,math.inf)]]
        p=m.predict(rows);self.assertAlmostEqual(sum(p),1)
        np.testing.assert_array_equal(p,pickle.loads(pickle.dumps(m)).predict(rows))
        del m.observation_calibration
        self.assertEqual(m.observation_protocol()['n_events'],0)
        self.assertFalse(np.allclose(p,m.predict(rows)))
        rows[0]['observed_max']=float('nan')
        with self.assertRaises(ValueError):m.predict(rows)
    def test_fit_keeps_external_calibration_fixed(self):
        c=calibrate(pairs(),{'fixture':'synthetic'})
        base=dict(remaining_forecast_max=75,remaining_hours=18,forecast_complete=True,series='KXHIGHNY',observed=True,observed_max=74)
        rows=[dict(base,event=str(i),lower=72,upper=78,y=1) for i in range(80)]
        m=RemainingMaximum(c).fit(rows)
        self.assertEqual(m.observation_protocol(),c)
        self.assertEqual(len(m.theta),m.inputs(rows)[1].shape[1]+1)
    def test_timestamp_offset_is_converted_not_relabelled(self):
        rows,_=parse_observations(b'station,valid,tmpf,metar\nNYC,2026-01-01T01:00:00-05:00,40,TEST\n','NYC')
        self.assertEqual(rows[0]['at'],'2026-01-01T06:00:00+00:00')
    def test_index_sorts_instants_and_rejects_negative_delay(self):
        index=ObservationIndex([dict(at='2026-01-01T01:00:00-05:00',temperature=40),dict(at='2026-01-01T05:30:00Z',temperature=39)])
        self.assertEqual(index.rows[0]['temperature'],39)
        with self.assertRaises(ValueError):index.feature('KXHIGHNY','KXHIGHNY-26JAN01','2026-01-01T07:00:00Z',-1)
