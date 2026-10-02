import json,math,tempfile,unittest
from pathlib import Path
from datetime import timedelta
from unittest.mock import patch
import numpy as np
from helper.weather_station_pilot import targets,join_day,fit,score,fetch,forecast_hours,VARIABLE
from helper.core import stamp

class WeatherStationPilotTests(unittest.TestCase):
    def label(self):return dict(target_f=70,label_issued_at='2024-07-02T08:00:00Z',target_product='202407020800-KOKX-CDUS41-CLINYC',target_station='KNYC')
    def hours(self):
        start=stamp('2024-07-01T05:00:00Z')
        return {start+timedelta(hours=i):60+i/2 for i in range(24)}
    def test_fixed_standard_day_and_no_future_assumed_forecast(self):
        row=join_day('KXHIGHNY','2024-07-01',self.label(),self.hours())
        self.assertEqual(row['day_start'],'2024-07-01T05:00:00+00:00')
        self.assertLess(stamp(row['forecast_available_at_assumed']),stamp(row['decision_at']))
        self.assertEqual(row['raw_forecast_f'],71.5)
        self.assertFalse(row['explicit_single_run']);self.assertFalse(row['availability_verified'])
    def test_missing_hour_and_premature_target_rejected(self):
        h=self.hours();h.pop(next(iter(h)))
        with self.assertRaises(ValueError):join_day('KXHIGHNY','2024-07-01',self.label(),h)
        with self.assertRaises(ValueError):join_day('KXHIGHNY','2024-07-01',dict(self.label(),label_issued_at='2024-07-01T20:00:00Z'),self.hours())
    def test_cli_identity_duplicates_and_missing_values(self):
        r=dict(station='KNYC',valid='2023-01-01',high=55,product='202301020625-KOKX-CDUS41-CLINYC')
        self.assertEqual(targets({'results':[r]},'KNYC',2023)[0]['2023-01-01']['target_f'],55)
        for body in [{'results':[r,r]},{'results':[dict(r,station='KMIA')]}]:
            with self.assertRaises(ValueError):targets(body,'KNYC',2023)
        self.assertFalse(targets({'results':[dict(r,high='M')]},'KNYC',2023)[0])
    def test_unit_validation(self):
        body=dict(latitude=40.779,longitude=-73.969,utc_offset_seconds=0,hourly_units={VARIABLE:'°C'},hourly={'time':[],VARIABLE:[]})
        with self.assertRaises(ValueError):forecast_hours(body,'KXHIGHNY')
    def test_calibration_uses_training_targets_only_and_crps_known_value(self):
        from helper.weather_model import STATIONS
        train=[dict(station=s,target_f=72,raw_forecast_f=70) for s in STATIONS for _ in range(90)]
        model=fit(train)
        self.assertEqual(model['KXHIGHNY']['bias_f'],2)
        before=json.dumps(model,sort_keys=True)
        val=[dict(station='KXHIGHNY',target_f=72,raw_forecast_f=70)]
        metrics=score(val,model,True)
        self.assertAlmostEqual(metrics['crps_f'],.5*(math.sqrt(2)-1)/math.sqrt(math.pi))
        score([dict(val[0],target_f=140)],model,True)
        self.assertEqual(json.dumps(model,sort_keys=True),before)
    def test_offline_cache_cannot_silently_fetch(self):
        with tempfile.TemporaryDirectory() as d,patch('helper.weather_station_pilot.urlopen') as get:
            with self.assertRaises(ValueError):fetch(d,'https://example.com',offline=True)
            get.assert_not_called()
