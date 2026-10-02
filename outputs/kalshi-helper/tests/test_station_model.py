import pickle
import unittest
from unittest.mock import patch
import numpy as np
from helper.station_observations import parse_observations,ObservationIndex,SITES
from helper.station_model import StationOffset,MarketOffset,station_features
from helper.weather_model import STATIONS,supported


class StationTests(unittest.TestCase):
    def test_midway_is_used_for_chicago(self):
        self.assertEqual(SITES['KXHIGHCHI'],'MDW')
        lat,lon,_=STATIONS['KXHIGHCHI']
        self.assertTrue(41.77<lat<41.80 and -87.77<lon<-87.74)

    def test_parser_rejects_wrong_station_corrections_and_conflicts(self):
        raw=b'station,valid,tmpf,metar\nNYC,2026-09-01 10:00,70,KNYC RMK 10250\nNYC,2026-09-01 11:00,71,KNYC COR RMK\nNYC,2026-09-01 12:00,72,KNYC\nNYC,2026-09-01 12:00,73,KNYC\n'
        rows,audit=parse_observations(raw,'NYC')
        self.assertEqual(len(rows),1);self.assertEqual(rows[0]['max_6h'],77)
        self.assertEqual(audit['rejected_rows'],1);self.assertEqual(audit['conflicting_timestamps'],1)
        with self.assertRaisesRegex(ValueError,'Unexpected'):parse_observations(raw,'MDW')

    def test_future_and_previous_day_reports_cannot_raise_running_max(self):
        rows=[dict(at='2026-09-01T04:59:00Z',temperature=100,max_6h=None),
              dict(at='2026-09-01T10:00:00Z',temperature=70,max_6h=99),
              dict(at='2026-09-01T11:00:00Z',temperature=72,max_6h=75),
              dict(at='2026-09-01T12:00:00Z',temperature=110,max_6h=None)]
        index=ObservationIndex(rows)
        a=index.feature('KXHIGHNY','KXHIGHNY-26SEP01-B70.5','2026-09-01T10:30Z')
        self.assertEqual(a['observed_max'],70) # six-hour window crosses midnight standard time
        b=index.feature('KXHIGHNY','KXHIGHNY-26SEP01-B70.5','2026-09-01T11:30Z')
        self.assertEqual(b['observed_max'],75)
        self.assertFalse(index.feature('KXHIGHNY','KXHIGHNY-26SEP01-B70.5','2026-09-01T10:29Z')['observed'])
        self.assertFalse(index.feature('KXHIGHNY','KXHIGHNY-26SEP01-B70.5','2026-09-01T17:00Z')['observed'])

    def test_delay_sensitivity_does_not_use_later_observations(self):
        ix=ObservationIndex([dict(at='2026-09-01T10:00Z',temperature=70,max_6h=None),dict(at='2026-09-01T11:00Z',temperature=75,max_6h=None)])
        self.assertEqual(ix.feature('KXHIGHNY','KXHIGHNY-26SEP01-T70','2026-09-01T11:30Z',30)['observed_max'],75)
        self.assertEqual(ix.feature('KXHIGHNY','KXHIGHNY-26SEP01-T70','2026-09-01T11:30Z',90)['observed_max'],70)

    def test_station_signal_learned_and_missing_observations_fall_back(self):
        data=[dict(p=.5,spread=.02,hours_left=6,series='KXHIGHNY',group=str(i//2),ticker=str(i),event=str(i),
                   lower=-float('inf'),upper=72,observed=True,observed_max=70 if i%2 else 80,
                   observed_temperature=69,observed_trend=-1,observation_age_hours=1,local_day_hour=18,gfs=75,ecmwf=75,y=i%2) for i in range(800)]
        with patch.dict('helper.station_model.PROTOCOL',iterations=10,minimum_leaf=20):model=StationOffset().fit(data[:600],data[600:])
        p=model.predict(data[600:]);self.assertLess(p[0],.5);self.assertGreater(p[1],.5)
        np.testing.assert_array_equal(p,pickle.loads(pickle.dumps(model)).predict(data[600:]))
        self.assertEqual(model.predict([dict(data[0],observed=False)])[0],.5)
        flipped=[dict(r,y=1-r['y']) for r in data]
        np.testing.assert_array_equal(station_features(data),station_features(flipped))

    def test_new_model_starts_exactly_at_market(self):
        self.assertEqual(StationOffset().predict([dict(p=.123)])[0],.123)

    def test_wrong_chicago_station_is_not_supported(self):
        rules="Highest temperature at Chicago O'Hare according to the Climatological Report"
        self.assertFalse(supported({'rules_primary':rules},'KXHIGHCHI'))
        self.assertTrue(supported({'rules_primary':rules.replace("O'Hare",'Midway')},'KXHIGHCHI'))

    def test_price_only_variant_requires_no_station_fields_and_freezes_rate(self):
        data=[dict(p=.2,y=i%2,spread=.02,hours_left=12,series='KXHIGHNY',group=str(i//2),ticker=str(i),event=str(i)) for i in range(800)]
        with patch.dict('helper.station_model.PROTOCOL',iterations=5):model=MarketOffset().fit(data[:600],data[600:])
        p=model.predict(data[600:])
        self.assertGreater(float(p.mean()),.2)
        with patch.dict('helper.station_model.PROTOCOL',learning_rate=10):np.testing.assert_array_equal(p,model.predict(data[600:]))
