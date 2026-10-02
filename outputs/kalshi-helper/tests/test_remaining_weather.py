import copy,unittest
from helper.remaining_weather import features

class RemainingWeatherTests(unittest.TestCase):
    def snapshot(self):
        return dict(series='KXHIGHNY',captured_at='2026-09-24T15:00:00Z',sources=[dict(received_at='2026-09-24T14:59:00Z',body=dict(properties=dict(updateTime='2026-09-24T14:00:00Z',periods=[
            dict(startTime='2026-09-24T15:00:00Z',endTime='2026-09-25T05:00:00Z',temperature=25,temperatureUnit='C')]))),
            dict(received_at='2026-09-24T14:59:00Z',body=dict(features=[dict(properties=dict(timestamp='2026-09-24T14:00:00Z',temperature=dict(value=24,unitCode='wmoUnit:degC')))]))])
    def test_units_and_standard_day(self):
        r=features(self.snapshot(),'KXHIGHNY-26SEP24-B75.5','2026-09-24T15:00:00Z')
        self.assertEqual(r['remaining_forecast_max'],77);self.assertAlmostEqual(r['observed_max'],75.2)
        self.assertEqual(r['remaining_hours'],14);self.assertTrue(r['forecast_complete'])
    def test_future_receipt_rejected(self):
        with self.assertRaises(ValueError):features(self.snapshot(),'KXHIGHNY-26SEP24-B75.5','2026-09-24T14:00:00Z')
    def test_future_observation_ignored(self):
        s=self.snapshot();s['sources'][1]['body']['features'][0]['properties']['timestamp']='2026-09-24T16:00:00Z'
        self.assertIsNone(features(s,'KXHIGHNY-26SEP24-B75.5','2026-09-24T15:00:00Z')['observed_max'])
    def test_overlap_does_not_hide_missing_hours(self):
        s=self.snapshot();p=s['sources'][0]['body']['properties'];p['periods'][0]['endTime']='2026-09-24T16:00:00Z'
        p['periods'].append(copy.deepcopy(p['periods'][0]))
        r=features(s,'KXHIGHNY-26SEP24-B75.5','2026-09-24T15:00:00Z')
        self.assertEqual(r['forecast_covered_hours'],1);self.assertFalse(r['forecast_complete'])
    def test_issue_after_receipt_rejected(self):
        s=self.snapshot();s['sources'][0]['body']['properties']['updateTime']='2026-09-24T15:00:00Z'
        with self.assertRaises(ValueError):features(s,'KXHIGHNY-26SEP24-B75.5','2026-09-24T15:00:00Z')
