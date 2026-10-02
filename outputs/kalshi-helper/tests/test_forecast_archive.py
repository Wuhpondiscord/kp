import copy,unittest,math,tempfile,json
from unittest.mock import patch,MagicMock
from datetime import datetime,timedelta,timezone
from helper.forecast_archive import archived_feature,fetch_run,MODELS
from helper.weather_model import STATIONS

def fixture():
    times=[(datetime(2026,4,4,tzinfo=timezone.utc)+timedelta(hours=i)).strftime('%Y-%m-%dT%H:%M') for i in range(48)]
    return dict(run_initialized_at='2026-04-04T06:00:00Z',url='synthetic',raw_sha256='synthetic',
        body=[dict(latitude=lat,longitude=lon,utc_offset_seconds=0,hourly_units={'temperature_2m_'+m:'°F' for m in MODELS},
        hourly=dict(time=times,**{'temperature_2m_'+m:[70+j]*48 for j,m in enumerate(MODELS)})) for lat,lon,_ in STATIONS.values()])

class ForecastArchiveTests(unittest.TestCase):
    def feature(self,r=None,at='2026-04-04T15:00:00Z'):
        return archived_feature(r or fixture(),'KXHIGHNY','KXHIGHNY-26APR04',at)
    def test_remaining_day_and_model_difference(self):
        f=self.feature();self.assertEqual(f['remaining_hours'],14)
        self.assertEqual(f['remaining_forecast_max'],70.5);self.assertEqual(f['model_disagreement_f'],1)
        self.assertFalse(f['prospective']);self.assertEqual(f['weather_available_at'],'2026-04-04T14:00:00+00:00')
    def test_publication_not_initialization(self):
        with self.assertRaises(ValueError):self.feature(at='2026-04-04T13:59:00Z')
        with self.assertRaises(ValueError):archived_feature(fixture(),'KXHIGHNY','KXHIGHNY-26APR04','2026-04-04T15:00:00Z',0)
    def test_past_and_next_day_temperatures_not_in_maximum(self):
        r=fixture();v=r['body'][0]['hourly']['temperature_2m_gfs_global'];v[14]=140;v[29]=140
        self.assertEqual(self.feature(r)['gfs_remaining_max'],70)
    def test_null_gaps_duplicates_units_and_location_rejected(self):
        mutations=[lambda b:b['hourly']['temperature_2m_gfs_global'].__setitem__(17,None),
            lambda b:b['hourly']['time'].__setitem__(17,b['hourly']['time'][16]),
            lambda b:b['hourly_units'].__setitem__('temperature_2m_gfs_global','K'),
            lambda b:b.__setitem__('latitude',0),lambda b:b.__setitem__('utc_offset_seconds',3600)]
        for change in mutations:
            r=fixture();change(r['body'][0])
            with self.assertRaises(ValueError):self.feature(r)
    def test_later_run_and_finished_day_rejected(self):
        r=fixture();r['run_initialized_at']='2026-04-05T06:00:00Z'
        with self.assertRaises(ValueError):self.feature(r)
        with self.assertRaises(ValueError):self.feature(at='2026-04-05T05:00:00Z')
    def test_cache_roundtrip_offline_and_tamper_detection(self):
        from pathlib import Path
        response=MagicMock();response.__enter__.return_value.read.return_value=json.dumps(fixture()['body']).encode()
        with tempfile.TemporaryDirectory() as d,patch('helper.forecast_archive.urlopen',return_value=response) as network,patch('helper.forecast_archive.time.sleep'):
            first=fetch_run(d,'2026-04-04')
            self.assertEqual(first,fetch_run(d,'2026-04-04',offline=True));self.assertEqual(network.call_count,1)
            with self.assertRaises(ValueError):fetch_run(d,'2026-04-05',offline=True)
            file=next(p for p in (Path(d)/'single-run-archive').glob('*.json') if not p.name.endswith('.raw.json'))
            changed=json.loads(file.read_text());changed['body'][0]['latitude']=0;file.write_text(json.dumps(changed))
            with self.assertRaises(ValueError):fetch_run(d,'2026-04-04',offline=True)
    def test_transient_network_retry_is_bounded(self):
        from urllib.error import URLError
        with tempfile.TemporaryDirectory() as d,patch('helper.forecast_archive.urlopen',side_effect=URLError('temporary')) as network,patch('helper.forecast_archive.time.sleep'):
            with self.assertRaises(URLError):fetch_run(d,'2026-04-04')
            self.assertEqual(network.call_count,3)
