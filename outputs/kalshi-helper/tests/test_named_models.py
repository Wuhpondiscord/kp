import unittest,tempfile
from unittest.mock import patch
from helper.named_models import configuration,LiveNamedModel,load
from helper.storage import Store
from urllib.error import URLError

class NamedModelsTests(unittest.TestCase):
    def test_socket_permission_failure_has_actionable_message_and_no_retries(self):
        from helper.kalshi import Kalshi
        error=OSError('blocked');error.winerror=10013
        with tempfile.TemporaryDirectory() as d,patch('helper.kalshi.urlopen',side_effect=URLError(error)) as request:
            with self.assertRaisesRegex(RuntimeError,'Restart BetCheck'):Kalshi(Store(d),interval=0).get('/exchange/status')
            self.assertEqual(request.call_count,1)
    def test_configuration_rejects_unknown_family_and_unsafe_checkpoint(self):
        for c in [{'name':'unknown'},{'weather_weight':float('nan')},{'weather_weight':1.1},{'checkpoint':'../../x'}]:
            with self.assertRaises(ValueError):configuration(c)
        self.assertEqual(configuration({'name':'WeatherSignal','weather_weight':.2})['weather_weight'],1)
        self.assertEqual(configuration({'name':'MarketGuard','weather_weight':.8})['weather_weight'],0)
    def test_live_never_calls_weather_outside_window_or_without_experimental_mode(self):
        with tempfile.TemporaryDirectory() as d,patch('helper.named_models.load',return_value=object()):
            m=LiveNamedModel(Store(d),{'name':'ConsensusBlend'})
            m.market=dict(ticker='KXHIGHNY-26APR04',rules='highest temperature central park climatological report')
            a=m.predict_signal('KXHIGHNY',.4,.5,'2026-04-04T15:00:00Z','2026-04-04T16:00:00Z',True)
            self.assertFalse(a.signal);self.assertIn('12–14',a.reason)
            b=m.predict_signal('KXHIGHNY',.4,.5,'2026-04-04T15:00:00Z','2026-04-05T04:00:00Z',False)
            self.assertFalse(b.signal);self.assertIn('Experimental',b.reason)
