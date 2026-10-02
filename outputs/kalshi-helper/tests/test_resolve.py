import tempfile
import unittest
from unittest.mock import patch
from helper.resolve import parse_reference,resolve_markets
from helper.kalshi import KalshiError,normalize_outcome
from helper.storage import Store
from helper.weather import rain_context

LINK='https://kalshi.com/markets/kxrainwknd/where-will-it-rain-this-weekend/kxrainwknd-26sep19?op_side=BUY&op_order_type=contracts'
M=dict(ticker='KXRAINWKND-26SEP19-NYC',event_ticker='KXRAINWKND-26SEP19',close_time='2099-09-21T05:00:00Z',status='active',
       yes_bid_dollars='0.3',yes_ask_dollars='0.4',rules_primary='If the total precipitation at CLINYC in New York City on any day within September 19, 2026 through September 20, 2026 is strictly greater than 0 inches, then the market resolves to Yes.')


class Client:
    def get(self,path,**params):
        if path.startswith('/series/'):
            return {'series':dict(ticker='KXRAINWKND',title='Weekend rain',fee_type='quadratic',fee_multiplier=1)}
        if path.endswith('-NYC'):raise KalshiError(404,path)
        return {'event':dict(series_ticker='KXRAINWKND',title='Weekend rain',markets=[M],fee_multiplier_override=2)}
    def market(self,ticker):return dict(M)
    def markets(self,ticker,limit):return [dict(M)]


class ResolveTests(unittest.TestCase):
    def test_url_query_and_case(self):
        self.assertEqual(parse_reference(LINK),'KXRAINWKND-26SEP19')
        self.assertEqual(parse_reference(' kxrainwknd '),'KXRAINWKND')
    def test_untrusted_hosts_paths_and_encoded_slashes_rejected(self):
        for value in ['https://evil.test/markets/x/a/b','https://kalshi.com.evil.test/markets/x','http://kalshi.com/markets/x','https://user@kalshi.com/markets/x','https://kalshi.com:443/markets/x','https://kalshi.com/markets/x/a/b%2F..','../a','x?foo=bar']:
            with self.subTest(value=value),self.assertRaises(ValueError):parse_reference(value)
    def test_event_series_and_contract_resolve_before_worker(self):
        with tempfile.TemporaryDirectory() as path:
            for value,kind in [(LINK,'event'),('kxrainwknd','series'),('kxrainwknd-26sep19-nyc','contract')]:
                r=resolve_markets(Store(path),value,Client())
                self.assertEqual(r['resolved_as'],kind)
                self.assertEqual(r['tickers'],[M['ticker']])
                self.assertEqual(r['markets'][0]['fee_rate'],'0.14')
    def test_non_404_error_does_not_fall_back(self):
        client=Client()
        with tempfile.TemporaryDirectory() as path,patch.object(client,'get',side_effect=KalshiError(503,'/events/x')):
            with self.assertRaises(KalshiError):resolve_markets(Store(path),'X-ABC',client)
    def test_finalized_market_settles(self):
        out=normalize_outcome(dict(M,status='finalized',result='yes'),'2026-09-22T00:00:00Z')
        self.assertEqual(out['yes_payout'],'1')
    def test_rain_context_never_invents_contract_odds(self):
        def source(store,url):
            if '/stations/' in url:return {'geometry':{'coordinates':[-73.9,40.7]}}
            if '/points/' in url:return {'properties':dict(timeZone='America/New_York',forecast='https://api.weather.gov/gridpoints/X/1,1/forecast')}
            return {'properties':dict(updateTime='2026-09-19T08:00:00Z',periods=[dict(name='Saturday',startTime='2026-09-19T06:00:00-04:00',endTime='2026-09-19T18:00:00-04:00',probabilityOfPrecipitation={'value':40},shortForecast='Rain possible'),dict(name='Sunday',startTime='2026-09-20T06:00:00-04:00',endTime='2026-09-20T18:00:00-04:00',probabilityOfPrecipitation={'value':50},shortForecast='Rain possible')])}
        with patch('helper.weather.nws_get',source),patch('helper.weather.utcnow',return_value='2026-09-19T09:00:00Z'):
            r=rain_context(None,{'rules':M['rules_primary']})
        self.assertIsNone(r['probability']);self.assertFalse(r['trade_eligible'])
        self.assertEqual(r['period_union_bounds'],[.5,.9])
        self.assertIn('already started',r['note'])
    def test_unknown_rule_station_and_stale_weather_fail_closed(self):
        for rule in ['custom rule',M['rules_primary'].replace('CLINYC','CLIXYZ')]:
            self.assertEqual(rain_context(None,{'rules':rule})['status'],'unavailable')

if __name__=='__main__':unittest.main()
