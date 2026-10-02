import json
import tempfile
import unittest
from io import BytesIO
from unittest.mock import patch
from urllib.error import HTTPError,URLError
from helper.kalshi import Kalshi,KalshiError,normalize_book
from helper.storage import Store
from helper.weather import nws_get,rain_context
from helper.research_store import ResearchStore
from helper.resolve import resolve_markets
from helper.discovery import refresh_board
from test_resolve import Client,M


class Response(BytesIO):
    def __enter__(self):return self
    def __exit__(self,*a):self.close()


class DataAudit(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.store=Store(self.temp.name)
    def test_success_archives_exact_raw_body(self):
        body=b'{"markets": []}'
        with patch('helper.kalshi.urlopen',return_value=Response(body)):
            self.assertEqual(Kalshi(self.store,0).get('/markets'),{'markets':[]})
        with self.store.connect() as db:row=db.execute('SELECT * FROM fetches').fetchone()
        self.assertEqual((self.store.root/row['path']).read_bytes(),body)
    def test_rate_limit_retries_then_succeeds(self):
        error=HTTPError('https://x',429,'Too many',{},None)
        with patch('helper.kalshi.urlopen',side_effect=[error,Response(b'{"ok":true}')]) as get,patch('helper.kalshi.time.sleep'):
            self.assertTrue(Kalshi(self.store,0).get('/markets')['ok']);self.assertEqual(get.call_count,2)
    def test_404_does_not_retry(self):
        with patch('helper.kalshi.urlopen',side_effect=HTTPError('https://x',404,'No',{},None)) as get:
            with self.assertRaises(KalshiError):Kalshi(self.store,0).get('/markets/A')
            self.assertEqual(get.call_count,1)
    def test_network_failure_has_bounded_retries(self):
        with patch('helper.kalshi.urlopen',side_effect=URLError('offline')) as get,patch('helper.kalshi.time.sleep'):
            with self.assertRaises(RuntimeError):Kalshi(self.store,0).get('/markets')
            self.assertEqual(get.call_count,3)
    def test_socket_timeout_has_bounded_retries(self):
        with patch('helper.kalshi.urlopen',side_effect=TimeoutError('timed out')) as get,patch('helper.kalshi.time.sleep'):
            with self.assertRaises(RuntimeError):Kalshi(self.store,0).get('/markets')
            self.assertEqual(get.call_count,3)
    def test_null_empty_book_sides_are_empty_not_crash(self):
        r=normalize_book('A',{'orderbook_fp':{'yes_dollars':None,'no_dollars':None}},'2026-01-01T00:00:00Z')
        self.assertEqual(r['yes'],[]);self.assertEqual(r['no'],[])
    def test_pagination_deduplicates_contracts(self):
        client=Kalshi(self.store,0)
        with patch.object(client,'get',side_effect=[{'markets':[{'ticker':'A'}],'cursor':'next'},{'markets':[{'ticker':'A'},{'ticker':'B'}],'cursor':''}]):
            self.assertEqual([m['ticker'] for m in client.markets('X',3)],['A','B'])
    def test_repeated_cursor_is_an_error(self):
        client=Kalshi(self.store,0)
        with patch.object(client,'get',return_value={'markets':[],'cursor':'repeat'}):
            with self.assertRaisesRegex(RuntimeError,'cursor'):client.markets('X',5)
    def test_series_resolution_honors_each_event_fee_override(self):
        result=resolve_markets(self.store,'KXRAINWKND',Client())
        self.assertEqual(result['markets'][0]['fee_rate'],'0.14')
    def test_discovery_honors_event_fee_override(self):
        series=dict(ticker='KXHIGHNY',title='NY weather',categories=['Climate and Weather'],frequency='daily',fee_type='quadratic',fee_multiplier=1)
        class DiscoveryClient(Client):
            def __init__(self,store):pass
            def get(self,path,**kw):
                if path=='/series':return {'series':[series]}
                return super().get(path,**kw)
        with patch('helper.discovery.Kalshi',DiscoveryClient):r=refresh_board(self.store)
        self.assertEqual(r['markets'][0]['fee_rate'],'0.14')
    def test_nws_cache_does_not_refetch_fresh_response(self):
        url='https://api.weather.gov/stations/KNYC'
        with patch('helper.weather.urlopen',return_value=Response(b'{"test":1}')) as get:
            self.assertEqual(nws_get(self.store,url),{'test':1});nws_get(self.store,url)
            self.assertEqual(get.call_count,1)
    def test_discovery_skips_inactive_series_before_counting_limit(self):
        series=[dict(ticker=f'MENTION{i}',title=f'Mentions {i}',categories=['Mentions'],fee_type='quadratic',fee_multiplier=1) for i in range(6)]
        class DiscoveryClient(Client):
            def __init__(self,store):pass
            def get(self,path,**kw):
                if path=='/series':return {'series':series}
                return super().get(path,**kw)
            def markets(self,ticker,limit):
                if ticker in ['MENTION0','MENTION1','MENTION2']:return []
                return super().markets(ticker,limit)
        with patch('helper.discovery.Kalshi',DiscoveryClient):r=refresh_board(self.store,['mentions'])
        self.assertEqual(r['scanned_series'],6)
        self.assertEqual(len(r['markets']),3)
    def test_future_dated_nws_cache_is_not_trusted(self):
        url='https://api.weather.gov/stations/KNYC'
        ResearchStore(self.store).set('nws:'+url,dict(at='2099-01-01T00:00:00Z',body={'stale':True}))
        with patch('helper.weather.urlopen',return_value=Response(b'{"fresh":true}')):
            self.assertEqual(nws_get(self.store,url),{'fresh':True})
    def test_weather_never_fetches_arbitrary_host(self):
        with patch('helper.weather.urlopen') as get:
            with self.assertRaises(ValueError):nws_get(self.store,'https://example.com/data')
            get.assert_not_called()

if __name__=='__main__':unittest.main()
