from datetime import datetime,timedelta,timezone
import tempfile
import unittest
from unittest.mock import patch
from helper.storage import Store
from helper.history import download_history,extract_samples


def market(day,ticker):
    return dict(ticker=ticker,event_ticker=ticker+'-EVENT',status='finalized',result='yes',market_type='binary',notional_value_dollars='1.0000',
        close_time=day.isoformat(),settlement_ts=(day+timedelta(hours=2)).isoformat(),settlement_value_dollars='1.0000')
def candles(m):
    close=int(datetime.fromisoformat(m['close_time']).timestamp())
    return [dict(end_period_ts=close-h*3600,yes_bid={'close_dollars':'.4'},yes_ask={'close_dollars':'.5'}) for h in range(27,2,-1)]


class HistoryAudit(unittest.TestCase):
    def test_backfill_routes_recent_archive_and_is_idempotent(self):
        now=datetime(2026,9,19,tzinfo=timezone.utc)
        recent=market(now-timedelta(days=2),'RECENT');old=market(now-timedelta(days=80),'ARCHIVE')
        calls=[]
        class Client:
            def __init__(self,store):pass
            def get(self,path,**params):
                calls.append(path)
                if path=='/historical/cutoff':return {'market_settled_ts':(now-timedelta(days=60)).isoformat()}
                if path=='/markets':return {'markets':[recent], 'cursor':''}
                if path=='/historical/markets':return {'markets':[old], 'cursor':''}
                if path=='/markets/candlesticks':return {'markets':[{'market_ticker':'RECENT','candlesticks':candles(recent)}]}
                if path=='/historical/markets/ARCHIVE/candlesticks':return {'candlesticks':candles(old)}
                raise AssertionError(path)
        with tempfile.TemporaryDirectory() as folder,patch('helper.history.Kalshi',Client),patch('helper.history.utcnow',return_value=now.isoformat()):
            store=Store(folder);r=download_history(store,series_list=['TEST'])
            self.assertEqual(r['stored_markets'],2);self.assertEqual(r['samples'],6);self.assertFalse(r['errors'])
            count=len([p for p in calls if 'candlesticks' in p]);again=download_history(store,series_list=['TEST'])
            self.assertEqual(again['data_hash'],r['data_hash']);self.assertEqual(len([p for p in calls if 'candlesticks' in p]),count)
    def test_inconsistent_outcome_or_notional_not_used_for_training(self):
        base=market(datetime(2026,9,19,tzinfo=timezone.utc),'A')
        for changed in [dict(base,settlement_value_dollars='0.0000'),dict(base,notional_value_dollars='100'),dict(base,market_type='scalar')]:
            with self.subTest(changed=changed):self.assertEqual(extract_samples('TEST',changed,candles(base)),[])
    def test_missing_settlement_and_missing_quotes_are_not_fabricated(self):
        m=market(datetime(2026,9,19,tzinfo=timezone.utc),'A')
        self.assertEqual(extract_samples('TEST',m,[]),[])
        m['settlement_ts']=None;self.assertEqual(extract_samples('TEST',m,candles(m)),[])

if __name__=='__main__':unittest.main()
