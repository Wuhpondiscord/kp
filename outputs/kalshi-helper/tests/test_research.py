from copy import deepcopy
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import tempfile
import unittest
import numpy as np
from helper.evaluation import chronological_plan, validate_samples, quote_scenario, paired_block_interval, Predictor, ModelService, score
from helper.history import valid_quote, extract_samples
from helper.storage import Store
from helper.research_store import ResearchStore


def rows(days=100):
    result=[]
    for day in range(days):
        at=datetime(2025,1,1,tzinfo=timezone.utc)+timedelta(days=day)
        for city in ('NY','CHI'):
            result.append(dict(ticker=f'{city}-{day}',event=f'{city}-{day}',series='KXHIGH'+city,
                group=at.date().isoformat(),at=at.isoformat(),feature_time=at.isoformat(),
                close_time=(at+timedelta(hours=24)).isoformat(),settled_at=(at+timedelta(hours=30)).isoformat(),
                execution_at=(at+timedelta(hours=1)).isoformat(),execution_bid=.39,execution_ask=.41,
                bid=.39,ask=.41,p=.4,spread=.02,hours_left=24,y=day%2,source='kalshi_hourly_candle',synthetic=False))
    return result


class ResearchTests(unittest.TestCase):
    def test_every_probability_at_bin_boundary_is_counted_once(self):
        data=rows(6)[:11]
        bins=score(data,[i/10 for i in range(11)])['calibration']
        self.assertEqual(sum(b['count'] for b in bins),11)
        self.assertEqual(len(bins),10)
        self.assertEqual(bins[6]['predicted'],.6)

    def test_folds_are_grouped_and_embargo_all_labels(self):
        development,test,folds=chronological_plan(rows())
        for train,val in [*folds,(development,test)]:
            self.assertFalse({r['group'] for r in train}&{r['group'] for r in val})
            latest=max(datetime.fromisoformat(r['settled_at']) for r in train)
            earliest=min(datetime.fromisoformat(r['at']) for r in val)
            self.assertLess(latest,earliest-timedelta(hours=48))
            self.assertFalse({r['ticker'] for r in train}&{r['ticker'] for r in val})
        self.assertEqual(len({r['group'] for r in test}),20)

    def test_future_features_known_outcomes_synthetic_duplicates_rejected(self):
        base=rows(1)
        for key,value in [('feature_time',base[0]['settled_at']),('settled_at',base[0]['at']),('synthetic',True),('p',float('nan'))]:
            changed=deepcopy(base);changed[0][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError): validate_samples(changed)
        with self.assertRaises(ValueError):validate_samples(base+base)

    def test_holdout_labels_do_not_change_fit_or_fold_selection(self):
        source=rows();development,test,folds=chronological_plan(source)
        altered=deepcopy(source)
        for r in altered:
            if r['group']>=min(x['group'] for x in test):r['y']=1-r['y']
        development2,test2,folds2=chronological_plan(altered)
        self.assertEqual(development,development2)
        self.assertEqual(folds,folds2)
        model=Predictor('calibrated_market').fit(development)
        np.testing.assert_array_equal(model.predict(test),model.predict(test2))

    def test_history_price_units_and_missing_values(self):
        self.assertEqual(valid_quote({'yes_bid':{'close':'0.2000'},'yes_ask':{'close':'0.3000'}}),(.2,.3))
        self.assertEqual(valid_quote({'yes_bid':{'close_dollars':'0.20'},'yes_ask':{'close_dollars':'0.30'}}),(.2,.3))
        for bid,ask in [(20,30),('20','30'),('0.5','0.4'),('nan','0.3'),(None,'0.3')]:
            self.assertIsNone(valid_quote({'yes_bid':{'close':bid},'yes_ask':{'close':ask}}))

    def test_later_candle_cannot_change_prediction_features(self):
        r=rows(1)[0];close=datetime.fromisoformat(r['close_time']);epoch=int(close.timestamp())
        market=dict(ticker=r['ticker'],event_ticker=r['event'],status='finalized',result='yes',close_time=r['close_time'],settlement_ts=r['settled_at'])
        candles=[dict(end_period_ts=epoch-24*3600,yes_bid={'close':'0.2'},yes_ask={'close':'0.3'}),
                 dict(end_period_ts=epoch-23*3600,yes_bid={'close':'0.8'},yes_ask={'close':'0.9'})]
        before=extract_samples('KXHIGHNY',market,candles)[0]
        candles[1]['yes_ask']['close']='1.0'
        after=extract_samples('KXHIGHNY',market,candles)[0]
        for key in ('p','bid','ask','spread','at','prior_move'):self.assertEqual(before[key],after[key])
        self.assertNotEqual(before['execution_ask'],after['execution_ask'])

    def test_scenario_cash_conservation_and_stress(self):
        data=rows(20)
        base=quote_scenario(data,[.85]*len(data))
        stress=quote_scenario(data,[.85]*len(data),slippage=.05,fee_multiplier=2)
        self.assertEqual(Decimal(base['ending_cash']),Decimal(base['bankroll'])+sum((Decimal(t['pnl']) for t in base['ledger']),Decimal(0)))
        self.assertLessEqual(Decimal(stress['pnl']),Decimal(base['pnl']))
        self.assertIsNone(base['verified_fills'])

    def test_no_next_quote_no_scenario_trade(self):
        data=rows(1)
        for r in data:r['execution_at']=None
        report=quote_scenario(data,[.99]*len(data))
        self.assertEqual(report['trades'],0)
        self.assertEqual(report['ending_cash'],'1000')

    def test_interval_is_reproducible_and_paired_baseline_zero(self):
        data=rows(20);p=[.4]*len(data)
        a=paired_block_interval(data,p,p,repetitions=50)
        b=paired_block_interval(data,p,p,repetitions=50)
        self.assertEqual(a,b);self.assertEqual(a['lower'],0);self.assertEqual(a['upper'],0)

    def test_model_service_stays_baseline_without_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            store=Store(folder);r=rows(1)[0]
            ResearchStore(store).set('active_model',dict(passed=False,trained_through='2024-01-01T00:00:00Z'))
            p,reason=ModelService(store).predict('KXHIGHNY',.3,.5,r['at'],r['close_time'])
            self.assertEqual(p,.4);self.assertIn('no validated',reason)

    def test_model_artifact_hash_must_match(self):
        with tempfile.TemporaryDirectory() as folder:
            store=Store(folder);r=rows(1)[0]
            (store.root/'models').mkdir();(store.root/'models'/'bad.pkl').write_bytes(b'not a model')
            ResearchStore(store).set('active_model',dict(passed=True,trained_through='2024-01-01T00:00:00Z',model_id='test',model_file='bad.pkl',sha256='wrong'))
            with self.assertRaisesRegex(ValueError,'hash mismatch'):
                ModelService(store).predict('KXHIGHNY',.3,.5,r['at'],r['close_time'])

if __name__=='__main__':unittest.main()
