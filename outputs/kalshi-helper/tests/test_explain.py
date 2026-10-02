import tempfile
import unittest
from unittest.mock import patch
from helper.explain import explain_market
from helper.storage import Store
from helper.research_store import ResearchStore
from helper.evaluation import PredictionResult


class FakePredictor:
    name='test'
    def predict(self,rows):return [.5*r['p']+.55 for r in rows]
class Service:
    bundle={'experimental':FakePredictor()}
    def predict_signal(self,*args,**kwargs):return PredictionResult(.75,'Renamed model label',True,'test')

CARD=dict(ticker='KXHIGHNY-TEST',event_ticker='KXHIGHNY-EVENT',title='Temperature in New York?',series='KXHIGHNY',
    bid=.39,ask=.41,no_ask=.61,midpoint=.4,fee_rate='.07',at='2026-01-01T00:00:00Z',close_time='2026-01-02T00:00:00Z')


class ExplainTests(unittest.TestCase):
    def test_calibrator_explanation_does_not_claim_weather_inputs(self):
        with tempfile.TemporaryDirectory() as folder,patch('helper.explain.utcnow',return_value=CARD['at']),patch.object(FakePredictor,'name','beta_market'):
            store=Store(folder);ResearchStore(store).set('active_model',dict(model_id='test',experiment_id='test',experimental='beta_market',passed=False))
            result=explain_market(store,CARD,Service())
            self.assertIn('market probability only',result['reasons'][1])
            self.assertIn('does not use weather',result['reasons'][1])

    def test_net_edge_has_fees_and_failed_gate_cannot_recommend_trade(self):
        with tempfile.TemporaryDirectory() as folder,patch('helper.explain.utcnow',return_value=CARD['at']):
            store=Store(folder);ResearchStore(store).set('active_model',dict(model_id='test',experiment_id='test',experimental='test',passed=False))
            r=explain_market(store,CARD,Service())
            self.assertEqual(r['action'],'WATCH');self.assertEqual(r['lean'],'YES')
            self.assertAlmostEqual(r['edges']['yes'],.75-.43)
            self.assertEqual(len(r['sensitivity']),4)
            self.assertTrue(r['kalshi_url'].startswith('https://kalshi.com/markets/'))

    def test_stale_quotes_never_get_trade_recommendation(self):
        with tempfile.TemporaryDirectory() as folder,patch('helper.explain.utcnow',return_value='2026-01-01T05:00:00Z'):
            store=Store(folder);ResearchStore(store).set('active_model',dict(model_id='test',experiment_id='test',experimental='test',passed=True))
            r=explain_market(store,CARD,Service())
            self.assertEqual(r['action'],'WATCH');self.assertIn('Refresh',r['label'])

    def test_unsupported_market_has_no_fake_model_probability(self):
        with tempfile.TemporaryDirectory() as folder:
            service=Service();service.predict_signal=lambda *a,**kw:PredictionResult(.4,'Experimental · misleading label',False)
            r=explain_market(Store(folder),CARD,service)
            self.assertIsNone(r['probability']);self.assertIsNone(r['edges']);self.assertEqual(r['action'],'WATCH')

if __name__=='__main__':unittest.main()
