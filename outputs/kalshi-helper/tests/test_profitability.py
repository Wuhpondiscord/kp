import unittest
from helper.live import qualify_signal
from helper.core import Engine
from helper.evaluation import quote_scenario
from helper.profitability import select_policy,evaluate_policy
from test_engine import market,book,prediction
from test_research import rows


class ProfitabilityTests(unittest.TestCase):
    def test_later_price_cannot_create_decision_time_edge(self):
        data=rows(1)[:1];data[0].update(bid=.49,ask=.51,p=.5,execution_bid=.19,execution_ask=.21,y=1)
        self.assertEqual(quote_scenario(data,[.5],slippage=0)['trades'],1)
        self.assertEqual(quote_scenario(data,[.5],slippage=0,decision_qualified=True)['trades'],0)

    def test_live_qualification_accounts_for_fees_and_missing_rates(self):
        self.assertIsNone(qualify_signal(.55,.49,.51,.07,.04))
        self.assertIsNone(qualify_signal(.8,.49,.51,None,.04))
        self.assertEqual(qualify_signal(.8,.49,.51,.07,.04)['signal_side'],'yes')

    def test_execution_cannot_flip_preselected_side(self):
        e=Engine();e.feed(market());e.feed(book())
        e.feed(dict(prediction(p='.6'),signal_side='yes'))
        e.feed(book(6,bid='.90',no='.09'))
        self.assertFalse(e.positions) # NO looks attractive now, but signal was YES.

    def test_rejected_signal_keeps_forecast_for_scoring_without_order(self):
        e=Engine();e.feed(market());e.feed(book())
        e.feed(dict(prediction(),signal_eligible=False))
        self.assertIn('A',e.predictions);self.assertFalse(e.pending)
        e.feed(book(6));self.assertFalse(e.positions)

    def test_newer_rejected_forecast_cancels_pending_intent(self):
        e=Engine();e.feed(market());e.feed(book());e.feed(prediction())
        self.assertTrue(e.pending)
        e.feed(dict(prediction(seconds=1,p='.4'),signal_eligible=False))
        self.assertFalse(e.pending)

    def test_no_trade_policy_stays_no_trade_regardless_of_test_labels(self):
        data=rows(3)
        selection=select_policy(data,[.4]*len(data))
        self.assertEqual(selection['action'],'no_trade')
        self.assertEqual(evaluate_policy(data,[.99]*len(data),selection)['trades'],0)
