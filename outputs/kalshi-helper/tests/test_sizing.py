import unittest
from helper.sizing import simulate,intervals
from helper.core import Engine,Settings
from test_research import rows
from test_engine import market,book,prediction,outcome

class SizingTests(unittest.TestCase):
    def test_rejects_silent_truncation_invalid_probability_and_backdated_fill(self):
        data=rows(1)[:1]
        for probabilities in [[],[.5,.6],[1.2],[float('nan')]]:
            with self.assertRaises(ValueError):simulate(data,probabilities)
        with self.assertRaises(ValueError):simulate([], [.5])
        with self.assertRaisesRegex(ValueError,'precede prediction'):
            simulate([dict(data[0],execution_at='2020-01-01T00:00:00Z')],[.9])
        same=dict(data[0],execution_at=data[0]['at'],execution_bid=data[0]['bid'],execution_ask=data[0]['ask'])
        self.assertGreater(simulate([same],[.9],slippage=0)['entries'],0)
    def test_city_day_budget_is_shared_across_different_event_ids(self):
        data=[]
        for i in range(6):
            r=dict(rows(1)[0],ticker=str(i),event='different-'+str(i),y=1)
            data.append(r)
        result=simulate(data,[.9]*6,policy='fixed_10_dollars',slippage=0)
        self.assertLessEqual(result['max_bet_cost'],10)
        self.assertLessEqual(result['max_city_day_cost'],30)
        self.assertLessEqual(result['capital_deployed'],30)
        self.assertEqual(result['unsettled_positions'],0)
        self.assertAlmostEqual(result['pnl'],sum(t['pnl'] for t in result['ledger']))
        self.assertAlmostEqual(result['return_on_deployed'],result['pnl']/result['capital_deployed'])

    def test_quantity_does_not_inflate_independent_events(self):
        data=rows(1)[:1];data[0]['y']=1
        a=simulate(data,[.9],policy='fixed_10_dollars',slippage=0,max_contracts=1)
        b=simulate(data,[.9],policy='fixed_10_dollars',slippage=0,max_contracts=100)
        self.assertGreater(b['contracts'],a['contracts'])
        self.assertEqual(a['traded_city_days'],b['traded_city_days'])
        self.assertTrue(intervals(data,b)['sparse'])

    def test_engine_bet_cap_and_settled_deployment_metrics(self):
        e=Engine();e.feed(market());e.feed(book());e.feed(prediction());e.feed(book(6))
        self.assertLessEqual(float(e.positions['A']['cost']),10)
        self.assertIsNone(e.report()['realized_return_on_deployed'])
        e.feed(outcome());r=e.report()
        self.assertAlmostEqual(float(r['realized_return_on_deployed']),float(r['realized_pnl'])/float(r['settled_capital_deployed']))
        with self.assertRaises(ValueError):Settings(bet_fraction='2')
