import unittest
from helper.btc_execution_frontier import tolerance, fixed_cohort
from helper.core import dec, fee_and_cost


class ExecutionFrontierTests(unittest.TestCase):
    def test_tolerance_is_last_passing_price_tick(self):
        for win,price in [(.6,.5),(.55,.51),(.9,.1),(1,.5),(.3,.5)]:
            maximum=tolerance(win,price)
            _,initial=fee_and_cost(dec(price),1,dec('.07'))
            if maximum is None:
                self.assertLess(dec(win)-initial,dec('.04'));continue
            _,cost=fee_and_cost(dec(price)+dec(maximum),1,dec('.07'))
            self.assertGreaterEqual(dec(win)-cost,dec('.04'))
            if dec(price)+dec(maximum)+dec('.0001')<=1:
                _,next_cost=fee_and_cost(dec(price)+dec(maximum)+dec('.0001'),1,dec('.07'))
                self.assertLess(dec(win)-next_cost,dec('.04'))
        for win,price in [(float('nan'),.5),(.5,-1),(1.1,.5)]:
            with self.assertRaises(ValueError):tolerance(win,price)

    def test_fixed_cohort_reconciles_and_costs_are_monotonic(self):
        rows=[dict(ticker='a',p=.5,bid=.49,ask=.51,y=1),dict(ticker='b',p=.5,bid=.49,ask=.51,y=1)]
        base=dict(ledger=[dict(ticker='a',quantity=10,cost=5.28,fees=.18,pnl=4.72),
                          dict(ticker='b',quantity=10,cost=5.28,fees=.18,pnl=-5.28)])
        zero=fixed_cohort(rows,[.8,.2],base,0)
        self.assertAlmostEqual(zero['pnl'],-.56)
        previous=zero
        for slip in (.005,.01,.02,.05):
            current=fixed_cohort(rows,[.8,.2],base,slip)
            self.assertEqual(current['contracts'],20)
            self.assertLessEqual(current['pnl'],previous['pnl'])
            self.assertLessEqual(current['expected_net'],previous['expected_net'])
            previous=current

    def test_tolerance_increases_with_probability_and_falls_with_price(self):
        self.assertGreater(tolerance(.70,.5),tolerance(.60,.5))
        self.assertLess(tolerance(.70,.55),tolerance(.70,.5))
        self.assertIsNone(tolerance(.51,.5))
        self.assertEqual(fixed_cohort([],[],{'ledger':[]},.02)['entries'],0)
