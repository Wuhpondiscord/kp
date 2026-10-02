import unittest
import numpy as np
from scipy.optimize import linprog
from helper.polymarket import quote_bounds,cdf_bounds,regional_features

class IntervalTests(unittest.TestCase):
    def test_one_sided_books_do_not_invent_midpoints(self):
        r=quote_bounds(dict(bids=[],asks=[dict(price='.01',size='5')]))
        self.assertIsNone(r['p']);self.assertEqual((r['bid'],r['ask']),(0,.01))
        self.assertEqual(quote_bounds(dict(bids=[],asks=[]))['quality'],'empty')
    def test_bounds_match_independent_linear_programs(self):
        rng=np.random.default_rng(7)
        for _ in range(15):
            p=rng.dirichlet(np.ones(5));lo=(p*rng.random(5)).tolist();hi=(p+(1-p)*rng.random(5)).tolist()
            lower,upper=cdf_bounds(lo,hi)
            for i in range(5):
                c=np.array([1. if j<=i else 0 for j in range(5)])
                a=linprog(c,A_eq=[np.ones(5)],b_eq=[1],bounds=list(zip(lo,hi)),method='highs')
                b=linprog(-c,A_eq=[np.ones(5)],b_eq=[1],bounds=list(zip(lo,hi)),method='highs')
                self.assertTrue(a.success and b.success);self.assertAlmostEqual(lower[i],a.fun);self.assertAlmostEqual(upper[i],-b.fun)
    def test_infeasible_or_invalid_quotes_fail(self):
        with self.assertRaises(ValueError):cdf_bounds([.8,.8],[1,1])
        with self.assertRaises(ValueError):quote_bounds(dict(bids=[dict(price='nan',size='1')],asks=[]))
    def test_unknown_tails_can_preserve_constrained_median(self):
        s=dict(captured_at='2026-09-24T12:00:00Z',series='KXHIGHNY',date='2026-09-24',markets=[
            dict(lower=None,upper=60,bid=0,ask=1,p=None,spread=1),dict(lower=60,upper=64,bid=.8,ask=.9,p=.85,spread=.1),
            dict(lower=64,upper=None,bid=0,ask=1,p=None,spread=1)])
        r=regional_features(s,'KXHIGHNY-26SEP24-T70','2026-09-24T12:00:00Z')
        self.assertTrue(r['poly_available']);self.assertEqual(r['poly_median_low'],60);self.assertEqual(r['poly_median_high'],64)
        self.assertAlmostEqual(r['poly_mean_spread'],.1)
        self.assertAlmostEqual(r['poly_mean_quote_interval_width'],.7)
        s['markets'][1].update(bid=0,ask=1,p=None,spread=1)
        self.assertFalse(regional_features(s,'KXHIGHNY-26SEP24-T70','2026-09-24T12:00:00Z')['poly_available'])
