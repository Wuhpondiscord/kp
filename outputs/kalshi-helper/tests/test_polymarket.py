import unittest
from helper.polymarket import bracket,midpoint,regional_features,read_json
from helper.hierarchy import HierarchicalOffset
from helper.poly_ablation import compare
import tempfile

class PolymarketTests(unittest.TestCase):
    def test_bounds_and_unsupported_units(self):
        self.assertEqual(bracket('58-59°F'),(57.5,59.5))
        self.assertEqual(bracket('57°F or below'),(None,57.5))
        with self.assertRaises(ValueError):bracket('20°C')
    def test_book_not_assumed_sorted(self):
        self.assertEqual(midpoint(dict(bids=[dict(price='.2',size='1'),dict(price='.3',size='2')],asks=[dict(price='.5',size='1'),dict(price='.4',size='1')]))['p'],.35)
    def test_order_endpoint_blocked(self):
        with self.assertRaises(ValueError):read_json('https://clob.polymarket.com/order')
    def test_future_and_mismatched_day_rejected(self):
        s=dict(captured_at='2026-09-24T12:00:00Z',series='KXHIGHNY',date='2026-09-24',markets=[])
        with self.assertRaises(ValueError):regional_features(s,'KXHIGHNY-26SEP24-T70','2026-09-24T11:00:00Z')
        with self.assertRaises(ValueError):regional_features(s,'KXHIGHNY-26SEP25-T70','2026-09-24T12:00:00Z')
        self.assertFalse(regional_features(s,'KXHIGHNY-26SEP24-T70','2026-09-24T12:00:00Z')['poly_available'])
    def test_complete_bracket_context_is_not_exact_match(self):
        s=dict(captured_at='2026-09-24T12:00:00Z',series='KXHIGHNY',date='2026-09-24',markets=[
            dict(lower=None,upper=60,p=.1,spread=.02),dict(lower=60,upper=70,p=.8,spread=.02),dict(lower=70,upper=None,p=.1,spread=.02)])
        f=regional_features(s,'KXHIGHNY-26SEP24-T70','2026-09-24T12:00:00Z')
        self.assertEqual(f['poly_median_low'],60);self.assertEqual(f['poly_relation'],'related_only')
    def test_ablation_does_not_invent_missing_data(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertFalse(compare(d,d+'/run')['performance_comparison_available'])
    def test_hierarchy_unseen_city_uses_shared_component(self):
        rows=[dict(p=.2+i*.01,hours_left=18,series='KXHIGHNY',group=str(i//4),y=i%3==0) for i in range(80)]
        m=HierarchicalOffset().fit(rows)
        _,x=m.inputs([dict(rows[0],series='unseen')]);self.assertTrue((x[0,3:]==0).all())
