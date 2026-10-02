import unittest
from helper.strategy_research import rejection_reasons

class StrategyRejectionTests(unittest.TestCase):
    def test_positive_sparse_profit_does_not_pass(self):
        r=dict(traded_city_days=10,pnl=5,uncertainty=dict(net_per_contract=[.01,.1]))
        self.assertTrue(rejection_reasons(r))
    def test_loss_uncertainty_does_not_pass(self):
        r=dict(traded_city_days=40,pnl=5,uncertainty=dict(net_per_contract=[-.01,.1]))
        self.assertTrue(rejection_reasons(r))
    def test_sufficient_positive_bound_passes_numerical_gate(self):
        r=dict(traded_city_days=40,pnl=5,uncertainty=dict(net_per_contract=[.01,.1]))
        self.assertEqual(rejection_reasons(r),[])
