import unittest
from helper.btc_loss_audit import attribute
from helper.core import dec,fee_and_cost


class LossAuditTests(unittest.TestCase):
    def test_exact_loss_waterfall_and_side_reconstruction(self):
        for side in ['yes','no']:
            row=dict(ticker='A',y=0 if side=='yes' else 1,p=.39,bid=.38,ask=.4)
            price=.4 if side=='yes' else .62
            fee,cost=fee_and_cost(dec(str(price))+dec('.02'),10,dec('.07'))
            ledger=[dict(ticker='A',day='2026-09-01',quantity=10,cost=float(cost),fees=float(fee),pnl=-float(cost))]
            result=attribute([row],[.9 if side=='yes' else .1],ledger)
            self.assertEqual(result['trades'][0]['side'],side)
            s=result['summary']
            self.assertAlmostEqual(s['gross_midpoint_pnl']-s['spread_cost']-s['slippage_cost']-s['fees'],s['net_pnl'])
            self.assertAlmostEqual(s['equal_contract_net'],s['net_pnl']/10)
            self.assertAlmostEqual(s['predicted_win_rate'],.9)

    def test_inconsistent_ledger_is_rejected(self):
        row=dict(ticker='A',y=0,p=.39,bid=.38,ask=.4)
        for ledger in [dict(ticker='A',day='x',quantity=10,cost=4,fees=0,pnl=-2),
                       dict(ticker='A',day='x',quantity=10,cost=4,fees=0,pnl=-4)]:
            with self.assertRaises(ValueError):attribute([row],[.9],[ledger])

    def test_empty_ledger_and_invalid_probabilities(self):
        self.assertEqual(attribute([],[],[])['summary']['entries'],0)
        with self.assertRaises(ValueError):attribute([{}],[float('nan')],[])
