import unittest
import numpy as np
from helper.btc_anchor_study import FixedAnchor, delayed_rows, trade_attribution, candidate_gate


class AnchorStudyTests(unittest.TestCase):
    def test_neutral_features_preserve_market_and_roundtrip(self):
        rng=np.random.default_rng(12)
        rows=[dict(p=.1+.8*rng.random(),flow_x=rng.normal(size=4).tolist(),y=i%2) for i in range(100)]
        model=FixedAnchor().fit(rows)
        neutral=[dict(r,flow_x=model.mean.tolist()) for r in rows]
        np.testing.assert_allclose(model.predict(neutral),[r['p'] for r in rows],atol=1e-12)
        self.assertEqual(len(model.coef),4)
        np.testing.assert_allclose(FixedAnchor.load(model.artifact()).predict(rows),model.predict(rows))
        with self.assertRaises(ValueError):FixedAnchor.load(dict(model.artifact(),coef=[1]*4))

    def test_age_changes_inputs_not_market_or_labels(self):
        row=dict(ticker='x',at='2026-07-01T01:00:05Z',p=.6,y=1,flow_x=[9]*4)
        f={'x':{'60':dict(flow_x=[1,2,3,4],flow_feature_time='2026-07-01T00:59:00Z',flow_available_at='2026-07-01T00:59:05Z')}}
        result=delayed_rows([row],f,60)[0]
        self.assertEqual(result['p'],.6);self.assertEqual(result['y'],1)
        self.assertEqual(row['flow_x'],[9]*4);self.assertEqual(result['flow_x'],[1,2,3,4])
        f['x']['60']['flow_available_at']='2026-07-01T01:00:05Z'
        with self.assertRaises(ValueError):delayed_rows([row],f,60)

    def test_fixed_trade_waterfall_reconciles_yes_and_no(self):
        rows=[dict(ticker='yes',p=.5,bid=.49,ask=.51,y=1),dict(ticker='no',p=.5,bid=.49,ask=.51,y=1)]
        paper=dict(entries=2,ledger=[dict(ticker=n,quantity=10,cost=5.48,fees=.18,pnl=pnl) for n,pnl in [('yes',4.52),('no',-5.48)]])
        result=trade_attribution(rows,[.8,.2],paper)
        totals=result['totals']
        self.assertAlmostEqual(totals['realized_net'],-.96)
        self.assertAlmostEqual(totals['midpoint_gross']-totals['spread_cost']-totals['slippage_cost']-totals['fees'],-.96)
        self.assertEqual([t['side'] for t in result['trades']],['yes','no'])
        self.assertEqual(result['funnel']['after_slippage'],2)
        paper['ledger'][0]['pnl']=999
        with self.assertRaisesRegex(ValueError,'attribution'):trade_attribution(rows,[.8,.2],paper)

    def test_gate_requires_both_metrics_and_enough_evidence(self):
        scores=lambda v:dict(scores=dict(brier=v,log_loss=v))
        results=dict(market=scores(.2),market_control=scores(.21),fixed_anchor=scores(.19))
        results['fixed_anchor']['paper']=dict(pnl=10,entries=3,traded_city_days=2)
        folds=[dict(models=results)]*4
        gate=candidate_gate(folds,results)
        self.assertEqual(gate['fold_wins'],4);self.assertFalse(gate['passed'])
        results['fixed_anchor']['paper'].update(entries=30,traded_city_days=10)
        self.assertTrue(candidate_gate(folds,results)['passed'])
        results['fixed_anchor']['scores']['log_loss']=.22
        self.assertFalse(candidate_gate(folds,results)['passed'])
