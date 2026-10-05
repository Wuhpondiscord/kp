import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from helper.btc_new_history import new_features, freeze, parse_flow, history_markets
from helper.btc_flow_model import FlowCorrection, single_fill_cost, run, trade_uncertainty


class BTCFlowTests(unittest.TestCase):
    def test_imbalance_sign_and_completed_candle_boundary(self):
        at='2026-07-08T12:00:05Z'
        from helper.core import stamp
        last=int(stamp(at).timestamp())-65
        flow={t:dict(close=100,volume=10,buy=8) for t in range(last-14*60,last+1,60)}
        bars={t:[t,99,101,100,100,10] for t in flow}
        flow[last+60]=dict(close=999,volume=100,buy=0)  # Unfinished next minute must not enter.
        r=new_features(flow,bars,at)
        np.testing.assert_allclose(r['flow_x'],[.6,.6,0,0])
        self.assertEqual(r['flow_available_at'],'2026-07-08T12:00:05+00:00')
        flow.pop(last)
        with self.assertRaises(ValueError):new_features(flow,bars,at)

    def test_single_fill_fee_precision(self):
        self.assertEqual(single_fill_cost(.5,1),(.02,.52))
        self.assertEqual(single_fill_cost(.5,1,precision='.0001'),(.0175,.5175))
        self.assertEqual(single_fill_cost(.5,100),(1.75,51.75))
        for price,quantity in [(float('nan'),1),(.5,0),(.5,1.5),(2,1)]:
            with self.assertRaises(ValueError):single_fill_cost(price,quantity)

    def test_artifact_roundtrip_and_training_only_scaler(self):
        rows=[dict(p=.2 if i%2==0 else .8,y=i%2,flow_x=[i%3,i%5,i%7,i%2]) for i in range(100)]
        model=FlowCorrection().fit(rows);before=model.mean.copy()
        changed=[dict(r,flow_x=[1000]*4) for r in rows]
        model.predict(changed);np.testing.assert_equal(before,model.mean)
        restored=FlowCorrection.load(model.artifact())
        np.testing.assert_allclose(restored.predict(rows),model.predict(rows))
        self.assertTrue((np.abs(model.coef)<=.25).all())

    def test_market_control_cannot_use_flow(self):
        rows=[dict(p=.2 if i%2==0 else .8,y=i%2,flow_x=[i%3]*4) for i in range(100)]
        model=FlowCorrection(False).fit(rows)
        np.testing.assert_allclose(model.predict(rows),model.predict([dict(r,flow_x=[999]*4) for r in rows]))

    def test_failed_gate_never_reads_holdout(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d,'validation-report.json').write_text(json.dumps(dict(holdout_release_eligible=False)))
            with patch('helper.btc_flow_model.read_partition',side_effect=AssertionError('Must not open')):
                with self.assertRaisesRegex(ValueError,'remains sealed'):run(d,True)

    def test_protocol_cannot_change_in_place(self):
        with tempfile.TemporaryDirectory() as d:
            freeze(d);Path(d,'protocol.json').write_text('{}')
            with self.assertRaises(ValueError):freeze(d)

    def test_archive_rejects_taker_volume_larger_than_total(self):
        import io,zipfile
        from helper.btc_research import unix
        t=unix('2026-07-01');out=io.BytesIO()
        with zipfile.ZipFile(out,'w') as z:
            z.writestr('prices.csv',f'{t*1000000},100,102,99,101,10,{(t+60)*1000000-1},1000,1,11,1100,0')
        with self.assertRaisesRegex(ValueError,'taker-buy'):parse_flow(out.getvalue(),'2026-07')

    def test_archive_pagination_cannot_silently_repeat(self):
        class Fake:
            def get(self,url):return dict(markets=[],cursor='same')
        with self.assertRaisesRegex(ValueError,'Repeated'):history_markets(Fake())

    def test_sparse_winners_do_not_create_confident_profit_interval(self):
        result=trade_uncertainty(dict(entries=1,ledger=[dict(day='2026-07-16',pnl=1,quantity=2,cost=1)]),{'2026-07-16'})
        self.assertIsNone(result['net_per_contract_95_interval'])
        self.assertEqual(result['status'],'insufficient_trade_evidence')
