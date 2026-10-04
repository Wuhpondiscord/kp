import json
from pathlib import Path
import tempfile
import unittest

from helper.btc_diagnostics import edge_summary, load_bars, source_summary
from helper.btc_research import digest, iso


class BTCDiagnosticsTests(unittest.TestCase):
    def row(self, **kw):
        return dict(dict(ticker='A',group='1970-01-01',close_time=iso(1800),strike=100,y=1,
            p=.5,bid=.49,ask=.51,spread=.02,x=[0,0,0,0,0,1,0,0,0,0]), **kw)

    def bars(self):
        return {840:[840,99,101,100,100,10],1740:[1740,99,101,100,99.5,10]}

    def test_close_mismatch_does_not_imply_whole_range_mismatch(self):
        r=source_summary([self.row()],self.bars())
        self.assertEqual(r['close_label_disagreements'],1)
        self.assertEqual(r['whole_range_opposes_outcome'],0)

    def test_whole_range_mismatch_and_tie_semantics(self):
        bars=self.bars();bars[1740]=[1740,100,101,100,100,10]
        self.assertEqual(source_summary([self.row(y=0)],bars)['whole_range_opposes_outcome'],1)
        self.assertEqual(source_summary([self.row(y=1)],bars)['close_label_disagreements'],0)

    def test_missing_rows_are_counted_and_invalid_candles_rejected(self):
        bars=self.bars()
        r=source_summary([self.row(),self.row(ticker='B',close_time=iso(2700))],bars)
        self.assertEqual(r['missing'],1);self.assertEqual(r['events'],1)
        bars[1740][4]=1000
        with self.assertRaisesRegex(ValueError,'Invalid audit'):source_summary([self.row()],bars)

    def test_edge_bins_use_prediction_but_realized_returns_use_outcome(self):
        yes=edge_summary([self.row(y=1)],[.6]);no=edge_summary([self.row(y=0)],[.6])
        self.assertEqual(yes['0.04-0.08']['opportunities'],1)
        self.assertEqual(yes['0.04-0.08']['predicted_net_per_contract'],no['0.04-0.08']['predicted_net_per_contract'])
        self.assertAlmostEqual(yes['0.04-0.08']['realized_net_per_contract']-no['0.04-0.08']['realized_net_per_contract'],1)
        with self.assertRaises(ValueError):edge_summary([self.row()],[])

    def test_cached_sources_must_match_hash(self):
        with tempfile.TemporaryDirectory() as d:
            raw=json.dumps([[0,99,101,100,100,10]]).encode()
            Path(d,'a.json').write_text(json.dumps(dict(url='https://api.exchange.coinbase.com/products/BTC-USD/candles',sha256=digest(raw))))
            Path(d,'a.raw').write_bytes(raw)
            self.assertIn(0,load_bars(d)[0])
            Path(d,'a.raw').write_bytes(b'[]')
            with self.assertRaisesRegex(ValueError,'integrity'):load_bars(d)
