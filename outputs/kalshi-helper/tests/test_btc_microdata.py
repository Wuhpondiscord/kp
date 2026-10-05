import io
from pathlib import Path
import tempfile
import unittest
import zipfile
import numpy as np
from helper.btc_microdata import parse_seconds,micro_features,attach,MicroAnchor,archive
from helper.btc_research import unix,iso


class MicrodataTests(unittest.TestCase):
    def bars(self):
        end=unix('2026-07-01')+120
        return end,{t:dict(close=100,volume=10,buy=6,trades=2) for t in range(end-121,end)}

    def test_only_completed_available_seconds_enter(self):
        end,bars=self.bars();at=iso(end+5);r=micro_features(bars,at)
        np.testing.assert_allclose(r['micro_x'],[.2,0,0,0],atol=1e-12)
        self.assertEqual(r['micro_available_at'],at)
        bars[end]=dict(close=999,volume=100,buy=0,trades=20)
        self.assertEqual(micro_features(bars,at),r)
        bars.pop(end-1)
        with self.assertRaisesRegex(ValueError,'contiguous'):micro_features(bars,at)

    def test_delay_and_zero_trading_are_not_future_or_missing(self):
        end,bars=self.bars();at=iso(end+5);before=micro_features(bars,at,60)
        bars[end-1]=dict(close=120,volume=20,buy=20,trades=10)
        self.assertEqual(before,micro_features(bars,at,60))
        self.assertNotEqual(before['micro_x'],micro_features(bars,at)['micro_x'])
        for r in bars.values():r.update(volume=0,buy=0,trades=0)
        self.assertEqual(micro_features(bars,at)['micro_x'][0],0)
        self.assertEqual(micro_features(bars,at)['trades_60s'],0)

    def test_full_day_parser_and_timestamp_units(self):
        start=unix('2026-07-01');stream=io.BytesIO()
        lines=[f'{t*1000000},100,100,100,100,10,{t*1000000+999999},1000,2,6,600,0\n' for t in range(start,start+86400)]
        with zipfile.ZipFile(stream,'w',zipfile.ZIP_DEFLATED) as z:z.writestr('bars.csv',''.join(lines))
        result=parse_seconds(stream.getvalue(),'2026-07-01',{start,start+86399})
        self.assertEqual(len(result),2);self.assertEqual(result[start]['buy'],6)
        stream=io.BytesIO()
        with zipfile.ZipFile(stream,'w') as z:z.writestr('bars.csv',lines[0]+lines[0])
        with self.assertRaisesRegex(ValueError,'timestamp'):parse_seconds(stream.getvalue(),'2026-07-01',{start})

    def test_future_attachment_and_bad_checksum_rejected(self):
        end,bars=self.bars();at=iso(end+5);f=micro_features(bars,at)
        with self.assertRaisesRegex(ValueError,'availability'):attach([dict(ticker='x',at=iso(end+4))],{'x':{'0':f}})
        with tempfile.TemporaryDirectory() as d:
            path=Path(d,'BTCUSDT-1s-2026-07-01.zip');path.write_bytes(b'wrong')
            Path(str(path)+'.CHECKSUM').write_text('0'*64)
            with self.assertRaisesRegex(ValueError,'checksum'):archive(d,'2026-07-01',offline=True)

    def test_model_serialization_and_scaler_frozen_at_inference(self):
        rng=np.random.default_rng(3)
        rows=[dict(p=.1+.8*rng.random(),y=i%2,flow_x=rng.normal(size=4).tolist(),micro_x=rng.normal(size=4).tolist()) for i in range(100)]
        for combined in (False,True):
            m=MicroAnchor(combined).fit(rows);saved=m.artifact()
            np.testing.assert_allclose(m.predict(rows),MicroAnchor.load(saved).predict(rows))
            m.predict([dict(r,micro_x=[100]*4) for r in rows]);self.assertEqual(saved,m.artifact())
