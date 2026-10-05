import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from helper.btc_flow_followup import Ablation, MASKS, freeze, read_rows, split
from helper.btc_flow_model import FlowCorrection
from helper.btc_new_history import new_features
from helper.btc_research import iso, unix, digest, save


class FlowFollowupTests(unittest.TestCase):
    def rows(self):
        rng=np.random.default_rng(42)
        return [dict(p=.2+.6*rng.random(),y=i%2,flow_x=rng.normal(size=4).tolist()) for i in range(100)]

    def test_full_and_control_reproduce_original(self):
        rows=self.rows()
        for indices,flag in [(MASKS['full_flow'],True),((),False)]:
            np.testing.assert_allclose(Ablation(indices).fit(rows).predict(rows),FlowCorrection(flag).fit(rows).predict(rows))

    def test_excluded_inputs_cannot_affect_prediction(self):
        rows=self.rows();model=Ablation((3,)).fit(rows)
        changed=[dict(r,flow_x=[999,-999,123,r['flow_x'][3]]) for r in rows]
        np.testing.assert_allclose(model.predict(rows),model.predict(changed))
        restored=Ablation.load(model.artifact())
        np.testing.assert_allclose(restored.predict(rows),model.predict(rows))
        with self.assertRaises(ValueError):Ablation((3,3))
        with self.assertRaises(ValueError):Ablation.load(dict(model.artifact(),scale=[0]*4))

    def test_lag_does_not_include_latest_minute(self):
        end=unix('2026-07-01')+3600
        flow={t:dict(close=100,volume=10,buy=5) for t in range(end-3600,end,60)}
        bars={t:[t,99,101,100,100,10] for t in flow}
        old=new_features(flow,bars,iso(end+5-60))
        flow[end-60]=dict(close=200,volume=10,buy=10)
        self.assertEqual(old,new_features(flow,bars,iso(end+5-60)))
        self.assertNotEqual(old['flow_x'],new_features(flow,bars,iso(end+5))['flow_x'])

    def test_split_purges_unsettled_labels_and_embargoes_evaluation(self):
        rows=[dict(close_time=iso(unix('2026-06-27')+i*60),settled_at=iso(unix('2026-06-27')+i*60+60),at=iso(unix('2026-06-27')+i*60-60)) for i in range(60)]
        late=dict(rows[0],settled_at='2026-06-28T01:00:00Z');rows.append(late)
        early=dict(at='2026-06-28T01:00:00Z',close_time='2026-06-28T01:05:00Z',settled_at='2026-06-28T01:06:00Z')
        valid=dict(at='2026-06-28T03:00:00Z',close_time='2026-06-28T03:05:00Z',settled_at='2026-06-28T03:06:00Z')
        train,test=split(rows+[early,valid],'2026-06-28','2026-07-04')
        self.assertEqual(len(train),60);self.assertEqual(test,[valid])

    def test_dataset_hash_and_holdout_date_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);freeze(root);path=root/'development.jsonl'
            path.write_text(json.dumps(dict(close_time='2026-07-21T12:00:00Z'))+'\n')
            save(root/'manifest.json',dict(data_sha256=digest(path.read_bytes()),protocol_sha256=digest((root/'protocol.json').read_bytes())))
            with self.assertRaisesRegex(ValueError,'Outside development'):read_rows(root)
            path.write_text('{}')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):read_rows(root)
