import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from helper.btc_tree_research import MarketTrees, inventory


class TreeResearchTests(unittest.TestCase):
    def rows(self):
        rng=np.random.default_rng(7)
        return [dict(p=.1+.8*rng.random(),y=i%2,x=rng.normal(size=10).tolist(),flow_x=rng.normal(size=4).tolist()) for i in range(150)]

    def test_market_fallback_exact_and_artifact_roundtrip(self):
        rows=self.rows()
        np.testing.assert_allclose(MarketTrees(0).fit(rows).predict(rows),[r['p'] for r in rows],atol=1e-12)
        model=MarketTrees(8).fit(rows);artifact=model.artifact()
        np.testing.assert_allclose(model.predict(rows),MarketTrees.load(artifact).predict(rows),atol=1e-12)
        changed=[dict(r,y=1-r['y']) for r in rows]
        np.testing.assert_allclose(model.predict(rows),model.predict(changed),atol=1e-12)

    def test_malformed_and_cyclic_tree_rejected(self):
        artifact=MarketTrees(8).fit(self.rows()).artifact();bad=copy.deepcopy(artifact)
        bad['trees'][0]['left'][0]=0
        with self.assertRaises(ValueError):MarketTrees.load(bad)
        bad=copy.deepcopy(artifact);bad['trees'][0]['value'][0]=float('nan')
        with self.assertRaises(ValueError):MarketTrees.load(bad)
        with self.assertRaises(ValueError):MarketTrees.load(dict(artifact,stages=0))

    def test_inventory_preserves_cohorts_without_ranking(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for n,count in [('btc-a',100),('btc-b',200)]:
                folder=root/n;folder.mkdir()
                (folder/'report.json').write_text(json.dumps(dict(aggregate={'candidate':dict(scores=dict(brier=.1,log_loss=.2,events=count,days=5))})))
            inventory(root,root/'ledger.json');r=json.loads((root/'ledger.json').read_text())
            self.assertEqual([e['models']['candidate']['events'] for e in r['experiments']],[100,200])
            self.assertTrue(all(len(e['sha256'])==64 for e in r['experiments']))
            self.assertIn('not a multiplicity correction',r['warning'])
