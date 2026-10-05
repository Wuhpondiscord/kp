from pathlib import Path
import tempfile
import unittest
import numpy as np
from helper.btc_tabpfn_research import select_weight,run
from helper.btc_tabpfn_anchor import fit_weight,predict


class PretrainedResearchTests(unittest.TestCase):
    def test_blend_selects_signal_or_market_on_known_examples(self):
        y=np.array([0,1]*30);market=np.full(60,.5);good=.1+.8*y
        self.assertEqual(select_weight(market,good,y),1)
        self.assertEqual(select_weight(market,1-good,y),0)
        self.assertEqual(select_weight(market,market,y),0)
        with self.assertRaises(ValueError):select_weight([.5],[float('nan')],[1])
        with self.assertRaises(ValueError):select_weight([.5,.5],[.5],[1])

    def test_blend_never_passes_with_worse_inner_brier(self):
        rng=np.random.default_rng(5)
        for _ in range(10):
            m=rng.uniform(.01,.99,50);p=rng.uniform(.01,.99,50);y=rng.integers(0,2,50)
            w=select_weight(m,p,y)
            self.assertLessEqual(np.mean(((1-w)*m+w*p-y)**2),np.mean((m-y)**2)+1e-12)

    def test_bad_checkpoint_rejected_before_optional_model_import(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=root/'checkpoint';p.write_bytes(b'wrong')
            with self.assertRaisesRegex(ValueError,'Checkpoint hash'):run(root,root,p,root/'output')

    def test_anchor_bounds_identity_and_label_response(self):
        y=np.array([0,1]*30);m=np.full(60,.5);good=.1+.8*y
        self.assertGreater(fit_weight(m,good,y),0)
        self.assertEqual(fit_weight(m,1-good,y),0)
        w=fit_weight(m,good,y);self.assertLessEqual(w,.25)
        np.testing.assert_allclose(predict(m,good,0),m)
        np.testing.assert_allclose(predict(m,m,.25),m)
        with self.assertRaises(ValueError):predict(m,good,.5)
