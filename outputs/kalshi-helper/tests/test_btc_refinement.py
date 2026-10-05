import unittest
import numpy as np
from helper.btc_refinement import Correction, logits, split


class RefinementTests(unittest.TestCase):
    def test_zero_correction_is_exact_baseline(self):
        market = np.array([.01,.2,.5,.8,.99]);signal = np.array([.8,.7,.6,.5,.4])
        for kind, width in [('market_calibrated',2),('market_anchored',3),('signal_calibrated',2)]:
            model=Correction(kind);model.coef=np.zeros(width)
            np.testing.assert_allclose(model.predict(market,signal), signal if kind=='signal_calibrated' else market)

    def test_calibration_softens_extreme_wrong_confidence(self):
        signal=np.tile([.01,.99],100);y=np.tile([0,1,1,0],50)
        model=Correction('signal_calibrated').fit(signal,signal,y)
        p=model.predict(signal,signal)
        self.assertGreater(p[0],signal[0]);self.assertLess(p[1],signal[1])
        self.assertGreaterEqual(model.coef[1],-.75)

    def test_prediction_uses_only_inputs_and_stays_bounded(self):
        p=np.tile([.2,.8],50);y=np.tile([0,1],50)
        model=Correction('market_anchored').fit(p,1-p,y)
        self.assertGreaterEqual(model.coef[2],0);self.assertLessEqual(model.coef[2],.5)
        result=model.predict([0,1],[1,0])
        self.assertTrue(np.isfinite(result).all());self.assertTrue(((result>0)&(result<1)).all())
        with self.assertRaises(ValueError):model.predict([.1],[.1,.2])
        with self.assertRaises(ValueError):logits([float('nan')])

    def test_settlement_cutoff_and_two_hour_embargo(self):
        rows=[dict(settled_at='2026-09-06T23:00:00Z',at='2026-09-06T22:50:00Z') for _ in range(50)]
        rows += [dict(settled_at='2026-09-07T03:00:00Z',at='2026-09-07T02:50:00Z') for _ in range(50)]
        rows += [dict(settled_at='2026-09-07T01:00:00Z',at='2026-09-07T00:50:00Z')]
        fit,score=split(rows,'2026-09-07','2026-09-11')
        self.assertEqual(len(fit),50);self.assertEqual(len(score),50)

    def test_invalid_or_one_class_labels_rejected(self):
        for labels in ([0]*100,[2,0]*50,[0,1]):
            with self.assertRaises(ValueError):Correction('market_anchored').fit([.5]*100,[.5]*100,labels)
