import unittest
import numpy as np
from helper.btc_stability_model import sample_days, StabilityAnchor, guarded_probabilities


class StabilityTests(unittest.TestCase):
    def rows(self):
        rng=np.random.default_rng(5)
        return [dict(group=f'2026-06-{20+i//20:02}',ticker=str(i),p=.2+.6*rng.random(),y=i%2,
                     flow_x=rng.normal(size=4).tolist()) for i in range(120)]

    def test_blocks_preserve_day_rows_and_seed(self):
        rows=self.rows();sample,days=sample_days(rows,np.random.default_rng(1729))
        other,otherdays=sample_days(rows,np.random.default_rng(1729))
        self.assertEqual(sample,other);self.assertEqual(days,otherdays)
        self.assertEqual(len(days),6)
        self.assertEqual(len(sample),120)
        for a,b in zip(days[::2],days[1::2]):self.assertEqual(int(b[-2:])-int(a[-2:]),1)
        flipped=[dict(r,y=1-r['y']) for r in rows]
        _,flipped_days=sample_days(flipped,np.random.default_rng(1729))
        self.assertEqual(days,flipped_days)
        with self.assertRaises(ValueError):sample_days([dict(group='2026-06-01'),dict(group='2026-06-04')],np.random.default_rng(1))

    def test_guard_cannot_flip_or_exaggerate_signal(self):
        rows=[dict(p=.5)]*3
        distribution=np.array([[.51,.1,.7],[.9,.49,.7],[.4,.6,.7]])
        guarded=guarded_probabilities(rows,distribution);mean=distribution.mean(axis=0)
        self.assertTrue(np.all(guarded>=np.minimum(.5,mean)))
        self.assertTrue(np.all(guarded<=np.maximum(.5,mean)))
        self.assertEqual(guarded[0],.5);self.assertEqual(guarded[1],.5)
        self.assertAlmostEqual(guarded[2],.7)
        with self.assertRaises(ValueError):guarded_probabilities(rows,[[float('nan')]*3]*2)

    def test_ensemble_roundtrip_and_predict_does_not_refit(self):
        rows=self.rows();model=StabilityAnchor().fit(rows)
        before=model.artifact();loaded=StabilityAnchor.load(before)
        np.testing.assert_allclose(model.predict(rows),loaded.predict(rows),atol=1e-12)
        model.predict([dict(r,y=1-r['y'],flow_x=[100]*4) for r in rows])
        self.assertEqual(model.artifact(),before)
        with self.assertRaises(ValueError):StabilityAnchor.load(dict(before,models=before['models'][:1]))
