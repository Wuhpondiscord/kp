import unittest
import numpy as np
from helper.btc_stability_repair import expected_day_counts, weighted_fit, repaired_models
from helper.btc_anchor_study import FixedAnchor


class StabilityRepairTests(unittest.TestCase):
    def rows(self):
        rng=np.random.default_rng(9)
        return [dict(group=f'2026-06-{20+i//20:02}',p=.1+.8*rng.random(),y=i%2,flow_x=rng.normal(size=4).tolist()) for i in range(100)]

    def test_expected_counts_include_odd_truncation(self):
        for n in (4,5,8):
            days=[f'2026-06-{20+i:02}' for i in range(n)];expected=expected_day_counts(days)
            self.assertAlmostEqual(sum(expected.values()),n)
            # Enumerate all equally probable starting positions for one full block
            # and (for odd n) the final truncated block.
            actual={d:0. for d in days}
            for start in range(n-1):
                for d in days[start:start+2]:actual[d]+=n//2/(n-1)
                if n%2:actual[days[start]]+=1/(n-1)
            np.testing.assert_allclose(list(expected.values()),list(actual.values()))
        counts=list(expected_day_counts([f'2026-06-{20+i:02}' for i in range(8)]).values())
        self.assertAlmostEqual(counts[0]*2,counts[1]);self.assertAlmostEqual(counts[-1]*2,counts[-2])

    def test_uniform_weights_reproduce_anchor(self):
        rows=self.rows()
        np.testing.assert_allclose(weighted_fit(rows,np.ones(len(rows))).predict(rows),FixedAnchor().fit(rows).predict(rows),atol=1e-10)
        with self.assertRaises(ValueError):weighted_fit(rows,[0]*len(rows))

    def test_weighted_loss_matches_duplicated_rows_with_fixed_scaling(self):
        rows=self.rows();x=np.array([r['flow_x'] for r in rows]);mean=x.mean(0);scale=x.std(0)
        weights=np.array([2 if i%3==0 else 1 for i in range(len(rows))])
        copied=[r for r,w in zip(rows,weights) for _ in range(w)]
        a=weighted_fit(rows,weights,mean,scale);b=weighted_fit(copied,np.ones(len(copied)),mean,scale)
        np.testing.assert_allclose(a.predict(rows),b.predict(rows),atol=1e-10)

    def test_shared_origin_and_samples_remain_training_only(self):
        rows=self.rows();days=sorted({r['group'] for r in rows});samples=[days,days[:2]+days[:3]]
        members=repaired_models(rows,samples,True,True);mean=np.array([r['flow_x'] for r in rows]).mean(0)
        neutral=[dict(r,flow_x=mean.tolist()) for r in rows]
        for m in members:np.testing.assert_allclose(m.predict(neutral),[r['p'] for r in rows],atol=1e-12)
        with self.assertRaises(ValueError):repaired_models(rows,[days[:-1]+['2026-07-01']],True,True)
