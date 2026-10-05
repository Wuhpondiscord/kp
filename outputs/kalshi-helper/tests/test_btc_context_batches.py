import unittest
import numpy as np
from helper.btc_context_batch import ContextAnchor, inner_split, select_and_fit
from helper.btc_directional_batch import DirectionalAnchor
from helper.core import stamp


class ContextBatchTests(unittest.TestCase):
    def rows(self):
        rng=np.random.default_rng(22)
        return [dict(group=f'2026-06-{20+i//20:02}',at=f'2026-06-{20+i//20:02}T{i%20:02}:00:00Z',
            settled_at=f'2026-06-{20+i//20:02}T{i%20:02}:10:00Z',p=.1+.8*rng.random(),y=i%2,
            x=rng.normal(size=10).tolist(),flow_x=rng.normal(size=4).tolist()) for i in range(160)]

    def test_nested_split_purges_late_labels_and_embargo(self):
        rows=self.rows();rows[0]['settled_at']='2026-06-26T03:00:00Z'
        fit,validation,boundary=inner_split(rows)
        self.assertEqual(boundary,'2026-06-26');self.assertNotIn(rows[0],fit)
        self.assertLess(max(stamp(r['settled_at']) for r in fit),min(stamp(r['at']) for r in validation))
        self.assertTrue(all(stamp(r['at'])>=stamp('2026-06-26T02:00:00Z') for r in validation))

    def test_context_artifact_and_inference_immutability(self):
        rows=self.rows()
        for kind in ('spot','combined','interactions'):
            model=ContextAnchor(kind).fit(rows);saved=model.artifact()
            np.testing.assert_allclose(model.predict(rows),ContextAnchor.load(saved).predict(rows))
            changed=[dict(r,y=1-r['y'],x=[999]*10) for r in rows]
            model.predict(changed);self.assertEqual(saved,model.artifact())

    def test_nested_choice_matches_recorded_inner_scores(self):
        model,selection=select_and_fit(self.rows(),'combined')
        chosen=min(selection['candidates'],key=lambda r:(r['scores']['log_loss'],-r['penalty']))['penalty']
        self.assertEqual(model.penalty,chosen)
        self.assertLess(selection['fit_rows']+selection['validation_rows'],len(self.rows()))

    def test_directional_complement_and_zero_anchor(self):
        rows=self.rows();mirrored=[];neutral=[]
        for r in rows:
            x=np.array(r['x']);f=np.array(r['flow_x']);x[:5]*=-1;f[[0,1,3]]*=-1
            mirrored.append(dict(r,p=1-r['p'],x=x.tolist(),flow_x=f.tolist()))
            z=np.array(r['x']);z[:5]=0;g=np.array(r['flow_x']);g[[0,1,3]]=0
            neutral.append(dict(r,x=z.tolist(),flow_x=g.tolist()))
        for kind in ('flow','price','joint'):
            model=DirectionalAnchor(kind).fit(rows)
            np.testing.assert_allclose(model.predict(mirrored),1-model.predict(rows),atol=1e-12)
            np.testing.assert_allclose(model.predict(neutral),[r['p'] for r in rows],atol=1e-12)
            np.testing.assert_allclose(DirectionalAnchor.load(model.artifact()).predict(rows),model.predict(rows))

    def test_directional_ignores_absolute_basis(self):
        rows=self.rows();model=DirectionalAnchor('joint').fit(rows)
        altered=[dict(r,flow_x=[r['flow_x'][0],r['flow_x'][1],999,r['flow_x'][3]]) for r in rows]
        np.testing.assert_allclose(model.predict(rows),model.predict(altered))
