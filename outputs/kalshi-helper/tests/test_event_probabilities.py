import math,unittest
import numpy as np
from helper.event_probabilities import project_event
from helper.fusion_blend import FusionBlend
from helper.observation_calibration import load_training_calibration

class Fixed:
    def __init__(self,p):self.p=p
    def predict(self,rows,paths):return np.asarray(self.p)

class EventProbabilityTests(unittest.TestCase):
    def test_station_identity_can_be_in_secondary_settlement_rules(self):
        from helper.weather_model import supported
        m=dict(rules_primary='maximum temperature New York City according to The Weather Company',rules_secondary='Data for CLINYC: Central Park NY')
        self.assertTrue(supported(m,'KXHIGHNY'))
        self.assertFalse(supported(m,'KXHIGHCHI'))
        self.assertFalse(supported(dict(m,rules_secondary=''), 'KXHIGHNY'))
    def rows(self):
        return [dict(event='e',series='s',at='2026-04-04T15:00:00Z',ticker=str(i),lower=a,upper=b) for i,(a,b) in enumerate([(-math.inf,70),(70,75),(75,math.inf)])]
    def test_partition_normalizes_without_labels_and_preserves_order(self):
        rows=self.rows();p=project_event(rows,[.2,.3,.7])
        self.assertAlmostEqual(sum(p),1)
        np.testing.assert_allclose(project_event(rows[::-1],[.7,.3,.2]),p[::-1])
        np.testing.assert_array_equal(project_event([dict(r,y=1) for r in rows],[.2,.3,.7]),p)
    def test_incomplete_overlapping_duplicate_and_asynchronous_fail_closed(self):
        rows=self.rows()
        for bad in [rows[1:],rows[:2],rows+[rows[0]],[dict(rows[0],upper=71)]+rows[1:],[dict(rows[0],at='2026-04-04T15:01:00Z')]+rows[1:]]:
            with self.assertRaises(ValueError):project_event(bad,[.3]*len(bad))
    def test_bad_probabilities_rejected(self):
        for p in [[0,0,0],[.1,float('nan'),.8],[-.1,.3,.8],[.2,.3]]:
            with self.assertRaises(ValueError):project_event(self.rows(),p)
    def test_blend_preserves_declared_weight_after_parent_normalization(self):
        a=[.2,.3,.7];b=[.1,.2,.3];m=FusionBlend([Fixed(a)]*3,[Fixed(b)]*3,.6)
        expected=.6*np.array(a)/sum(a)+.4*np.array(b)/sum(b)
        np.testing.assert_allclose(m.predict_event(self.rows(),None),expected)
        np.testing.assert_allclose(m.predict(self.rows(),None),.6*np.array(a)+.4*np.array(b))
    def test_calibration_requires_explicit_legacy_opt_in(self):
        self.assertIsNone(load_training_calibration(legacy=True))
        with self.assertRaisesRegex(ValueError,'missing'):load_training_calibration('nonexistent-calibration.json')
        with self.assertRaises(ValueError):load_training_calibration('anything',legacy=True)
