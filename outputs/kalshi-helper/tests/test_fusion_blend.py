import unittest
import numpy as np
from helper.fusion_blend import FusionBlend,SeedEnsemble

class Member:
    def __init__(self,p):self.p=p
    def predict(self,rows,paths):return np.full(len(rows),self.p)

class FusionBlendTests(unittest.TestCase):
    def test_weighted_blend_endpoints_and_nested_user_example(self):
        parents=([Member(.2)]*3,[Member(.8)]*3)
        for weight,expected in [(0,.8),(1,.2),(.7,.38),(.85,.29)]:
            np.testing.assert_allclose(FusionBlend(*parents,weather_weight=weight).predict([{}],None),[expected])
        nested=.7*.2+.3*FusionBlend(*parents).predict([{}],None)
        np.testing.assert_allclose(FusionBlend(*parents,weather_weight=.85).predict([{}],None),nested)
    def test_invalid_weight_and_legacy_default(self):
        parents=([Member(.2)]*3,[Member(.8)]*3)
        for weight in [-.1,1.1,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):FusionBlend(*parents,weather_weight=weight)
        model=FusionBlend(*parents);del model.weather_weight
        np.testing.assert_allclose(model.predict([{}],None),[.5])
    def test_named_seed_ensemble_preserves_all_seeds(self):
        model=SeedEnsemble([Member(.1),Member(.3),Member(.8)],'WeatherSignal')
        np.testing.assert_allclose(model.predict([{}],None),[.4])
        self.assertEqual(model.name,'WeatherSignal')
        with self.assertRaises(ValueError):SeedEnsemble([Member(.5)],'MarketGuard')
    def test_equal_parent_weight_without_labels(self):
        model=FusionBlend([Member(.2)]*3,[Member(.8)]*3)
        np.testing.assert_allclose(model.predict([{'y':0},{'y':1}],None),[.5,.5])
    def test_requires_all_seeds(self):
        with self.assertRaises(ValueError):FusionBlend([Member(.2)],[Member(.8)]*3)
    def test_rejects_invalid_member_probability(self):
        for p in (float('nan'),1.1,-.1):
            with self.assertRaises(ValueError):FusionBlend([Member(p)]*3,[Member(.8)]*3).predict([{}],None)
