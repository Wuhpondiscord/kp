import unittest
from copy import deepcopy
from helper.evaluation import validate_samples
from test_research import rows


class FeatureIntegrity(unittest.TestCase):
    def test_inconsistent_quote_features_rejected(self):
        for changes in ({'p':1.1},{'p':.91},{'spread':-.1},{'spread':.7},{'hours_left':-2}):
            source=deepcopy(rows());source[0].update(changes)
            with self.subTest(changes=changes),self.assertRaises(ValueError):validate_samples(source)
