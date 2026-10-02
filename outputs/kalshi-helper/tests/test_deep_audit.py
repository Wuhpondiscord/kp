import tempfile
import unittest
import pickle
from unittest.mock import patch
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from helper.core import Engine,book_midpoint,validate_record
from helper.evaluation import ModelService
from helper.storage import Store
from helper.station_model import NewtonTree
from test_engine import market,book,prediction,outcome


class DeepAuditTests(unittest.TestCase):
    def test_zero_size_prices_do_not_create_midpoint_or_cross(self):
        b=book();b['yes'].append(['.99','0'])
        validate_record(b)
        self.assertEqual(float(book_midpoint(b)),.395)
        b['no']=[['.6','0']]
        self.assertIsNone(book_midpoint(b))

    def test_filled_scores_use_actual_signal_not_first_prediction(self):
        engine=Engine();engine.feed(market());engine.feed(book())
        engine.feed(prediction(p='.5'))
        engine.feed(prediction(seconds=1,p='.8'))
        engine.feed(book(7,bid='.49',no='.50'))
        engine.feed(outcome())
        r=engine.report();cohorts=r['score_cohorts']
        self.assertAlmostEqual(r['scores']['model_brier'],.25)
        self.assertAlmostEqual(cohorts['filled_signal_vs_decision_market']['model_brier'],.04)
        self.assertAlmostEqual(cohorts['filled_signal_vs_execution_market']['market_brier'],.505**2)
        fill=next(row for row in r['ledger'] if row['action']=='fill')
        self.assertEqual(fill['probability'],'.8');self.assertEqual(fill['signal_age_seconds'],6)

    def test_inference_rejects_invalid_quotes_even_without_model(self):
        with tempfile.TemporaryDirectory() as folder:
            model=ModelService(Store(folder))
            for bid,ask in [(float('nan'),.5),(.8,.2),(-.1,.4),(.1,float('inf'))]:
                with self.subTest(bid=bid,ask=ask),self.assertRaises(ValueError):
                    model.predict_signal('KXHIGHNY',bid,ask,'2026-01-01T00:00Z','2026-01-02T00:00Z')

    def test_newton_leaf_values_are_external_and_serializable(self):
        X=np.array([[0],[1],[2],[3]])
        tree=DecisionTreeRegressor(max_leaf_nodes=2,random_state=1).fit(X,[0,0,1,1])
        original=tree.predict(X).copy();nodes=tree.apply(X)
        correction=NewtonTree(tree,{int(nodes[0]):-2.,int(nodes[-1]):1.5})
        np.testing.assert_array_equal(correction.predict(X),[-2,-2,1.5,1.5])
        np.testing.assert_array_equal(tree.predict(X),original)
        np.testing.assert_array_equal(pickle.loads(pickle.dumps(correction)).predict(X),correction.predict(X))
