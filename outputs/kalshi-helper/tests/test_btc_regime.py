import unittest
from unittest.mock import patch
import numpy as np
from helper.btc_regime import BTCRegime, TradeGate, gate_training, side_features
from helper.btc_research import iso, unix
from helper.core import stamp


class BTCRegimeTests(unittest.TestCase):
    def rows(self):
        rng=np.random.default_rng(17);rows=[]
        for i in range(700):
            at=unix('2026-09-01')+i*900;x=rng.normal(size=10).tolist()
            rows.append(dict(ticker=str(i),at=iso(at),settled_at=iso(at+310),close_time=iso(at+300),
                group=iso(at)[:10],x=x,y=int(x[0]>0),p=.5,bid=.49,ask=.51,spread=.02,
                execution_at=iso(at+65),execution_bid=.49,execution_ask=.51,
                execution_spot={'x':x},quote_at=iso(at-60),quote_available_at=iso(at-55)))
        return rows

    def test_forecaster_learns_and_does_not_read_labels_or_market_at_inference(self):
        rows=self.rows();m=BTCRegime().fit(rows[:500]);p=m.predict(rows[500:])
        self.assertGreater(np.mean((p>.5)==[r['y'] for r in rows[500:]]),.9)
        np.testing.assert_array_equal(p,m.predict([dict(r,y=1-r['y'],p=.99) for r in rows[500:]]))

    def test_side_costs_and_gate_inference_ignore_labels(self):
        rows=self.rows();p=np.array([.8 if r['y'] else .2 for r in rows])
        gate=TradeGate().fit(rows[:500],p[:500]);a=gate.filter(rows[500:],p[500:])
        b=gate.filter([dict(r,y=1-r['y']) for r in rows[500:]],p[500:])
        np.testing.assert_array_equal(a[0],b[0]);self.assertEqual(a[1],b[1])
        features,cost=side_features(rows[0],.8,'yes')
        self.assertGreater(cost,.53);self.assertAlmostEqual(features[1],.8-cost)

    def test_gate_refuses_mismatched_or_invalid_probabilities(self):
        rows=self.rows()[:100]
        for probabilities in [[.5], [float('nan')]*100, [1.5]*100]:
            with self.assertRaises(ValueError):TradeGate().fit(rows,probabilities)

    def test_gate_abstention_does_not_mutate_forecasts(self):
        gate=TradeGate()
        class Reject:
            def predict(self,x): return [-1]*len(x)
        gate.model=Reject();rows=self.rows()[:10];p=np.full(10,.9)
        filtered,count=gate.filter(rows,p)
        self.assertEqual(count,0);np.testing.assert_array_equal(filtered,np.full(10,.5))
        np.testing.assert_array_equal(p,np.full(10,.9))

    def test_gate_training_predictions_are_strictly_forward(self):
        class AuditModel:
            def fit(self,rows):
                self.ids={r['ticker'] for r in rows};self.latest=max(stamp(r['settled_at']) for r in rows)
                return self
            def predict(self,rows):
                assert not self.ids.intersection(r['ticker'] for r in rows)
                assert min(stamp(r['at']) for r in rows)>self.latest
                return np.full(len(rows),.5)
        with patch('helper.btc_regime.BTCRegime',AuditModel):
            rows,p,audit=gate_training(self.rows())
        self.assertEqual(len(rows),len(p));self.assertEqual(len(audit),3)
        self.assertEqual(len({r['ticker'] for r in rows}),len(rows))
        for f in audit:
            self.assertGreater((stamp(f['score_earliest_decision'])-stamp(f['train_latest_settlement'])).total_seconds(),7200)
