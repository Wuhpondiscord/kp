import unittest,tempfile,math
import numpy as np
from helper.core import Engine,validate_record
from helper.weather_research import RemainingMaximum,train
from test_engine import market,book,prediction,ts

class PriorityTests(unittest.TestCase):
    def engine(self):
        e=Engine();e.feed(market());e.feed(book());p=prediction();p.update(require_execution_check=True,signal_side='yes');e.feed(p);return e
    def check(self,p,signal=True):
        b=book(6);b['execution_check']=dict(model_id='test',made_at=ts(6),signal=signal,probability=p);return b
    def test_missing_execution_check_blocks_fill(self):
        e=self.engine();e.feed(book(6));self.assertFalse(e.positions)
    def test_refresh_removes_old_edge_without_flipping_side(self):
        e=self.engine();e.feed(self.check(.1));self.assertFalse(e.positions)
    def test_refreshed_probability_is_scored(self):
        e=self.engine();e.feed(self.check(.7));self.assertIn('A',e.positions)
        self.assertEqual(e.filled_predictions['A']['probability'],.7)
        self.assertEqual(e.filled_predictions['A']['decision_probability'],'0.80')
    def test_future_recheck_rejected(self):
        b=self.check(.7);b['execution_check']['made_at']=ts(7)
        with self.assertRaises(ValueError):validate_record(b)
    def test_remaining_model_fits_interval_targets_and_conserves_mass(self):
        base=dict(remaining_forecast_max=75,remaining_hours=18,forecast_complete=True,series='KXHIGHNY',observed=False,group='2026-01-01')
        rows=[dict(base,event=str(i),lower=72,upper=78,y=1) for i in range(80)]
        model=RemainingMaximum().fit(rows)
        pred=model.predict([dict(base,lower=a,upper=b) for a,b in [(-math.inf,72),(72,78),(78,math.inf)]])
        self.assertAlmostEqual(sum(pred),1);self.assertTrue(np.all(np.isfinite(pred)))
    def test_insufficient_data_is_reported_not_trained(self):
        with tempfile.TemporaryDirectory() as d:
            r=train(d,d+'/experiment');self.assertEqual(r['status'],'insufficient_data')
