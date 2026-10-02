import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from helper.core import Engine, Settings
from helper.live import LiveManager
from helper.replay import demo_records, save_replay
from helper.storage import Store
from helper.evaluation import PredictionResult
from test_engine import ts, prediction, book


class LiveTests(unittest.TestCase):
    def test_auto_baseline_cannot_trade_on_price_movement(self):
        clock=[0]
        class Stop:
            def is_set(self):return False
            def wait(self,seconds):clock[0]+=seconds
        class Client:
            def __init__(self,store):pass
            def market(self,ticker):return dict(ticker=ticker,event_ticker='KXHIGHNY-EVENT',close_time=ts(3600),status='active')
            def book(self,ticker):return book(clock[0],ticker,bid='0.10' if clock[0] else '0.70',no='0.80' if clock[0] else '0.20')
        class Model:
            loaded_id=None
            def __init__(self,store,active_model=None):pass
            def predict_signal(self,*args):return PredictionResult(.75 if clock[0]==0 else .15,'Experimental · misleading label',False)
        with tempfile.TemporaryDirectory() as folder:
            store=Store(folder);manager=LiveManager(store);manager.stop_event=Stop()
            details=dict(dataset='live-gate',start=ts(),end=ts(120),tickers=['A'],poll_seconds=60,predictions_file=None,auto_model=True)
            with patch('helper.live.Kalshi',Client),patch('helper.live.utcnow',lambda:ts(clock[0])),patch('helper.evaluation.ModelService',Model):
                manager._run('run',Engine(),details,'0.07')
            r=store.run('run')
            self.assertEqual(r['fills'],0)
            self.assertFalse(any(x['type']=='prediction' for x in store.records('live-gate')))
            self.assertEqual(r['market_health']['A']['status'],'watching')

    def test_bad_reference_rejected_before_worker_starts(self):
        with tempfile.TemporaryDirectory() as folder:
            manager=LiveManager(Store(folder))
            with patch('helper.resolve.resolve_markets',side_effect=ValueError('No open markets')):
                with self.assertRaises(ValueError):manager.start(reference='KXINVALID')
            self.assertIsNone(manager.thread)
            self.assertEqual(manager.store.runs(),[])

    def test_model_initialization_failure_persists_failed_status(self):
        with tempfile.TemporaryDirectory() as folder:
            manager=LiveManager(Store(folder))
            details=dict(dataset='live-failure',start=ts(),end=ts(120),tickers=['A'],poll_seconds=60,predictions_file=None,auto_model=True)
            with patch('helper.evaluation.ModelService',side_effect=RuntimeError('Model unavailable')),patch('helper.live.utcnow',return_value=ts()):
                manager._run('run',Engine(),details,'0.07')
            self.assertEqual(manager.store.runs()[0]['status'],'failed')
            self.assertIn('Model unavailable',manager.store.run('run')['errors'][0]['message'])

    def test_two_poll_session_fills_then_stops_at_deadline(self):
        clock = [0]

        class Stop:
            def is_set(self):
                return False

            def wait(self, seconds):
                clock[0] += seconds

        class Client:
            def __init__(self, store):
                pass

            def market(self, ticker):
                return dict(ticker=ticker, event_ticker="EVENT", close_time=ts(3600),
                            status="active", notional_value_dollars="1.0000")

            def book(self, ticker):
                return book(clock[0], ticker)

        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder)
            path = Path(folder)/"predictions.jsonl"
            path.write_text(json.dumps(prediction()), encoding="utf-8")
            manager = LiveManager(store)
            manager.stop_event = Stop()
            details = dict(dataset="live-test", start=ts(), end=ts(120),
                           tickers=["A"], poll_seconds=60, predictions_file=str(path))
            with patch("helper.live.Kalshi", Client), patch("helper.live.utcnow", lambda: ts(clock[0])):
                manager._run("run", Engine(Settings()), details, "0.07")
            run = store.runs()[0]
            self.assertEqual(run["status"], "complete")
            self.assertEqual(run["report"]["fills"], 1)
            self.assertEqual(run["report"]["realized_pnl"], "0")
            self.assertEqual(clock[0], 120)
            self.assertIn("A", run["report"]["positions"])
            self.assertEqual(len([r for r in store.records("live-test") if r["type"] == "prediction"]), 1)

    def test_synthetic_label_survives_import_name(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder)
            store.append(demo_records(1), "imported")
            _, report = save_replay(store, "imported")
            self.assertTrue(report["synthetic"])

    def test_restart_marks_running_report_interrupted(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder)
            store.save_run("run", "live paper", "running", Engine().report())
            store.interrupt_previous_runs()
            self.assertEqual(store.runs()[0]["status"], "interrupted")


if __name__ == "__main__":
    unittest.main()
