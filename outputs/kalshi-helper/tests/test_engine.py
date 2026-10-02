from datetime import datetime, timedelta, timezone
from decimal import Decimal
import json
import tempfile
import unittest

from helper.core import Engine, Settings, fee_and_cost, validate_record
from helper.kalshi import normalize_book, normalize_outcome
from helper.models import weather_probability
from helper.replay import demo_records, parse_jsonl, replay
from helper.storage import Store


BASE = datetime(2025, 1, 1, tzinfo=timezone.utc)


def ts(seconds=0):
    return (BASE+timedelta(seconds=seconds)).isoformat()


def market(ticker="A", fee="0.07"):
    return dict(type="market", ticker=ticker, event_ticker="EVENT", available_at=ts(),
                close_time=ts(3600), status="open", fee_rate=fee, fee_version="test")


def book(seconds=0, ticker="A", bid="0.39", no="0.60", size="1000"):
    return dict(type="book", ticker=ticker, available_at=ts(seconds), observed_at=ts(seconds),
                yes=[[bid, size]], no=[[no, size]])


def prediction(ticker="A", seconds=0, p="0.80"):
    return dict(type="prediction", ticker=ticker, available_at=ts(seconds), made_at=ts(seconds),
                features_available_at=ts(seconds), expires_at=ts(1800), probability=p, model_id="test")


def outcome(ticker="A", payout="1"):
    return dict(type="outcome", ticker=ticker, available_at=ts(4000), settled_at=ts(4000), yes_payout=payout)


class EngineTests(unittest.TestCase):
    def initialized(self, fee="0.07"):
        engine = Engine()
        engine.feed(market(fee=fee))
        engine.feed(book())
        engine.feed(prediction())
        return engine

    def test_delay_uses_future_executable_price(self):
        engine = self.initialized()
        self.assertEqual(len(engine.positions), 0)
        engine.feed(book(3))
        self.assertEqual(len(engine.positions), 0)
        engine.feed(book(6, no="0.10"))  # Ask moves to .90; old .40 cannot fill.
        self.assertEqual(len(engine.positions), 0)
        self.assertTrue(any(r.get("reason")=='edge_lost_after_costs' for r in engine.ledger))

    def test_bankroll_conservation_and_duplicate_settlement(self):
        engine = self.initialized()
        engine.feed(book(6))
        pos = engine.positions["A"]
        self.assertEqual(engine.cash+pos["cost"], engine.initial)
        quantity = pos["quantity"]
        cost = pos["cost"]
        engine.feed(outcome())
        self.assertEqual(engine.cash, engine.initial-cost+quantity)
        self.assertEqual(engine.realized, quantity-cost)
        balance = engine.cash
        engine.feed(outcome())
        self.assertEqual(engine.cash, balance)

    def test_conflicting_outcome_rejected(self):
        engine = self.initialized()
        engine.feed(outcome())
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            engine.feed(outcome(payout="0"))

    def test_no_side_payout(self):
        engine = Engine()
        engine.feed(market())
        engine.feed(book(bid="0.70", no="0.29"))
        engine.feed(prediction(p="0.05"))
        engine.feed(book(6, bid="0.70", no="0.29"))
        self.assertEqual(engine.positions["A"]["side"], "no")
        engine.feed(outcome(payout="0"))
        self.assertGreater(engine.realized, 0)

    def test_event_risk_shared_across_brackets(self):
        engine = Engine()
        for t in ("A", "B"):
            engine.feed(market(t))
            engine.feed(book(ticker=t))
            engine.feed(prediction(t))
        for t in ("A", "B"):
            engine.feed(book(6, ticker=t))
        exposure = sum(p["cost"] for p in engine.positions.values())
        self.assertLessEqual(exposure, Decimal("30"))

    def test_unknown_fee_no_trade(self):
        engine = self.initialized(fee=None)
        engine.feed(book(6))
        self.assertFalse(engine.positions)

    def test_stale_observation_never_fills(self):
        engine = self.initialized()
        r = book(1000)
        r["observed_at"] = ts(6)
        engine.feed(r)
        self.assertFalse(engine.positions)

    def test_repeated_book_does_not_replenish(self):
        engine = self.initialized()
        r = book(6)
        engine.feed(r)
        count = len([x for x in engine.ledger if x["action"] == "fill"])
        engine.feed(r)
        engine.feed(prediction(seconds=7))
        engine.feed(book(13))
        self.assertEqual(count, len([x for x in engine.ledger if x["action"] == "fill"]))

    def test_unpriced_position_is_not_realized_profit(self):
        engine = self.initialized()
        engine.feed(book(6))
        engine.finish(ts(1000))
        r = engine.report()
        self.assertEqual(r["realized_pnl"], "0")
        self.assertGreater(r["unpriced_contracts"], 0)
        self.assertEqual(r["equity_lower_bound"], r["cash"])

    def test_future_features_rejected(self):
        p = prediction()
        p["features_available_at"] = ts(10)
        with self.assertRaises(ValueError):
            validate_record(p)

    def test_naive_timestamp_rejected(self):
        p = prediction()
        p["made_at"] = "2025-01-01T00:00:00"
        with self.assertRaises(ValueError):
            validate_record(p)

    def test_closed_market_no_fill(self):
        engine = self.initialized()
        m = dict(market(), available_at=ts(5), status="closed")
        engine.feed(m)
        engine.feed(book(6))
        self.assertFalse(engine.positions)

    def test_depth_limit_and_book_walk(self):
        engine = Engine(Settings(bankroll="10000", event_fraction="1", total_fraction="1"))
        engine.feed(market())
        engine.feed(prediction())
        r = book(6, size="50")
        r["no"].append(["0.55", "30"])
        engine.feed(r)
        self.assertEqual(engine.positions["A"]["quantity"], 8)
        self.assertEqual(len([x for x in engine.ledger if x["action"] == "fill"][0]["levels"]), 2)

    def test_prediction_expiry_cancels(self):
        engine = self.initialized()
        engine.feed(book(1800))
        self.assertFalse(engine.positions)
        self.assertFalse(engine.pending)

    def test_outcome_at_prediction_time_is_not_scored(self):
        p = prediction(seconds=4000)
        p["expires_at"] = ts(5000)
        r = replay([market(), book(), p, outcome()])
        self.assertEqual(r["scores"]["paired_markets"], 0)

    def test_fee_rounding_and_subpenny(self):
        fee, cost = fee_and_cost(Decimal("0.50"), 100, Decimal("0.07"))
        self.assertEqual(fee, Decimal("1.75"))
        self.assertEqual(cost, Decimal("51.75"))
        fee, cost = fee_and_cost(Decimal("0.4055"), 1, Decimal("0.07"))
        self.assertEqual(cost, Decimal("0.43"))
        self.assertEqual(fee, Decimal("0.0245"))

    def test_brackets_sum_to_one(self):
        ps = [weather_probability(72, 4, a, b) for a, b in ((None,68.5),(68.5,72.5),(72.5,76.5),(76.5,None))]
        self.assertAlmostEqual(sum(ps), 1)

    def test_demo_deterministic(self):
        self.assertEqual(replay(demo_records()), replay(demo_records()))

    def test_replay_cutoff_does_not_use_later_settlement(self):
        rows = [market(), book(), prediction(), book(6), outcome()]
        r = replay(rows, end=ts(10))
        self.assertIn("A", r["positions"])
        self.assertEqual(r["realized_pnl"], "0")

    def test_small_bankroll_never_negative(self):
        r = replay(demo_records(), Settings(bankroll="1"))
        self.assertGreaterEqual(Decimal(r["cash"]), 0)

    def test_book_parser_and_crossed_validation(self):
        r = normalize_book("A", {"orderbook_fp":{"yes_dollars":[["0.4","2.5"]], "no_dollars":[["0.5","1"]]}}, ts())
        self.assertEqual(r["yes"][0][1], "2.5")
        with self.assertRaises(ValueError):
            validate_record(book(bid="0.6", no="0.6"))

    def test_settlement_waits_until_final(self):
        self.assertIsNone(normalize_outcome({"status":"determined", "result":"yes"}, ts()))

    def test_import_is_atomic_and_deduplicates(self):
        with tempfile.TemporaryDirectory() as path:
            store = Store(path)
            rows = [market(), book()]
            self.assertEqual(store.append(rows, "test"), 2)
            self.assertEqual(store.append(rows, "test"), 0)
            with self.assertRaises(ValueError):
                store.append([market("B"), dict(prediction("B"), probability="NaN")], "test")
            self.assertEqual(len(store.records("test")), 2)

    def test_jsonl_reports_bad_line(self):
        with self.assertRaisesRegex(ValueError, "Line 2"):
            parse_jsonl(json.dumps(market())+"\nnot json")


if __name__ == "__main__":
    unittest.main()
