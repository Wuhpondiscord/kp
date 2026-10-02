from datetime import datetime, timedelta, timezone
import json
import random
import uuid

from .core import Engine, Settings, stamp, validate_record
from .models import weather_prediction, weather_probability


PRIORITY = {"market": 0, "outcome": 1, "book": 2, "prediction": 3}
DEMO_DATASET = "demo-weather-v2"


def parse_jsonl(text):
    records = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
            if r.get("type") == "weather_forecast":
                r = weather_prediction(r)
            records.append(validate_record(r))
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            raise ValueError(f"Line {line_number}: {exc}") from exc
    if not records:
        raise ValueError("The dataset is empty")
    return records


def replay(records, settings=None, start=None, end=None):
    records = sorted(records, key=lambda r: (stamp(r["available_at"]), PRIORITY[r["type"]]))
    if not records:
        raise ValueError("No records to replay")
    start = stamp(start) if start else stamp(records[0]["available_at"])
    end = stamp(end) if end else stamp(records[-1]["available_at"])
    if start > end:
        raise ValueError("Replay start must precede end")
    engine = Engine(settings or Settings())
    for record in records:
        at = stamp(record["available_at"])
        if at > end:
            break
        if at < start and record["type"] == "prediction":
            continue
        engine.feed(record, allow_orders=(start <= at < end))
    engine.finish(end.isoformat())
    report = engine.report()
    report.update(start=start.isoformat(), end=end.isoformat(),
                  evaluation="Fixed supplied predictions; no training or walk-forward tuning is performed")
    return report


def save_replay(store, dataset, settings=None, start=None, end=None):
    records = store.records(dataset)
    report = replay(records, settings, start, end)
    report["dataset"] = dataset
    report["synthetic"] = dataset.startswith("demo-") or any(
        r.get("synthetic") or r.get("model_id", "").startswith("synthetic-") for r in records)
    run_id = uuid.uuid4().hex[:12]
    store.save_run(run_id, "synthetic demo" if report["synthetic"] else "historical replay", "complete", report)
    return run_id, report


def demo_records(days=20, seed=17):
    """Invented weather, books and outcomes. Never evidence of profitability."""
    rng = random.Random(seed)
    base = datetime(2025, 1, 1, 12, tzinfo=timezone.utc)
    records = []
    brackets = [(None, 68.5), (68.5, 72.5), (72.5, 76.5), (76.5, None)]
    for day in range(days):
        t = base + timedelta(days=day)
        actual = rng.gauss(72, 4)
        model_mean = 72 + rng.gauss(0, 2.5)
        market_mean = 72 + rng.gauss(0, 2.5)
        event = f"SYNTH-WEATHER-{day:03d}"
        for i, (lower, upper) in enumerate(brackets):
            ticker = f"{event}-B{i}"
            records.append(dict(type="market", available_at=t.isoformat(), ticker=ticker,
                                event_ticker=event, close_time=(t+timedelta(hours=12)).isoformat(),
                                status="open", fee_rate="0.07", fee_version="synthetic-example-0.07",
                                title=f"Invented station / day {day+1} / bracket {i+1}",
                                rules="SYNTHETIC. Nearest-integer temperature; not a real contract."))
            mid = weather_probability(market_mean, 4, lower, upper)
            for seconds in (0, 30, 7200):
                observed = t + timedelta(seconds=seconds)
                shift = rng.uniform(-0.015, 0.015) if seconds else 0
                bid = round(max(0.01, min(0.94, mid+shift-0.03)), 4)
                ask = round(min(0.99, bid+0.06), 4)
                records.append(dict(type="book", ticker=ticker, available_at=observed.isoformat(),
                                    observed_at=observed.isoformat(), yes=[[str(bid), "250"]],
                                    no=[[str(round(1-ask, 4)), "250"]]))
            prediction = weather_prediction(dict(ticker=ticker, available_at=t.isoformat(),
                made_at=t.isoformat(), features_available_at=t.isoformat(),
                expires_at=(t+timedelta(hours=4)).isoformat(), model_id="synthetic-normal-v1",
                station="INVENTED", rule_reference="synthetic nearest-integer bracket",
                mean=model_mean, std=4, lower=lower, upper=upper))
            records.append(prediction)
            won = (lower is None or actual >= lower) and (upper is None or actual < upper)
            settled = t+timedelta(hours=20)
            records.append(dict(type="outcome", ticker=ticker, available_at=settled.isoformat(),
                                settled_at=settled.isoformat(), yes_payout="1" if won else "0"))
    for record in records:
        record["synthetic"] = True
    return sorted(records, key=lambda r: (stamp(r["available_at"]), PRIORITY[r["type"]]))
