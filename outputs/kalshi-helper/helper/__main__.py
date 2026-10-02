import argparse
import json
from pathlib import Path
import time

from .core import Settings
from .kalshi import collect_once
from .live import LiveManager
from .models import weather_prediction
from .replay import DEMO_DATASET, demo_records, parse_jsonl, save_replay
from .storage import Store


def main():
    parser = argparse.ArgumentParser(description="Kalshi Helper — local research and paper simulation")
    parser.add_argument("--data", default="data", help="Local database and raw-response directory")
    commands = parser.add_subparsers(dest="command", required=True)
    web = commands.add_parser("serve", help="Open the local dashboard")
    web.add_argument("--port", type=int, default=8765)
    web.add_argument('--attach',action='store_true',help='Attach a UI while another process finishes its data job; preserve running state')
    demo = commands.add_parser("demo", help="Run a clearly labeled synthetic weather demonstration")
    demo.add_argument("--bankroll", default="1000")
    demo.add_argument("--export", default="reports/demo-report.json")
    imp = commands.add_parser("import", help="Import normalized JSONL, preserving supplied timestamps")
    imp.add_argument("file")
    imp.add_argument("--dataset", required=True)
    backtest = commands.add_parser("replay", help="Replay a named dataset with a simulated balance")
    backtest.add_argument("dataset")
    backtest.add_argument("--bankroll", default="1000")
    backtest.add_argument("--start")
    backtest.add_argument("--end")
    backtest.add_argument("--export", default="reports/replay-report.json")
    collector = commands.add_parser("collect", help="Collect one batch of live market metadata and books")
    collector.add_argument("--series", required=True)
    collector.add_argument("--limit", type=int, default=12)
    live = commands.add_parser("paper", help="Run a bounded live paper session (foreground)")
    live.add_argument("--tickers", default='', help="Comma-separated tickers; omit to discover weather markets")
    live.add_argument('--prospective',action='store_true',help='Freeze model and record weather/books in a hashed evidence journal')
    live.add_argument('--experimental',action='store_true',help='Allow unvalidated models with simulated money')
    live.add_argument("--bankroll", default="1000")
    live.add_argument("--hours", type=float, default=24)
    live.add_argument("--poll-seconds", type=int, default=60)
    live.add_argument("--fee-rate", help="Explicit assumed taker coefficient; unknown fees disable fills")
    live.add_argument("--predictions", help="JSONL file; reloaded every polling cycle")
    forecast = commands.add_parser("forecast", help="Convert reviewed normal-distribution inputs to predictions")
    forecast.add_argument("file")
    forecast.add_argument("--export", default="reports/predictions.jsonl")
    remaining=commands.add_parser('train-remaining-weather',help='Fit remaining-day model from settled prospective journals')
    poly=commands.add_parser('compare-polymarket',help='Paired with/without Polymarket weather ablation')
    poly.add_argument('--output',required=True)
    spread=commands.add_parser('compare-weather-spread',help='Validation-only constant versus disagreement spread comparison')
    spread.add_argument('--output',required=True)
    archive=commands.add_parser('prepare-weather-replay',help='Download explicit forecast runs and evaluate historical validation data')
    archive.add_argument('--features',required=True,help='Frozen original price features.jsonl used to retain split membership')
    archive.add_argument('--output',required=True)
    archive.add_argument('--offline',action='store_true',help='Use saved run responses only; do not contact the archive')
    archive.add_argument('--observation-audit',required=True,help='Exact-settlement observation audit; calibration is restricted to prior dates')
    for command in (remaining,poly,spread):
        choice=command.add_mutually_exclusive_group()
        choice.add_argument('--observation-calibration',help='External development-only calibration JSON; defaults to frozen pre-training calibration')
        choice.add_argument('--legacy-observation-assumption',action='store_true',help='Explicitly use the uncalibrated zero-bias, 1F assumption')
    commands.add_parser('collect-weather-outcomes',help='Fetch settled targets for recorded weather features')
    remaining.add_argument('--output',required=True,help='New experiment folder; existing results never overwritten')
    strategy=commands.add_parser('validate-strategy',help='Validation-only strategy search for a saved price model')
    strategy.add_argument('--training-folder',required=True)
    strategy.add_argument('--output',required=True)
    args = parser.parse_args()
    if args.command in ('compare-weather-spread','compare-polymarket','train-remaining-weather'):
        from .observation_calibration import load_training_calibration
        calibration=load_training_calibration(args.observation_calibration,args.legacy_observation_assumption)
    store = Store(args.data)
    if args.command == 'prepare-weather-replay':
        from .archive_replay import run
        run(store.root,args.features,args.output,args.observation_audit,offline=args.offline)
    elif args.command == 'compare-weather-spread':
        from .spread_research import compare
        print(json.dumps(compare(store.root,args.output,calibration),indent=2))
    elif args.command == 'compare-polymarket':
        from .poly_ablation import compare
        print(json.dumps(compare(store.root,args.output,calibration),indent=2))
    elif args.command == 'collect-weather-outcomes':
        from .weather_research import collect_outcomes
        print(json.dumps(collect_outcomes(store),indent=2))
    elif args.command == 'train-remaining-weather':
        from .weather_research import train
        print(json.dumps(train(store.root,args.output,calibration),indent=2))
    elif args.command == 'validate-strategy':
        from .strategy_research import run
        result=run(args.training_folder,args.output)
        print(json.dumps(dict(action=result['selection']['action'],reason=result['reason'],test_rows=result['test_rows']),indent=2))
    elif args.command == "serve":
        from .server import serve
        serve(store, args.port,attach=args.attach)
    elif args.command in ("demo", "replay"):
        dataset = DEMO_DATASET if args.command == "demo" else args.dataset
        if args.command == "demo":
            store.append(demo_records(), dataset)
        run_id, report = save_replay(store, dataset, Settings(bankroll=args.bankroll),
                                    getattr(args, "start", None), getattr(args, "end", None))
        path = Path(args.export)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({"run_id": run_id, "report": str(path.resolve()),
                          "synthetic": report["synthetic"], "fills": report["fills"],
                          "realized_pnl": report["realized_pnl"], "scores": report["scores"]}, indent=2))
    elif args.command == "import":
        if args.dataset.startswith(("demo-", "live-")):
            raise ValueError("demo- and live- dataset prefixes are reserved")
        count = store.append(parse_jsonl(Path(args.file).read_text(encoding="utf-8-sig")), args.dataset)
        print(f"Imported {count} new records into {args.dataset}")
    elif args.command == "collect":
        print(json.dumps(collect_once(store, args.series, args.limit), indent=2))
    elif args.command == "paper":
        manager = LiveManager(store)
        run_id = manager.start(args.tickers.split(","), Settings(bankroll=args.bankroll), args.hours,
                               args.poll_seconds, args.fee_rate, args.predictions,auto_model=not bool(args.predictions),
                               prospective=args.prospective,experimental=args.experimental)
        print(f"Paper run {run_id}. Keep this process running; Ctrl+C stops it.", flush=True)
        try:
            while manager.thread.is_alive():
                manager.thread.join(timeout=1)
        except KeyboardInterrupt:
            manager.stop()
            manager.thread.join(timeout=60)
        print(json.dumps(store.run(run_id), indent=2))
    elif args.command == "forecast":
        rows = json.loads(Path(args.file).read_text(encoding="utf-8-sig"))
        text = "\n".join(json.dumps(weather_prediction(row)) for row in rows) + "\n"
        parse_jsonl(text)
        path = Path(args.export)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print(f"Saved {len(rows)} predictions to {path.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, OSError) as exc:
        raise SystemExit(str(exc))
