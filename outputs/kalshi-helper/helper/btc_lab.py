"""Opt-in historical BTC paper lab. No live orders or automatic promotion."""
from functools import lru_cache
import json
import math
from pathlib import Path
from threading import Lock

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .btc_research import digest, execution_rows, metrics
from .btc_regime import BTCRegime
from .sizing import simulate

ROOT = Path(__file__).resolve().parents[1] / 'reports'
MODELS = {
    'volatility': ('BTC Volatility', 'A simple probability baseline using recent price volatility.'),
    'logistic': ('BTC Signal', 'A logistic model trained on January–April 2025 Binance candles.'),
    'regime': ('BTC Regime', 'Shallow boosted trees trained on the same Binance examples.'),
}
LOCK = Lock()


def read_verified(folder, name, sha):
    raw = (folder / name).read_bytes()
    if digest(raw) != sha:
        raise ValueError('BTC research artifact failed its integrity check: ' + name)
    return raw


def catalog():
    audit_dir = ROOT / 'btc-loss-audit'
    manifest = json.loads((audit_dir / 'manifest.json').read_text())
    report = json.loads(read_verified(audit_dir, 'report.json', manifest['report_sha256']))
    cards = []
    for key, (name, description) in MODELS.items():
        result = report['models'][key]
        cards.append(dict(id=key, name=name, description=description,
                          summary=result['summary'], scores=result['execution_forecast_scores']))
    return dict(mode='historical_paper_only', live_enabled=False, automatic_promotion=False,
                models=cards, market_scores=report['models']['volatility']['execution_market_scores'],
                warning='All three models lost money after assumed costs. These dates were previously inspected; no proven edge.',
                period='September 2026', contracts=2830, days=30,
                sources=['Binance BTCUSDT training candles (January–April 2025)',
                         'Coinbase BTC-USD inputs and official Kalshi outcomes (September 2026)'])


@lru_cache(maxsize=1)
def prepared():
    experiment = ROOT / 'btc-binance-history-v1'
    manifest = json.loads((experiment / 'manifest.json').read_text())
    # Only local, hash-checked numerical training arrays; never deserialize user pickle files.
    import io
    raw = read_verified(experiment, 'proxy_dataset.npz', manifest['outputs']['proxy_dataset.npz'])
    with np.load(io.BytesIO(raw), allow_pickle=False) as data:
        train = data['close_time'] < '2025-05-01T00:00:00+00:00'
        x, y = data['x'][train], data['y'][train]
    rows = []
    for name in ('train.jsonl', 'validation.jsonl', 'test.jsonl'):
        raw = read_verified(ROOT / 'btc15m-pilot-v1', name, manifest['kalshi_inputs'][name])
        rows.extend(json.loads(line) for line in raw.decode().splitlines())
    if len({r['ticker'] for r in rows}) != len(rows):
        raise ValueError('Duplicate BTC contract')
    rows = execution_rows(sorted(rows, key=lambda r: r['at']))
    logistic = make_pipeline(StandardScaler(), LogisticRegression(C=.1, max_iter=1000, random_state=1729)).fit(x, y)
    regime = BTCRegime().fit([dict(x=a.tolist(), y=int(b)) for a, b in zip(x, y)])
    probabilities = dict(volatility=np.array([r['diffusion'] for r in rows]),
                         logistic=logistic.predict_proba([r['x'] for r in rows])[:, 1],
                         regime=regime.predict(rows))
    return rows, probabilities


def replay(body):
    model = body.get('model', 'volatility')
    if model not in MODELS:
        raise ValueError('Choose BTC Volatility, BTC Signal or BTC Regime')
    bankroll = float(body.get('bankroll', 1000))
    slippage = float(body.get('slippage', .02))
    if not math.isfinite(bankroll) or not 10 <= bankroll <= 1000:
        raise ValueError('Use a paper balance between $10 and $1,000')
    if slippage not in (0, .02, .05):
        raise ValueError('Choose the fixed 0, 2 or 5 cent cost scenario')
    # Bound shared hosted CPU use, including the first deterministic training pass.
    if not LOCK.acquire(blocking=False):
        raise RuntimeError('Another BTC replay is running. Try again shortly.')
    try:
        rows, predictions = prepared()
        p = predictions[model]
        paper = simulate(rows, p, policy='bankroll_1_percent', slippage=slippage,
                         bankroll=bankroll, min_edge=.04)
        paper['execution_recomputed'] = True
        return dict(model=model, name=MODELS[model][0], mode='historical_paper_only',
                    previously_inspected=True, paper=paper, scores=metrics(rows, p),
                    market_scores=metrics(rows, np.array([r['p'] for r in rows])),
                    protocol=dict(min_edge=.04, fee_rate=.07, max_contracts=100,
                                  bet_fraction=.01, day_open_fraction=.03, portfolio_fraction=.20,
                                  decision_to_execution_seconds=5, execution_feature_age_seconds=5),
                    note='Reused September dates. Assumed quote fills without historical depth. Coinbase/Binance proxies differ from the settlement index. Cost scenarios are sensitivity checks, not measured execution.')
    finally:
        LOCK.release()
