"""Frozen out-of-fold BTC challengers; experimental historical paper replay only."""
import json
from functools import lru_cache
from pathlib import Path
import numpy as np
from .btc_research import digest, metrics
from .btc_flow_followup import read_rows, split, PROTOCOL
from .btc_anchor_study import FixedAnchor
from .btc_directional_batch import DirectionalAnchor
from .btc_microdata import MicroAnchor, attach
from .sizing import simulate

ROOT = Path(__file__).resolve().parents[1] / 'reports'
MODELS = {
    'fixed_anchor': ('FixedAnchor', 'Best log loss in this comparison. Small corrections to market odds using minute order flow and exchange basis.'),
    'directional_joint': ('DirectionalAnchor Joint', 'Best Brier score in this comparison. Symmetric corrections using price movement and signed flow.'),
    'delayed_micro': ('Delayed MicroFlow', 'Timing stress test with the highest simulated profit, but worse forecasts. Not a validated trading winner.'),
}

@lru_cache(maxsize=1)
def prepared():
    manifest = json.loads((ROOT / 'btc-experimental-manifest.json').read_text())
    for name, sha in manifest.items():
        if digest((ROOT / name).read_bytes()) != sha:
            raise ValueError('Experimental BTC artifact hash mismatch: ' + name)
    rows = read_rows(ROOT / 'btc-flow-followup-v1')
    features = json.loads((ROOT / 'btc-microdata-v1/features.json').read_text())
    allrows = []; predictions = {k: [] for k in MODELS}
    for start, end in PROTOCOL['folds']:
        _, test = split(rows, start, end)
        allrows.extend(test)
        def load(folder):
            return json.loads((ROOT / folder / f'models-{start}.json').read_text())
        predictions['fixed_anchor'].extend(FixedAnchor.load(load('btc-anchor-study-v1')['fixed_anchor']).predict(test))
        predictions['directional_joint'].extend(DirectionalAnchor.load(load('btc-directional-batch-v1')['joint']).predict(test))
        predictions['delayed_micro'].extend(MicroAnchor.load(load('btc-microdata-v1')['flow_plus_micro']).predict(attach(test, features, 60)))
    return allrows, {k: np.asarray(v) for k, v in predictions.items()}

def replay(model, bankroll, slippage):
    rows, predictions = prepared(); p = predictions[model]
    return dict(model=model, name=MODELS[model][0] + ' · EXPERIMENTAL', experimental=True,
        mode='historical_paper_only', live_enabled=False, automatic_promotion=False,
        previously_inspected=True, period='June 28–July 20, 2026',
        paper=simulate(rows, p, policy='bankroll_1_percent', slippage=slippage, bankroll=bankroll, min_edge=.04),
        scores=metrics(rows, p), market_scores=metrics(rows, np.asarray([r['p'] for r in rows])),
        protocol=dict(min_edge=.04, fee_rate=.07, max_contracts=100, bet_fraction=.01,
                      day_open_fraction=.03, portfolio_fraction=.20, micro_delay_seconds=60 if model=='delayed_micro' else 0),
        note='527 contracts / 23 previously inspected days. Frozen chronological fold models, not in-sample fitted predictions. Assumed fills without depth; no proven edge. Delayed MicroFlow is a timing stress test. All models share the same trading rules.')

def cards():
    result = []
    for key, (name, description) in MODELS.items():
        r = replay(key, 1000, .02)
        result.append(dict(id=key, name=name + ' · EXPERIMENTAL', description=description,
            experimental=True, period=r['period'], scores=r['scores'],
            summary=dict(net_pnl=r['paper']['pnl'], entries=r['paper']['entries'])))
    return result
