"""Development-only BTC settlement-average experiment. Never opens test.jsonl."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import log_ndtr, ndtr

from .btc_research import BTCModels, digest, metrics, save, unix
from .core import stamp

PROTOCOL = {
    'version': 'btc-average-calibration-v1',
    'purpose': 'Development comparison, not a new holdout or promotion test',
    'inputs': ['train.jsonl', 'validation.jsonl'],
    'folds': [['2026-09-07', '2026-09-11'], ['2026-09-11', '2026-09-15'],
              ['2026-09-15', '2026-09-19']],
    'embargo_hours': 2,
    'candidate': 'BTCAverage: final-minute Gaussian-average approximation with two fitted probit parameters',
    'ablation': 'BTCAverageRaw: identical averaging correction without calibration',
    'regularization': 0.01,
    'selection': 'No hyperparameter grid; no strategy tuning; original test not loaded',
    'limitations': ['Coinbase is not BRTI', 'Log-price average approximates arithmetic price average',
                    'Continuous average approximates 60 discrete index samples',
                    'Constant local diffusion ignores jumps and changing volatility',
                    'Development dates already inspected; no independent edge claim'],
}


def average_z(rows):
    """For Brownian increments, Var(mean over [T-1,T]) = sigma²(T-2/3).

    T is minutes since the observed price. The averaging interval must be fully
    in the future. This is an approximation for a small log return, not exact BRTI.
    """
    horizon = np.array([(stamp(r['close_time'])-stamp(r['feature_time'])).total_seconds()/60
                        for r in rows])
    z = np.array([r['x'][0] for r in rows], dtype=float)
    if not np.isfinite(horizon).all() or not np.isfinite(z).all() or np.any(horizon < 1):
        raise ValueError('Require finite features and an entirely future averaging window')
    return z*np.sqrt(horizon/(horizon-2/3))


class BTCAverage:
    def __init__(self, bias=0., log_slope=0.):
        self.bias = float(bias)
        self.log_slope = float(log_slope)

    def fit(self, rows):
        z = average_z(rows)
        y = np.array([r['y'] for r in rows])
        if set(y) != {0, 1}: raise ValueError('Require both binary outcomes')
        def objective(w):
            score = w[0]+np.exp(w[1])*z
            return float(-np.mean(np.where(y == 1, log_ndtr(score), log_ndtr(-score)))
                         + PROTOCOL['regularization']*np.sum(w*w))
        fit = minimize(objective, [0., 0.], method='L-BFGS-B', bounds=[(-2, 2), (-2, 2)])
        if not fit.success: raise ValueError('Calibration fit failed: '+fit.message)
        self.bias, self.log_slope = map(float, fit.x)
        return self

    def predict(self, rows):
        return np.clip(ndtr(self.bias+np.exp(self.log_slope)*average_z(rows)), 1e-6, 1-1e-6)

    def artifact(self):
        return dict(bias=self.bias, log_slope=self.log_slope)


def read_development(source):
    """Enforce the old test boundary even if the source files are mislabelled."""
    result = []
    ids = set()
    for name, start, end in [('train', '2026-09-01', '2026-09-19'),
                             ('validation', '2026-09-19', '2026-09-25')]:
        path = Path(source)/(name+'.jsonl')
        manifest = Path(source)/'manifest.json'
        if manifest.exists():
            expected = json.loads(manifest.read_text())['files'][path.name]
            if digest(path.read_bytes()) != expected: raise ValueError('Development input hash mismatch')
        rows = [json.loads(line) for line in path.read_text(encoding='utf8').splitlines()]
        for r in rows:
            at, close, settled = [stamp(r[k]).timestamp() for k in ('at', 'close_time', 'settled_at')]
            # Pilot cohorts use close dates; the first decision can precede
            # midnight by five minutes for the contract closing at midnight.
            if not (unix(start)-300 <= at < close <= settled < unix(end)
                    and close >= unix(start)):
                raise ValueError('Development row crosses its declared date boundary')
            if stamp(r['feature_available_at']).timestamp() > at or stamp(r['quote_available_at']).timestamp() > at:
                raise ValueError('Future input in development row')
            if 'feature_time' in r and stamp(r['feature_time']) > stamp(r['feature_available_at']):
                raise ValueError('Feature availability precedes observation')
            if r['ticker'] in ids: raise ValueError('Duplicate contract across development splits')
            ids.add(r['ticker'])
        result.append(rows)
    return result


def diagnostics(rows, probabilities):
    """Buckets use the observed market midpoint, never the eventual label."""
    result = {}
    market = np.array([r['p'] for r in rows])
    for low, high in [(0, .1), (.1, .3), (.3, .7), (.7, .9), (.9, 1.01)]:
        mask = (market >= low) & (market < high)
        if not mask.any(): continue
        subset = [r for r, keep in zip(rows, mask) if keep]
        result[f'{low:g}-{min(high,1):g}'] = {n: metrics(subset, p[mask]) for n, p in probabilities.items()}
    return result


def evaluate(train, validation):
    old = BTCModels().fit(train)
    candidate = BTCAverage().fit(train)
    probabilities = {n: old.predict(validation, n) for n in
                     ['market', 'BTCVolatility', 'BTCSignal', 'BTCMarketGuard', 'price_only_control']}
    probabilities.update(BTCAverageRaw=BTCAverage().predict(validation),
                         BTCAverage=candidate.predict(validation))
    return dict(train_events=len(train), evaluated_events=len(validation),
                fit=candidate.artifact(), scores={n: metrics(validation, p) for n, p in probabilities.items()},
                price_buckets=diagnostics(validation, probabilities)), candidate, probabilities


def run(source, output):
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    protocol_file = output/'protocol.json'
    if protocol_file.exists() and json.loads(protocol_file.read_text()) != PROTOCOL:
        raise ValueError('Use a new output directory for a changed protocol')
    save(protocol_file, PROTOCOL)
    train, validation = read_development(source)
    folds = []; pooled_rows = []; pooled = {}
    for start, end in PROTOCOL['folds']:
        fitting = [r for r in train if stamp(r['settled_at']).timestamp() < unix(start)]
        scoring = [r for r in train if unix(start)+7200 <= stamp(r['at']).timestamp()
                   and stamp(r['settled_at']).timestamp() < unix(end)]
        if min(len(fitting), len(scoring)) < 50: raise ValueError('Insufficient forward fold')
        result, _, p = evaluate(fitting, scoring)
        folds.append(dict(start=start, end_exclusive=end, **result))
        pooled_rows.extend(scoring)
        for name, values in p.items(): pooled.setdefault(name, []).extend(values)
    final, model, _ = evaluate(train, validation)
    report = dict(protocol=PROTOCOL, test_loaded=False, deployment_allowed=False,
                  folds=folds, forward_scores={n: metrics(pooled_rows, p) for n,p in pooled.items()},
                  validation=final, no_proven_edge=True)
    save(output/'model.json', model.artifact())
    save(output/'report.json', report)
    save(output/'manifest.json', dict(
        inputs={n: digest((source/n).read_bytes()) for n in ['train.jsonl', 'validation.jsonl']},
        code={p.name: digest(p.read_bytes()) for p in [Path(__file__), Path(__file__).with_name('btc_research.py')]},
        outputs={p.name: digest(p.read_bytes()) for p in output.iterdir() if p.name != 'manifest.json' and p.is_file()}))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='reports/btc15m-pilot-v1')
    parser.add_argument('--output', default='reports/btc-average-calibration-v1')
    args = parser.parse_args()
    r = run(args.source, args.output)
    print(json.dumps({n: {k: round(v[k], 6) for k in ['log_loss', 'brier']}
                      for n, v in r['validation']['scores'].items()}, indent=2))
