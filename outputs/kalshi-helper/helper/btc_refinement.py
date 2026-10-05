"""Fixed BTC calibration/market-anchor ablation; reused development dates only."""
import argparse
import io
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .btc_calibration import read_development, PROTOCOL as OLD
from .btc_lab import read_verified
from .btc_regime import BTCRegime
from .btc_research import metrics, paired_interval, pnl_interval, save, digest, unix
from .core import stamp
from .sizing import simulate
from .btc_inputs import validated_execution_rows

PROTOCOL = dict(version='btc-refinement-v1', test_loaded=False,
    folds=OLD['folds'], calibration_train='Earlier settled Kalshi development outcomes only',
    base_train='January-April 2025 Binance; frozen architectures, never fit on September labels',
    candidates=['market', 'volatility', 'signal', 'regime', 'signal_calibrated',
                'market_calibrated', 'market_anchored'],
    ablations=['signal_calibrated_old_clock'],
    calibration='Two parameter sigmoid, L2=.01 toward identity, positive slope [0.25,2]',
    anchor='Market log-odds offset; intercept, price calibration, external signal disagreement; L2=.1 toward market',
    bounds='Anchor intercept [-1,1], market slope correction [-.25,.25], signal disagreement weight [0,.5]',
    execution='All evaluation probabilities recomputed on execution rows; original five-second quote availability assumption',
    trading=dict(bankroll=1000, min_edge=.04, slippage=.02, policy='bankroll_1_percent', fee_rate=.07),
    selection='No hyperparameter or trading threshold search; every candidate reported; no automatic promotion',
    limitations=['Reused dates, not a fresh holdout', 'No historical depth', 'Proxy prices differ from BRTI',
                 'Only six validation days; uncertainty is exploratory', 'No initial decision-time intent requirement in quote scenario'])


def logits(values):
    a = np.asarray(values, dtype=float)
    if a.ndim != 1 or not np.isfinite(a).all() or np.any((a < 0) | (a > 1)):
        raise ValueError('Require finite probability vector')
    return logit(np.clip(a, 1e-6, 1-1e-6))


class Correction:
    def __init__(self, kind):
        if kind not in ('signal_calibrated', 'market_calibrated', 'market_anchored'):
            raise ValueError('Unknown correction')
        self.kind = kind

    def design(self, market, signal):
        m, s = logits(market), logits(signal)
        if m.shape != s.shape: raise ValueError('Mismatched probabilities')
        if self.kind == 'signal_calibrated':
            return s, np.column_stack([np.ones(len(s)), s])
        columns = [np.ones(len(m)), m]
        if self.kind == 'market_anchored': columns.append(np.clip(s-m, -4, 4))
        return m, np.column_stack(columns)

    def fit(self, market, signal, y):
        offset, x = self.design(market, signal)
        y = np.asarray(y)
        if y.shape != offset.shape or set(y.tolist()) != {0, 1} or len(y) < 50:
            raise ValueError('Need at least 50 aligned binary labels with both outcomes')
        penalty = .01 if self.kind == 'signal_calibrated' else .1
        bounds = [(-1, 1), (-.75, 1)] if self.kind == 'signal_calibrated' else [(-1, 1), (-.25, .25)]
        if self.kind == 'market_anchored': bounds.append((0, .5))
        def objective(w):
            z = offset+x@w
            return (float(np.mean(np.logaddexp(0, z)-y*z)+penalty*(w@w)),
                    x.T@(expit(z)-y)/len(y)+2*penalty*w)
        result = minimize(objective, np.zeros(x.shape[1]), jac=True, method='L-BFGS-B', bounds=bounds)
        if not result.success: raise ValueError('Correction fit failed: '+result.message)
        self.coef = result.x
        return self

    def predict(self, market, signal):
        offset, x = self.design(market, signal)
        return np.clip(expit(offset+x@self.coef), 1e-6, 1-1e-6)


def external_models(experiment):
    experiment = Path(experiment)
    manifest = json.loads((experiment/'manifest.json').read_text())
    raw = read_verified(experiment, 'proxy_dataset.npz', manifest['outputs']['proxy_dataset.npz'])
    with np.load(io.BytesIO(raw), allow_pickle=False) as data:
        use = data['close_time'] < '2025-05-01T00:00:00+00:00'
        x, y = data['x'][use], data['y'][use]
    signal = make_pipeline(StandardScaler(), LogisticRegression(C=.1, max_iter=1000, random_state=1729)).fit(x, y)
    regime = BTCRegime().fit([dict(x=a.tolist(), y=int(b)) for a, b in zip(x, y)])
    return signal, regime


def split(train, start, end):
    fit = [r for r in train if stamp(r['settled_at']).timestamp() < unix(start)]
    score = [r for r in train if unix(start)+7200 <= stamp(r['at']).timestamp()
             and stamp(r['settled_at']).timestamp() < unix(end)]
    if min(len(fit), len(score)) < 50: raise ValueError('Insufficient temporal split')
    return fit, score


def evaluate(fitting, scoring, models):
    train, rows = validated_execution_rows(fitting), validated_execution_rows(scoring)
    if max(stamp(r['settled_at']) for r in train) >= min(stamp(r['at']) for r in rows):
        raise ValueError('Calibration overlaps evaluation')
    signal, regime = models
    predict = lambda data: signal.predict_proba([r['x'] for r in data])[:, 1]
    m, s = np.array([r['p'] for r in rows]), predict(rows)
    probabilities = dict(market=m, volatility=np.array([r['diffusion'] for r in rows]), signal=s, regime=regime.predict(rows))
    fitted = {}
    for kind in ('signal_calibrated', 'market_calibrated', 'market_anchored', 'signal_calibrated_old_clock'):
        data = fitting if kind.endswith('old_clock') else train
        model = Correction('signal_calibrated' if kind.endswith('old_clock') else kind).fit(
            [r['p'] for r in data], predict(data), [r['y'] for r in data])
        probabilities[kind] = model.predict(m, s)
        fitted[kind] = model.coef.tolist()
    results = {}
    for name, p in probabilities.items():
        paper = simulate(
            rows, p, policy='bankroll_1_percent', slippage=.02, bankroll=1000, min_edge=.04)
        paper['execution_recomputed'] = True
        paper['uncertainty'] = pnl_interval(paper, {r['group'] for r in rows})
        results[name] = dict(scores=metrics(rows, p), paired_brier=paired_interval(rows, p), paper=paper)
    return dict(training_events=len(train), scoring_events=len(rows),
                latest_training_settlement=max(r['settled_at'] for r in train),
                earliest_evaluation=min(r['at'] for r in rows), coefficients=fitted, models=results), rows, probabilities


def run(source, experiment, output):
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    protocol = output/'protocol.json'
    if protocol.exists() and json.loads(protocol.read_text()) != PROTOCOL: raise ValueError('Protocol changed')
    save(protocol, PROTOCOL)  # Freeze before observing candidate results.
    train, validation = read_development(source)
    models = external_models(experiment)
    folds, pooled_rows, pooled = [], [], {}
    for start, end in PROTOCOL['folds']:
        fit, score = split(train, start, end)
        result, rows, predictions = evaluate(fit, score, models)
        folds.append(dict(start=start, end=end, **result)); pooled_rows.extend(rows)
        for name, p in predictions.items(): pooled.setdefault(name, []).extend(p)
    final, _, _ = evaluate(train, validation, models)
    forward = {}
    for name, p in pooled.items():
        paper = simulate(pooled_rows, p, policy='bankroll_1_percent', slippage=.02, bankroll=1000, min_edge=.04)
        paper['uncertainty'] = pnl_interval(paper, {r['group'] for r in pooled_rows})
        forward[name] = dict(scores=metrics(pooled_rows, p), paper=paper)
    report = dict(protocol=PROTOCOL, folds=folds, forward=forward, validation=final,
                  test_loaded=False, deployment_allowed=False, no_proven_edge=True)
    save(output/'report.json', report)
    save(output/'model.json', dict(coefficients=final['coefficients'], protocol=PROTOCOL))
    save(output/'manifest.json', dict(inputs={n:digest((source/n).read_bytes()) for n in ('train.jsonl','validation.jsonl')},
        external_manifest=digest((Path(experiment)/'manifest.json').read_bytes()),
        code={n:digest(Path(__file__).with_name(n).read_bytes()) for n in
              ('btc_refinement.py','btc_inputs.py','btc_research.py','btc_calibration.py','btc_regime.py','btc_lab.py','sizing.py','core.py','live.py')},
        outputs={name:digest((output/name).read_bytes()) for name in ('protocol.json','report.json','model.json')}))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='reports/btc15m-pilot-v1')
    parser.add_argument('--experiment', default='reports/btc-binance-history-v1')
    parser.add_argument('--output', default='reports/btc-refinement-v1')
    args = parser.parse_args()
    report = run(args.source, args.experiment, args.output)
    print(json.dumps({part:{n:dict(log_loss=v['scores']['log_loss'], brier=v['scores']['brier'],
        pnl=v['paper']['pnl'], entries=v['paper']['entries']) for n,v in
        (report['forward'] if part=='forward' else report['validation']['models']).items()}
        for part in ('forward','validation')}, indent=2))
