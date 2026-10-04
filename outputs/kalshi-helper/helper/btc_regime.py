"""Nonlinear BTC forecasting and a separately learned trade filter; development only."""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .btc_calibration import read_development, PROTOCOL as CALIBRATION
from .btc_research import BTCModels, metrics, execution_rows, save, digest, unix, pnl_interval
from .core import stamp, fee_and_cost, dec
from .sizing import simulate

PROTOCOL = dict(version='btc-regime-gate-v1', test_loaded=False,
    forecast='Shallow histogram trees: 80 iterations, 4 leaves, min leaf 60, L2 10, learning rate .05; no early stopping',
    forecast_inputs='Ten original Coinbase features; no Kalshi prices',
    gate='Standardized ridge alpha=100 on realized YES/NO per-contract net returns from chronological out-of-fold forecasts',
    gate_inputs=['win_probability','estimated_net_edge','spread','absolute_target_distance','volatility_ratio'],
    gate_training='First 40% of training dates warmup; three expanding blocks of remaining dates, two-hour embargo',
    execution='Recomputed next-minute inputs; assumed fills with 2-cent slippage and .07 quadratic fees',
    policies=['fixed_edge','learned_gate','learned_gate_quarter_kelly_stops'],
    selection='Fixed experiment; no parameter grid or original test access',
    limitations=['Reused development dates','Coinbase proxy','No historical depth',
                 'Gate return estimates are not confidence bounds','Shift from original training to next-minute horizon'])


class BTCRegime:
    def fit(self, rows):
        self.model = HistGradientBoostingClassifier(max_iter=80, max_leaf_nodes=4,
            min_samples_leaf=60, l2_regularization=10, learning_rate=.05,
            early_stopping=False, random_state=1729).fit([r['x'] for r in rows], [r['y'] for r in rows])
        return self

    def predict(self, rows):
        return np.clip(self.model.predict_proba([r['x'] for r in rows])[:,1], 1e-6, 1-1e-6)


def side_features(row, probability, side):
    win = probability if side == 'yes' else 1-probability
    price = min(1., (row['ask'] if side == 'yes' else 1-row['bid'])+.02)
    _, cost = fee_and_cost(dec(str(price)), 1, dec('.07'))
    return [win, win-float(cost), row['spread'], abs(row['x'][0]), row['x'][5]], float(cost)


class TradeGate:
    def fit(self, rows, probabilities):
        self.validate(rows, probabilities)
        x, y = [], []
        for row, p in zip(rows, probabilities):
            for side in ['yes','no']:
                features, cost = side_features(row, float(p), side)
                x.append(features); y.append((row['y'] if side=='yes' else 1-row['y'])-cost)
        if len(x)<100: raise ValueError('Insufficient out-of-fold gate examples')
        self.model = make_pipeline(StandardScaler(), Ridge(alpha=100)).fit(x,y)
        return self

    @staticmethod
    def validate(rows, probabilities):
        p=np.asarray(probabilities,dtype=float)
        if p.shape!=(len(rows),) or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
            raise ValueError('Require one finite probability per row')

    def filter(self, rows, probabilities):
        self.validate(rows, probabilities)
        # Preserve the forecast for reporting; only the execution copy abstains.
        filtered=[]; accepted=0
        for r,p in zip(rows, probabilities):
            choices=[side_features(r,float(p),s)[0] for s in ['yes','no']]
            side=int(choices[1][1]>choices[0][1])
            keep=choices[side][1]>=.04 and self.model.predict([choices[side]])[0]>0
            filtered.append(float(p) if keep else r['p']); accepted+=int(keep)
        return np.asarray(filtered),accepted


def gate_training(rows):
    days=sorted({r['group'] for r in rows}); start=max(2,int(len(days)*.4))
    blocks=np.array_split(days[start:],3); output=[]; probabilities=[]; audit=[]
    for block in blocks:
        if len(block)==0:continue
        cutoff=unix(str(block[0]))
        fitting=[r for r in rows if stamp(r['settled_at']).timestamp()<cutoff]
        scoring=[r for r in rows if r['group'] in block and stamp(r['at']).timestamp()>=cutoff+7200]
        if len(fitting)<100 or not scoring:continue
        execution=execution_rows(scoring)
        if not execution:continue
        model=BTCRegime().fit(fitting)
        output.extend(execution);probabilities.extend(model.predict(execution))
        audit.append(dict(train_latest_settlement=max(r['settled_at'] for r in fitting),
                          score_earliest_decision=min(r['at'] for r in scoring),
                          training_events=len(fitting),scored_events=len(execution)))
    return output,probabilities,audit


def evaluate(train, validation):
    model=BTCRegime().fit(train); old=BTCModels().fit(train)
    gate_rows,gate_p,audit=gate_training(train); gate=TradeGate().fit(gate_rows,gate_p)
    scores={n:metrics(validation,old.predict(validation,n)) for n in
            ['market','BTCVolatility','BTCSignal','BTCMarketGuard','price_only_control']}
    scores['BTCRegime']=metrics(validation,model.predict(validation))
    execution=execution_rows(validation); p=model.predict(execution)
    filtered,accepted=gate.filter(execution,p)
    strategies={}
    for name,probability,policy,stops in [
        ('BTCMarketGuard_fixed',old.predict(execution,'BTCMarketGuard'),'bankroll_1_percent',False),
        ('BTCRegime_fixed',p,'bankroll_1_percent',False),
        ('BTCRegime_gate',filtered,'bankroll_1_percent',False),
        ('BTCRegime_gate_risk',filtered,'quarter_kelly',True)]:
        result=simulate(execution,probability,policy=policy,slippage=.02,bankroll=1000,min_edge=.04,loss_stops=stops)
        result['execution_recomputed']=True
        result['uncertainty']=pnl_interval(result,{r['group'] for r in validation})
        strategies[name]=result
    return dict(scores=scores,strategies=strategies,gate_training_audit=audit,
        gate_training_events=len(gate_rows),execution_events=len(execution),gate_accepted=accepted,
        gate_coefficients=gate.model[-1].coef_.tolist(),gate_intercept=float(gate.model[-1].intercept_))


def run(source,output):
    source,output=Path(source),Path(output);output.mkdir(parents=True,exist_ok=True)
    frozen=output/'protocol.json'
    if frozen.exists() and json.loads(frozen.read_text())!=PROTOCOL:raise ValueError('Protocol mismatch')
    save(frozen,PROTOCOL)
    train,validation=read_development(source); folds=[]
    for start,end in CALIBRATION['folds']:
        fitting=[r for r in train if stamp(r['settled_at']).timestamp()<unix(start)]
        scoring=[r for r in train if unix(start)+7200<=stamp(r['at']).timestamp() and stamp(r['settled_at']).timestamp()<unix(end)]
        folds.append(dict(start=start,end_exclusive=end,**evaluate(fitting,scoring)))
    report=dict(protocol=PROTOCOL,folds=folds,validation=evaluate(train,validation),
                deployment_allowed=False,no_proven_edge=True)
    save(output/'report.json',report)
    app=Path(__file__).parent.parent
    save(output/'manifest.json',dict(inputs={n:digest((source/n).read_bytes()) for n in ['train.jsonl','validation.jsonl']},
        code={n:digest((app/n).read_bytes()) for n in ['helper/btc_regime.py','helper/btc_research.py','helper/btc_calibration.py','helper/sizing.py']},
        outputs={p.name:digest(p.read_bytes()) for p in output.iterdir() if p.name!='manifest.json' and p.is_file()}))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',default='reports/btc15m-pilot-v1')
    parser.add_argument('--output',default='reports/btc-regime-gate-v1');args=parser.parse_args()
    r=run(args.source,args.output)
    print(json.dumps(dict(scores=r['validation']['scores'],trading={n:{k:v[k] for k in
        ['pnl','entries','contracts','capital_deployed','return_on_deployed']} for n,v in r['validation']['strategies'].items()}),indent=2))
