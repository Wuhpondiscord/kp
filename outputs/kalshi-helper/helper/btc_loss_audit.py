"""Fixed-cohort loss attribution, not a strategy search or a new holdout."""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .btc_research import digest,execution_rows,metrics,save
from .btc_regime import BTCRegime
from .core import stamp


def attribute(rows,probabilities,ledger,slippage=.02):
    if len(rows)!=len(probabilities):raise ValueError('Mismatched predictions')
    if not np.isfinite(probabilities).all() or np.any((np.asarray(probabilities)<0)|(np.asarray(probabilities)>1)):
        raise ValueError('Invalid probabilities')
    lookup={r['ticker']:(r,float(p)) for r,p in zip(rows,probabilities)}
    trades=[]
    for t in ledger:
        r,p=lookup[t['ticker']];q=t['quantity']
        if q<1 or int(q)!=q:raise ValueError('Invalid filled quantity')
        payout=(t['pnl']+t['cost'])/q
        if min(abs(payout),abs(payout-1))>1e-7:raise ValueError('Nonbinary ledger payout')
        win=int(payout>.5);side='yes' if r['y']==win else 'no'
        price=r['ask'] if side=='yes' else 1-r['bid']
        midpoint=r['p'] if side=='yes' else 1-r['p']
        executed=min(1,price+slippage)
        if abs(q*executed+t['fees']-t['cost'])>1e-7:raise ValueError('Ledger cost does not reconcile')
        predicted=p if side=='yes' else 1-p
        trades.append(dict(ticker=t['ticker'],day=t['day'],side=side,quantity=q,price=price,
            predicted_win=predicted,actual_win=win,market_win=midpoint,
            gross_midpoint_pnl=q*(win-midpoint),spread_cost=q*(price-midpoint),
            slippage_cost=q*(executed-price),fees=t['fees'],net_pnl=t['pnl'],
            predicted_net=q*predicted-t['cost'],equal_contract_net=win-t['cost']/q))
    sums={key:sum(t[key] for t in trades) for key in
          ['gross_midpoint_pnl','spread_cost','slippage_cost','fees','net_pnl','predicted_net','equal_contract_net']}
    if abs(sums['gross_midpoint_pnl']-sums['spread_cost']-sums['slippage_cost']-sums['fees']-sums['net_pnl'])>1e-6:
        raise ValueError('Loss attribution fails to reconcile')
    quantity=sum(t['quantity'] for t in trades)
    if not trades:return dict(summary=dict(entries=0,contracts=0,**sums),entry_price_buckets={},trades=[])
    summary=dict(entries=len(trades),contracts=quantity,**sums,
        predicted_win_rate=float(np.mean([t['predicted_win'] for t in trades])),
        actual_win_rate=float(np.mean([t['actual_win'] for t in trades])),
        market_implied_win_rate=float(np.mean([t['market_win'] for t in trades])),
        weighted_predicted_win_rate=sum(t['quantity']*t['predicted_win'] for t in trades)/quantity,
        weighted_actual_win_rate=sum(t['quantity']*t['actual_win'] for t in trades)/quantity,
        note='Costs removed on identical filled contracts and quantities, not a reoptimized cost-free strategy')
    days=sorted({t['day'] for t in trades})
    blocks=[np.array([t['predicted_win']-t['actual_win'] for t in trades if t['day']==d]) for d in days]
    rng=np.random.default_rng(1729)
    differences=[float(np.concatenate([blocks[i] for i in rng.integers(0,len(blocks),len(blocks))]).mean()) for _ in range(3000)]
    summary['selected_probability_overstatement_95_day_interval']=np.quantile(differences,[.025,.975]).tolist()
    summary['uncertainty_note']='Exploratory day bootstrap on fixed selected trades; ignores selection history and cross-day dependence'
    buckets={}
    for low,high in [(0,.1),(.1,.3),(.3,.7),(.7,1.01)]:
        group=[t for t in trades if low<=t['price']<high]
        buckets[f'{low}-{min(high,1)}']=dict(entries=len(group),contracts=sum(t['quantity'] for t in group),
            net_pnl=sum(t['net_pnl'] for t in group),
            equal_contract_net=sum(t['equal_contract_net'] for t in group))
    return dict(summary=summary,entry_price_buckets=buckets,trades=trades)


def run(source,experiment,output):
    source,experiment,output=Path(source),Path(experiment),Path(output);output.mkdir(parents=True,exist_ok=True)
    report=json.loads((experiment/'report.json').read_text());manifest=json.loads((experiment/'manifest.json').read_text())
    for name,sha in manifest['outputs'].items():
        if digest((experiment/name).read_bytes())!=sha:raise ValueError('Experiment artifact mismatch')
    rows=[]
    for name in ['train.jsonl','validation.jsonl','test.jsonl']:
        if digest((source/name).read_bytes())!=manifest['kalshi_inputs'][name]:raise ValueError('Kalshi input mismatch')
        rows.extend(json.loads(x) for x in (source/name).read_text().splitlines())
    rows.sort(key=lambda r:r['at']);execution=execution_rows(rows)
    with np.load(experiment/'proxy_dataset.npz',allow_pickle=False) as data:
        train=data['close_time']<'2025-05-01T00:00:00+00:00';x=data['x'][train];y=data['y'][train]
    logistic=make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=1000,random_state=1729)).fit(x,y)
    regime=BTCRegime().fit([dict(x=a.tolist(),y=int(b)) for a,b in zip(x,y)])
    probabilities=dict(volatility=np.array([r['diffusion'] for r in execution]),
        logistic=logistic.predict_proba([r['x'] for r in execution])[:,1],regime=regime.predict(execution))
    timing=dict(decision_to_execution_seconds=sorted({(stamp(r['execution_at'])-stamp(r['at'])).total_seconds() for r in rows if r['execution_at']}),
        decision_feature_age_seconds=sorted({(stamp(r['at'])-stamp(r['feature_time'])).total_seconds() for r in rows}),
        execution_feature_age_seconds=sorted({(stamp(r['at'])-stamp(r['feature_time'])).total_seconds() for r in execution}))
    result=dict(purpose='Retrospective diagnosis of already reported losing policies; no tuning',timing=timing,models={})
    for name,p in probabilities.items():
        paper=report['retrospective_september_replay']['models'][name]['paper']
        audit=attribute(execution,p,paper['ledger'])
        if abs(audit['summary']['net_pnl']-paper['pnl'])>1e-6:raise ValueError('Reported P&L mismatch')
        audit['execution_forecast_scores']=metrics(execution,p)
        audit['execution_market_scores']=metrics(execution,np.array([r['p'] for r in execution]))
        result['models'][name]=audit
    train_mean=x.mean(axis=0);train_sd=x.std(axis=0)
    target=np.array([r['x'] for r in execution])
    result['feature_shift']=dict(mean_shift_training_std=((target.mean(axis=0)-train_mean)/np.maximum(train_sd,1e-8)).tolist(),
        features=['distance','return1','return5','return15','return60','volatility_ratio','volume_ratio','range','hour_sin','hour_cos'])
    save(output/'report.json',result)
    save(output/'manifest.json',dict(code_sha256=digest(Path(__file__).read_bytes()),
        source_manifest_sha256=digest((experiment/'manifest.json').read_bytes()),report_sha256=digest((output/'report.json').read_bytes())))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',default='reports/btc15m-pilot-v1')
    parser.add_argument('--experiment',default='reports/btc-binance-history-v1');parser.add_argument('--output',default='reports/btc-loss-audit')
    args=parser.parse_args();r=run(args.source,args.experiment,args.output)
    print(json.dumps(dict(timing=r['timing'],models={k:v['summary'] for k,v in r['models'].items()}),indent=2))
