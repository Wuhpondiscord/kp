"""Fixed market anchor, age stress and trade attribution on development data only."""
import argparse
import json
from pathlib import Path
from urllib.parse import urlencode
import numpy as np
from .btc_flow_followup import read_rows, split, evaluate, PROTOCOL as SOURCE_PROTOCOL
from .btc_flow_model import FlowCorrection
from .btc_new_history import new_features, parse_flow
from .btc_long_history import download
from .btc_research import Cache, digest, save, iso, unix, paired_interval
from .core import stamp, dec, fee_and_cost
from .live import qualify_signal

PROTOCOL = dict(version='btc-anchor-study-v1', delays_seconds=[0,60,120,300],
    folds=SOURCE_PROTOCOL['folds'],
    candidate='Fixed market log odds plus four standardized external features; no fitted intercept or market slope',
    controls=['unmodified market','market-only calibration','previous full-flow correction'],
    fitting='L2 .1, bounds +/-.25, training-only scaling unchanged; no tuning',
    stress='Fit on current inputs, delay external inputs only at inference, keep current market quote',
    trade='Same 1000 bankroll, 1% sizing, .04 net edge, .02 slippage, .07 fee coefficient',
    gate='Both Brier and log loss beat both market controls in >=3 of 4 folds and pooled; positive net P&L, >=30 trades and >=10 trade days; never automatic deployment/holdout release',
    limitations=['Previously inspected development dates','Historical arrival times unavailable; delay is a scenario',
                 'No historical depth/fills','June fee coefficient assumed; no override audit'])


class FixedAnchor(FlowCorrection):
    """Learn only external-feature corrections, with an unchanged market offset."""
    def design(self,rows,fit=False):
        offset,x=super().design(rows,fit)
        return offset,x[:,2:]
    def artifact(self):return dict(super().artifact(),kind='fixed_market_anchor')
    @classmethod
    def load(cls,data):
        if data.get('kind')!='fixed_market_anchor':raise ValueError('Wrong model kind')
        model=cls()
        for key in ('mean','scale','coef'):
            value=np.asarray(data[key],dtype=float)
            if value.shape!=(4,) or not np.isfinite(value).all():raise ValueError('Invalid anchor artifact')
            setattr(model,key,value)
        if np.any(model.scale<=0) or np.any(np.abs(model.coef)>.25):raise ValueError('Invalid anchor parameters')
        return model


def freeze(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);path=root/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=PROTOCOL:raise ValueError('Protocol changed')
    if not path.exists():save(path,PROTOCOL)


def prepare(source,cache_path,root):
    """Reconstruct stress features from verified public cache; no network or holdout."""
    root=Path(root);freeze(root);rows=read_rows(Path(source));flow={};archives={}
    for month in ('2026-06','2026-07'):
        _,archives[month]=download(cache_path,month,offline=True)
        flow.update(parse_flow((Path(cache_path)/f'BTCUSDT-1m-{month}.zip').read_bytes(),month))
    cache=Cache(cache_path,offline=True);bars={};end=unix(SOURCE_PROTOCOL['end'])
    for a in range(unix(SOURCE_PROTOCOL['start'])-3*3600,end,240*60):
        params=dict(start=iso(a),end=iso(min(a+240*60,end)),granularity=60)
        for r in cache.get('https://api.exchange.coinbase.com/products/BTC-USD/candles?'+urlencode(params)):
            if a<=r[0]<min(a+240*60,end):bars[int(r[0])]=r
    # All scenarios must use exactly the same contracts. Fail rather than silently drop rows.
    features={}
    for r in rows:
        features[r['ticker']]={str(delay):new_features(flow,bars,iso(stamp(r['at']).timestamp()-delay))
                               for delay in PROTOCOL['delays_seconds']}
        np.testing.assert_allclose(features[r['ticker']]['0']['flow_x'],r['flow_x'],rtol=0,atol=1e-12)
        np.testing.assert_allclose(features[r['ticker']]['60']['flow_x'],r['lag_flow_x'],rtol=0,atol=1e-12)
    save(root/'age-features.json',features)
    save(root/'inputs.json',dict(source_sha256=digest((Path(source)/'development.jsonl').read_bytes()),
        age_sha256=digest((root/'age-features.json').read_bytes()),archives=archives,requests=cache.used))


def delayed_rows(rows,features,delay):
    if delay not in PROTOCOL['delays_seconds']:raise ValueError('Unregistered age scenario')
    result=[]
    for r in rows:
        f=features[r['ticker']][str(delay)]
        available=stamp(f['flow_available_at']);end=stamp(f['flow_feature_time'])
        cutoff=stamp(r['at']).timestamp()-delay
        if not end<=available or available.timestamp()>cutoff or cutoff-end.timestamp()>65:
            raise ValueError('Invalid aged input availability')
        if len(f['flow_x'])!=4 or not np.isfinite(f['flow_x']).all():raise ValueError('Invalid aged vector')
        result.append(dict(r,**f))
    return result


def trade_attribution(rows,p,paper):
    """Fixed executed cohort: midpoint gross minus spread, slippage and fees."""
    lookup={r['ticker']:(r,float(v)) for r,v in zip(rows,p)};details=[]
    totals={k:0. for k in ('midpoint_gross','spread_cost','slippage_cost','fees','predicted_net','realized_net')}
    funnel=dict(raw_edge=0,after_spread=0,after_fees=0,after_slippage=0,filled=paper['entries'])
    for r,v in zip(rows,p):
        midpoint=(r['bid']+r['ask'])/2
        funnel['raw_edge']+=int(abs(v-midpoint)>=.04)
        funnel['after_spread']+=int(max(v-r['ask'],1-v-(1-r['bid']))>=.04)
        intent=qualify_signal(float(v),r['bid'],r['ask'],'.07',.04)
        if intent:
            funnel['after_fees']+=1;yes=intent['signal_side']=='yes'
            win=dec(v) if yes else 1-dec(v)
            price=min(dec(1),(dec(r['ask']) if yes else 1-dec(r['bid']))+dec('.02'))
            _,cost=fee_and_cost(price,1,dec('.07'))
            funnel['after_slippage']+=int(win-cost>=dec('.04'))
    for trade in paper['ledger']:
        r,v=lookup[trade['ticker']];intent=qualify_signal(v,r['bid'],r['ask'],'.07',.04)
        if intent is None:raise ValueError('Fill without signal')
        yes=intent['signal_side']=='yes';q=trade['quantity']
        mid=r['p'] if yes else 1-r['p'];ask=r['ask'] if yes else 1-r['bid']
        win=v if yes else 1-v;outcome=r['y'] if yes else 1-r['y']
        components=dict(midpoint_gross=(outcome-mid)*q,spread_cost=(ask-mid)*q,
            slippage_cost=(min(1,ask+.02)-ask)*q,fees=trade['fees'],
            predicted_net=win*q-trade['cost'],realized_net=trade['pnl'])
        reconstructed=components['midpoint_gross']-components['spread_cost']-components['slippage_cost']-components['fees']
        if abs(reconstructed-trade['pnl'])>1e-8:raise ValueError('Cost attribution mismatch')
        for k in totals:totals[k]+=components[k]
        details.append(dict(ticker=r['ticker'],side='yes' if yes else 'no',quantity=q,probability=win,outcome=outcome,**components))
    return dict(funnel=funnel,totals=totals,trades=details,
        note='Fixed filled cohort attribution; expected profit uses model probabilities, not independently verified edge')


def drift(train,test):
    a=np.asarray([r['flow_x'] for r in train]);b=np.asarray([r['flow_x'] for r in test])
    scale=np.maximum(a.std(axis=0),1e-6)
    return dict(train_mean=a.mean(axis=0).tolist(),test_mean=b.mean(axis=0).tolist(),
        mean_shift_training_sd=((b.mean(axis=0)-a.mean(axis=0))/scale).tolist(),
        test_to_train_sd=(b.std(axis=0)/scale).tolist(),
        clipped_fraction=(np.abs((b-a.mean(axis=0))/scale)>5).mean(axis=0).tolist())


def candidate_gate(folds,aggregate):
    def beats(result):
        return all(result['fixed_anchor']['scores'][metric]<result[c]['scores'][metric]
            for metric in ('brier','log_loss') for c in ('market','market_control'))
    wins=sum(beats(f['models']) for f in folds);paper=aggregate['fixed_anchor']['paper']
    checks=dict(folds_beating_both_controls=wins>=3,pooled_beats_both_controls=beats(aggregate),
        positive_net=paper['pnl']>0,enough_trades=paper['entries']>=30,enough_trade_days=paper['traded_city_days']>=10)
    return dict(checks=checks,fold_wins=wins,passed=all(checks.values()),automatic_release=False)


def run(source,root):
    source=Path(source);root=Path(root);freeze(root);rows=read_rows(source)
    meta=json.loads((root/'inputs.json').read_text())
    if digest((source/'development.jsonl').read_bytes())!=meta['source_sha256'] or digest((root/'age-features.json').read_bytes())!=meta['age_sha256']:
        raise ValueError('Input hash mismatch')
    features=json.loads((root/'age-features.json').read_text());allrows=[];folds=[]
    pooled={str(d):{n:[] for n in ('market','market_control','full_flow','fixed_anchor')} for d in PROTOCOL['delays_seconds']}
    for start,end in PROTOCOL['folds']:
        train,test=split(rows,start,end);allrows.extend(test)
        models=dict(market_control=FlowCorrection(False).fit(train),full_flow=FlowCorrection().fit(train),fixed_anchor=FixedAnchor().fit(train))
        save(root/f'models-{start}.json',{n:m.artifact() for n,m in models.items()})
        scenarios={}
        for delay in PROTOCOL['delays_seconds']:
            current=delayed_rows(test,features,delay)
            predictions=dict(market=np.asarray([r['p'] for r in test]),**{n:m.predict(current) for n,m in models.items()})
            scores={n:evaluate(test,p) for n,p in predictions.items()}
            for n,p in predictions.items():pooled[str(delay)][n].extend(p.tolist())
            scenarios[str(delay)]=scores
        folds.append(dict(start=start,end=end,training_rows=len(train),evaluation_rows=len(test),
            drift=drift(train,test),models=scenarios['0'],age_scenarios=scenarios))
        print('Evaluated',start,flush=True)
    aggregate={};ages={}
    for delay,predictions in pooled.items():
        results={}
        for n,p in predictions.items():
            result=evaluate(allrows,np.asarray(p))
            result['trade_attribution']=trade_attribution(allrows,p,result['paper'])
            results[n]=result
        controlrows=[dict(r,p=p) for r,p in zip(allrows,predictions['market_control'])]
        results['fixed_anchor']['paired_vs_market_control']=paired_interval(controlrows,np.asarray(predictions['fixed_anchor']))
        ages[delay]=results
    aggregate=ages['0']
    report=dict(protocol=PROTOCOL,rerun=(root/'report.json').exists(),holdout_loaded=False,deployment_allowed=False,
        historical_arrival_times_measured=False,evaluation_rows=len(allrows),folds=folds,aggregate=aggregate,
        age_scenarios=ages,candidate_gate=candidate_gate(folds,aggregate),
        input_hashes=meta,code_hashes={p.name:digest(p.read_bytes()) for p in [Path(__file__),
        *[Path(__file__).with_name(n) for n in ('btc_flow_followup.py','btc_flow_model.py','btc_new_history.py','btc_research.py','sizing.py','core.py','live.py')]]})
    save(root/'report.json',report)
    for n,v in aggregate.items():print(n,v['scores']['brier'],v['scores']['log_loss'],v['paper']['pnl'],v['paper']['entries'])
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',default='reports/btc-flow-followup-v1')
    p.add_argument('--output',default='reports/btc-anchor-study-v1');p.add_argument('--cache');a=p.parse_args()
    if a.cache:prepare(a.source,a.cache,a.output)
    run(a.source,a.output)
