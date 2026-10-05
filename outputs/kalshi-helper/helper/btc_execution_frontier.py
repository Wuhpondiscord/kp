"""Execution feasibility diagnostic; frozen models, unchanged entry rule, no fitting."""
import argparse
import json
from pathlib import Path
import numpy as np
from .btc_anchor_study import FixedAnchor
from .btc_flow_model import FlowCorrection, trade_uncertainty
from .btc_flow_followup import read_rows, split, PROTOCOL as DATA_PROTOCOL
from .btc_research import save, digest, metrics
from .core import dec, fee_and_cost
from .live import qualify_signal
from .sizing import simulate

PROTOCOL=dict(version='btc-execution-frontier-v1',slippage=[0,.005,.01,.02,.05],
    models=['fixed_anchor','full_flow'],bankroll=1000,min_edge=.04,
    selection='No cost scenario may be selected as a production setting or promotion evidence',
    sizing='Existing bankroll_1_percent and shared exposure caps, .07 fee, conservative cent rounding',
    frontier='Maximum extra price per contract preserving the existing one-contract .04 qualification; 0.0001 grid',
    fixed_cohort='Zero-slippage filled trades, same side/quantity for all cost scenarios; diagnostic only',
    restrictions='Previously inspected development dates; no training, new strategies or holdout access')


def tolerance(probability,price):
    """Max extra buy price passing conservative one-contract gate, no outcome input."""
    win=dec(probability);base=dec(price)
    if not win.is_finite() or not base.is_finite() or not 0<=win<=1 or not 0<=base<=1:
        raise ValueError('Invalid probability/price')
    def accepts(ticks):
        _,cost=fee_and_cost(base+dec(ticks)/10000,1,dec('.07'))
        return win-cost>=dec('.04')
    if not accepts(0):return None
    low=0;high=int((1-base)*10000)
    while low<high:
        mid=(low+high+1)//2
        if accepts(mid):low=mid
        else:high=mid-1
    return low/10000


def fixed_cohort(rows,p,base_paper,slippage):
    """Reprice a fixed observed fill cohort, without selecting different trades."""
    lookup={r['ticker']:(r,float(v)) for r,v in zip(rows,p)};ledger=[];expected=0.
    for trade in base_paper['ledger']:
        r,v=lookup[trade['ticker']];intent=qualify_signal(v,r['bid'],r['ask'],'.07',.04)
        if intent is None:raise ValueError('Missing fixed-cohort intent')
        yes=intent['signal_side']=='yes';win=v if yes else 1-v;outcome=r['y'] if yes else 1-r['y']
        price=min(dec(1),(dec(r['ask']) if yes else 1-dec(r['bid']))+dec(slippage))
        fee,cost=fee_and_cost(price,trade['quantity'],dec('.07'))
        pnl=float(dec(outcome)*trade['quantity']-cost);expected+=win*trade['quantity']-float(cost)
        ledger.append(dict(trade,cost=float(cost),fees=float(fee),pnl=pnl))
    contracts=sum(t['quantity'] for t in ledger);invested=sum(t['cost'] for t in ledger);net=sum(t['pnl'] for t in ledger)
    return dict(entries=len(ledger),contracts=contracts,capital_deployed=invested,pnl=net,expected_net=expected,
        net_per_contract=net/contracts if contracts else None,return_on_deployed=net/invested if invested else None,
        ledger=ledger,note='Same zero-slip fills and quantities. Not executable policy replay; higher costs may violate original sizing or edge caps.')


def analyze(rows,p):
    scenarios={};signals=[]
    for r,v in zip(rows,p):
        intent=qualify_signal(float(v),r['bid'],r['ask'],'.07',.04)
        if not intent:continue
        yes=intent['signal_side']=='yes';win=float(v) if yes else 1-float(v);price=r['ask'] if yes else 1-r['bid']
        signals.append(dict(ticker=r['ticker'],day=r['group'],side=intent['signal_side'],
            probability=win,quote_price=price,max_slippage=tolerance(win,price)))
    for slip in PROTOCOL['slippage']:
        paper=simulate(rows,p,policy='bankroll_1_percent',slippage=slip,bankroll=1000,min_edge=.04)
        paper['uncertainty']=trade_uncertainty(paper,{r['group'] for r in rows})
        scenarios[str(slip)]=paper
    fixed={str(s):fixed_cohort(rows,p,scenarios['0'],s) for s in PROTOCOL['slippage']}
    return dict(scores=metrics(rows,p),signals=signals,scenarios=scenarios,fixed_cohort=fixed,
        tolerance_summary=dict(signals=len(signals),median=float(np.median([s['max_slippage'] for s in signals])) if signals else None,
            maximum=max((s['max_slippage'] for s in signals),default=None)))


def run(source,models,output):
    source=Path(source);models=Path(models);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    files=[source/'development.jsonl',source/'manifest.json',source/'protocol.json',models/'report.json',
           *[models/f'models-{a}.json' for a,_ in DATA_PROTOCOL['folds']]]
    frozen=dict(protocol=PROTOCOL,inputs={str(p.resolve()):digest(p.read_bytes()) for p in files})
    path=output/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=frozen:raise ValueError('Frozen inputs/protocol changed')
    if not path.exists():save(path,frozen)
    rows=read_rows(source);allrows=[];preds={n:[] for n in PROTOCOL['models']};folds=[]
    original=json.loads((models/'report.json').read_text())
    for start,end in DATA_PROTOCOL['folds']:
        _,test=split(rows,start,end);allrows.extend(test)
        data=json.loads((models/f'models-{start}.json').read_text())
        fitted=dict(fixed_anchor=FixedAnchor.load(data['fixed_anchor']),full_flow=FlowCorrection.load(data['full_flow']))
        results={}
        for n,m in fitted.items():
            p=m.predict(test);preds[n].extend(p.tolist());results[n]=analyze(test,p)
        folds.append(dict(start=start,end=end,models=results))
    results={n:analyze(allrows,np.asarray(p)) for n,p in preds.items()}
    # Fail if loading saved models has changed the primary experiment.
    for n,result in results.items():
        for metric in ('brier','log_loss'):
            if abs(result['scores'][metric]-original['aggregate'][n]['scores'][metric])>1e-12:
                raise ValueError('Frozen model score mismatch')
        if result['scenarios']['0.02']['ledger']!=original['aggregate'][n]['paper']['ledger']:
            raise ValueError('Frozen primary ledger mismatch')
    report=dict(protocol=PROTOCOL,rerun=(output/'report.json').exists(),holdout_loaded=False,training_performed=False,
        deployment_allowed=False,rows=len(allrows),models=results,folds=folds,
        code_hashes={p.name:digest(p.read_bytes()) for p in [Path(__file__),*[Path(__file__).with_name(n) for n in
            ('btc_anchor_study.py','btc_flow_model.py','btc_flow_followup.py','sizing.py','core.py','live.py')]]})
    save(output/'report.json',report)
    for n,r in results.items():
        print(n,r['tolerance_summary'])
        for s,v in r['scenarios'].items():print(s,v['pnl'],v['entries'])
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',default='reports/btc-flow-followup-v1')
    p.add_argument('--models',default='reports/btc-anchor-study-v1');p.add_argument('--output',default='reports/btc-execution-frontier-v1')
    a=p.parse_args();run(a.source,a.models,a.output)
