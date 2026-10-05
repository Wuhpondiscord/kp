"""Controlled audit of bootstrap boundary weights and moving feature origins."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from .btc_anchor_study import FixedAnchor
from .btc_stability_model import StabilityAnchor, paper_scenarios, guarded_probabilities
from .btc_flow_followup import read_rows, split, evaluate, PROTOCOL as DATA_PROTOCOL
from .btc_research import digest, save, paired_interval

PROTOCOL=dict(version='btc-stability-repair-v1',
    variants=['single','original_ensemble','shared_scaler','boundary_weighted','both_repairs'],
    samples='Reuse exact 32 saved two-day training resamples in each fold, no resampling search',
    boundary='Inverse expected day multiplicity under original sampler; normalize weighted loss per member',
    scaler='For shared variants use full training fold mean/std, never evaluation data',
    constants='Unchanged L2 .1, coefficient bounds +/-.25, feature set, costs, risk and .04 threshold',
    limits='Development-only 2x2 mechanism audit, no holdout access, automatic promotion or claim of independent confirmation')


def expected_day_counts(days):
    """Expected multiplicity of each day under saved nonwrapping 2-day sampler."""
    from datetime import date
    n=len(days)
    starts=[i for i in range(n-1) if (date.fromisoformat(days[i+1])-date.fromisoformat(days[i])).days==1]
    if not starts:raise ValueError('No contiguous pairs')
    result=np.zeros(n)
    for i in starts:
        result[i:i+2]+=n//2/len(starts)
        if n%2:result[i]+=1/len(starts)
    if np.any(result<=0):raise ValueError('Sampler cannot include every training day')
    return dict(zip(days,result.tolist()))


def weighted_fit(rows,weights,mean=None,scale=None):
    w=np.asarray(weights,dtype=float)
    if w.shape!=(len(rows),) or not np.isfinite(w).all() or np.any(w<=0):raise ValueError('Invalid sample weights')
    model=FixedAnchor()
    if mean is None:offset,x=model.design(rows,True)
    else:
        model.mean=np.asarray(mean).copy();model.scale=np.asarray(scale).copy();offset,x=model.design(rows)
    y=np.asarray([r['y'] for r in rows]);w=w/w.sum()
    if len(rows)<50 or set(y)!={0,1}:raise ValueError('Insufficient training labels')
    def objective(coef):
        z=offset+x@coef
        return float(w@(np.logaddexp(0,z)-y*z)+.1*(coef@coef)),x.T@(w*(expit(z)-y))+.2*coef
    result=minimize(objective,np.zeros(4),jac=True,method='L-BFGS-B',bounds=[(-.25,.25)]*4)
    if not result.success:raise ValueError(result.message)
    model.coef=result.x;return model


def repaired_models(train,sampled_days,shared,weighted):
    groups={d:[r for r in train if r['group']==d] for d in sorted({r['group'] for r in train})}
    expected=expected_day_counts(list(groups));x=np.asarray([r['flow_x'] for r in train])
    mean=x.mean(0);scale=np.maximum(x.std(0),1e-6);members=[]
    for days in sampled_days:
        if len(days)!=len(groups) or any(d not in groups for d in days):raise ValueError('Invalid training resample')
        rows=[r for day in days for r in groups[day]]
        weights=[1/expected[r['group']] if weighted else 1 for r in rows]
        members.append(weighted_fit(rows,weights,mean if shared else None,scale if shared else None))
    return members


def run(source,old,anchor,output):
    source=Path(source);old=Path(old);anchor=Path(anchor);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    inputs=[source/'development.jsonl',old/'report.json',anchor/'report.json',
        *[root/f'models-{a}.json' for root in (old,anchor) for a,_ in DATA_PROTOCOL['folds']]]
    frozen=dict(protocol=PROTOCOL,inputs={str(p):digest(p.read_bytes()) for p in inputs})
    path=output/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=frozen:raise ValueError('Frozen inputs changed')
    if not path.exists():save(path,frozen)
    rows=read_rows(source);allrows=[];preds={n:[] for n in ['market',*PROTOCOL['variants']]};folds=[];guards=[]
    for start,end in DATA_PROTOCOL['folds']:
        train,test=split(rows,start,end);allrows.extend(test)
        saved=json.loads((old/f'models-{start}.json').read_text());samples=saved['sampled_days']
        original=StabilityAnchor.load(saved);single=FixedAnchor.load(json.loads((anchor/f'models-{start}.json').read_text())['fixed_anchor'])
        variants=dict(shared_scaler=repaired_models(train,samples,True,False),
            boundary_weighted=repaired_models(train,samples,False,True),both_repairs=repaired_models(train,samples,True,True))
        save(output/f'models-{start}.json',{n:[m.artifact() for m in members] for n,members in variants.items()})
        distributions={n:np.asarray([m.predict(test) for m in members]) for n,members in variants.items()}
        current=dict(market=np.asarray([r['p'] for r in test]),single=single.predict(test),original_ensemble=original.predict(test),
            **{n:d.mean(0) for n,d in distributions.items()})
        for n,p in current.items():preds[n].extend(p.tolist())
        guards.extend(guarded_probabilities(test,distributions['both_repairs']).tolist())
        days=sorted({r['group'] for r in train});counts=Counter(d for sample in samples for d in sample)
        mean=np.asarray([r['flow_x'] for r in train]).mean(0);neutral=[dict(r,flow_x=mean.tolist()) for r in test]
        shifts={n:float(np.mean(np.abs(np.asarray([m.predict(neutral) for m in members])-np.asarray([r['p'] for r in test]))))
                for n,members in dict(original_ensemble=original.models,**variants).items()}
        folds.append(dict(start=start,end=end,training_days=len(days),
            median_unique_days=float(np.median([len(set(s)) for s in samples])),day_counts=dict(counts),
            expected_day_counts=expected_day_counts(days),neutral_mean_absolute_member_shift=shifts,
            models={n:evaluate(test,p) for n,p in current.items()}))
        print('Audited',start,flush=True)
    aggregate={n:evaluate(allrows,np.asarray(p)) for n,p in preds.items()}
    previous=json.loads((old/'report.json').read_text())
    for metric in ('brier','log_loss'):
        if abs(aggregate['original_ensemble']['scores'][metric]-previous['aggregate']['ensemble']['scores'][metric])>1e-12:raise ValueError('Old ensemble did not reproduce')
    control=[dict(r,p=p) for r,p in zip(allrows,preds['single'])]
    aggregate['both_repairs']['paired_vs_single']=paired_interval(control,np.asarray(preds['both_repairs']))
    report=dict(protocol=PROTOCOL,rerun=(output/'report.json').exists(),holdout_loaded=False,deployment_allowed=False,
        folds=folds,aggregate=aggregate,trading={n:paper_scenarios(allrows,np.asarray(p)) for n,p in preds.items() if n!='market'},
        guarded_both=paper_scenarios(allrows,np.asarray(guards)),
        code_hashes={p.name:digest(p.read_bytes()) for p in [Path(__file__),*[Path(__file__).with_name(n) for n in
            ('btc_stability_model.py','btc_anchor_study.py','btc_flow_model.py','btc_flow_followup.py','sizing.py','core.py','live.py')]]})
    save(output/'report.json',report)
    for n,v in aggregate.items():print(n,v['scores']['brier'],v['scores']['log_loss'],v['paper']['pnl'],v['paper']['entries'])
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',default='reports/btc-flow-followup-v1');p.add_argument('--old',default='reports/btc-stability-v1')
    p.add_argument('--anchor',default='reports/btc-anchor-study-v1');p.add_argument('--output',default='reports/btc-stability-repair-v1')
    a=p.parse_args();run(a.source,a.old,a.anchor,a.output)
