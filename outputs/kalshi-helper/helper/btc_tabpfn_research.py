"""Pinned local TabPFN-v2 benchmark; no hosted inference or checkpoint fine-tuning."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import time
import numpy as np
from scipy.optimize import minimize_scalar
from .btc_tree_research import matrix, inventory
from .btc_context_batch import inner_split
from .btc_flow_followup import read_rows, split, evaluate, PROTOCOL as DATA_PROTOCOL
from .btc_anchor_study import FixedAnchor
from .btc_research import digest, save, metrics, paired_interval
from .btc_stability_model import paper_scenarios

CHECKPOINT_SHA='cf8c519c01eaf1613ee91239006d57b1c806ff5f23ac1aeb1315ba1015210e49'
PROTOCOL=dict(version='btc-tabpfn-v1',package='tabpfn==2.2.1',checkpoint_sha256=CHECKPOINT_SHA,
    checkpoint_revision='f851f2a3c941544733b712d8c0f96dfae9b28862',n_estimators=4,random_state=1729,
    device='cpu',torch_threads=4,inputs='Market probability plus 10 spot and four flow features; no labels as inputs',
    adaptation='Frozen pretrained weights; fit supplies training context, not gradient fine-tuning',
    variants=['raw_tabpfn','inner_market_blend'],
    blend='Convex probability blend weight fitted by inner log loss on earlier training dates; set to zero if inner Brier is worse than market',
    split='Existing chronological outer folds and embargoed inner training split; no outer selection',
    limits='Reused development dates; no fine-tuning, holdout access, remote inference or automatic promotion')


def select_weight(market,prediction,labels):
    m=np.asarray(market,dtype=float);p=np.asarray(prediction,dtype=float);y=np.asarray(labels,dtype=float)
    if m.shape!=p.shape or m.shape!=y.shape or m.ndim!=1 or not len(m) or not np.isfinite([m,p,y]).all() or np.any((m<0)|(m>1)|(p<0)|(p>1)) or not set(y)<= {0,1}:
        raise ValueError('Invalid blend inputs')
    def loss(a):
        v=np.clip((1-a)*m+a*p,1e-6,1-1e-6)
        return float(np.mean(-y*np.log(v)-(1-y)*np.log1p(-v)))
    result=minimize_scalar(loss,bounds=(0,1),method='bounded',options={'xatol':1e-7})
    if not result.success:raise ValueError('Blend optimization failed')
    a=min([0.,float(result.x),1.],key=lambda w:(loss(w),w))
    if np.mean(((1-a)*m+a*p-y)**2)>np.mean((m-y)**2):a=0.
    return a


def local_model(checkpoint):
    os.environ['TABPFN_DISABLE_TELEMETRY']='1';os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
    os.environ['TABPFN_STATE_DIR']=str(Path(checkpoint).resolve().parent/'local-state')
    if importlib.metadata.version('tabpfn')!='2.2.1':raise ValueError('Requires pinned TabPFN 2.2.1 research environment')
    from tabpfn import TabPFNClassifier
    return TabPFNClassifier(model_path=str(checkpoint),device='cpu',n_estimators=4,random_state=1729,n_jobs=1)


def infer(train,test,checkpoint):
    model=local_model(checkpoint);start=time.perf_counter()
    model.fit(matrix(train),np.asarray([r['y'] for r in train]))
    p=model.predict_proba(matrix(test))[:,list(model.classes_).index(1)]
    if p.shape!=(len(test),) or not np.isfinite(p).all() or np.any((p<0)|(p>1)):raise ValueError('Invalid pretrained predictions')
    return p,time.perf_counter()-start


def run(source,previous,checkpoint,output):
    source=Path(source);previous=Path(previous);checkpoint=Path(checkpoint);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if digest(checkpoint.read_bytes())!=CHECKPOINT_SHA:raise ValueError('Checkpoint hash mismatch')
    frozen=dict(protocol=PROTOCOL,data_hash=digest((source/'development.jsonl').read_bytes()),
        controls={f'models-{a}.json':digest((previous/f'models-{a}.json').read_bytes()) for a,_ in DATA_PROTOCOL['folds']})
    path=output/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=frozen:raise ValueError('Frozen experiment changed')
    if not path.exists():save(path,frozen)
    import torch
    torch.set_num_threads(4)
    rows=read_rows(source);allrows=[];folds=[];preds={n:[] for n in ['market','fixed_anchor','raw_tabpfn','inner_market_blend']}
    for start,end in DATA_PROTOCOL['folds']:
        train,test=split(rows,start,end);allrows.extend(test);fit,validation,boundary=inner_split(train)
        print('Starting pretrained fold',start,'inner',len(fit),len(validation),'outer',len(train),len(test),flush=True)
        inner,inner_time=infer(fit,validation,checkpoint)
        weight=select_weight([r['p'] for r in validation],inner,[r['y'] for r in validation])
        # Freeze inner choice before outer outcomes are scored.
        config=dict(weight=weight,boundary=boundary,training_tickers=[r['ticker'] for r in train],
            inner_fit_tickers=[r['ticker'] for r in fit],inner_validation_tickers=[r['ticker'] for r in validation])
        save(output/f'context-{start}.json',config)
        raw,outer_time=infer(train,test,checkpoint)
        market=np.asarray([r['p'] for r in test]);control=FixedAnchor.load(json.loads((previous/f'models-{start}.json').read_text())['fixed_anchor'])
        current=dict(market=market,fixed_anchor=control.predict(test),raw_tabpfn=raw,inner_market_blend=(1-weight)*market+weight*raw)
        for n,p in current.items():preds[n].extend(p.tolist())
        save(output/f'predictions-{start}.json',dict(tickers=[r['ticker'] for r in test],predictions={n:p.tolist() for n,p in current.items()},
            inner_predictions=inner.tolist(),inner_tickers=[r['ticker'] for r in validation]))
        folds.append(dict(start=start,end=end,weight=weight,inner_seconds=inner_time,outer_seconds=outer_time,
            inner_market=metrics(validation,np.asarray([r['p'] for r in validation])),inner_tabpfn=metrics(validation,inner),
            models={n:evaluate(test,p) for n,p in current.items()}))
        print('Completed',start,'blend',weight,'seconds',round(inner_time+outer_time,2),flush=True)
    repeat,repeat_time=infer(train,test,checkpoint)
    difference=float(np.max(np.abs(repeat-raw)))
    if difference>1e-6:raise ValueError('Pretrained last-fold refit not reproducible within 1e-6')
    aggregate={n:evaluate(allrows,np.asarray(p)) for n,p in preds.items()}
    controlrows=[dict(r,p=p) for r,p in zip(allrows,preds['fixed_anchor'])]
    for n in ('raw_tabpfn','inner_market_blend'):aggregate[n]['paired_vs_fixed_anchor']=paired_interval(controlrows,np.asarray(preds[n]))
    report=dict(protocol=PROTOCOL,rerun=(output/'report.json').exists(),holdout_loaded=False,deployment_allowed=False,
        folds=folds,aggregate=aggregate,trading={n:paper_scenarios(allrows,np.asarray(p)) for n,p in preds.items() if n!='market'},
        reproduction=dict(last_fold_max_abs_difference=difference,seconds=repeat_time),
        environment={name:importlib.metadata.version(name) for name in ['tabpfn','torch','numpy','scikit-learn','scipy','pandas','einops','huggingface-hub','tabpfn-common-utils']},
        code_hashes={p.name:digest(p.read_bytes()) for p in [Path(__file__),*[Path(__file__).with_name(n) for n in
            ('btc_tree_research.py','btc_context_batch.py','btc_flow_followup.py','btc_anchor_study.py','btc_research.py','btc_stability_model.py','sizing.py','core.py','live.py')]]})
    save(output/'report.json',report);inventory('reports',output/'experiment-ledger.json')
    for n,v in aggregate.items():print(n,v['scores']['brier'],v['scores']['log_loss'],v['paper']['pnl'],v['paper']['entries'])
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',default='reports/btc-flow-followup-v1');p.add_argument('--previous',default='reports/btc-anchor-study-v1')
    p.add_argument('--checkpoint',required=True);p.add_argument('--output',default='reports/btc-tabpfn-v1')
    a=p.parse_args();run(a.source,a.previous,a.checkpoint,a.output)
