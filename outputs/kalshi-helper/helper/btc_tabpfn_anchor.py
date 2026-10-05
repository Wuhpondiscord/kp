"""One fixed output-calibration repair using cached, chronological TabPFN predictions."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit
from .btc_refinement import logits
from .btc_flow_followup import read_rows, split, evaluate, PROTOCOL as DATA_PROTOCOL
from .btc_context_batch import inner_split
from .btc_research import save,digest,paired_interval
from .btc_stability_model import paper_scenarios

PROTOCOL=dict(version='btc-tabpfn-anchor-v1',weight_bounds=[0,.25],l2=.1,disagreement_clip=4,
    design='Fixed market log odds plus nonnegative scalar times clipped TabPFN-minus-market log odds',
    selection='Fit scalar on cached earlier inner validation predictions/labels only; fixed bound and L2, no parameter search',
    purpose='Test overlarge pretrained corrections without changing checkpoint, features or execution costs',
    limits='Post-diagnostic exploratory repair; reused development dates; no holdout or deployment')


def predict(market,signal,weight):
    if not 0<=weight<=.25:raise ValueError('Invalid correction weight')
    m=logits(market);s=logits(signal)
    if m.shape!=s.shape:raise ValueError('Mismatched probabilities')
    return expit(m+weight*np.clip(s-m,-4,4))


def fit_weight(market,signal,y):
    y=np.asarray(y,dtype=float)
    if y.shape!=np.asarray(market).shape or not set(y)<={0,1} or not len(y):raise ValueError('Invalid labels')
    def loss(w):
        p=np.clip(predict(market,signal,w),1e-6,1-1e-6)
        return float(np.mean(-y*np.log(p)-(1-y)*np.log1p(-p))+.1*w*w)
    result=minimize_scalar(loss,bounds=(0,.25),method='bounded',options={'xatol':1e-8})
    if not result.success:raise ValueError('Correction fit failed')
    return min([0.,float(result.x),.25],key=lambda w:(loss(w),w))


def run(source,previous,output):
    source=Path(source);previous=Path(previous);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    files=[previous/f'predictions-{a}.json' for a,_ in DATA_PROTOCOL['folds']]
    frozen=dict(protocol=PROTOCOL,data_hash=digest((source/'development.jsonl').read_bytes()),
        predictions={p.name:digest(p.read_bytes()) for p in files})
    path=output/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=frozen:raise ValueError('Frozen inputs changed')
    if not path.exists():save(path,frozen)
    rows=read_rows(source);allrows=[];allp=[];control=[];folds=[]
    for start,end in DATA_PROTOCOL['folds']:
        train,test=split(rows,start,end);_,validation,_=inner_split(train);saved=json.loads((previous/f'predictions-{start}.json').read_text())
        if [r['ticker'] for r in test]!=saved['tickers'] or [r['ticker'] for r in validation]!=saved['inner_tickers']:raise ValueError('Prediction alignment mismatch')
        weight=fit_weight([r['p'] for r in validation],saved['inner_predictions'],[r['y'] for r in validation])
        save(output/f'weight-{start}.json',dict(weight=weight))
        p=predict([r['p'] for r in test],saved['predictions']['raw_tabpfn'],weight)
        allrows.extend(test);allp.extend(p.tolist());control.extend(saved['predictions']['fixed_anchor'])
        folds.append(dict(start=start,end=end,weight=weight,result=evaluate(test,p)))
    result=evaluate(allrows,np.asarray(allp));result['paired_vs_fixed_anchor']=paired_interval([dict(r,p=p) for r,p in zip(allrows,control)],np.asarray(allp))
    report=dict(protocol=PROTOCOL,rerun=(output/'report.json').exists(),holdout_loaded=False,deployment_allowed=False,
        folds=folds,aggregate={'anchored_tabpfn':result},trading=paper_scenarios(allrows,np.asarray(allp)),code_sha256=digest(Path(__file__).read_bytes()))
    save(output/'report.json',report)
    print('weights',[f['weight'] for f in folds]);print(result['scores']['brier'],result['scores']['log_loss'],result['paper']['pnl'],result['paper']['entries'])
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',default='reports/btc-flow-followup-v1');p.add_argument('--previous',default='reports/btc-tabpfn-v1')
    p.add_argument('--output',default='reports/btc-tabpfn-anchor-v1');a=p.parse_args();run(a.source,a.previous,a.output)
