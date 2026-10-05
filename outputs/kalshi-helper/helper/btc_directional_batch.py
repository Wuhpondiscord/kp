"""Physical zero and sign-symmetric correction; a fixed second research batch."""
import argparse
import json
from pathlib import Path
import numpy as np
from .btc_context_batch import ContextAnchor
from .btc_refinement import logits
from .btc_anchor_study import FixedAnchor
from .btc_flow_followup import read_rows, split, evaluate, PROTOCOL as DATA_PROTOCOL
from .btc_research import digest, save, paired_interval
from .btc_stability_model import paper_scenarios

PROTOCOL=dict(version='btc-directional-batch-v1',kinds=['flow','price','joint'],penalty=.1,
    design='Physical zero, training RMS scaling without centering, fixed market-logit offset; reversal flips correction',
    flow='5/15-minute signed imbalance and five-minute cross-exchange basis change; no absolute basis',
    price='Distance in sigma; 1/5/15/60-minute normalized returns; distance and return1 times (volatility ratio - 1)',
    selection='Fixed .1 penalty, no inner/outer tuning. All three variants reported',
    controls='Previously frozen fixed anchor and unmodified market',
    limits='Development-only hypothesis test, price/flow sign symmetry is a prior rather than a universal market law; no holdout or deployment')


class DirectionalAnchor(ContextAnchor):
    def __init__(self,kind):
        if kind not in PROTOCOL['kinds']:raise ValueError('Unknown directional kind')
        self.kind=kind;self.penalty=.1
    def design(self,rows,fit=False):
        spot=np.asarray([r['x'] for r in rows],dtype=float);flow=np.asarray([r['flow_x'] for r in rows],dtype=float)
        if spot.shape!=(len(rows),10) or flow.shape!=(len(rows),4) or not np.isfinite(spot).all() or not np.isfinite(flow).all():raise ValueError('Invalid context features')
        f=flow[:,[0,1,3]]
        price=np.column_stack([spot[:,:5],spot[:,0]*(spot[:,5]-1),spot[:,1]*(spot[:,5]-1)])
        raw=f if self.kind=='flow' else price if self.kind=='price' else np.column_stack([price,f])
        if fit:self.mean=np.zeros(raw.shape[1]);self.scale=np.maximum(np.sqrt(np.mean(raw**2,axis=0)),1e-6)
        return logits([r['p'] for r in rows]),np.clip(raw/self.scale,-5,5)
    def artifact(self):return dict(super().artifact(),family='directional_anchor')
    @classmethod
    def load(cls,data):
        if data.get('family')!='directional_anchor':raise ValueError('Wrong model family')
        model=cls(data['kind']);n={'flow':3,'price':7,'joint':10}[model.kind]
        for key in ('mean','scale','coef'):
            v=np.asarray(data[key],dtype=float)
            if v.shape!=(n,) or not np.isfinite(v).all():raise ValueError('Invalid artifact')
            setattr(model,key,v)
        if np.any(model.mean!=0) or np.any(model.scale<=0) or np.any(abs(model.coef)>.25):raise ValueError('Invalid directional parameters')
        return model


def run(source,previous,output):
    source=Path(source);previous=Path(previous);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    frozen=dict(protocol=PROTOCOL,data_sha256=digest((source/'development.jsonl').read_bytes()),
        previous_models={f'models-{a}.json':digest((previous/f'models-{a}.json').read_bytes()) for a,_ in DATA_PROTOCOL['folds']})
    path=output/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=frozen:raise ValueError('Frozen inputs changed')
    if not path.exists():save(path,frozen)
    rows=read_rows(source);allrows=[];folds=[];preds={n:[] for n in ['market','fixed_anchor',*PROTOCOL['kinds']]}
    for start,end in DATA_PROTOCOL['folds']:
        train,test=split(rows,start,end);allrows.extend(test)
        models={n:DirectionalAnchor(n).fit(train) for n in PROTOCOL['kinds']}
        save(output/f'models-{start}.json',{n:m.artifact() for n,m in models.items()})
        baseline=FixedAnchor.load(json.loads((previous/f'models-{start}.json').read_text())['fixed_anchor'])
        current=dict(market=np.asarray([r['p'] for r in test]),fixed_anchor=baseline.predict(test),**{n:m.predict(test) for n,m in models.items()})
        for n,p in current.items():preds[n].extend(p.tolist())
        folds.append(dict(start=start,end=end,models={n:evaluate(test,p) for n,p in current.items()}))
    aggregate={n:evaluate(allrows,np.asarray(p)) for n,p in preds.items()}
    control=[dict(r,p=p) for r,p in zip(allrows,preds['fixed_anchor'])]
    for n in PROTOCOL['kinds']:aggregate[n]['paired_vs_fixed_anchor']=paired_interval(control,np.asarray(preds[n]))
    report=dict(protocol=PROTOCOL,rerun=(output/'report.json').exists(),holdout_loaded=False,deployment_allowed=False,
        folds=folds,aggregate=aggregate,trading={n:paper_scenarios(allrows,np.asarray(p)) for n,p in preds.items() if n!='market'},
        code_hashes={p.name:digest(p.read_bytes()) for p in [Path(__file__),*[Path(__file__).with_name(n) for n in
            ('btc_context_batch.py','btc_anchor_study.py','btc_flow_followup.py','btc_stability_model.py','btc_research.py','sizing.py','core.py','live.py')]]})
    save(output/'report.json',report)
    for n,v in aggregate.items():print(n,v['scores']['brier'],v['scores']['log_loss'],v['paper']['pnl'],v['paper']['entries'])
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',default='reports/btc-flow-followup-v1')
    p.add_argument('--previous',default='reports/btc-anchor-study-v1');p.add_argument('--output',default='reports/btc-directional-batch-v1')
    a=p.parse_args();run(a.source,a.previous,a.output)
