"""Nested training-only regularization for market-relative price/flow context."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from .btc_anchor_study import FixedAnchor
from .btc_refinement import logits
from .btc_flow_followup import read_rows, split, evaluate, PROTOCOL as DATA_PROTOCOL
from .btc_research import digest, save, metrics, paired_interval, unix
from .btc_stability_model import paper_scenarios
from .core import stamp

PROTOCOL=dict(version='btc-context-batch-v1',kinds=['spot','combined','interactions'],penalties=[.1,1.,10.],
    inner='Last max(2, floor(training days/4)) days; two-hour embargo; earlier settled labels only',
    selection='Lowest inner log loss; ties prefer larger penalty; no outer labels or P&L selection',
    architecture='Fixed market logit offset, standardized external features, no intercept or learned market slope, coefficient bounds +/-.25',
    features='10 existing timestamped Coinbase price/volatility/volume/time features; four Binance flow/basis inputs; interactions use standardized distance*volatility, return1*flow5, return5*flow15',
    trading='Same bankroll 1000, 1% risk, .04 net edge, .02 primary slippage, .07 fees',
    restrictions='Previously inspected development dates; all candidates reported; no holdout access or automatic deployment')


class ContextAnchor:
    def __init__(self,kind,penalty=.1):
        if kind not in PROTOCOL['kinds'] or penalty not in PROTOCOL['penalties']:raise ValueError('Unregistered model')
        self.kind=kind;self.penalty=penalty
    def design(self,rows,fit=False):
        spot=np.asarray([r['x'] for r in rows],dtype=float)
        if spot.shape!=(len(rows),10) or not np.isfinite(spot).all():raise ValueError('Invalid spot features')
        raw=spot
        if self.kind!='spot':
            flow=np.asarray([r['flow_x'] for r in rows],dtype=float)
            if flow.shape!=(len(rows),4) or not np.isfinite(flow).all():raise ValueError('Invalid flow features')
            raw=np.column_stack([spot,flow])
        if fit:self.mean=raw.mean(0);self.scale=np.maximum(raw.std(0),1e-6)
        z=np.clip((raw-self.mean)/self.scale,-5,5)
        if self.kind=='interactions':z=np.column_stack([z,np.clip(z[:,0]*z[:,5],-5,5),np.clip(z[:,1]*z[:,10],-5,5),np.clip(z[:,2]*z[:,11],-5,5)])
        return logits([r['p'] for r in rows]),z
    def fit(self,rows):
        offset,x=self.design(rows,True);y=np.asarray([r['y'] for r in rows])
        if len(rows)<50 or set(y)!={0,1}:raise ValueError('Insufficient training outcomes')
        def objective(w):
            z=offset+x@w
            return float(np.mean(np.logaddexp(0,z)-y*z)+self.penalty*(w@w)),x.T@(expit(z)-y)/len(y)+2*self.penalty*w
        r=minimize(objective,np.zeros(x.shape[1]),jac=True,method='L-BFGS-B',bounds=[(-.25,.25)]*x.shape[1])
        if not r.success:raise ValueError(r.message)
        self.coef=r.x;return self
    def predict(self,rows):
        offset,x=self.design(rows);return np.clip(expit(offset+x@self.coef),1e-6,1-1e-6)
    def artifact(self):return dict(kind=self.kind,penalty=self.penalty,mean=self.mean.tolist(),scale=self.scale.tolist(),coef=self.coef.tolist())
    @classmethod
    def load(cls,data):
        model=cls(data['kind'],data['penalty']);n=10 if model.kind=='spot' else 14
        for key,length in [('mean',n),('scale',n),('coef',n+(3 if model.kind=='interactions' else 0))]:
            v=np.asarray(data[key],dtype=float)
            if v.shape!=(length,) or not np.isfinite(v).all():raise ValueError('Invalid artifact')
            setattr(model,key,v)
        if np.any(model.scale<=0) or np.any(abs(model.coef)>.25):raise ValueError('Invalid model parameters')
        return model


def inner_split(train):
    days=sorted({r['group'] for r in train});boundary=days[-max(2,len(days)//4)];a=unix(boundary)
    fit=[r for r in train if stamp(r['settled_at']).timestamp()<a]
    validation=[r for r in train if stamp(r['at']).timestamp()>=a+7200]
    if len(fit)<50 or len(validation)<30:raise ValueError('Insufficient nested chronological split')
    return fit,validation,boundary


def select_and_fit(train,kind):
    fit,validation,boundary=inner_split(train);scores=[]
    for penalty in PROTOCOL['penalties']:
        model=ContextAnchor(kind,penalty).fit(fit)
        scores.append(dict(penalty=penalty,scores=metrics(validation,model.predict(validation))))
    chosen=min(scores,key=lambda r:(r['scores']['log_loss'],-r['penalty']))['penalty']
    return ContextAnchor(kind,chosen).fit(train),dict(boundary=boundary,fit_rows=len(fit),validation_rows=len(validation),candidates=scores,chosen=chosen)


def run(source,previous,output):
    source=Path(source);previous=Path(previous);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    frozen=dict(protocol=PROTOCOL,data_hash=digest((source/'development.jsonl').read_bytes()),
        previous_models={f'models-{a}.json':digest((previous/f'models-{a}.json').read_bytes()) for a,_ in DATA_PROTOCOL['folds']})
    path=output/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=frozen:raise ValueError('Frozen inputs changed')
    if not path.exists():save(path,frozen)
    rows=read_rows(source);allrows=[];folds=[];preds={n:[] for n in ['market','fixed_anchor',*PROTOCOL['kinds']]}
    for start,end in DATA_PROTOCOL['folds']:
        train,test=split(rows,start,end);allrows.extend(test);models={};selection={}
        for kind in PROTOCOL['kinds']:models[kind],selection[kind]=select_and_fit(train,kind)
        save(output/f'models-{start}.json',dict(selection=selection,models={n:m.artifact() for n,m in models.items()}))
        baseline=FixedAnchor.load(json.loads((previous/f'models-{start}.json').read_text())['fixed_anchor'])
        current=dict(market=np.asarray([r['p'] for r in test]),fixed_anchor=baseline.predict(test),**{n:m.predict(test) for n,m in models.items()})
        for n,p in current.items():preds[n].extend(p.tolist())
        folds.append(dict(start=start,end=end,selection=selection,models={n:evaluate(test,p) for n,p in current.items()}))
        print('Evaluated',start,{n:s['chosen'] for n,s in selection.items()},flush=True)
    aggregate={n:evaluate(allrows,np.asarray(p)) for n,p in preds.items()}
    for n in PROTOCOL['kinds']:
        controls=[dict(r,p=p) for r,p in zip(allrows,preds['fixed_anchor'])]
        aggregate[n]['paired_vs_fixed_anchor']=paired_interval(controls,np.asarray(preds[n]))
    report=dict(protocol=PROTOCOL,rerun=(output/'report.json').exists(),holdout_loaded=False,deployment_allowed=False,
        evaluation_rows=len(allrows),folds=folds,aggregate=aggregate,
        trading={n:paper_scenarios(allrows,np.asarray(p)) for n,p in preds.items() if n!='market'},
        code_hashes={p.name:digest(p.read_bytes()) for p in [Path(__file__),*[Path(__file__).with_name(n) for n in
            ('btc_flow_followup.py','btc_anchor_study.py','btc_stability_model.py','btc_research.py','btc_inputs.py','sizing.py','core.py','live.py')]]})
    save(output/'report.json',report)
    for n,v in aggregate.items():print(n,v['scores']['brier'],v['scores']['log_loss'],v['paper']['pnl'],v['paper']['entries'])
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',default='reports/btc-flow-followup-v1')
    p.add_argument('--previous',default='reports/btc-anchor-study-v1');p.add_argument('--output',default='reports/btc-context-batch-v1')
    a=p.parse_args();run(a.source,a.previous,a.output)
