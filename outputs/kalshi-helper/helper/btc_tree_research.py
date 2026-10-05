"""Public-API gradient boosting from market probabilities; chronological selection."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier
from .btc_refinement import logits
from .btc_context_batch import inner_split
from .btc_flow_followup import read_rows, split, evaluate, PROTOCOL as DATA_PROTOCOL
from .btc_anchor_study import FixedAnchor
from .btc_research import save, digest, metrics, paired_interval
from .btc_stability_model import paper_scenarios

PROTOCOL=dict(version='btc-trees-v1',stages=[0,8,32],learning_rate=.05,max_depth=2,min_samples_leaf=30,
    initial='Market probability via sklearn public init estimator; no in-place tree value updates',
    inputs='Market probability, 10 timestamped spot features, four flow/basis features',
    selection='Inner chronological log loss, requiring Brier no worse than raw market; market is explicit stage-zero fallback',
    comparisons=['market','fixed_anchor','fixed_32','selected'],
    limits='All outer dates previously inspected; limited new-family search only, not global multiplicity correction; no holdout or deployment')


class MarketInit(ClassifierMixin,BaseEstimator):
    def fit(self,x,y,sample_weight=None):
        self.classes_=np.array([0,1]);self.n_features_in_=x.shape[1];return self
    def predict_proba(self,x):
        p=np.clip(np.asarray(x)[:,0],1e-6,1-1e-6)
        return np.column_stack([1-p,p])


def matrix(rows):
    x=np.asarray([[r['p'],*r['x'],*r['flow_x']] for r in rows],dtype=float)
    if x.shape!=(len(rows),15) or not np.isfinite(x).all() or np.any((x[:,0]<0)|(x[:,0]>1)):
        raise ValueError('Invalid tree inputs')
    return x


class MarketTrees:
    def __init__(self,stages=32):
        if stages not in PROTOCOL['stages']:raise ValueError('Unregistered stage count')
        self.stages=stages;self.trees=[]
    def fit(self,rows):
        x=matrix(rows);y=np.asarray([r['y'] for r in rows])
        if len(rows)<50 or set(y)!={0,1}:raise ValueError('Insufficient training labels')
        self.trees=[]
        if not self.stages:return self
        model=GradientBoostingClassifier(init=MarketInit(),n_estimators=self.stages,learning_rate=.05,
            max_depth=2,min_samples_leaf=30,random_state=1729).fit(x,y)
        for estimator in model.estimators_[:,0]:
            t=estimator.tree_
            self.trees.append(dict(left=t.children_left.tolist(),right=t.children_right.tolist(),feature=t.feature.tolist(),
                threshold=t.threshold.tolist(),value=t.value.ravel().tolist()))
        # JSON inference must preserve public sklearn predictions to float32-input precision.
        if not np.allclose(self.predict(rows),model.predict_proba(x)[:,1],rtol=0,atol=2e-7):raise ValueError('Tree serialization mismatch')
        return self
    def predict(self,rows):
        raw=matrix(rows);x=raw.astype(np.float32);z=logits(raw[:,0])
        for tree in self.trees:
            for i,row in enumerate(x):
                node=0
                while tree['left'][node]!=-1:
                    node=tree['left'][node] if row[tree['feature'][node]]<=tree['threshold'][node] else tree['right'][node]
                z[i]+=.05*tree['value'][node]
        return np.clip(expit(z),1e-6,1-1e-6)
    def artifact(self):return dict(kind='market_trees',stages=self.stages,trees=self.trees)
    @classmethod
    def load(cls,data):
        if data.get('kind')!='market_trees':raise ValueError('Wrong artifact type')
        model=cls(data['stages']);model.trees=data['trees']
        if len(model.trees)!=model.stages:raise ValueError('Tree count mismatch')
        for t in model.trees:
            n=len(t['left'])
            if not n or any(len(t[k])!=n for k in ('right','feature','threshold','value')):raise ValueError('Malformed tree')
            if not np.isfinite(t['threshold']).all() or not np.isfinite(t['value']).all():raise ValueError('Nonfinite tree')
            for i in range(n):
                a,b=t['left'][i],t['right'][i]
                if a==-1 and b==-1:continue
                if not (i<a<n and i<b<n and 0<=t['feature'][i]<15):raise ValueError('Invalid or cyclic tree')
        return model


def choose(train):
    fit,validation,boundary=inner_split(train);scores=[]
    for stages in PROTOCOL['stages']:
        model=MarketTrees(stages).fit(fit)
        scores.append(dict(stages=stages,scores=metrics(validation,model.predict(validation))))
    market=scores[0]['scores']
    eligible=[r for r in scores if r['scores']['brier']<=market['brier']]
    selected=min(eligible,key=lambda r:(r['scores']['log_loss'],r['stages']))['stages']
    return selected,dict(boundary=boundary,fit_rows=len(fit),validation_rows=len(validation),candidates=scores,selected=selected)


def run(source,previous,output):
    source=Path(source);previous=Path(previous);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    frozen=dict(protocol=PROTOCOL,data_hash=digest((source/'development.jsonl').read_bytes()),
        controls={f'models-{a}.json':digest((previous/f'models-{a}.json').read_bytes()) for a,_ in DATA_PROTOCOL['folds']})
    path=output/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=frozen:raise ValueError('Protocol changed')
    if not path.exists():save(path,frozen)
    rows=read_rows(source);allrows=[];folds=[];preds={n:[] for n in PROTOCOL['comparisons']}
    for start,end in DATA_PROTOCOL['folds']:
        train,test=split(rows,start,end);allrows.extend(test);stages,selection=choose(train)
        full=MarketTrees(32).fit(train);selected=full if stages==32 else MarketTrees(stages).fit(train)
        save(output/f'models-{start}.json',dict(selection=selection,fixed_32=full.artifact(),selected=selected.artifact()))
        control=FixedAnchor.load(json.loads((previous/f'models-{start}.json').read_text())['fixed_anchor'])
        current=dict(market=np.asarray([r['p'] for r in test]),fixed_anchor=control.predict(test),fixed_32=full.predict(test),selected=selected.predict(test))
        for n,p in current.items():preds[n].extend(p.tolist())
        folds.append(dict(start=start,end=end,selection=selection,models={n:evaluate(test,p) for n,p in current.items()}))
        print('Selected stages',start,stages,flush=True)
    aggregate={n:evaluate(allrows,np.asarray(p)) for n,p in preds.items()}
    for n in ('fixed_32','selected'):
        controlrows=[dict(r,p=p) for r,p in zip(allrows,preds['fixed_anchor'])]
        aggregate[n]['paired_vs_fixed_anchor']=paired_interval(controlrows,np.asarray(preds[n]))
    report=dict(protocol=PROTOCOL,rerun=(output/'report.json').exists(),holdout_loaded=False,deployment_allowed=False,
        folds=folds,aggregate=aggregate,trading={n:paper_scenarios(allrows,np.asarray(p)) for n,p in preds.items() if n!='market'},
        code_hashes={p.name:digest(p.read_bytes()) for p in [Path(__file__),*[Path(__file__).with_name(n) for n in
            ('btc_flow_followup.py','btc_context_batch.py','btc_anchor_study.py','btc_research.py','btc_stability_model.py','sizing.py','core.py','live.py')]]})
    save(output/'report.json',report)
    for n,v in aggregate.items():print(n,v['scores']['brier'],v['scores']['log_loss'],v['paper']['pnl'],v['paper']['entries'])
    return report


def inventory(reports,output):
    """Audit ledger only: incomparable cohorts must not become one leaderboard."""
    items=[]
    for path in sorted(Path(reports).glob('btc*/report.json')):
        report=json.loads(path.read_text());models=report.get('aggregate',{})
        if not isinstance(models,dict) or not models:continue
        entries={n:dict(brier=v['scores']['brier'],log_loss=v['scores']['log_loss'],
            events=v['scores'].get('events'),days=v['scores'].get('days'),primary_pnl=v.get('paper',{}).get('pnl'))
            for n,v in models.items() if isinstance(v,dict) and 'scores' in v}
        items.append(dict(report=path.as_posix(),sha256=digest(path.read_bytes()),models=entries,
            holdout_loaded=report.get('holdout_loaded','unreported'),deployment_allowed=report.get('deployment_allowed','unreported')))
    save(Path(output),dict(experiments=items,scope='Reports exposing aggregate scores; not a complete count of every historical fit',
        warning='Do not rank different cohorts together. Controls repeat across reports. This inventory is not a multiplicity correction or probability-of-overfitting estimate.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',default='reports/btc-flow-followup-v1')
    p.add_argument('--previous',default='reports/btc-anchor-study-v1');p.add_argument('--output',default='reports/btc-trees-v1')
    a=p.parse_args();run(a.source,a.previous,a.output);inventory('reports',Path(a.output)/'experiment-ledger.json')
