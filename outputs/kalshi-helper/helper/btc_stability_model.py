"""Training-day block ensemble; dispersion is sensitivity, not a confidence bound."""
import argparse
import json
from pathlib import Path
import numpy as np
from .btc_anchor_study import FixedAnchor, delayed_rows
from .btc_flow_followup import read_rows, split, evaluate, PROTOCOL as DATA_PROTOCOL
from .btc_flow_model import trade_uncertainty
from .btc_research import digest, save, paired_interval
from .sizing import simulate

PROTOCOL=dict(version='btc-stability-v1',members=32,block_days=2,seed=1729,
    candidate='Average probabilities from fixed-anchor models fitted to moving two-day training blocks',
    fitting='Same L2 .1 and coefficient bounds +/-.25; each member fits its own training-only scaler',
    resampling='Sample nonwrapping contiguous day blocks with replacement until original day count, truncate last block',
    guard='5th/95th percentile across fitted models, clamped toward market to prevent side reversal',
    interpretation='Model dispersion is training-sample sensitivity, NOT a calibrated confidence interval',
    primary='Same 1000 bankroll, 1% sizing, .04 edge, .02 slippage and .07 fees',
    diagnostic_slippage=[0,.005,.01,.02,.05],delay_seconds=[0,60],
    restrictions='Previously inspected development data; no threshold/weight search, holdout access or automatic deployment')


def sample_days(rows,rng,block_days=2):
    days=sorted({r['group'] for r in rows})
    if len(days)<block_days:raise ValueError('Insufficient training days')
    groups={d:[r for r in rows if r['group']==d] for d in days};chosen=[]
    # Do not silently treat nonadjacent calendar dates as adjacent blocks.
    from datetime import date
    starts=[i for i in range(len(days)-block_days+1)
            if (date.fromisoformat(days[i+block_days-1])-date.fromisoformat(days[i])).days==block_days-1]
    if not starts:raise ValueError('No contiguous day blocks')
    while len(chosen)<len(days):
        start=int(rng.choice(starts));chosen.extend(days[start:start+block_days])
    chosen=chosen[:len(days)]
    return [r for d in chosen for r in groups[d]],chosen


class StabilityAnchor:
    def fit(self,rows):
        rng=np.random.default_rng(PROTOCOL['seed']);self.models=[];self.samples=[]
        for _ in range(PROTOCOL['members']):
            sampled,days=sample_days(rows,rng,PROTOCOL['block_days'])
            # Fail visibly on invalid resamples instead of retrying until favorable.
            self.models.append(FixedAnchor().fit(sampled));self.samples.append(days)
        return self
    def distribution(self,rows):return np.array([m.predict(rows) for m in self.models])
    def predict(self,rows):return self.distribution(rows).mean(axis=0)
    def artifact(self):return dict(kind='stability_anchor',models=[m.artifact() for m in self.models],sampled_days=self.samples)
    @classmethod
    def load(cls,data):
        if data.get('kind')!='stability_anchor' or len(data['models'])!=PROTOCOL['members']:raise ValueError('Invalid ensemble artifact')
        obj=cls();obj.models=[FixedAnchor.load(m) for m in data['models']];obj.samples=data['sampled_days'];return obj


def guarded_probabilities(rows,distribution):
    """Conservative decision inputs only; these are not forecast probabilities."""
    d=np.asarray(distribution,dtype=float)
    if d.ndim!=2 or d.shape[1]!=len(rows) or d.shape[0]<2 or not np.isfinite(d).all() or np.any((d<0)|(d>1)):
        raise ValueError('Invalid member probabilities')
    market=np.asarray([r['p'] for r in rows]);mean=d.mean(axis=0)
    lower,upper=np.quantile(d,[.05,.95],axis=0)
    return np.where(mean>=market,np.maximum(market,np.minimum(mean,lower)),np.minimum(market,np.maximum(mean,upper)))


def paper_scenarios(rows,p):
    result={}
    for slip in PROTOCOL['diagnostic_slippage']:
        paper=simulate(rows,p,policy='bankroll_1_percent',slippage=slip,bankroll=1000,min_edge=.04)
        paper['uncertainty']=trade_uncertainty(paper,{r['group'] for r in rows});result[str(slip)]=paper
    return result


def run(source,previous,output):
    source=Path(source);previous=Path(previous);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    frozen=dict(protocol=PROTOCOL,source_hash=digest((source/'development.jsonl').read_bytes()),
        previous_report_hash=digest((previous/'report.json').read_bytes()),
        previous_models={f'models-{a}.json':digest((previous/f'models-{a}.json').read_bytes()) for a,_ in DATA_PROTOCOL['folds']},
        age_hash=digest((previous/'age-features.json').read_bytes()))
    path=output/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=frozen:raise ValueError('Frozen inputs changed')
    if not path.exists():save(path,frozen)
    rows=read_rows(source);features=json.loads((previous/'age-features.json').read_text());allrows=[];folds=[]
    original=json.loads((previous/'report.json').read_text())
    allpred={n:[] for n in ('market','fixed_anchor','ensemble','ensemble_60s')};allguard=[];details=[]
    for start,end in DATA_PROTOCOL['folds']:
        train,test=split(rows,start,end);allrows.extend(test)
        ensemble=StabilityAnchor().fit(train)
        save(output/f'models-{start}.json',ensemble.artifact())
        baseline=FixedAnchor.load(json.loads((previous/f'models-{start}.json').read_text())['fixed_anchor'])
        distribution=ensemble.distribution(test);p=distribution.mean(axis=0)
        guard=guarded_probabilities(test,distribution);allguard.extend(guard.tolist())
        predictions=dict(market=np.asarray([r['p'] for r in test]),fixed_anchor=baseline.predict(test),
            ensemble=p,ensemble_60s=ensemble.predict(delayed_rows(test,features,60)))
        scores={n:evaluate(test,v) for n,v in predictions.items()}
        for n,v in predictions.items():allpred[n].extend(v.tolist())
        widths=np.quantile(distribution,.95,axis=0)-np.quantile(distribution,.05,axis=0)
        for r,mean,width,g in zip(test,p,widths,guard):
            details.append(dict(ticker=r['ticker'],at=r['at'],market=r['p'],mean=float(mean),
                member_90_percent_span=float(width),guarded_decision_input=float(g)))
        folds.append(dict(start=start,end=end,training_rows=len(train),evaluation_rows=len(test),models=scores,
            median_member_span=float(np.median(widths))))
        print('Trained and evaluated',start,flush=True)
    aggregate={n:evaluate(allrows,np.asarray(p)) for n,p in allpred.items()}
    for metric in ('brier','log_loss'):
        if abs(aggregate['fixed_anchor']['scores'][metric]-original['aggregate']['fixed_anchor']['scores'][metric])>1e-12:
            raise ValueError('Baseline reproduction mismatch')
    controlrows=[dict(r,p=p) for r,p in zip(allrows,allpred['fixed_anchor'])]
    aggregate['ensemble']['paired_vs_fixed_anchor']=paired_interval(controlrows,np.asarray(allpred['ensemble']))
    report=dict(protocol=PROTOCOL,rerun=(output/'report.json').exists(),holdout_loaded=False,deployment_allowed=False,
        evaluation_rows=len(allrows),folds=folds,aggregate=aggregate,
        trading=dict(single=paper_scenarios(allrows,np.asarray(allpred['fixed_anchor'])),
            ensemble=paper_scenarios(allrows,np.asarray(allpred['ensemble'])),
            guarded=paper_scenarios(allrows,np.asarray(allguard))),
        sensitivity=details,code_hashes={p.name:digest(p.read_bytes()) for p in [Path(__file__),*[Path(__file__).with_name(n) for n in
            ('btc_anchor_study.py','btc_flow_followup.py','btc_flow_model.py','btc_research.py','sizing.py','core.py','live.py')]]})
    save(output/'report.json',report)
    for n,v in aggregate.items():print(n,v['scores']['brier'],v['scores']['log_loss'])
    for n,v in report['trading'].items():print(n,{s:(p['pnl'],p['entries']) for s,p in v.items()})
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',default='reports/btc-flow-followup-v1')
    p.add_argument('--previous',default='reports/btc-anchor-study-v1');p.add_argument('--output',default='reports/btc-stability-v1')
    a=p.parse_args();run(a.source,a.previous,a.output)
