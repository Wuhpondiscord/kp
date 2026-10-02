"""Observation-conditioned market-offset boosting, explicitly experimental."""
from copy import deepcopy
from datetime import timedelta
import hashlib
import json
import math
import pickle
from pathlib import Path
import numpy as np
from scipy.special import expit,logit
from sklearn.tree import DecisionTreeRegressor
from threadpoolctl import threadpool_limits
from .core import utcnow,stamp
from .residual import date_weights
from .evaluation import features,score,paired_block_interval,quote_scenario
from .training import split_dataset,partition_summary,segment_scores
from .weather_experiment import collect_features
from .station_observations import SITES,fetch_observations,ObservationIndex
from .weather_model import event_date
from .research_store import digest,ResearchStore

PROTOCOL=dict(version='station-offset-v1',candidate_leaves=[4,7],iterations=120,learning_rate=.05,
    minimum_leaf=100,leaf_regularization=10,weights=[0,.25,.5,.75,1],observation_delay_minutes=30,
    sensitivity_delay_minutes=90,selection='Validation mean daily log loss; market baseline wins ties',
    embargo_hours=48,test_status='Reused historical dates: exploratory only',
    future_evaluation='Requires newly recorded observations, quotes and resolved outcomes after protocol creation')


def station_features(rows):
    extra=[]
    for r in rows:
        peak=r.get('observed_max',65);temp=r.get('observed_temperature',65)
        lo=r['lower'];hi=r['upper'];g=r['gfs'];e=r['ecmwf']
        extra.append([float(r.get('observed',False)),np.clip((lo-peak)/10,-3,3),np.clip((hi-peak)/10,-3,3),
            float(not math.isfinite(lo)),float(not math.isfinite(hi)),(peak-temp)/10,
            r.get('observed_trend',0)/10,r.get('observation_age_hours',3)/3,r.get('local_day_hour',0)/24,
            (g-peak)/10,(e-peak)/10,abs(g-e)/10,np.clip((lo-(g+e)/2)/10,-3,3),np.clip((hi-(g+e)/2)/10,-3,3)])
    return np.column_stack([features(rows),np.asarray(extra)])


class NewtonTree:
    """Public sklearn routing plus explicit Newton leaf corrections.

    Never mutate sklearn's private tree value buffer. Existing pickles with
    plain DecisionTreeRegressor entries remain readable via predict().
    """
    def __init__(self,tree,values):self.tree=tree;self.values=values
    def predict(self,X):return np.asarray([self.values[int(node)] for node in self.tree.apply(X)])


class StationOffset:
    """Shallow Newton boosting learns corrections to contemporaneous market odds."""
    name='station_offset'
    feature_mode='station'
    learning_rate=.05
    def __init__(self,leaves=4,feature_mode='station'):
        self.leaves=leaves;self.trees=[];self.weight=1.;self.feature_mode=feature_mode;self.learning_rate=PROTOCOL['learning_rate']
    def inputs(self,rows):return features(rows) if self.feature_mode=='market' else station_features(rows)
    def predict(self,rows):
        market=np.asarray([r['p'] for r in rows],dtype=float)
        if not self.trees or self.weight==0:return market
        X=self.inputs(rows);z=logit(np.clip(market,1e-6,1-1e-6))
        for tree in self.trees:z+=self.learning_rate*tree.predict(X)
        p=self.weight*expit(z)+(1-self.weight)*market
        return np.where([r.get('observed',False) for r in rows],p,market)

    def fit(self,train,val):
        usable=[r for r in train if r.get('observed')]
        if len(usable)<500:raise ValueError('Too few rows with available station observations')
        X=self.inputs(usable);y=np.asarray([r['y'] for r in usable]);w=date_weights(usable)
        z=logit(np.clip([r['p'] for r in usable],1e-6,1-1e-6));best=deepcopy(self)
        best_loss=score(val,[r['p'] for r in val])['log_loss'];self.history=[]
        for iteration in range(1,PROTOCOL['iterations']+1):
            p=expit(z);h=np.maximum(p*(1-p),1e-5);gradient=y-p
            tree=DecisionTreeRegressor(max_leaf_nodes=self.leaves,min_samples_leaf=PROTOCOL['minimum_leaf'],random_state=1729)
            tree.fit(X,np.clip(gradient/h,-20,20),sample_weight=w*h)
            nodes=tree.apply(X)
            values={}
            for node in np.unique(nodes):
                ix=nodes==node
                values[int(node)]=float(np.clip(np.sum(w[ix]*gradient[ix])/(np.sum(w[ix]*h[ix])+PROTOCOL['leaf_regularization']),-2,2))
            correction=NewtonTree(tree,values)
            self.trees.append(correction);z+=self.learning_rate*correction.predict(X)
            if iteration%5==0:
                loss=score(val,self.predict(val))['log_loss'];self.history.append(dict(iteration=iteration,validation_loss=loss))
                if loss<best_loss-1e-8:best_loss=loss;best=deepcopy(self)
        best.history=self.history;return best


class TreeMarketOffset(StationOffset):
    """Same Newton correction architecture using deployable price inputs only."""
    name='market_offset'
    def __init__(self,leaves=4):super().__init__(leaves,feature_mode='market')
    def fit(self,train,val):
        return super().fit([dict(r,observed=True) for r in train],val)
    def predict(self,rows):
        return super().predict([dict(r,observed=True) for r in rows])


MarketOffset=TreeMarketOffset # Historical pickle/import compatibility, not the logistic control.


def build_station_rows(store,progress=None):
    rows,excluded,forecast_sources=collect_features(store,progress);sources={};indices={}
    for series,station in SITES.items():
        group=[r for r in rows if r['series']==series]
        if not group:continue
        dates=[event_date(r['ticker']) for r in group]
        if progress:progress('Downloading observations for '+station,0,4)
        payload=fetch_observations(store,station,min(dates).date().isoformat(),(max(dates)+timedelta(days=1)).date().isoformat())
        sources[station]={k:v for k,v in payload.items() if k!='observations'};indices[series]=ObservationIndex(payload['observations'])
    joined=[];delayed=[]
    for r in rows:
        joined.append(dict(r,**indices[r['series']].feature(r['series'],r['ticker'],r['at'],30)))
        delayed.append(dict(r,**indices[r['series']].feature(r['series'],r['ticker'],r['at'],90)))
    return joined,delayed,dict(observations=sources,forecasts=forecast_sources,excluded=excluded)


def run_station_experiment(store,progress=None):
    rs=ResearchStore(store);run_id='station-'+utcnow().replace(':','').replace('+','-').replace('.','-')
    folder=store.root/'station-model'/run_id;folder.mkdir(parents=True)
    from .weather_model import STATIONS
    protocol=dict(PROTOCOL,created_at=utcnow(),stations=STATIONS,station_ids=SITES)
    # Written before data fitting or any final-test scoring.
    (folder/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    rows,delayed,sources=build_station_rows(store,progress);train,val,test=split_dataset(rows)
    # Preserve the actual feature dataset, not only a hash of mutable caches.
    safe=lambda data:[dict(r,lower=None if not math.isfinite(r['lower']) else r['lower'],upper=None if not math.isfinite(r['upper']) else r['upper']) for r in data]
    (folder/'features.jsonl').write_text('\n'.join(json.dumps(r,allow_nan=False) for r in safe(rows)),encoding='utf-8')
    (folder/'delayed-features.jsonl').write_text('\n'.join(json.dumps(r,allow_nan=False) for r in safe(delayed)),encoding='utf-8')
    candidates=[dict(leaves=0,iterations=0,weight=0,validation=score(val,[r['p'] for r in val]))]
    best=StationOffset();best_score=candidates[0]['validation']['log_loss']
    with threadpool_limits(limits=2):
        for leaves in PROTOCOL['candidate_leaves']:
            if progress:progress(f'Training station-conditioned correction: {leaves} leaves',0,120)
            model=StationOffset(leaves).fit(train,val)
            for weight in PROTOCOL['weights'][1:]:
                model.weight=weight;metrics=score(val,model.predict(val))
                candidates.append(dict(leaves=leaves,iterations=len(model.trees),weight=weight,validation=metrics))
                if metrics['log_loss']<best_score-1e-8:best=deepcopy(model);best_score=metrics['log_loss']
        # Freeze artifact before final test labels are scored.
        model_path=folder/'model.pkl';model_path.write_bytes(pickle.dumps(best))
        predicted=best.predict(test);market=[r['p'] for r in test]
        testkeys={(r['ticker'],r['at']) for r in test};slow=[r for r in delayed if (r['ticker'],r['at']) in testkeys]
        report=dict(id=run_id,created_at=utcnow(),protocol=protocol,passed=False,synthetic=False,sources=sources,
            data_hash=digest([dict(r,lower=None if not math.isfinite(r['lower']) else r['lower'],upper=None if not math.isfinite(r['upper']) else r['upper']) for r in rows]),
            source_hash=hashlib.sha256(Path(__file__).read_bytes()+Path(__file__).with_name('station_observations.py').read_bytes()).hexdigest(),
            artifact=str(model_path),artifact_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
            selected=dict(leaves=best.leaves,iterations=len(best.trees),weight=best.weight),candidates=candidates,
            splits={k:partition_summary(v) for k,v in [('train',train),('validation',val),('test',test)]},
            observation_coverage={k:sum(r['observed'] for r in v)/len(v) for k,v in [('train',train),('validation',val),('test',test)]},
            holdout=score(test,predicted),market_baseline=score(test,market),paired_interval=paired_block_interval(test,predicted,market),
            delayed_observations=score(slow,best.predict(slow)),diagnostics=segment_scores(test,predicted),
            scenarios=[quote_scenario(test,predicted,slippage=s,fee_multiplier=f) for s,f in ((0,1),(.02,1),(.05,2))],
            limits=['Reused historical test dates. No automatic promotion or proof of executable profit.',
                'IEM observations are retrospective. Thirty-minute availability is assumed; ninety-minute sensitivity is also reported.',
                'Known corrected and conflicting observations are excluded; unknown backfills remain possible.',
                'Hourly and six-hour station reports are imperfect proxies for final climate settlement maxima.',
                'Model uses market odds plus forecasts and observations; missing observations fall back exactly to market odds.',
                'This artifact requires station/forecast features and is not installed as the live market-price model.'])
    (folder/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    rs.set('station_model',report);return report


if __name__=='__main__':
    import argparse
    from .storage import Store
    parser=argparse.ArgumentParser();parser.add_argument('--data',default='data');parser.add_argument('--report',default='reports/station-model.json');args=parser.parse_args()
    # Pickles must reference the importable class, never __main__.StationOffset.
    from helper.station_model import run_station_experiment as run_importable
    report=run_importable(Store(args.data),lambda message,*_:print(message,flush=True))
    output=Path(args.report);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('selected','paired_interval','observation_coverage')},indent=2))
