"""Multi-source, lead-aware weather distribution experiments on real contracts."""
from collections import Counter
from datetime import timedelta
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import ndtr
from .core import stamp,utcnow
from .weather_model import STATIONS,fetch_hours,forecast_feature,event_date,bounds,supported,design,blend
from .history import sample_rows
from .training import split_dataset,partition_summary
from .evaluation import score,paired_block_interval
from .research_store import ResearchStore,digest

PROVIDERS={'gfs':'gfs_seamless','ecmwf':'ecmwf_ifs025'}


def freshest_feature(series,ticker,archives,decision):
    errors=[]
    for lead in (1,2):
        try:return forecast_feature(series,ticker,archives[lead],decision,lead)
        except (ValueError,KeyError) as exc:errors.append(str(exc))
    raise ValueError('No complete forecast available before decision: '+'; '.join(errors))


def collect_features(store,progress=None):
    rs=ResearchStore(store);history=rs.history()
    markets={x['market']['ticker']:x['market'] for x in history}
    source=sample_rows(store);rows=[];excluded=Counter();requests=[]
    for city in STATIONS:
        group=[r for r in source if r['series']==city]
        if not group:continue
        dates=[event_date(r['ticker']) for r in group]
        start=min(dates).date().isoformat();end=(max(dates)+timedelta(days=1)).date().isoformat()
        archives={}
        for provider,model in PROVIDERS.items():
            archives[provider]={}
            for lead in (1,2):
                if progress:progress(f'Archived {provider.upper()} day {lead}: {city}',len(requests),16)
                archives[provider][lead]=fetch_hours(store,city,start,end,lead,model)
                requests.append(dict(city=city,provider=model,lead_days=lead,start=start,end=end,hours=len(archives[provider][lead])))
        for r in group:
            try:
                if not supported(markets[r['ticker']],city):raise ValueError('Unsupported settlement rules')
                low,high=bounds(markets[r['ticker']])
                old=forecast_feature(city,r['ticker'],archives['gfs'][2],r['at'])
                g=freshest_feature(city,r['ticker'],archives['gfs'],r['at'])
                e=freshest_feature(city,r['ticker'],archives['ecmwf'],r['at'])
                rows.append(dict(r,lower=low,upper=high,weather_day=g['weather_day'],
                    gfs_old=old['forecast_high'],gfs=g['forecast_high'],ecmwf=e['forecast_high'],
                    gfs_lead=g['weather_lead_days'],ecmwf_lead=e['weather_lead_days'],
                    weather_available_at=max(g['weather_available_at'],e['weather_available_at'],key=stamp)))
            except (KeyError,ValueError) as exc:excluded[str(exc)]+=1
    return sorted(rows,key=lambda r:(r['at'],r['ticker'])),dict(excluded),requests


class WeatherDistribution:
    """Learn location and log-scale; disagreement is a predictor, not an ensemble spread."""
    def __init__(self,source):self.source=source

    def inputs(self,rows):
        mean=np.asarray([(r['gfs']+r['ecmwf'])/2 if self.source=='multi' else r[self.source] for r in rows])
        lead=np.asarray([2 if self.source=='gfs_old' else max(r['gfs_lead'],r['ecmwf_lead']) if self.source=='multi' else r[self.source+'_lead'] for r in rows])
        adjusted=[dict(r,forecast_high=float(m)) for r,m in zip(rows,mean)]
        X=np.column_stack([design(adjusted),lead-1])
        # City, forecast lead and model disagreement allow uncertainty to vary.
        Z=np.column_stack([np.ones(len(rows)),lead-1,
            *[[float(r['series']==s) for r in rows] for s in list(STATIONS)[1:]],
            [abs(r['gfs']-r['ecmwf'])/5 if self.source=='multi' else 0 for r in rows]])
        return mean,X,Z

    def predict(self,rows):
        mean,X,Z=self.inputs(rows);n=X.shape[1]
        mu=mean+X@self.parameters[:n];sigma=np.exp(np.clip(Z@self.parameters[n:],math.log(.5),math.log(20)))
        low=np.asarray([r['lower'] for r in rows]);high=np.asarray([r['upper'] for r in rows])
        return np.clip(ndtr((high-mu)/sigma)-ndtr((low-mu)/sigma),1e-6,1-1e-6)

    def fit(self,train):
        # Repeated market quotes do not create new temperature outcomes.
        unique={}
        for r in train:
            if r['y']==1:unique[(r['event'],r['gfs_lead'],r['ecmwf_lead'])]=r
        rows=list(unique.values());counts=Counter(r['event'] for r in rows)
        if len(counts)<80:raise ValueError('Weather fitting needs 80 resolved training events')
        weights=np.asarray([1/counts[r['event']] for r in rows]);weights/=weights.sum()
        _,X,Z=self.inputs(rows);initial=np.zeros(X.shape[1]+Z.shape[1]);initial[X.shape[1]]=math.log(4)
        def objective(theta):
            self.parameters=theta
            penalty=np.r_[theta[:X.shape[1]],theta[X.shape[1]+1:]]
            return float(-np.dot(weights,np.log(self.predict(rows)))+.002*np.square(penalty).sum())
        result=minimize(objective,initial,method='L-BFGS-B',bounds=[(-10,10)]*len(initial),options={'maxiter':500})
        if not result.success:raise ValueError('Weather fit did not converge: '+str(result.message))
        self.parameters=result.x;self.events=len(counts);return self


def train_weather_experiment(store,progress=None):
    rs=ResearchStore(store)
    def update(message,done=0,total=16):
        rs.set('weather_training',dict(status='running',message=message,at=utcnow()))
        if progress:progress(message,done,total)
    try:
        update('Building timestamp-aligned multi-source weather dataset')
        rows,excluded,requests=collect_features(store,update)
        train,val,test=split_dataset(rows);candidates=[];models={}
        for source in ('gfs_old','gfs','ecmwf','multi'):
            update('Fitting conditional temperature distribution: '+source)
            model=WeatherDistribution(source).fit(train);models[source]=model
            probs=model.predict(val)
            for weight in (0,.1,.25,.5,.75,1):
                metrics=score(val,blend(probs,[r['p'] for r in val],weight))
                candidates.append(dict(source=source,weight=weight,validation=metrics))
        best=min(candidates,key=lambda c:c['validation']['log_loss'])
        model=models[best['source']];weather=model.predict(test)
        predictions=blend(weather,[r['p'] for r in test],best['weight'])
        # All ablations were declared above; test results are descriptive only.
        ablations={name:score(test,m.predict(test)) for name,m in models.items()}
        safe=[dict(r,lower=None if not math.isfinite(r['lower']) else r['lower'],upper=None if not math.isfinite(r['upper']) else r['upper']) for r in rows]
        report=dict(status='complete',at=utcnow(),model='Multi-source conditional Gaussian weather calibration',
            selected_source=best['source'],parameters=model.parameters.tolist(),weight=best['weight'],
            winning_training_events=model.events,excluded_samples=sum(excluded.values()),exclusion_reasons=excluded,
            lead_counts=dict(Counter(str(r['gfs_lead']) for r in rows)),sources=requests,
            splits={k:partition_summary(v) for k,v in [('train',train),('validation',val),('test',test)]},
            candidates=candidates,holdout=score(test,predictions),weather_only=score(test,weather),
            baseline=score(test,[r['p'] for r in test]),ablations=ablations,
            paired_interval=paired_block_interval(test,predictions,[r['p'] for r in test]),
            data_hash=digest(safe),trained_through=max(r['settled_at'] for r in train+val),passed=False,
            source_hash=hashlib.sha256(Path(__file__).read_bytes()+Path(__file__).with_name('weather_model.py').read_bytes()).hexdigest(),
            protocol=dict(geography_version=2,stations=STATIONS,providers=PROVIDERS,leads=[1,2],availability_allowance_hours=6,
                selection='Validation log loss chooses source and blend; whole-date chronological split with 48-hour embargo',
                objective='Event-weighted winning-interval likelihood; city/lead/disagreement-dependent log scale',test_reused=True),
            limits=['Exploratory reused test dates; no automatic promotion or live-weather deployment.',
                'Fixed-lead archive availability is inferred with a six-hour allowance, not verified original ingestion.',
                'Gridpoint hourly maxima proxy station maxima. Contract rules and station/provider changes need verification.',
                'Two deterministic model outputs are not a probabilistic ensemble. Disagreement is learned only as a scale feature.',
                'Winning contract brackets are censored labels, not measured exact temperatures.',
                'A zero selected weather weight means validation found no added benefit over market prices.'])
        folder=store.root/'weather-model';folder.mkdir(exist_ok=True)
        (folder/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        (folder/'features.jsonl').write_text('\n'.join(json.dumps(r,allow_nan=False) for r in safe),encoding='utf-8')
        rs.set('weather_model',report);rs.set('weather_training',dict(status='complete',message='Multi-source fit and final test complete',at=utcnow()))
        return report
    except Exception as exc:
        rs.set('weather_training',dict(status='failed',message=str(exc),at=utcnow()));raise
