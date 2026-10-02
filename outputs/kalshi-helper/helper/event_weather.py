"""Coherent maximum-temperature distribution prototype, research only.

Existing archived daily maxima proxy the remaining-day maximum. The new
prospective hourly source is needed to validate a true remaining-path model.
"""
import math
from collections import Counter,defaultdict
import numpy as np
from scipy.special import ndtr
from scipy.optimize import minimize
from .weather_model import STATIONS

def complete_events(rows):
    groups=defaultdict(list)
    for r in rows:groups[(r['event'],r['at'])].append(r)
    keep=[];excluded=0
    for group in groups.values():
        group=sorted(group,key=lambda r:r['lower'])
        if sum(r['y'] for r in group)!=1 or group[0]['lower']!=-math.inf or group[-1]['upper']!=math.inf or any(a['upper']!=b['lower'] for a,b in zip(group,group[1:])):
            excluded+=1;continue
        keep.extend(group)
    return sorted(keep,key=lambda r:(r['at'],r['ticker'])),excluded

def normalized_market(rows):
    totals=defaultdict(float)
    for r in rows:totals[(r['event'],r['at'])]+=r['p']
    return np.asarray([r['p']/totals[(r['event'],r['at'])] for r in rows])

def event_loss(rows,p):
    daily=defaultdict(list)
    for r,v in zip(rows,p):
        if r['y']:daily[r['group']].append(-math.log(max(float(v),1e-12)))
    return float(np.mean([np.mean(v) for v in daily.values()]))

class EventMaximum:
    def __init__(self, observation_calibration=None, remaining_spread='constant'):
        # External calibration only: never add this scale to the censored MLE.
        from copy import deepcopy
        from .observation_calibration import validate_calibration
        self.observation_calibration=deepcopy(observation_calibration)
        if observation_calibration is not None:validate_calibration(observation_calibration)
        if remaining_spread not in ('constant','disagreement'):raise ValueError('Unknown remaining spread model')
        self.remaining_spread=remaining_spread

    def spread_feature(self,rows):
        values=[r.get('model_disagreement_f') for r in rows]
        if any(v is None or not math.isfinite(v) or v<0 for v in values):
            raise ValueError('Disagreement spread requires finite nonnegative disagreement; missing is not agreement')
        # Bounded covariate avoids exponential extrapolation far outside training.
        return np.minimum(np.asarray(values,dtype=float)/5,4)

    def distribution(self,rows):
        g,X=self.inputs(rows);n=X.shape[1]
        mu=g+X@self.theta[:n]
        log_sigma=np.full(len(rows),self.theta[n])
        if getattr(self,'remaining_spread','constant')=='disagreement':
            log_sigma+=self.theta[n+1]*self.spread_feature(rows)
        return mu,np.exp(log_sigma)

    def observation_parameters(self,row):
        c=self.observation_protocol()
        if c.get('pooling')=='station':
            from .station_observations import SITES
            s=c['by_station'].get(SITES.get(row['series']))
            if s is not None and s['n']>=80:return s['bias_f'],s['sigma_f']
        return c['bias_f'],c['sigma_f']

    def observation_protocol(self):
        calibration=getattr(self,'observation_calibration',None)
        return calibration or dict(method='legacy_assumption',bias_f=0.,sigma_f=1.,n_events=0,
            warning='Uncalibrated 1F assumption; not an empirical sensor-noise estimate')

    def inputs(self,rows):
        g=np.asarray([(r['gfs']+r['ecmwf'])/2 for r in rows])
        X=np.asarray([[1,(r['gfs']-r['ecmwf'])/5,r['hours_left']/24,*[float(r['series']==s) for s in list(STATIONS)[1:]]] for r in rows])
        return g,X
    def predict(self,rows):
        if any(r.get('observed') and (r.get('observed_max') is None or not math.isfinite(r['observed_max'])) for r in rows):
            raise ValueError('Observed rows require a finite observed maximum')
        mu,sigma=self.distribution(rows)
        parameters=[self.observation_parameters(r) if r.get('observed') else (0,1) for r in rows]
        def cdf(bound):
            b=np.asarray([r[bound] for r in rows]);remaining=ndtr((b-mu)/sigma)
            observed=np.asarray([ndtr((r[bound]-(r['observed_max']+bias))/scale) if r.get('observed') else 1 for r,(bias,scale) in zip(rows,parameters)])
            return remaining*observed
        return np.clip(cdf('upper')-cdf('lower'),0,1)
    def fit(self,rows):
        calibration=getattr(self,'observation_calibration',None)
        if calibration is not None:
            from .observation_calibration import validate_training_scope
            validate_training_scope(calibration,rows)
        winning=[r for r in rows if r['y']==1];counts=Counter(r['event'] for r in winning)
        if len(counts)<80:raise ValueError('Need 80 distinct training events')
        weights=np.asarray([1/counts[r['event']] for r in winning]);weights/=weights.sum()
        _,X=self.inputs(winning);initial=np.zeros(X.shape[1]+1);initial[-1]=math.log(4)
        bounds=[(-10,10)]*X.shape[1]+[(math.log(.5),math.log(20))]
        if getattr(self,'remaining_spread','constant')=='disagreement':
            self.spread_feature(winning)
            initial=np.append(initial,0.);bounds.append((0.,1.))
        def objective(theta):
            self.theta=theta
            # Same mean penalty in both arms; one nonnegative spread coefficient.
            return float(-weights@np.log(np.clip(self.predict(winning),1e-12,1))+.002*np.square(theta[:X.shape[1]]).sum())
        result=minimize(objective,initial,method='L-BFGS-B',bounds=bounds)
        if not result.success:raise ValueError(result.message)
        self.theta=result.x;return self
