"""Regularized market-offset calibration and validation-only model selection.

Fixed bases keep this CPU-sized and preserve the market as a zero-correction
baseline. No test rows enter fitting, selection, or shrinkage.
"""
from copy import deepcopy
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.preprocessing import StandardScaler
from .discovery import WEATHER_SERIES
from .residual import date_weights
from .evaluation import score


def basis(rows,kind):
    p=np.clip([r['p'] for r in rows],1e-6,1-1e-6)
    z=logit(p);h=np.asarray([r['hours_left']/24 for r in rows])
    columns=[np.log(p),-np.log1p(-p)]
    if kind!='beta':
        columns += [h,h*h,np.asarray([r['spread'] for r in rows]),h*z]
        for city in WEATHER_SERIES[:-1]:
            c=np.asarray([float(r['series']==city) for r in rows])
            columns += [c,c*z]
    if kind=='spline':
        columns += [np.maximum(z-k,0) for k in (-4,-2,0,2,4)]
    return np.column_stack(columns)


class OffsetCalibrator:
    name='adaptive_market'
    def __init__(self,kind='context',penalty=.001):
        self.kind=kind;self.penalty=penalty;self.weight=1.

    def fit(self,rows):
        self.scaler=StandardScaler().fit(basis(rows,self.kind))
        X=np.column_stack([np.ones(len(rows)),self.scaler.transform(basis(rows,self.kind))])
        y=np.asarray([r['y'] for r in rows]);w=date_weights(rows)
        offset=logit(np.clip([r['p'] for r in rows],1e-6,1-1e-6))
        def objective(theta):
            z=offset+X@theta
            loss=np.mean(w*(np.logaddexp(0,z)-y*z))+.5*self.penalty*np.dot(theta,theta)
            grad=X.T@(w*(expit(z)-y))/len(y)+self.penalty*theta
            return loss,grad
        result=minimize(objective,np.zeros(X.shape[1]),jac=True,method='L-BFGS-B',options={'maxiter':500,'ftol':1e-12})
        if not result.success or not np.all(np.isfinite(result.x)):
            raise ValueError('Calibration optimizer failed: '+str(result.message))
        self.coef=result.x;return self

    def predict(self,rows):
        p=np.asarray([r['p'] for r in rows],dtype=float)
        if self.weight==0:return p
        X=np.column_stack([np.ones(len(rows)),self.scaler.transform(basis(rows,self.kind))])
        learned=expit(logit(np.clip(p,1e-6,1-1e-6))+X@self.coef)
        return self.weight*learned+(1-self.weight)*p


class SelectedPredictor:
    name='adaptive_market'
    def __init__(self,model,weight,label):
        self.model=model;self.weight=weight;self.label=label
    def predict(self,rows):
        market=np.asarray([r['p'] for r in rows],dtype=float)
        if self.weight==0:return market
        return self.weight*self.model.predict(rows)+(1-self.weight)*market


def select_adaptive(train,val,neural,cancel=None):
    """Select by validation log loss, requiring no worse validation Brier.

The small fixed candidate grid is declared before any final-test scoring.
The baseline wins exact ties. Neural epochs remain visible independently.

Log loss is the primary proper score for likelihood-based probability fitting;
Brier is a secondary guard against worsening mean squared probability error.
This is an engineering multi-objective rule, not a statistical test or a profit
criterion. Its historical choice was not independently preregistered; do not
claim otherwise or reverse the priority after inspecting the same holdout.
"""
    baseline=score(val,[r['p'] for r in val]);candidates=[]
    best=SelectedPredictor(None,0,'Market baseline');best_loss=baseline['log_loss']
    models=[('Neural checkpoint',deepcopy(neural))]
    models[0][1].weight=1
    for kind in ('beta','context','spline'):
        for penalty in (.001,.01):
            if cancel and cancel.is_set():raise InterruptedError('Training cancelled during model selection')
            models.append((f'{kind} offset / penalty {penalty}',OffsetCalibrator(kind,penalty).fit(train)))
    from .station_model import TreeMarketOffset
    for leaves in ((4,7) if len(train)>=500 else ()):
        if cancel and cancel.is_set():raise InterruptedError('Training cancelled during model selection')
        models.append((f'Market-offset trees / {leaves} leaves',TreeMarketOffset(leaves).fit(train,val)))
    for label,model in models:
        raw=model.predict(val);market=np.asarray([r['p'] for r in val])
        for weight in (.25,.5,.75,1):
            metrics=score(val,weight*raw+(1-weight)*market)
            eligible=metrics['brier']<=baseline['brier']
            candidates.append(dict(label=label,weight=weight,log_loss=metrics['log_loss'],brier=metrics['brier'],eligible=eligible))
            if eligible and metrics['log_loss']<best_loss-1e-8:
                best_loss=metrics['log_loss'];best=SelectedPredictor(model,weight,label)
    return best,candidates
