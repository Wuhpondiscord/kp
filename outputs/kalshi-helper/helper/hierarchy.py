"""Shared price calibration plus regularized city deviations; research only."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import logit,expit
from .residual import date_weights
from .weather_model import STATIONS

class HierarchicalOffset:
    def __init__(self,city_penalty=.1,pooled=False):
        self.city_penalty=city_penalty;self.pooled=pooled
    def inputs(self,rows):
        z=logit(np.clip([r['p'] for r in rows],1e-6,1-1e-6))
        shared=np.column_stack([np.ones(len(rows)),z,np.asarray([r['hours_left']/24 for r in rows])])
        if self.pooled:return z,shared
        children=np.column_stack([shared[:,j]*np.asarray([float(r['series']==s) for r in rows]) for s in STATIONS for j in (0,1)])
        return z,np.column_stack([shared,children])
    def fit(self,rows):
        offset,x=self.inputs(rows);y=np.asarray([r['y'] for r in rows]);w=date_weights(rows)
        penalty=np.full(x.shape[1],self.city_penalty);penalty[:3]=.01
        def objective(theta):
            z=offset+x@theta
            return np.mean(w*(np.logaddexp(0,z)-y*z))+.5*np.sum(penalty*theta**2),x.T@(w*(expit(z)-y))/len(y)+penalty*theta
        r=minimize(objective,np.zeros(x.shape[1]),jac=True,method='L-BFGS-B')
        if not r.success:raise ValueError('Hierarchy optimizer failed')
        self.theta=r.x
        fitted=expit(offset+x@self.theta)
        curvature=x.T@((w*fitted*(1-fitted))[:,None]*x)/len(y)
        self.effective_df=float(np.trace(np.linalg.pinv(curvature+np.diag(penalty))@curvature))
        self.penalty_convention='Mean date-weighted log loss + 0.5 * sum(penalty * coefficient squared)'
        return self
    def predict(self,rows):
        offset,x=self.inputs(rows);return expit(offset+x@self.theta)
