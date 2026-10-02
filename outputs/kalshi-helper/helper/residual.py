"""Market-offset neural correction, with train-only scaling and weighted Adam.

The zero-initialized output layer starts at the market's log odds. Hidden
layers learn a correction, rather than relearning the entire base forecast.
"""
from collections import Counter
import numpy as np
from scipy.special import expit, logit
from sklearn.preprocessing import StandardScaler
from .evaluation import features


def date_weights(rows):
    counts=Counter(r['group'] for r in rows)
    weights=np.array([1/counts[r['group']] for r in rows])
    return weights/weights.mean()


class ResidualNetwork:
    name='neural_trained'
    def __init__(self,rows,config):
        self.weight=1.
        self.scaler=StandardScaler().fit(features(rows))
        self.rng=np.random.default_rng(config['seed'])
        sizes=[features(rows).shape[1],*map(int,config['architecture'].split(',')),1]
        self.coefs_=[self.rng.normal(0,np.sqrt(2/(a+b)),(a,b)) for a,b in zip(sizes,sizes[1:])]
        self.coefs_[-1][:]=0
        self.biases_=[np.zeros(b) for b in sizes[1:]]
        self.alpha=config['alpha']/len(rows)
        self.rate=config['learning_rate'];self.batch=config['batch_size'];self.step=0
        self.m=[np.zeros_like(p) for p in self.parameters()]
        self.v=[np.zeros_like(p) for p in self.parameters()]

    def parameters(self):return self.coefs_+self.biases_

    def forward(self,X,offset):
        layers=[X]
        for W,b in zip(self.coefs_[:-1],self.biases_[:-1]):layers.append(np.tanh(layers[-1]@W+b))
        logits=offset+(layers[-1]@self.coefs_[-1]+self.biases_[-1]).ravel()
        return logits,layers

    def loss_grad(self,X,offset,y,weights):
        logits,layers=self.forward(X,offset)
        # Weights have mean 1 globally; do not renormalize each minibatch.
        loss=np.mean(weights*(np.logaddexp(0,logits)-y*logits))
        loss+=.5*self.alpha*sum(np.square(W).sum() for W in self.coefs_)
        delta=((expit(logits)-y)*weights/len(y))[:,None]
        dW=[None]*len(self.coefs_);db=[None]*len(self.biases_)
        for i in reversed(range(len(self.coefs_))):
            dW[i]=layers[i].T@delta+self.alpha*self.coefs_[i];db[i]=delta.sum(axis=0)
            if i:delta=(delta@self.coefs_[i].T)*(1-layers[i]**2)
        return float(loss),dW+db

    def fit_epoch(self,rows):
        X=self.scaler.transform(features(rows));p=np.clip([r['p'] for r in rows],1e-6,1-1e-6)
        y=np.array([r['y'] for r in rows]);weights=date_weights(rows);offset=logit(p)
        order=self.rng.permutation(len(rows))
        for start in range(0,len(rows),self.batch):
            indices=order[start:start+self.batch]
            _,grads=self.loss_grad(X[indices],offset[indices],y[indices],weights[indices])
            self.step+=1
            for i,(param,grad) in enumerate(zip(self.parameters(),grads)):
                self.m[i]=.9*self.m[i]+.1*grad;self.v[i]=.999*self.v[i]+.001*grad**2
                param-=self.rate*(self.m[i]/(1-.9**self.step))/(np.sqrt(self.v[i]/(1-.999**self.step))+1e-8)
        self.loss_=self.loss_grad(X,offset,y,weights)[0]
        if not np.isfinite(self.loss_):raise ValueError('Non-finite residual training loss')

    def predict(self,rows):
        market=np.array([r['p'] for r in rows],dtype=float)
        if self.weight==0:return market
        logits,_=self.forward(self.scaler.transform(features(rows)),logit(np.clip(market,1e-6,1-1e-6)))
        return np.clip(self.weight*expit(logits)+(1-self.weight)*market,1e-6,1-1e-6)
