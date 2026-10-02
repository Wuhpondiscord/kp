"""Small weather/fusion models; legacy artifacts also serve experimental live inference."""
import math
from collections import Counter
import numpy as np
import torch
from torch import nn
from scipy.optimize import minimize
from scipy.special import expit,logit
from .weather_model import STATIONS,event_date
from .forecast_archive import MODELS
from .core import stamp

def features(rows):
    return np.asarray([[*[float(r['series']==s) for s in STATIONS],
        r['observation_innovation_f']/5,r['observed_trend']/5,
        (r['observed_temperature']-r['remaining_forecast_max'])/10,
        (r['observed_max']-r['remaining_forecast_max'])/10,r['remaining_hours']/24,
        (r['gfs_remaining_max']-r['ecmwf_remaining_max'])/5,r['model_disagreement_f']/5,
        r['observation_age_hours']/3] for r in rows],dtype=float)

def sequence(record,row):
    """Two already-issued forecast paths, 16 relative-time samples, not observations."""
    from datetime import timedelta
    now=stamp(row['at']);run=stamp(record['run_initialized_at'])
    if run+timedelta(hours=8)>now:raise ValueError('Future forecast run')
    body=record['body'][list(STATIONS).index(row['series'])]
    times=np.asarray([stamp(t+'Z').timestamp() for t in body['hourly']['time']])
    end=event_date(row['ticker'])+timedelta(hours=STATIONS[row['series']][2]+24)
    target=np.linspace(now.timestamp(),(end-timedelta(hours=1)).timestamp(),16)
    if target[-1]<target[0] or target[0]<times[0] or target[-1]>times[-1]:raise ValueError('Incomplete forecast path')
    if np.any(np.diff(times)!=3600):raise ValueError('Forecast path has gaps')
    out=[]
    for model in MODELS:
        key='temperature_2m_'+model
        if body['hourly_units'][key]!='°F':raise ValueError('Path requires Fahrenheit')
        values=np.asarray(body['hourly'][key],dtype=float)
        interpolated=np.interp(target,times,values)
        if not np.all(np.isfinite(interpolated)):raise ValueError('Missing forecast path')
        out.append((interpolated-row['remaining_forecast_max'])/10)
    return np.asarray(out)

def weights(rows):
    counts=Counter((r['group'],r['event']) for r in rows);days=Counter(d for d,e in counts)
    return np.asarray([1/(len(days)*days[r['group']]*counts[(r['group'],r['event'])]) for r in rows])

class LogisticMarketOffset:
    """Penalized market-only control, zero correction is the identity baseline."""
    def x(self,rows):
        return np.asarray([[1,logit(np.clip(r['p'],.001,.999))/4,r['spread'],
            *[float(r['series']==s) for s in list(STATIONS)[1:]]] for r in rows])
    def fit(self,rows):
        x=self.x(rows);base=logit(np.clip([r['p'] for r in rows],1e-6,1-1e-6));y=np.asarray([r['y'] for r in rows]);w=weights(rows)
        def objective(b):
            z=base+x@b
            return float(w@(np.logaddexp(0,z)-y*z)+.1*(b@b))
        opt=minimize(objective,np.zeros(x.shape[1]),method='L-BFGS-B')
        if not opt.success:raise ValueError(opt.message)
        self.coef=opt.x;return self
    def predict(self,rows):return expit(logit(np.clip([r['p'] for r in rows],1e-6,1-1e-6))+self.x(rows)@self.coef)

MarketOffset=LogisticMarketOffset # Compatibility for historical pickles and scripts.

class SmallJoint(nn.Module):
    def __init__(self,temporal=False,fusion=False):
        super().__init__();self.temporal=temporal;self.fusion=fusion
        self.conv=nn.Conv1d(2,4,3,padding=1) if temporal else None
        self.encoder=nn.Sequential(nn.Linear(12+(8 if temporal else 0),8),nn.Tanh())
        self.weather=nn.Linear(8,2)
        nn.init.zeros_(self.weather.weight);nn.init.zeros_(self.weather.bias)
        if fusion:
            self.market=nn.Sequential(nn.Linear(12,4),nn.Tanh(),nn.Linear(4,1))
            nn.init.zeros_(self.market[-1].weight);nn.init.zeros_(self.market[-1].bias)

    def forward(self,x,path,baseline,low,high,obs,obs_bias,obs_sigma,market,spread,event_groups=None):
        if self.temporal:
            t=torch.tanh(self.conv(path));x=torch.cat([x,t.mean(-1),t.amax(-1)],dim=1)
        latent=self.encoder(x);raw=self.weather(latent)
        mu=baseline+10*torch.tanh(raw[:,0]/10);sigma=torch.exp(math.log(3)+1.5*torch.tanh(raw[:,1]))
        # Infinite interval endpoints are CDF limits with zero gradient. Clipping
        # standardized endpoints avoids inf * 0 gradients in erf backward.
        def cdf(b):
            finite=torch.isfinite(b);safe=torch.where(finite,b,torch.zeros_like(b))
            z=torch.where(finite,torch.clamp((safe-mu)/sigma,-12,12),torch.sign(b)*12)
            a=.5*(1+torch.erf(z/math.sqrt(2)))
            zo=torch.where(finite,torch.clamp((safe-obs-obs_bias)/obs_sigma,-12,12),torch.sign(b)*12)
            o=.5*(1+torch.erf(zo/math.sqrt(2)))
            return a*o
        weather=torch.clamp(cdf(high)-cdf(low),1e-10,1-1e-10)
        if not self.fusion:return weather,weather
        base=torch.logit(market.clamp(1e-6,1-1e-6))
        market_inputs=torch.cat([latent,base[:,None]/4,spread[:,None],
            torch.logit(weather)[:,None]/4,(sigma/5)[:,None]],dim=1)
        # One model trained jointly: bounded correction, market baseline at init.
        correction=.5*torch.tanh(self.market(market_inputs).squeeze(1))
        if getattr(self,'event_coherent',False):
            if event_groups is None:raise ValueError('Event fusion requires complete bracket groups')
            # Categorical probabilities require log probabilities, NOT binary
            # log odds. At zero correction this is normalized market mass.
            logits=torch.log(market.clamp(min=1e-6))+correction
            fused=torch.zeros_like(logits)
            for indices in event_groups:
                fused=fused.index_copy(0,indices,torch.softmax(logits[indices],dim=0))
            return weather,fused
        return weather,torch.sigmoid(base+correction)

class NeuralPredictor:
    def __init__(self,calibration,temporal=False,fusion=False,seed=1729,weather_loss_weight=1.,residual_penalty=0.,penalty_deadzone=0.,average_last=0):
        from .event_weather import EventMaximum
        if not math.isfinite(weather_loss_weight) or weather_loss_weight<=0:
            raise ValueError('Weather loss weight must be finite and positive')
        if not math.isfinite(residual_penalty) or residual_penalty<0:
            raise ValueError('Residual penalty must be finite and nonnegative')
        if not math.isfinite(penalty_deadzone) or not 0<=penalty_deadzone<=.5:
            raise ValueError('Penalty deadzone must be between 0 and .5 logit units')
        if not isinstance(average_last,int) or average_last<0:
            raise ValueError('Averaging window must be a nonnegative integer')
        self.penalty_deadzone=penalty_deadzone;self.average_last=average_last
        self.weather_loss_weight=weather_loss_weight;self.residual_penalty=residual_penalty
        self.calibration=calibration;self.observation=EventMaximum(calibration)
        torch.manual_seed(seed);self.network=SmallJoint(temporal,fusion).double();self.seed=seed
    def batch(self,rows,paths):
        tensor=lambda a:torch.as_tensor(np.asarray(a),dtype=torch.float64)
        x=features(rows)
        if not np.all(np.isfinite(x)):raise ValueError('Missing weather features')
        bias,scale=zip(*(self.observation.observation_parameters(r) for r in rows))
        return (tensor((x-self.mean)/self.scale),tensor(paths),tensor([r['remaining_forecast_max'] for r in rows]),
            tensor([r['lower'] for r in rows]),tensor([r['upper'] for r in rows]),tensor([r['observed_max'] for r in rows]),
            tensor(bias),tensor(scale),tensor([r['p'] for r in rows]),tensor([r['spread'] for r in rows]))
    def fit(self,rows,paths,epochs=200,warm_start=False,learning_rate=.01,freeze_backbone=False,progress=None):
        from .observation_calibration import validate_training_scope
        validate_training_scope(self.calibration,rows)
        average_last=getattr(self,'average_last',0)
        if average_last>epochs:raise ValueError('Averaging window exceeds training epochs')
        averaged=None;average_count=0
        if not math.isfinite(learning_rate) or not 0<learning_rate<=.05:raise ValueError('Invalid learning rate')
        x=features(rows)
        if not warm_start:self.mean=x.mean(0);self.scale=np.maximum(x.std(0),.1)
        elif not hasattr(self,'mean'):raise ValueError('Warm start requires a fitted model')
        for name,p in self.network.named_parameters():p.requires_grad_(not freeze_backbone or name.startswith('market.'))
        batch=self.batch(rows,paths);y=torch.tensor([r['y'] for r in rows],dtype=torch.float64)
        w=torch.tensor(weights(rows),dtype=torch.float64);winning=y==1
        ww=torch.tensor(weights([r for r in rows if r['y']==1]),dtype=torch.float64)
        optimizer=torch.optim.AdamW([p for p in self.network.parameters() if p.requires_grad],lr=learning_rate,weight_decay=.01)
        self.trace=[]
        for epoch in range(epochs):
            optimizer.zero_grad();weather,fused=self.network(*batch)
            weather_loss=-(ww*torch.log(weather[winning])).sum()
            loss=getattr(self,'weather_loss_weight',1.)*weather_loss
            if self.network.fusion:
                if getattr(self.network,'event_coherent',False):
                    loss=loss-(ww*torch.log(fused[winning].clamp(min=1e-12))).sum()
                else:
                    loss=loss+(w*torch.nn.functional.binary_cross_entropy(fused,y,reduction='none')).sum()
                # Penalize deviations from contemporaneous market odds, without
                # using labels to choose a city, price bucket, or gating rule.
                if getattr(self.network,'event_coherent',False):
                    reference=torch.zeros_like(fused)
                    for indices in batch[-1]:
                        p=batch[-3][indices].clamp(min=1e-6)
                        reference=reference.index_copy(0,indices,p/p.sum())
                    correction=torch.log(fused.clamp(min=1e-12))-torch.log(reference)
                else:
                    correction=torch.logit(fused)-torch.logit(batch[-2].clamp(1e-6,1-1e-6))
                penalized=torch.relu(correction.abs()-getattr(self,'penalty_deadzone',0.))
                loss=loss+getattr(self,'residual_penalty',0.)*(w*penalized.square()).sum()
            if not torch.isfinite(loss):raise ValueError('Nonfinite neural training loss')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in self.network.parameters()):raise ValueError('Nonfinite neural gradients')
            torch.nn.utils.clip_grad_norm_(self.network.parameters(),5);optimizer.step()
            if average_last and epoch>=epochs-average_last:
                state=self.network.state_dict();average_count+=1
                if averaged is None:averaged={k:v.detach().clone() for k,v in state.items()}
                else:
                    for k,v in state.items():averaged[k].add_((v.detach()-averaged[k])/average_count)
            if epoch%20==0 or epoch==epochs-1:self.trace.append(dict(epoch=epoch+1,loss=float(loss.detach())))
            if progress:progress(epoch+1,epochs)
        if averaged is not None:self.network.load_state_dict(averaged)
        self.averaged_checkpoints=average_count
        self.network.eval();return self
    def predict(self,rows,paths):
        with torch.no_grad():weather,fused=self.network(*self.batch(rows,paths))
        return (fused if self.network.fusion else weather).numpy()

class EventNeuralPredictor(NeuralPredictor):
    """Version 1 categorical fusion; requires full ladders before price filtering.

    Not interchangeable with legacy weights/benchmarks. No old validation search
    or automatic live promotion is implied by constructing this model.
    """
    architecture_version='event-softmax-v1'

    def __init__(self,calibration,seed=1729,weather_loss_weight=1.,residual_penalty=0.):
        super().__init__(calibration,fusion=True,seed=seed,weather_loss_weight=weather_loss_weight,residual_penalty=residual_penalty)
        self.network.event_coherent=True
        self.training_protocol=dict(architecture=self.architecture_version,
            objective='Date/event-balanced categorical NLL plus auxiliary weather interval NLL',
            reference='Per-event normalized market probabilities, floored at 1e-6 before log',
            residual_penalty='Mean squared log categorical probability ratio to normalized market reference',
            regularization='AdamW decoupled weight_decay=0.01; not equivalent to NumPy Adam coupled L2',
            coverage='Complete simultaneous event ladders only, shared weather inputs; price filtering after prediction',
            status='Correctness tested on synthetic fixtures; no fitted production artifact or performance claim')

    def groups(self,rows,paths,training=False):
        from collections import defaultdict
        from .event_probabilities import validate_event
        if not rows:raise ValueError('Empty event batch')
        paths=np.asarray(paths)
        if paths.shape!=(len(rows),2,16) or not np.isfinite(paths).all():raise ValueError('Invalid event forecast paths')
        groups=defaultdict(list)
        for i,r in enumerate(rows):groups[(r['event'],stamp(r['at']))].append(i)
        for indices in groups.values():
            rr=[rows[i] for i in indices];validate_event(rr)
            if training and (any(r.get('y') not in (0,1) for r in rr) or sum(r['y'] for r in rr)!=1):
                raise ValueError('Training event requires exactly one settled winning bracket')
            if len({r.get('group') for r in rr})!=1:raise ValueError('Inconsistent event date group')
            x=features(rr)
            if not np.all(x==x[0]):raise ValueError('Event weather features must be shared across brackets')
            for key in ('remaining_forecast_max','observed_max','observed','series'):
                if any(r.get(key)!=rr[0].get(key) for r in rr):raise ValueError('Event weather state must be shared across brackets')
            if not rr[0].get('observed'):raise ValueError('Event model requires available observations')
            if not np.all(paths[indices]==paths[indices[0]]):raise ValueError('Event forecast paths must be shared')
            if any(not math.isfinite(r['p']) or not 0<=r['p']<=1 or not math.isfinite(r['spread']) or not 0<=r['spread']<=1 for r in rr):raise ValueError('Invalid event market inputs')
            if sum(r['p'] for r in rr)<=0:raise ValueError('Event market reference has zero mass')
        return [torch.tensor(ix,dtype=torch.long) for ix in groups.values()]

    def batch(self,rows,paths):
        groups=self.groups(rows,paths)
        return super().batch(rows,paths)+(groups,)

    def fit(self,rows,paths,**kwargs):
        self.groups(rows,paths,training=True)
        if kwargs.get('warm_start'):raise ValueError('Event v1 does not accept legacy or warm-start training')
        return super().fit(rows,paths,**kwargs)
