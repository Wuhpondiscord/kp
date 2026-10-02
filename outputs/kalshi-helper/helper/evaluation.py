"""Chronological model selection, locked holdout, and conservative quote scenarios.

No future labels/features, no random train/test split, no candle OHLC fills,
and no claim that quoted historical prices prove executable trading returns.
"""
from collections import defaultdict
from dataclasses import dataclass
from datetime import timedelta
import hashlib
import json
import math
from pathlib import Path
import pickle
import warnings
import os
os.environ.setdefault('LOKY_MAX_CPU_COUNT','2')

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .core import dec, fee_and_cost, stamp, utcnow
from .discovery import WEATHER_SERIES
from .history import sample_rows
from .research_store import ResearchStore, digest


PROTOCOL = {
    'version':'chronological-weather-prices-v1', 'seed':1729,
    'candidates':['market','calibrated_market','beta_market','boosted_small','boosted_regularized','neural_small'],
    'embargo_hours':48, 'holdout_fraction':0.20, 'minimum_holdout_days':14,
    'bootstrap_repetitions':2000, 'bootstrap_block_days':7,
    'minimum_promotion_days':30, 'minimum_promotion_markets':100,
    'min_edge':0.04, 'horizons':[24,12,6],
    'selection_metric':'Mean daily log loss on expanding-window validation folds',
    'features':['logit_market','market_probability','market_variance','spread','hours_left','series_one_hot'],
}
LABELS = {'adaptive_market':'Adaptive calibration · validation checkpoint','market':'Market odds baseline','calibrated_market':'Calibrated market odds',
          'beta_market':'Beta-calibrated market odds',
          'neural_trained':'Neural network · validation checkpoint',
          'boosted_small':'Gradient boosting · small','boosted_regularized':'Gradient boosting · regularized',
          'neural_small':'Neural network · 16 / 8 units'}


def validate_samples(rows):
    seen, outcomes = set(), {}
    for r in rows:
        if r.get('synthetic') or r.get('source') != 'kalshi_hourly_candle':
            raise ValueError('Research evaluation requires real Kalshi candle rows')
        at=stamp(r['at'])
        if stamp(r['feature_time']) > at or stamp(r['settled_at']) <= at or stamp(r['close_time']) <= at:
            raise ValueError('Feature/settlement times would leak future information')
        if r['execution_at'] and stamp(r['execution_at']) <= at:
            raise ValueError('Execution scenario must use a later quote')
        for name in ('p','bid','ask','spread','hours_left'):
            if not math.isfinite(float(r[name])):
                raise ValueError('Non-finite feature')
        if not 0 <= r['bid'] <= r['ask'] <= 1 or r['y'] not in (0,1):
            raise ValueError('Invalid quote or outcome')
        if not 0<=r['p']<=1 or not math.isclose(r['p'],(r['bid']+r['ask'])/2,abs_tol=1e-8):
            raise ValueError('Market probability must match the source quote midpoint')
        if not math.isclose(r['spread'],r['ask']-r['bid'],abs_tol=1e-8):
            raise ValueError('Spread must match the source bid/ask quote')
        if r['hours_left']<=0:
            raise ValueError('Prediction horizon must be positive')
        key=(r['ticker'],r['at'])
        if key in seen:
            raise ValueError('Duplicate market/time sample')
        seen.add(key)
        if r['ticker'] in outcomes and outcomes[r['ticker']] != r['y']:
            raise ValueError('Conflicting labels for a market')
        outcomes[r['ticker']]=r['y']


def features(rows):
    values=[]
    for r in rows:
        p=float(np.clip(r['p'],0.001,0.999))
        values.append([math.log(p/(1-p)),p,p*(1-p),r['spread'],r['hours_left']/24,
                       *[float(r['series']==s) for s in WEATHER_SERIES]])
    return np.array(values,dtype=float)


class Predictor:
    def __init__(self, name):
        self.name=name
        self.model=None
        self.warnings=[]

    def fit(self, rows):
        if self.name=='market':
            return self
        X,y=features(rows),np.array([r['y'] for r in rows])
        if len(set(y)) < 2:
            raise ValueError('Training needs both outcomes')
        if self.name=='beta_market':
            p=np.clip([r['p'] for r in rows],.001,.999)
            X=np.column_stack([np.log(p),-np.log1p(-p)])
            self.model=LogisticRegression(C=.1,max_iter=1000,random_state=1729)
        elif self.name=='calibrated_market':
            self.model=LogisticRegression(C=0.1,max_iter=1000,random_state=1729)
            X=X[:,:1]
        elif self.name in ('boosted_small','boosted_regularized'):
            self.model=HistGradientBoostingClassifier(max_iter=90,learning_rate=0.04,
                max_leaf_nodes=7 if self.name=='boosted_small' else 4,
                min_samples_leaf=50,l2_regularization=5 if self.name=='boosted_small' else 15,
                early_stopping=False,random_state=1729)
        elif self.name=='neural_small':
            # No random early-stopping split. Fixed CPU-sized architecture and budget.
            self.model=make_pipeline(StandardScaler(),MLPClassifier(hidden_layer_sizes=(16,8),
                activation='relu',alpha=3.0,max_iter=250,early_stopping=False,
                shuffle=False,random_state=1729,tol=1e-5))
        else:
            raise ValueError('Unknown model')
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always',ConvergenceWarning)
            with threadpool_limits(limits=2):
                self.model.fit(X,y)
            self.warnings=[str(w.message) for w in caught]
        return self

    def predict(self, rows):
        market=np.array([r['p'] for r in rows],dtype=float)
        if self.name=='market':
            return market
        X=features(rows)
        if self.name=='calibrated_market':
            X=X[:,:1]
        if self.name=='beta_market':
            p=np.clip(market,.001,.999)
            X=np.column_stack([np.log(p),-np.log1p(-p)])
        with threadpool_limits(limits=2):
            fitted=self.model.predict_proba(X)[:,1]
        # Fixed shrinkage toward market odds; this blend is not tuned on holdout.
        weight=getattr(self,'weight',.5)
        return np.clip(weight*fitted+(1-weight)*market,0.0001,0.9999)


def day_mean(rows,values):
    grouped=defaultdict(list)
    for r,value in zip(rows,values):
        grouped[r['group']].append(float(value))
    return {day:float(np.mean(v)) for day,v in sorted(grouped.items())}


def losses(rows,probabilities):
    p=np.clip(np.asarray(probabilities),1e-8,1-1e-8)
    y=np.array([r['y'] for r in rows])
    return -(y*np.log(p)+(1-y)*np.log(1-p))


def score(rows,probabilities):
    y=np.array([r['y'] for r in rows])
    p=np.asarray(probabilities)
    calibration=[]
    for bucket in range(10):
        low=bucket/10
        indices=[i for i,v in enumerate(p) if min(9,int(float(v)*10))==bucket]
        if indices:
            calibration.append(dict(low=round(float(low),1),count=len(indices),
                predicted=float(np.mean(p[indices])),actual=float(np.mean(y[indices]))))
    return dict(log_loss=float(np.mean(list(day_mean(rows,losses(rows,p)).values()))),
        brier=float(np.mean(list(day_mean(rows,(p-y)**2).values()))),calibration=calibration,
        samples=len(rows),markets=len({r['ticker'] for r in rows}),events=len({r['event'] for r in rows}),
        days=len({r['group'] for r in rows}))


def paired_block_interval(rows, first, second, repetitions=2000, block=7):
    by_day=day_mean(rows,losses(rows,first)-losses(rows,second))
    values=np.array(list(by_day.values()))
    if len(values)<2:
        return dict(mean=float(np.mean(values)) if len(values) else None,lower=None,upper=None,days=len(values))
    rng=np.random.default_rng(1729)
    estimates=[]
    # Circular moving blocks retain local date dependence and all same-day markets.
    for _ in range(repetitions):
        indices=[]
        while len(indices)<len(values):
            start=int(rng.integers(0,len(values)))
            indices.extend((start+j)%len(values) for j in range(min(block,len(values))))
        estimates.append(float(np.mean(values[indices[:len(values)]])))
    return dict(mean=float(np.mean(values)),lower=float(np.quantile(estimates,0.025)),
                upper=float(np.quantile(estimates,0.975)),days=len(values),block_days=block,
                interpretation='Negative favors the selected model. Exploratory interval; not proof of tradable profit.')


def chronological_plan(rows):
    validate_samples(rows)
    days=sorted({r['group'] for r in rows})
    if len(days)<45:
        raise ValueError(f'Need at least 45 calendar-day groups; currently {len(days)}')
    holdout_count=max(14,math.ceil(len(days)*0.2))
    development_days=days[:-holdout_count]
    test_days=set(days[-holdout_count:])
    test=[r for r in rows if r['group'] in test_days]
    boundary=min(stamp(r['at']) for r in test)-timedelta(hours=48)
    development=[r for r in rows if r['group'] not in test_days and stamp(r['settled_at']) < boundary]
    usable_days=sorted({r['group'] for r in development})
    first=max(15,int(len(usable_days)*0.4))
    edges=np.linspace(first,len(usable_days),4,dtype=int)
    folds=[]
    for begin,end in zip(edges[:-1],edges[1:]):
        val_days=set(usable_days[begin:end])
        val=[r for r in development if r['group'] in val_days]
        if not val:
            continue
        cutoff=min(stamp(r['at']) for r in val)-timedelta(hours=48)
        train=[r for r in development if r['group'] < usable_days[begin] and stamp(r['settled_at']) < cutoff]
        if len({r['group'] for r in train}) < 10:
            continue
        folds.append((train,val))
    if len(folds)<3:
        raise ValueError('Not enough data for three purged chronological validation folds')
    return development,test,folds


def quote_scenario(rows, probabilities, bankroll='1000', slippage=0.02, fee_multiplier=1.0, execution_mode='later_quote', decision_qualified=False, min_edge=.04):
    """Indicative 1-contract entries at LATER hourly quotes; never a fill replay.

    Empty depth means a verified fill count cannot be supplied. Fixed quantity,
    conservative missing-quote skips, event caps, and shared cash still apply.
    """
    if execution_mode not in ('later_quote','decision_quote'):
        raise ValueError('Unknown execution scenario')
    if not math.isfinite(min_edge) or not 0<=min_edge<1:raise ValueError('Invalid minimum edge')
    if execution_mode=='decision_quote':
        rows=[dict(r,execution_at=r['at'],execution_bid=r['bid'],execution_ask=r['ask']) for r in rows]
    cash=dec(bankroll);initial=cash
    positions={};seen=set();trades=[];curve=[];fees=dec(0)
    probability={ (r['ticker'],r['at']):float(p) for r,p in zip(rows,probabilities)}
    events=[]
    for r in rows:
        events.append((stamp(r['settled_at']),0,r))
        if r['execution_at']:
            events.append((stamp(r['execution_at']),1,r))
    for at,kind,r in sorted(events,key=lambda x:(x[0],x[1],x[2]['ticker'])):
        ticker=r['ticker']
        if kind==0:
            pos=positions.pop(ticker,None)
            if pos:
                proceeds=dec(r['y'] if pos['side']=='yes' else 1-r['y'])
                cash+=proceeds
                trades.append(dict(ticker=ticker,event=r['event'],at=at.isoformat(),
                    side=pos['side'],cost=str(pos['cost']),payout=str(proceeds),pnl=str(proceeds-pos['cost'])))
                curve.append(dict(at=at.isoformat(),cash=str(cash),open_cost=str(sum((p['cost'] for p in positions.values()),dec(0)))))
            continue
        if ticker in seen or at >= stamp(r['close_time']) or at >= stamp(r['settled_at']):
            continue
        p=probability[(ticker,r['at'])]
        initial_choices=[(p-r['ask'],'yes',p),(1-p-(1-r['bid']),'no',1-p)]
        _,side,win=max(initial_choices)
        if decision_qualified:
            choices=[]
            for choice,prob,quote in (('yes',p,r['ask']),('no',1-p,1-r['bid'])):
                _,debit=fee_and_cost(dec(quote),1,dec('0.07')*dec(fee_multiplier))
                choices.append((prob-float(debit),choice,prob))
            edge,side,win=max(choices)
            if edge<min_edge:continue
        quoted=r['execution_ask'] if side=='yes' else 1-r['execution_bid']
        price=dec(min(1,quoted+slippage))
        fee,cost=fee_and_cost(price,1,dec(0.07*fee_multiplier))
        if win-float(cost)<min_edge:
            continue
        event_cost=sum((v['cost'] for v in positions.values() if v['event']==r['event']),dec(0))
        total_cost=sum((v['cost'] for v in positions.values()),dec(0))
        if cost>cash or event_cost+cost>initial*dec('.03') or total_cost+cost>initial*dec('.20'):
            continue
        cash-=cost;fees+=fee;seen.add(ticker)
        positions[ticker]=dict(side=side,cost=cost,event=r['event'])
    return dict(label=('Same-quote optimistic counterfactual — fills unverified' if execution_mode=='decision_quote' else 'Delayed stale-signal scenario — fills unverified'),execution_mode=execution_mode,
        decision_qualified=decision_qualified,minimum_edge=min_edge,
        signal_policy=('Decision-time net edge required, side frozen, edge rechecked at execution' if decision_qualified else 'Prediction fixed at decision time; edge rechecked at assumed execution price'),bankroll=str(initial),
        ending_cash=str(cash),pnl=str(cash-initial),fees=str(fees),trades=len(trades),
        assumed_quantity=1,slippage_cents=slippage*100,fee_multiplier=fee_multiplier,
        missing_next_quotes=sum(not r['execution_at'] for r in rows),verified_fills=None,
        ledger=trades,curve=curve)


def run_evaluation(store, progress=None):
    rs=ResearchStore(store)
    rows=sample_rows(store)
    data_hash=digest(rows)
    source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    experiment_id=digest(dict(data=data_hash,protocol=PROTOCOL,code=source_hash))[:20]
    cached=rs.experiment(experiment_id)
    if cached:
        rs.set('latest_experiment',experiment_id)
        rs.set('active_model',dict(experiment_id=experiment_id,model_id=cached['model_id'],model_file=cached['model_file'],
            sha256=cached['model_sha256'],passed=cached['promotion_passed'],selected=cached['selected_model'],
            experimental=cached['experimental_model'],trained_through=cached['training_last_settlement']))
        if progress:progress('Loaded cached model comparison; use Train new model for fresh epoch training',1,1)
        return cached
    development,test,folds=chronological_plan(rows)
    lock=rs.get('holdout_lock')
    reused=bool(lock and min(r['group'] for r in test) <= lock['last_test_day'])
    candidates=[]
    for i,name in enumerate(PROTOCOL['candidates']):
        if progress:
            progress('Comparing '+LABELS[name],i,len(PROTOCOL['candidates'])+1)
        fold_reports=[];all_rows=[];all_predictions=[];candidate_warnings=[]
        for train,val in folds:
            model=Predictor(name).fit(train)
            probabilities=model.predict(val)
            metrics=score(val,probabilities)
            fold_reports.append(dict(train_days=len({r['group'] for r in train}),
                train_last_settlement=max(r['settled_at'] for r in train),
                validation_first_prediction=min(r['at'] for r in val),
                validation_start=min(r['group'] for r in val),validation_end=max(r['group'] for r in val),
                log_loss=metrics['log_loss']))
            all_rows.extend(val);all_predictions.extend(probabilities)
            candidate_warnings.extend(model.warnings)
        metrics=score(all_rows,all_predictions)
        candidates.append(dict(name=name,label=LABELS[name],validation=metrics,folds=fold_reports,
                               warnings=list(set(candidate_warnings))))
    winner=min(candidates,key=lambda c:c['validation']['log_loss'])
    challenger=min([c for c in candidates if c['name']!='market'],key=lambda c:c['validation']['log_loss'])
    selected=Predictor(winner['name']).fit(development)
    experimental=Predictor(challenger['name']).fit(development)
    predicted=selected.predict(test)
    market=np.array([r['p'] for r in test])
    interval=paired_block_interval(test,predicted,market)
    selected_score=score(test,predicted)
    market_score=score(test,market)
    passed=(winner['name']!='market' and interval['upper'] is not None and interval['upper']<0 and
            selected_score['days']>=PROTOCOL['minimum_promotion_days'] and
            selected_score['markets']>=PROTOCOL['minimum_promotion_markets'] and not reused)
    status='Forecast improvement detected; execution still unverified' if passed else 'No proven edge'
    model_id=f"price-model-{experiment_id}"
    folder=store.root/'models';folder.mkdir(exist_ok=True)
    model_path=folder/(model_id+'.pkl')
    # Only locally generated artifacts are loaded; no user-supplied pickle endpoint.
    model_path.write_bytes(pickle.dumps(dict(selected=selected,experimental=experimental,
        trained_through=max(r['settled_at'] for r in development),series=WEATHER_SERIES)))
    artifact_hash=hashlib.sha256(model_path.read_bytes()).hexdigest()
    forecasts=[dict(r,model_probability=float(p)) for r,p in zip(test,predicted)]
    report=dict(id=experiment_id,created_at=utcnow(),synthetic=False,status=status,promotion_passed=passed,
        selected_model=winner['name'],selected_label=winner['label'],experimental_model=challenger['name'],
        model_id=model_id,model_file=model_path.name,model_sha256=artifact_hash,source_hash=source_hash,
        data_hash=data_hash,protocol=PROTOCOL,candidates=candidates,
        development_days=len({r['group'] for r in development}),
        test_start=min(r['group'] for r in test),test_end=max(r['group'] for r in test),
        holdout_reused=reused,holdout=selected_score,market_baseline=market_score,paired_interval=interval,
        training_last_settlement=max(r['settled_at'] for r in development),
        history_manifest=rs.get('history_manifest'),forecasts=forecasts,
        scenarios=[quote_scenario(test,predicted,slippage=s,fee_multiplier=f) for s,f in ((0,1),(.02,1),(.05,2))],
        limits=['Models currently use market prices, spread, time remaining and series; no weather forecasts or text embeddings yet.',
            'Hourly quotes cannot establish execution profit. All displayed dollar backtests are explicitly assumption-based.',
            'Current general fee coefficient is an assumption for history, not a reconstructed historical fee schedule.',
            'Final test is not used for model or parameter selection. Repeated/revised holdouts cannot auto-promote.',
            'Small sample, source changes, survivorship and missing snapshots can invalidate apparent edge.'])
    rs.save_experiment(report)
    rs.set('holdout_lock',dict(last_test_day=max(r['group'] for r in test),first_experiment=lock['first_experiment'] if lock else experiment_id))
    rs.set('active_model',dict(experiment_id=experiment_id,model_id=model_id,model_file=model_path.name,
        sha256=artifact_hash,passed=passed,selected=winner['name'],experimental=challenger['name'],
        trained_through=report['training_last_settlement']))
    return report


@dataclass(frozen=True)
class PredictionResult:
    probability: float
    reason: str
    signal: bool = False
    model_id: str | None = None


class ModelService:
    def __init__(self, store, active_model=None):
        self.store=store;self.loaded_id=None;self.bundle=None;self.active_override=active_model

    def predict(self, series, bid, ask, at, close_time, experimental=False):
        result=self.predict_signal(series,bid,ask,at,close_time,experimental)
        return result.probability,result.reason

    def predict_signal(self, series, bid, ask, at, close_time, experimental=False):
        bid,ask=float(bid),float(ask)
        if not (math.isfinite(bid) and math.isfinite(ask) and 0<=bid<=ask<=1):
            raise ValueError('Model inference requires finite, uncrossed dollar quotes')
        active=self.active_override if self.active_override is not None else ResearchStore(self.store).get('active_model')
        market=(bid+ask)/2
        if not active:
            return PredictionResult(market,'Market odds — model not trained')
        hours=(stamp(close_time)-stamp(at)).total_seconds()/3600
        from .coverage import coverage_issue
        issue=coverage_issue(active,series,at,close_time)
        if issue:return PredictionResult(market,issue)
        if stamp(at)<=stamp(active['trained_through']):
            return PredictionResult(market,'Prediction predates the model’s training cutoff')
        if not active['passed'] and not experimental:
            return PredictionResult(market,'Auto mode: no validated model advantage')
        if self.loaded_id != active['model_id']:
            path=self.store.root/'models'/active['model_file']
            if path.parent.resolve()!=(self.store.root/'models').resolve():
                raise ValueError('Invalid model path')
            payload=path.read_bytes()
            if hashlib.sha256(payload).hexdigest()!=active['sha256']:
                raise ValueError('Model artifact hash mismatch')
            self.bundle=pickle.loads(payload)
            self.loaded_id=active['model_id']
        row=dict(p=market,spread=ask-bid,hours_left=hours,series=series)
        predictor=self.bundle['experimental' if experimental else 'selected']
        if getattr(predictor,'weight',.5)==0:
            return PredictionResult(market,'Market odds — validation rejected the learned model contribution')
        if predictor.name=='market':return PredictionResult(market,'Market odds baseline')
        probability=float(predictor.predict([row])[0])
        if not math.isfinite(probability) or not 0<=probability<=1:
            raise ValueError('Model returned an invalid probability')
        return PredictionResult(probability,('Experimental · ' if experimental else '')+LABELS[predictor.name],True,active['model_id'])
