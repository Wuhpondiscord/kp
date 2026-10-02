"""Visible epoch training with chronological train/validation/test partitions."""
from copy import deepcopy
from datetime import timedelta
import csv
import hashlib
import json
import math
import pickle
from pathlib import Path
import time
import uuid
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from .core import stamp, utcnow
from .evaluation import Predictor, features, score, validate_samples, paired_block_interval, quote_scenario, PROTOCOL
from .history import sample_rows
from .research_store import ResearchStore, digest
from .discovery import WEATHER_SERIES


DEFAULTS=dict(epochs=150,patience=20,batch_size=128,learning_rate=.001,alpha=1.0,architecture='32,16',seed=1729,method='adaptive')


def settings(config=None):
    config=config or {}
    if set(config)-set(DEFAULTS):raise ValueError('Unknown training setting')
    result=dict(DEFAULTS,**config)
    for key,low,high in [('epochs',5,500),('patience',3,100),('batch_size',16,1024),('seed',0,2147483647)]:
        value=float(result[key])
        if not math.isfinite(value) or not value.is_integer() or not low<=value<=high:
            raise ValueError(f'{key} must be an integer between {low} and {high}')
        result[key]=int(value)
    for key,low,high in [('learning_rate',.00001,.05),('alpha',.00001,100)]:
        value=float(result[key])
        if not math.isfinite(value) or not low<=value<=high:raise ValueError(f'{key} must be between {low} and {high}')
        result[key]=value
    if result['architecture'] not in ('16,8','32,16','64,32'):raise ValueError('Choose a supported network size')
    if result['method'] not in ('residual','legacy','adaptive'):raise ValueError('Choose adaptive, residual or legacy training')
    return result


def split_dataset(rows):
    validate_samples(rows)
    days=sorted({r['group'] for r in rows})
    if len(days)<45:raise ValueError('Training needs at least 45 date groups. Download real history first.')
    test_days=set(days[int(len(days)*.8):]);val_days=set(days[int(len(days)*.6):int(len(days)*.8)])
    test=[r for r in rows if r['group'] in test_days]
    test_cut=min(stamp(r['at']) for r in test)-timedelta(hours=48)
    # Purge entire dates, including their slowest-settling contract.
    latest={}
    for r in rows:
        day=r['group'];settled=stamp(r['settled_at'])
        latest[day]=max(latest.get(day,settled),settled)
    val_days={d for d in val_days if latest[d]<test_cut}
    val=[r for r in rows if r['group'] in val_days]
    if not val:raise ValueError('No validation dates remain after the settlement embargo')
    val_cut=min(stamp(r['at']) for r in val)-timedelta(hours=48)
    train_days={d for d in days[:int(len(days)*.6)] if latest[d]<val_cut}
    train=[r for r in rows if r['group'] in train_days]
    if len(train_days)<10 or len(val_days)<5:raise ValueError('Too few train/validation dates after embargo')
    return train,val,test


def partition_summary(rows):
    return dict(samples=len(rows),markets=len({r['ticker'] for r in rows}),days=len({r['group'] for r in rows}),
        start=min(r['group'] for r in rows),end=max(r['group'] for r in rows),yes_rate=sum(r['y'] for r in rows)/len(rows))


def fit_epochs(train,val,config,callback=None,cancel=None):
    """No test argument: checkpoint selection cannot consult test labels."""
    config=settings(config)
    scaler=StandardScaler().fit(features(train))
    X=scaler.transform(features(train));y=np.array([r['y'] for r in train])
    if len(set(y))<2:raise ValueError('Training data must contain both YES and NO outcomes')
    network=MLPClassifier(hidden_layer_sizes=tuple(map(int,config['architecture'].split(','))),
        solver='adam',alpha=config['alpha'],batch_size=min(config['batch_size'],len(train)),
        learning_rate_init=config['learning_rate'],shuffle=True,random_state=config['seed'],early_stopping=False)
    model=Predictor('neural_trained');model.model=make_pipeline(scaler,network)
    if config['method']!='legacy':
        from .residual import ResidualNetwork
        model=ResidualNetwork(train,config)
    history=[];best=None;best_loss=float('inf');best_epoch=0;wait=0
    with threadpool_limits(limits=2):
        for epoch in range(1,config['epochs']+1):
            if cancel and cancel.is_set():raise InterruptedError('Training cancelled; previous active model retained')
            if config['method']!='legacy':model.fit_epoch(train)
            else:network.partial_fit(X,y,classes=np.array([0,1]))
            a,b=score(train,model.predict(train)),score(val,model.predict(val))
            row=dict(epoch=epoch,train_loss=a['log_loss'],validation_loss=b['log_loss'],
                train_brier=a['brier'],validation_brier=b['brier'],optimizer_loss=float(model.loss_ if config['method']!='legacy' else network.loss_))
            history.append(row)
            meaningful=b['log_loss']<best_loss-1e-5
            if b['log_loss']<best_loss:
                best_loss=b['log_loss'];best_epoch=epoch;best=deepcopy(model)
            wait=0 if meaningful else wait+1
            if callback:callback(row,history,best_epoch,best)
            if wait>=config['patience']:break
    return best,history,best_epoch


def train_model(store,config=None,progress=None,cancel=None):
    rs=ResearchStore(store)
    rs.set('training',dict(status='preparing',message='Loading data and building time-ordered partitions',started_at=utcnow()))
    if progress:progress('Preparing dataset and checking timestamps',0,1)
    try:return _train_model(store,config,progress,cancel)
    except Exception as exc:
        record=rs.get('training') or {}
        record.update(status='cancelled' if isinstance(exc,InterruptedError) else 'failed',message=str(exc),finished_at=utcnow())
        rs.set('training',record)
        raise


def choose_weight(model,val):
    """Validation-only shrinkage, including the option to reject learned odds."""
    candidates=[]
    for weight in [0,.1,.25,.5,.75,1]:
        model.weight=weight
        candidates.append(dict(weight=weight,log_loss=score(val,model.predict(val))['log_loss']))
    model.weight=min(candidates,key=lambda c:c['log_loss'])['weight']
    return candidates


def segment_scores(rows,predictions):
    """Descriptive diagnostics only: never used to select checkpoints or gates."""
    from collections import defaultdict
    groups=defaultdict(list)
    for i,r in enumerate(rows):
        groups['City: '+r['series']].append(i)
        groups['Horizon: '+str(r.get('horizon',min((6,12,24),key=lambda h:abs(h-r['hours_left']))))+'h'].append(i)
    output=[]
    for name,indices in sorted(groups.items()):
        subset=[rows[i] for i in indices]
        learned=score(subset,[predictions[i] for i in indices]);baseline=score(subset,[r['p'] for r in subset])
        output.append(dict(segment=name,samples=len(subset),days=learned['days'],model=learned,baseline=baseline,
            log_loss_difference=learned['log_loss']-baseline['log_loss']))
    return output


def _train_model(store,config=None,progress=None,cancel=None):
    config=settings(config);rs=ResearchStore(store);run_id='train-'+uuid.uuid4().hex[:12]
    folder=store.root/'training'/run_id;folder.mkdir(parents=True)
    rows=sample_rows(store,horizons=(26,24,18,12,6,3));train,val,test=split_dataset(rows)
    from .coverage import training_coverage
    coverage=training_coverage((train,val,test))
    splits={k:partition_summary(v) for k,v in [('train',train),('validation',val),('test',test)]}
    splits['purged_samples']=len(rows)-len(train)-len(val)-len(test)
    record=dict(id=run_id,status='running',started_at=utcnow(),config=config,splits=splits,history=[],best_epoch=None,
        message='Training weights on the training partition; validation selects the checkpoint')
    rs.set('training',record)
    (folder/'config.json').write_text(json.dumps(config,indent=2))
    members={k:[dict(ticker=r['ticker'],at=r['at'],date=r['group']) for r in v] for k,v in [('train',train),('validation',val),('test',test)]}
    (folder/'split-manifest.json').write_text(json.dumps(dict(data_hash=digest(rows),partitions=members),indent=2))
    (folder/'features.jsonl').write_text('\n'.join(json.dumps(r,allow_nan=False) for r in rows),encoding='utf-8')
    from .environment import capture
    environment=capture()
    (folder/'environment.json').write_text(json.dumps(environment,indent=2),encoding='utf-8')
    start=time.monotonic()
    def update(row,history,best_epoch,best):
        record.update(epoch=row['epoch'],history=history,best_epoch=best_epoch,elapsed_seconds=round(time.monotonic()-start,2))
        rs.set('training',record)
        if row['epoch']==best_epoch:(folder/'best.pkl').write_bytes(pickle.dumps(best))
        if progress:progress(f"Epoch {row['epoch']}/{config['epochs']} · train {row['train_loss']:.4f} · validation {row['validation_loss']:.4f}",row['epoch'],config['epochs'])
    try:
        model,history,best_epoch=fit_epochs(train,val,config,update,cancel)
        if cancel and cancel.is_set():raise InterruptedError('Training cancelled before evaluation')
        record.update(status='evaluating',message='Weights finished; selecting blend on validation and scoring final test')
        rs.set('training',record)
        if progress:progress('Evaluating checkpoint and saving results',len(history),config['epochs'])
        if config['method']=='adaptive':
            from .adaptive import select_adaptive
            model,blend_candidates=select_adaptive(train,val,model,cancel)
        else:blend_candidates=choose_weight(model,val)
        if cancel and cancel.is_set():raise InterruptedError('Training cancelled during evaluation; previous active model retained')
        (folder/'best.pkl').write_bytes(pickle.dumps(model))
        train_score=score(train,model.predict(train));val_probs=model.predict(val);val_score=score(val,val_probs)
        val_baseline=score(val,[r['p'] for r in val])
        # The test is evaluated once, after checkpoint selection is complete.
        predicted=model.predict(test);baseline=[r['p'] for r in test]
        holdout=score(test,predicted);market_score=score(test,baseline)
        interval=paired_block_interval(test,predicted,baseline)
        lock=rs.get('holdout_lock');reused=bool(lock and splits['test']['start']<=lock['last_test_day'])
        passed=(not reused and holdout['days']>=30 and holdout['markets']>=100 and interval['upper']<0 and val_score['log_loss']<val_baseline['log_loss'])
        models=store.root/'models';models.mkdir(exist_ok=True);path=models/(run_id+'.pkl')
        trained_through=max(r['settled_at'] for r in train+val) # validation labels influence checkpoint choice
        path.write_bytes(pickle.dumps(dict(selected=model,experimental=model,trained_through=trained_through,series=coverage['series'],coverage=coverage)))
        sha=hashlib.sha256(path.read_bytes()).hexdigest()
        protocol=dict(PROTOCOL,candidates=['market','neural_checkpoint','beta_offset','context_offset','spline_offset','market_offset_4','market_offset_7'] if config['method']=='adaptive' else ['market','neural_checkpoint'],version='chronological-epoch-training-v4',horizons=[26,24,18,12,6,3],selection_metric='Validation daily log loss selects epoch; final test never selects weights',split='60/20/20 by date, whole-day purging and 48-hour settlement embargo',config=config,
            objective='Market log-odds offset plus neural correction; equal date weights' if config['method']!='legacy' else 'Legacy unweighted binary classification')
        diagnostics=dict(validation=segment_scores(val,val_probs),test=segment_scores(test,predicted))
        data_quality=dict(samples=len(rows),contracts=len({r['ticker'] for r in rows}),events=len({r['event'] for r in rows}),dates=len({r['group'] for r in rows}),
            warning='Repeated quotes and contracts from one event are correlated; sample count is not independent outcome count.')
        report=dict(id=run_id,created_at=utcnow(),synthetic=False,status='Forecast gate passed; execution unverified' if passed else 'No proven edge',
            promotion_passed=passed,selected_model=model.name,selected_label=(f'Adaptive selection: {model.label}' if config['method']=='adaptive' else f"Neural network · {config['method']} · {config['architecture']} · epoch {best_epoch}"),experimental_model=model.name,
            model_id=run_id,model_file=path.name,model_sha256=sha,source_hash=hashlib.sha256(b''.join(Path(__file__).with_name(n).read_bytes() for n in ('training.py','evaluation.py','residual.py','history.py','coverage.py','adaptive.py','station_model.py'))).hexdigest(),
            adaptive_candidates=blend_candidates if config['method']=='adaptive' else [],
            diagnostics=diagnostics,data_quality=data_quality,coverage=coverage,
            data_hash=digest(rows),protocol=protocol,candidates=[dict(name=model.name,label='Validation-selected model' if config['method']=='adaptive' else 'Best validation checkpoint',validation=val_score,folds=[],warnings=[]),dict(name='market',label='Market odds baseline',validation=val_baseline,folds=[],warnings=[])],
            development_days=splits['train']['days']+splits['validation']['days'],test_start=splits['test']['start'],test_end=splits['test']['end'],
            holdout_reused=reused,holdout=holdout,market_baseline=market_score,paired_interval=interval,training_last_settlement=trained_through,
            history_manifest=rs.get('history_manifest'),training=dict(config=config,splits=splits,history=history,best_epoch=best_epoch,train_score=train_score,validation_score=val_score,validation_baseline=val_baseline,blend_weight=model.weight,blend_candidates=blend_candidates),
            validation_predictions=[dict(r,model_probability=float(p)) for r,p in zip(val,val_probs)],
            forecasts=[dict(r,model_probability=float(p)) for r,p in zip(test,predicted)],
            scenarios=[quote_scenario(test,predicted,slippage=s,fee_multiplier=f) for s,f in ((0,1),(.02,1),(.05,2))],
            limits=['This neural model uses market prices, spread, horizon and city, not independent weather forecasts.',
                    'Residual/adaptive epochs train market-offset neural weights. Adaptive also compares regularized calibrators using validation log loss and a validation Brier constraint; the market baseline may win.',
                    'Validation selects weights; revisiting final-test dates blocks automatic promotion.',
                    'Dollar scenarios lack historical depth and cannot establish executable profit.'])
        if cancel and cancel.is_set():raise InterruptedError('Training cancelled before publication; previous active model retained')
        rs.save_experiment(report)
        rs.set('holdout_lock',dict(last_test_day=max(splits['test']['end'],lock['last_test_day'] if lock else ''),first_experiment=lock['first_experiment'] if lock else run_id))
        rs.set('active_model',dict(experiment_id=run_id,model_id=run_id,model_file=path.name,sha256=sha,passed=passed,
            selected=model.name,experimental=model.name,trained_through=trained_through,coverage=coverage))
        record.update(status='complete',elapsed_seconds=round(time.monotonic()-start,2),finished_at=utcnow(),message=f'Best weights saved from epoch {best_epoch}. Final test completed.',report_id=run_id,
            metrics=dict(train=train_score,validation=val_score,test=holdout),diagnostics=diagnostics,data_quality=data_quality,blend_weight=model.weight,early_stopped=len(history)<config['epochs'])
        rs.set('training',record)
        (folder/'report.json').write_text(json.dumps(report,indent=2))
        with (folder/'results.csv').open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(history[0]));writer.writeheader();writer.writerows(history)
        return report
    except Exception as exc:
        record.update(status='cancelled' if isinstance(exc,InterruptedError) else 'failed',message=str(exc),finished_at=utcnow())
        rs.set('training',record)
        raise
