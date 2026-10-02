"""Download station-sized run archives, join frozen splits, fit on development only."""
import json,hashlib,sqlite3,time,pickle,math
from pathlib import Path
from collections import Counter
import numpy as np
from helper.forecast_archive import fetch_run,archived_feature
from helper.weather_model import event_date,bounds,supported
from helper.station_observations import ObservationIndex,SITES
from helper.training import split_dataset
from helper.weather_research import RemainingMaximum
from helper.observation_calibration import calibrate
from helper.spread_research import calibration_metrics
from helper.evaluation import paired_block_interval
from helper.core import stamp

def run(root,features,output,observation_audit,offline=False):
    root=Path(root).resolve();folder=Path(output);features=Path(features)
    folder.mkdir(parents=True,exist_ok=True)
    allrows=[json.loads(s) for s in features.read_text(encoding='utf-8').splitlines()]
    parts=split_dataset(allrows);groups={r['group']:name for name,part in zip(('train','validation','reserved_test'),parts) for r in part}
    candidates=[r for r in allrows if r['horizon']==12 and event_date(r['ticker']).date().isoformat()>='2026-04-02' and r['group'] in groups]
    protocol=dict(source='Open-Meteo single runs, gfs_global + ecmwf_ifs025',horizon='Original nominal 12h candle only',
        run_cycle='06 UTC on event date',availability_allowance_hours=8,
        splits='Original full price-feature split membership retained BEFORE archive/date/availability filtering',
        test_evaluated=False,promotion=False,feature_sha256=hashlib.sha256(features.read_bytes()).hexdigest(),
        observation_delay_minutes=30,baseline='Mean of two model remaining maxima, not NWS',
        warning='Historical dates previously inspected; availability allowance assumed. Retrospective exploratory weather study, not live tradability evidence.')
    (folder/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    days=sorted({event_date(r['ticker']).date().isoformat() for r in candidates})
    records={};errors={}
    for i,day in enumerate(days):
        try:records[day]=fetch_run(root,day,offline=offline)
        except (OSError,ValueError) as e:errors[day]=str(e)
        if i%10==0:print(f'Archive {i+1}/{len(days)}, usable {len(records)}, errors {len(errors)}',flush=True)
    caches={}
    for path in sorted((root/'station-observations').glob('*.json')):
        d=json.loads(path.read_text(encoding='utf-8'));s=d['station']
        if s not in caches or len(d['observations'])>len(caches[s][1]['observations']):caches[s]=(path,d)
    indices={s:ObservationIndex(d['observations']) for s,(_,d) in caches.items()}
    db=sqlite3.connect((root/'research.sqlite3').as_uri()+'?mode=ro',uri=True)
    markets={t:json.loads(m) for t,m in db.execute('select ticker,market from history_markets')};db.close()
    selected={k:[] for k in ('train','validation','reserved_test')};excluded=Counter()
    for row in candidates:
        day=event_date(row['ticker']).date().isoformat()
        try:
            if not supported(markets[row['ticker']],row['series']):raise ValueError('Unsupported settlement')
            f=archived_feature(records[day],row['series'],row['ticker'],row['at'])
            obs=indices[SITES[row['series']]].feature(row['series'],row['ticker'],row['at'])
            low,high=bounds(markets[row['ticker']])
            selected[groups[row['group']]].append(dict(row,lower=low,upper=high,**f,**obs))
        except (ValueError,KeyError) as e:excluded[str(e)]+=1
    counts={k:dict(rows=len(v),events=len({r['event'] for r in v}),dates=len({r['group'] for r in v})) for k,v in selected.items()}
    audit=dict(splits=counts,download_errors=errors,excluded=dict(excluded),archive_runs=len(records),
        station_cache_hashes={s:hashlib.sha256(p.read_bytes()).hexdigest() for s,(p,_) in caches.items()})
    (folder/'data-audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    for name,rows in selected.items():
        safe=[dict(r,lower=None if math.isinf(r['lower']) else r['lower'],upper=None if math.isinf(r['upper']) else r['upper']) for r in rows]
        (folder/(name+'.jsonl')).write_text('\n'.join(json.dumps(r,allow_nan=False) for r in safe),encoding='utf-8')
    training=selected['train'];validation=[r for r in selected['validation'] if .05<=r['p']<=.95]
    if len({r['event'] for r in training})<80 or len({r['group'] for r in validation})<10:
        report=dict(status='insufficient_data',audit=audit,test_evaluated=False)
    else:
        # Entire external calibration precedes the first training decision.
        previous=json.loads(Path(observation_audit).read_text(encoding='utf-8'))
        first=min(stamp(r['at']) for r in training)
        pairs=[r for r in previous['pairs'] if stamp(r['settled_at'])<first]
        calibration=calibrate(pairs,previous['calibration']['source_hashes']);calibration['pooling']='station'
        (folder/'observation-calibration.json').write_text(json.dumps(calibration,indent=2),encoding='utf-8')
        predictions={};metrics={};models={}
        for arm in ('constant','disagreement'):
            print('Fitting '+arm,flush=True)
            model=RemainingMaximum(calibration,remaining_spread=arm).fit(training)
            payload=pickle.dumps(model);(folder/(arm+'.pkl')).write_bytes(payload)
            predictions[arm]=model.predict(validation);metrics[arm]=calibration_metrics(validation,predictions[arm])
            models[arm]=dict(sha256=hashlib.sha256(payload).hexdigest(),theta=model.theta.tolist())
        metrics['market']=calibration_metrics(validation,[r['p'] for r in validation])
        report=dict(status='retrospective_validation_only',audit=audit,validation=metrics,models=models,
            paired_log_loss=paired_block_interval(validation,predictions['disagreement'],predictions['constant']),
            test_evaluated=False,promotion=False,live_model_changed=False,profitability_tested=False,
            warning=protocol['warning'])
    (folder/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(status=report['status'],audit=audit),indent=2),flush=True)
    return report


