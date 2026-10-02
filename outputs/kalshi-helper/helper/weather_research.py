"""Prospective remaining-day training; refuses insufficient/unsettled evidence."""
import json,math,pickle,hashlib
from pathlib import Path
from collections import Counter
from datetime import timedelta
import numpy as np
from .core import stamp
from .prospective import verify
from .weather_model import bounds,supported,STATIONS
from .event_weather import EventMaximum
from .evaluation import score,paired_block_interval

def collect_outcomes(store):
    from .kalshi import Kalshi,normalize_outcome
    from .prospective import Recorder
    from .core import utcnow
    import uuid
    tickers=set();settled=set()
    for path in store.root.glob('prospective/*/journal.jsonl'):
        verify(path)
        with path.open(encoding='utf-8') as handle:
            for line in handle:
                r=json.loads(line)
                if r['kind']=='weather_features':tickers.add(r['payload']['ticker'])
                if r['kind']=='outcome':settled.add(r['payload']['ticker'])
    pending=sorted(tickers-settled)
    if not pending:return dict(checked=0,settled=0)
    recorder=Recorder(store,'outcomes-'+uuid.uuid4().hex[:12],dict(purpose='Target-only settlement collection',tickers=pending))
    client=Kalshi(store);count=0;errors=[]
    for ticker in pending:
        try:
            result=normalize_outcome(client.market(ticker),utcnow())
            if result:recorder.append('outcome',result);count+=1
        except (OSError,ValueError,RuntimeError,KeyError) as exc:errors.append(dict(ticker=ticker,error=str(exc)))
    recorder.append('session_end',dict(settled=count,errors=errors))
    return dict(checked=len(pending),settled=count,errors=errors,evidence=str(recorder.folder))

class RemainingMaximum(EventMaximum):
    """Maximum model; external observation calibration or declared legacy 1F assumption.

    Product CDF assumes independence of observation discrepancy and remaining max.
    """
    def inputs(self,rows):
        g=np.asarray([r['remaining_forecast_max'] for r in rows],dtype=float)
        values=[]
        for r in rows:
            if not r['forecast_complete'] or not math.isfinite(r['remaining_forecast_max']):
                raise ValueError('Complete remaining forecast required')
            x=[1,r['remaining_hours']/24,*[float(r['series']==s) for s in list(STATIONS)[1:]]]
            for key,scale in [('forecast_revision_f',5),('mean_relative_humidity',100),('max_wind_mph',30),('mean_cloud_cover',100),('model_disagreement_f',5)]:
                v=r.get(key);x.extend([float(v)/scale if v is not None else 0,float(v is None)])
            for key in ('gfs_remaining_max','ecmwf_remaining_max'):
                v=r.get(key);x.extend([(float(v)-r['remaining_forecast_max'])/5 if v is not None else 0,float(v is None)])
            values.append(x)
        return g,np.asarray(values)

def build_dataset(root):
    """Only join weather/books available then; settlements are target-only fields."""
    candidates=[];outcomes={};counts=Counter();heads={}
    for path in sorted(Path(root).glob('prospective/*/journal.jsonl')):
        heads[str(path)]=verify(path)['head'];markets={};books={}
        with path.open(encoding='utf-8') as handle:
            for line in handle:
                record=json.loads(line);kind=record['kind'];r=record['payload']
                if kind=='market':markets[r['ticker']]=r
                elif kind=='book':books[r['ticker']]=r
                elif kind=='outcome':
                    if r['ticker'] in outcomes and outcomes[r['ticker']]['yes_payout']!=r['yes_payout']:raise ValueError('Conflicting settlements')
                    outcomes[r['ticker']]=r
                elif kind=='weather_features':
                    if r.get('poly_available') and r.get('poly_feature_version',1)<3:
                        # Older producers mixed observed spreads with missing-side bounds.
                        r=dict(r,poly_mean_quote_interval_width=r.get('poly_mean_spread'),poly_mean_spread=None)
                    ticker=r['ticker'];m=markets.get(ticker);b=books.get(ticker)
                    if not m or not b:counts['missing_market_or_book']+=1;continue
                    raw=m.get('raw_metadata',{});series=ticker.split('-')[0]
                    if not supported(raw,series):counts['unsupported_rules']+=1;continue
                    now=stamp(r['at'])
                    if r.get('poly_available') and (not r.get('poly_received_at') or stamp(r['poly_received_at'])>now):raise ValueError('Future or undated Polymarket feature')
                    if stamp(r['available_at'])>now or stamp(b['available_at'])>now:raise ValueError('Future feature join')
                    if (now-stamp(b['observed_at'])).total_seconds()>180:counts['stale_book']+=1;continue
                    yes=[float(p) for p,q in b['yes'] if float(q)>0];no=[float(p) for p,q in b['no'] if float(q)>0]
                    if not yes or not no:counts['empty_book']+=1;continue
                    bid=max(yes);ask=1-max(no)
                    if not 0<=bid<=ask<=1:counts['invalid_book']+=1;continue
                    lo,hi=bounds(raw)
                    candidates.append(dict(r,event=m['event_ticker'],series=series,group=r['event_start'][:10],
                        p=(bid+ask)/2,bid=bid,ask=ask,lower=lo,upper=hi,close_time=m['close_time'],
                        observed=r['observed_max'] is not None,source='prospective_weather',journal_head=heads[str(path)]))
    rows=[];seen=set()
    for r in candidates:
        outcome=outcomes.get(r['ticker'])
        if not outcome:counts['unsettled']+=1;continue
        if float(outcome['yes_payout']) not in (0,1):counts['nonbinary_settlement']+=1;continue
        if stamp(outcome['settled_at'])<=stamp(r['at']) or stamp(r['close_time'])<=stamp(r['at']):counts['already_closed']+=1;continue
        if not r['forecast_complete'] or r['forecast_age_hours'] is None or r['forecast_age_hours']>12:counts['incomplete_or_stale_forecast']+=1;continue
        # One sample per contract/UTC hour, avoiding poll-rate dependent weighting.
        key=(r['ticker'],stamp(r['at']).strftime('%Y-%m-%dT%H'))
        if key in seen:continue
        seen.add(key);rows.append(dict(r,y=int(float(outcome['yes_payout'])),settled_at=outcome['settled_at']))
    return rows,dict(excluded=dict(counts),journals=heads,rows=len(rows),events=len({r['event'] for r in rows}))

def partitions(rows):
    days=sorted({r['group'] for r in rows})
    if len(days)<90:raise ValueError('Need at least 90 settled date groups for remaining-day training')
    groups=[set(days[:int(.6*len(days))]),set(days[int(.6*len(days)):int(.8*len(days))]),set(days[int(.8*len(days)):])]
    parts=[[r for r in rows if r['group'] in g] for g in groups]
    for i in (1,0):
        cutoff=min(stamp(r['at']) for r in parts[i+1])-timedelta(hours=48)
        latest={d:max(stamp(r['settled_at']) for r in parts[i] if r['group']==d) for d in groups[i]}
        parts[i]=[r for r in parts[i] if latest[r['group']]<cutoff]
        if not parts[i]:raise ValueError('Embargo leaves empty partition')
    return parts

def train(root,output,observation_calibration=None):
    folder=Path(output);folder.mkdir(parents=True,exist_ok=False)
    protocol=dict(primary='12–24 hours remaining; market midpoint 5–95 cents',
        validation_weights=[0,.25,.5,1],embargo_hours=48,minimum_dates=90,
        selection='Daily validation log loss, Brier must not worsen; test never selects',
        observation_calibration=RemainingMaximum(observation_calibration).observation_protocol(),
        promotion=False,limitations='NWS plus two deterministic forecasts, not an ensemble; independent maximum components are an approximation. External full-day calibration may not transfer to running maxima. No automatic live integration.')
    (folder/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    rows,audit=build_dataset(root)
    (folder/'data-audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    primary=[r for r in rows if 12<=r['remaining_hours']<=24 and .05<=r['p']<=.95]
    safe=[dict(r,lower=None if math.isinf(r['lower']) else r['lower'],upper=None if math.isinf(r['upper']) else r['upper']) for r in rows]
    (folder/'features.jsonl').write_text('\n'.join(json.dumps(r,allow_nan=False) for r in safe),encoding='utf-8')
    try:
        training,validation,test=partitions(rows)
        # Fit the physical outcome distribution without selecting events by price.
        # Price filtering belongs only to the declared validation/test objective.
        validation=[r for r in validation if 12<=r['remaining_hours']<=24 and .05<=r['p']<=.95]
        test=[r for r in test if 12<=r['remaining_hours']<=24 and .05<=r['p']<=.95]
        if not validation or not test:raise ValueError('No primary-cohort validation or test rows')
    except ValueError as exc:
        report=dict(status='insufficient_data',reason=str(exc),primary_rows=len(primary),audit=audit)
        (folder/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');return report
    model=RemainingMaximum(observation_calibration).fit(training)
    baseline=score(validation,[r['p'] for r in validation]);v=model.predict(validation);choices=[]
    for weight in protocol['validation_weights']:
        metrics=score(validation,weight*v+(1-weight)*np.asarray([r['p'] for r in validation]))
        choices.append(dict(weight=weight,metrics=metrics))
    chosen=min((c for c in choices if c['metrics']['brier']<=baseline['brier']),key=lambda c:c['metrics']['log_loss'])
    (folder/'selection.json').write_text(json.dumps(dict(selected=chosen,candidates=choices),indent=2),encoding='utf-8')
    payload=pickle.dumps(model);(folder/'model.pkl').write_bytes(payload)
    probabilities=chosen['weight']*model.predict(test)+(1-chosen['weight'])*np.asarray([r['p'] for r in test])
    simple=RemainingMaximum(observation_calibration);_,x=simple.inputs(test);simple.theta=np.zeros(x.shape[1]+1);simple.theta[-1]=math.log(4)
    report=dict(status='research_only',weight=chosen['weight'],test=score(test,probabilities),
        market=score(test,[r['p'] for r in test]),simple_weather=score(test,simple.predict(test)),
        paired_interval=paired_block_interval(test,probabilities,[r['p'] for r in test]),
        split_dates=[[min(r['group'] for r in part),max(r['group'] for r in part)] for part in (training,validation,test)],
        artifact_sha256=hashlib.sha256(payload).hexdigest(),promotion=False,
        warning='Prospective recording alone does not establish untouched test dates; manual pre-registration required.')
    diagnostics={}
    for label,subset in [('late_hours',[r for r in rows if r['group']>=min(x['group'] for x in test) and r['remaining_hours']<12]),
                         ('extreme_prices',[r for r in rows if r['group']>=min(x['group'] for x in test) and (r['p']<.05 or r['p']>.95)])]:
        if subset:diagnostics[label]=dict(model=score(subset,chosen['weight']*model.predict(subset)+(1-chosen['weight'])*np.asarray([r['p'] for r in subset])),market=score(subset,[r['p'] for r in subset]))
    report['separate_diagnostics']=diagnostics
    (folder/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report
