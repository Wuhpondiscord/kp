"""Experimental forecast post-processing using archived GFS and resolved brackets.

This is interval-censored Gaussian MOS, not ensemble EMOS: GFS supplies a
deterministic forecast, and winning Kalshi brackets supply interval labels.
Fixed-lead archives infer availability with a six-hour publication allowance.
They cannot establish original ingestion times, so automatic promotion is off.
"""
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import numpy as np
from scipy.optimize import minimize
from scipy.special import ndtr, expit, logit
from .core import stamp, utcnow
from .storage import Store
from .research_store import ResearchStore, digest
from .history import sample_rows
from .training import split_dataset, partition_summary
from .evaluation import score, paired_block_interval

# Station coordinates, plus fixed standard UTC offset (not civil DST).
STATIONS = {'KXHIGHNY':(40.779,-73.969,5), 'KXHIGHCHI':(41.7841,-87.7551,6),
            'KXHIGHMIA':(25.791,-80.316,5), 'KXHIGHDEN':(39.856,-104.673,7)}
VARIABLE='temperature_2m_previous_day2'
MONTHS={m:i+1 for i,m in enumerate(['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'])}

def event_date(ticker):
    part=ticker.split('-')[1]
    return datetime(2000+int(part[:2]),MONTHS[part[2:5]],int(part[5:7]),tzinfo=timezone.utc)

def bounds(market):
    kind=market.get('strike_type');lo=market.get('floor_strike');hi=market.get('cap_strike')
    if kind=='between' and lo is not None and hi is not None and float(lo)<=float(hi):
        return float(lo)-.5,float(hi)+.5
    if kind=='greater' and lo is not None:return float(lo)+.5,math.inf
    if kind=='less' and hi is not None:return -math.inf,float(hi)-.5
    raise ValueError('Unsupported temperature strike or rounding rule')

def supported(market,series):
    # Some contracts identify the station only in secondary settlement rules.
    text=' '.join(str(market.get(k) or '') for k in ('rules_primary','rules_secondary','rules')).lower()
    station_tokens={'KXHIGHNY':('central park','clinyc'),'KXHIGHCHI':('midway','climdw'),'KXHIGHMIA':('miami','climia'),'KXHIGHDEN':('denver','cliden')}
    if not any(token in text for token in station_tokens.get(series,())):return False
    return series in STATIONS and ('highest temperature' in text or 'maximum temperature' in text) and (
        'climatological report' in text or 'weather company' in text)

def fetch_hours(store,series,start,end,lead_days=2,model='gfs_seamless'):
    if lead_days not in (1,2):raise ValueError('Use a supported archived lead')
    variable=f'temperature_2m_previous_day{lead_days}'
    lat,lon,_=STATIONS[series]
    params=dict(latitude=lat,longitude=lon,start_date=start,end_date=end,hourly=variable,
                models=model,temperature_unit='fahrenheit',timezone='GMT')
    url='https://previous-runs-api.open-meteo.com/v1/forecast?'+urlencode(params)
    rs=ResearchStore(store);key='gfs:'+hashlib.sha256(url.encode()).hexdigest()
    cached=rs.get(key)
    # Historical payload is reused; today/future responses expire after one hour.
    if cached and (end<utcnow()[:10] or 0 <= (stamp(utcnow())-stamp(cached['at'])).total_seconds()<3600):
        return cached['hours']
    for attempt in range(3):
        try:
            with urlopen(Request(url,headers={'User-Agent':'BetCheck/0.5 local noncommercial research'}),timeout=25) as response:raw=response.read()
            store.raw(url,raw);body=json.loads(raw)
            if body.get('hourly_units',{}).get(variable)!='°F':raise ValueError('Forecast unit must be Fahrenheit')
            times=body['hourly']['time'];values=body['hourly'][variable]
            if len(times)!=len(values) or len(set(times))!=len(times):raise ValueError('Invalid forecast timestamps')
            hours={t:float(v) for t,v in zip(times,values) if v is not None and math.isfinite(float(v)) and -100<float(v)<150}
            rs.set(key,dict(at=utcnow(),hours=hours,url=url));return hours
        except (OSError,ValueError,KeyError):
            if attempt==2:raise
            time.sleep(attempt+1)

def forecast_feature(series,ticker,hours,decision,lead_days=2):
    if lead_days not in (1,2):raise ValueError('Use a supported archived lead')
    begin=event_date(ticker)+timedelta(hours=STATIONS[series][2])
    valid=[begin+timedelta(hours=i) for i in range(24)]
    available=valid[-1]-timedelta(hours=24*lead_days)+timedelta(hours=6)
    if available>stamp(decision):raise ValueError('Forecast was not available at the decision time')
    values=[hours.get(t.strftime('%Y-%m-%dT%H:%M')) for t in valid]
    if any(v is None for v in values):raise ValueError('Incomplete 24-hour forecast window')
    return dict(forecast_high=max(values),weather_lead_days=lead_days,weather_available_at=available.isoformat(),
                weather_day=begin.date().isoformat())

def design(rows):
    out=[]
    for r in rows:
        day=datetime.fromisoformat(r['weather_day']).timetuple().tm_yday
        out.append([1,(r['forecast_high']-65)/20,math.sin(2*math.pi*day/365.25),math.cos(2*math.pi*day/365.25),
                    *[float(r['series']==s) for s in list(STATIONS)[1:]]])
    return np.asarray(out)

def probability(rows,parameters):
    mu=np.array([r['forecast_high'] for r in rows])+design(rows)@np.asarray(parameters[:-1])
    sigma=math.exp(parameters[-1])
    low=np.array([r['lower'] for r in rows]);high=np.array([r['upper'] for r in rows])
    return np.clip(ndtr((high-mu)/sigma)-ndtr((low-mu)/sigma),1e-6,1-1e-6)

def fit_distribution(train):
    # One winning interval per event; never treat six mutually exclusive brackets
    # or three horizons as independent observed temperatures.
    winners={r['event']:r for r in train if r['y']==1}
    if len(winners)<80:raise ValueError('Weather fitting requires at least 80 resolved winning events')
    rows=list(winners.values())
    initial=np.zeros(design(rows).shape[1]+1);initial[-1]=math.log(4)
    def objective(theta):return float(-np.log(probability(rows,theta)).mean()+.002*np.square(theta[:-1]).sum())
    result=minimize(objective,initial,method='L-BFGS-B',bounds=[(-15,15)]*(len(initial)-1)+[(math.log(.5),math.log(20))])
    if not result.success:raise ValueError('Weather optimizer did not converge: '+str(result.message))
    return result.x.tolist(),len(rows)

def blend(weather,market,weight):
    if weight==0:return np.asarray(market,dtype=float)
    return expit(weight*logit(np.clip(weather,1e-6,1-1e-6))+(1-weight)*logit(np.clip(market,1e-6,1-1e-6)))

def train_weather_legacy(store,progress=None):
    rs=ResearchStore(store)
    def update(message,done=0,total=5):
        rs.set('weather_training',dict(status='running',message=message,at=utcnow()))
        if progress:progress(message,done,total)
    update('Loading resolved temperature brackets')
    try:
        history=rs.history();markets={x['market']['ticker']:x['market'] for x in history}
        source=sample_rows(store);rows=[];excluded=0
        for i,series in enumerate(STATIONS):
            group=[r for r in source if r['series']==series]
            if not group:continue
            dates=[event_date(r['ticker']) for r in group]
            update('Downloading archived GFS forecasts for '+series,i,5)
            hours=fetch_hours(store,series,min(dates).date().isoformat(),(max(dates)+timedelta(days=1)).date().isoformat())
            for r in group:
                try:
                    market=markets[r['ticker']]
                    if not supported(market,series):raise ValueError('Unsupported settlement rules')
                    low,high=bounds(market)
                    rows.append(dict(r,lower=low,upper=high,**forecast_feature(series,r['ticker'],hours,r['at'])))
                except (KeyError,ValueError):excluded+=1
        train,val,test=split_dataset(rows)
        update('Fitting forecast bias and temperature uncertainty',4,5)
        parameters,event_count=fit_distribution(train)
        val_weather=probability(val,parameters);test_weather=probability(test,parameters)
        weights=[0,.1,.25,.5,.75,1]
        candidates=[dict(weight=w,validation=score(val,blend(val_weather,[r['p'] for r in val],w))) for w in weights]
        best=min(candidates,key=lambda c:c['validation']['log_loss'])
        predictions=blend(test_weather,[r['p'] for r in test],best['weight'])
        safe_rows=[dict(r,lower=None if not math.isfinite(r['lower']) else r['lower'],upper=None if not math.isfinite(r['upper']) else r['upper']) for r in rows]
        report=dict(status='complete',at=utcnow(),model='Interval-censored Gaussian GFS post-processing',
            parameters=parameters,weight=best['weight'],winning_training_events=event_count,excluded_samples=excluded,
            splits={k:partition_summary(v) for k,v in [('train',train),('validation',val),('test',test)]},
            candidates=candidates,holdout=score(test,predictions),weather_only=score(test,test_weather),
            baseline=score(test,[r['p'] for r in test]),paired_interval=paired_block_interval(test,predictions,[r['p'] for r in test]),
            data_hash=digest(safe_rows),trained_through=max(r['settled_at'] for r in train+val),passed=False,
            source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            protocol=dict(forecast='GFS seamless previous_day2',availability_allowance_hours=6,
                stations=STATIONS,rounding='Integer Fahrenheit bounds expanded by half a degree',
                weights=weights,selection='Validation daily mean log loss',test_reused=True),
            limits=['Exploratory reused test dates; no automatic promotion.',
                'GFS fixed-lead availability is inferred with a six-hour allowance, not original ingestion timestamps.',
                'Hourly gridpoint maxima are proxies for station maxima; bias and variance are learned from winning settlement intervals.',
                'Station/day and settlement-provider changes require prospective verification.',
                'Forecast weight selected only on validation; a zero weight means weather added no validated benefit.'])
        rs.set('weather_model',report);rs.set('weather_training',dict(status='complete',message='Weather fit and final test complete',at=utcnow()))
        folder=store.root/'weather-model';folder.mkdir(exist_ok=True)
        (folder/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        (folder/'features.jsonl').write_text('\n'.join(json.dumps(r,allow_nan=False) for r in safe_rows),encoding='utf-8')
        return report
    except Exception as exc:
        rs.set('weather_training',dict(status='failed',message=str(exc),at=utcnow()));raise

def train_weather(store,progress=None):
    from .weather_experiment import train_weather_experiment
    return train_weather_experiment(store,progress)


if __name__=='__main__':
    report=train_weather(Store('data'),lambda message,*args:print(message,flush=True))
    print(json.dumps({k:report[k] for k in ['weight','splits','holdout','baseline','weather_only']},indent=2))
