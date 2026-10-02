"""Weather-only seasonal pilot. Fixed-lead archives are NOT explicit-run replay."""
import argparse,hashlib,json,math,time
from collections import defaultdict,Counter
from datetime import datetime,timedelta,timezone
from pathlib import Path
from urllib.request import Request,urlopen
from urllib.parse import urlencode
import numpy as np
from scipy.special import ndtr
from .weather_model import STATIONS
from .station_observations import SITES
from .core import stamp,utcnow

VARIABLE='temperature_2m_previous_day2'

def fetch(root,url,offline=False):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    key=hashlib.sha256(url.encode()).hexdigest();raw=root/(key+'.raw.json');meta=root/(key+'.json')
    if raw.exists() and meta.exists():
        payload=raw.read_bytes();record=json.loads(meta.read_text())
        if record['url']!=url or hashlib.sha256(payload).hexdigest()!=record['sha256']:raise ValueError('Cache integrity failure')
        return json.loads(payload),record
    if offline:raise ValueError('Offline source missing: '+url)
    with urlopen(Request(url,headers={'User-Agent':'BetCheck noncommercial weather research'}),timeout=60) as response:payload=response.read()
    body=json.loads(payload)
    record=dict(url=url,downloaded_at=utcnow(),sha256=hashlib.sha256(payload).hexdigest())
    raw.write_bytes(payload);meta.write_text(json.dumps(record,indent=2));time.sleep(.4)
    return body,record

def targets(body,station,year):
    result={};excluded=Counter()
    for r in body['results']:
        if r['station']!=station:raise ValueError('Wrong CLI station')
        day=datetime.strptime(r['valid'],'%Y-%m-%d').replace(tzinfo=timezone.utc)
        if day.year!=year:raise ValueError('Wrong CLI year')
        if r['valid'] in result:raise ValueError('Duplicate CLI target day')
        try:
            y=float(r['high'])
            if not math.isfinite(y) or not -100<y<150:raise ValueError()
            issued=datetime.strptime(r['product'].split('-')[0],'%Y%m%d%H%M').replace(tzinfo=timezone.utc)
            if issued<=day:raise ValueError()
        except (ValueError,TypeError,KeyError):excluded['missing_or_invalid_label']+=1;continue
        result[r['valid']]=dict(target_f=y,label_issued_at=issued.isoformat(),target_product=r['product'],target_station=station)
    return result,dict(excluded)

def forecast_hours(body,series):
    lat,lon,_=STATIONS[series]
    if body.get('utc_offset_seconds')!=0 or body['hourly_units'][VARIABLE]!='°F':raise ValueError('Forecast requires UTC and Fahrenheit')
    if abs(body['latitude']-lat)>.6 or abs(body['longitude']-lon)>.6:raise ValueError('Wrong forecast location')
    times=body['hourly']['time'];values=body['hourly'][VARIABLE]
    if len(times)!=len(values) or len(set(times))!=len(times):raise ValueError('Invalid forecast axis')
    parsed=[stamp(t+'Z') for t in times]
    if parsed!=sorted(parsed):raise ValueError('Forecast axis not ordered')
    return dict(zip(parsed,values))

def join_day(series,day,label,hours):
    begin=stamp(day+'T00:00:00Z')+timedelta(hours=STATIONS[series][2]);end=begin+timedelta(days=1)
    valid=[begin+timedelta(hours=i) for i in range(24)]
    values=[hours.get(t) for t in valid]
    if any(v is None or not math.isfinite(float(v)) or not -100<float(v)<150 for v in values):raise ValueError('Incomplete/invalid 24-hour forecast')
    # Fixed-lead product: infer latest possible availability, not a real issue ID.
    assumed_available=valid[-1]-timedelta(hours=48)+timedelta(hours=8)
    decision=begin-timedelta(hours=12)
    if assumed_available>decision:raise ValueError('Forecast not available under declared allowance')
    if stamp(label['label_issued_at'])<end:raise ValueError('Target was issued before the station day ended')
    return dict(station=series,day=day,decision_at=decision.isoformat(),day_start=begin.isoformat(),day_end=end.isoformat(),
        raw_forecast_f=float(max(values)),forecast_available_at_assumed=assumed_available.isoformat(),
        availability_verified=False,explicit_single_run=False,**label)

def fit(rows):
    if not rows:raise ValueError('No training rows')
    model={}
    for station in STATIONS:
        rr=[r for r in rows if r['station']==station]
        if len(rr)<90:raise ValueError('Need 90 training days for each station')
        e=np.array([r['target_f']-r['raw_forecast_f'] for r in rr]);bias=float(e.mean())
        model[station]=dict(n=len(rr),bias_f=bias,sigma_raw_f=max(.5,float(np.sqrt(np.mean(e*e)))),sigma_corrected_f=max(.5,float(np.sqrt(np.mean((e-bias)**2)))))
    return model

def losses(rows,model,corrected):
    y=np.array([r['target_f'] for r in rows]);mu=np.array([r['raw_forecast_f']+(model[r['station']]['bias_f'] if corrected else 0) for r in rows]);sigma=np.array([model[r['station']]['sigma_corrected_f' if corrected else 'sigma_raw_f'] for r in rows])
    z=(y-mu)/sigma;phi=np.exp(-z*z/2)/math.sqrt(2*math.pi)
    crps=sigma*(z*(2*ndtr(z)-1)+2*phi-1/math.sqrt(math.pi))
    return dict(error=mu-y,crps=crps,nll=np.log(sigma)+.5*z*z+.5*math.log(2*math.pi),pit=ndtr(z),coverage90=(abs(z)<=1.6448536269514722))

def score(rows,model,corrected):
    v=losses(rows,model,corrected)
    return dict(n=len(rows),mae_f=float(np.mean(abs(v['error']))),rmse_f=float(np.sqrt(np.mean(v['error']**2))),bias_f=float(np.mean(v['error'])),crps_f=float(v['crps'].mean()),gaussian_nll=float(v['nll'].mean()),coverage90=float(v['coverage90'].mean()),pit_bins=np.histogram(v['pit'],bins=np.linspace(0,1,11))[0].tolist())

def run(root,output,offline=False):
    root=Path(root);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    protocol=dict(version='weather-independent-pilot-v1',training_year=2023,validation_year=2024,
        train_last_day='2023-12-28',stations=STATIONS,targets='NWS CLI daily highs retrieved from IEM; final archive vintage, not certified Kalshi settlement values',
        forecasts='Open-Meteo GFS global previous_day2; mixed fixed-lead forecast hours, NOT a single initialized run',
        day='24h fixed local standard time; 24 hourly forecast samples; does not capture subhourly forecast peaks',
        decision='12h before station day start',publication_assumption='latest valid hour minus 48h plus 8h; not historical receipt evidence',
        models=['Raw GFS maximum + station train-RMSE Gaussian width','Same forecast + train station bias + train residual-SD Gaussian width'],
        minimum_sigma_f=.5,primary_metric='Validation CRPS',selection='One declared baseline comparison; no tuning or live promotion',
        market_data_used=False,trading_evaluated=False,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    rows=[];sources=[];excluded=[]
    for year in [2023,2024]:
        for series,(lat,lon,_) in STATIONS.items():
            station='K'+SITES[series];print('Loading',year,station,flush=True)
            target,tm=fetch(root,'https://mesonet.agron.iastate.edu/json/cli.py?'+urlencode(dict(station=station,year=year)),offline)
            labels,reasons=targets(target,station,year);sources.append(tm)
            params=dict(latitude=lat,longitude=lon,hourly=VARIABLE,models='gfs_global',start_date=f'{year}-01-01',end_date=f'{year+1}-01-01',temperature_unit='fahrenheit',timezone='GMT')
            forecast,fm=fetch(root,'https://previous-runs-api.open-meteo.com/v1/forecast?'+urlencode(params),offline);sources.append(fm)
            hours=forecast_hours(forecast,series)
            for day,label in sorted(labels.items()):
                try:rows.append(dict(join_day(series,day,label,hours),forecast_sha256=fm['sha256'],target_sha256=tm['sha256']))
                except ValueError as e:excluded.append(dict(station=series,day=day,reason=str(e)))
            if reasons:excluded.append(dict(station=series,year=year,target_exclusions=reasons))
    train=[r for r in rows if r['day']<='2023-12-28'];val=[r for r in rows if r['day'].startswith('2024-')]
    cutoff=min(stamp(r['decision_at']) for r in val)
    late=[r for r in train if stamp(r['label_issued_at'])>=cutoff]
    train=[r for r in train if stamp(r['label_issued_at'])<cutoff]
    if late:excluded.extend(dict(station=r['station'],day=r['day'],reason='Training label unavailable before validation') for r in late)
    model=fit(train)
    for name,part in [('train',train),('validation',val)]:
        (output/(name+'.jsonl')).write_text('\n'.join(json.dumps(r,allow_nan=False) for r in part))
    (output/'model.json').write_text(json.dumps(model,indent=2))
    reports={}
    for name,corrected in [('raw',False),('station_bias',True)]:
        reports[name]=dict(overall=score(val,model,corrected),by_station={s:score([r for r in val if r['station']==s],model,corrected) for s in STATIONS},
            by_quarter={str(q):score([r for r in val if (int(r['day'][5:7])-1)//3+1==q],model,corrected) for q in range(1,5)})
    delta=losses(val,model,True)['crps']-losses(val,model,False)['crps'];daily=defaultdict(list)
    for r,d in zip(val,delta):daily[r['day']].append(d)
    d=np.array([np.mean(daily[k]) for k in sorted(daily)]);rng=np.random.default_rng(1729);draws=[]
    for _ in range(3000):
        starts=rng.integers(0,len(d),math.ceil(len(d)/7));ix=np.concatenate([(s+np.arange(7))%len(d) for s in starts])[:len(d)];draws.append(float(d[ix].mean()))
    report=dict(protocol=protocol,counts=dict(train=len(train),validation=len(val),train_dates=len({r['day'] for r in train}),validation_dates=len(daily)),results=reports,
        paired_crps=dict(mean=float(d.mean()),interval95=list(map(float,np.quantile(draws,[.025,.975]))),method='Seven-date blocks, cities synchronized; conditional on this fixed pilot and archive vintage'),sources=sources,excluded=excluded,
        limitations=['Not an explicit-run historical trading replay','NWS CLI label equivalence to every Kalshi settlement source is not established','Hourly grid maxima differ from reported station maxima','One train year and one validation year do not prove multi-year robustness','No price edge or profitability claim'])
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file()}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(dict(counts=report['counts'],results={k:v['overall'] for k,v in reports.items()},paired_crps=report['paired_crps']),indent=2))
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--cache',required=True);parser.add_argument('--output',required=True);parser.add_argument('--offline',action='store_true');args=parser.parse_args()
    run(args.cache,args.output,args.offline)
