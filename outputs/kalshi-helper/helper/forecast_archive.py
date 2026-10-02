"""Explicit-run historical forecast adapter. Retrospective, never prospective.

Run initialization is not publication. Availability uses a declared allowance;
there is no claim that historical ingestion timestamps have been verified.
"""
import hashlib,json,math,time
from datetime import timedelta
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request,urlopen
from urllib.error import HTTPError,URLError
from .core import stamp,utcnow
from .weather_model import STATIONS,event_date

MODELS=('gfs_global','ecmwf_ifs025')
HOST='https://single-runs-api.open-meteo.com/v1/forecast'

def observation_innovation(record,series,row):
    """Observed minus forecast temperature at the SAME available observation time.

    Interpolate already-issued forecast hours only; never interpolate observations
    with a later reading. This measures an error known at the decision, not a label.
    """
    from bisect import bisect_right
    if not row.get('observed'):return dict(observation_innovation_f=None)
    if row['series']!=series:raise ValueError('Innovation station mismatch')
    now=stamp(row['at']);at=stamp(row['observation_at']);available=stamp(row['observation_available_at'])
    if at>available or available>now:raise ValueError('Observation innovation would use future data')
    if stamp(record['run_initialized_at'])+timedelta(hours=8)>now:raise ValueError('Forecast not available')
    if at<stamp(record['run_initialized_at']):raise ValueError('Observation predates forecast initialization')
    body=record['body'][list(STATIONS).index(series)]
    times=[stamp(t+'Z') for t in body['hourly']['time']]
    if times!=sorted(set(times)):raise ValueError('Invalid forecast time axis')
    index=bisect_right(times,at)-1
    if index<0 or index+1>=len(times) or times[index+1]-times[index]!=timedelta(hours=1):
        return dict(observation_innovation_f=None)
    fraction=(at-times[index]).total_seconds()/3600;predictions=[]
    for model in MODELS:
        key='temperature_2m_'+model
        if body['hourly_units'].get(key)!='°F':raise ValueError('Forecast innovation requires Fahrenheit')
        a,b=body['hourly'][key][index:index+2]
        if a is None or b is None:return dict(observation_innovation_f=None)
        if not all(math.isfinite(v) for v in (a,b,row['observed_temperature'])):raise ValueError('Nonfinite innovation')
        predictions.append(a+(b-a)*fraction)
    forecast=sum(predictions)/2
    return dict(observation_innovation_f=row['observed_temperature']-forecast,
        forecast_at_observation_f=forecast,innovation_observation_at=at.isoformat())

def fetch_run(root,day,offline=False):
    run=stamp(day+'T06:00:00Z')
    if day<'2026-04-02':raise ValueError('Paired single-run archive is not verified before 2026-04-02')
    params=dict(latitude=','.join(str(v[0]) for v in STATIONS.values()),
        longitude=','.join(str(v[1]) for v in STATIONS.values()),hourly='temperature_2m',models=','.join(MODELS),
        run=run.strftime('%Y-%m-%dT%H:%M'),forecast_days=3,temperature_unit='fahrenheit',timezone='GMT')
    url=HOST+'?'+urlencode(params);key=hashlib.sha256(url.encode()).hexdigest()
    folder=Path(root)/'single-run-archive';folder.mkdir(parents=True,exist_ok=True)
    path=folder/(key+'.json')
    if path.exists():
        cached=json.loads(path.read_text(encoding='utf-8'))
        if cached['url']!=url or hashlib.sha256(json.dumps(cached['body'],sort_keys=True).encode()).hexdigest()!=cached['body_sha256']:
            raise ValueError('Forecast cache integrity failure')
        return cached
    if offline:raise ValueError('Run missing from offline cache: '+day)
    time.sleep(.35)
    for attempt in range(3):
        try:
            with urlopen(Request(url,headers={'User-Agent':'BetCheck noncommercial research'}),timeout=45) as response:raw=response.read()
            break
        except HTTPError as exc:
            if exc.code not in (429,500,502,503,504):
                detail=exc.read(1500).decode('utf-8',errors='replace')
                raise ValueError(f'Archive HTTP {exc.code}: {detail}') from exc
            delay=exc.headers.get('Retry-After','')
            delay=int(delay) if delay.isdigit() else 2**(attempt+1)
            if attempt==2 or delay>45:raise
            time.sleep(delay)
        except (URLError,TimeoutError,ConnectionError):
            if attempt==2:raise
            time.sleep(2**(attempt+1))
    body=json.loads(raw)
    if not isinstance(body,list) or len(body)!=len(STATIONS):raise ValueError('Expected four station responses')
    result=dict(url=url,run_initialized_at=run.isoformat(),downloaded_at=utcnow(),raw_sha256=hashlib.sha256(raw).hexdigest(),
        body_sha256=hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest(),body=body)
    (folder/(key+'.raw.json')).write_bytes(raw)
    path.write_text(json.dumps(result,allow_nan=False),encoding='utf-8');return result

def archived_feature(record,series,ticker,decision,publication_allowance_hours=8):
    now=stamp(decision);run=stamp(record['run_initialized_at'])
    if not math.isfinite(publication_allowance_hours) or publication_allowance_hours<8:
        raise ValueError('Retrospective allowance must be at least the declared 8h')
    available=run+timedelta(hours=publication_allowance_hours)
    if available>now:raise ValueError('Run not available under publication allowance')
    start=event_date(ticker)+timedelta(hours=STATIONS[series][2]);end=start+timedelta(days=1)
    if now>=end:raise ValueError('Station day already ended')
    body=record['body'][list(STATIONS).index(series)]
    lat,lon,_=STATIONS[series]
    if abs(body['latitude']-lat)>.6 or abs(body['longitude']-lon)>.6:raise ValueError('Forecast grid location mismatch')
    if body.get('utc_offset_seconds')!=0:raise ValueError('Archive timestamps must be UTC')
    hours=body['hourly'];times=[stamp(t+'Z') for t in hours['time']]
    if times!=sorted(set(times)):raise ValueError('Duplicate or unordered forecast hours')
    maxima={};expected=(end-max(start,now)).total_seconds()/3600
    for model in MODELS:
        key='temperature_2m_'+model
        if body['hourly_units'].get(key)!='°F':raise ValueError('Expected Fahrenheit 2m temperature')
        values=hours.get(key,[])
        if len(values)!=len(times):raise ValueError('Forecast array length mismatch')
        used=[];covered=0
        for at,v in zip(times,values):
            overlap=(min(at+timedelta(hours=1),end)-max(at,start,now)).total_seconds()/3600
            if overlap<=0:continue
            if at<run or v is None or not math.isfinite(v) or not -100<v<150:raise ValueError('Missing/invalid forecast temperature')
            used.append(v);covered+=overlap
        if not used or not math.isclose(covered,expected,abs_tol=1e-6):raise ValueError('Incomplete remaining-day forecast coverage')
        maxima[model]=max(used)
    return dict(remaining_forecast_max=sum(maxima.values())/2,remaining_hours=expected,forecast_complete=True,
        gfs_remaining_max=maxima['gfs_global'],ecmwf_remaining_max=maxima['ecmwf_ifs025'],
        model_disagreement_f=abs(maxima['gfs_global']-maxima['ecmwf_ifs025']),
        forecast_age_hours=(now-run).total_seconds()/3600,forecast_revision_f=None,
        mean_relative_humidity=None,max_wind_mph=None,mean_cloud_cover=None,
        forecast_run=run.isoformat(),weather_available_at=available.isoformat(),archive_url=record['url'],
        archive_sha256=record['raw_sha256'],forecast_source='open_meteo_single_run',
        availability_basis=f'Assumed initialization + {publication_allowance_hours:g}h, not historical receipt evidence',
        baseline_source='Mean of GFS and ECMWF remaining maxima; not NWS',prospective=False)
