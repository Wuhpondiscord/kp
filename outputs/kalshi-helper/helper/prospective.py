"""Local append-only evidence chain and fresh weather source snapshots."""
from pathlib import Path
import hashlib,json,os
from urllib.request import Request,urlopen
from urllib.parse import urlparse,urlencode
from .core import utcnow,stamp
from .weather_model import STATIONS

STATION_IDS={'KXHIGHNY':'KNYC','KXHIGHCHI':'KMDW','KXHIGHMIA':'KMIA','KXHIGHDEN':'KDEN'}

class Recorder:
    def __init__(self,store,run_id,protocol):
        self.folder=store.root/'prospective'/run_id;self.folder.mkdir(parents=True,exist_ok=False)
        self.previous='0'*64;self.count=0
        self.append('protocol',protocol)
    def append(self,kind,payload):
        record=dict(sequence=self.count,recorded_at=utcnow(),kind=kind,payload=payload,previous=self.previous)
        encoded=json.dumps(record,sort_keys=True,allow_nan=False).encode()
        record['sha256']=hashlib.sha256(encoded).hexdigest()
        with (self.folder/'journal.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(record,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
        self.previous=record['sha256'];self.count+=1
        return record['sha256']

def verify(path):
    previous='0'*64;count=0;last=None
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        r=json.loads(line);saved=r.pop('sha256')
        if r['sequence']!=count or r['previous']!=previous or hashlib.sha256(json.dumps(r,sort_keys=True,allow_nan=False).encode()).hexdigest()!=saved:
            raise ValueError('Prospective journal integrity failed')
        now=stamp(r['recorded_at'])
        if last and now<last:raise ValueError('Recorder clock moved backward')
        last=now;previous=saved;count+=1
    return dict(records=count,head=previous,verified=True,note='Local integrity check; not independent timestamp attestation')

def fetch_nws(url):
    if urlparse(url).scheme!='https' or urlparse(url).hostname!='api.weather.gov':raise ValueError('Unexpected NWS host')
    started=utcnow()
    with urlopen(Request(url,headers={'User-Agent':'BetCheck local paper research','Accept':'application/geo+json'}),timeout=15) as response:raw=response.read()
    return dict(url=url,request_started=started,received_at=utcnow(),sha256=hashlib.sha256(raw).hexdigest(),raw_utf8=raw.decode('utf-8'),body=json.loads(raw))

def weather_snapshot(series):
    lat,lon,_=STATIONS[series]
    point=fetch_nws(f'https://api.weather.gov/points/{lat},{lon}')
    forecast=fetch_nws(point['body']['properties']['forecastHourly'])
    observations=fetch_nws(f'https://api.weather.gov/stations/{STATION_IDS[series]}/observations?limit=100')
    sources=[point,forecast,observations];warnings=[]
    if point['body']['properties'].get('forecastGridData'):
        try:sources.append(fetch_nws(point['body']['properties']['forecastGridData']))
        except (OSError,ValueError) as exc:warnings.append('Grid source unavailable: '+str(exc))
    url='https://api.open-meteo.com/v1/forecast?'+urlencode(dict(latitude=lat,longitude=lon,
        hourly='temperature_2m',models='gfs_seamless,ecmwf_ifs025',temperature_unit='fahrenheit',timezone='GMT',forecast_days=3))
    try:
        started=utcnow()
        with urlopen(Request(url,headers={'User-Agent':'BetCheck local paper research'}),timeout=15) as response:raw=response.read()
        sources.append(dict(url=url,request_started=started,received_at=utcnow(),sha256=hashlib.sha256(raw).hexdigest(),
            raw_utf8=raw.decode('utf-8'),body=json.loads(raw),source='open_meteo',
            note='Current stitched forecasts archived at receipt, not an independently timestamped model initialization.'))
    except (OSError,ValueError) as exc:warnings.append('Model comparison source unavailable: '+str(exc))
    return dict(series=series,station=STATION_IDS[series],captured_at=utcnow(),sources=sources,warnings=warnings,
        note='Recorded current hourly forecast and recent station reports. Not 5-minute ASOS, not guaranteed complete settlement-day maxima; no new model signal implied.')
