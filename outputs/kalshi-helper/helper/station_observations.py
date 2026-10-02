"""Exact-station observations with conservative retrospective availability.

Observation timestamps plus a delay are an assumption, not verified ingestion.
METAR maxima are not guaranteed to equal the final climate settlement value.
"""
from bisect import bisect_right
import csv
from datetime import datetime,timedelta,timezone
import hashlib
import io
import json
import math
from pathlib import Path
import re
import time
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request,urlopen
from .core import stamp,utcnow
from .weather_model import event_date,STATIONS

SITES={'KXHIGHNY':'NYC','KXHIGHCHI':'MDW','KXHIGHMIA':'MIA','KXHIGHDEN':'DEN'}


def parse_observations(payload,station):
    parsed={};conflicts=set();excluded=0
    lines=[line for line in payload.decode('utf-8-sig').splitlines() if not line.startswith('#')]
    reader=csv.DictReader(io.StringIO('\n'.join(lines)))
    if not {'station','valid','tmpf','metar'}<=set(reader.fieldnames or []):raise ValueError('Invalid station CSV schema')
    for row in reader:
        if row['station']!=station:raise ValueError('Unexpected observation station')
        try:
            when=datetime.fromisoformat(row['valid'])
            when=when.replace(tzinfo=timezone.utc) if when.tzinfo is None else when.astimezone(timezone.utc)
            temp=float(row['tmpf'])
            if not math.isfinite(temp) or not -100<temp<150:raise ValueError('Invalid temperature')
            if re.search(r'\bCOR\b',row['metar']):raise ValueError('Retrospective correction excluded')
            maximum=None
            remarks=row['metar'].partition(' RMK ')[2]
            m=re.search(r'(?:^|\s)1([01])(\d{3})(?:\s|$)',remarks)
            if m:
                maximum=(int(m[2])/10*(-1 if m[1]=='1' else 1))*9/5+32
                if not -100<maximum<150:maximum=None
            record=dict(at=when.isoformat(),temperature=temp,max_6h=maximum)
            key=record['at']
            if key in parsed and parsed[key]!=record:conflicts.add(key)
            parsed[key]=record
        except (ValueError,KeyError):excluded+=1
    records=[r for k,r in sorted(parsed.items()) if k not in conflicts]
    if not records:raise ValueError('No usable station observations')
    return records,dict(rejected_rows=excluded,conflicting_timestamps=len(conflicts),usable_rows=len(records))


def fetch_observations(store,station,start,end):
    if station not in SITES.values():raise ValueError('Unsupported station')
    params=dict(station=station,data='tmpf,metar',sts=start+'T00:00:00Z',ets=end+'T00:00:00Z',
                tz='UTC',format='onlycomma',report_type='3,4',missing='M',latlon='no',elev='no')
    url='https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?'+urlencode(params)
    key=hashlib.sha256(url.encode()).hexdigest();folder=store.root/'station-observations';folder.mkdir(exist_ok=True)
    path=folder/(key+'.json')
    if path.exists():
        cached=json.loads(path.read_text(encoding='utf-8'))
        if end<utcnow()[:10] or (stamp(utcnow())-stamp(cached['downloaded_at'])).total_seconds()<3600:return cached
    for attempt in range(4):
        try:
            with urlopen(Request(url,headers={'User-Agent':'BetCheck local research'}),timeout=45) as response:payload=response.read()
            break
        except HTTPError as exc:
            if exc.code not in (429,503) or attempt==3:raise
            retry=exc.headers.get('Retry-After','')
            delay=int(retry) if retry.isdigit() else (5,15,30)[attempt]
            if delay>60:raise
            time.sleep(max(1,delay))
    store.raw(url,payload);records,quality=parse_observations(payload,station)
    result=dict(station=station,url=url,downloaded_at=utcnow(),raw_sha256=hashlib.sha256(payload).hexdigest(),quality=quality,observations=records)
    path.write_text(json.dumps(result),encoding='utf-8');return result


class ObservationIndex:
    def __init__(self,observations):
        self.rows=sorted(observations,key=lambda r:stamp(r['at']));self.times=[stamp(r['at']) for r in self.rows]

    def feature(self,series,ticker,decision,delay_minutes=30):
        if not math.isfinite(delay_minutes) or delay_minutes<0:raise ValueError('Observation delay must be finite and nonnegative')
        start=event_date(ticker)+timedelta(hours=STATIONS[series][2]);end=start+timedelta(days=1)
        cutoff=stamp(decision)-timedelta(minutes=delay_minutes)
        left=bisect_right(self.times,start-timedelta(microseconds=1));right=bisect_right(self.times,min(cutoff,end-timedelta(microseconds=1)))
        rows=self.rows[left:right]
        if not rows:return dict(observed=False,reason='No same-day observation before availability cutoff')
        latest=stamp(rows[-1]['at']);age=(stamp(decision)-latest).total_seconds()/3600
        if age>3:return dict(observed=False,reason='Station observation older than three hours')
        values=[r['temperature'] for r in rows]
        # A six-hour maximum may only be included if its entire window belongs
        # to this settlement day. Never pull the next day's daily summary.
        maxima=[r['max_6h'] for r in rows if r.get('max_6h') is not None and stamp(r['at'])-timedelta(hours=6)>=start]
        older=[r for r in rows if stamp(r['at'])<=latest-timedelta(hours=2)]
        return dict(observed=True,observed_max=max(values+maxima),observed_temperature=values[-1],
            observed_trend=values[-1]-older[-1]['temperature'] if older else 0,
            observation_age_hours=age,observation_count=len(rows),observation_at=rows[-1]['at'],
            observation_available_at=(latest+timedelta(minutes=delay_minutes)).isoformat(),
            local_day_hour=(stamp(decision)-start).total_seconds()/3600)
