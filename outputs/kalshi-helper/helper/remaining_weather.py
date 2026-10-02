"""Point-in-time remaining-day features. No labels or inferred publication times."""
from datetime import timedelta
import math
import re
from .core import stamp
from .weather_model import STATIONS,event_date


def features(snapshot,ticker,at,previous=None):
    now=stamp(at);series=ticker.split('-')[0]
    if snapshot['series']!=series:raise ValueError('Weather city mismatch')
    if stamp(snapshot['captured_at'])>now:raise ValueError('Weather snapshot arrived after decision')
    start=event_date(ticker)+timedelta(hours=STATIONS[series][2]);end=start+timedelta(days=1)
    forecasts=[];observations=[];issues=[];receipts=[];humidity=[];wind=[];cloud=[];model_maxima={}
    for source in snapshot['sources']:
        received=stamp(source['received_at'])
        if received>now:raise ValueError('Source arrived after decision')
        receipts.append(received)
        body=source['body'];props=body.get('properties',{})
        if source.get('source')=='open_meteo':
            hourly=body.get('hourly',{});units=body.get('hourly_units',{})
            for model in ('gfs_seamless','ecmwf_ifs025'):
                key='temperature_2m_'+model;values=[];covered_model=0
                if units.get(key) not in ('°F','°C'):continue
                for t,v in zip(hourly.get('time',[]),hourly.get(key,[])):
                    a=stamp(t+'Z') # Request explicitly sets GMT; API omits the suffix.
                    overlap=(min(a+timedelta(hours=1),end)-max(a,start,now)).total_seconds()/3600
                    if v is not None and math.isfinite(float(v)) and overlap>0:
                        values.append(float(v) if units[key]=='°F' else float(v)*1.8+32)
                        covered_model+=overlap
                if values and abs(covered_model-max(0,(end-max(now,start)).total_seconds()/3600))<1e-6:model_maxima[model]=max(values)
        if 'periods' in props:
            issue=props.get('updateTime') or props.get('generatedAt')
            if not issue:raise ValueError('Forecast issue time missing')
            if stamp(issue)>received:raise ValueError('Forecast issue follows receipt')
            issues.append(stamp(issue))
            for period in props['periods']:
                a,b=stamp(period['startTime']),stamp(period['endTime'])
                value=period.get('temperature');unit=period.get('temperatureUnit')
                if value is None or unit not in ('F','C'):continue
                value=float(value) if unit=='F' else float(value)*1.8+32
                if not math.isfinite(value):continue
                a=max(a,start,now);b=min(b,end)
                if b>a:
                    forecasts.append((a,b,value))
                    rh=(period.get('relativeHumidity') or {}).get('value')
                    if rh is not None and math.isfinite(float(rh)):humidity.append(float(rh))
                    speed=period.get('windSpeed','')
                    if isinstance(speed,str) and 'mph' in speed:wind.extend(float(x) for x in re.findall(r'\d+(?:\.\d+)?',speed))
        sky=props.get('skyCover',{})
        if sky:
            if not props.get('updateTime') or stamp(props['updateTime'])>received:raise ValueError('Invalid grid issue time')
            for item in sky.get('values',[]):
                valid,duration=item['validTime'].split('/')
                match=re.fullmatch(r'P(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?)?',duration)
                if not match:continue
                a=stamp(valid);b=a+timedelta(days=int(match[1] or 0),hours=int(match[2] or 0),minutes=int(match[3] or 0))
                v=item.get('value')
                if v is not None and math.isfinite(float(v)) and min(b,end)>max(a,start,now):cloud.append(float(v))
        for item in body.get('features',[]):
            prop=item.get('properties',{});temperature=prop.get('temperature') or {}
            value=temperature.get('value');unit=temperature.get('unitCode')
            if value is None or not prop.get('timestamp'):continue
            observed=stamp(prop['timestamp'])
            if not start<=observed<end or observed>now or observed>received:continue
            if unit not in ('wmoUnit:degC','wmoUnit:degF'):continue
            value=float(value)*(1.8 if unit.endswith('degC') else 1)+(32 if unit.endswith('degC') else 0)
            if math.isfinite(value):observations.append((observed,value))
    # Union coverage avoids counting overlapping forecast periods twice.
    cursor=max(now,start);covered=0.
    for a,b,_ in sorted(forecasts):
        if b>cursor:covered+=(b-max(a,cursor)).total_seconds()/3600;cursor=b
    expected=max(0,(end-max(now,start)).total_seconds()/3600)
    revision=None
    if previous and stamp(previous['captured_at'])<stamp(snapshot['captured_at']):
        old=features(previous,ticker,at)
        if forecasts and old['remaining_forecast_max'] is not None:revision=max(v for _,_,v in forecasts)-old['remaining_forecast_max']
    return dict(schema_version=2,ticker=ticker,at=at,event_start=start.isoformat(),event_end=end.isoformat(),
        gfs_remaining_max=model_maxima.get('gfs_seamless'),ecmwf_remaining_max=model_maxima.get('ecmwf_ifs025'),
        model_disagreement_f=abs(model_maxima['gfs_seamless']-model_maxima['ecmwf_ifs025']) if len(model_maxima)==2 else None,
        forecast_revision_f=revision,mean_relative_humidity=sum(humidity)/len(humidity) if humidity else None,
        max_wind_mph=max(wind,default=None),mean_cloud_cover=sum(cloud)/len(cloud) if cloud else None,
        remaining_hours=expected,forecast_covered_hours=covered,
        forecast_complete=expected>0 and abs(covered-expected)<1e-6,
        remaining_forecast_max=max((v for _,_,v in forecasts),default=None),
        observed_max=max((v for _,v in observations),default=None),observation_count=len({a for a,_ in observations}),
        latest_observation_age_minutes=(now-max(a for a,_ in observations)).total_seconds()/60 if observations else None,
        forecast_age_hours=(now-max(issues)).total_seconds()/3600 if issues else None,
        available_at=max(receipts).isoformat() if receipts else snapshot['captured_at'],
        limitations='Hourly sampled maximum is not the settlement maximum. Forecast coverage does not establish forecast accuracy.')
