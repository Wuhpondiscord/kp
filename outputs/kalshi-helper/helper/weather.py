"""Real NWS precipitation context. No fabricated full-event probabilities."""
from datetime import datetime, timedelta
import json
import math
import re
from urllib.request import Request, urlopen
from urllib.parse import urlparse
from zoneinfo import ZoneInfo
from .core import stamp, utcnow
from .research_store import ResearchStore

STATIONS = dict(NYC='KNYC',ORD='KORD',AUS='KAUS',MIA='KMIA',DEN='KDEN',PHL='KPHL',
    LAX='KLAX',LAS='KLAS',MSY='KMSY',SFO='KSFO',DCA='KDCA',SEA='KSEA',BOS='KBOS',
    PHX='KPHX',ATL='KATL',MSP='KMSP',DFW='KDFW',SAT='KSAT',HOU='KHOU',OKC='KOKC',EWR='KEWR',TTN='KTTN')


def nws_get(store, url):
    if urlparse(url).hostname != 'api.weather.gov' or urlparse(url).scheme != 'https':
        raise ValueError('Unexpected weather data host')
    rs = ResearchStore(store)
    cached = rs.get('nws:' + url)
    if cached and 0 <= (stamp(utcnow())-stamp(cached['at'])).total_seconds()<900:
        return cached['body']
    with urlopen(Request(url,headers={'User-Agent':'KalshiHelper/0.2 (local paper research)',
                                    'Accept':'application/geo+json'}),timeout=12) as response:
        raw = response.read()
    store.raw(url,raw)
    body = json.loads(raw)
    rs.set('nws:' + url,dict(at=utcnow(),body=body))
    return body


def rain_context(store, card):
    unavailable = dict(status='unavailable',probability=None,trade_eligible=False,periods=[])
    match = re.fullmatch(r'If the total precipitation at CLI([A-Z]+) in .+ on any day within (.+) through (.+) is strictly greater than 0 inches, then the market resolves to Yes\.',card['rules'])
    if not match:
        return dict(unavailable,note='An independent forecast adapter is not available for these exact settlement rules.')
    code, start, end = match.groups()
    if code not in STATIONS:
        return dict(unavailable,note='Settlement station mapping has not been verified for this location.')
    station = STATIONS[code]
    station_url = 'https://api.weather.gov/stations/' + station
    station_data = nws_get(store,station_url)
    lon,lat = station_data['geometry']['coordinates'][:2]
    point = nws_get(store,f'https://api.weather.gov/points/{lat:.4f},{lon:.4f}')['properties']
    zone = ZoneInfo(point['timeZone'])
    begin = datetime.strptime(start,'%B %d, %Y').replace(tzinfo=zone)
    finish = datetime.strptime(end,'%B %d, %Y').replace(tzinfo=zone)+timedelta(days=1)
    source = point['forecast']
    forecast = nws_get(store,source)['properties']
    updated = forecast.get('updateTime') or forecast.get('generatedAt')
    now = stamp(utcnow())
    if not updated or not -300 <= (now-stamp(updated)).total_seconds() <= 18*3600:
        return dict(unavailable,note='NWS forecast is stale or has an invalid publication time.',source_url=source)
    periods = []
    for p in forecast['periods']:
        a,b = stamp(p['startTime']),stamp(p['endTime'])
        pop = (p.get('probabilityOfPrecipitation') or {}).get('value')
        if b <= begin or a >= finish or pop is None:
            continue
        if not isinstance(pop,(float,int)) or not math.isfinite(pop) or not 0 <= pop <= 100:
            continue
        periods.append(dict(name=p['name'],start=p['startTime'],end=p['endTime'],probability=pop/100,
            summary=p.get('shortForecast',''),fully_inside=a>=begin and b<=finish))
    if not periods:
        return dict(unavailable,note='No current NWS forecast periods overlap these market dates.',source_url=source)
    # Correlated periods: never multiply independent no-rain probabilities.
    # Bounds describe the union of the displayed NWS periods only, not settlement.
    bounds=[max(p['probability'] for p in periods),min(1,sum(p['probability'] for p in periods))]
    return dict(status='weather_context',probability=None,trade_eligible=False,station=station,
        source_url=source,station_url=station_url,updated_at=updated,fetched_at=utcnow(),
        market_start=begin.isoformat(),market_end=finish.isoformat(),periods=periods,
        period_union_bounds=bounds,
        note='NWS chance of at least 0.01 inch during each displayed period. These are weather forecasts, not calibrated contract odds. '
             + ('The market window has already started; earlier observed rain must also be checked. ' if now>begin else '')
             + 'Kalshi settles from The Weather Company. Periods may cross market boundaries; daily accumulation and station/report differences prevent an exact match. No automated rain trades until that mapping is validated.')
