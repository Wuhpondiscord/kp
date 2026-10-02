"""Public data only. No credentials, wallets, or order placement endpoints."""
import json,hashlib,re,math
from urllib.request import Request,urlopen
from urllib.parse import urlencode,urlparse
from .core import utcnow,stamp
from .weather_model import event_date

CITIES={'KXHIGHNY':'nyc','KXHIGHCHI':'chicago','KXHIGHMIA':'miami','KXHIGHDEN':'denver'}

def read_json(url):
    u=urlparse(url)
    if u.scheme!='https' or (u.hostname,u.path) not in {('gamma-api.polymarket.com','/events'),('clob.polymarket.com','/book')}:
        raise ValueError('Unapproved public data endpoint')
    started=utcnow()
    with urlopen(Request(url,headers={'User-Agent':'BetCheck/1.0','Accept':'application/json'}),timeout=10) as response:raw=response.read()
    return dict(url=url,request_started=started,received_at=utcnow(),sha256=hashlib.sha256(raw).hexdigest(),raw_utf8=raw.decode('utf-8'),body=json.loads(raw))

def bracket(text):
    s=text.replace('−','-').replace('–','-').strip()
    if '°F' not in s:raise ValueError('Only explicitly Fahrenheit brackets supported')
    m=re.fullmatch(r'(-?\d+)(?:-(-?\d+))?°F(?: or (below|higher|above))?',s)
    if not m:raise ValueError('Unknown bracket wording')
    lo=int(m[1]);hi=int(m[2] or m[1])
    if hi<lo:raise ValueError('Reversed bracket')
    return (None,hi+.5) if m[3]=='below' else (lo-.5,None) if m[3] in ('higher','above') else (lo-.5,hi+.5)

def midpoint(book):
    bids=[float(v['price']) for v in book['bids'] if float(v['size'])>0]
    asks=[float(v['price']) for v in book['asks'] if float(v['size'])>0]
    if not bids or not asks:raise ValueError('Empty Polymarket book')
    bid,ask=max(bids),min(asks)
    if not 0<=bid<=ask<=1:raise ValueError('Invalid Polymarket book')
    return dict(bid=bid,ask=ask,p=(bid+ask)/2,spread=ask-bid)

def quote_bounds(book):
    """Missing sides leave an interval, never a fabricated midpoint."""
    values={}
    for side in ('bids','asks'):
        levels=[]
        for v in book.get(side,[]):
            price,size=float(v['price']),float(v['size'])
            if not math.isfinite(price) or not math.isfinite(size) or not 0<=price<=1 or size<0:raise ValueError('Invalid book level')
            if size>0:levels.append(price)
        values[side]=levels
    bid=max(values['bids'],default=0);ask=min(values['asks'],default=1)
    if bid>ask:raise ValueError('Crossed Polymarket book')
    both=bool(values['bids']) and bool(values['asks'])
    return dict(bid=bid,ask=ask,p=(bid+ask)/2 if both else None,spread=ask-bid,
        quality='two_sided' if both else 'one_sided' if values['bids'] or values['asks'] else 'empty')

def cdf_bounds(lows,highs):
    if len(lows)!=len(highs) or not lows or any(not 0<=a<=b<=1 for a,b in zip(lows,highs)) or sum(lows)>1+1e-8 or sum(highs)<1-1e-8:
        raise ValueError('Infeasible quote intervals')
    return ([max(sum(lows[:i+1]),1-sum(highs[i+1:])) for i in range(len(lows))],
            [min(sum(highs[:i+1]),1-sum(lows[i+1:])) for i in range(len(lows))])

def collect(ticker):
    series=ticker.split('-')[0];day=event_date(ticker)
    slug=f"highest-temperature-in-{CITIES[series]}-on-{day.strftime('%B').lower()}-{day.day}-{day.year}"
    receipt=read_json('https://gamma-api.polymarket.com/events?'+urlencode({'slug':slug}))
    events=[e for e in receipt['body'] if e.get('slug')==slug]
    if len(events)!=1:raise ValueError('No unique corresponding Polymarket city/date event')
    event=events[0];markets=[];receipts=[receipt];errors=[]
    for m in event['markets'][:24]:
        try:
            lo,hi=bracket(m['groupItemTitle'])
            outcomes=json.loads(m['outcomes']) if isinstance(m['outcomes'],str) else m['outcomes']
            tokens=json.loads(m['clobTokenIds']) if isinstance(m['clobTokenIds'],str) else m['clobTokenIds']
            token=tokens[outcomes.index('Yes')]
            response=read_json('https://clob.polymarket.com/book?'+urlencode({'token_id':token}));receipts.append(response)
            # Receipt time is known; exchange timestamp describes book state, not local availability.
            quoted=quote_bounds(response['body'])
            markets.append(dict(id=m['id'],lower=lo,upper=hi,**quoted))
        except (OSError,ValueError,KeyError,IndexError) as exc:errors.append(dict(id=m.get('id'),error=str(exc)))
    return dict(series=series,date=day.date().isoformat(),slug=slug,captured_at=utcnow(),markets=markets,errors=errors,
        relation='related_only',exact_match=False,rules=event.get('description',''),sources=receipts,
        note='City/date discovery only. Station, observation convention, fallback and revision rules may differ; never a substitute Kalshi probability or label.')

def regional_features(snapshot,ticker,at):
    if stamp(snapshot['captured_at'])>stamp(at):raise ValueError('Future Polymarket snapshot')
    if (stamp(at)-stamp(snapshot['captured_at'])).total_seconds()>1200:raise ValueError('Stale Polymarket snapshot')
    if snapshot['series']!=ticker.split('-')[0] or snapshot['date']!=event_date(ticker).date().isoformat():raise ValueError('Polymarket city/date mismatch')
    rows=sorted(snapshot['markets'],key=lambda r:-math.inf if r['lower'] is None else r['lower'])
    complete=bool(rows) and rows[0]['lower'] is None and rows[-1]['upper'] is None and all(a['upper']==b['lower'] for a,b in zip(rows,rows[1:]))
    if not complete:return dict(poly_available=False,poly_reason='Incomplete bracket definitions')
    lows=[r.get('bid',max(0,r['p']-r['spread']/2) if r.get('p') is not None else 0) for r in rows]
    highs=[r.get('ask',min(1,r['p']+r['spread']/2) if r.get('p') is not None else 1) for r in rows]
    if any(not 0<=a<=b<=1 for a,b in zip(lows,highs)) or sum(lows)>1+1e-8 or sum(highs)<1-1e-8:
        return dict(poly_available=False,poly_reason='No unit-mass distribution consistent with quote intervals')
    # Feasible CDF bounds under sum(probabilities)=1. These are quote-implied
    # constraints, not confidence intervals for the true outcome probabilities.
    cdf_low,cdf_high=cdf_bounds(lows,highs)
    quantiles={}
    for q,name in ((.25,'q25'),(.5,'median'),(.75,'q75')):
        left=next((i for i,v in enumerate(cdf_high) if v>=q-1e-8),len(rows)-1)
        right=next((i for i,v in enumerate(cdf_low) if v>=q-1e-8),len(rows)-1)
        quantiles['poly_'+name+'_low']=rows[left]['lower'];quantiles['poly_'+name+'_high']=rows[right]['upper']
    a,b=quantiles['poly_median_low'],quantiles['poly_median_high']
    usable=a is not None and b is not None and b-a<=10
    return dict(poly_available=usable,**quantiles,poly_median_width_f=b-a if a is not None and b is not None else None,
        poly_reason='Quote-constrained regional distribution' if usable else 'Median unbounded or wider than predefined 10F limit',
        poly_mean_spread=(sum(r['spread'] for r in rows if r.get('p') is not None)/sum(r.get('p') is not None for r in rows)) if any(r.get('p') is not None for r in rows) else None,
        poly_mean_quote_interval_width=sum(r['spread'] for r in rows)/len(rows),
        poly_two_sided_fraction=sum(r.get('p') is not None for r in rows)/len(rows),
        poly_received_at=snapshot['captured_at'],poly_relation='related_only',poly_feature_version=3)
