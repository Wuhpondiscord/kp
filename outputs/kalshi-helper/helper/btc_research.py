"""Separate BTC15M research pilot: real labels, archived quotes, no live orders.

Coinbase is a predictor, NOT the BRTI settlement source. Historical timestamps
include a declared publication allowance, not verified real-time availability.
"""
import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
import re
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit, ndtr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .core import stamp
from .history import valid_quote
from .sizing import simulate

SERIES='KXBTC15M'
BASE='https://external-api.kalshi.com/trade-api/v2'
PROTOCOL={
    'version':'btc15m-pilot-v1','series':SERIES,
    'start':'2026-09-01','end_exclusive':'2026-10-01',
    'train_end_exclusive':'2026-09-19','validation_end_exclusive':'2026-09-25',
    'decision_seconds_before_close':300,'publication_allowance_seconds':5,
    'lookback_minutes':120,'split_embargo_hours':2,
    'models':['market','BTCVolatility','BTCSignal','BTCMarketGuard','price_only_control'],
    'selection':'Fixed architectures; no test tuning or strategy search',
    'signal_logistic_C':.1,'offset_l2':.01,'random_seed':1729,
    'trade_policy':'bankroll_1_percent','bankroll':1000,'minimum_edge':.04,
    'slippage_dollars':[0,.01,.02],'fee_assumption':.07,
    'settlement':'Final-minute mean BRTI >= opening target; ties YES; never touch labels',
    'limitations':['Coinbase spot is not BRTI','No historical depth or queue evidence',
        'Five-second publication allowance is assumed, not measured',
        'Final metadata supplies opening target; original publication not archived',
        '0.07 quadratic fee assumption is not a verified historical fee schedule',
        'Six test days cannot establish a durable edge'],
}
FEATURES=['distance_in_sigma','return_1m','return_5m','return_15m','return_60m',
          'volatility_ratio','volume_ratio','range_in_sigma','hour_sin','hour_cos']

def iso(ts):return datetime.fromtimestamp(ts,timezone.utc).isoformat()
def unix(day):return int(stamp(day+'T00:00:00Z').timestamp())
def digest(raw):return hashlib.sha256(raw).hexdigest()
def save(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf8')

class Cache:
    def __init__(self,root,offline=False):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True);self.offline=offline
        self.used={}
    def get(self,url):
        key=digest(url.encode());meta=self.root/(key+'.json');rawpath=self.root/(key+'.raw')
        if meta.exists() and rawpath.exists():
            record=json.loads(meta.read_text());raw=rawpath.read_bytes()
            if record['url']!=url or digest(raw)!=record['sha256']:raise ValueError('Cache hash mismatch')
        else:
            if self.offline:raise ValueError('Offline cache missing: '+url)
            for attempt in range(5):
                try:
                    time.sleep(.15)
                    with urlopen(Request(url,headers={'User-Agent':'BetCheck public research','Accept':'application/json'}),timeout=40) as response:raw=response.read()
                    break
                except (HTTPError,URLError,TimeoutError) as exc:
                    if isinstance(exc,HTTPError) and exc.code not in (429,500,502,503,504):raise
                    if attempt==4:raise
                    time.sleep(2**attempt)
            record=dict(url=url,sha256=digest(raw),fetched_at=datetime.now(timezone.utc).isoformat())
            rawpath.write_bytes(raw);save(meta,record)
        self.used[key]=record
        return json.loads(raw)

def supported(m):
    rules=m.get('rules_primary','').lower()
    # Some finalized markets omit the index value; their official result is
    # still a usable label. Cross-check the index only when it is published.
    try:
        value=m.get('expiration_value')
        if value not in (None,''):
            if not re.fullmatch(r'(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d+)?',str(value)):return False
            observed=Decimal(str(value).replace(',',''));target=Decimal(str(m.get('floor_strike')))
            if not observed.is_finite() or not target.is_finite():return False
            if (observed>=target)!=(m.get('result')=='yes'):return False
    except InvalidOperation:return False
    literal=re.search(r'is at least ([0-9]+(?:\.[0-9]+)?), then the market resolves to yes\.',rules)
    comparison=('is at least the simple average of the sixty seconds' in rules
        or bool(literal and Decimal(literal[1])==Decimal(str(m.get('floor_strike')))))
    return (m.get('ticker','').startswith(SERIES+'-') and m.get('strike_type')=='greater_or_equal'
        and rules.startswith('if the simple average of the sixty seconds of cf benchmarks\' brti before ')
        and comparison
        and m.get('result') in ('yes','no') and m.get('status') in ('finalized','settled')
        and float(m.get('notional_value_dollars',0))==1 and float(m.get('floor_strike') or 0)>0
        and (stamp(m['close_time'])-stamp(m['open_time'])).total_seconds()==900
        and stamp(m['settlement_ts'])>=stamp(m['close_time'])
        and float(m['settlement_value_dollars'])==int(m['result']=='yes'))

def collect(cache,protocol=PROTOCOL):
    start,end=unix(protocol['start']),unix(protocol['end_exclusive'])
    series=cache.get(BASE+'/series/'+SERIES)['series']
    cutoff=cache.get(BASE+'/historical/cutoff')
    if unix(protocol['start'])<stamp(cutoff['market_settled_ts']).timestamp():
        raise ValueError('This pilot window requires the historical archive adapter; refusing incomplete live-only history')
    markets=[];cursor='';seen=set()
    while True:
        params=dict(series_ticker=SERIES,status='settled',min_close_ts=start,max_close_ts=end-1,limit=1000)
        if cursor:params['cursor']=cursor
        page=cache.get(BASE+'/markets?'+urlencode(params));markets.extend(page['markets'])
        cursor=page.get('cursor')
        if not cursor:break
        if cursor in seen:raise ValueError('Repeated cursor')
        seen.add(cursor)
    if len({m['ticker'] for m in markets})!=len(markets):raise ValueError('Duplicate markets')
    markets.sort(key=lambda m:m['close_time']);candles={}
    for i in range(0,len(markets),20):
        batch=markets[i:i+20]
        params=dict(market_tickers=','.join(m['ticker'] for m in batch),period_interval=1,
            start_ts=min(int(stamp(m['open_time']).timestamp()) for m in batch),
            end_ts=max(int(stamp(m['close_time']).timestamp()) for m in batch))
        page=cache.get(BASE+'/markets/candlesticks?'+urlencode(params))
        for item in page['markets']:candles[item['market_ticker']]=item['candlesticks']
        if i%200==0:print(f'Kalshi quotes: {min(i+20,len(markets))}/{len(markets)}',flush=True)
    bars={}
    # Include feature history before the first event. Do not invent missing bars.
    for a in range(start-3*3600,end,240*60):
        b=min(a+240*60,end)
        params=dict(start=iso(a),end=iso(b),granularity=60)
        data=cache.get('https://api.exchange.coinbase.com/products/BTC-USD/candles?'+urlencode(params))
        for row in data:
            if a<=row[0]<b:
                if row[0] in bars and bars[row[0]]!=row:raise ValueError('Conflicting Coinbase bars')
                bars[row[0]]=row
        if (a-(start-3*3600))%(24*3600)==0:print('Coinbase candles through '+iso(b),flush=True)
    return markets,candles,bars,series

def spot_features(bars,decision,close,strike,allowance=5):
    """Bar timestamps denote START; only completed, available bars enter features."""
    last=int((decision-allowance)//60)*60-60
    history=[bars.get(t) for t in range(last-119*60,last+1,60)]
    if any(r is None for r in history):raise ValueError('Missing contiguous Coinbase lookback')
    data=np.asarray(history,dtype=float)
    if data.shape!=(120,6) or not np.isfinite(data).all():raise ValueError('Invalid Coinbase bar')
    if np.any(data[:,1:5]<=0) or np.any(data[:,5]<0) or np.any(data[:,1]>data[:,2]):raise ValueError('Invalid OHLCV')
    if np.any(data[:,3:5]<data[:,1,None]) or np.any(data[:,3:5]>data[:,2,None]):raise ValueError('OHLC outside range')
    prices=data[:,4];returns=np.diff(np.log(prices))
    weights=.94**np.arange(59,-1,-1)
    sigma=max(float(np.sqrt(np.average(returns[-60:]**2,weights=weights))),1e-6)
    horizon=(close-(last+60))/60
    if horizon<2 or not math.isfinite(strike) or strike<=0:raise ValueError('Unsupported horizon or target')
    z=math.log(prices[-1]/strike)/(sigma*math.sqrt(horizon))
    hourly=2*math.pi*((decision%86400)/86400)
    x=[z,*[math.log(prices[-1]/prices[-1-k])/(sigma*math.sqrt(k)) for k in [1,5,15,60]],
        float(np.std(returns[-15:]))/sigma,float(np.mean(data[-5:,5]))/max(float(np.mean(data[:,5])),1e-8),
        math.log(data[-1,2]/data[-1,1])/sigma,math.sin(hourly),math.cos(hourly)]
    return dict(x=x,diffusion=float(np.clip(ndtr(z),1e-6,1-1e-6)),spot=float(prices[-1]),
        feature_time=iso(last+60),feature_available_at=iso(last+60+allowance))

def build_rows(markets,candles,bars,protocol=PROTOCOL):
    rows=[];excluded=Counter()
    for m in markets:
        try:
            if not supported(m):excluded['unsupported_rules_or_outcome']+=1;continue
            close=int(stamp(m['close_time']).timestamp());decision=close-protocol['decision_seconds_before_close']
            cs=sorted(candles.get(m['ticker'],[]),key=lambda c:c['end_period_ts'])
            eligible=[c for c in cs if c['end_period_ts']+5<=decision and valid_quote(c)]
            if not eligible:excluded['no_quote']+=1;continue
            c=eligible[-1];t=c['end_period_ts']
            if decision-t>65 or t<=stamp(m['open_time']).timestamp():excluded['stale_quote']+=1;continue
            bid,ask=valid_quote(c)
            if bid<=0 or ask>=1:excluded['no_two_sided_price']+=1;continue
            spot=spot_features(bars,decision,close,float(m['floor_strike']))
            later=[e for e in cs if e['end_period_ts']>t and e['end_period_ts']+5<close and valid_quote(e)]
            e=later[0] if later and later[0]['end_period_ts']-t==60 else None
            ex=valid_quote(e) if e else (None,None)
            execution_at=e['end_period_ts']+5 if e else None
            execution_spot=spot_features(bars,execution_at,close,float(m['floor_strike'])) if e else None
            previous=valid_quote(eligible[-2]) if len(eligible)>1 else (bid,ask)
            rows.append(dict(ticker=m['ticker'],event=m['event_ticker'],series=SERIES,group=iso(close)[:10],
                at=iso(decision),close_time=m['close_time'],settled_at=m['settlement_ts'],
                p=(bid+ask)/2,bid=bid,ask=ask,spread=ask-bid,prior_move=(bid+ask-sum(previous))/2,
                quote_at=iso(t),quote_available_at=iso(t+5),y=int(m['result']=='yes'),strike=m['floor_strike'],
                **spot,execution_at=iso(execution_at) if e else None,execution_bid=ex[0],execution_ask=ex[1],
                execution_spot=execution_spot,synthetic=False,source='Kalshi minute quotes and Coinbase BTC-USD',
                target_provenance='Opening target from final Kalshi metadata; historical publication assumed'))
        except (ValueError,KeyError,TypeError) as exc:excluded[str(exc)]+=1
    return rows,dict(excluded)

def split_rows(rows,protocol=PROTOCOL):
    a=unix(protocol['train_end_exclusive']);b=unix(protocol['validation_end_exclusive']);embargo=protocol['split_embargo_hours']*3600
    train=[r for r in rows if stamp(r['settled_at']).timestamp()<a]
    val=[r for r in rows if a+embargo<=stamp(r['at']).timestamp() and stamp(r['settled_at']).timestamp()<b]
    test=[r for r in rows if stamp(r['at']).timestamp()>=b+embargo]
    if min(map(len,[train,val,test]))<50:raise ValueError('Insufficient chronological data')
    return train,val,test

class BTCModels:
    def matrix(self,rows,kind):
        market=np.array([[float(logit(np.clip(r['p'],1e-6,1-1e-6))),r['spread']] for r in rows])
        spot=np.array([r['x'] for r in rows])
        return spot if kind=='BTCSignal' else market if kind=='price_only_control' else np.column_stack([spot,market])
    def fit(self,rows):
        y=np.array([r['y'] for r in rows]);self.models={}
        if len(set(y))!=2:raise ValueError('Training needs both labels')
        for name in ['BTCSignal','BTCMarketGuard','price_only_control']:
            scaler=StandardScaler().fit(self.matrix(rows,name));x=np.clip(scaler.transform(self.matrix(rows,name)),-10,10)
            if name=='BTCSignal':
                model=LogisticRegression(C=.1,max_iter=1000,random_state=1729).fit(x,y)
                coef=np.r_[model.intercept_,model.coef_[0]]
            else:
                x=np.column_stack([np.ones(len(x)),x]);offset=logit(np.clip([r['p'] for r in rows],1e-6,1-1e-6))
                def objective(w):
                    logits=offset+x@w
                    return float(np.mean(np.logaddexp(0,logits)-y*logits)+.01*np.sum(w*w)), x.T@(expit(logits)-y)/len(y)+.02*w
                result=minimize(objective,np.zeros(x.shape[1]),jac=True,method='L-BFGS-B')
                if not result.success:raise ValueError('Offset fit failed: '+result.message)
                coef=result.x
            self.models[name]=dict(mean=scaler.mean_.tolist(),scale=scaler.scale_.tolist(),coef=coef.tolist())
        return self
    def predict(self,rows,name):
        if name=='market':return np.array([r['p'] for r in rows])
        if name=='BTCVolatility':return np.array([r['diffusion'] for r in rows])
        m=self.models[name];x=np.clip((self.matrix(rows,name)-m['mean'])/m['scale'],-10,10)
        logits=np.column_stack([np.ones(len(x)),x])@m['coef']
        if name!='BTCSignal':logits+=logit(np.clip([r['p'] for r in rows],1e-6,1-1e-6))
        return np.clip(expit(logits),1e-6,1-1e-6)

def metrics(rows,p):
    y=np.array([r['y'] for r in rows]);p=np.clip(p,1e-6,1-1e-6)
    return dict(events=len(rows),days=len({r['group'] for r in rows}),accuracy=float(np.mean((p>=.5)==y)),
        log_loss=float(np.mean(-y*np.log(p)-(1-y)*np.log1p(-p))),brier=float(np.mean((p-y)**2)),
        calibration=[dict(count=int(mask.sum()),predicted=float(np.mean(p[mask])),observed=float(np.mean(y[mask])))
            for low in np.arange(0,1,.1) if (mask:=(p>=low)&(p<low+.1)).any()])

def paired_interval(rows,p,metric='brier'):
    """Cluster by UTC day; the six-day test interval remains very underpowered."""
    y=np.array([r['y'] for r in rows]);q=np.clip([r['p'] for r in rows],1e-6,1-1e-6);p=np.clip(p,1e-6,1-1e-6)
    diff=(p-y)**2-(q-y)**2 if metric=='brier' else (-y*np.log(p)-(1-y)*np.log1p(-p))-(-y*np.log(q)-(1-y)*np.log1p(-q))
    groups=sorted({r['group'] for r in rows});blocks=[diff[[r['group']==g for r in rows]] for g in groups]
    rng=np.random.default_rng(1729);values=[]
    for _ in range(3000):
        sample=[blocks[i] for i in rng.integers(0,len(blocks),len(blocks))];values.append(float(np.concatenate(sample).mean()))
    return dict(model_minus_market=float(diff.mean()),day_cluster_95_interval=np.quantile(values,[.025,.975]).tolist(),days=len(groups))

def execution_rows(rows):
    # Forecast is recomputed from completed inputs available at execution, avoiding
    # the weather experiment's stale forecast versus later quote mismatch.
    out=[]
    for r in rows:
        if not r['execution_at'] or not 0<r['execution_bid']<=r['execution_ask']<1:continue
        bid,ask=r['execution_bid'],r['execution_ask']
        out.append(dict(r,at=r['execution_at'],p=(bid+ask)/2,bid=bid,ask=ask,spread=ask-bid,
            quote_at=iso(stamp(r['execution_at']).timestamp()-PROTOCOL['publication_allowance_seconds']),
            quote_available_at=r['execution_at'],
            prior_move=(bid+ask)/2-r['p'],**r['execution_spot']))
    return out

def pnl_interval(result,days=None):
    ledger=result['ledger'];groups=sorted(set(days) if days is not None else {r['day'] for r in ledger})
    if not groups:return None
    # Net cents per filled contract; resample calendar-day trade blocks, not fills.
    blocks=[(sum(r['pnl'] for r in ledger if r['day']==g),sum(r['quantity'] for r in ledger if r['day']==g),
             sum(r['cost'] for r in ledger if r['day']==g)) for g in groups]
    rng=np.random.default_rng(1729);values=[];returns=[];zero=0
    for _ in range(3000):
        sample=[blocks[i] for i in rng.integers(0,len(blocks),len(blocks))]
        pnl=sum(x[0] for x in sample);quantity=sum(x[1] for x in sample);cost=sum(x[2] for x in sample)
        if not quantity or not cost:zero+=1;continue
        values.append(pnl/quantity);returns.append(pnl/cost)
    return dict(net_per_contract_95_interval=np.quantile(values,[.025,.975]).tolist() if values else None,
        return_on_deployed_95_interval=np.quantile(returns,[.025,.975]).tolist() if returns else None,
        calendar_days=len(groups),trade_days=len({r['day'] for r in ledger}),zero_trade_resamples=zero,
        note='Exploratory day-block ratios including no-trade days; undefined zero-trade ratios excluded. Fixed historical fills, not a policy replay or reliable six-day edge test.')

def run(root,output,offline=False):
    root=Path(root);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    frozen=output/'protocol.json'
    if frozen.exists() and json.loads(frozen.read_text())!=PROTOCOL:raise ValueError('Protocol changed; use a new named experiment and disclose reuse')
    save(frozen,PROTOCOL)  # Freeze before collecting or evaluating any outcomes.
    reused=(output/'report.json').exists()
    cache=Cache(root,offline);markets,candles,bars,series=collect(cache)
    rows,excluded=build_rows(markets,candles,bars);train,val,test=split_rows(rows)
    model=BTCModels().fit(train);save(output/'model.json',dict(features=FEATURES,models=model.models))
    report=dict(protocol=PROTOCOL,test_previously_evaluated=reused,downloaded_markets=len(markets),excluded=excluded,
        rows=len(rows),split_counts=dict(train=len(train),validation=len(val),test=len(test)),series_fee_snapshot=series,
        validation={},test={},no_proven_edge=True)
    for name in PROTOCOL['models']:
        p=model.predict(test,name);entry=dict(metrics=metrics(test,p),brier_difference=paired_interval(test,p),
            log_loss_difference=paired_interval(test,p,'log_loss'),scenarios={})
        for scenario,data in [('same_quote_optimistic',[dict(r,execution_at=r['at'],execution_bid=r['bid'],execution_ask=r['ask']) for r in test]),
            ('next_minute_recomputed',execution_rows(test))]:
            probabilities=model.predict(data,name)
            for slip in PROTOCOL['slippage_dollars']:
                result=simulate(data,probabilities,policy='bankroll_1_percent',slippage=slip,bankroll=1000,min_edge=.04)
                entry['scenarios'][scenario+'_'+str(slip)]=dict(result,
                    execution_recomputed=scenario=='next_minute_recomputed',
                    evaluated_events=len(data),risk_group='BTC UTC day',
                    uncertainty=pnl_interval(result,{r['group'] for r in test}))
        report['test'][name]=entry;report['validation'][name]=metrics(val,model.predict(val,name))
    for split,items in [('train',train),('validation',val),('test',test)]:
        (output/(split+'.jsonl')).write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in items),encoding='utf8')
    save(output/'report.json',report)
    app=Path(__file__).resolve().parent.parent
    source_paths=['helper/btc_research.py','helper/sizing.py','helper/core.py','helper/live.py',
                  'helper/history.py','tests/test_btc_research.py','requirements.txt']
    save(output/'manifest.json',dict(sources=cache.used,
        code_files={name:digest((app/name).read_bytes()) for name in source_paths},
        files={p.name:digest(p.read_bytes()) for p in output.iterdir() if p.name!='manifest.json' and p.is_file()}))
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--cache',required=True);parser.add_argument('--output',required=True);parser.add_argument('--offline',action='store_true')
    args=parser.parse_args();report=run(args.cache,args.output,args.offline)
    print(json.dumps(dict(split_counts=report['split_counts'],results={k:v['metrics'] for k,v in report['test'].items()}),indent=2))
