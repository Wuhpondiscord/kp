"""New July BTC cohort and timestamped aggressor-flow features; public data only."""
import argparse
from collections import Counter
import csv
import io
import json
import math
from pathlib import Path
from urllib.parse import urlencode, quote
from urllib.request import urlopen
import zipfile

import numpy as np
from .btc_research import Cache, BASE, SERIES, build_rows, iso, unix, save, digest
from .btc_long_history import timestamp, parse_archive
from .core import stamp

PROTOCOL = dict(version='btc-july-flow-v1', start='2026-07-08', train_end='2026-07-16',
    validation_end='2026-07-21', end='2026-08-01', embargo_seconds=7200,
    sampling='Contracts closing at whole UTC hours, fixed before downloading outcomes; all other quarter-hours excluded',
    holdout='July 21-31 reserved; model/protocol frozen before holdout reader is permitted',
    features=['taker_imbalance_5m','taker_imbalance_15m','binance_coinbase_basis_bps','basis_change_5m_bps'],
    source='Binance BTCUSDT minute taker-buy base volume and Coinbase BTCUSD closes; not order-book depth',
    availability='Completed bars plus 5-second assumed publication allowance',
    model='Market log-odds offset + standardized four new features; L2 .1, coefficients bounded [-.25,.25], fixed; market-only control',
    trading='Bankroll 1000; 1% risk; existing shared-day limits; edge .04; slippage .02 primary and 0/.05 sensitivity; .07 fee',
    limitations=['USDT/USD basis includes stablecoin effects','No historical depth or measured fill latency',
                 'Final metadata opening target publication assumed','Small hourly sample; not all BTC15M opportunities'])


def freeze(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    path=output/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=PROTOCOL:raise ValueError('Use a new protocol directory')
    if not path.exists():save(path,PROTOCOL)


def parse_flow(raw, month):
    # Reuse OHLC/minute/dedup validation before reading the additional columns.
    bars=parse_archive(raw,month);flow={}
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        with archive.open(archive.namelist()[0]) as stream:
            for r in csv.reader(io.TextIOWrapper(stream,encoding='utf8')):
                t=timestamp(r[0]);volume=float(r[5]);buy=float(r[9])
                if not math.isfinite(buy) or not 0<=buy<=volume:raise ValueError('Invalid taker-buy base volume')
                flow[t]=dict(close=bars[t][4],volume=volume,buy=buy)
    return flow


def new_features(flow,coinbase,at):
    at=stamp(at).timestamp();last=int((at-5)//60)*60-60
    history=[flow.get(t) for t in range(last-14*60,last+1,60)]
    if any(r is None for r in history):raise ValueError('Missing Binance flow window')
    def imbalance(n):
        total=sum(r['volume'] for r in history[-n:])
        if total<=0:raise ValueError('Zero flow volume')
        return 2*sum(r['buy'] for r in history[-n:])/total-1
    def basis(t):
        if t not in coinbase or t not in flow:raise ValueError('Missing cross-exchange timestamp')
        return math.log(flow[t]['close']/coinbase[t][4])*10000
    x=[imbalance(5),imbalance(15),basis(last),basis(last)-basis(last-300)]
    if not np.isfinite(x).all():raise ValueError('Nonfinite flow features')
    return dict(flow_x=x,flow_feature_time=iso(last+60),flow_available_at=iso(last+65))


def history_markets(cache):
    markets=[];cursor='';seen=set();tickers=set()
    # API filters are mutually exclusive: filter series server-side, dates locally.
    for _ in range(100):
        params=dict(series_ticker=SERIES,limit=1000)
        if cursor:params['cursor']=cursor
        page=cache.get(BASE+'/historical/markets?'+urlencode(params))
        block=page['markets']
        for m in block:
            if m['ticker'] in tickers:raise ValueError('Duplicate historical contract')
            tickers.add(m['ticker']);t=stamp(m['close_time']).timestamp()
            if unix(PROTOCOL['start'])<=t<unix(PROTOCOL['end']) and t%3600==0:markets.append(m)
        # Do not assume pagination order when deciding whether the interval is complete.
        cursor=page.get('cursor')
        if not cursor:return sorted(markets,key=lambda m:m['close_time'])
        if cursor in seen:raise ValueError('Repeated archive cursor')
        seen.add(cursor)
    raise ValueError('Archive pagination exceeded bound; incomplete cohort')


def collect(cache_path,output,offline=False):
    output=Path(output);freeze(output);cache=Cache(cache_path,offline)
    series=cache.get(BASE+'/series/'+SERIES)['series']
    changes=cache.get(BASE+'/series/fee_changes?'+urlencode(dict(series_ticker=SERIES,show_historical='true')))
    save(output/'fee-evidence.json',dict(series=series,changes=changes,
        assessed_coefficient=.07,confidence='Current series corroboration, not a complete historical event-override audit',
        schedule='https://kalshi.com/docs/kalshi-fee-schedule.pdf',effective='2026-07-07',
        precision_source='https://docs.kalshi.com/getting_started/fee_rounding',
        slippage='2 cents remains an assumption; 0/5 cents are sensitivity scenarios'))
    archive=Path(cache_path)/'BTCUSDT-1m-2026-07.zip';check=Path(str(archive)+'.CHECKSUM')
    url='https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1m/'+archive.name
    if not archive.exists() or not check.exists():
        if offline:raise ValueError('Missing Binance archive')
        with urlopen(url,timeout=45) as r:archive.write_bytes(r.read(20*1024*1024+1))
        with urlopen(url+'.CHECKSUM',timeout=30) as r:check.write_bytes(r.read(4096))
    raw=archive.read_bytes()
    if len(raw)>20*1024*1024 or digest(raw)!=check.read_text().split()[0]:raise ValueError('Archive checksum mismatch')
    flow=parse_flow(raw,'2026-07')
    markets=history_markets(cache);print('Fixed hourly sample:',len(markets),flush=True)
    if not markets:raise ValueError('No historical BTC markets in reserved interval')
    candles={}
    for i,m in enumerate(markets):
        params=dict(start_ts=int(stamp(m['open_time']).timestamp()),end_ts=int(stamp(m['close_time']).timestamp()),period_interval=1)
        page=cache.get(BASE+'/historical/markets/'+quote(m['ticker'],safe='')+'/candlesticks?'+urlencode(params))
        candles[m['ticker']]=page['candlesticks']
        if i%48==0:print('Archived quotes:',i+1,'/',len(markets),flush=True)
    bars={};start=unix(PROTOCOL['start'])-3*3600;end=unix(PROTOCOL['end'])
    for a in range(start,end,240*60):
        params=dict(start=iso(a),end=iso(min(a+240*60,end)),granularity=60)
        for r in cache.get('https://api.exchange.coinbase.com/products/BTC-USD/candles?'+urlencode(params)):
            if a<=r[0]<min(a+240*60,end):bars[int(r[0])]=r
    rows,excluded=build_rows(markets,candles,bars);failures=Counter();groups=dict(train=[],validation=[],holdout=[])
    for r in rows:
        try:
            if not r['execution_at']:raise ValueError('Missing execution quote')
            r.update(new_features(flow,bars,r['execution_at']))
            t=stamp(r['close_time']).timestamp();at=stamp(r['execution_at']).timestamp()
            if t<unix(PROTOCOL['train_end']):part='train'
            elif t<unix(PROTOCOL['validation_end']):
                if at<unix(PROTOCOL['train_end'])+7200:raise ValueError('Validation embargo')
                part='validation'
            else:
                if at<unix(PROTOCOL['validation_end'])+7200:raise ValueError('Holdout embargo')
                part='holdout'
            groups[part].append(r)
        except (ValueError,KeyError) as exc:failures[str(exc)]+=1
    for name,part in groups.items():
        (output/(name+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in part),encoding='utf8')
    save(output/'manifest.json',dict(protocol_sha256=digest((output/'protocol.json').read_bytes()),
        counts={n:len(r) for n,r in groups.items()},sampled_markets=len(markets),row_exclusions=excluded,feature_exclusions=dict(failures),
        files={n+'.jsonl':digest((output/(n+'.jsonl')).read_bytes()) for n in groups},
        binance=dict(url=url,sha256=digest(raw)),requests=cache.used,
        holdout_status='Sealed from model evaluation; acquisition/structural validation only'))
    return {n:len(r) for n,r in groups.items()}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--cache',required=True)
    parser.add_argument('--output',default='reports/btc-july-flow-v1');parser.add_argument('--offline',action='store_true')
    args=parser.parse_args();print(collect(args.cache,args.output,args.offline))
