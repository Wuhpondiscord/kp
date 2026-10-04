"""Post-outcome source and edge diagnostics. Never usable as forecast features."""
import argparse
import json
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

from .btc_calibration import read_development
from .btc_regime import BTCRegime, side_features
from .btc_research import BTCModels, digest, execution_rows, save
from .core import stamp


def load_bars(cache):
    bars={};sources={}
    for meta in sorted(Path(cache).glob('*.json')):
        record=json.loads(meta.read_text());url=urlparse(record.get('url',''))
        if url.netloc!='api.exchange.coinbase.com' or url.path!='/products/BTC-USD/candles':continue
        raw=meta.with_suffix('.raw').read_bytes()
        if digest(raw)!=record['sha256']:raise ValueError('Coinbase cache integrity failure')
        sources[meta.stem]=record['sha256']
        for row in json.loads(raw):
            t=int(row[0])
            if t in bars and bars[t]!=row:raise ValueError('Conflicting historical Coinbase candles')
            bars[t]=row
    if not bars:raise ValueError('No cached Coinbase candles')
    return bars,sources


def source_rows(rows,bars):
    """Final-minute OHLC is future information, used ONLY to diagnose labels."""
    out=[];missing=0
    for r in rows:
        close=int(stamp(r['close_time']).timestamp());target=float(r['strike'])
        final=bars.get(close-60);opening=bars.get(close-900-60)
        if final is None or opening is None:missing+=1;continue
        for bar in (opening,final):
            a=np.asarray(bar,dtype=float)
            if a.shape!=(6,) or not np.isfinite(a).all() or not 0<a[1]<=a[3]<=a[2] or not a[1]<=a[4]<=a[2]:
                raise ValueError('Invalid audit candle')
        proxy=int(final[4]>=target)
        # If the entire Coinbase range is on the opposite side, averaging its
        # prices within that minute cannot reconcile the official outcome.
        opposite=(r['y']==0 and final[1]>=target) or (r['y']==1 and final[2]<target)
        out.append(dict(ticker=r['ticker'],day=r['group'],official_yes=r['y'],
            final_coinbase_close_yes=proxy,close_label_disagreement=proxy!=r['y'],
            whole_range_opposes_outcome=bool(opposite),
            opening_close_minus_target_bps=10000*(opening[4]/target-1),
            opening_target_outside_coinbase_range=not opening[1]<=target<=opening[2],
            final_range_crosses_target=final[1]<target<=final[2]))
    return out,missing


def source_summary(rows,bars):
    out,missing=source_rows(rows,bars)
    if not out:raise ValueError('No matched source audit rows')
    difference=np.array([r['opening_close_minus_target_bps'] for r in out])
    return dict(events=len(out),missing=missing,
        close_label_disagreements=sum(r['close_label_disagreement'] for r in out),
        whole_range_opposes_outcome=sum(r['whole_range_opposes_outcome'] for r in out),
        opening_target_outside_coinbase_range=sum(r['opening_target_outside_coinbase_range'] for r in out),
        opening_close_minus_target_bps=dict(mean=float(difference.mean()),
            median_absolute=float(np.median(abs(difference))),p95_absolute=float(np.quantile(abs(difference),.95))),
        rows=out)


def edge_summary(rows,probabilities):
    if len(rows)!=len(probabilities):raise ValueError('Prediction length mismatch')
    bins={f'{lo:.2f}-{hi:.2f}':[] for lo,hi in [(.04,.08),(.08,.15),(.15,1.01)]}
    for r,p in zip(rows,probabilities):
        if not np.isfinite(p) or not 0<=p<=1:raise ValueError('Invalid probability')
        options=[side_features(r,float(p),s) for s in ['yes','no']]
        side=int(options[1][0][1]>options[0][0][1]);features,cost=options[side]
        edge=features[1];realized=(r['y'] if side==0 else 1-r['y'])-cost
        for lo,hi in [(.04,.08),(.08,.15),(.15,1.01)]:
            if lo<=edge<hi:bins[f'{lo:.2f}-{hi:.2f}'].append((edge,realized))
    return {name:dict(opportunities=len(values),
        predicted_net_per_contract=float(np.mean([x[0] for x in values])) if values else None,
        realized_net_per_contract=float(np.mean([x[1] for x in values])) if values else None)
        for name,values in bins.items()}


def run(source,cache,output):
    source,output=Path(source),Path(output);output.mkdir(parents=True,exist_ok=True)
    train,val=read_development(source);bars,sources=load_bars(cache)
    old=BTCModels().fit(train);model=BTCRegime().fit(train);execution=execution_rows(val)
    report=dict(purpose='Diagnostic only; final-minute values are post-outcome and never forecasting inputs',
        deployment_allowed=False,old_test_scored=False,
        training_source=source_summary(train,bars),validation_source=source_summary(val,bars),
        validation_edges={n:edge_summary(execution,p) for n,p in {
            'BTCSignal':old.predict(execution,'BTCSignal'),
            'BTCMarketGuard':old.predict(execution,'BTCMarketGuard'),
            'BTCRegime':model.predict(execution)}.items()},
        limitations=['Close proxy is not a final-minute mean',
            'Range mismatch may reflect venue basis, timestamps or source errors; cause not identified',
            'Edge bins are retrospective descriptions, not proposed strategy thresholds',
            'One hypothetical contract per opportunity; no bankroll sizing or executable depth',
            'Only development contracts scored; no fresh independent evidence'])
    save(output/'report.json',report)
    save(output/'manifest.json',dict(inputs={n:digest((source/n).read_bytes()) for n in ['train.jsonl','validation.jsonl']},
        coinbase_sources=sources,code_sha256=digest(Path(__file__).read_bytes()),
        report_sha256=digest((output/'report.json').read_bytes())))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',default='reports/btc15m-pilot-v1')
    parser.add_argument('--cache',required=True);parser.add_argument('--output',default='reports/btc-source-edge-audit')
    args=parser.parse_args();r=run(args.source,args.cache,args.output)
    print(json.dumps({k:({a:b for a,b in v.items() if a!='rows'} if k.endswith('_source') else v)
                      for k,v in r.items()},indent=2))
