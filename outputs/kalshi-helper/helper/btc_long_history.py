"""External BTC history pretraining and separate real-Kalshi transfer evaluation."""
import argparse
from collections import Counter
import csv
import io
import json
from pathlib import Path
from urllib.request import urlopen
import zipfile

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .btc_calibration import read_development
from .btc_regime import BTCRegime
from .btc_research import BTCModels, digest, execution_rows, iso, metrics, pnl_interval, save, spot_features, unix
from .sizing import simulate

PROTOCOL=dict(version='btc-binance-history-v1',months=[f'2025-{m:02d}' for m in range(1,7)],
    train='2025-01-01 through 2025-04-30',validation='2025-05-01 through 2025-05-31',
    test='2025-06-01 through 2025-06-30',embargo_seconds=7200,
    task='At five minutes before a quarter-hour, predict final Binance minute close >= minute close immediately before interval opening',
    target='BTCUSDT close-price proxy; NOT BRTI averaging and NOT a Kalshi contract',
    models=['volatility','logistic','regime'],selection='Fixed model settings; no hyperparameter or strategy selection',
    transfer='Existing September Kalshi validation rows only; already inspected development data',
    execution='Recomputed next-minute input, fixed 4-cent edge, 2-cent slippage, 0.07 fee assumption',
    limitations=['USDT differs from USD','Exchange closes differ from BRTI averages',
        'Binance-to-Coinbase feature distribution shift','Historical depth unavailable',
        'Proxy test is not a trading-profit test','Transfer dates reused; no independent profitability claim'])


def timestamp(value):
    value=int(value)
    unit=1000000 if value>=10**15 else 1000 if value>=10**12 else None
    if unit is None or value%unit:raise ValueError('Expected exact second in documented millisecond/microsecond units')
    return value//unit


def parse_archive(raw,month):
    result={}
    start=unix(month+'-01');year,m=map(int,month.split('-'))
    end=unix(f'{year+int(m==12):04d}-{m%12+1:02d}-01')
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        names=archive.namelist()
        if len(names)!=1 or not names[0].endswith('.csv'):raise ValueError('Unexpected archive members')
        if archive.getinfo(names[0]).file_size>50*1024*1024:raise ValueError('Archive expands beyond limit')
        with archive.open(names[0]) as stream:
            for row in csv.reader(io.TextIOWrapper(stream,encoding='utf8')):
                t=timestamp(row[0])
                if not start<=t<end or t%60:raise ValueError('Candle outside declared month or minute boundary')
                unit=1000000 if int(row[0])>=10**15 else 1000
                if int(row[6])!=(t+60)*unit-1:raise ValueError('Candle is not exactly one minute')
                # Normalize to the existing Coinbase-format feature adapter.
                values=[t,float(row[3]),float(row[2]),float(row[1]),float(row[4]),float(row[5])]
                a=np.array(values)
                if not np.isfinite(a).all() or not 0<a[1]<=a[3]<=a[2] or not a[1]<=a[4]<=a[2] or a[5]<0:
                    raise ValueError('Invalid external OHLCV')
                if t in result:raise ValueError('Duplicate external minute')
                result[t]=values
    return result


def download(root,month,offline=False):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    name=f'BTCUSDT-1m-{month}.zip';path=root/name;checksum=root/(name+'.CHECKSUM')
    url='https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1m/'+name
    if not path.exists() or not checksum.exists():
        if offline:raise ValueError('Offline monthly archive missing')
        with urlopen(url+'.CHECKSUM',timeout=60) as r: expected=r.read(4096)
        with urlopen(url,timeout=60) as r:raw=r.read(20*1024*1024+1)
        if len(raw)>20*1024*1024:raise ValueError('Archive exceeds limit')
        if digest(raw)!=expected.decode().split()[0]:raise ValueError('Source checksum mismatch')
        path.write_bytes(raw);checksum.write_bytes(expected)
    raw=path.read_bytes()
    if digest(raw)!=checksum.read_text().split()[0]:raise ValueError('Cached archive checksum mismatch')
    return parse_archive(raw,month),dict(url=url,sha256=digest(raw),bytes=len(raw))


def build_rows(bars):
    rows=[];excluded=Counter()
    for close in range(unix('2025-01-01')+900,unix('2025-07-01'),900):
        try:
            target=bars[close-960][4];final=bars[close-60][4]
            feature=spot_features(bars,close-300,close,target)
            rows.append(dict(ticker=f'BINANCE-PROXY-{close}',group=iso(close)[:10],
                close_time=iso(close),at=iso(close-300),y=int(final>=target),
                strike=target,label_source='Binance close proxy, not Kalshi',**feature))
        except (KeyError,ValueError):excluded['missing_or_invalid_history']+=1
    return rows,dict(excluded)


def run(cache,source,output,offline=False):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if (output/'protocol.json').exists() and json.loads((output/'protocol.json').read_text())!=PROTOCOL:
        raise ValueError('Changed protocol; use a new experiment')
    repeated=(output/'report.json').exists();save(output/'protocol.json',PROTOCOL)
    bars={};sources={}
    for month in PROTOCOL['months']:
        block,record=download(cache,month,offline);bars.update(block);sources[month]=record
        print('Loaded '+month+': '+str(len(block))+' minutes',flush=True)
    rows,excluded=build_rows(bars)
    train=[r for r in rows if r['close_time']<iso(unix('2025-05-01'))]
    val=[r for r in rows if iso(unix('2025-05-01')+7200)<=r['at'] and r['close_time']<iso(unix('2025-06-01'))]
    test=[r for r in rows if r['at']>=iso(unix('2025-06-01')+7200)]
    if min(map(len,[train,val,test]))<100:raise ValueError('Insufficient historical partitions')
    logistic=make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=1000,random_state=1729)).fit(
        [r['x'] for r in train],[r['y'] for r in train])
    regime=BTCRegime().fit(train)
    def predict(data):
        return dict(volatility=np.array([r['diffusion'] for r in data]),
            logistic=logistic.predict_proba([r['x'] for r in data])[:,1],regime=regime.predict(data))
    report=dict(protocol=PROTOCOL,proxy_test_previously_evaluated=repeated,minutes=len(bars),exclusions=excluded,
        split_counts=dict(train=len(train),validation=len(val),test=len(test)),proxy_validation={},proxy_test={})
    for name,data in [('proxy_validation',val),('proxy_test',test)]:
        report[name]={n:metrics(data,p) for n,p in predict(data).items()}
        report[name]['constant_training_rate']=metrics(data,np.full(len(data),np.mean([r['y'] for r in train])))
    original_train,transfer=read_development(source);execution=execution_rows(transfer)
    report['kalshi_development_transfer']={}
    transfer_p=predict(transfer);execution_p=predict(execution)
    for name in transfer_p:
        result=simulate(execution,execution_p[name],policy='bankroll_1_percent',slippage=.02,bankroll=1000,min_edge=.04)
        result['execution_recomputed']=True;result['uncertainty']=pnl_interval(result,{r['group'] for r in transfer})
        report['kalshi_development_transfer'][name]=dict(scores=metrics(transfer,transfer_p[name]),paper=result)
    report['kalshi_market_scores']=metrics(transfer,np.array([r['p'] for r in transfer]))
    # A full retrospective month is possible without waiting in real time. These
    # September dates were inspected previously; this is not a fresh holdout.
    test_path=Path(source)/'test.jsonl'
    expected=json.loads((Path(source)/'manifest.json').read_text())['files']['test.jsonl']
    if digest(test_path.read_bytes())!=expected:raise ValueError('Kalshi replay input hash mismatch')
    september_test=[json.loads(line) for line in test_path.read_text().splitlines()]
    month=sorted(original_train+transfer+september_test,key=lambda r:r['at'])
    if len({r['ticker'] for r in month})!=len(month):raise ValueError('Duplicate month replay contract')
    month_execution=execution_rows(month);month_p=predict(month);month_ep=predict(month_execution)
    report['retrospective_september_replay']=dict(previously_inspected=True,events=len(month),
        days=len({r['group'] for r in month}),omitted_original_split_boundary_rows=18,
        market_scores=metrics(month,np.array([r['p'] for r in month])),
        note='External models trained only on January-April. Reused September evaluation; no strategy tuning. No historical depth.',models={})
    for name in month_p:
        paper=simulate(month_execution,month_ep[name],policy='bankroll_1_percent',slippage=.02,bankroll=1000,min_edge=.04)
        paper['execution_recomputed']=True;paper['uncertainty']=pnl_interval(paper,{r['group'] for r in month})
        paper['uncertainty']['note']='Exploratory 30-day block ratios, not a full resimulation of bankroll paths; ignores cross-day dependence and prior experiment selection.'
        report['retrospective_september_replay']['models'][name]=dict(scores=metrics(month,month_p[name]),paper=paper)
    old=BTCModels().fit(original_train);old_regime=BTCRegime().fit(original_train)
    report['original_training_comparison']={}
    for name,p,ep in [('BTCSignal',old.predict(transfer,'BTCSignal'),old.predict(execution,'BTCSignal')),
                      ('BTCRegime',old_regime.predict(transfer),old_regime.predict(execution))]:
        paper=simulate(execution,ep,policy='bankroll_1_percent',slippage=.02,bankroll=1000,min_edge=.04)
        report['original_training_comparison'][name]=dict(scores=metrics(transfer,p),paper=paper)
    report['deployment_allowed']=False
    save(output/'report.json',report)
    np.savez_compressed(output/'proxy_dataset.npz',x=np.array([r['x'] for r in rows]),y=np.array([r['y'] for r in rows]),
        close_time=np.array([r['close_time'] for r in rows]))
    save(output/'manifest.json',dict(sources=sources,code_sha256=digest(Path(__file__).read_bytes()),
        kalshi_inputs={n:digest((Path(source)/n).read_bytes()) for n in ['train.jsonl','validation.jsonl','test.jsonl']},
        outputs={p.name:digest(p.read_bytes()) for p in output.iterdir() if p.name!='manifest.json' and p.is_file()}))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--cache',required=True)
    parser.add_argument('--source',default='reports/btc15m-pilot-v1');parser.add_argument('--output',default='reports/btc-binance-history-v1')
    parser.add_argument('--offline',action='store_true');args=parser.parse_args()
    r=run(args.cache,args.source,args.output,args.offline)
    print(json.dumps(dict(split_counts=r['split_counts'],proxy_test={n:{k:v[k] for k in ['log_loss','brier','accuracy']} for n,v in r['proxy_test'].items()},
        transfer={n:dict(log_loss=v['scores']['log_loss'],brier=v['scores']['brier'],pnl=v['paper']['pnl']) for n,v in r['kalshi_development_transfer'].items()}),indent=2))
