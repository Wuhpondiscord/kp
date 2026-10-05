"""Public one-second BTC bars with explicit availability and immutable checksums."""
import argparse
import csv
from datetime import datetime,timezone
import io
import json
import math
from pathlib import Path
from urllib.request import urlopen
import zipfile
import numpy as np
from .core import stamp
from .btc_flow_followup import read_rows,split,evaluate,PROTOCOL as DATA_PROTOCOL
from .btc_flow_model import FlowCorrection
from .btc_anchor_study import FixedAnchor
from .btc_refinement import logits
from .btc_research import digest,save,iso,paired_interval
from .btc_stability_model import paper_scenarios

PROTOCOL=dict(version='btc-microdata-v1',source='Binance public daily BTCUSDT one-second spot klines',
    availability_seconds=5,features=['imbalance_15s','return_15s_bps','imbalance_5s_minus_60s','return_5s_bps'],
    variants=['micro_only','flow_plus_micro'],latency_stress_seconds=60,
    model='Existing fixed market-logit correction, training-only scaling, L2 .1, bounds +/-.25; no parameter search',
    evaluation='Same development contracts, chronology, fees, 1% risk, .04 edge and .02 slippage',
    limits='Event timestamps plus assumed publication allowance, not measured receipt times; trades aggregated into bars, not order-book depth; no holdout/deployment')


def freeze(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);path=root/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=PROTOCOL:raise ValueError('Protocol changed')
    if not path.exists():save(path,PROTOCOL)


def archive(cache,day,offline=False):
    cache=Path(cache);cache.mkdir(parents=True,exist_ok=True);name=f'BTCUSDT-1s-{day}.zip';path=cache/name
    check=cache/(name+'.CHECKSUM');url='https://data.binance.vision/data/spot/daily/klines/BTCUSDT/1s/'+name
    if not path.exists() or not check.exists():
        if offline:raise ValueError('Missing one-second archive: '+day)
        with urlopen(url+'.CHECKSUM',timeout=40) as r:expected=r.read(4096)
        with urlopen(url,timeout=60) as r:raw=r.read(25*1024*1024+1)
        if len(raw)>25*1024*1024 or digest(raw)!=expected.decode().split()[0]:raise ValueError('Archive size/checksum error')
        path.write_bytes(raw);check.write_bytes(expected)
    raw=path.read_bytes()
    if digest(raw)!=check.read_text().split()[0]:raise ValueError('Cached checksum mismatch')
    return raw,dict(url=url,sha256=digest(raw),bytes=len(raw))


def parse_seconds(raw,day,needed):
    start=int(datetime.fromisoformat(day).replace(tzinfo=timezone.utc).timestamp());result={};previous=None;count=0
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        if len(z.infolist())!=1 or z.infolist()[0].file_size>80*1024*1024:raise ValueError('Invalid archive structure')
        with z.open(z.infolist()[0]) as f:
            for row in csv.reader(io.TextIOWrapper(f,encoding='utf8')):
                if len(row)!=12:raise ValueError('Unexpected second-bar schema')
                micros=int(row[0]);t=micros//1000000
                if micros%1000000 or not start<=t<start+86400 or (previous is not None and t!=previous+1):raise ValueError('Noncontiguous/invalid second timestamp')
                if int(row[6])!=micros+999999:raise ValueError('Incorrect close timestamp')
                previous=t;count+=1
                o,h,l,c,volume=map(float,row[1:6]);buy=float(row[9]);trades=int(row[8])
                if not all(math.isfinite(v) for v in (o,h,l,c,volume,buy)) or not 0<l<=min(o,c)<=max(o,c)<=h or not 0<=buy<=volume or trades<0:
                    raise ValueError('Invalid second-bar values')
                if t in needed:result[t]=dict(close=c,volume=volume,buy=buy,trades=trades)
    if count!=86400 or previous!=start+86399:raise ValueError('Incomplete source day')
    return result


def micro_features(bars,at,delay=0):
    if delay not in (0,60):raise ValueError('Unregistered delay')
    end=math.floor(stamp(at).timestamp()-5-delay)
    history=[bars.get(t) for t in range(end-61,end)]
    if any(r is None for r in history):raise ValueError('Missing contiguous micro window')
    def imbalance(n):
        volume=sum(r['volume'] for r in history[-n:])
        return 2*sum(r['buy'] for r in history[-n:])/volume-1 if volume else 0.
    def ret(n):return math.log(history[-1]['close']/history[-1-n]['close'])*10000
    x=[imbalance(15),ret(15),imbalance(5)-imbalance(60),ret(5)]
    if not np.isfinite(x).all():raise ValueError('Invalid micro features')
    return dict(micro_x=x,micro_feature_time=iso(end),micro_available_at=iso(end+5),
        source_seconds=61,trades_60s=sum(r['trades'] for r in history[-60:]))


def collect(source,cache,output,offline=False):
    output=Path(output);freeze(output);rows=read_rows(Path(source));needed=set()
    for r in rows:
        end=math.floor(stamp(r['at']).timestamp()-5)
        needed.update(range(end-121,end))
    days=sorted({datetime.fromtimestamp(t,timezone.utc).date().isoformat() for t in needed});bars={};records={}
    for day in days:
        raw,records[day]=archive(cache,day,offline);bars.update(parse_seconds(raw,day,needed))
        print('Loaded second bars',day,flush=True)
    data={r['ticker']:{str(d):micro_features(bars,r['at'],d) for d in (0,60)} for r in rows}
    save(output/'features.json',data)
    save(output/'manifest.json',dict(data_sha256=digest((Path(source)/'development.jsonl').read_bytes()),
        features_sha256=digest((output/'features.json').read_bytes()),archives=records,rows=len(rows),matched_seconds=len(bars)))


class MicroAnchor(FlowCorrection):
    def __init__(self,combined=False):super().__init__(True);self.combined=combined
    def design(self,rows,fit=False):
        x=np.asarray([([*r['flow_x'],*r['micro_x']] if self.combined else r['micro_x']) for r in rows],dtype=float)
        if x.shape!=(len(rows),8 if self.combined else 4) or not np.isfinite(x).all():raise ValueError('Invalid micro vector')
        if fit:self.mean=x.mean(0);self.scale=np.maximum(x.std(0),1e-6)
        return logits([r['p'] for r in rows]),np.clip((x-self.mean)/self.scale,-5,5)
    def artifact(self):return dict(super().artifact(),combined=self.combined,kind='micro_anchor')
    @classmethod
    def load(cls,data):
        if data.get('kind')!='micro_anchor':raise ValueError('Wrong model type')
        model=cls(data['combined']);n=8 if model.combined else 4
        for key in ('mean','scale','coef'):
            a=np.asarray(data[key],dtype=float)
            if a.shape!=(n,) or not np.isfinite(a).all():raise ValueError('Invalid artifact')
            setattr(model,key,a)
        if np.any(model.scale<=0) or np.any(abs(model.coef)>.25):raise ValueError('Invalid model parameters')
        return model


def attach(rows,features,delay=0):
    result=[]
    for r in rows:
        f=features[r['ticker']][str(delay)];cutoff=stamp(r['at']).timestamp()-delay
        if not stamp(f['micro_feature_time']).timestamp()+5==stamp(f['micro_available_at']).timestamp() or not cutoff-1<stamp(f['micro_available_at']).timestamp()<=cutoff:
            raise ValueError('Invalid feature availability')
        result.append(dict(r,**f))
    return result


def audit_aggregation(source,second_cache,minute_cache,output):
    """Compare disjoint 60-second bars with independently archived minute totals."""
    from .btc_long_history import download
    from .btc_new_history import parse_flow
    rows=read_rows(Path(source));needed=set();ends=[]
    for r in rows:
        end=math.floor(stamp(r['at']).timestamp()-5)
        if end%60:raise ValueError('Aggregation audit requires minute-aligned decisions')
        ends.append(end);needed.update(range(end-60,end))
    seconds={}
    for day in sorted({datetime.fromtimestamp(t,timezone.utc).date().isoformat() for t in needed}):
        raw,_=archive(second_cache,day,offline=True);seconds.update(parse_seconds(raw,day,needed))
    minutes={};sources={}
    for month in ('2026-06','2026-07'):
        _,sources[month]=download(minute_cache,month,offline=True)
        minutes.update(parse_flow((Path(minute_cache)/f'BTCUSDT-1m-{month}.zip').read_bytes(),month))
    differences=[]
    for end in ends:
        bars=[seconds[t] for t in range(end-60,end)];minute=minutes[end-60]
        differences.append([bars[-1]['close']-minute['close'],sum(r['volume'] for r in bars)-minute['volume'],sum(r['buy'] for r in bars)-minute['buy']])
    d=np.asarray(differences);limits=np.array([1e-8,1e-7,1e-7]);bad=np.any(abs(d)>limits,axis=1)
    report=dict(windows=len(ends),matching_windows=int((~bad).sum()),mismatching_windows=int(bad.sum()),
        fields=['close','base_volume','taker_buy_base_volume'],maximum_absolute_difference=np.max(abs(d),axis=0).tolist(),
        absolute_tolerances=limits.tolist(),minute_sources=sources,passed=not bool(bad.any()))
    save(Path(output)/'aggregation-audit.json',report)
    if bad.any():raise ValueError('Second/minute source aggregation mismatch; inspect audit before interpreting model results')
    return report


def run(source,previous,output):
    source=Path(source);previous=Path(previous);output=Path(output);freeze(output)
    manifest=json.loads((output/'manifest.json').read_text())
    if digest((source/'development.jsonl').read_bytes())!=manifest['data_sha256'] or digest((output/'features.json').read_bytes())!=manifest['features_sha256']:raise ValueError('Input hash mismatch')
    rows=read_rows(source);features=json.loads((output/'features.json').read_text());allrows=[];folds=[]
    preds={n:[] for n in ('market','fixed_anchor','micro_only','flow_plus_micro','combined_delayed_micro')}
    for start,end in DATA_PROTOCOL['folds']:
        train,test=split(rows,start,end);train=attach(train,features);test=attach(test,features);allrows.extend(test)
        models=dict(micro_only=MicroAnchor().fit(train),flow_plus_micro=MicroAnchor(True).fit(train))
        save(output/f'models-{start}.json',{n:m.artifact() for n,m in models.items()})
        control=FixedAnchor.load(json.loads((previous/f'models-{start}.json').read_text())['fixed_anchor'])
        current=dict(market=np.asarray([r['p'] for r in test]),fixed_anchor=control.predict(test),**{n:m.predict(test) for n,m in models.items()},
            combined_delayed_micro=models['flow_plus_micro'].predict(attach(test,features,60)))
        for n,p in current.items():preds[n].extend(p.tolist())
        folds.append(dict(start=start,end=end,models={n:evaluate(test,p) for n,p in current.items()}))
    aggregate={n:evaluate(allrows,np.asarray(p)) for n,p in preds.items()}
    controls=[dict(r,p=p) for r,p in zip(allrows,preds['fixed_anchor'])]
    for n in ('micro_only','flow_plus_micro','combined_delayed_micro'):aggregate[n]['paired_vs_fixed_anchor']=paired_interval(controls,np.asarray(preds[n]))
    save(output/'report.json',dict(protocol=PROTOCOL,rerun=(output/'report.json').exists(),holdout_loaded=False,deployment_allowed=False,
        folds=folds,aggregate=aggregate,trading={n:paper_scenarios(allrows,np.asarray(p)) for n,p in preds.items() if n!='market'},
        code_hashes={p.name:digest(p.read_bytes()) for p in [Path(__file__),*[Path(__file__).with_name(n) for n in
            ('btc_flow_followup.py','btc_flow_model.py','btc_anchor_study.py','btc_research.py','btc_stability_model.py','sizing.py','core.py','live.py')]]}))
    for n,v in aggregate.items():print(n,v['scores']['brier'],v['scores']['log_loss'],v['paper']['pnl'],v['paper']['entries'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',default='reports/btc-flow-followup-v1');p.add_argument('--previous',default='reports/btc-anchor-study-v1')
    p.add_argument('--output',default='reports/btc-microdata-v1');p.add_argument('--cache');p.add_argument('--offline',action='store_true');a=p.parse_args()
    if a.cache:collect(a.source,a.cache,a.output,a.offline)
    run(a.source,a.previous,a.output)
