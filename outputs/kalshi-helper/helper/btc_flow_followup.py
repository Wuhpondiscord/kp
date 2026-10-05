"""Fixed feature ablations on expanded development dates. Never opens July holdout."""
import argparse
from collections import Counter
import json
from pathlib import Path
from urllib.parse import urlencode, quote
import numpy as np
from .btc_research import Cache, BASE, SERIES, build_rows, iso, unix, save, digest, metrics, paired_interval
from .btc_long_history import download
from .btc_new_history import parse_flow, new_features
from .btc_flow_model import FlowCorrection, trade_uncertainty
from .btc_inputs import validated_execution_rows
from .core import stamp
from .sizing import simulate

MASKS = {'market_control': (), 'full_flow': (0,1,2,3), 'volume_only': (0,1),
         'basis_only': (2,3), 'basis_change_only': (3,)}
PROTOCOL = dict(version='btc-flow-followup-v1', start='2026-06-20', end='2026-07-21',
    folds=[['2026-06-28','2026-07-04'],['2026-07-04','2026-07-10'],
           ['2026-07-10','2026-07-16'],['2026-07-16','2026-07-21']],
    masks={k:list(v) for k,v in MASKS.items()}, embargo_seconds=7200,
    sampling='Whole UTC hour closes only; fixed before fetching earlier outcomes',
    model='Market-offset logistic correction, L2 .1 and coefficient bounds +/-.25 unchanged',
    execution='Recompute inputs at execution quote; 1% bankroll risk, .04 edge, .02 slippage, .07 fee',
    stress='Delay only new flow/basis inputs 60 seconds; retain current executable quote',
    inference='Exploratory rolling development, July dates reused; no promotion or holdout release',
    fee_caveat='June fee coefficient .07 is assumed, not verified historical overrides')


class Ablation(FlowCorrection):
    def __init__(self, indices):
        super().__init__(True)
        self.indices=tuple(indices)
        if len(set(self.indices))!=len(self.indices) or any(i not in range(4) for i in self.indices):
            raise ValueError('Invalid feature indices')
    def design(self, rows, fit=False):
        offset,x=super().design(rows,fit)
        return offset,x[:,[0,1]+[i+2 for i in self.indices]]
    def artifact(self):
        return dict(super().artifact(),indices=list(self.indices))
    @classmethod
    def load(cls,data):
        model=cls(data['indices']);model.mean=np.asarray(data['mean']);model.scale=np.asarray(data['scale'])
        model.coef=np.asarray(data['coef'])
        if model.coef.shape!=(2+len(model.indices),) or model.mean.shape!=(4,) or model.scale.shape!=(4,):
            raise ValueError('Invalid ablation artifact shape')
        if not all(np.isfinite(x).all() for x in (model.mean,model.scale,model.coef)) or np.any(model.scale<=0):
            raise ValueError('Invalid ablation artifact values')
        return model


def freeze(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    path=root/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=PROTOCOL:raise ValueError('Protocol mismatch')
    if not path.exists():save(path,PROTOCOL)


def collect(cache_path, root, offline=False):
    root=Path(root);freeze(root);cache=Cache(cache_path,offline)
    flow={};archives={}
    for month in ('2026-06','2026-07'):
        _,archives[month]=download(cache_path,month,offline)
        flow.update(parse_flow((Path(cache_path)/f'BTCUSDT-1m-{month}.zip').read_bytes(),month))
    markets=[];cursor='';seen=set();tickers=set()
    for _ in range(100):
        params=dict(series_ticker=SERIES,limit=1000)
        if cursor:params['cursor']=cursor
        page=cache.get(BASE+'/historical/markets?'+urlencode(params))
        for m in page['markets']:
            if m['ticker'] in tickers:raise ValueError('Duplicate contract')
            tickers.add(m['ticker']);t=stamp(m['close_time']).timestamp()
            if unix(PROTOCOL['start'])<=t<unix(PROTOCOL['end']) and t%3600==0:markets.append(m)
        cursor=page.get('cursor')
        if not cursor:break
        if cursor in seen:raise ValueError('Repeated cursor')
        seen.add(cursor)
    else:raise ValueError('Incomplete pagination')
    markets.sort(key=lambda m:m['close_time']);candles={}
    print('Selected markets',len(markets),flush=True)
    for i,m in enumerate(markets):
        params=dict(start_ts=int(stamp(m['open_time']).timestamp()),end_ts=int(stamp(m['close_time']).timestamp()),period_interval=1)
        candles[m['ticker']]=cache.get(BASE+'/historical/markets/'+quote(m['ticker'],safe='')+'/candlesticks?'+urlencode(params))['candlesticks']
        if i%48==0:print('Quotes',i+1,'/',len(markets),flush=True)
    bars={};end=unix(PROTOCOL['end'])
    for a in range(unix(PROTOCOL['start'])-3*3600,end,240*60):
        params=dict(start=iso(a),end=iso(min(a+240*60,end)),granularity=60)
        for r in cache.get('https://api.exchange.coinbase.com/products/BTC-USD/candles?'+urlencode(params)):
            if a<=r[0]<min(a+240*60,end):bars[int(r[0])]=r
    raw,excluded=build_rows(markets,candles,bars);rows=[];failures=Counter()
    for r in raw:
        try:
            if not r['execution_at']:raise ValueError('Missing execution quote')
            r.update(new_features(flow,bars,r['execution_at']))
            lag=new_features(flow,bars,iso(stamp(r['execution_at']).timestamp()-60))
            r.update(lag_flow_x=lag['flow_x'],lag_flow_available_at=lag['flow_available_at'])
            rows.append(r)
        except (ValueError,KeyError) as exc:failures[str(exc)]+=1
    path=root/'development.jsonl'
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf8')
    save(root/'manifest.json',dict(rows=len(rows),markets=len(markets),excluded=excluded,
        feature_exclusions=dict(failures),data_sha256=digest(path.read_bytes()),
        protocol_sha256=digest((root/'protocol.json').read_bytes()),archives=archives,requests=cache.used))
    print('Development rows',len(rows),flush=True)


def read_rows(root):
    root=Path(root);freeze(root);manifest=json.loads((root/'manifest.json').read_text())
    raw=(root/'development.jsonl').read_bytes()
    if digest(raw)!=manifest['data_sha256'] or digest((root/'protocol.json').read_bytes())!=manifest['protocol_sha256']:
        raise ValueError('Data/protocol hash mismatch')
    rows=[json.loads(line) for line in raw.decode().splitlines()]
    for r in rows:
        if not unix(PROTOCOL['start'])<=stamp(r['close_time']).timestamp()<unix(PROTOCOL['end']):
            raise ValueError('Outside development dates')
        if not stamp(r['flow_feature_time'])<=stamp(r['flow_available_at'])<=stamp(r['execution_at']):raise ValueError('Future input')
        if stamp(r['lag_flow_available_at']).timestamp()>stamp(r['execution_at']).timestamp()-60:raise ValueError('Invalid lag')
        if len(r['lag_flow_x'])!=4 or not np.isfinite(r['lag_flow_x']).all():raise ValueError('Invalid lag vector')
    return sorted(validated_execution_rows(rows),key=lambda r:r['at'])


def split(rows,start,end):
    a,b=unix(start),unix(end)
    train=[r for r in rows if stamp(r['close_time']).timestamp()<a and stamp(r['settled_at']).timestamp()<a]
    test=[r for r in rows if a<=stamp(r['close_time']).timestamp()<b and
          stamp(r['at']).timestamp()>=a+7200 and stamp(r['settled_at']).timestamp()<b]
    if len(train)<50 or not test:raise ValueError('Insufficient fold')
    return train,test


def evaluate(rows,p):
    paper=simulate(rows,p,policy='bankroll_1_percent',slippage=.02,bankroll=1000,min_edge=.04)
    paper['uncertainty']=trade_uncertainty(paper,{r['group'] for r in rows})
    return dict(scores=metrics(rows,p),paired_brier=paired_interval(rows,p),paper=paper)


def run(root):
    root=Path(root);rows=read_rows(root);repeated=(root/'report.json').exists()
    folds=[];allrows=[];preds={n:[] for n in ['market',*MASKS]};lags={n:[] for n in MASKS}
    for start,end in PROTOCOL['folds']:
        train,test=split(rows,start,end);allrows.extend(test)
        models={n:Ablation(mask).fit(train) for n,mask in MASKS.items()}
        # Save fitted parameters before inspecting this fold's scores.
        save(root/f'models-{start}.json',{n:m.artifact() for n,m in models.items()})
        delayed=[dict(r,flow_x=r['lag_flow_x']) for r in test]
        current={'market':np.array([r['p'] for r in test]),**{n:m.predict(test) for n,m in models.items()}}
        lagged={n:m.predict(delayed) for n,m in models.items()}
        for n,p in current.items():preds[n].extend(p.tolist())
        for n,p in lagged.items():lags[n].extend(p.tolist())
        folds.append(dict(start=start,end=end,training_rows=len(train),evaluation_rows=len(test),
            models={n:evaluate(test,p) for n,p in current.items()},
            lagged={n:evaluate(test,p) for n,p in lagged.items()}))
        print('Scored fold',start,len(train),len(test),flush=True)
    report=dict(protocol=PROTOCOL,rerun=repeated,holdout_loaded=False,deployment_allowed=False,
        data_sha256=digest((root/'development.jsonl').read_bytes()),
        code_sha256=digest(Path(__file__).read_bytes()),folds=folds,evaluation_rows=len(allrows),
        dependency_hashes={p.name:digest(p.read_bytes()) for p in
            [Path(__file__).with_name(n) for n in ('btc_flow_model.py','btc_new_history.py','btc_research.py','btc_inputs.py','sizing.py')]},
        aggregate={n:evaluate(allrows,np.array(p)) for n,p in preds.items()},
        lagged={n:evaluate(allrows,np.array(p)) for n,p in lags.items()})
    save(root/'report.json',report)
    for n,v in report['aggregate'].items():print(n,v['scores']['brier'],v['paper']['pnl'],v['paper']['entries'])
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='reports/btc-flow-followup-v1')
    parser.add_argument('--cache');parser.add_argument('--offline',action='store_true');args=parser.parse_args()
    if args.cache:collect(args.cache,args.output,args.offline)
    run(args.output)
