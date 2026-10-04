"""Local prospective BTC paper experiment. Public data only; no order endpoints.

Only load experiment directories created locally by init(): model.joblib is a
Python artifact, not a safe format for files received from other people.
"""
import argparse
from contextlib import closing
from datetime import datetime, timedelta, timezone
import json
from importlib.metadata import version
from pathlib import Path
import sqlite3
import time
from urllib.parse import urlencode, quote
from urllib.request import Request, urlopen

import joblib
from .btc_calibration import read_development
from .btc_research import BASE, BTCModels, digest, iso, save, spot_features, metrics, pnl_interval
from .btc_regime import BTCRegime, TradeGate, gate_training
from .core import dec, fee_and_cost, stamp


def now(): return datetime.now(timezone.utc).isoformat()


def connect(root):
    db=sqlite3.connect(Path(root)/'records.sqlite3')
    db.execute('CREATE TABLE IF NOT EXISTS records (id INTEGER PRIMARY KEY, kind TEXT NOT NULL, ticker TEXT, received TEXT NOT NULL, body TEXT NOT NULL, sha256 TEXT NOT NULL)')
    db.execute('CREATE UNIQUE INDEX IF NOT EXISTS prediction_once ON records(ticker) WHERE kind="prediction"')
    return db


def append(db,kind,ticker,body,received=None):
    raw=json.dumps(body,sort_keys=True,allow_nan=False)
    db.execute('INSERT INTO records(kind,ticker,received,body,sha256) VALUES(?,?,?,?,?)',
               (kind,ticker,received or now(),raw,digest(raw.encode())))
    db.commit()


def records(db):
    for seq,kind,ticker,received,raw,sha in db.execute('SELECT * FROM records ORDER BY id'):
        if digest(raw.encode())!=sha:raise ValueError('Recorded payload hash mismatch')
        yield dict(id=seq,kind=kind,ticker=ticker,received=received,body=json.loads(raw))


def initialize(root,source):
    root=Path(root)
    if root.exists() and any(root.iterdir()):raise ValueError('Initialize an empty experiment directory')
    root.mkdir(parents=True,exist_ok=True)
    train,_=read_development(source)
    guard=BTCModels().fit(train);regime=BTCRegime().fit(train)
    rows,p,audit=gate_training(train);gate=TradeGate().fit(rows,p)
    joblib.dump(dict(guard=guard,regime=regime,gate=gate),root/'model.joblib')
    start=datetime.now(timezone.utc)
    code_files=['btc_forward.py','btc_research.py','btc_regime.py','btc_calibration.py','core.py']
    protocol=dict(version='btc-forward-v1',start=start.isoformat(),end=(start+timedelta(days=30)).isoformat(),
        bankroll=1000,bet_cap=10,day_open_cap=30,portfolio_cap=200,max_contracts=100,
        fee_rate=.07,slippage=.02,min_edge=.04,min_latency_seconds=2,max_latency_seconds=30,
        snapshot_max_age_seconds=20,decision_seconds_left=[285,315],
        models=['market','BTCMarketGuard','BTCRegime_gate','abstain'],
        model_sha256=digest((root/'model.joblib').read_bytes()),
        package_versions={n:version(n) for n in ['numpy','scipy','scikit-learn','joblib']},
        code_hashes={n:digest((Path(__file__).parent/n).read_bytes()) for n in code_files},
        training_sha256=digest((Path(source)/'train.jsonl').read_bytes()),
        gate_audit=audit,model_selection='Frozen before first prospective record; no parameter tuning',
        limitations=['REST receipt time is not exchange generation time','Assumed fills from later displayed depth, not actual orders',
                     'Fixed 0.07 fee assumption; verify against exchange before interpreting P&L',
                     'Coinbase proxy, not BRTI','30 days is a checkpoint, not proof of statistical power'])
    save(root/'protocol.json',protocol)
    with closing(connect(root)) as db:append(db,'initialized',None,{'protocol_sha256':digest((root/'protocol.json').read_bytes())})
    return protocol


def verify(root):
    root=Path(root);p=json.loads((root/'protocol.json').read_text())
    with closing(connect(root)) as db:
        first=next(records(db))
    if first['kind']!='initialized' or first['body']['protocol_sha256']!=digest((root/'protocol.json').read_bytes()):
        raise ValueError('Frozen protocol changed')
    if digest((root/'model.joblib').read_bytes())!=p['model_sha256']:raise ValueError('Frozen model changed')
    for name,sha in p['code_hashes'].items():
        if digest((Path(__file__).parent/name).read_bytes())!=sha:raise ValueError('Frozen experiment code changed: '+name)
    for name,expected in p.get('package_versions',{}).items():
        if version(name)!=expected:raise ValueError('Frozen dependency version changed: '+name)
    return p


def fetch(db,url,ticker=None):
    started=now()
    try:
        with urlopen(Request(url,headers={'User-Agent':'BetCheck prospective paper research'}),timeout=20) as response:
            raw=response.read(8*1024*1024+1)
        received=now()
        if len(raw)>8*1024*1024:raise ValueError('Response size limit')
        data=json.loads(raw)
        append(db,'response',ticker,dict(url=url,started=started,raw=raw.decode('utf8'),source_sha256=digest(raw)),received)
        return data,received
    except Exception as exc:
        append(db,'fetch_error',ticker,dict(url=url,started=started,error_type=type(exc).__name__))
        raise


def levels(body,side):
    book=body['orderbook_fp'];key='no_dollars' if side=='yes' else 'yes_dollars'
    out=[];seen=set()
    for price,quantity in book[key]:
        p,q=dec(price),dec(quantity)
        if not p.is_finite() or not q.is_finite() or not 0<=p<=1 or q<0:raise ValueError('Invalid depth')
        if p in seen:raise ValueError('Duplicate depth price')
        seen.add(p)
        if q>=1:out.append((1-p,int(q)))
    return sorted(out)


def quote_from_book(body):
    yes,no=levels(body,'yes'),levels(body,'no')
    if not yes or not no:raise ValueError('No two-sided whole-contract depth')
    bid,ask=float(1-no[0][0]),float(yes[0][0])
    if not 0<bid<=ask<1:raise ValueError('Unsupported or crossed book')
    return bid,ask


def poll(root,settle_only=False):
    root=Path(root);protocol=verify(root)
    if not settle_only and not stamp(protocol['start'])<=stamp(now())<stamp(protocol['end']):raise ValueError('Outside collection window')
    models=None if settle_only else joblib.load(root/'model.joblib')
    with closing(connect(root)) as db:
        # Keep settlement polling independent of current open-market discovery.
        predicted={r[0] for r in db.execute('SELECT ticker FROM records WHERE kind="prediction"')}
        settled={r[0] for r in db.execute('SELECT ticker FROM records WHERE kind="outcome"')}
        for ticker in sorted(predicted-settled)[:20]:
            data,received=fetch(db,BASE+'/markets/'+quote(ticker,safe=''),ticker);m=data['market']
            if m.get('status') in ('settled','finalized') and m.get('result') in ('yes','no'):
                append(db,'outcome',ticker,dict(y=int(m['result']=='yes'),settlement_ts=m.get('settlement_ts')),received)
        if settle_only:return
        page,received=fetch(db,BASE+'/markets?'+urlencode(dict(series_ticker='KXBTC15M',status='open',limit=1000)))
        if page.get('cursor'):raise ValueError('Market page incomplete; refusing silent truncation')
        current=[m for m in page['markets'] if 0<(stamp(m['close_time'])-stamp(now())).total_seconds()<960]
        for m in current:
            ticker=m['ticker'];rules=m.get('rules_primary','').lower()
            if not (m.get('strike_type')=='greater_or_equal' and 'simple average of the sixty seconds' in rules
                    and "cf benchmarks' brti" in rules and 'is at least' in rules
                    and (stamp(m['close_time'])-stamp(m['open_time'])).total_seconds()==900):
                append(db,'skip',ticker,{'reason':'unsupported_contract'});continue
            book_started=now()
            book,book_at=fetch(db,BASE+'/markets/'+quote(ticker,safe='')+'/orderbook?depth=20',ticker)
            book['request_started_at']=book_started
            append(db,'book',ticker,book,book_at)
            left=(stamp(m['close_time'])-stamp(book_at)).total_seconds()
            if ticker in predicted or not protocol['decision_seconds_left'][0]<=left<=protocol['decision_seconds_left'][1]:continue
            try:
                bid,ask=quote_from_book(book)
                t=stamp(now());params=dict(start=(t-timedelta(hours=3)).isoformat(),end=t.isoformat(),granularity=60)
                bars,spot_at=fetch(db,'https://api.exchange.coinbase.com/products/BTC-USD/candles?'+urlencode(params),ticker)
                decision=now()
                if (stamp(decision)-stamp(book_at)).total_seconds()>protocol['snapshot_max_age_seconds']:raise ValueError('Stale book')
                left=(stamp(m['close_time'])-stamp(decision)).total_seconds()
                if not 285<=left<=315:raise ValueError('Missed decision window')
                strike=float(m['floor_strike'])
                feature=spot_features({int(b[0]):b for b in bars},stamp(decision).timestamp(),stamp(m['close_time']).timestamp(),strike)
                row=dict(ticker=ticker,at=decision,close_time=m['close_time'],strike=strike,
                    p=(bid+ask)/2,bid=bid,ask=ask,spread=ask-bid,group=m['close_time'][:10],**feature)
                guard=float(models['guard'].predict([row],'BTCMarketGuard')[0]);regime=float(models['regime'].predict([row])[0])
                _,accepted=models['gate'].filter([row],[regime])
                predictions=dict(market=row['p'],BTCMarketGuard=guard,BTCRegime_gate=regime)
                append(db,'prediction',ticker,dict(row=row,probabilities=predictions,gate_accepted=bool(accepted),
                    book_received_at=book_at,coinbase_received_at=spot_at,target_received_at=received),decision)
                predicted.add(ticker)
            except (ValueError,KeyError,TypeError) as exc:
                append(db,'skip',ticker,{'reason':str(exc)})
        append(db,'poll_complete',None,dict(open_contracts=len(current)))


def depth_fill(book,side,probability,budget,protocol):
    total=dec(0);fees=dec(0);quantity=0
    win=dec(str(probability if side=='yes' else 1-probability))
    for price,available in levels(book,side):
        price=min(dec(1),price+dec(str(protocol['slippage'])))
        count=0
        for q in range(1,min(available,protocol['max_contracts']-quantity)+1):
            fee,cost=fee_and_cost(price,q,dec(str(protocol['fee_rate'])))
            if total+cost>budget or win-cost/q<dec(str(protocol['min_edge'])):break
            count=q
        if count:
            fee,cost=fee_and_cost(price,count,dec(str(protocol['fee_rate'])))
            total+=cost;fees+=fee;quantity+=count
        if quantity>=protocol['max_contracts']:break
    return quantity,total,fees


def replay(root):
    protocol=verify(root)
    events=[];record_count=0;errors=0
    with closing(connect(root)) as db:
        for event in records(db):
            record_count+=1;errors+=int(event['kind']=='fetch_error')
            if event['kind'] in ('prediction','book','outcome'):events.append(event)
    # Local receipts determine when information became usable. Never backdate
    # settlements to source timestamps, nor fill at the prediction's old book.
    events.sort(key=lambda r:(stamp(r['received']),r['id']))
    result={}
    for strategy in ['BTCMarketGuard','BTCRegime_gate','abstain']:
        cash=dec(protocol['bankroll']);positions={};pending={};closed=set();ledger=[];skips={}
        for event in events:
            ticker=event['ticker'];body=event['body'];at=stamp(event['received'])
            if event['kind']=='outcome':
                closed.add(ticker);pending.pop(ticker,None)
                pos=positions.pop(ticker,None)
                if pos:
                    payout=dec(body['y'] if pos['side']=='yes' else 1-body['y'])*pos['quantity'];cash+=payout
                    ledger.append(dict(ticker=ticker,**pos,pnl=float(payout-dec(pos['cost'])),settled_received=event['received']))
            elif event['kind']=='prediction' and strategy!='abstain' and ticker not in closed:
                if strategy=='BTCRegime_gate' and not body['gate_accepted']:continue
                row=body['row'];p=body['probabilities'][strategy]
                side='yes' if p-row['ask'] >= (1-p)-(1-row['bid']) else 'no'
                price=min(dec(1),dec(str(row['ask'] if side=='yes' else 1-row['bid']))+dec(str(protocol['slippage'])))
                _,cost=fee_and_cost(price,1,dec(str(protocol['fee_rate'])))
                win=dec(str(p if side=='yes' else 1-p))
                if win-cost<dec(str(protocol['min_edge'])):
                    skips['no_decision_edge']=skips.get('no_decision_edge',0)+1;continue
                pending[ticker]=event
            elif event['kind']=='book' and ticker in pending:
                prediction=pending[ticker];row=prediction['body']['row']
                delay=(at-stamp(prediction['received'])).total_seconds()
                if delay<protocol['min_latency_seconds']:continue
                # A slow response requested before the prediction is not a later
                # execution opportunity, even if it arrives afterward.
                requested=stamp(body.get('request_started_at',event['received']))
                if (requested-stamp(prediction['received'])).total_seconds()<protocol['min_latency_seconds']:continue
                pending.pop(ticker)
                if delay>protocol['max_latency_seconds'] or at>=stamp(row['close_time']):
                    skips['late_or_closed']=skips.get('late_or_closed',0)+1;continue
                p=prediction['body']['probabilities'][strategy]
                # Commit to the side supported at decision time, then recheck cost
                # using the later depth. Do not switch sides with future prices.
                side='yes' if p-row['ask'] >= (1-p)-(1-row['bid']) else 'no'
                open_cost=sum((dec(v['cost']) for v in positions.values()),dec(0))
                day_cost=sum((dec(v['cost']) for v in positions.values() if v['day']==row['group']),dec(0))
                budget=min(cash,(cash+open_cost)*dec('.01'),dec(protocol['bet_cap']),dec(protocol['day_open_cap'])-day_cost,dec(protocol['portfolio_cap'])-open_cost)
                try:
                    quote_from_book(body)
                    q,cost,fee=depth_fill(body,side,p,budget,protocol)
                except (ValueError,KeyError,TypeError):
                    skips['invalid_execution_book']=skips.get('invalid_execution_book',0)+1;continue
                if q:
                    cash-=cost;positions[ticker]=dict(side=side,quantity=q,cost=str(cost),fee=str(fee),day=row['group'],filled_received=event['received'])
                else:skips['no_eligible_depth_or_edge']=skips.get('no_eligible_depth_or_edge',0)+1
        deployed=sum(float(t['cost']) for t in ledger);profit=sum(t['pnl'] for t in ledger)
        contracts=sum(t['quantity'] for t in ledger)
        result[strategy]=dict(realized_pnl=profit,entries=len(ledger),cash=float(cash),
            open_positions=len(positions),pending_predictions=len(pending),capital_deployed_settled=deployed,
            settled_contracts=contracts,net_per_contract=profit/contracts if contracts else None,
            return_on_deployed=profit/deployed if deployed else None,ledger=ledger,skips=skips)
    outcomes={e['ticker']:e for e in events if e['kind']=='outcome'}
    scored=[e for e in events if e['kind']=='prediction' and e['ticker'] in outcomes
            and stamp(e['received'])<stamp(outcomes[e['ticker']]['received'])]
    score_rows=[dict(e['body']['row'],y=outcomes[e['ticker']]['body']['y']) for e in scored]
    forecast_scores={name:metrics(score_rows,[e['body']['probabilities'][name] for e in scored])
                     for name in ['market','BTCMarketGuard','BTCRegime_gate']} if scored else {}
    for strategy in result.values():
        numeric=dict(strategy,ledger=[dict(t,cost=float(t['cost'])) for t in strategy['ledger']])
        strategy['uncertainty']=pnl_interval(numeric,{r['group'] for r in score_rows}) if score_rows else None
    report=dict(protocol=protocol,strategies=result,record_count=record_count,forecast_scores=forecast_scores,
        predictions=sum(e['kind']=='prediction' for e in events),errors=errors,
        note='Paper replay only; displayed depth does not guarantee an executable fill. Unsettled cost is not treated as a realized loss.')
    save(Path(root)/'replay.json',report);return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['init','record','settle','replay'])
    parser.add_argument('--directory',default='data/btc-forward');parser.add_argument('--source',default='reports/btc15m-pilot-v1')
    parser.add_argument('--polls',type=int,default=1);parser.add_argument('--interval',type=float,default=15)
    args=parser.parse_args()
    if args.action=='init':print(json.dumps(initialize(args.directory,args.source),indent=2))
    elif args.action=='replay':print(json.dumps(replay(args.directory),indent=2))
    else:
        if args.polls<1 or args.interval<5:parser.error('Use positive polls and interval >=5 seconds')
        for i in range(args.polls):
            # A source outage is recorded and retried next cycle; a modified
            # frozen experiment still fails before entering the polling attempt.
            protocol=verify(args.directory)
            if args.action=='record' and stamp(now())>=stamp(protocol['end']):
                print('Prediction window ended; use settle to collect remaining outcomes.',flush=True);break
            try:poll(args.directory,settle_only=args.action=='settle')
            except Exception as exc:
                with closing(connect(args.directory)) as db:append(db,'cycle_error',None,{'error_type':type(exc).__name__})
                print('Cycle failed: '+type(exc).__name__,flush=True)
            if i+1<args.polls:time.sleep(args.interval)
