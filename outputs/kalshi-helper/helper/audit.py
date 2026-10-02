"""Repeatable live-source and stored-data audit: python -m helper.audit."""
import hashlib
import json
from pathlib import Path
import tempfile
from .core import utcnow,stamp
from .storage import Store
from .research_store import ResearchStore,digest
from .kalshi import Kalshi,normalize_book
from .resolve import resolve_markets
from .history import sample_rows,extract_samples
from .training import split_dataset,fit_epochs
from .evaluation import score,validate_samples
from .weather import rain_context


def run_audit(store,train=True):
    report=dict(at=utcnow(),checks=[],limitations=[])
    def check(name,fn):
        try:detail=fn();report['checks'].append(dict(name=name,status='pass',detail=detail))
        except Exception as exc:report['checks'].append(dict(name=name,status='fail',detail=f'{type(exc).__name__}: {exc}'))
    rs=ResearchStore(store)
    def integrity():
        with store.connect() as db:
            result=db.execute('PRAGMA integrity_check').fetchone()[0]
            assert result=='ok',result
            fetches=db.execute('SELECT path,sha256 FROM fetches ORDER BY id DESC LIMIT 30').fetchall()
        for f in fetches:assert hashlib.sha256((store.root/f['path']).read_bytes()).hexdigest()==f['sha256']
        return dict(sqlite=result,raw_response_hashes_verified=len(fetches))
    check('Database and raw data integrity',integrity)
    rows=sample_rows(store)
    def dataset():
        validate_samples(rows);train_rows,val,test=split_dataset(rows)
        return dict(samples=len(rows),markets=len({r['ticker'] for r in rows}),days=len({r['group'] for r in rows}),
            partition_samples=[len(train_rows),len(val),len(test)],sha256=digest(rows))
    check('Historical labels, quotes and time splits',dataset)
    client=Kalshi(store)
    def live():
        result=resolve_markets(store,'KXHIGHNY',client)
        assert result['tickers'],'No active NYC contracts returned'
        ticker=result['tickers'][0];book=client.book(ticker)
        assert book['ticker']==ticker
        return dict(contracts=len(result['markets']),ticker=ticker,yes_bid_levels=len(book['yes']),no_bid_levels=len(book['no']),at=book['available_at'])
    check('Live Kalshi markets and order books',live)
    def archived():
        history=rs.history();old=min(history,key=lambda x:x['market']['close_time'])
        ticker=old['market']['ticker'];close=int(stamp(old['market']['close_time']).timestamp())
        cutoff=stamp(client.get('/historical/cutoff')['market_settled_ts'])
        if stamp(old['market']['settlement_ts'])<cutoff:
            current=client.get('/historical/markets/'+ticker)['market']
            candles=client.get('/historical/markets/'+ticker+'/candlesticks',start_ts=close-27*3600,end_ts=close-3*3600,period_interval=60)['candlesticks']
        else:
            current=client.market(ticker)
            candles=client.get('/markets/candlesticks',market_tickers=ticker,start_ts=close-27*3600,end_ts=close-3*3600,period_interval=60)['markets'][0]['candlesticks']
        assert current['result']==old['market']['result'],'Settlement differs from archived label'
        extracted=extract_samples(old['series'],current,candles)
        assert extracted,'No valid historical samples returned'
        return dict(ticker=ticker,outcome=current['result'],candles=len(candles),samples=len(extracted))
    check('Archived Kalshi settlement and candles',archived)
    def weather():
        card=resolve_markets(store,'KXRAINWKND-26SEP19-NYC',client)['markets'][0]
        context=rain_context(store,card)
        assert context['status']=='weather_context',context['note']
        assert context['periods'];assert context['probability'] is None
        return dict(station=context['station'],periods=len(context['periods']),updated_at=context['updated_at'],source=context['source_url'])
    check('NWS station, forecast and timestamp checks',weather)
    def artifact():
        active=rs.get('active_model');assert active,'No active trained model'
        path=store.root/'models'/active['model_file'];assert hashlib.sha256(path.read_bytes()).hexdigest()==active['sha256']
        from .evaluation import ModelService
        from datetime import timedelta
        now=stamp(utcnow());p,note=ModelService(store).predict('KXHIGHNY',.4,.5,now.isoformat(),(now+timedelta(hours=12)).isoformat(),True)
        assert 0<=p<=1 and note.startswith('Experimental')
        return dict(model_id=active['model_id'],probability=p,passed_validation=active['passed'])
    check('Saved model integrity and inference',artifact)
    if train:
        def learning():
            train_rows,val,test=split_dataset(rows)
            model,history,best=fit_epochs(train_rows,val,{'epochs':150})
            assert best==min(history,key=lambda r:r['validation_loss'])['epoch']
            assert len(history)>1 and history[0]['optimizer_loss']!=history[-1]['optimizer_loss']
            return dict(epochs=len(history),best_epoch=best,train=score(train_rows,model.predict(train_rows)),
                validation=score(val,model.predict(val)),test=score(test,model.predict(test)),baseline=score(test,[r['p'] for r in test]))
        check('Fresh training, validation checkpoint and final test',learning)
    report['status']='pass' if all(c['status']=='pass' for c in report['checks']) else 'issues_found'
    report['limitations']=['Source checks are point-in-time samples, not proof every external record is correct.',
        'Current learned inputs are market-derived. Rain forecasts are context, not validated contract probabilities.',
        'No full 24-hour soak or real-money execution test. A passing software audit does not establish profitable predictions.']
    rs.set('audit',report)
    return report


if __name__=='__main__':
    store=Store('data');report=run_audit(store)
    Path('reports').mkdir(exist_ok=True)
    Path('reports/system-audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    raise SystemExit(0 if report['status']=='pass' else 1)
