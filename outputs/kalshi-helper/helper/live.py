"""Bounded local paper sessions. The machine must stay awake for collection."""
from datetime import timedelta
import hashlib
import json
from pathlib import Path
import threading
import uuid

from .core import Engine, Settings, stamp, utcnow
from .kalshi import Kalshi, normalize_market, normalize_outcome
from .replay import parse_jsonl


def qualify_signal(probability,bid,ask,fee_rate,minimum_edge):
    """Conservative one-contract decision-time qualification for built-in ML."""
    from .core import dec,fee_and_cost
    if fee_rate is None:return None
    choices=[]
    for side,win,price in (('yes',dec(probability),dec(ask)),('no',1-dec(probability),1-dec(bid))):
        _,cost=fee_and_cost(price,1,dec(fee_rate));choices.append((win-cost,side))
    edge,side=max(choices)
    return dict(signal_side=side,decision_net_edge=str(edge)) if edge>=dec(minimum_edge) else None


def practice_candidates(markets,now=None,active_model=None):
    """Prefer model-supported horizons and usable two-sided quotes over expiry."""
    from .discovery import WEATHER_SERIES
    now=stamp(now or utcnow())
    def rank(m):
        hours=(stamp(m['close_time'])-now).total_seconds()/3600
        liquid=0<float(m.get('bid',0))<=float(m.get('ask',1))<1
        from .coverage import model_coverage
        c=model_coverage(active_model)
        supported=m.get('series') in c['series'] and c['min_hours']<=hours<=c['max_hours']
        return (not supported,not liquid,m.get('spread',1),m['close_time'])
    return sorted([m for m in markets if stamp(m['close_time'])>now and m.get('tradeable',True)],key=rank)[:24]


class LiveManager:
    def __init__(self, store):
        self.store = store
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = None
        self.active_id = None

    def start(self, tickers=None, settings=None, hours=24, poll_seconds=60,
              fee_rate=None, predictions_file=None, categories=None, auto_model=False, experimental=False, reference=None, prospective=False,named_configuration=None):
        from .core import dec
        if named_configuration:
            from .named_models import configuration,load
            named_configuration=configuration(named_configuration);load(self.store,named_configuration)
            if predictions_file:raise ValueError('Choose a named model or a predictions file, not both')
        with self.lock:
            if self.thread and self.thread.is_alive():
                raise ValueError('A paper session is already running. Stop it before starting another.')
        if isinstance(tickers,str):
            tickers = tickers.split(',')
        tickers = list(dict.fromkeys(x.strip() for x in (tickers or []) if x.strip()))
        fee_map = {}
        if reference or tickers:
            from .resolve import resolve_markets
            resolved = []
            for value in ([reference] if reference else tickers):
                result = resolve_markets(self.store,value)
                resolved.extend(result['tickers'])
                fee_map.update({m['ticker']:m['fee_rate'] for m in result['markets']})
            tickers = list(dict.fromkeys(resolved))
            if not tickers:
                raise ValueError('These markets are closed. Choose an open market for a live paper session.')
        automatic=not tickers
        if automatic:
            from .discovery import refresh_board
            from .research_store import ResearchStore
            categories=categories or ['weather']
            board=ResearchStore(self.store).get('board')
            if not board or (stamp(utcnow())-stamp(board['at'])).total_seconds()>600 or not set(categories)<=set(board['categories']):
                board=refresh_board(self.store,categories)
            coverage_model=dict(coverage=dict(series=['KXHIGHNY','KXHIGHCHI','KXHIGHMIA','KXHIGHDEN'],min_hours=12,max_hours=14)) if named_configuration else ResearchStore(self.store).get('active_model')
            tickers=[m['ticker'] for m in practice_candidates([m for m in board['markets'] if m['category'] in categories],active_model=coverage_model)]
            fee_map.update({m['ticker']:m['fee_rate'] for m in board['markets']})
        if not 1 <= len(tickers) <= 48:
            raise ValueError("No open markets found, or more than 48 custom markets selected")
        if not 0 < float(hours) <= 168 or not 15 <= int(poll_seconds) <= 3600:
            raise ValueError("Use 0–168 hours and a 15–3600 second poll interval")
        if fee_rate is not None and not 0 <= dec(fee_rate) <= 1:
            raise ValueError("Fee rate must be between 0 and 1")
        if predictions_file and not Path(predictions_file).is_file():
            raise ValueError("Predictions file does not exist")
        with self.lock:
            if self.thread and self.thread.is_alive():
                raise ValueError("A paper session is already running. Stop it before starting another.")
            self.stop_event.clear()
            run_id = uuid.uuid4().hex[:12]
            self.active_id = run_id
            engine = Engine(settings or Settings())
            start = stamp(utcnow())
            end = start+timedelta(hours=float(hours))
            details = dict(dataset=f"live-{run_id}", synthetic=False, start=start.isoformat(),
                           end=end.isoformat(), tickers=tickers, poll_seconds=poll_seconds,
                           predictions_file=str(predictions_file) if predictions_file else None,
                           automatic=automatic,categories=categories or ['weather'],auto_model=auto_model,
                           experimental=experimental, fee_map=fee_map, reference=reference,
                           prospective=prospective,
                           named_configuration=named_configuration,
                           strategy=('Experimental ML paper strategy' if experimental else 'Auto model with validation gate') if auto_model else
                           ('Imported timestamped predictions' if predictions_file else 'Observe only; no predictions supplied'))
            from .research_store import ResearchStore
            details['model_snapshot']=ResearchStore(self.store).get('active_model') or {}
            if named_configuration:
                details['model_snapshot']=dict(named_configuration=named_configuration,coverage=dict(series=['KXHIGHNY','KXHIGHCHI','KXHIGHMIA','KXHIGHDEN'],min_hours=12,max_hours=14))
                details['strategy']=('Experimental '+named_configuration['name']+' · '+str(round(named_configuration['weather_weight']*100))+'% WeatherSignal') if auto_model and experimental else 'Observe only · named models have no proven edge'
            self.store.save_run(run_id, "live paper", "running", dict(engine.report(), **details))
            self.thread = threading.Thread(target=self._run, args=(run_id, engine, details, fee_rate),
                                           daemon=True, name="paper-collector")
            self.thread.start()
            return run_id

    def stop(self):
        self.stop_event.set()

    def _run(self, run_id, engine, details, fee_rate):
        client, seen, errors = Kalshi(self.store), set(), []
        end = stamp(details["end"])
        status = "complete"
        last_discovery=stamp(utcnow())
        model_service=None
        health={}
        recorder=None;weather_times={};weather_snapshots={};previous_weather={};poly_snapshots={}

        def ingest(record):
            if recorder:recorder.append(record['type'],record)
            self.store.append([record], details["dataset"])
            engine.feed(record, allow_orders=stamp(record["available_at"]) < end)

        def persist(state):
            report = dict(engine.report(), **details, errors=errors[-50:], heartbeat_at=utcnow(),market_health=health)
            self.store.save_run(run_id, "live paper", state, report)

        try:
            if details.get('prospective'):
                from .prospective import Recorder,weather_snapshot,STATION_IDS
                recorder=Recorder(self.store,run_id,dict(settings=engine.report()['settings'],model=details.get('model_snapshot'),
                    started_at=details['start'],ends_at=details['end'],mode='paper only',poll_seconds=details['poll_seconds']))
                snapshot=details.get('model_snapshot') or {}
                if snapshot.get('model_file'):
                    source=self.store.root/'models'/snapshot['model_file']
                    if source.resolve().parent!=(self.store.root/'models').resolve():raise ValueError('Invalid model snapshot path')
                    payload=source.read_bytes()
                    if hashlib.sha256(payload).hexdigest()!=snapshot['sha256']:raise ValueError('Frozen model checksum mismatch')
                    (recorder.folder/'model.pkl').write_bytes(payload)
                details['evidence_directory']=str(recorder.folder)
            if details.get('auto_model'):
                if details.get('named_configuration'):
                    from .named_models import LiveNamedModel
                    model_service=LiveNamedModel(self.store,details['named_configuration'])
                else:
                    from .evaluation import ModelService
                    model_service=ModelService(self.store,active_model=details.get('model_snapshot'))
            while not self.stop_event.is_set() and stamp(utcnow()) < end:
                if details.get('automatic') and (stamp(utcnow())-last_discovery).total_seconds()>600:
                    try:
                        from .discovery import refresh_board
                        board=refresh_board(self.store,details['categories'])
                        details.setdefault('fee_map',{}).update({m['ticker']:m['fee_rate'] for m in board['markets']})
                        found=[m['ticker'] for m in practice_candidates(board['markets'],active_model=details.get('model_snapshot'))]
                        details['tickers']=list(dict.fromkeys([*engine.positions,*engine.pending,*found]))[:48]
                        last_discovery=stamp(utcnow())
                    except (RuntimeError,ValueError,KeyError) as exc:
                        errors.append(dict(at=utcnow(),message=f'Market discovery: {exc}'))
                from .research_store import ResearchStore
                from .discovery import series_fee
                catalog=ResearchStore(self.store).get('series_catalog',{'series':[]})
                by_series={s['ticker']:s for s in catalog['series']}
                for ticker in details["tickers"]:
                    if self.stop_event.is_set() or stamp(utcnow()) >= end:
                        break
                    try:
                        market = client.market(ticker)
                        series=market['event_ticker'].split('-')[0]
                        if recorder and series in STATION_IDS and (series not in weather_times or (stamp(utcnow())-weather_times[series]).total_seconds()>900):
                            try:
                                snapshot=weather_snapshot(series)
                                recorder.append('weather',snapshot)
                                previous_weather[series]=weather_snapshots.get(series)
                                weather_snapshots[series]=snapshot
                            except (OSError,ValueError,KeyError) as exc:recorder.append('weather_error',dict(series=series,error=str(exc)))
                            weather_times[series]=stamp(utcnow())
                        poly_key=market['event_ticker']
                        if recorder and series in STATION_IDS and (poly_key not in poly_snapshots or (stamp(utcnow())-stamp(poly_snapshots[poly_key]['captured_at'])).total_seconds()>900):
                            from .polymarket import collect
                            try:
                                poly_snapshots[poly_key]=collect(ticker)
                                recorder.append('polymarket',poly_snapshots[poly_key])
                            except (OSError,ValueError,KeyError) as exc:
                                poly_snapshots[poly_key]=dict(captured_at=utcnow(),failed=True)
                                recorder.append('polymarket_error',dict(ticker=ticker,error=str(exc)))
                        current_fee=fee_rate if fee_rate is not None else details.get('fee_map',{}).get(ticker)
                        ingest(normalize_market(market, utcnow(), current_fee))
                        health[ticker]=dict(at=utcnow(),status=market['status'],note='Waiting for settlement')
                        outcome = normalize_outcome(market, utcnow())
                        if outcome:
                            ingest(outcome)
                        elif market["status"] in ("open", "active"):
                            if details.get('named_configuration') and model_service:model_service.market=market
                            book=client.book(ticker)
                            # Zero-size levels must not supply model prices or baseline odds.
                            book=dict(book,yes=[x for x in book['yes'] if float(x[1])>0],no=[x for x in book['no'] if float(x[1])>0])
                            if model_service and book['yes'] and book['no']:
                                fresh=model_service.predict_signal(series,max(float(x[0]) for x in book['yes']),
                                    1-max(float(x[0]) for x in book['no']),utcnow(),market['close_time'],details.get('experimental',False))
                                book['execution_check']=dict(probability=fresh.probability,signal=fresh.signal,
                                    model_id=fresh.model_id or model_service.loaded_id or 'market-baseline',made_at=utcnow())
                                book['available_at']=utcnow()
                            ingest(book)
                            health[ticker]=dict(at=utcnow(),status='collecting',note='Watching prices; no prediction strategy selected')
                            persist('running')
                            if not book['yes'] or not book['no']:
                                health[ticker]['note']='One side of the order book is empty; no reliable two-sided price'
                            if model_service and book['yes'] and book['no'] and stamp(utcnow())<end:
                                bid=max(float(x[0]) for x in book['yes'])
                                ask=1-max(float(x[0]) for x in book['no'])
                                at=utcnow()
                                if recorder and series in weather_snapshots:
                                    from .remaining_weather import features
                                    try:
                                        row=features(weather_snapshots[series],ticker,at,previous_weather.get(series))
                                        poly=poly_snapshots.get(poly_key)
                                        if poly and not poly.get('failed'):
                                            from .polymarket import regional_features
                                            try:row.update(regional_features(poly,ticker,at))
                                            except ValueError as exc:row.update(poly_available=False,poly_reason=str(exc))
                                        recorder.append('weather_features',row)
                                    except (ValueError,KeyError,TypeError) as exc:recorder.append('weather_feature_error',dict(ticker=ticker,error=str(exc)))
                                prediction=model_service.predict_signal(series,bid,ask,at,market['close_time'],details.get('experimental',False))
                                probability,reason=prediction.probability,prediction.reason
                                if recorder:recorder.append('model_check',dict(ticker=ticker,at=at,bid=bid,ask=ask,probability=probability,signal=prediction.signal,reason=reason,model_id=prediction.model_id))
                                health[ticker]=dict(at=at,status='forecast',note=reason,probability=probability,bid=bid,ask=ask)
                                if not prediction.signal:
                                    health[ticker]['status']='watching'
                                    continue  # A stale market midpoint is NOT a trading signal.
                                intent=qualify_signal(probability,bid,ask,current_fee,engine.settings.min_edge)
                                if intent is None:
                                    health[ticker].update(status='watching',note='Model checked: no decision-time edge after one-contract fees, or fees unknown')
                                ingest(dict(type='prediction',ticker=ticker,available_at=at,made_at=at,
                                    features_available_at=book['available_at'],
                                    expires_at=(stamp(at)+timedelta(minutes=5)).isoformat(),
                                    probability=probability,model_id=model_service.loaded_id or 'market-baseline',
                                    explanation=reason,require_execution_check=True,signal_eligible=intent is not None,**(intent or {})))
                    except (RuntimeError, ValueError, KeyError) as exc:
                        errors.append(dict(at=utcnow(), ticker=ticker, message=str(exc)))
                        errors=errors[-50:]
                        health[ticker]=dict(at=utcnow(),status='error',note=str(exc))
                        from .kalshi import KalshiError
                        if isinstance(exc,KalshiError) and exc.status==404:
                            raise ValueError(f'{ticker} is no longer available. Session stopped; paste a current market link.') from exc
                path = details["predictions_file"]
                if path and stamp(utcnow()) < end and not self.stop_event.is_set():
                    try:
                        rows = parse_jsonl(Path(path).read_text(encoding="utf-8-sig"))
                        for r in rows:
                            if r["type"] != "prediction":
                                raise ValueError("Live prediction file must contain predictions or weather_forecast rows only")
                            key = hashlib.sha256(json.dumps(r, sort_keys=True).encode()).hexdigest()
                            if key in seen or r["ticker"] not in details["tickers"]:
                                continue
                            if stamp(r["available_at"]) > stamp(utcnow()):
                                continue
                            # Local arrival time, never a backdated fill from an imported file.
                            r = dict(r, source_available_at=r["available_at"], available_at=utcnow())
                            ingest(r)
                            seen.add(key)
                    except (OSError, ValueError, KeyError) as exc:
                        errors.append(dict(at=utcnow(), message=f"Prediction import: {exc}"))
                engine.now = stamp(utcnow())
                engine.mark()
                persist("running")
                remaining = max(0, (end-stamp(utcnow())).total_seconds())
                self.stop_event.wait(min(details["poll_seconds"], remaining))
            if self.stop_event.is_set():
                status = "stopped"
        except Exception as exc:
            status = "failed"
            errors.append(dict(at=utcnow(), message=f"Worker stopped: {type(exc).__name__}: {exc}"))
        finally:
            engine.finish(utcnow())
            if recorder:
                recorder.append('session_end',dict(status=status,errors=errors[-50:]))
                details['evidence_head']=recorder.previous
            persist(status)
