"""Loopback-only UI, with same-origin checks on mutations."""
import json
import os
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from .core import Settings
from .kalshi import collect_once
from .live import LiveManager
from .replay import DEMO_DATASET, demo_records, parse_jsonl, save_replay
from .research_store import ResearchStore
from .discovery import CATEGORIES, refresh_board
from .history import download_history
from .jobs import Jobs


class LocalServer(ThreadingHTTPServer):
    allow_reuse_address=False

    def server_bind(self):
        if hasattr(socket,'SO_EXCLUSIVEADDRUSE'):
            self.socket.setsockopt(socket.SOL_SOCKET,socket.SO_EXCLUSIVEADDRUSE,1)
        super().server_bind()


def serve(store, port=8765, attach=False):
    # Claim the port before recovering jobs: a duplicate launch must not alter
    # a running instance's training or paper-session state.
    public_origin=os.environ.get('BETCHECK_PUBLIC_ORIGIN','').rstrip('/')
    parsed_origin=urlparse(public_origin)
    if public_origin and (parsed_origin.scheme!='https' or not parsed_origin.netloc or parsed_origin.path or parsed_origin.query or parsed_origin.fragment or parsed_origin.username):
        raise ValueError('BETCHECK_PUBLIC_ORIGIN must be an HTTPS origin')
    server=LocalServer(('0.0.0.0' if public_origin else '127.0.0.1',port),BaseHTTPRequestHandler)
    if not attach:
        store.interrupt_previous_runs()
    manager = LiveManager(store)
    research=ResearchStore(store)
    jobs=Jobs(store,recover=not attach)
    analysis_jobs=Jobs(store,recover=not attach,state_key='analysis_job')
    model_service=None
    from . import named_models
    if research.get('named_selection') is None and named_models.state(store)['available']:
        research.set('named_selection',named_models.configuration(None))
    static = Path(__file__).parent / "static"

    class Handler(BaseHTTPRequestHandler):
        def send(self, status, body, mime="application/json"):
            if not isinstance(body, bytes):
                body = json.dumps(body, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            ancestors='https://huggingface.co' if public_origin else "'none'"
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; frame-ancestors "+ancestors)
            self.end_headers()
            self.wfile.write(body)

        def host_ok(self):
            return self.headers.get("Host") in (f"127.0.0.1:{port}", f"localhost:{port}", parsed_origin.netloc if public_origin else 'invalid-host')

        def do_GET(self):
            if not self.host_ok():
                return self.send(403, {"error": "Local host required"})
            path = urlparse(self.path).path
            try:
                if path == '/healthz':
                    return self.send(200,dict(status='ok',revision=os.environ.get('BETCHECK_REVISION','local'),paper_only=True,hosted=bool(public_origin)))
                files = {"/": "home.html", "/app.js": "main.js", "/training.js":"training.js", "/named.js":"named.js", "/style.css": "main.css"}
                if path in files:
                    mime = {"/": "text/html; charset=utf-8", "/app.js": "text/javascript", "/training.js":"text/javascript", "/named.js":"text/javascript", "/style.css": "text/css"}[path]
                    content=(static / files[path]).read_bytes()
                    if path=='/' and public_origin:
                        content=content.replace(b'<body>',b'<body><p class="fine" role="status">Shared public paper-trading demo: model selections, training and practice sessions are shared with other visitors. Data may reset when the Space restarts. No real orders.</p>')
                        content=content.replace(b'Keep your computer awake while it runs.',b'The Space runs the session while its server stays awake; restarting it interrupts practice.')
                    return self.send(200, content, mime)
                if path == "/api/readiness":
                    from .readiness import assess
                    return self.send(200, assess(store))
                if path == "/api/state":
                    from .model_library import list_models
                    from .coverage import model_coverage
                    experiment=research.experiment((research.get('active_model') or {}).get('experiment_id'))
                    if experiment:
                        experiment={k:v for k,v in experiment.items() if k not in ('forecasts','validation_predictions')}
                        experiment['scenarios']=[{k:v for k,v in s.items() if k not in ('ledger','curve')} for s in experiment['scenarios']]
                    station=research.get('station_model')
                    if station:
                        station={k:v for k,v in station.items() if k not in ('sources','diagnostics','candidates')}
                        station['scenarios']=[{k:v for k,v in scenario.items() if k not in ('ledger','curve')} for scenario in station['scenarios']]
                    return self.send(200, dict(named_models=named_models.state(store),priority_research=research.get('priority_research'),station_model=station,model_coverage=model_coverage(research.get('active_model')),models=list_models(store),runs=store.runs(), datasets=store.datasets(), active_id=manager.active_id,
                        board=research.get('board'),categories=CATEGORIES,job=research.get('job'),analysis_job=research.get('analysis_job'),
                        history=research.get('history_manifest'),experiment=experiment,active_model=research.get('active_model'),analysis=research.get('last_analysis'),training=research.get('training'),weather_model=research.get('weather_model'),weather_training=research.get('weather_training'),audit=research.get('audit'),app_version='0.9.0'))
                if path == '/api/board':
                    nonlocal model_service
                    from .evaluation import ModelService
                    if model_service is None:
                        model_service=ModelService(store)
                    board=research.get('board',dict(markets=[],at=None,categories=[]))
                    for market in board['markets']:
                        from .explain import explain_market
                        market['recommendation']=explain_market(store,market,model_service,diagnostics=False)
                        if research.get('named_selection'):
                            probability=market['recommendation']['probability']
                            probability=market['midpoint'] if probability is None else probability
                            note='; '.join(market['recommendation']['reasons'])
                        else:
                            probability,note=model_service.predict(market['series'],market['bid'],market['ask'],
                                market['at'],market['close_time'],experimental=True)
                        market['model_probability']=probability
                        market['model_note']=note
                        market['estimated_edge']=max(probability-market['ask'],1-probability-market['no_ask'])
                    return self.send(200,board)
                if path == '/api/experiment':
                    return self.send(200,research.experiment((research.get('active_model') or {}).get('experiment_id')))
                if path.startswith("/api/report/"):
                    return self.send(200, store.run(path.rsplit("/", 1)[-1]))
                return self.send(404, {"error": "Not found"})
            except (ValueError, OSError) as exc:
                return self.send(400, {"error": str(exc)})

        def do_POST(self):
            if not self.host_ok() or self.headers.get("Origin") not in (
                f"http://127.0.0.1:{port}", f"http://localhost:{port}", public_origin or 'invalid-origin'
            ):
                return self.send(403, {"error": "Same-origin local requests only"})
            try:
                length = int(self.headers.get("Content-Length", 0))
                if not 0 < length <= 10000000:
                    raise ValueError("Request must be between 1 byte and 10 MB")
                body = json.loads(self.rfile.read(length))
                if not isinstance(body,dict):
                    raise ValueError('Request body must be a JSON object')
                path = urlparse(self.path).path
                if public_origin and body.get('predictions_file'):
                    raise ValueError('Hosted sessions cannot read server file paths; import predictions through the upload form')
                if path in ('/api/named/select','/api/named/evaluate','/api/named/train'):
                    config=named_models.configuration(body.get('configuration'))
                    if path=='/api/named/select':
                        if manager.thread and manager.thread.is_alive():raise ValueError('Stop practice before changing its model')
                        named_models.load(store,config)
                        research.set('named_selection',config)
                        return self.send(200,config)
                    if path=='/api/named/evaluate':
                        bankroll=float(body.get('bankroll',1000))
                        if not 1<=bankroll<=1000000:raise ValueError('Use a bankroll between 1 and 1000000')
                        return self.send(200,dict(job_id=jobs.start('Test named model',lambda p:named_models.evaluate(store,config,p,bankroll=bankroll))))
                    epochs=int(body.get('epochs',20))
                    if not 1<=epochs<=100:raise ValueError('Use 1–100 extra epochs')
                    return self.send(200,dict(job_id=jobs.start('Continue named model training',lambda p:named_models.train(store,config,epochs,bool(body.get('freeze_backbone',True)),p))))
                if path=='/api/network-check':
                    from .kalshi import Kalshi
                    return self.send(200,dict(status='connected',exchange=Kalshi(store).get('/exchange/status')))
                if path == '/api/priority-research':
                    if manager.thread and manager.thread.is_alive():raise ValueError('Finish or stop practice before freezing its evidence for research')
                    action=body.get('action')
                    if action not in ('outcomes','weather','strategy','polymarket'):raise ValueError('Choose outcomes, weather, strategy or polymarket')
                    def run_priority(progress):
                        import uuid
                        from .weather_research import collect_outcomes,train
                        folder=store.root/'experiments'/('priority-'+uuid.uuid4().hex[:12])
                        if action=='outcomes':result=collect_outcomes(store)
                        elif action=='weather':result=train(store.root,folder)
                        elif action=='polymarket':
                            from .poly_ablation import compare
                            result=compare(store.root,folder)
                        else:
                            from .strategy_research import run
                            model_id=(research.get('active_model') or {}).get('model_id','')
                            source=(store.root/'training'/model_id).resolve()
                            if source.parent!=(store.root/'training').resolve() or not (source/'best.pkl').exists():raise ValueError('Select an epoch-trained price model first')
                            result=run(source,folder)
                        research.set('priority_research',dict(action=action,result=result,folder=str(folder) if action!='outcomes' else None))
                        return result
                    return self.send(200,{'job_id':jobs.start('Evaluate '+action,run_priority)})
                if path == '/api/activate-model':
                    if manager.thread and manager.thread.is_alive():raise ValueError('Stop practice before switching its model')
                    if (research.get('job') or {}).get('status')=='running':raise ValueError('Wait for the current training or data job to finish')
                    from .model_library import activate_model
                    return self.send(200,activate_model(store,body['experiment_id']))
                if path == '/api/train':
                    from .training import train_model,settings
                    config=settings(body.get('config'))
                    return self.send(200,{'job_id':jobs.start('Train neural model',lambda p:train_model(store,config,p,jobs.cancel_event))})
                if path == '/api/station-train':
                    from .station_model import run_station_experiment
                    return self.send(200,{'job_id':jobs.start('Train station-informed model',lambda p:run_station_experiment(store,p))})
                if path == '/api/weather-train':
                    from .weather_model import train_weather
                    return self.send(200,{'job_id':jobs.start('Train weather challenger',lambda p:train_weather(store,p))})
                if path == '/api/cancel-training':
                    current=research.get('job') or {}
                    if current.get('name')!='Train neural model' or current.get('status')!='running':
                        raise ValueError('No neural training job is running')
                    jobs.cancel_event.set()
                    return self.send(200,{'status':'Training will stop after the current epoch; the previous active model remains available.'})
                if path == '/api/market-detail':
                    from .resolve import resolve_markets
                    from .explain import explain_market
                    result=resolve_markets(store,body['reference'])
                    if len(result['markets'])!=1:raise ValueError('Choose an individual contract from the market list')
                    card=result['markets'][0]
                    card['recommendation']=explain_market(store,card)
                    card['model_probability']=card['recommendation']['probability']
                    card['model_note']=card['recommendation'].get('model_note') or '; '.join(card['recommendation']['reasons'])
                    return self.send(200,card)
                if path == '/api/analyze':
                    from .analysis import analyze
                    from .resolve import parse_reference
                    reference=body['reference']
                    parse_reference(reference)
                    return self.send(200,{'job_id':analysis_jobs.start('Check market link',lambda p:analyze(store,reference,p))})
                if path == '/api/discover':
                    categories=body.get('categories') or ['weather']
                    return self.send(200,{'job_id':analysis_jobs.start('Update market prices',lambda p:refresh_board(store,categories,p))})
                if path == '/api/history':
                    days=int(body.get('days',100))
                    return self.send(200,{'job_id':jobs.start('Download real history',lambda p:download_history(store,days=days,progress=p))})
                if path == '/api/tune':
                    from .evaluation import run_evaluation
                    return self.send(200,{'job_id':jobs.start('Auto-tune and evaluate',lambda p:run_evaluation(store,progress=p))})
                if path == "/api/demo":
                    dataset = DEMO_DATASET
                    store.append(demo_records(), dataset)
                    run_id, _ = save_replay(store, dataset, Settings(**body.get("settings", {})))
                    return self.send(200, {"run_id": run_id})
                if path == "/api/import":
                    dataset = body["dataset"].strip()
                    if not dataset or dataset.startswith(("demo-", "live-")):
                        raise ValueError("Choose a dataset name without a reserved demo-/live- prefix")
                    count = store.append(parse_jsonl(body["text"]), dataset)
                    return self.send(200, {"records": count})
                if path == "/api/replay":
                    run_id, _ = save_replay(store, body["dataset"], Settings(**body.get("settings", {})),
                                            body.get("start") or None, body.get("end") or None)
                    return self.send(200, {"run_id": run_id})
                if path == "/api/collect":
                    return self.send(200, collect_once(store, body["series"].strip(), int(body.get("limit", 12))))
                if path == "/api/live":
                    run_id = manager.start(body.get("tickers"), Settings(**body.get("settings", {})),
                        float(body.get("hours", 24)), int(body.get("poll_seconds", 60)),
                        body.get("fee_rate") or None, body.get("predictions_file") or None,
                        categories=body.get('categories'),auto_model=body.get('auto_model',True),
                        experimental=body.get('experimental',False),reference=body.get('reference'),
                        prospective=body.get('prospective',False),named_configuration=body.get('named_configuration'))
                    return self.send(200, {"run_id": run_id})
                if path == "/api/stop":
                    manager.stop()
                    return self.send(200, {"status": "Stop requested; an in-flight request may take up to its timeout"})
                return self.send(404, {"error": "Not found"})
            except (ValueError, KeyError, TypeError, OSError, RuntimeError) as exc:
                return self.send(400, {"error": str(exc)})

        def log_message(self, fmt, *args):
            print(fmt % args)

    server.RequestHandlerClass=Handler
    print(f"Kalshi Helper: http://127.0.0.1:{port} — paper simulation only", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        manager.stop()
        if manager.thread:
            manager.thread.join(timeout=60)
        server.server_close()
