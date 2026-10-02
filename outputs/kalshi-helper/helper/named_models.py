"""Named research families: selection, reproducible practice and warm-start training."""
import copy,hashlib,json,math,pickle,re,uuid,threading
from pathlib import Path
from datetime import timedelta
from .core import stamp,utcnow
from .research_store import ResearchStore

PROJECT=Path(__file__).resolve().parents[1]
FOLDER=PROJECT/'reports/named-model-families'
NAMES=('WeatherSignal','MarketGuard','ConsensusBlend')
_recommend_lock=threading.Lock()
_services={}

def recommend(store,card):
    with _recommend_lock:return _recommend(store,card)

def _recommend(store,card):
    from .core import dec,fee_and_cost
    from .kalshi import Kalshi
    config=ResearchStore(store).get('named_selection')
    key=(str(store.root),json.dumps(config,sort_keys=True))
    if key not in _services:
        _services.clear();_services[key]=LiveNamedModel(store,config)
    service=_services[key]
    result=dict(action='WATCH',label='Collecting only',reasons=[],probability=None,market_probability=card.get('midpoint'),
        kalshi_url='https://kalshi.com/markets/'+card['series'].lower()+'/'+card['event_ticker'].lower(),
        fees=None,edges=None,sensitivity=[],validation=None,test=None,inputs=dict(series=card['series'],quote_at=card['at']))
    if card.get('bid') is None or card.get('ask') is None:
        result['reasons']=['Waiting for a two-sided quote'];return result
    age=(stamp(utcnow())-stamp(card['at'])).total_seconds()
    if age<0 or age>180:
        result.update(label='Refresh prices before using this estimate',reasons=['Quote is stale; no weather model was run']);return result
    # Avoid downloading metadata/weather for unsupported decision times.
    hours=(stamp(card['close_time'])-stamp(card['at'])).total_seconds()/3600
    service.market=Kalshi(store).market(card['ticker']) if 12<=hours<=14 and card['series'] in ('KXHIGHNY','KXHIGHCHI','KXHIGHMIA','KXHIGHDEN') else card
    prediction=service.predict_signal(card['series'],card['bid'],card['ask'],card['at'],card['close_time'],True)
    result['reasons']=[prediction.reason];result['model_note']=prediction.reason
    if not prediction.signal:return result
    p=prediction.probability;result.update(probability=p,model_id=service.loaded_id)
    if card.get('fee_rate') is not None:
        yf,yc=fee_and_cost(dec(card['ask']),1,dec(card['fee_rate']));nf,nc=fee_and_cost(1-dec(card['bid']),1,dec(card['fee_rate']))
        edges=dict(yes=p-float(yc),no=1-p-float(nc));side=max(edges,key=edges.get)
        result.update(edges=edges,fees=dict(yes=float(yf),no=float(nf)),lean=side.upper(),minimum_edge=.04,
            label='Experimental '+side.upper()+' lean' if edges[side]>=.04 else 'Pass — edge does not cover costs',action='WATCH' if edges[side]>=.04 else 'PASS')
    if (stamp(utcnow())-stamp(card['at'])).total_seconds()>180:result.update(action='WATCH',label='Refresh prices before using this estimate')
    return result

def configuration(value):
    value=value or {};name=value.get('name','ConsensusBlend')
    if name not in NAMES:raise ValueError('Choose WeatherSignal, MarketGuard, or ConsensusBlend')
    weight=float(value.get('weather_weight',.5))
    if not math.isfinite(weight) or not 0<=weight<=1:raise ValueError('WeatherSignal weight must be between 0 and 100%')
    checkpoint=value.get('checkpoint') or None
    if checkpoint and not re.fullmatch('[a-f0-9]{12}',checkpoint):raise ValueError('Invalid checkpoint')
    return dict(name=name,weather_weight=1. if name=='WeatherSignal' else 0. if name=='MarketGuard' else weight,checkpoint=checkpoint)

def state(store):
    rs=ResearchStore(store)
    return dict(selection=configuration(rs.get('named_selection')),available=all((FOLDER/(n+'.pkl')).exists() for n in NAMES),
        checkpoints=rs.get('named_checkpoints') or [],evaluation=rs.get('named_evaluation'),
        coverage='NYC, Chicago, Miami and Denver daily highs. Experimental live inputs near 13 hours before close; other times collect prices only.',
        descriptions={'WeatherSignal':'Stronger weather-based corrections to market odds','MarketGuard':'Smaller corrections with a penalty for departing from market odds','ConsensusBlend':'Your chosen mix of WeatherSignal and MarketGuard'})

def load(store,config):
    from .fusion_blend import FusionBlend
    config=configuration(config)
    registry=json.loads((FOLDER/'registry.json').read_text())
    def read(name):
        item=registry[name];path=FOLDER/item['file'];payload=path.read_bytes()
        if hashlib.sha256(payload).hexdigest()!=item['sha256']:raise ValueError('Named model integrity check failed')
        return pickle.loads(payload)
    if config['checkpoint']:
        folder=store.root/'named-checkpoints'/config['checkpoint'];meta=json.loads((folder/'metadata.json').read_text())
        if meta['name']!=config['name']:raise ValueError('Checkpoint belongs to a different model family')
        payload=(folder/'model.pkl').read_bytes()
        if hashlib.sha256(payload).hexdigest()!=meta['sha256']:raise ValueError('Checkpoint integrity check failed')
        model=pickle.loads(payload)
        if config['name']=='ConsensusBlend':model.weather_weight=config['weather_weight']
        return model
    if config['name']!='ConsensusBlend':return read(config['name'])
    return FusionBlend(read('WeatherSignal').members,read('MarketGuard').members,config['weather_weight'])

def dataset(store,split):
    import numpy as np
    from .forecast_archive import fetch_run,observation_innovation
    from .joint_research import sequence
    from .weather_model import event_date
    if split not in ('train','validation'):raise ValueError('Only development partitions are available here')
    rows=[];paths=[];cache={}
    for r in map(json.loads,(PROJECT/'reports/historical-forecast-replay'/(split+'.jsonl')).read_text().splitlines()):
        if split=='validation' and not .05<=r['p']<=.95:continue
        r['lower']=-math.inf if r['lower'] is None else r['lower'];r['upper']=math.inf if r['upper'] is None else r['upper']
        day=event_date(r['ticker']).date().isoformat()
        if day not in cache:cache[day]=fetch_run(store.root,day,offline=True)
        r.update(observation_innovation(cache[day],r['series'],r))
        if r.get('observation_innovation_f') is None:continue
        rows.append(r);paths.append(sequence(cache[day],r))
    return rows,np.asarray(paths)

def evaluate(store,config,progress=lambda *a:None,model=None,bankroll=1000):
    from .spread_research import calibration_metrics
    from .sizing import simulate
    config=configuration(config);progress('Loading held-out development dates',1,3)
    rows,paths=dataset(store,'validation');model=model or load(store,config);p=model.predict(rows,paths)
    progress('Comparing prices, fees and paper returns',2,3)
    immediate=[dict(r,execution_at=r['at'],execution_bid=r['bid'],execution_ask=r['ask']) for r in rows]
    results={}
    for name,data,slip in [('Immediate quote (optimistic)',immediate,0),('Next hourly quote',rows,0),('Next hourly quote + 2c',rows,.02)]:
        result=simulate(data,p,policy='bankroll_1_percent',slippage=slip,bankroll=float(bankroll))
        results[name]={k:v for k,v in result.items() if k!='ledger'}
    report=dict(at=utcnow(),configuration=config,metrics=calibration_metrics(rows,p),scenarios=results,
        events=len({r['event'] for r in rows}),dates=len({r['group'] for r in rows}),
        note='Previously inspected validation. Assumed fills without depth; delayed forecasts are stale. No proven edge. Reserved test untouched.')
    ResearchStore(store).set('named_evaluation',report);return report

def train(store,config,epochs=20,freeze=True,progress=lambda *a:None):
    from .joint_research import torch
    if isinstance(epochs,bool) or int(epochs)!=epochs or not 1<=epochs<=100:raise ValueError('Choose 1–100 extra epochs')
    config=configuration(config);model=copy.deepcopy(load(store,config));torch.set_num_threads(2)
    rows,paths=dataset(store,'train')
    members=model.original+model.revised if config['name']=='ConsensusBlend' else model.members
    for i,member in enumerate(members):
        member.average_last=0
        member.fit(rows,paths,epochs=int(epochs),warm_start=True,learning_rate=.001,freeze_backbone=freeze,
            progress=lambda e,total:progress(f'Continuing seed {i+1}/{len(members)} · epoch {e}/{total}',i*total+e,len(members)*total))
    report=evaluate(store,config,progress,model)
    identity=uuid.uuid4().hex[:12];folder=store.root/'named-checkpoints'/identity;folder.mkdir(parents=True)
    payload=pickle.dumps(model);(folder/'model.pkl').write_bytes(payload)
    metadata=dict(id=identity,name=config['name'],at=utcnow(),parent=config,epochs=epochs,freeze_backbone=freeze,
        sha256=hashlib.sha256(payload).hexdigest(),metrics=report['metrics'],training_events=len({r['event'] for r in rows}),
        note='Continued from saved weights on existing training partition; validation reused; not automatically selected')
    (folder/'metadata.json').write_text(json.dumps(metadata,indent=2))
    rs=ResearchStore(store);rs.set('named_checkpoints',[metadata]+(rs.get('named_checkpoints') or []))
    report['configuration']=dict(config,checkpoint=identity);report['at']=utcnow();report['note']='New checkpoint, not selected automatically. '+report['note']
    rs.set('named_evaluation',report);return metadata

class LiveNamedModel:
    """Same issued-run/observation feature pipeline as research; explicit live gate."""
    def __init__(self,store,config):
        self.store=store;self.config=configuration(config);self.model=load(store,self.config)
        self.loaded_id=self.config['name']+':'+str(self.config['weather_weight'])+':'+str(self.config['checkpoint'] or 'base')
        self.market=None;self.cache={}
    def predict_signal(self,series,bid,ask,at,close,experimental=False):
        from types import SimpleNamespace
        def result(p,signal,reason):return SimpleNamespace(probability=p,signal=signal,reason=reason,model_id=self.loaded_id)
        midpoint=(bid+ask)/2
        if not experimental:return result(midpoint,False,'Named models are unproven; choose Experimental practice to use them')
        from .weather_model import supported,bounds,event_date,STATIONS
        if series not in STATIONS or not self.market or not supported(self.market,series):return result(midpoint,False,'Named models support four-city daily high temperature contracts only')
        hours=(stamp(close)-stamp(at)).total_seconds()/3600
        if not 12<=hours<=14:return result(midpoint,False,'Collecting only: this model was trained near 13 hours before close (experimental window 12–14h)')
        try:
            from .forecast_archive import fetch_run,archived_feature,observation_innovation
            from .station_observations import fetch_observations,ObservationIndex,SITES
            from .joint_research import sequence
            import numpy as np
            ticker=self.market['ticker'];day=event_date(ticker).date();key=(series,day.isoformat())
            if key not in self.cache or (stamp(at)-self.cache[key][0]).total_seconds()>900:
                run=fetch_run(self.store.root,day.isoformat())
                observations=fetch_observations(self.store,SITES[series],day.isoformat(),(day+timedelta(days=2)).isoformat())
                self.cache[key]=(stamp(at),run,ObservationIndex(observations['observations']))
            _,run,index=self.cache[key];low,high=bounds(self.market)
            row=dict(ticker=ticker,series=series,at=at,p=midpoint,spread=ask-bid,lower=low,upper=high,
                **archived_feature(run,series,ticker,at),**index.feature(series,ticker,at))
            row.update(observation_innovation(run,series,row))
            if row.get('observation_innovation_f') is None:raise ValueError('No sufficiently recent station observation')
            p=float(self.model.predict([row],np.asarray([sequence(run,row)]))[0])
            if not math.isfinite(p) or not 0<=p<=1:raise ValueError('Invalid named-model probability')
            return result(p,True,self.loaded_id+' · experimental live weather inputs; historical research did not establish an edge')
        except (OSError,ValueError,KeyError,RuntimeError) as exc:
            return result(midpoint,False,'Collecting only: weather inputs unavailable · '+str(exc))
