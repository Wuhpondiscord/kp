"""Functional checks of the hosted app, not just a successful Docker build."""
import argparse,json,time
from urllib.request import Request,urlopen
from urllib.error import URLError

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--url',required=True);parser.add_argument('--revision',required=True);parser.add_argument('--wait',type=int,default=60)
    args=parser.parse_args();base=args.url.rstrip('/')
    def request(path,body=None):
        headers={'Origin':base,'Content-Type':'application/json'}
        data=None if body is None else json.dumps(body).encode()
        with urlopen(Request(base+path,data=data,headers=headers),timeout=90) as response:return json.load(response)
    deadline=time.monotonic()+args.wait
    while True:
        try:
            health=request('/healthz')
            if health.get('revision')==args.revision and health.get('paper_only'):break
        except (URLError,ValueError,TimeoutError):pass
        if time.monotonic()>deadline:raise RuntimeError('Expected deployment revision did not become healthy')
        time.sleep(15)
    for path,marker in [('/','BetCheck'),('/app.js','fetch'),('/named.js','WeatherSignal')]:
        with urlopen(base+path,timeout=60) as r:assert marker in r.read().decode()
    state=request('/api/state');assert state['named_models']['available']
    def job(path,body,key='job'):
        request(path,body)
        end=time.monotonic()+300
        while time.monotonic()<end:
            state=request('/api/state');status=(state.get(key) or {}).get('status')
            if status=='complete':return state
            if status in ('failed','cancelled','interrupted'):raise RuntimeError(str(state[key]))
            time.sleep(2)
        raise RuntimeError('Job timed out: '+path+' '+str(state.get(key)))
    original=state['named_models']['selection']
    config={'name':'ConsensusBlend','weather_weight':.6}
    try:
        assert request('/api/named/select',{'configuration':config})['weather_weight']==.6
        evaluated=job('/api/named/evaluate',{'configuration':config,'bankroll':1000})
        assert evaluated['named_models']['evaluation']['events']>0
        trained=job('/api/named/train',{'configuration':config,'epochs':1,'freeze_backbone':True})
        assert trained['named_models']['checkpoints']
        demo=request('/api/demo',{})
        assert request('/api/report/'+demo['run_id'])
        assert request('/api/network-check',{})['status']=='connected'
        board=job('/api/discover',{'categories':['weather']},'analysis_job')['board']
        assert board and board['markets'],'No weather markets returned'
        print(json.dumps({'revision':args.revision,'models_available':True,'evaluation_events':evaluated['named_models']['evaluation']['events'],'one_epoch_training':True,'synthetic_paper_engine':True,'kalshi_connection':True,'weather_markets':len(board['markets'])},indent=2))
    finally:request('/api/named/select',{'configuration':original})

if __name__=='__main__':main()
