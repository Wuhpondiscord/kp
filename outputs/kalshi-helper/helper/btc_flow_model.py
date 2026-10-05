"""Market-relative aggressor flow experiment with an explicit holdout lock."""
import argparse
from decimal import Decimal, ROUND_CEILING
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from .btc_refinement import logits
from .btc_inputs import validated_execution_rows
from .btc_research import digest, save, metrics, paired_interval, pnl_interval
from .sizing import simulate


def single_fill_cost(price, quantity, rate=.07, precision='.01'):
    """Official single-buy-fill rounding mechanics, no invented partial-fill rebates."""
    p=Decimal(str(price));q=Decimal(str(quantity));r=Decimal(str(rate));grid=Decimal(precision)
    if not all(x.is_finite() for x in (p,q,r,grid)) or not 0<=p<=1 or q<=0 or q!=q.to_integral_value() or r<0 or grid not in (Decimal('.01'),Decimal('.0001')):
        raise ValueError('Invalid fee inputs')
    gross=p*q;trade=(r*q*p*(1-p)).quantize(Decimal('.000001'),rounding=ROUND_CEILING)
    debit=(gross+trade).quantize(grid,rounding=ROUND_CEILING)
    return float(debit-gross),float(debit)


class FlowCorrection:
    def __init__(self, flow=True):self.flow=flow
    def design(self,rows,fit=False):
        m=logits([r['p'] for r in rows]);x=np.array([r['flow_x'] for r in rows],dtype=float)
        if x.shape!=(len(rows),4) or not np.isfinite(x).all():raise ValueError('Invalid flow vector')
        if fit:self.mean=x.mean(axis=0);self.scale=np.maximum(x.std(axis=0),1e-6)
        z=np.clip((x-self.mean)/self.scale,-5,5)
        return m,np.column_stack([np.ones(len(rows)),m,z]) if self.flow else np.column_stack([np.ones(len(rows)),m])
    def fit(self,rows):
        offset,x=self.design(rows,True);y=np.array([r['y'] for r in rows])
        if len(rows)<50 or set(y)!={0,1}:raise ValueError('Insufficient training outcomes')
        def objective(w):
            z=offset+x@w
            return float(np.mean(np.logaddexp(0,z)-y*z)+.1*(w@w)),x.T@(expit(z)-y)/len(y)+.2*w
        result=minimize(objective,np.zeros(x.shape[1]),jac=True,method='L-BFGS-B',bounds=[(-.25,.25)]*x.shape[1])
        if not result.success:raise ValueError(result.message)
        self.coef=result.x;return self
    def predict(self,rows):
        offset,x=self.design(rows);return np.clip(expit(offset+x@self.coef),1e-6,1-1e-6)
    def artifact(self):return dict(flow=self.flow,mean=self.mean.tolist(),scale=self.scale.tolist(),coef=self.coef.tolist())
    @classmethod
    def load(cls,data):
        model=cls(data['flow']);model.mean=np.array(data['mean']);model.scale=np.array(data['scale']);model.coef=np.array(data['coef']);return model


def read_partition(root,name):
    manifest=json.loads((root/'manifest.json').read_text());path=root/(name+'.jsonl');raw=path.read_bytes()
    if digest(raw)!=manifest['files'][path.name]:raise ValueError('Dataset hash mismatch')
    if digest((root/'protocol.json').read_bytes())!=manifest['protocol_sha256']:raise ValueError('Protocol hash mismatch')
    rows=[json.loads(line) for line in raw.decode().splitlines()]
    from .core import stamp
    from .btc_new_history import PROTOCOL
    from .btc_research import unix
    bounds={'train':(PROTOCOL['start'],PROTOCOL['train_end']),
            'validation':(PROTOCOL['train_end'],PROTOCOL['validation_end']),
            'holdout':(PROTOCOL['validation_end'],PROTOCOL['end'])}
    a,b=bounds[name]
    for r in rows:
        if not unix(a)<=stamp(r['close_time']).timestamp()<unix(b):raise ValueError('Partition date overlap')
        if name!='train' and stamp(r['execution_at']).timestamp()<unix(a)+7200:raise ValueError('Missing embargo')
        if stamp(r['settled_at']).timestamp()>=unix(b):raise ValueError('Outcome crosses partition')
        if not stamp(r['flow_feature_time'])<=stamp(r['flow_available_at'])<=stamp(r['execution_at']):raise ValueError('Future flow input')
    return validated_execution_rows(rows)


def trade_uncertainty(paper,days):
    result=pnl_interval(paper,days)
    if result and (paper['entries']<30 or result['trade_days']<10):
        # Resampling three observed winners cannot represent unseen losing trades.
        result['net_per_contract_95_interval']=None
        result['return_on_deployed_95_interval']=None
        result['status']='insufficient_trade_evidence'
        result['note']='Intervals suppressed: fewer than 30 entries or 10 trading days. Resampling sparse winners would produce misleadingly narrow positive bounds.'
    return result


def score(rows,models):
    predictions=dict(market=np.array([r['p'] for r in rows]),volatility=np.array([r['diffusion'] for r in rows]),
        **{n:m.predict(rows) for n,m in models.items()})
    results={}
    for name,p in predictions.items():
        scenarios={}
        for slip in (0,.02,.05):
            paper=simulate(rows,p,policy='bankroll_1_percent',slippage=slip,bankroll=1000,min_edge=.04)
            paper['uncertainty']=trade_uncertainty(paper,{r['group'] for r in rows});scenarios[str(slip)]=paper
        results[name]=dict(scores=metrics(rows,p),paired_brier=paired_interval(rows,p),scenarios=scenarios)
    return results


def run(root,release=False):
    root=Path(root);frozen=root/'frozen-models.json'
    if release:
        validation=json.loads((root/'validation-report.json').read_text())
        if not validation['holdout_release_eligible']:raise ValueError('Validation failed the predeclared gate; holdout remains sealed')
        if digest(frozen.read_bytes())!=validation['frozen_model_sha256']:raise ValueError('Frozen model changed')
        artifact=json.loads(frozen.read_text())
        if artifact['code_sha256']!=digest(Path(__file__).read_bytes()):raise ValueError('Model code changed after validation')
        if artifact['source_manifest_sha256']!=digest((root/'manifest.json').read_bytes()):raise ValueError('Data changed after validation')
        if (root/'holdout-opened.json').exists():raise ValueError('Holdout already opened; do not present a rerun as untouched')
        save(root/'holdout-opened.json',dict(model_sha256=digest(frozen.read_bytes()),purpose='Single evaluation after validation gate'))
        models={n:FlowCorrection.load(m) for n,m in artifact['models'].items()}
        result=score(read_partition(root,'holdout'),models)
        save(root/'holdout-report.json',result);return result
    if (root/'holdout-opened.json').exists():raise ValueError('Research locked after holdout opening')
    previously_evaluated=(root/'validation-report.json').exists()
    train=read_partition(root,'train');validation=read_partition(root,'validation')
    from .core import stamp
    if max(stamp(r['settled_at']) for r in train)>=min(stamp(r['at']) for r in validation):raise ValueError('Overlapping labels')
    models=dict(market_control=FlowCorrection(False).fit(train),flow=FlowCorrection(True).fit(train))
    save(frozen,dict(models={n:m.artifact() for n,m in models.items()},
        source_manifest_sha256=digest((root/'manifest.json').read_bytes()),
        code_sha256=digest(Path(__file__).read_bytes()),
        release_gate='Validation Brier below market and price control, positive primary net P&L, >=30 entries; no parameter tuning'))
    result=score(validation,models);f=result['flow'];paper=f['scenarios']['0.02']
    passed=(f['scores']['brier']<min(result[n]['scores']['brier'] for n in ('market','market_control')) and paper['pnl']>0 and paper['entries']>=30)
    report=dict(training_rows=len(train),validation_rows=len(validation),models=result,holdout_loaded=False,validation_previously_evaluated=previously_evaluated,
        holdout_release_eligible=passed,frozen_model_sha256=digest(frozen.read_bytes()),deployment_allowed=False)
    save(root/'validation-report.json',report);return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',default='reports/btc-july-flow-v1')
    parser.add_argument('--release-holdout',action='store_true');args=parser.parse_args()
    r=run(args.source,args.release_holdout)
    print(json.dumps({n:dict(brier=v['scores']['brier'],log_loss=v['scores']['log_loss'],pnl=v['scenarios']['0.02']['pnl'],entries=v['scenarios']['0.02']['entries']) for n,v in r.get('models',r).items()},indent=2))
