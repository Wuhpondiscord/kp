"""Paired prospective ablation; never substitutes missing external data."""
import json,pickle
from pathlib import Path
import numpy as np
from .weather_research import RemainingMaximum,build_dataset,partitions
from .evaluation import score,paired_block_interval

class PolyMaximum(RemainingMaximum):
    def inputs(self,rows):
        g,x=super().inputs(rows);extra=[]
        for r in rows:
            a,b=r.get('poly_median_low'),r.get('poly_median_high')
            if not r.get('poly_available') or a is None or b is None:raise ValueError('Complete finite Polymarket median required')
            low,high=r.get('poly_q25_low'),r.get('poly_q75_high')
            spread=r.get('poly_mean_spread')
            extra.append([((a+b)/2-r['remaining_forecast_max'])/5,(b-a)/10,spread if spread is not None else 0,float(spread is None),
                r.get('poly_mean_quote_interval_width',0),
                r.get('poly_two_sided_fraction',1),(high-low)/10 if low is not None and high is not None else 0,
                float(low is None or high is None)])
        return g,np.column_stack([x,np.asarray(extra)])

def compare(root,output,observation_calibration=None):
    folder=Path(output);folder.mkdir(parents=True,exist_ok=False)
    protocol=dict(arms=['without_polymarket','with_polymarket'],weights=[0,.25,.5,1],
        observation_calibration=RemainingMaximum(observation_calibration).observation_protocol(),
        matching='Same eligible Kalshi rows, dates and outcomes for both arms; Polymarket is related-station context only',
        selection='Validation log loss with non-worsening Brier gate; no test tuning',promotion=False)
    (folder/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    rows,audit=build_dataset(root)
    paired=[r for r in rows if r.get('poly_available') and r.get('poly_median_low') is not None and r.get('poly_median_high') is not None]
    try:train,val,test=partitions(paired)
    except ValueError as exc:
        result=dict(status='insufficient_paired_data',paired_rows=len(paired),reason=str(exc),audit=audit,performance_comparison_available=False)
        (folder/'report.json').write_text(json.dumps(result,indent=2),encoding='utf-8');return result
    primary=lambda rs:[r for r in rs if 12<=r['remaining_hours']<=24 and .05<=r['p']<=.95]
    val,test=primary(val),primary(test)
    if not val or not test:raise ValueError('Empty primary comparison cohort')
    vb=np.asarray([r['p'] for r in val]);market=score(val,vb);models={};locks={}
    for name,cls in [('without_polymarket',RemainingMaximum),('with_polymarket',PolyMaximum)]:
        m=cls(observation_calibration).fit(train);v=m.predict(val);choices=[]
        for weight in protocol['weights']:
            metric=score(val,weight*v+(1-weight)*vb);choices.append(dict(weight=weight,metrics=metric))
        chosen=min((c for c in choices if c['metrics']['brier']<=market['brier']),key=lambda c:c['metrics']['log_loss'])
        locks[name]=chosen;models[name]=m;(folder/(name+'.pkl')).write_bytes(pickle.dumps(m))
    (folder/'selection.json').write_text(json.dumps(locks,indent=2),encoding='utf-8')
    predictions={name:locks[name]['weight']*m.predict(test)+(1-locks[name]['weight'])*np.asarray([r['p'] for r in test]) for name,m in models.items()}
    result=dict(status='research_only',paired_rows=len(paired),metrics={name:score(test,p) for name,p in predictions.items()},
        paired_difference=paired_block_interval(test,predictions['with_polymarket'],predictions['without_polymarket']),
        promotion=False,performance_comparison_available=True,warning='No automatic assertion of untouched dates or executable profitability.')
    (folder/'report.json').write_text(json.dumps(result,indent=2),encoding='utf-8');return result
