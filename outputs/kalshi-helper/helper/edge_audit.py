"""Descriptive fixed-model diagnostics. Never select weights or trade thresholds."""
from collections import defaultdict
import math
from .evaluation import score,losses,quote_scenario
from .core import dec,fee_and_cost


def bucket(p):
    if p<.02:return '0–2%'
    if p<.1:return '2–10%'
    if p<.3:return '10–30%'
    if p<.7:return '30–70%'
    if p<.9:return '70–90%'
    return '90–100%'


def audit(rows,probabilities):
    groups=defaultdict(list);edge_groups=defaultdict(list)
    for i,(r,p) in enumerate(zip(rows,probabilities)):
        groups['Horizon '+str(r.get('horizon'))].append(i)
        groups['Price '+bucket(r['p'])].append(i)
        groups['Horizon '+str(r.get('horizon'))+' / price '+bucket(r['p'])].append(i)
        choices=[]
        for side,win,price in (('yes',p,r['ask']),('no',1-p,1-r['bid'])):
            _,cost=fee_and_cost(dec(price),1,dec('.07'))
            choices.append((win-float(cost),side,float(cost)))
        edge,side,cost=max(choices)
        label='negative' if edge<0 else '0–2 cents' if edge<.02 else '2–4 cents' if edge<.04 else '4–8 cents' if edge<.08 else '8+ cents'
        payout=r['y'] if side=='yes' else 1-r['y']
        later=None
        if r['execution_at']:
            price=r['execution_ask'] if side=='yes' else 1-r['execution_bid']
            _,debit=fee_and_cost(dec(price),1,dec('.07'));later=payout-float(debit)
        edge_groups[label].append(dict(edge=edge,pnl=payout-cost,later=later,event=r['event']))
    model_loss=losses(rows,probabilities);market_loss=losses(rows,[r['p'] for r in rows])
    # Sum row differences divided by original date sizes gives additive attribution.
    sizes=defaultdict(int)
    for r in rows:sizes[r['group']]+=1
    out=[]
    for label,ix in sorted(groups.items()):
        subset=[rows[i] for i in ix];pred=[probabilities[i] for i in ix]
        out.append(dict(segment=label,model=score(subset,pred),market=score(subset,[r['p'] for r in subset]),
            headline_logloss_contribution=sum((model_loss[i]-market_loss[i])/sizes[rows[i]['group']] for i in ix)/len(sizes),
            zero_bid_one_cent_ask_fraction=sum(r['bid']==0 and r['ask']<=.01 for r in subset)/len(subset)))
    edges=[]
    for label,values in sorted(edge_groups.items()):
        delayed=[v['later'] for v in values if v['later'] is not None]
        edges.append(dict(bucket=label,rows=len(values),events=len({v['event'] for v in values}),
            mean_claimed_net_edge=sum(v['edge'] for v in values)/len(values),
            same_quote_mean_pnl=sum(v['pnl'] for v in values)/len(values),
            delayed_fixed_cohort_mean_pnl=sum(delayed)/len(delayed) if delayed else None,delayed_rows=len(delayed)))
    events=defaultdict(list)
    for i,r in enumerate(rows):events[(r['event'],r['at'])].append(i)
    complete=[];excluded=0
    for key,ix in events.items():
        ordered=sorted(ix,key=lambda i:rows[i]['lower'])
        valid=(sum(rows[i]['y'] for i in ix)==1 and rows[ordered[0]]['lower']==-math.inf and rows[ordered[-1]]['upper']==math.inf
            and all(rows[a]['upper']==rows[b]['lower'] for a,b in zip(ordered,ordered[1:])))
        if not valid:excluded+=1;continue
        q=[float(probabilities[i]) for i in ix];m=[rows[i]['p'] for i in ix]
        if sum(q)<=0 or sum(m)<=0:excluded+=1;continue
        winner=next(j for j,i in enumerate(ix) if rows[i]['y'])
        complete.append(dict(model_loss=-math.log(max(q[winner]/sum(q),1e-8)),market_loss=-math.log(max(m[winner]/sum(m),1e-8)),mass_error=abs(sum(q)-1)))
    return dict(note='Post-review descriptive audit on reused dates. No tuning. Edge buckets count correlated rows, not a deployable strategy; fixed decision-time cohorts compare prices without later re-selection.',
        segments=out,edge_buckets=edges,event_distribution=dict(complete_event_times=len(complete),excluded_event_times=excluded,
            note='Diagnostic normalization of complete bracket partitions only; not raw model outputs or a new trained distribution.',
            normalized_model_logloss=sum(v['model_loss'] for v in complete)/len(complete) if complete else None,
            normalized_market_logloss=sum(v['market_loss'] for v in complete)/len(complete) if complete else None,
            mean_raw_probability_mass_error=sum(v['mass_error'] for v in complete)/len(complete) if complete else None),
        scenarios=[{k:v for k,v in quote_scenario(rows,probabilities,slippage=0,execution_mode=mode).items() if k not in ('ledger','curve')} for mode in ('decision_quote','later_quote')])
