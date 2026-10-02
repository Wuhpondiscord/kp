"""Auditable quote-based recommendations and model diagnostics, not invented rationales."""
import re
from .core import stamp,utcnow,dec,fee_and_cost
from .evaluation import ModelService
from .research_store import ResearchStore


def explain_market(store,card,service=None,diagnostics=True):
    if ResearchStore(store).get('named_selection'):
        from .named_models import recommend
        return recommend(store,card)
    service=service or ModelService(store);rs=ResearchStore(store);active=rs.get('active_model')
    slug=re.sub(r'[^a-z0-9]+','-',card.get('series_title',card['title']).lower()).strip('-')
    link=f"https://kalshi.com/markets/{card['series'].lower()}/{slug}/{card['event_ticker'].lower()}"
    result=dict(action='WATCH',label='Watch — no supported model',reasons=[],probability=None,
        market_probability=card.get('midpoint'),kalshi_url=link,fees=None,edges=None,sensitivity=[],validation=None,test=None)
    if card.get('bid') is None or card.get('ask') is None:
        result['reasons']=['No valid two-sided market quote is available.'];return result
    prediction=service.predict_signal(card['series'],card['bid'],card['ask'],card['at'],card['close_time'],experimental=True)
    p,note=prediction.probability,prediction.reason
    learned=prediction.signal
    hours=(stamp(card['close_time'])-stamp(card['at'])).total_seconds()/3600
    result['inputs']=dict(market_probability=card['midpoint'],spread=card['ask']-card['bid'],hours_left=hours,series=card['series'],quote_at=card['at'])
    if not learned:
        result['reasons']=[note,'A market price is not an independent forecast.'];return result
    result.update(probability=p,model_id=active['model_id'],model_note=note)
    report=(rs.experiment(active['experiment_id']) or {}) if diagnostics else {}
    candidate=next((c for c in report.get('candidates',[]) if c['name']==active['experimental']),None)
    result['validation']=candidate['validation'] if candidate else None
    if report.get('selected_model')==active['experimental']:
        result['test']=report.get('holdout');result['test_market_baseline']=report.get('market_baseline')
        result['paired_interval']=report.get('paired_interval')
    predictor=service.bundle['experimental']
    row=dict(p=card['midpoint'],spread=card['ask']-card['bid'],hours_left=hours,series=card['series'])
    changes=[('Market odds 5 points lower',dict(row,p=max(.001,row['p']-.05))),
                         ('Market odds 5 points higher',dict(row,p=min(.999,row['p']+.05))),
                         ('Spread narrowed to 1 cent',dict(row,spread=.01)),
                         ('12 hours remaining',dict(row,hours_left=12))]
    for name,changed in changes if diagnostics else []:
        alt=float(predictor.predict([changed])[0])
        result['sensitivity'].append(dict(change=name,probability=alt,difference=alt-p))
    result['reasons']=[f"The trained model changes the YES probability by {(p-card['midpoint'])*100:+.1f} percentage points relative to market midpoint.",
        ('This calibrator uses market probability only; it does not use weather forecasts, spread or city as predictive inputs.'
         if predictor.name in ('beta_market','calibrated_market') else
         'Inputs are market price, spread, time remaining and city. What-if changes show model sensitivity, not causal explanations.')]
    if card.get('fee_rate') is not None:
        yes_fee,yes_cost=fee_and_cost(dec(card['ask']),1,dec(card['fee_rate']))
        no_fee,no_cost=fee_and_cost(dec(card['no_ask']),1,dec(card['fee_rate']))
        edges=dict(yes=p-float(yes_cost),no=1-p-float(no_cost));side=max(edges,key=edges.get)
        result.update(edges=edges,fees=dict(yes=float(yes_fee),no=float(no_fee)),lean=side.upper(),minimum_edge=.04)
        if edges[side]<.04:
            result.update(action='PASS',label='Pass — not enough edge after estimated fees')
        elif not active['passed']:
            result.update(action='WATCH',label=f'Experimental {side.upper()} lean — validation gate not passed')
        else:result.update(action='PAPER_'+side.upper(),label=f'Paper {side.upper()} candidate — check fresh book depth')
        result['reasons'].append(f'Estimated {side.upper()} edge is {edges[side]*100:.1f} cents per $1 contract after conservative one-contract fees; the entry threshold is 4 cents.')
    else:result['reasons'].append('Fees are unknown; no paper entry recommendation is available.')
    age=(stamp(utcnow())-stamp(card['at'])).total_seconds()
    if age>180 or age<0:
        result.update(action='WATCH',label='Refresh prices before considering a paper bet')
        result['reasons'].append('Quote is stale or has an invalid timestamp.')
    if stamp(card['close_time'])<=stamp(utcnow()) or card.get('tradeable') is False:
        result.update(action='WATCH',label='Closed market — no new paper entry')
    if not active['passed']:result['reasons'].append('The model has not demonstrated a validated advantage over market prices. This is an experimental forecast.')
    return result
