"""Frozen-forecast trading diagnostics; validation-only policy selection."""
import numpy as np
from .evaluation import quote_scenario

PROTOCOL=dict(thresholds=[.02,.04,.06,.08],minimum_validation_trades=30,minimum_validation_events=30,
    block_days=7,draws=2000,seed=1729,selection='Positive simultaneous lower bound on validation daily P/L; otherwise no trade',
    caveat='Validation already selected forecast weights; strategy results are exploratory, not nested validation or executable profit.')

def uncertainty(rows,report,comparisons=1):
    days=sorted({r['group'] for r in rows});totals={d:0. for d in days}
    groups={r['ticker']:r['group'] for r in rows}
    for trade in report['ledger']:totals[groups[trade['ticker']]]+=float(trade['pnl'])
    values=np.array(list(totals.values()));rng=np.random.default_rng(PROTOCOL['seed'])
    draws=[]
    for _ in range(PROTOCOL['draws']):
        ix=[]
        while len(ix)<len(days):
            start=int(rng.integers(len(days)));ix.extend((start+j)%len(days) for j in range(min(len(days),PROTOCOL['block_days'])))
        draws.append(float(values[ix[:len(days)]].mean()))
    alpha=.025/comparisons
    return dict(mean_daily_pnl=float(values.mean()),lower=float(np.quantile(draws,alpha)),upper=float(np.quantile(draws,1-alpha)),
        dates=len(days),traded_events=len({r['event'] for r in report['ledger']}),comparisons=comparisons,
        sparse_events=len({r['event'] for r in report['ledger']})<PROTOCOL['minimum_validation_events'],
        note='Calendar-group daily P/L, including zero-trade days; circular seven-day blocks. Sparse-event intervals are unreliable even when positive. Assumed fills, not verified returns.')

def compact(rows,report,comparisons=1):
    return dict({k:v for k,v in report.items() if k not in ('ledger','curve')},interval=uncertainty(rows,report,comparisons))

def select_policy(val,predictions):
    candidates=[]
    for threshold in PROTOCOL['thresholds']:
        scenario=quote_scenario(val,predictions,slippage=.02,decision_qualified=True,min_edge=threshold)
        result=compact(val,scenario,len(PROTOCOL['thresholds']))
        result['eligible']=result['trades']>=PROTOCOL['minimum_validation_trades'] and not result['interval']['sparse_events'] and result['interval']['lower']>0
        candidates.append(result)
    eligible=[r for r in candidates if r['eligible']]
    winner=max(eligible,key=lambda r:r['interval']['lower']) if eligible else None
    return dict(action='trade' if winner else 'no_trade',minimum_edge=winner['minimum_edge'] if winner else None,candidates=candidates)

def evaluate_policy(test,predictions,selection):
    if selection['action']=='no_trade':return dict(action='no_trade',trades=0,pnl='0',reason='No validation policy cleared the predeclared evidence threshold. Abstention is not proof of a profitable model.')
    report=quote_scenario(test,predictions,slippage=.02,decision_qualified=True,min_edge=selection['minimum_edge'])
    return compact(test,report)
