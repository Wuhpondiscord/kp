"""Small predeclared validation search, locked before inspecting final P/L."""
import json,pickle
from pathlib import Path
from .training import split_dataset
from .sizing import simulate,intervals
from .cohorts import price_primary,PRICE_COHORT

def rejection_reasons(result):
    reasons=[]
    if result['traded_city_days']<30:reasons.append('Fewer than 30 independent traded city/day groups')
    lower=result['uncertainty']['net_per_contract'][0]
    if lower is None:reasons.append('No estimable net-profit interval')
    elif lower<=0:reasons.append('Adjusted net-profit lower bound is not positive')
    if result['pnl']<=0:reasons.append('Validation net profit is not positive')
    return reasons

def run(training_folder,output):
    folder=Path(output);folder.mkdir(parents=True,exist_ok=False)
    source=Path(training_folder)
    protocol=dict(edges=[.02,.04,.06,.08],policies=['one_contract','bankroll_1_percent'],
        hold='settlement',slippage=.02,bankroll=1000,loss_stops=True,execution='recompute price-only model at later quote',
        primary=PRICE_COHORT,minimum_validation_city_days=30,
        selection='Highest validation lower net-per-contract bound; eight-comparison adjustment; otherwise no trade',
        warning='Historical quote scenarios lack depth; reused dates cannot establish a new edge. No early-exit tuning without intratrade books.')
    (folder/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    rows=[json.loads(x) for x in (source/'features.jsonl').read_text(encoding='utf-8').splitlines()]
    model=pickle.loads((source/'best.pkl').read_bytes())
    _,validation,test=split_dataset(rows)
    # Predefined cohort, not selected from test profitability.
    cohort=lambda rs:[r for r in rs if price_primary(r)]
    validation=cohort(validation);test=cohort(test)
    if not validation or not test:raise ValueError('No primary-cohort validation or test rows')
    candidates=[];vp=model.predict(validation)
    for edge in protocol['edges']:
        for policy in protocol['policies']:
            result=simulate(validation,vp,policy,min_edge=edge,execution_predict=model.predict,loss_stops=True)
            result['uncertainty']=intervals(validation,result,comparisons=8);result.pop('ledger')
            result['rejection_reasons']=rejection_reasons(result);candidates.append(result)
    eligible=[r for r in candidates if not r['rejection_reasons']]
    chosen=max(eligible,key=lambda r:r['uncertainty']['net_per_contract'][0]) if eligible else None
    selection=dict(action='research_only' if chosen else 'no_trade',selected=chosen,candidates=candidates)
    (folder/'selection.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
    results=[]
    if chosen:
        for slip in (0,.02,.05):
            result=simulate(test,model.predict(test),chosen['policy'],slippage=slip,min_edge=chosen['minimum_edge'],execution_predict=model.predict,loss_stops=True)
            result['uncertainty']=intervals(test,result);result.pop('ledger');results.append(result)
    report=dict(selection=selection,test_scenarios=results,promotion=False,test_rows=len(test),
        reason='Validation rejected all strategies; no test strategy selected' if not chosen else 'Locked validation choice; test is descriptive only')
    (folder/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');return report
