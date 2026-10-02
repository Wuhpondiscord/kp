"""Explicit hypothetical-fill sizing experiments, never historical depth replay."""
from collections import defaultdict
import math
import numpy as np
from .core import dec,stamp,fee_and_cost
from .live import qualify_signal

PROTOCOL=dict(bankroll=1000,min_edge=.04,bet_fraction=.01,event_fraction=.03,portfolio_fraction=.20,
    policies=['one_contract','fixed_10_dollars','bankroll_1_percent','quarter_kelly'],
    primary_slippage=.02,stress_slippage=[0,.05],maximum_contracts=100,
    selection='Require credible validation edge before increasing risk; no test selection',
    limitations='Reused dates and assumed quote fills. No historical quantity/depth evidence. Cost-valued open positions are not marked to market.')

def simulate(rows,probabilities,policy='one_contract',slippage=.02,max_contracts=100,bankroll=1000,min_edge=.04,execution_predict=None,loss_stops=False):
    rows=list(rows);probabilities=np.asarray(probabilities,dtype=float)
    if probabilities.shape!=(len(rows),) or not np.isfinite(probabilities).all() or np.any((probabilities<0)|(probabilities>1)):
        raise ValueError('Require exactly one finite probability in [0,1] per row')
    for r in rows:
        decision=stamp(r['at'])
        if stamp(r['settled_at'])<=decision or stamp(r['close_time'])<=decision:raise ValueError('Decision must precede close and settlement')
        if r['y'] not in (0,1):raise ValueError('Invalid settled outcome')
        def quote(bid,ask):
            if not all(math.isfinite(float(v)) for v in (bid,ask)) or not 0<=float(bid)<=float(ask)<=1:
                raise ValueError('Invalid scenario quote')
        quote(r['bid'],r['ask'])
        if r['execution_at']:
            # Equality is the explicitly optimistic same-quote scenario.
            if stamp(r['execution_at'])<decision:raise ValueError('Execution cannot precede prediction')
            quote(r['execution_bid'],r['execution_ask'])
    if policy not in PROTOCOL['policies']:raise ValueError('Unknown sizing policy')
    if not math.isfinite(bankroll) or not 0<=slippage<=1 or not 1<=max_contracts<=10000 or bankroll<=0:raise ValueError('Invalid scenario parameters')
    if not 0<=min_edge<1:raise ValueError('Invalid edge threshold')
    initial=dec(bankroll);cash=initial;positions={};done=set();ledger=[];opportunities=set();opportunity_groups=set()
    realized=dec(0);daily=defaultdict(lambda:dec(0));halted=False
    events=[];skips=defaultdict(int);peak=initial;drawdown=dec(0);max_open=dec(0);max_event=dec(0);max_bet=dec(0)
    for r,p in zip(rows,probabilities):
        events.append((stamp(r['settled_at']),0,r,None))
        intent=qualify_signal(float(p),r['bid'],r['ask'],'.07',min_edge)
        if not intent:continue
        opportunities.add(r['ticker']);opportunity_groups.add((r['series'],r['group']))
        if not r['execution_at']:skips['missing_later_quote']+=1;continue
        events.append((stamp(r['execution_at']),1,r,(float(p),intent['signal_side'])))
    for at,kind,r,signal in sorted(events,key=lambda x:(x[0],x[1],x[2]['ticker'])):
        ticker=r['ticker'];group=(r['series'],r['group'])
        if kind==0:
            pos=positions.pop(ticker,None)
            if pos:
                payout=dec(r['y'] if pos['side']=='yes' else 1-r['y'])*pos['quantity'];cash+=payout
                profit=payout-pos['cost'];realized+=profit;daily[at.date().isoformat()]+=profit
                if realized<=-initial*dec('.05'):halted=True
                ledger.append(dict(ticker=ticker,event_group=' / '.join(group),day=r['group'],quantity=pos['quantity'],cost=float(pos['cost']),fees=float(pos['fee']),pnl=float(payout-pos['cost'])))
        else:
            if ticker in done:continue
            if loss_stops and (halted or daily[at.date().isoformat()]<=-initial*dec('.03')):skips['loss_stop']+=1;continue
            if at>=stamp(r['close_time']) or at>=stamp(r['settled_at']):skips['closed']+=1;continue
            p,side=signal;win=dec(p) if side=='yes' else 1-dec(p)
            if execution_predict:
                # Price-only model adapter; must not silently reuse weather inputs.
                refreshed=dict(r,at=r['execution_at'],p=(r['execution_bid']+r['execution_ask'])/2,
                    bid=r['execution_bid'],ask=r['execution_ask'],spread=r['execution_ask']-r['execution_bid'],
                    hours_left=(stamp(r['close_time'])-at).total_seconds()/3600)
                p=float(execution_predict([refreshed])[0])
                if not np.isfinite(p) or not 0<=p<=1:raise ValueError('Invalid execution prediction')
                win=dec(p) if side=='yes' else 1-dec(p)
            price=min(dec(1),(dec(r['execution_ask']) if side=='yes' else 1-dec(r['execution_bid']))+dec(slippage))
            _,one=fee_and_cost(price,1,dec('.07'))
            if win-one<dec(min_edge):skips['edge_lost_at_execution']+=1;continue
            open_cost=sum((x['cost'] for x in positions.values()),dec(0));capital=cash+open_cost
            exposure=sum((x['cost'] for x in positions.values() if x['group']==group),dec(0))
            per_bet=initial*dec(PROTOCOL['bet_fraction'])
            budget=min(cash,per_bet,initial*dec(PROTOCOL['event_fraction'])-exposure,initial*dec(PROTOCOL['portfolio_fraction'])-open_cost)
            if policy=='fixed_10_dollars':budget=min(budget,dec(10))
            if policy=='bankroll_1_percent':budget=min(budget,capital*dec('.01'))
            if policy=='quarter_kelly':
                fraction=max(dec(0),(win-one)/(1-one))*dec('.25') if one<1 else dec(0)
                budget=min(budget,capital*fraction)
            limit=1 if policy=='one_contract' else max_contracts
            quantity=0
            for q in range(1,limit+1):
                fee,cost=fee_and_cost(price,q,dec('.07'))
                if cost<=budget and win-cost/q>=dec(min_edge):quantity=q
            if not quantity:skips['risk_or_whole_contract_limit']+=1;continue
            fee,cost=fee_and_cost(price,quantity,dec('.07'));cash-=cost;done.add(ticker)
            positions[ticker]=dict(side=side,quantity=quantity,cost=cost,fee=fee,group=group)
            max_open=max(max_open,open_cost+cost);max_event=max(max_event,exposure+cost);max_bet=max(max_bet,cost)
        equity=cash+sum((x['cost'] for x in positions.values()),dec(0))
        peak=max(peak,equity);drawdown=max(drawdown,peak-equity)
    invested=sum(t['cost'] for t in ledger);contracts=sum(t['quantity'] for t in ledger);pnl=float(cash-initial)
    return dict(policy=policy,slippage=slippage,max_contracts=max_contracts,starting_bankroll=float(initial),pnl=pnl,
        minimum_edge=min_edge,execution_recomputed=execution_predict is not None,loss_stops=loss_stops,
        entries=len(ledger),contracts=contracts,opportunity_contracts=len(opportunities),opportunity_city_days=len(opportunity_groups),
        traded_city_days=len({x['event_group'] for x in ledger}),capital_deployed=invested,fees=sum(t['fees'] for t in ledger),
        net_per_contract=pnl/contracts if contracts else None,return_on_deployed=pnl/invested if invested else None,
        bankroll_return=pnl/float(initial),max_open_cost=float(max_open),max_city_day_cost=float(max_event),max_bet_cost=float(max_bet),
        realized_cost_equity_drawdown=float(drawdown),unsettled_positions=len(positions),skipped_rows=dict(skips),ledger=ledger)

def intervals(rows,result,comparisons=1):
    days=sorted({r['group'] for r in rows});values={d:np.zeros(3) for d in days}
    for trade in result['ledger']:values[trade['day']]+=np.array([trade['pnl'],trade['quantity'],trade['cost']])
    x=np.array(list(values.values()));rng=np.random.default_rng(1729);ratios=[];zero=0
    for _ in range(2000):
        ix=[]
        while len(ix)<len(x):
            start=int(rng.integers(len(x)));ix.extend((start+j)%len(x) for j in range(min(7,len(x))))
        total=x[ix[:len(x)]].sum(axis=0)
        if total[1]<=0 or total[2]<=0:zero+=1;continue
        ratios.append([total[0]/total[1],total[0]/total[2]])
    bounds=np.quantile(ratios,[.025/comparisons,1-.025/comparisons],axis=0).tolist() if ratios else [[None,None],[None,None]]
    return dict(net_per_contract=[bounds[0][0],bounds[1][0]],return_on_deployed=[bounds[0][1],bounds[1][1]],
        sparse=result['traded_city_days']<30,zero_trade_resamples=zero,comparisons=comparisons,
        note='Exploratory seven-day paired cluster bootstrap of ratios; zero-trade resamples excluded from ratios. Sparse groups cannot substantiate edge. Quantities do not create independent outcomes.')
