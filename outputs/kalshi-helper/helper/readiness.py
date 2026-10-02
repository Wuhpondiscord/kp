"""Read-only readiness checklist. Never enables real orders."""
from pathlib import Path
from .core import dec
from .prospective import verify

def assess(store):
    sessions=[]
    for r in store.runs():
        report=r['report']
        if not report.get('prospective'):continue
        path=store.root/'prospective'/r['id']/'journal.jsonl'
        try:
            integrity=verify(path)
            if report.get('evidence_head') and integrity['head']!=report['evidence_head']:
                raise ValueError('Journal head differs from saved session report')
        except (OSError,ValueError,KeyError,TypeError) as exc:integrity=dict(verified=False,error=str(exc))
        reconciled=dec(report['initial_bankroll'])+dec(report['realized_pnl'])-dec(report['open_cost'])==dec(report['cash'])
        sessions.append(dict(id=r['id'],status=r['status'],integrity=integrity,cash_reconciled=reconciled,fills=report['fills'],evidence_directory=str(path.parent)))
    return dict(real_money_ready=False,sessions=sessions,blocking_requirements=[
        'No completed fresh prospective profitability study with adequate independent events and positive net-cost uncertainty bound.',
        'No validated full-season or multi-year generalization.',
        'No verified real-money execution adapter, exchange reconciliation, or live fee/queue equivalence.',
        'The coherent event-weather prototype was rejected by validation; fresh remaining-day inputs need their own study.'],
        next_action='Continue frozen paper collection; no real-money pilot is authorized or implemented by this checklist.')
