"""Audit fee precision on identical past fills; never relabel slippage as observed."""
import argparse
import json
from pathlib import Path
from .btc_research import digest, save
from .btc_flow_model import single_fill_cost


def audit(source,output):
    source,output=Path(source),Path(output);output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((source/'manifest.json').read_text());raw=(source/'report.json').read_bytes()
    if digest(raw)!=manifest['report_sha256']:raise ValueError('Loss audit hash mismatch')
    result=dict(purpose='Fixed fills/quantities; account-precision fee sensitivity, not strategy reruns',models={})
    for name,model in json.loads(raw)['models'].items():
        scenarios={}
        for precision in ('.01','.0001'):
            fee=sum(single_fill_cost(min(1,t['price']+.02),t['quantity'],precision=precision)[0] for t in model['trades'])
            old=model['summary'];scenarios[precision]=dict(fees=fee,net_pnl=old['net_pnl']+old['fees']-fee,
                fee_saving_vs_prior=old['fees']-fee)
        result['models'][name]=scenarios
    result['limitations']=['Single fills only; no partial-fill accumulator/rebate simulation',
        'Account type not supplied; cent precision retained as primary',
        'Official .07 schedule plus current series metadata support coefficient, but historical event overrides not proven',
        '2-cent slippage remains unmeasured; no historical depth evidence']
    result['sources']=['https://kalshi.com/docs/kalshi-fee-schedule.pdf','https://docs.kalshi.com/getting_started/fee_rounding']
    save(output/'report.json',result)
    save(output/'manifest.json',dict(input_sha256=digest(raw),
        code_sha256=digest(Path(__file__).read_bytes()),
        fee_code_sha256=digest(Path(__file__).with_name('btc_flow_model.py').read_bytes()),
        report_sha256=digest((output/'report.json').read_bytes())))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',default='reports/btc-loss-audit');p.add_argument('--output',default='reports/btc-cost-audit-v1')
    a=p.parse_args();print(json.dumps(audit(a.source,a.output),indent=2))
