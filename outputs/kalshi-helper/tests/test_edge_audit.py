import unittest
from copy import deepcopy
from helper.evaluation import quote_scenario
from helper.edge_audit import audit
from test_research import rows

class EdgeAuditTests(unittest.TestCase):
    def test_same_quote_counterfactual_does_not_use_future_quote(self):
        data=rows(1)[:1];data[0]['y']=1
        other=deepcopy(data);other[0].update(execution_bid=.98,execution_ask=.99)
        a=quote_scenario(data,[.7],slippage=0,execution_mode='decision_quote')
        b=quote_scenario(other,[.7],slippage=0,execution_mode='decision_quote')
        self.assertEqual(a,b);self.assertEqual(a['trades'],1)
        self.assertEqual(quote_scenario(other,[.7],slippage=0)['trades'],0)
        self.assertNotEqual(data[0]['execution_at'],data[0]['at'])

    def test_event_partition_and_contribution_accounting(self):
        data=rows(1)
        data[0].update(event='E',lower=-float('inf'),upper=70,y=1,horizon=24)
        data[1].update(event='E',lower=70,upper=float('inf'),y=0,horizon=24)
        result=audit(data,[.7,.4])
        self.assertEqual(result['event_distribution']['complete_event_times'],1)
        self.assertAlmostEqual(result['event_distribution']['mean_raw_probability_mass_error'],.1)
        horizon=next(x for x in result['segments'] if x['segment']=='Horizon 24')
        self.assertAlmostEqual(horizon['headline_logloss_contribution'],horizon['model']['log_loss']-horizon['market']['log_loss'])
        data[1]['lower']=71
        self.assertEqual(audit(data,[.7,.4])['event_distribution']['complete_event_times'],0)
