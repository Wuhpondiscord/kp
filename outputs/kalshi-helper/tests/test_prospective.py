import json,tempfile,unittest
from pathlib import Path
import numpy as np
from helper.storage import Store
from helper.prospective import Recorder,verify
from helper.event_weather import EventMaximum
from helper.core import Engine,Settings
from test_engine import market,book,prediction,outcome,ts

class ProspectiveTests(unittest.TestCase):
    def test_evidence_chain_detects_mutation_and_duplicate_run(self):
        with tempfile.TemporaryDirectory() as folder:
            store=Store(folder);r=Recorder(store,'run',{'paper':True});r.append('prediction',{'p':.5})
            path=r.folder/'journal.jsonl';self.assertEqual(verify(path)['records'],2)
            with self.assertRaises(FileExistsError):Recorder(store,'run',{})
            text=path.read_text(encoding='utf-8').replace('"p": 0.5','"p": 0.9');path.write_text(text,encoding='utf-8')
            with self.assertRaises(ValueError):verify(path)

    def test_event_distribution_conserves_probability_and_ignores_labels(self):
        model=EventMaximum();model.theta=np.array([0,0,0,0,0,0,np.log(4)])
        base=dict(gfs=75,ecmwf=75,hours_left=12,series='KXHIGHNY',observed=True,observed_max=73)
        rows=[dict(base,lower=a,upper=b,y=i%2) for i,(a,b) in enumerate([(-np.inf,70),(70,75),(75,80),(80,np.inf)])]
        p=model.predict(rows);self.assertAlmostEqual(sum(p),1)
        np.testing.assert_array_equal(p,model.predict([dict(r,y=1-r['y']) for r in rows]))
        self.assertLess(p[0],.01)

    def test_session_loss_stop_rejects_next_trade_but_keeps_accounting(self):
        e=Engine(Settings(daily_loss_fraction='.001',session_loss_fraction='.001'))
        e.feed(market());e.feed(book());e.feed(prediction());e.feed(book(6));e.feed(outcome(payout='0'))
        m=market('B');m.update(available_at=ts(4001),close_time=ts(8000));e.feed(m)
        e.feed(book(4001,'B'));p=prediction('B',4001);p['expires_at']=ts(7000);e.feed(p);e.feed(book(4007,'B'))
        self.assertNotIn('B',e.positions);self.assertTrue(e.risk_halted)
        r=e.report();self.assertEqual(float(r['cash']),float(r['initial_bankroll'])+float(r['realized_pnl']))
