import json
from pathlib import Path
import tempfile
import unittest
from helper.btc_forward import append,connect,depth_fill,levels,replay,verify
from helper.btc_research import digest,save
from helper.core import dec


class ForwardTests(unittest.TestCase):
    def protocol(self):
        return dict(bankroll=1000,bet_cap=10,day_open_cap=30,portfolio_cap=200,max_contracts=100,
                    fee_rate=.07,slippage=.02,min_edge=.04,min_latency_seconds=2,max_latency_seconds=30,code_hashes={})

    def book(self):
        return dict(orderbook_fp={'yes_dollars':[['0.39','5']], 'no_dollars':[['0.59','2.9'],['0.58','1']]})

    def experiment(self,d,probability=.9):
        Path(d,'model.joblib').write_bytes(b'not loaded by replay')
        p=dict(self.protocol(),model_sha256=digest(Path(d,'model.joblib').read_bytes()))
        save(Path(d,'protocol.json'),p)
        db=connect(d);append(db,'initialized',None,{'protocol_sha256':digest(Path(d,'protocol.json').read_bytes())})
        prediction=dict(row=dict(at='2026-10-04T12:10:00Z',close_time='2026-10-04T12:15:00Z',
                    ask=.41,bid=.39,group='2026-10-04',p=.4),gate_accepted=True,
                    probabilities={'market':.4,'BTCMarketGuard':probability,'BTCRegime_gate':probability})
        append(db,'prediction','A',prediction,'2026-10-04T12:10:00Z')
        return db

    def test_complement_price_depth_and_fractional_quantity(self):
        self.assertEqual(levels(self.book(),'yes'),[(dec('.41'),2),(dec('.42'),1)])
        q,cost,fee=depth_fill(self.book(),'yes',.9,dec(10),self.protocol())
        self.assertEqual(q,3);self.assertLessEqual(cost,dec(10));self.assertGreater(fee,0)
        q,_,_=depth_fill(self.book(),'yes',.9,dec('.45'),self.protocol())
        self.assertLessEqual(q,1)

    def test_replay_waits_for_later_book_and_uses_actual_depth(self):
        with tempfile.TemporaryDirectory() as d:
            db=self.experiment(d)
            append(db,'book','A',self.book(),'2026-10-04T12:10:01Z')
            self.assertEqual(replay(d)['strategies']['BTCMarketGuard']['open_positions'],0)
            append(db,'book','A',self.book(),'2026-10-04T12:10:05Z')
            report=replay(d)['strategies']['BTCMarketGuard']
            self.assertEqual(report['open_positions'],1);self.assertEqual(report['realized_pnl'],0)
            append(db,'outcome','A',{'y':1},'2026-10-04T12:15:05Z')
            report=replay(d)
            self.assertEqual(report['strategies']['BTCMarketGuard']['settled_contracts'],3)
            self.assertGreater(report['strategies']['BTCMarketGuard']['realized_pnl'],0)
            self.assertEqual(report['strategies']['abstain']['realized_pnl'],0)
            self.assertEqual(report['forecast_scores']['market']['events'],1)
            db.close()

    def test_late_or_pre_requested_book_cannot_fill(self):
        for received,started in [('12:10:40','12:10:39'),('12:10:05','12:09:59')]:
            with tempfile.TemporaryDirectory() as d:
                db=self.experiment(d)
                append(db,'book','A',dict(self.book(),request_started_at='2026-10-04T'+started+'Z'),'2026-10-04T'+received+'Z')
                self.assertEqual(replay(d)['strategies']['BTCMarketGuard']['open_positions'],0)
                db.close()

    def test_frozen_protocol_and_record_tampering_fail(self):
        with tempfile.TemporaryDirectory() as d:
            db=self.experiment(d)
            original=Path(d,'protocol.json').read_bytes()
            Path(d,'protocol.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'protocol'):verify(d)
            Path(d,'protocol.json').write_bytes(original)
            db.execute('UPDATE records SET body=? WHERE kind=?',('{}','prediction'));db.commit()
            with self.assertRaisesRegex(ValueError,'hash'):replay(d)
            db.close()

    def test_malformed_depth_rejected(self):
        for levels_ in [[['.5','-1']],[['NaN','1']],[['.5','2'],['.5','3']]]:
            with self.assertRaises(ValueError):levels({'orderbook_fp':{'no_dollars':levels_}},'yes')

    def test_later_price_cannot_create_an_unqualified_intent(self):
        with tempfile.TemporaryDirectory() as d:
            db=self.experiment(d,probability=.4)
            cheap={'orderbook_fp':{'yes_dollars':[['.19','100']],'no_dollars':[['.79','100']]}}
            append(db,'book','A',cheap,'2026-10-04T12:10:05Z')
            report=replay(d)['strategies']['BTCMarketGuard']
            self.assertEqual(report['open_positions'],0)
            self.assertEqual(report['skips']['no_decision_edge'],1)
            db.close()
