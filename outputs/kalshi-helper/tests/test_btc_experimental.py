import unittest
from unittest.mock import patch
from helper import btc_lab, btc_experimental as b

class ExperimentalTests(unittest.TestCase):
    def test_frozen_forecasts_and_full_ledger_match_research(self):
        import json
        specs=[('fixed_anchor','btc-anchor-study-v1','fixed_anchor'),('directional_joint','btc-directional-batch-v1','joint'),('delayed_micro','btc-microdata-v1','combined_delayed_micro')]
        for model,folder,key in specs:
            expected=json.loads((b.ROOT/folder/'report.json').read_text())['aggregate'][key]
            result=btc_lab.replay(dict(model=model,bankroll=1000))
            for metric in ('brier','log_loss'):
                self.assertAlmostEqual(result['scores'][metric],expected['scores'][metric],places=12)
            self.assertEqual(result['paper']['ledger'],expected['paper']['ledger'])
            self.assertFalse(result['live_enabled'])
            self.assertTrue(result['experimental'])
        rows,p=b.prepared();self.assertEqual(len(rows),527)
        self.assertEqual(len({r['ticker'] for r in rows}),527)

    def test_experimental_catalog_and_risk_caps(self):
        cards=btc_lab.catalog()['experimental_models']
        self.assertEqual({r['id'] for r in cards},set(b.MODELS))
        for model in b.MODELS:
            p=btc_lab.replay(dict(model=model,bankroll=100))['paper']
            self.assertLessEqual(p['max_bet_cost'],1)
            self.assertLessEqual(p['max_city_day_cost'],3)

    def test_corrupt_artifact_fails_before_inference(self):
        b.prepared.cache_clear()
        with patch.object(b,'digest',return_value='corrupted'):
            with self.assertRaisesRegex(ValueError,'hash mismatch'):b.prepared()
        b.prepared.cache_clear()
