import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from helper import btc_lab


class BTCLabTests(unittest.TestCase):
    def test_published_replays_match_frozen_results(self):
        for model, expected in [('volatility', -122.52), ('logistic', -319.83), ('regime', -222.99)]:
            result = btc_lab.replay(dict(model=model, bankroll=1000))
            self.assertAlmostEqual(result['paper']['pnl'], expected, places=2)
            self.assertGreater(result['scores']['brier'], result['market_scores']['brier'])
            self.assertEqual(result['mode'], 'historical_paper_only')

    def test_catalog_does_not_promote_losing_models(self):
        data = btc_lab.catalog()
        self.assertFalse(data['live_enabled'])
        self.assertFalse(data['automatic_promotion'])
        self.assertEqual(len(data['models']), 3)
        self.assertTrue(all(m['summary']['net_pnl'] < 0 for m in data['models']))

    def test_invalid_parameters_rejected_before_training(self):
        with patch.object(btc_lab, 'prepared', side_effect=AssertionError('Must not train')):
            for params in [dict(model='weather'), dict(bankroll=float('nan')),
                           dict(bankroll=1001), dict(bankroll=0), dict(slippage=.01)]:
                with self.assertRaises(ValueError):
                    btc_lab.replay(params)

    def test_hash_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, 'rows').write_text('changed')
            with self.assertRaises(ValueError):
                btc_lab.read_verified(Path(folder), 'rows', 'invalid')

    def test_busy_lab_rejects_duplicate_work(self):
        with btc_lab.LOCK:
            with self.assertRaisesRegex(RuntimeError, 'Another BTC replay'):
                btc_lab.replay({})

    def test_bankroll_is_resimulated_not_scaled(self):
        result = btc_lab.replay(dict(model='volatility', bankroll=100))['paper']
        self.assertEqual(result['starting_bankroll'], 100)
        self.assertLessEqual(result['max_bet_cost'], 1)
        self.assertLessEqual(result['max_city_day_cost'], 3)
        self.assertNotAlmostEqual(result['pnl'], -12.252, places=2)
