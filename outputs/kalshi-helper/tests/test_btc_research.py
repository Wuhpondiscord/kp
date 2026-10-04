import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from helper.btc_research import (BTCModels, Cache, digest, execution_rows,
    iso, metrics, paired_interval, pnl_interval, supported, spot_features, split_rows, unix)


class BTCResearchTests(unittest.TestCase):
    def test_settlement_average_is_not_a_touch_and_ties_are_yes(self):
        market = dict(ticker='KXBTC15M-TEST', strike_type='greater_or_equal',
            rules_primary="If the simple average of the sixty seconds of CF Benchmarks' BRTI before CLOSE is at least the simple average of the sixty seconds before OPEN, then Yes.",
            result='yes', status='settled', notional_value_dollars='1',
            floor_strike=100, expiration_value='100.00', open_time=iso(0),
            close_time=iso(900), settlement_ts=iso(905), settlement_value_dollars='1')
        self.assertTrue(supported(market))
        self.assertTrue(supported(dict(market, expiration_value='')))
        self.assertFalse(supported(dict(market, expiration_value='NaN')))
        literal="If the simple average of the sixty seconds of CF Benchmarks' BRTI before CLOSE is at least 100, then the market resolves to Yes."
        self.assertTrue(supported(dict(market, rules_primary=literal)))
        self.assertFalse(supported(dict(market, rules_primary=literal.replace('least 100', 'least 101'))))
        self.assertTrue(supported(dict(market, expiration_value='1,000.00')))
        self.assertFalse(supported(dict(market, expiration_value='10,00.00')))
        self.assertFalse(supported(dict(market, expiration_value='99.99')))
        self.assertFalse(supported(dict(market, rules_primary="If CF Benchmarks' BRTI touches at least 100 during sixty seconds")))

    def test_profit_interval_includes_no_trade_days(self):
        result = dict(ledger=[dict(day='one', pnl=1, quantity=2, cost=1)])
        interval = pnl_interval(result, ['one', 'two'])
        self.assertEqual(interval['calendar_days'], 2)
        self.assertEqual(interval['trade_days'], 1)
        self.assertGreater(interval['zero_trade_resamples'], 0)
        self.assertEqual(interval['net_per_contract_95_interval'], [.5, .5])
        self.assertEqual(interval['return_on_deployed_95_interval'], [1, 1])

    def bars(self):
        return {t: [t, 99, 102, 100, 100 + (t // 60 % 7) / 100, 10]
                for t in range(0, 180 * 60, 60)}

    def test_bar_start_timestamp_and_publication_delay(self):
        bars = self.bars()
        r = spot_features(bars, 150 * 60, 155 * 60, 100)
        self.assertEqual(r['feature_time'], iso(149 * 60))
        self.assertEqual(r['feature_available_at'], iso(149 * 60 + 5))
        # Anything not yet available, including the current minute, is irrelevant.
        changed = dict(bars)
        for t in range(149 * 60, 180 * 60, 60):
            changed[t] = [t, 1, 10000, 5000, 5000, 100000]
        self.assertEqual(r, spot_features(changed, 150 * 60, 155 * 60, 100))
        self.assertEqual(spot_features(bars, 150 * 60 + 5, 155 * 60, 100)
                         ['feature_time'], iso(150 * 60))

    def test_missing_or_invalid_spot_data_fails_closed(self):
        for change in ('missing', 'nan', 'range'):
            bars = self.bars()
            if change == 'missing': del bars[100 * 60]
            elif change == 'nan': bars[100 * 60][4] = float('nan')
            else: bars[100 * 60][4] = 103
            with self.assertRaises(ValueError):
                spot_features(bars, 150 * 60, 155 * 60, 100)

    def data(self):
        rng = np.random.default_rng(13)
        rows = []
        for i in range(200):
            x = rng.normal(size=10)
            rows.append(dict(x=x.tolist(), p=.5, spread=.02,
                             diffusion=.5, y=int(x[0] > 0), group=str(i // 20)))
        return rows

    def test_learning_serialization_and_label_independent_inference(self):
        rows = self.data()
        model = BTCModels().fit(rows[:150])
        restored = BTCModels()
        restored.models = json.loads(json.dumps(model.models))
        for name in model.models:
            p = model.predict(rows[150:], name)
            np.testing.assert_array_equal(p, restored.predict(rows[150:], name))
            np.testing.assert_array_equal(p, model.predict(
                [dict(r, y=1-r['y']) for r in rows[150:]], name))
            if name != 'price_only_control':
                self.assertLess(metrics(rows[150:], p)['log_loss'], .69)
        # Saved preprocessing uses training only.
        np.testing.assert_allclose(model.models['BTCSignal']['mean'],
                                   np.mean([r['x'] for r in rows[:150]], axis=0))

    def test_recomputed_execution_updates_all_input_provenance(self):
        original = dict(at=iso(9000), p=.4, bid=.39, ask=.41, spread=.02,
                        quote_at=iso(8940), quote_available_at=iso(8945),
                        execution_at=iso(9005), execution_bid=.59, execution_ask=.61,
                        x=[0]*10, execution_spot=dict(x=[1]*10, spot=101,
                        feature_time=iso(9000), feature_available_at=iso(9005)))
        r = execution_rows([original])[0]
        self.assertEqual(r['x'], [1]*10)
        self.assertEqual(r['p'], .6)
        self.assertEqual(r['at'], iso(9005))
        self.assertEqual(r['quote_at'], iso(9000))
        self.assertEqual(r['quote_available_at'], iso(9005))
        self.assertEqual(original['p'], .4)

    def test_chronological_embargo_uses_settlement(self):
        rows = []
        for day in range(30):
            for hour in range(24):
                t = unix('2026-09-01') + day*86400 + hour*3600
                rows.append(dict(at=iso(t), settled_at=iso(t+600)))
        train, val, test = split_rows(rows)
        self.assertLess(max(r['settled_at'] for r in train), iso(unix('2026-09-19')))
        self.assertGreaterEqual(min(r['at'] for r in val), iso(unix('2026-09-19')+7200))
        self.assertLess(max(r['settled_at'] for r in val), iso(unix('2026-09-25')))
        self.assertGreaterEqual(min(r['at'] for r in test), iso(unix('2026-09-25')+7200))

    def test_market_comparison_is_exactly_zero_and_clusters_days(self):
        rows = self.data()
        result = paired_interval(rows, [r['p'] for r in rows])
        self.assertEqual(result['day_cluster_95_interval'], [0, 0])
        self.assertEqual(result['days'], 10)

    def test_offline_cache_checks_integrity_and_never_fetches_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            url = 'https://example.test/public'; key = digest(url.encode())
            raw = b'{"ok": true}'
            root = Path(directory)
            (root/(key+'.raw')).write_bytes(raw)
            (root/(key+'.json')).write_text(json.dumps(dict(url=url, sha256=digest(raw))))
            cache = Cache(root, offline=True)
            self.assertEqual(cache.get(url), {'ok': True})
            (root/(key+'.raw')).write_bytes(b'{}')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'): cache.get(url)
            with self.assertRaisesRegex(ValueError, 'Offline cache missing'): cache.get(url+'/missing')
