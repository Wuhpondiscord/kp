import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.special import ndtr

from helper.btc_calibration import BTCAverage, average_z, read_development, run
from helper.btc_research import iso, metrics, unix


class BTCAverageTests(unittest.TestCase):
    def rows(self):
        rng = np.random.default_rng(17)
        rows = []
        for i in range(1200):
            z = rng.normal()
            rows.append(dict(x=[z], feature_time=iso(0), close_time=iso(360),
                y=int(rng.random() < ndtr(.45 + .65*z)), group=str(i//100)))
        return rows

    def test_average_variance_matches_brownian_covariance(self):
        times = 5 + (np.arange(100)+.5)/100
        variance = np.minimum.outer(times, times).mean()
        self.assertAlmostEqual(variance, 6-2/3, places=4)
        z = average_z([dict(x=[1], feature_time=iso(0), close_time=iso(360))])[0]
        self.assertAlmostEqual(z, np.sqrt(6/(6-2/3)))

    def test_rejects_partial_average_and_nonfinite_signal(self):
        for row in [dict(x=[1], feature_time=iso(330), close_time=iso(360)),
                    dict(x=[float('nan')], feature_time=iso(0), close_time=iso(360))]:
            with self.assertRaises(ValueError): average_z([row])

    def test_calibration_learns_and_predictions_ignore_labels(self):
        rows = self.rows()
        model = BTCAverage().fit(rows[:900])
        p = model.predict(rows[900:])
        self.assertLess(metrics(rows[900:], p)['log_loss'],
                        metrics(rows[900:], BTCAverage().predict(rows[900:]))['log_loss'])
        clone = BTCAverage(**json.loads(json.dumps(model.artifact())))
        np.testing.assert_array_equal(p, clone.predict(rows[900:]))
        np.testing.assert_array_equal(p, model.predict([dict(r, y=1-r['y']) for r in rows[900:]]))
        self.assertGreater(model.bias, 0)
        self.assertLess(model.log_slope, 0)

    def test_rejects_wrong_split_and_future_availability(self):
        with tempfile.TemporaryDirectory() as d:
            base = dict(ticker='A', at=iso(unix('2026-09-01')+100),
                close_time=iso(unix('2026-09-01')+400), settled_at=iso(unix('2026-09-01')+405),
                feature_available_at=iso(unix('2026-09-01')), quote_available_at=iso(unix('2026-09-01')))
            Path(d, 'validation.jsonl').write_text('')
            Path(d, 'train.jsonl').write_text(json.dumps(base)+'\n')
            self.assertEqual(len(read_development(d)[0]), 1)
            Path(d, 'train.jsonl').write_text(json.dumps(dict(base, feature_available_at=base['close_time']))+'\n')
            with self.assertRaisesRegex(ValueError, 'Future input'): read_development(d)
            Path(d, 'train.jsonl').write_text(json.dumps(dict(base, settled_at='2026-09-26T00:00:00Z'))+'\n')
            with self.assertRaisesRegex(ValueError, 'date boundary'): read_development(d)

    def test_development_loader_never_opens_test(self):
        opened = []
        def reading(path, *args, **kwargs):
            opened.append(path.name)
            if path.name == 'test.jsonl': raise AssertionError('Test set read')
            return ''
        with patch.object(Path, 'read_text', reading):
            self.assertEqual(read_development('unused'), [[], []])
        self.assertEqual(opened, ['train.jsonl', 'validation.jsonl'])
