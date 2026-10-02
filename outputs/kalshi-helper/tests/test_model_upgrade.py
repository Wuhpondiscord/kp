from datetime import timedelta
import math
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from helper.weather_model import bounds,forecast_feature,event_date,probability,blend,fit_distribution
from helper.live import practice_candidates
from helper.training import choose_weight,train_model
from helper.storage import Store
from helper.research_store import ResearchStore

class ModelUpgrade(unittest.TestCase):
    def test_temperature_partition_conserves_probability(self):
        specs=[dict(strike_type='less',cap_strike=70),dict(strike_type='between',floor_strike=70,cap_strike=71),dict(strike_type='greater',floor_strike=71)]
        rows=[dict(series='KXHIGHNY',weather_day='2026-09-01',forecast_high=70,lower=bounds(m)[0],upper=bounds(m)[1]) for m in specs]
        p=probability(rows,[0]*7+[math.log(3)])
        self.assertAlmostEqual(float(p.sum()),1,places=6)
    def test_forecast_leakage_and_missing_hours_rejected(self):
        begin=event_date('KXHIGHNY-26SEP01-B70.5')+timedelta(hours=5)
        hours={(begin+timedelta(hours=i)).strftime('%Y-%m-%dT%H:%M'):70+i/24 for i in range(24)}
        f=forecast_feature('KXHIGHNY','KXHIGHNY-26SEP01-B70.5',hours,'2026-09-01T00:00:00Z')
        self.assertGreater(f['forecast_high'],70)
        with self.assertRaisesRegex(ValueError,'not available'):forecast_feature('KXHIGHNY','KXHIGHNY-26SEP01-B70.5',hours,'2026-08-30T00:00:00Z')
        hours.pop(next(iter(hours)))
        with self.assertRaisesRegex(ValueError,'Incomplete'):forecast_feature('KXHIGHNY','KXHIGHNY-26SEP01-B70.5',hours,'2026-09-01T00:00:00Z')
    def test_weather_fitter_deduplicates_event_horizons(self):
        rows=[dict(event=str(i),series='KXHIGHNY',weather_day='2026-09-01',forecast_high=70+i%5,lower=69+i%5,upper=72+i%5,y=1) for i in range(90)]
        a,n=fit_distribution(rows);b,m=fit_distribution(rows*3)
        self.assertEqual(n,90);self.assertEqual(m,90);np.testing.assert_allclose(a,b)
    def test_zero_weather_blend_is_exact_market(self):
        np.testing.assert_array_equal(blend([.99,.01],[.2,.8],0),[.2,.8])
    def test_failed_preparation_is_visible(self):
        with tempfile.TemporaryDirectory() as folder:
            store=Store(folder)
            with patch('helper.training.sample_rows',side_effect=ValueError('bad archive')):
                with self.assertRaises(ValueError):train_model(store)
            self.assertEqual(ResearchStore(store).get('training')['status'],'failed')
    def test_validation_can_reject_harmful_neural_blend(self):
        class BadModel:
            weight=.5
            def predict(self,rows):return np.array([self.weight*.9+(1-self.weight)*r['p'] for r in rows])
        rows=[dict(group=str(i),p=.01,y=0,ticker=str(i),event=str(i)) for i in range(10)]
        model=BadModel();choose_weight(model,rows);self.assertEqual(model.weight,0)
    def test_practice_prefers_supported_liquid_next_day(self):
        markets=[dict(ticker='empty',series='KXHIGHNY',close_time='2026-09-01T02:00:00Z',bid=0,ask=1,spread=1),dict(ticker='usable',series='KXHIGHNY',close_time='2026-09-02T00:00:00Z',bid=.3,ask=.35,spread=.05)]
        self.assertEqual(practice_candidates(markets,'2026-09-01T00:00:00Z')[0]['ticker'],'usable')
    def test_practice_excludes_expired_contracts(self):
        self.assertEqual(practice_candidates([dict(close_time='2026-09-01T00:00:00Z')],'2026-09-02T00:00:00Z'),[])
