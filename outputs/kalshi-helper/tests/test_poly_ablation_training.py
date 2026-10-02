import unittest,tempfile,json
from unittest.mock import patch
from datetime import datetime,timedelta,timezone
from pathlib import Path
from helper.poly_ablation import compare

class AblationTrainingTests(unittest.TestCase):
    def test_both_arms_train_on_same_chronological_synthetic_fixture(self):
        rows=[]
        for d in range(100):
            day=datetime(2020,1,1,tzinfo=timezone.utc)+timedelta(days=d)
            for city in ('KXHIGHNY','KXHIGHCHI','KXHIGHMIA','KXHIGHDEN'):
                event=city+str(d)
                for side in (0,1):
                    rows.append(dict(event=event,ticker=event+str(side),group=day.date().isoformat(),at=day.isoformat(),
                        settled_at=(day+timedelta(days=1)).isoformat(),p=.5,y=int(side==d%2),
                        lower=-float('inf') if side==0 else 70,upper=70 if side==0 else float('inf'),
                        remaining_forecast_max=70.,remaining_hours=18.,forecast_complete=True,series=city,observed=False,
                        poly_available=True,poly_median_low=69.,poly_median_high=71.,poly_mean_spread=.02))
        with tempfile.TemporaryDirectory() as d:
            with patch('helper.poly_ablation.build_dataset',return_value=(rows,{'synthetic_fixture':True})):
                result=compare(d,d+'/result')
            self.assertTrue(result['performance_comparison_available']);self.assertFalse(result['promotion'])
            self.assertEqual(result['metrics']['with_polymarket']['samples'],result['metrics']['without_polymarket']['samples'])
            self.assertTrue((Path(d)/'result/selection.json').exists())
