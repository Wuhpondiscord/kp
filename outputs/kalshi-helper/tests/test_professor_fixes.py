import unittest,tempfile,json,math
from unittest.mock import patch
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from helper.environment import capture
from helper.cohorts import price_primary
from helper.training import train_model
from helper.storage import Store
from helper.ladder import augment
from test_research import rows

class ProfessorFixTests(unittest.TestCase):
    def test_metadata_failure_does_not_abort_training(self):
        with tempfile.TemporaryDirectory() as d,patch('helper.environment.metadata.version',side_effect=PackageNotFoundError('tzdata')),patch('helper.training.sample_rows',return_value=rows()):
            report=train_model(Store(d),dict(epochs=5,method='residual'))
            env=json.loads((Path(d)/'training'/report['id']/'environment.json').read_text())
            self.assertIsNone(env['packages']['tzdata']);self.assertIn('tzdata',env['metadata_errors'])
    def test_nominal_24h_included_despite_candle_age(self):
        self.assertTrue(price_primary(dict(horizon=24,hours_left=24.983,p=.5)))
        self.assertFalse(price_primary(dict(horizon=26,hours_left=26.983,p=.5)))
    def test_ladder_ignores_labels_and_does_not_join_future_quote(self):
        rs=[dict(ticker='a',event='e',at='2026-01-01T00:00:00Z',p=.2,y=0),dict(ticker='b',event='e',at='2026-01-01T00:00:00Z',p=.8,y=1)]
        b={'a':(-math.inf,70),'b':(70,math.inf)}
        first,_=augment(rs,b);second,_=augment([dict(r,y=1-r['y']) for r in rs],b)
        self.assertEqual([{k:v for k,v in r.items() if k!='y'} for r in first],[{k:v for k,v in r.items() if k!='y'} for r in second])
        rs[1]['at']='2026-01-01T01:00:00Z';self.assertEqual(augment(rs,b)[0],[])
