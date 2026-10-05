from copy import deepcopy
import json
from pathlib import Path
import unittest
from helper.btc_inputs import validated_execution_rows


class BTCInputTests(unittest.TestCase):
    def setUp(self):
        self.row=json.loads((Path(__file__).resolve().parents[1]/'reports/btc15m-pilot-v1/train.jsonl').read_text().splitlines()[0])

    def test_real_input_and_absent_execution(self):
        self.assertEqual(len(validated_execution_rows([self.row])),1)
        self.row['execution_at']=None
        self.assertEqual(validated_execution_rows([self.row]),[])

    def test_future_features_are_rejected(self):
        self.row['execution_spot']['feature_available_at']=self.row['close_time']
        with self.assertRaisesRegex(ValueError,'Future'):validated_execution_rows([self.row])

    def test_stale_missing_or_malformed_features_are_rejected(self):
        for change in ({'feature_time':'2020-01-01T00:00:00Z'}, {'x':[0]*9}, {'diffusion':float('nan')}):
            row=deepcopy(self.row);row['execution_spot'].update(change)
            with self.assertRaises(ValueError):validated_execution_rows([row])
        self.row['execution_spot']=None
        with self.assertRaises(ValueError):validated_execution_rows([self.row])

    def test_duplicate_and_postclose_execution_rejected(self):
        with self.assertRaises(ValueError):validated_execution_rows([self.row,self.row])
        self.row['execution_at']=self.row['close_time']
        with self.assertRaisesRegex(ValueError,'chronology'):validated_execution_rows([self.row])
