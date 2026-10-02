import hashlib
import tempfile
import unittest
from helper.storage import Store
from helper.research_store import ResearchStore
from helper.model_library import list_models,activate_model


class ModelLibraryTests(unittest.TestCase):
    def test_select_saved_model_preserves_gate_and_checks_artifact(self):
        with tempfile.TemporaryDirectory() as folder:
            store=Store(folder);rs=ResearchStore(store);(store.root/'models').mkdir()
            p=store.root/'models'/'test.pkl';p.write_bytes(b'locally recorded fixture')
            report=dict(id='test',model_id='test',model_file=p.name,model_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),promotion_passed=False,selected_model='neural_trained',experimental_model='neural_trained',selected_label='Test',training_last_settlement='2026-01-01')
            rs.save_experiment(report)
            self.assertEqual(len(list_models(store)),1)
            self.assertFalse(activate_model(store,'test')['passed'])
            previous=rs.get('active_model');p.write_bytes(b'changed')
            with self.assertRaises(ValueError):activate_model(store,'test')
            self.assertEqual(rs.get('active_model'),previous)
            with self.assertRaises(ValueError):activate_model(store,'missing')
