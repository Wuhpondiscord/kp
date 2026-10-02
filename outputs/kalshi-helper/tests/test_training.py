import tempfile
import threading
import unittest
from unittest.mock import patch
from copy import deepcopy
import numpy as np
from helper.training import settings,split_dataset,fit_epochs,train_model
from helper.evaluation import features
from helper.storage import Store
from helper.research_store import ResearchStore
from test_research import rows


class TrainingTests(unittest.TestCase):
    def test_whole_date_partitions_and_embargo(self):
        train,val,test=split_dataset(rows())
        groups=[{r['group'] for r in x} for x in (train,val,test)]
        for a,b in [(0,1),(0,2),(1,2)]:self.assertFalse(groups[a]&groups[b])
        from helper.core import stamp
        from datetime import timedelta
        for a,b in [(train,val),(val,test)]:
            self.assertLess(max(stamp(r['settled_at']) for r in a),min(stamp(r['at']) for r in b)-timedelta(hours=48))

    def test_scaler_fits_only_train_and_weights_really_update(self):
        train,val,_=split_dataset(rows())
        traces=[]
        def collect(row,history,epoch,best):traces.append(deepcopy(best.coefs_))
        model,history,best=fit_epochs(train,val,{'epochs':10,'patience':10},collect)
        np.testing.assert_allclose(model.scaler.mean_,features(train).mean(axis=0))
        self.assertEqual(len(history),10)
        self.assertEqual(best,min(history,key=lambda h:h['validation_loss'])['epoch'])
        self.assertTrue(any(not np.array_equal(a,b) for a,b in zip(traces[0],traces[-1])))

    def test_test_labels_cannot_select_checkpoint(self):
        source=rows();train,val,test=split_dataset(source)
        changed=deepcopy(source)
        for r in changed:
            if r['group']>=test[0]['group']:r['y']=1-r['y']
        train2,val2,_=split_dataset(changed)
        self.assertEqual(train,train2);self.assertEqual(val,val2)
        a,ha,ea=fit_epochs(train,val,{'epochs':5})
        b,hb,eb=fit_epochs(train2,val2,{'epochs':5})
        self.assertEqual(ha,hb);self.assertEqual(ea,eb)
        np.testing.assert_array_equal(a.predict(test),b.predict(test))

    def test_config_bounds_and_unknown_keys(self):
        for config in [dict(epochs=float('nan')),dict(epochs=4),dict(batch_size=1),dict(learning_rate=1),dict(alpha=-1),dict(architecture='9999'),dict(hidden_test=True)]:
            with self.subTest(config=config),self.assertRaises(ValueError):settings(config)

    def test_cancel_preserves_active_model(self):
        with tempfile.TemporaryDirectory() as folder:
            store=Store(folder);rs=ResearchStore(store);rs.set('active_model',{'model_id':'previous'})
            cancel=threading.Event();cancel.set()
            with patch('helper.training.sample_rows',return_value=rows()),self.assertRaises(InterruptedError):
                train_model(store,{'epochs':5},cancel=cancel)
            self.assertEqual(rs.get('active_model'),{'model_id':'previous'})
            self.assertEqual(rs.get('training')['status'],'cancelled')

    def test_cancel_during_evaluation_does_not_publish_model(self):
        with tempfile.TemporaryDirectory() as folder:
            store=Store(folder);rs=ResearchStore(store);rs.set('active_model',{'model_id':'previous'})
            cancel=threading.Event()
            def progress(message,*args):
                if message.startswith('Evaluating checkpoint'):cancel.set()
            with patch('helper.training.sample_rows',return_value=rows()),self.assertRaises(InterruptedError):
                train_model(store,{'epochs':5},progress,cancel)
            self.assertEqual(rs.get('active_model'),{'model_id':'previous'})
            self.assertEqual(rs.get('training')['status'],'cancelled')

    def test_saved_checkpoint_reload_and_reused_holdout_gate(self):
        with tempfile.TemporaryDirectory() as folder:
            store=Store(folder);rs=ResearchStore(store)
            rs.set('holdout_lock',dict(last_test_day='2099-01-01',first_experiment='prior'))
            with patch('helper.training.sample_rows',return_value=rows()):r=train_model(store,{'epochs':5})
            self.assertTrue(r['holdout_reused']);self.assertFalse(r['promotion_passed'])
            self.assertEqual(rs.get('training')['status'],'complete')
            self.assertTrue((store.root/'training'/r['id']/'results.csv').is_file())
            self.assertTrue((store.root/'training'/r['id']/'split-manifest.json').is_file())
            from helper.evaluation import ModelService
            p,note=ModelService(store).predict('KXHIGHNY',.3,.5,'2026-01-01T00:00:00Z','2026-01-02T00:00:00Z',True)
            self.assertTrue(0<=p<=1);self.assertIn('validation checkpoint',note)
            pinned=rs.get('active_model');rs.set('active_model',{'model_id':'a different model'})
            p2,note2=ModelService(store,active_model=pinned).predict('KXHIGHNY',.3,.5,'2026-01-01T00:00:00Z','2026-01-02T00:00:00Z',True)
            self.assertEqual((p,note),(p2,note2))

if __name__=='__main__':unittest.main()
