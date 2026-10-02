import tempfile
import threading
import unittest
from unittest.mock import patch
from helper.storage import Store
from helper.research_store import ResearchStore
from helper.jobs import Jobs
from helper.evaluation import run_evaluation
from test_research import rows


class WorkflowAudit(unittest.TestCase):
    def test_market_checks_can_run_during_history_download(self):
        with tempfile.TemporaryDirectory() as folder:
            store=Store(folder);rs=ResearchStore(store);rs.set('job',dict(status='running',name='download'))
            job=Jobs(store,recover=False,state_key='analysis_job');job.start('market check',lambda p:None);job.thread.join(5)
            self.assertEqual(rs.get('job')['status'],'running')
            self.assertEqual(rs.get('analysis_job')['status'],'complete')
    def test_attached_ui_preserves_and_does_not_duplicate_collector_job(self):
        with tempfile.TemporaryDirectory() as folder:
            store=Store(folder);rs=ResearchStore(store);rs.set('job',dict(status='running',name='download'))
            jobs=Jobs(store,recover=False)
            self.assertEqual(rs.get('job')['status'],'running')
            with self.assertRaises(ValueError):jobs.start('another',lambda p:None)
    def test_full_model_comparison_caches_same_data_and_restores_active_model(self):
        with tempfile.TemporaryDirectory() as folder:
            store=Store(folder);rs=ResearchStore(store)
            with patch('helper.evaluation.sample_rows',return_value=rows()):
                report=run_evaluation(store)
                self.assertEqual(len(report['candidates']),6)
                self.assertTrue((store.root/'models'/report['model_file']).is_file())
                rs.set('active_model',{'model_id':'other'})
                with patch('helper.evaluation.Predictor.fit',side_effect=AssertionError('Cached call must not train')):
                    same=run_evaluation(store)
                self.assertEqual(same['id'],report['id'])
                self.assertEqual(rs.get('active_model')['model_id'],report['model_id'])
    def test_job_rejects_concurrent_work_and_persists_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            store=Store(folder);jobs=Jobs(store);ready=threading.Event();release=threading.Event()
            def work(progress):
                progress('Testing',1,2);ready.set();release.wait(5);raise RuntimeError('Deliberate fixture failure')
            jobs.start('fixture',work);self.assertTrue(ready.wait(3))
            try:
                with self.assertRaises(ValueError):jobs.start('second',lambda p:None)
            finally:release.set();jobs.thread.join(5)
            result=ResearchStore(store).get('job')
            self.assertEqual(result['status'],'failed');self.assertIn('Deliberate',result['message'])
    def test_restart_marks_training_and_job_interrupted(self):
        with tempfile.TemporaryDirectory() as folder:
            store=Store(folder);rs=ResearchStore(store)
            rs.set('training',dict(status='running'));rs.set('job',dict(status='running'))
            Jobs(store)
            self.assertEqual(rs.get('training')['status'],'interrupted')
            self.assertEqual(rs.get('job')['status'],'interrupted')
    def test_cancelled_job_has_distinct_status(self):
        with tempfile.TemporaryDirectory() as folder:
            store=Store(folder);jobs=Jobs(store)
            def cancel(progress):raise InterruptedError('cancelled')
            jobs.start('cancel fixture',cancel);jobs.thread.join(5)
            self.assertEqual(ResearchStore(store).get('job')['status'],'cancelled')

if __name__=='__main__':unittest.main()
