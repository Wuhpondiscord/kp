import tempfile
import unittest
from http.server import BaseHTTPRequestHandler
from helper.server import LocalServer,serve
from helper.storage import Store
from helper.research_store import ResearchStore


class ExclusiveServerTests(unittest.TestCase):
    def test_duplicate_launch_cannot_bind_or_change_running_job(self):
        with tempfile.TemporaryDirectory() as folder:
            store=Store(folder);rs=ResearchStore(store)
            original=dict(status='running',name='Train neural model',id='existing')
            rs.set('job',original)
            with LocalServer(('127.0.0.1',0),BaseHTTPRequestHandler) as first:
                with self.assertRaises(OSError):serve(store,port=first.server_port)
                self.assertEqual(rs.get('job'),original)
