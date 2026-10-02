import json
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from urllib.request import Request,urlopen
from urllib.error import HTTPError,URLError


class HttpAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder=tempfile.TemporaryDirectory()
        with socket.socket() as s:s.bind(('127.0.0.1',0));cls.port=s.getsockname()[1]
        cls.base=f'http://127.0.0.1:{cls.port}'
        cls.process=subprocess.Popen([sys.executable,'-m','helper','--data',cls.folder.name,'serve','--port',str(cls.port)],
            cwd=Path(__file__).resolve().parents[1],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        for _ in range(100):
            try:
                with urlopen(cls.base+'/api/state',timeout=1):return
            except (URLError,TimeoutError):time.sleep(.1)
        cls.process.terminate();cls.process.wait();cls.folder.cleanup();raise RuntimeError('Test server failed to start')
    @classmethod
    def tearDownClass(cls):
        cls.process.terminate();cls.process.wait(timeout=10);cls.folder.cleanup()
    def request(self,path,body=None,origin=None,host=None):
        headers={'Origin':origin or self.base,'Content-Type':'application/json'}
        if host:headers['Host']=host
        request=Request(self.base+path,data=None if body is None else json.dumps(body).encode(),headers=headers)
        try:
            with urlopen(request,timeout=15) as response:return response.status,response.headers,response.read()
        except HTTPError as exc:return exc.code,exc.headers,exc.read()
    def test_static_assets_and_content_security(self):
        for path in ['/','/app.js','/training.js','/style.css']:
            status,headers,body=self.request(path)
            self.assertEqual(status,200);self.assertGreater(len(body),100)
            self.assertIn("default-src 'self'",headers['Content-Security-Policy'])
    def test_fresh_database_state(self):
        status,_,body=self.request('/api/state');self.assertEqual(status,200)
        self.assertIn('runs',json.loads(body))
    def test_origin_and_host_rejected(self):
        self.assertEqual(self.request('/api/stop',{},origin='https://example.com')[0],403)
        self.assertEqual(self.request('/api/state',host='example.com')[0],403)
    def test_non_object_json_rejected(self):
        for body in [[], 'invalid',42]:
            self.assertEqual(self.request('/api/train',body)[0],400)
    def test_bad_training_settings_rejected(self):
        self.assertEqual(self.request('/api/train',{'config':{'epochs':1}})[0],400)
    def test_arbitrary_market_url_rejected(self):
        self.assertEqual(self.request('/api/market-detail',{'reference':'https://example.com/markets/x'})[0],400)
    def test_unknown_routes_do_not_expose_files(self):
        for path in ['/api/no-such-route','/../data/research.sqlite3','/data/research.sqlite3']:
            self.assertEqual(self.request(path)[0],404)
    def test_synthetic_demo_report_end_to_end(self):
        status,_,body=self.request('/api/demo',{});self.assertEqual(status,200)
        run=json.loads(body)['run_id'];status,_,body=self.request('/api/report/'+run)
        report=json.loads(body);self.assertEqual(status,200);self.assertTrue(report['synthetic']);self.assertGreater(report['fills'],0)
    def test_import_rejects_reserved_name_and_bad_jsonl(self):
        for data in [dict(dataset='live-x',text='{}'),dict(dataset='bad',text='not json')]:
            self.assertEqual(self.request('/api/import',data)[0],400)

if __name__=='__main__':unittest.main()
