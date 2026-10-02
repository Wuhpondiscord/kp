"""Exercise HTTP host/origin checks against an actual server process."""
import json,os,socket,subprocess,sys,tempfile,time,unittest
from urllib.request import Request,urlopen
from urllib.error import HTTPError,URLError
from pathlib import Path

class HostedTests(unittest.TestCase):
    def test_hosted_security_and_health(self):
        with tempfile.TemporaryDirectory() as folder:
            with socket.socket() as s:
                s.bind(('127.0.0.1',0));port=s.getsockname()[1]
            env=dict(os.environ,BETCHECK_PUBLIC_ORIGIN='https://wuhp-kp.hf.space',BETCHECK_REVISION='test-revision')
            process=subprocess.Popen([sys.executable,'-m','helper','--data',folder,'serve','--port',str(port)],env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,cwd=Path(__file__).resolve().parents[1])
            try:
                base=f'http://127.0.0.1:{port}'
                for _ in range(100):
                    try:
                        with urlopen(base+'/healthz',timeout=2) as response:health=json.load(response)
                        break
                    except URLError:time.sleep(.1)
                else:self.fail('Server did not start')
                self.assertEqual(health['revision'],'test-revision')
                with urlopen(Request(base+'/',headers={'Host':'wuhp-kp.hf.space'})) as response:
                    self.assertIn(b'Shared public paper-trading demo',response.read())
                    self.assertIn('https://huggingface.co',response.headers['Content-Security-Policy'])
                for host,origin,payload,status in [('evil.test','https://wuhp-kp.hf.space',{},403),('wuhp-kp.hf.space','https://evil.test',{},403),('wuhp-kp.hf.space',None,{},403),('wuhp-kp.hf.space','https://wuhp-kp.hf.space',{'predictions_file':'/etc/passwd'},400)]:
                    headers={'Host':host,'Content-Type':'application/json'}
                    if origin:headers['Origin']=origin
                    with self.assertRaises(HTTPError) as err:urlopen(Request(base+'/api/live',data=json.dumps(payload).encode(),headers=headers))
                    self.assertEqual(err.exception.code,status)
            finally:process.terminate();process.wait(timeout=15)
