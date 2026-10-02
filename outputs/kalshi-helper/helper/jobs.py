"""Serialized long-running research jobs, visible and persisted in the app."""
import threading
import uuid

from .core import utcnow
from .research_store import ResearchStore


class Jobs:
    def __init__(self, store, recover=True, state_key='job'):
        self.store=store;self.lock=threading.Lock();self.thread=None;self.cancel_event=threading.Event()
        self.state_key=state_key
        previous=ResearchStore(store).get(state_key)
        if recover and previous and previous.get('status')=='running':
            ResearchStore(store).set(state_key,dict(previous,status='interrupted',message='Previous process stopped'))
        training=ResearchStore(store).get('training')
        if recover and state_key=='job' and training and training.get('status') in ('running','preparing','evaluating'):
            ResearchStore(store).set('training',dict(training,status='interrupted',message='Training process stopped; the previous active model is retained'))

    def start(self, name, function):
        with self.lock:
            if self.thread and self.thread.is_alive():
                raise ValueError('A data or model job is already running')
            current=ResearchStore(self.store).get(self.state_key)
            if current and current.get('status')=='running':
                raise ValueError('A data or model job is already running in the collector process')
            job_id=uuid.uuid4().hex[:12]
            self.cancel_event.clear()
            rs=ResearchStore(self.store)
            rs.set(self.state_key,dict(id=job_id,name=name,status='running',message='Starting',done=0,total=1,at=utcnow()))
            def progress(message,done,total):
                rs.set(self.state_key,dict(id=job_id,name=name,status='running',message=message,done=done,total=total,at=utcnow()))
            def worker():
                try:
                    result=function(progress)
                    rs.set(self.state_key,dict(id=job_id,name=name,status='complete',message='Ready',done=1,total=1,at=utcnow()))
                except InterruptedError as exc:
                    rs.set(self.state_key,dict(id=job_id,name=name,status='cancelled',message=str(exc),done=0,total=1,at=utcnow()))
                except Exception as exc:
                    rs.set(self.state_key,dict(id=job_id,name=name,status='failed',message=str(exc),done=0,total=1,at=utcnow()))
            self.thread=threading.Thread(target=worker,daemon=True,name='research-job')
            self.thread.start()
            return job_id
