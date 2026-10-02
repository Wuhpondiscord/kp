"""Best-effort metadata; diagnostics must never abort a training run."""
import sys
from importlib import metadata

def capture():
    packages={};errors={}
    for name in ('numpy','scipy','scikit-learn','threadpoolctl','tzdata'):
        try:packages[name]=metadata.version(name)
        except Exception as exc:
            packages[name]=None;errors[name]=type(exc).__name__+': '+str(exc)
    return dict(python=sys.version,packages=packages,metadata_errors=errors)
