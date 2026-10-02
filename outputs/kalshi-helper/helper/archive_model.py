"""Compact archive-only physical model; not interchangeable with live NWS inputs."""
import math
import numpy as np
from .event_weather import EventMaximum
from .weather_model import STATIONS

class ArchiveMaximum(EventMaximum):
    def __init__(self,observation_calibration=None,innovation=False):
        super().__init__(observation_calibration)
        self.innovation=innovation

    def inputs(self,rows):
        g=np.asarray([r['remaining_forecast_max'] for r in rows],dtype=float)
        x=[]
        for r in rows:
            if r['series'] not in STATIONS:raise ValueError('Unsupported station')
            if not r['forecast_complete'] or not math.isfinite(r['remaining_forecast_max']):raise ValueError('Incomplete archive forecast')
            v=[1,*[float(r['series']==s) for s in list(STATIONS)[1:]],
                r['model_disagreement_f']/5,(r['gfs_remaining_max']-r['ecmwf_remaining_max'])/5]
            if self.innovation:
                error=r.get('observation_innovation_f')
                if error is None or not math.isfinite(error):raise ValueError('Observed forecast error required for innovation model')
                v.append(error/5)
            if not all(math.isfinite(value) for value in v):raise ValueError('Nonfinite archive feature')
            x.append(v)
        return g,np.asarray(x,dtype=float)
