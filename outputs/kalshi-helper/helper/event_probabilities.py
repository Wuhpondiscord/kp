"""Explicit categorical projection for complete, simultaneous bracket partitions.

This is a versioned inference transformation, not the old binary predictor or a
newly validated forecasting model. Never normalize a price-filtered subset.
"""
import math
import numpy as np
from .core import stamp

VERSION='complete-event-projection-v1'

def validate_event(rows):
    if len(rows)<2:raise ValueError('A complete event requires all brackets, including both tails')
    first=rows[0]
    identity=(first['event'],first['series'],stamp(first['at']))
    tickers=set()
    for row in rows:
        if (row['event'],row['series'],stamp(row['at']))!=identity:
            raise ValueError('Brackets must share an event, station and decision timestamp')
        if row['ticker'] in tickers:raise ValueError('Duplicate bracket ticker')
        tickers.add(row['ticker'])
        lo,hi=row['lower'],row['upper']
        if math.isnan(lo) or math.isnan(hi) or not lo<hi:raise ValueError('Invalid bracket bounds')
    ordered=sorted(rows,key=lambda r:r['lower'])
    if ordered[0]['lower']!=-math.inf or ordered[-1]['upper']!=math.inf:
        raise ValueError('Complete event must contain both unbounded tails')
    if any(a['upper']!=b['lower'] for a,b in zip(ordered,ordered[1:])):
        raise ValueError('Bracket partition has a gap or overlap')

def project_event(rows,probabilities):
    validate_event(rows)
    p=np.asarray(probabilities,dtype=float)
    if p.shape!=(len(rows),) or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise ValueError('Invalid bracket probabilities')
    mass=float(p.sum())
    if mass<=0:raise ValueError('Cannot normalize zero event mass')
    return p/mass
