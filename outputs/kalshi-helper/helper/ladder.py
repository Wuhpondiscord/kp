"""Same-time Kalshi ladder covariates; no outcomes enter feature construction."""
from collections import defaultdict
import math
import numpy as np
from .hierarchy import HierarchicalOffset

def augment(rows,contract_bounds):
    groups=defaultdict(list)
    for r in rows:groups[(r['event'],r['at'])].append(r)
    output=[];complete=0
    for group in groups.values():
        if any(r['ticker'] not in contract_bounds for r in group):continue
        ordered=sorted(group,key=lambda r:contract_bounds[r['ticker']][0])
        intervals=[contract_bounds[r['ticker']] for r in ordered]
        if intervals[0][0]!=-math.inf or intervals[-1][1]!=math.inf or any(a[1]!=b[0] for a,b in zip(intervals,intervals[1:])):continue
        mass=sum(r['p'] for r in ordered)
        if mass<=0:continue
        qbounds=[]
        for q in (.25,.5,.75):
            acc=0
            for r,interval in zip(ordered,intervals):
                acc+=r['p']/mass
                if acc>=q:qbounds.append(interval);break
        if len(qbounds)!=3:continue
        median=sum(qbounds[1])/2 if all(math.isfinite(v) for v in qbounds[1]) else None
        spread=qbounds[2][1]-qbounds[0][0]
        for r in ordered:
            lo,hi=contract_bounds[r['ticker']]
            center=(lo+hi)/2 if math.isfinite(lo) and math.isfinite(hi) else lo if math.isfinite(lo) else hi
            output.append(dict(r,ladder_mass=mass,ladder_distance=(center-median)/10 if median is not None else 0,
                ladder_median_missing=float(median is None),ladder_iqr=spread/10 if math.isfinite(spread) else 0,
                ladder_iqr_missing=float(not math.isfinite(spread))))
        complete+=1
    return sorted(output,key=lambda r:(r['at'],r['ticker'])),dict(input_events=len({r['event'] for r in rows}),
        included_events=len({r['event'] for r in output}),complete_event_times=complete,excluded_rows=len(rows)-len(output))

class LadderOffset(HierarchicalOffset):
    def __init__(self,city_penalty=.1,use_ladder=False):
        super().__init__(city_penalty=city_penalty);self.use_ladder=use_ladder
    def inputs(self,rows):
        z,x=super().inputs(rows)
        if self.use_ladder:
            extra=np.array([[r['ladder_mass']-1,r['ladder_distance'],r['ladder_median_missing'],r['ladder_iqr'],r['ladder_iqr_missing']] for r in rows])
            x=np.column_stack([x,extra])
        return z,x
