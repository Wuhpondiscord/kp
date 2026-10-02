"""External full-day reporting-discrepancy calibration, not pure sensor noise.

Do not fit this alongside remaining-day spread. Late pre-close residuals also
contain remaining warming and are diagnostics only. Use exact expiration values,
one record per station/event, and development dates only.
"""
import math
import numpy as np
from .core import stamp

METHOD='completed_station_day_reporting_discrepancy_v1'

def load_training_calibration(path=None,legacy=False):
    """Default to the frozen pre-training calibration; never silently use 1F."""
    import json
    from pathlib import Path
    if path and legacy:raise ValueError('Choose a calibration file OR the explicit legacy assumption')
    if legacy:return None
    source=Path(path) if path else Path(__file__).resolve().parents[1]/'reports/historical-forecast-replay/observation-calibration.json'
    if not source.is_file():raise ValueError('Observation calibration missing: supply --observation-calibration or explicitly opt into --legacy-observation-assumption')
    calibration=json.loads(source.read_text(encoding='utf-8'))
    validate_calibration(calibration)
    return calibration

def validate_calibration(c):
    if c.get('method')!=METHOD:raise ValueError('Unsupported observation calibration')
    if not math.isfinite(c['bias_f']) or not math.isfinite(c['sigma_f']) or c['sigma_f']<=0:
        raise ValueError('Invalid observation location/scale')
    if c.get('n_events',0)<80 or c['n_events']!=len(set(c.get('events',[]))):
        raise ValueError('Need at least 80 unique exact-settlement event pairs')
    if not c.get('source_hashes') or not c.get('latest_settled_at'):
        raise ValueError('Calibration requires source hashes and settlement cutoff')
    stamp(c['latest_settled_at'])
    if c.get('pooling','pooled') not in ('pooled','station'):raise ValueError('Invalid pooling mode')
    if c.get('pooling')=='station':
        if not c.get('by_station'):raise ValueError('Station calibration requires station estimates')
        for s in c['by_station'].values():
            if s['n']>=80 and (not math.isfinite(s['bias_f']) or not math.isfinite(s['sigma_f']) or s['sigma_f']<=0):
                raise ValueError('Invalid station calibration')

def validate_training_scope(c,rows):
    validate_calibration(c)
    if not rows:raise ValueError('Empty training data')
    if set(c['events'])<={r['event'] for r in rows}:return
    if stamp(c['latest_settled_at'])<min(stamp(r['at']) for r in rows):return
    raise ValueError('Observation calibration must use training events or previously settled history only')

def summarize(values):
    a=np.asarray(values,dtype=float)
    if not len(a):return dict(n=0)
    return dict(n=len(a),bias_f=float(a.mean()),sigma_f=float(a.std(ddof=1)) if len(a)>1 else None,
        rmse_f=float(np.sqrt(np.mean(a*a))),quantiles_f=dict(zip(['p05','p50','p95'],map(float,np.quantile(a,[.05,.5,.95])))))

def calibrate(pairs,source_hashes):
    """Pairs must be complete-day, exact settlement records, not bracket proxies."""
    seen=set()
    for r in pairs:
        if r['event'] in seen:raise ValueError('Repeated event would overweight a station-day')
        seen.add(r['event'])
        if r.get('target_source')!='kalshi_expiration_value' or not r.get('complete_day'):
            raise ValueError('Need exact expiration value and complete-day observations')
        if not all(math.isfinite(r[k]) for k in ('settled_max','observed_max')):
            raise ValueError('Nonfinite calibration temperature')
    if len(pairs)<80:raise ValueError('Need 80 complete station-day pairs')
    summary=summarize([r['settled_max']-r['observed_max'] for r in pairs])
    result=dict(method=METHOD,bias_f=summary['bias_f'],sigma_f=summary['sigma_f'],
        n_events=len(pairs),events=sorted(seen),source_hashes=source_hashes,
        latest_settled_at=max((r['settled_at'] for r in pairs),key=stamp),
        by_station={s:summarize([r['settled_max']-r['observed_max'] for r in pairs if r['station']==s]) for s in sorted({r['station'] for r in pairs})},
        interpretation='Pooled Gaussian approximation to completed-day sampling/reporting discrepancy. Transfer to running maxima is an unvalidated modeling assumption; not pure sensor error.',
        selection='Development/training events only; no validation/test optimization')
    validate_calibration(result)
    return result

def block_uncertainty(pairs,repetitions=2000,seed=1729):
    """Station-stratified month cluster bootstrap; plus synchronized-month check.

    Resample whole observed station-months, never individual rows. Synchronized
    months preserve cross-station dependence as a separate sensitivity estimate.
    Neither fixes sparse seasons or dependence across calendar-month boundaries.
    """
    from collections import defaultdict
    clusters=defaultdict(dict)
    for r in pairs:
        month=r['event'].split('-')[1][:5]
        clusters[r['station']].setdefault(month,[]).append(r['settled_max']-r['observed_max'])
    rng=np.random.default_rng(seed);stations=sorted(clusters)
    def interval(draws):
        a=np.asarray(draws)
        return dict(bias_interval_f=np.quantile(a[:,0],[.025,.975]).tolist(),
            sigma_interval_f=np.quantile(a[:,1],[.025,.975]).tolist(),
            sigma_bootstrap_se_f=float(a[:,1].std(ddof=1)))
    def stats(a):return [float(np.mean(a)),float(np.std(a,ddof=1))]
    pooled=[];individual={s:[] for s in stations};synchronized=[]
    months=sorted({m for c in clusters.values() for m in c})
    for _ in range(repetitions):
        all_values=[]
        for s in stations:
            blocks=[clusters[s][m] for m in sorted(clusters[s])]
            v=[x for i in rng.integers(0,len(blocks),len(blocks)) for x in blocks[i]]
            individual[s].append(stats(v));all_values.extend(v)
        pooled.append(stats(all_values))
        v=[x for i in rng.integers(0,len(months),len(months)) for s in stations for x in clusters[s].get(months[i],[])]
        synchronized.append(stats(v))
    return dict(method='station-stratified month-cluster percentile bootstrap',repetitions=repetitions,seed=seed,
        pooled=interval(pooled),by_station={s:dict(month_blocks=len(clusters[s]),**interval(individual[s])) for s in stations},
        station_month_blocks=sum(len(c) for c in clusters.values()),calendar_months=len(months),
        synchronized_month_sensitivity=interval(synchronized),
        warning='Few observed months; incomplete seasons and between-month persistence remain. Intervals are not evidence of stable future station-specific noise.')
