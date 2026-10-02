"""One fixed variance-link challenger, evaluated on validation only, never test."""
import json,math,pickle,hashlib
from pathlib import Path
from collections import Counter
import numpy as np
from .weather_research import RemainingMaximum,build_dataset,partitions
from .evaluation import score,paired_block_interval

def calibration_metrics(rows,probabilities):
    result=score(rows,probabilities)
    # Equal date weight, then equal event weight, then equal row weight.
    events=Counter((r['group'],r['event']) for r in rows)
    date_events=Counter(d for d,e in events)
    w=np.asarray([1/(len(date_events)*date_events[r['group']]*events[(r['group'],r['event'])]) for r in rows])
    p=np.asarray(probabilities);y=np.asarray([r['y'] for r in rows]);bins=[]
    for b in range(10):
        mask=np.minimum((p*10).astype(int),9)==b
        if mask.any():
            weight=float(w[mask].sum())
            bins.append(dict(low=b/10,weight=weight,rows=int(mask.sum()),
                predicted=float(w[mask]@p[mask]/weight),observed=float(w[mask]@y[mask]/weight)))
    result['event_date_weighted_reliability']=bins
    result['event_date_weighted_ece']=sum(b['weight']*abs(b['predicted']-b['observed']) for b in bins)
    result['event_date_weighted_brier']=float(w@((p-y)**2))
    result['event_date_weighted_log_loss']=float(-w@(y*np.log(np.clip(p,1e-12,1))+(1-y)*np.log(np.clip(1-p,1e-12,1))))
    result['calibration_note']='Fixed 10 probability bins. ECE is descriptive, bin-dependent and noisy; Brier also measures resolution.'
    return result

def compare(root,output,observation_calibration=None):
    folder=Path(output);folder.mkdir(parents=True,exist_ok=False)
    protocol=dict(arms=['constant','disagreement'],test_evaluated=False,promotion=False,
        formula='sigma = exp(log_base_sigma + beta * min(disagreement_f/5, 4))',
        bounds=dict(base_sigma_f=[.5,20],beta=[0,1]),mean_features='Identical in both arms; disagreement mean term retained',
        missing_disagreement='Exclude from both arms; never impute agreement',
        selection='No automatic winner or tuning; one paired validation comparison',
        split='Existing full eligible cohort chronological 60/20/20 with 48h embargo; split BEFORE disagreement filter',
        observation=RemainingMaximum(observation_calibration).observation_protocol(),
        warning='Two deterministic model maxima are a spread proxy, not a calibrated ensemble. Prior research inspection still creates selection risk.')
    (folder/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    rows,audit=build_dataset(root)
    result=dict(audit=audit,eligible_rows=len(rows),dates=len({r['group'] for r in rows}),test_evaluated=False,promotion=False)
    try:
        training,validation,_=partitions(rows)
        valid=lambda r:r.get('model_disagreement_f') is not None and math.isfinite(r['model_disagreement_f']) and r['model_disagreement_f']>=0
        training=[r for r in training if valid(r)]
        validation=[r for r in validation if valid(r) and 12<=r['remaining_hours']<=24 and .05<=r['p']<=.95]
        if len({r['event'] for r in training})<80 or len({r['group'] for r in validation})<10:
            raise ValueError('Need 80 training events and 10 primary validation dates with observed disagreement')
    except ValueError as exc:
        result.update(status='insufficient_data',reason=str(exc),calibration_improvement_established=False)
        (folder/'report.json').write_text(json.dumps(result,indent=2),encoding='utf-8');return result
    result.update(status='research_only',validation={},models={},split_counts=dict(training=len(training),validation=len(validation)))
    predictions={}
    for arm in protocol['arms']:
        model=RemainingMaximum(observation_calibration,remaining_spread=arm).fit(training)
        payload=pickle.dumps(model);(folder/(arm+'.pkl')).write_bytes(payload)
        predictions[arm]=model.predict(validation)
        result['validation'][arm]=calibration_metrics(validation,predictions[arm])
        strata={}
        for name,lo,hi in [('under_2F',0,2),('2_to_5F',2,5),('5F_or_more',5,math.inf)]:
            indices=[i for i,r in enumerate(validation) if lo<=r['model_disagreement_f']<hi]
            if indices:strata[name]=calibration_metrics([validation[i] for i in indices],predictions[arm][indices])
        result['validation'][arm]['by_disagreement']=strata
        result['models'][arm]=dict(sha256=hashlib.sha256(payload).hexdigest(),theta=model.theta.tolist())
    result['paired_validation_log_loss']=paired_block_interval(validation,predictions['disagreement'],predictions['constant'])
    result['calibration_improvement_established']=False
    result['interpretation']='Review reliability, ECE and Brier together; a single inspected validation comparison cannot establish future calibration or profit.'
    (folder/'report.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result
