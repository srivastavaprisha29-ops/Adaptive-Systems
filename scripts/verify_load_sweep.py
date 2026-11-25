"""Verify a frozen-policy load sweep from saved records without ML inference."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from adaptive_systems import live_burst,load_sweep


def verify(run):
    lock=json.loads((run/'lock.json').read_text());summary=json.loads((run/'summary.json').read_text())
    sessions=json.loads((run/'sessions.json').read_text());prior=json.loads((run/'prior-lock.json').read_text())
    digest=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
    if digest(run/'lock.json')!=summary['lock_sha256'] or digest(run/'prior-lock.json')!=lock['prior_lock_sha256']:
        raise ValueError('Lock checksum mismatch')
    for name,value in summary['evidence_sha256'].items():
        if digest(run/name)!=value:raise ValueError('Raw checksum mismatch: '+name)
    if lock['policies']['always_active']['thresholds']!=[prior['threshold']]*2 or lock['policies']['bounded_queue']!=dict(
            thresholds=[prior['threshold'],prior['relaxation_limit']],depth=prior['policies']['safeguarded_queue']['depth']):
        raise ValueError('Confidence policies were retuned')
    frozen=json.loads((run/'evaluation-frozen.json').read_text())
    if frozen['lock_sha256']!=summary['lock_sha256'] or not frozen['calibration_complete']:
        raise ValueError('Calibration selection was not frozen')
    cal=json.loads((run/'calibration-scores.json').read_text())
    if (cal['preparation']!=lock['preparation'] or cal['head_sha256']!=lock['head_sha256']
            or cal['cache_sha256']!=prior['feature_cache_sha256']['calibration']):
        raise ValueError('Calibration score identity mismatch')
    cal_rows={r['image_id']:r for r in cal['records']}
    cal_labels={i:r['label'] for i,r in cal_rows.items()}
    calibration_checked=0
    # Recompute timing profiles and the exact machine-selection rule.
    for profile in lock['profile']:
        for name,saved in profile['metrics'].items():
            raw=json.loads((run/'calibration-profile'/str(profile['repeat'])/f'{name}.json').read_text())
            threshold=next(c['threshold'] for c in lock['fixed_candidates'] if c['name']==name)
            for row in raw:
                ref=cal_rows[row['image_id']]
                exited=threshold is not None and row['confidence']['margin']>=threshold
                expected_pred=ref['early_prediction'] if exited else ref['full_prediction']
                if row['threshold']!=threshold or row['early_exit']!=exited or row['prediction']!=expected_pred:
                    raise ValueError('Calibration profile dispatch mismatch')
            calibration_checked+=len(raw)
            if live_burst.summarize(profile['trace'],raw,cal_labels)!=saved:
                raise ValueError('Calibration live summary mismatch')
    for candidate in lock['fixed_candidates']:
        diffs=[]
        for r in cal_rows.values():
            early=candidate['threshold'] is not None and r['margin']>=candidate['threshold']
            pred=r['early_prediction'] if early else r['full_prediction']
            diffs.append(int(pred==r['label'])-int(r['full_prediction']==r['label']))
        if not math.isclose(100*statistics.mean(diffs),candidate['calibration_delta_pp'],abs_tol=1e-5):
            raise ValueError('Calibration quality point estimate mismatch')
        if candidate['quality_feasible']!=(candidate['calibration_interval_95_pp'][0]>=-lock['settings']['quality_tolerance_pp']):
            raise ValueError('Quality eligibility mismatch')
        if candidate['quality_feasible']:
            means=[p['metrics'][candidate['name']]['mean_service_ms'] for p in lock['profile']]
            if means!=candidate['profile_mean_service_ms'] or statistics.median(means)!=candidate['selection_service_ms']:
                raise ValueError('Fixed policy selection score mismatch')
    selected=min((c for c in lock['fixed_candidates'] if c['quality_feasible']),key=lambda c:(c['selection_service_ms'],c['name']))
    if selected!=lock['fixed_selection'] or lock['policies']['best_fixed']['thresholds']!=[selected['threshold']]*2:
        raise ValueError('Incorrect best fixed selection')
    if lock['dataset']['train'] is not False:raise ValueError('Evaluation is not the fresh dynamic test namespace')
    expected_plans=load_sweep.make_plans(lock['settings'],lock['capacity_service_ms'],list(range(10000)))
    if live_burst.sha_json(expected_plans)!=live_burst.sha_json(lock['plans']):raise ValueError('Fresh trace generation mismatch')
    seen=set();rebuilt=[];checked=0;alias_predictions=0
    for plan in lock['plans']:
        ids=[r['image_id'] for r in plan['trace']]
        if len(ids)!=len(set(ids)) or seen.intersection(ids):raise ValueError('Test panels overlap')
        seen.update(ids);metrics={};all_rows={}
        for name,policy in lock['policies'].items():
            rows=json.loads((run/f"session-{plan['session']}"/plan['level']/f'{name}.json').read_text())
            live_burst.validate_records(plan['trace'],rows)
            labels={r['image_id']:r['label'] for r in rows}
            for row in rows:
                mode=policy['thresholds'][0]!=policy['thresholds'][1] and row['pending_depth']>=policy['depth']
                threshold=policy['thresholds'][int(mode)]
                exited=threshold is not None and row['confidence']['margin']>=threshold
                prediction=row['confidence']['early_prediction'] if exited else row['full_prediction']
                if row['threshold']!=threshold or row['early_exit']!=exited or row['prediction']!=prediction:
                    raise ValueError('Dispatch or confidence decision mismatch')
                if row['monitor_ns']+row['decision_ns']>row['service_ns']:
                    raise ValueError('Nested timing mismatch')
            metrics[name]=live_burst.summarize(plan['trace'],rows,labels)
            metrics[name]['observed_offered_load']=metrics[name]['mean_service_ms']/(plan['gap_ns']/1e6)
            checked+=len(rows);all_rows[name]=rows
        if lock['policies']['best_fixed']['thresholds']==lock['policies']['always_active']['thresholds']:
            for a,b in zip(all_rows['best_fixed'],all_rows['always_active']):
                if (a['prediction'],a['early_exit'])!=(b['prediction'],b['early_exit']):
                    raise ValueError('Equivalent fixed gates disagree')
                alias_predictions+=1
        rebuilt.append(dict(session=plan['session'],level=plan['level'],metrics=metrics))
    if rebuilt!=sessions or load_sweep.analyze(rebuilt,lock['settings'])!=summary['results']:
        raise ValueError('Sweep metrics or adoption bounds mismatch')
    wins=[level for level,value in summary['results'].items() if value['claimed_win']]
    if wins!=summary['claimed_load_wins']:raise ValueError('Claimed ranges mismatch')
    if summary['research_validated']:raise ValueError('One-host exploratory sweep is not broader validation')
    return dict(verified_evaluation_requests=checked,verified_calibration_requests=calibration_checked,unique_test_images=len(seen),
                equivalent_fixed_predictions_checked=alias_predictions,frozen_policy_violations=0,claimed_ranges=wins)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    print(json.dumps(verify(p.parse_args().run)))
