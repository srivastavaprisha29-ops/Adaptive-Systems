"""Recompute the safeguarded-controller study from raw records without PyTorch."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from adaptive_systems import live_burst, safeguard_study


def verify(directory):
    summary=json.loads((directory/'summary.json').read_text())
    lock=json.loads((directory/'lock.json').read_text())
    sessions=json.loads((directory/'sessions.json').read_text())
    digest=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
    if digest(directory/'lock.json') != summary['lock_sha256']:
        raise ValueError('Lock checksum mismatch')
    for name,expected in summary['evidence_sha256'].items():
        if digest(directory/name) != expected: raise ValueError(f'Checksum mismatch: {name}')
    frozen=json.loads((directory/'evaluation-frozen.json').read_text())
    if frozen['lock_sha256'] != summary['lock_sha256'] or not frozen['no_threshold_changes']:
        raise ValueError('Evaluation not frozen to calibration lock')
    floor=lock['threshold']
    if floor != summary['frozen_threshold'] or not lock['calibration']['quality_pass']:
        raise ValueError('Missing calibrated confidence safeguard')
    seen=set();checked=0;rebuilt=[]
    calibration_ids={r['image_id'] for r in json.loads((directory/'calibration-live/best_fixed.json').read_text())}
    for plan,saved in zip(lock['plans'],sessions):
        ids=[r['image_id'] for r in plan['traces']['burst']]
        if len(ids)!=len(set(ids)) or seen.intersection(ids) or calibration_ids.intersection(ids):
            raise ValueError('Evaluation/calibration panels overlap')
        seen.update(ids)
        record=dict(session=plan['session'],metrics={},error_groups={})
        for workload,trace in plan['traces'].items():
            if ids != [r['image_id'] for r in trace]: raise ValueError('Workload image panels differ')
            record['metrics'][workload]={};record['error_groups'][workload]={}
            for name in lock['policies']:
                rows=json.loads((directory/f"session-{plan['session']}"/workload/f'{name}.json').read_text())
                live_burst.validate_records(trace,rows)
                labels={r['image_id']:r['label'] for r in rows}
                previous=False
                for row in rows:
                    policy=lock['policies'][name]
                    mode=policy['thresholds'][0]!=policy['thresholds'][1] and row['pending_depth']>=policy['depth']
                    threshold=policy['thresholds'][int(mode)]
                    expected_exit=threshold is not None and row['confidence']['margin']>=threshold
                    if row['threshold'] != threshold or row['early_exit'] != expected_exit:
                        raise ValueError('Policy/confidence decision mismatch')
                    if row['relaxed'] != mode or row['state_changed'] != (mode!=previous):
                        raise ValueError('Policy state mismatch')
                    previous=mode
                    if name!='best_fixed' and row['early_exit'] and row['confidence']['margin']<floor:
                        raise ValueError('Safeguard violated')
                    expected_pred=row['confidence']['early_prediction'] if expected_exit else row['full_prediction']
                    if row['prediction'] != expected_pred: raise ValueError('Prediction branch mismatch')
                    if row['state_update_ns']>row['decision_ns'] or row['monitor_ns']+row['decision_ns']>row['service_ns']:
                        raise ValueError('Nested timing mismatch')
                checked+=len(rows)
                record['metrics'][workload][name]=live_burst.summarize(trace,rows,labels)
                record['error_groups'][workload][name]=safeguard_study.error_groups(rows,
                    lock['settings']['confidence_bins_margin'],lock['settings']['queue_bins'])
        if record!=saved: raise ValueError('Session summaries or error groups mismatch')
        rebuilt.append(record)
    if len(rebuilt)!=lock['settings']['sessions']: raise ValueError('Missing sessions')
    if safeguard_study.compare(rebuilt,lock['settings'])!=summary['comparisons']:
        raise ValueError('Session-level comparisons mismatch')
    if summary['research_validated']: raise ValueError('Exploratory sessions cannot validate the broader study')
    return dict(verified_timed_requests=checked,unique_evaluation_images=len(seen),
                calibrated_floor=floor,safeguard_violations=0)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    print(json.dumps(verify(p.parse_args().run)))
