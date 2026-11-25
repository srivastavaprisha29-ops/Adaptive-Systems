"""Audit the fixed width/precision grid from saved records without ML inference."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from adaptive_systems import live_burst,width_precision


def verify(run):
    lock=json.loads((run/'lock.json').read_text());summary=json.loads((run/'summary.json').read_text())
    sessions=json.loads((run/'sessions.json').read_text());prior=json.loads((run/'prior-pca-lock.json').read_text())
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    if digest(run/'lock.json')!=summary['lock_sha256'] or digest(run/'heads.pt')!=lock['checkpoint_sha256']:
        raise ValueError('Lock/checkpoint mismatch')
    if digest(run/'prior-pca-lock.json')!=lock['prior_lock_sha256'] or prior['selected_width']!=lock['selected_width']:
        raise ValueError('Prior width/checkpoint changed')
    for name,value in summary['evidence_sha256'].items():
        if digest(run/name)!=value:raise ValueError('Evidence checksum mismatch: '+name)
    for p in lock['profile']:
        for action,expected in p['metrics'].items():
            rows=json.loads((run/'calibration-profile'/str(p['repeat'])/f'{action}.json').read_text())
            labels={r['image_id']:r['label'] for r in rows}
            if live_burst.summarize(p['trace'],rows,labels)!=expected:raise ValueError('Calibration profile mismatch')
    costs={a:statistics.median(p['metrics'][a]['mean_service_ms'] for p in lock['profile']) for a in width_precision.ACTIONS}
    selected=min((a for a in width_precision.ACTIONS if lock['calibration_quality'][a]['quality_pass']),key=lambda a:(costs[a],a))
    if costs!=lock['profile_cost_ms'] or selected!=lock['baseline']:raise ValueError('Calibration fixed selection mismatch')
    scored=json.loads((run/'held-out-predictions.json').read_text());lookup={r['image_id']:r for r in scored}
    if list(r['image_id'] for r in scored)!=prior['held_out_indices']:raise ValueError('Held-out panel changed')
    for a in width_precision.ACTIONS:
        accuracy=100*statistics.mean(r['predictions'][a]==r['label'] for r in scored)
        if not math.isclose(accuracy,summary['quality'][a]['accuracy_pct'],abs_tol=1e-9):raise ValueError('Accuracy mismatch')
        for b in width_precision.ACTIONS:
            if a!=b:
                other=100*statistics.mean(r['predictions'][b]==r['label'] for r in scored)
                if not math.isclose(accuracy-other,summary['quality'][a]['pairwise'][b]['delta_pp'],abs_tol=1e-5):raise ValueError('Paired accuracy point mismatch')
    rebuilt=[];checked=0
    for plan in lock['plans']:
        original=prior['plans'][plan['session']]['trace']
        if [r['image_id'] for r in plan['trace']]!=[r['image_id'] for r in original]:raise ValueError('Prior live image trace changed')
        if len(plan['order'])!=4 or set(plan['order'])!=set(width_precision.ACTIONS):raise ValueError('Missing action')
        metrics={}
        for a in width_precision.ACTIONS:
            rows=json.loads((run/f"session-{plan['session']}"/plan['level']/f'{a}.json').read_text())
            live_burst.validate_records(plan['trace'],rows)
            width=lock['selected_width'] if a.startswith('reduced') else 512
            expected_bytes=width if 'int8' in a else width*4
            for row in rows:
                if row['action']!=a or row['prediction']!=lookup[row['image_id']]['predictions'][a] or row['label']!=lookup[row['image_id']]['label']:
                    raise ValueError('Prediction/action mismatch')
                stage=row['stages']
                if stage['feature_bytes']!=expected_bytes or stage['classifier_input_bytes']!=width*4:
                    raise ValueError('Storage/FP32 classifier byte mismatch')
                if stage['scale_bytes']!=(4 if 'int8' in a else 0):raise ValueError('Scale format byte mismatch')
                keys=('preprocess_ns','backbone_ns','projection_ns','quantize_ns','dequantize_ns','classifier_ns')
                if min(stage[k] for k in keys)<0 or sum(stage[k] for k in keys)>row['service_ns']:raise ValueError('Timing boundary mismatch')
                if 'int8' not in a and (stage['quantize_ns']!=0 or stage['dequantize_ns']!=0):raise ValueError('FP32 path converted storage')
            labels={r['image_id']:r['label'] for r in rows}
            m=live_burst.summarize(plan['trace'],rows,labels)
            m['stages_mean_ms']={k:statistics.mean(r['stages'][k]/1e6 for r in rows) for k in
                                ('preprocess_ns','backbone_ns','projection_ns','quantize_ns','dequantize_ns','classifier_ns')}
            m['feature_bytes']=rows[0]['stages']['feature_bytes'];m['scale_bytes']=rows[0]['stages']['scale_bytes'];metrics[a]=m
            checked+=len(rows)
        rebuilt.append(dict(session=plan['session'],level=plan['level'],metrics=metrics))
    if rebuilt!=sessions or width_precision.analyze(rebuilt,summary['quality'],lock['baseline'],lock['settings'])!=summary['results']:
        raise ValueError('Timing/ranking/crossover results mismatch')
    if summary['research_validated']:raise ValueError('Fixed pilot is not controller validation')
    return dict(held_out_images=len(scored),verified_live_requests=checked,baseline=selected,
                storage_only=True,classifier_dtype='FP32',supported_crossovers=summary['results']['supported_crossovers'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    print(json.dumps(verify(p.parse_args().run)))
