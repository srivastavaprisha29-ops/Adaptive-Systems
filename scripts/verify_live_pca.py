"""Verify saved PCA live traces, quality points and nested timing without ML inference."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from adaptive_systems import live_burst,live_pca


def verify(run):
    lock=json.loads((run/'lock.json').read_text());summary=json.loads((run/'summary.json').read_text())
    calibration=json.loads((run/'calibration.json').read_text());sessions=json.loads((run/'sessions.json').read_text())
    prior=json.loads((run/'prior-sweep-lock.json').read_text())
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    if digest(run/'lock.json')!=summary['lock_sha256'] or digest(run/'heads.pt')!=lock['checkpoint_sha256']:
        raise ValueError('Lock or checkpoint checksum mismatch')
    for name,expected in summary['evidence_sha256'].items():
        if digest(run/name)!=expected:raise ValueError('Evidence checksum mismatch: '+name)
    if digest(run/'calibration.json')!=lock['calibration_sha256'] or digest(run/'prior-sweep-lock.json')!=lock['prior_sweep_lock_sha256']:
        raise ValueError('Input lock checksum mismatch')
    chosen,qualified=live_pca.select_width(calibration['candidates'])
    if chosen['width']!=lock['selected_width'] or qualified!=lock['calibration_qualified']:
        raise ValueError('Calibration width selection mismatch')
    if set(calibration['training_indices'])&set(calibration['calibration_indices']):
        raise ValueError('PCA training uses calibration images')
    scored=json.loads((run/'held-out-predictions.json').read_text());lookup={r['image_id']:r for r in scored}
    ids=lock['held_out_indices'];used={r['image_id'] for p in prior['plans'] for r in p['trace']}
    if len(lookup)!=len(ids) or set(lookup)!=set(ids) or used.intersection(ids):
        raise ValueError('Held-out panel duplicate, missing or reused from prior sweep')
    full=100*statistics.mean(r['full_prediction']==r['label'] for r in scored)
    pca=100*statistics.mean(r['pca_prediction']==r['label'] for r in scored)
    for actual,key in [(full,'full_accuracy_pct'),(pca,'pca_accuracy_pct'),(pca-full,'delta_pp')]:
        if not math.isclose(actual,summary['quality'][key],abs_tol=1e-5):raise ValueError('Held-out quality mismatch')
    if summary['quality']['quality_pass']!=(summary['quality']['interval_95_pp'][0]>=-lock['settings']['quality_tolerance_pp']):
        raise ValueError('Held-out accuracy gate mismatch')
    seen=set();rebuilt=[];checked=0
    for plan in lock['plans']:
        panel=[r['image_id'] for r in plan['trace']]
        if seen.intersection(panel) or not set(panel)<=set(ids):raise ValueError('Live panels overlap or leave held-out set')
        seen.update(panel);metrics={}
        for name in ('full_fp32','pca_fp32'):
            rows=json.loads((run/f"session-{plan['session']}"/f'{name}.json').read_text())
            live_burst.validate_records(plan['trace'],rows)
            for row in rows:
                stages=row['stages']
                if row['action']!=name or row['prediction']!=lookup[row['image_id']]['full_prediction' if name=='full_fp32' else 'pca_prediction']:
                    raise ValueError('Live prediction/dispatch mismatch')
                if row['label']!=lookup[row['image_id']]['label']:raise ValueError('Live label mismatch')
                if any(stages[k]<0 for k in ('preprocess_ns','backbone_ns','projection_ns','classifier_ns')):
                    raise ValueError('Negative component timing')
                total=sum(stages[k] for k in ('preprocess_ns','backbone_ns','projection_ns','classifier_ns'))
                if total>row['service_ns']:raise ValueError('Component timers exceed outer service')
                if name=='full_fp32' and stages['projection_ns']!=0:raise ValueError('Full path projected features')
                if stages['feature_bytes']!=lock['feature_bytes'][name] or stages['backbone_feature_bytes']!=2048:
                    raise ValueError('Feature byte accounting mismatch')
            labels={r['image_id']:r['label'] for r in rows}
            value=live_burst.summarize(plan['trace'],rows,labels)
            value['stages_mean_ms']={key:statistics.mean(r['stages'][key]/1e6 for r in rows) for key in
                                    ('preprocess_ns','backbone_ns','projection_ns','classifier_ns')}
            value['feature_bytes']=rows[0]['stages']['feature_bytes']
            metrics[name]=value;checked+=len(rows)
        rebuilt.append(dict(session=plan['session'],metrics=metrics))
    if rebuilt!=sessions or live_pca.timing_comparison(rebuilt,lock['settings'])!=summary['timing']:
        raise ValueError('Timing summaries or paired-session interval mismatch')
    if summary['research_validated']:raise ValueError('A one-host pilot is not broad validation')
    return dict(held_out_examples=len(scored),verified_live_requests=checked,selected_width=lock['selected_width'],
                full_feature_bytes=lock['feature_bytes']['full_fp32'],pca_feature_bytes=lock['feature_bytes']['pca_fp32'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    print(json.dumps(verify(p.parse_args().run)))
