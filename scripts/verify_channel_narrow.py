"""Audit structural narrowing, split isolation, live timings and matched predictions."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from adaptive_systems import live_burst,channel_narrow


def verify(run):
    lock=json.loads((run/'lock.json').read_text());summary=json.loads((run/'summary.json').read_text())
    sessions=json.loads((run/'sessions.json').read_text());scored=json.loads((run/'held-out-predictions.json').read_text())
    prior=json.loads((run/'prior-pca-lock.json').read_text());sweep=json.loads((run/'prior-sweep-lock.json').read_text())
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    if digest(run/'lock.json')!=summary['lock_sha256'] or digest(run/'checkpoint.pt')!=lock['checkpoint_sha256']:
        raise ValueError('Lock or checkpoint checksum mismatch')
    if digest(run/'prior-pca-lock.json')!=lock['prior_pca_lock_sha256'] or digest(run/'prior-sweep-lock.json')!=lock['prior_sweep_lock_sha256']:
        raise ValueError('Prior input lock mismatch')
    for name,value in summary['evidence_sha256'].items():
        if digest(run/name)!=value:raise ValueError('Evidence checksum mismatch: '+name)
    width,qualified=channel_narrow.width_selection(lock['calibration_candidates'])
    if width['width']!=lock['selected_inner_width'] or qualified!=lock['calibration_qualified']:
        raise ValueError('Calibration selection mismatch')
    if set(lock['training_indices'])&set(lock['calibration_indices']):raise ValueError('Fitting/calibration overlap')
    old_used=set(prior['held_out_indices'])|{r['image_id'] for p in sweep['plans'] for r in p['trace']}
    ids=lock['held_out_indices']
    if len(ids)!=len(set(ids)) or old_used.intersection(ids):raise ValueError('Test panel is not fresh to prior dynamic trials')
    graph={r['name']:r for r in lock['narrow_graph']['convolutions']}
    reference={r['name']:r for r in lock['reference_graph']['convolutions']}
    selected=lock['selected_inner_width']
    for name,row in graph.items():
        expected=list(reference[name]['weight_shape'])
        if name in ('layer4.0.conv1','layer4.1.conv1'):expected[0]=selected
        if name in ('layer4.0.conv2','layer4.1.conv2'):expected[1]=selected
        if row['weight_shape']!=expected:raise ValueError('Convolution tensor shape mismatch: '+name)
    if lock['narrow_graph']['convolution_macs']>=lock['reference_graph']['convolution_macs'] or lock['narrow_graph']['encoder_parameters']>=lock['reference_graph']['encoder_parameters']:
        raise ValueError('No structural encoder-work reduction')
    lookup={r['image_id']:r for r in scored}
    if set(lookup)!=set(ids) or len(scored)!=len(ids):raise ValueError('Held-out scoring IDs mismatch')
    full=100*statistics.mean(r['full_prediction']==r['label'] for r in scored)
    narrow=100*statistics.mean(r['narrow_prediction']==r['label'] for r in scored)
    for actual,key in [(full,'reference_accuracy_pct'),(narrow,'narrow_accuracy_pct'),(narrow-full,'delta_pp')]:
        if not math.isclose(actual,summary['quality'][key],abs_tol=1e-5):raise ValueError('Held-out accuracy mismatch')
    quality_pass=summary['quality']['interval_95_pp'][0]>=-lock['settings']['quality_tolerance_pp']
    if quality_pass!=summary['quality']['quality_pass']:raise ValueError('Quality gate mismatch')
    rebuilt=[];seen=set();checked=0
    for plan in lock['plans']:
        panel=[r['image_id'] for r in plan['trace']]
        if seen.intersection(panel) or not set(panel)<=set(ids):raise ValueError('Live panel overlap')
        seen.update(panel);metrics={}
        for action in ('full','narrow'):
            rows=json.loads((run/f"session-{plan['session']}"/f'{action}.json').read_text())
            live_burst.validate_records(plan['trace'],rows)
            for row in rows:
                if row['action']!=action or row['prediction']!=lookup[row['image_id']][f'{action}_prediction'] or row['label']!=lookup[row['image_id']]['label']:
                    raise ValueError('Live/offline prediction mismatch')
                stages=row['stages'];keys=('preprocess_ns','encoder_ns','classifier_ns')
                if min(stages[k] for k in keys)<0 or sum(stages[k] for k in keys)>row['service_ns']:
                    raise ValueError('Nested encoder timing mismatch')
                if stages['feature_bytes']!=2048:raise ValueError('Output embedding should remain 512D FP32')
            labels={r['image_id']:r['label'] for r in rows}
            m=live_burst.summarize(plan['trace'],rows,labels)
            m['encoder_mean_ms']=statistics.mean(r['stages']['encoder_ns']/1e6 for r in rows);m['feature_bytes']=2048
            metrics[action]=m;checked+=len(rows)
        rebuilt.append(dict(session=plan['session'],metrics=metrics))
    if rebuilt!=sessions or channel_narrow.timing_result(rebuilt,lock['settings'])!=summary['timing']:
        raise ValueError('Live metrics or paired timing interval mismatch')
    gate=qualified and quality_pass and summary['timing']['latency_pass']
    if gate!=summary['matched_pilot_gate_passed']:raise ValueError('Joint gate mismatch')
    if summary['research_validated'] or summary['adopted']:raise ValueError('A matched pilot does not establish deployment validation')
    return dict(selected_internal_channels=selected,held_out_images=len(scored),verified_live_requests=checked,
                structurally_smaller_convolutions=True,embedding_bytes=2048,matched_pilot_gate_passed=gate)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    print(json.dumps(verify(p.parse_args().run)))
