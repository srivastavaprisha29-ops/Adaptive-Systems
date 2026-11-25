"""Recompute the compact-encoder control from saved predictions and live records."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from adaptive_systems import live_burst
from adaptive_systems.width_precision import paired_interval


def verify(run):
    lock=json.loads((run/'lock.json').read_text());summary=json.loads((run/'summary.json').read_text())
    sessions=json.loads((run/'sessions.json').read_text());scored=json.loads((run/'predictions.json').read_text())
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    if digest(run/'lock.json')!=summary['lock_sha256'] or digest(run/'compact.pt')!=lock['compact_checkpoint_sha256']:
        raise ValueError('Lock/checkpoint mismatch')
    for name,value in summary['evidence_sha256'].items():
        if digest(run/name)!=value:raise ValueError('Evidence checksum mismatch')
    lookup={r['image_id']:r for r in scored}
    if len(lookup)!=2000 or set(lookup)!=set(lock['held_out_indices']):raise ValueError('Held-out IDs mismatch')
    accuracy={key:100*statistics.mean(r[key]==r['label'] for r in scored) for key in
              ('reference_prediction','compact_prediction','grid_prediction')}
    if not math.isclose(accuracy['compact_prediction'],summary['quality']['compact_accuracy_pct'],abs_tol=1e-9):
        raise ValueError('Compact held-out accuracy mismatch')
    if not math.isclose(accuracy['compact_prediction']-accuracy['reference_prediction'],summary['quality']['delta_pp'],abs_tol=1e-5):
        raise ValueError('Reference quality delta mismatch')
    if not math.isclose(accuracy['compact_prediction']-accuracy['grid_prediction'],summary['quality_vs_grid_fixed']['delta_pp'],abs_tol=1e-5):
        raise ValueError('Grid quality delta mismatch')
    checked=0;rebuilt=[]
    for plan in lock['plans']:
        metrics={}
        for action,column in [('resnet18','reference_prediction'),('grid_fixed','grid_prediction'),('compact','compact_prediction')]:
            rows=json.loads((run/f"session-{plan['session']}"/f'{action}.json').read_text())
            live_burst.validate_records(plan['trace'],rows)
            for row in rows:
                if row['action']!=action or row['prediction']!=lookup[row['image_id']][column] or row['label']!=lookup[row['image_id']]['label']:
                    raise ValueError('Live prediction mismatch')
                keys=('preprocess_ns','backbone_ns','adapter_ns','classifier_ns')
                if min(row['stages'][k] for k in keys)<0 or sum(row['stages'][k] for k in keys)>row['service_ns']:
                    raise ValueError('Timing decomposition mismatch')
            labels={r['image_id']:r['label'] for r in rows}
            m=live_burst.summarize(plan['trace'],rows,labels)
            m['stages_mean_ms']={k:statistics.mean(r['stages'][k]/1e6 for r in rows) for k in
                                ('preprocess_ns','backbone_ns','adapter_ns','classifier_ns')}
            m['feature_bytes']=rows[0]['stages']['feature_bytes'];metrics[action]=m;checked+=len(rows)
        rebuilt.append(dict(session=plan['session'],metrics=metrics))
    if rebuilt!=sessions:raise ValueError('Live summaries mismatch')
    for reference,key in [('resnet18','timing'),('grid_fixed','timing_vs_grid_fixed')]:
        gains=[100*(1-s['metrics']['compact']['p95_response_ms']/s['metrics'][reference]['p95_response_ms']) for s in sessions]
        if paired_interval(gains,dict(seed=20261129,bootstrap_samples=8000))!=summary[key]:raise ValueError('Paired timing interval mismatch')
    gate=(summary['timing']['interval_95'][0]>=5 and summary['quality']['quality_pass'] and
          summary['timing_vs_grid_fixed']['interval_95'][0]>=5 and summary['quality_vs_grid_fixed']['quality_pass'])
    if gate!=summary['matched_pilot_gate_passed']:raise ValueError('Pilot adoption gate mismatch')
    if summary['research_validated'] or summary['adopted']:raise ValueError('A one-load architecture control cannot claim deployment validation')
    return dict(held_out_examples=len(scored),verified_live_requests=checked,matched_pilot_gate_passed=gate,
                architecture_control=True,isolated_resnet_width_ablation=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    print(json.dumps(verify(p.parse_args().run)))
