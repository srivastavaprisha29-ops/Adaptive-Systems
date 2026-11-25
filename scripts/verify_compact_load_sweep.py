"""Audit frozen compact multi-load evidence without rerunning inference."""
import argparse,itertools,json,math,statistics,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from adaptive_systems import live_burst,experiment
from adaptive_systems.compact_load_sweep import ACTIONS

def verify(run):
 lock=json.loads((run/'lock.json').read_text());summary=json.loads((run/'summary.json').read_text());sessions=json.loads((run/'sessions.json').read_text());scored=json.loads((run/'predictions.json').read_text())
 if experiment.digest(run/'lock.json')!=summary['lock_sha256']:raise ValueError('Lock hash mismatch')
 for name,digest in summary['evidence_sha256'].items():
  if experiment.digest(run/name)!=digest:raise ValueError('Evidence hash mismatch: '+name)
 supplement=lock.get('supplemental_low_load_check',False);replay=lock.get('diagnostic_anomaly_replay',False)
 if len(sessions)!=(1 if replay else 6 if supplement else 30) or len(scored)!=(48 if replay else 288 if supplement else 2880):raise ValueError('Incomplete experiment')
 source=run/'runner.py'
 if not source.exists():source=Path(__file__).resolve().parents[1]/'src/adaptive_systems/compact_load_sweep.py'
 if experiment.digest(source)!=lock['source_sha256']:raise ValueError('Runner source changed')
 ids=[r['image_id'] for r in scored]
 if len(set(ids))!=len(ids) or set(ids)!=set(lock['fresh_ids']) or set(ids)&set(lock['excluded_ids']):raise ValueError('Freshness or uniqueness failure')
 lookup={r['image_id']:r for r in scored};checked=0
 for s in sessions:
  plan=lock['plans'][s['session']]
  if s['level']!=plan['level']:raise ValueError('Load mismatch')
  for action in ACTIONS:
   rows=json.loads((run/f"session-{s['session']}"/f'{action}.json').read_text());live_burst.validate_records(plan['trace'],rows)
   for r in rows:
    expected=lookup[r['image_id']]
    if r['action']!=action or r['label']!=expected['label'] or r['prediction']!=expected['predictions'][action]:raise ValueError('Prediction mismatch')
    stages=r['stages'];times=[stages[k] for k in ('preprocess_ns','encoder_ns','adapter_ns','classifier_ns')]
    if min(times)<0 or sum(times)>r['service_ns']:raise ValueError('Invalid stage boundaries')
    if stages['feature_bytes']!={'compact':2304,'resnet18':2048,'grid_fixed':256}[action]:raise ValueError('Feature byte mismatch')
   m=live_burst.summarize(plan['trace'],rows,{r['image_id']:r['label'] for r in rows});m['mean_encoder_ms']=statistics.mean(r['stages']['encoder_ns']/1e6 for r in rows);m['mean_wait_ms']=statistics.mean(r['wait_ns']/1e6 for r in rows)
   if m!=s['metrics'][action]:raise ValueError('Timing summary mismatch')
   checked+=len(rows)
 for level in lock['arrival_gap_ms']:
  plans=[p for p in lock['plans'] if p['level']==level]
  if not replay and {tuple(p['order']) for p in plans}!=set(itertools.permutations(ACTIONS)):raise ValueError('Unbalanced execution order')
  for reference in ACTIONS[:2]:
   c=summary['results'][level]['comparisons'][reference]
   if c['passed']!=(c['timing']['adoption_lower']>=5 and c['accuracy_delta_pp']['adoption_lower']>=-1):raise ValueError('Gate mismatch')
 if summary['all_loads_passed']!=all(c['passed'] for r in summary['results'].values() for c in r['comparisons'].values()):raise ValueError('Aggregate gate mismatch')
 return dict(verified_live_requests=checked,fresh_images=len(scored),load_levels=len(lock['arrival_gap_ms']),matched_sessions=len(sessions),all_loads_passed=summary['all_loads_passed'])
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);print(json.dumps(verify(p.parse_args().run)))
