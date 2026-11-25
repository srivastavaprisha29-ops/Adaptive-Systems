"""Frozen compact architecture validation on fresh matched live FIFO traces."""
import argparse,itertools,json,platform,random,statistics,time
from pathlib import Path
from . import live_burst,experiment,early_exit
from .width_precision import projected,quantize_storage,reconstruct,paired_interval
ACTIONS=('resnet18','grid_fixed','compact')
def main():
 import torch
 from torchvision import models
 from torchvision.datasets import CIFAR10
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--low-load-check',action='store_true');p.add_argument('--replay-low-anomaly',action='store_true');a=p.parse_args();out=a.output;root=Path('runs')
 if out.exists():p.error('Use a fresh output directory')
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
 prior=json.loads((root/'compact-encoder/lock.json').read_text());grid=json.loads((root/'width-precision/lock.json').read_text())
 paths={'compact':root/'compact-encoder/compact.pt','heads':root/'live-pca/heads.pt','encoder':Path('../repro-package/frozen/prepared/encoder.pt')}
 if experiment.digest(paths['compact'])!=prior['compact_checkpoint_sha256'] or experiment.digest(paths['heads'])!=prior['reference_checkpoint_sha256']:raise ValueError('Frozen checkpoint mismatch')
 full,_,transform=early_exit.load_models(Path('../repro-package/frozen/prepared'))
 compact=models.mobilenet_v3_small(weights=None);compact.classifier=torch.nn.Identity();saved=torch.load(paths['compact'],weights_only=True);compact.load_state_dict(saved['encoder']);compact.eval()
 heads=torch.load(paths['heads'],weights_only=True);action=prior['grid_fixed_action'];test=CIFAR10('data',train=False);train=CIFAR10('data',train=True)
 @torch.inference_mode()
 def predict(image,which):
  t0=time.perf_counter_ns();x=transform(image).unsqueeze(0);t1=time.perf_counter_ns();cpu1=time.thread_time_ns();z=(compact if which=='compact' else full)(x);t2=time.perf_counter_ns();cpu2=time.thread_time_ns();payload=z.numel()*z.element_size()
  if which=='grid_fixed':
   state=heads[grid['selected_width'] if action.startswith('reduced') else 512];z=projected(z,state)
   if 'int8' in action:
    scale=grid['scales'][str(state['width'])];q=quantize_storage(z,scale);payload=q.numel()*q.element_size();z=reconstruct(q,scale)
   else:payload=z.numel()*z.element_size()
   w,b=state['weight'],state['bias']
  else:w,b=(saved['weight'],saved['bias']) if which=='compact' else (heads[512]['weight'],heads[512]['bias'])
  t3=time.perf_counter_ns();pred=(z@w+b).argmax(1).item();t4=time.perf_counter_ns()
  return pred,False,dict(preprocess_ns=t1-t0,encoder_ns=t2-t1,encoder_thread_cpu_ns=cpu2-cpu1,adapter_ns=t3-t2,classifier_ns=t4-t3,feature_bytes=payload)
 profile={}
 for which in ACTIONS:
  imgs=[train[i][0] for i in prior['calibration_indices'][:24]]
  for img in imgs[:6]:predict(img,which)
  samples=[]
  for _ in range(3):
   for img in imgs:
    start=time.perf_counter_ns();predict(img,which);samples.append((time.perf_counter_ns()-start)/1e6)
  profile[which]=statistics.median(samples)
 excluded=set()
 for name in ('live-pca','load-sweep','channel-narrow') + (('compact-load-sweep',) if a.low_load_check else ()):
  lock=json.loads((root/name/'lock.json').read_text());excluded.update(lock.get('held_out_indices',[]))
  for plan in lock.get('plans',[]):excluded.update(r['image_id'] for r in plan['trace'])
 fresh=sorted(set(range(10000))-excluded);rng=random.Random(2026112607);rng.shuffle(fresh)
 required=288 if a.low_load_check else 2880
 if len(fresh)<required:raise ValueError(f'Insufficient fresh images: {len(fresh)}')
 levels={f'reference_{name}':profile['resnet18']/rho for name,rho in [('below',.65),('near',.95),('above',1.15)]};levels.update({f'compact_{name}':profile['compact']/rho for name,rho in [('near',.95),('above',1.15)]})
 if a.low_load_check:levels={'verified_low':250.0}
 count=48 if a.low_load_check else 96
 orders=list(itertools.permutations(ACTIONS));plans=[];cursor=0
 for level,gap in levels.items():
  for repeat in range(6):
   ids=fresh[cursor:cursor+count];cursor+=count;trace=[dict(request_id=i,image_id=image,arrival_ns=round(i*gap*1e6)) for i,image in enumerate(ids)];plans.append(dict(level=level,repeat=repeat,order=list(orders[repeat]),trace=trace))
 rng.shuffle(plans)
 if a.replay_low_anomaly:
  if not a.low_load_check:raise ValueError('Replay requires --low-load-check')
  plans=[json.loads((root/'compact-low-load/lock.json').read_text())['plans'][0]];fresh=[r['image_id'] for r in plans[0]['trace']];cursor=len(fresh)
 out.mkdir(parents=True);(out/'runner.py').write_text(Path(__file__).read_text())
 lock=dict(seed=2026112607,diagnostic_anomaly_replay=a.replay_low_anomaly,supplemental_low_load_check=a.low_load_check,actions=ACTIONS,grid_action=action,profile_median_service_ms=profile,arrival_gap_ms=levels,checkpoint_sha256={k:experiment.digest(v) for k,v in paths.items()},plans=plans,fresh_ids=fresh[:cursor],excluded_ids=sorted(excluded),threads=1,platform=platform.platform(),torch_version=str(torch.__version__),source_sha256=experiment.digest(__file__),quality_tolerance_pp=1,p95_target_pct=5,scope='Frozen architecture comparison, one host, no retraining or tuning; grid INT8 is storage-only')
 (out/'lock.json').write_text(json.dumps(lock,indent=2));print('Locked',len(plans),'sessions',cursor,'fresh images',profile,flush=True)
 sessions=[];scored=[]
 for j,plan in enumerate(plans):
  images=[test[r['image_id']][0] for r in plan['trace']];metrics={};predictions={};folder=out/f'session-{j}';folder.mkdir()
  for which in plan['order']:
   for img in images[:6]:predict(img,which)
   rows=live_burst.run_fifo(plan['trace'],images,dict(thresholds=[which,which],depth=1),predict)
   for row in rows:row['stages']=row.pop('confidence');row['action']=row.pop('threshold');row['label']=test.targets[row['image_id']]
   (folder/f'{which}.json').write_text(json.dumps(rows,indent=2));m=live_burst.summarize(plan['trace'],rows,test.targets);m['mean_encoder_ms']=statistics.mean(r['stages']['encoder_ns']/1e6 for r in rows);m['mean_wait_ms']=statistics.mean(r['wait_ns']/1e6 for r in rows);metrics[which]=m;predictions[which]=[r['prediction'] for r in rows]
  for i,r in enumerate(plan['trace']):scored.append(dict(image_id=r['image_id'],label=test.targets[r['image_id']],predictions={k:v[i] for k,v in predictions.items()}))
  sessions.append(dict(session=j,level=plan['level'],repeat=plan['repeat'],metrics=metrics));(out/'sessions.json').write_text(json.dumps(sessions,indent=2));print('Completed',j+1,'/',len(plans),plan['level'],flush=True)
 (out/'predictions.json').write_text(json.dumps(scored,indent=2));tail=.05/(20*2);results={}
 for level in levels:
  subset=[s for s in sessions if s['level']==level];ids={r['image_id'] for s in subset for r in plans[s['session']]['trace']};q=[r for r in scored if r['image_id'] in ids];comparisons={}
  for reference in ACTIONS[:2]:
   gains=[100*(1-s['metrics']['compact']['p95_response_ms']/s['metrics'][reference]['p95_response_ms']) for s in subset];timing=paired_interval(gains,dict(seed=2026112607,bootstrap_samples=20000),tail)
   diffs=[100*(int(r['predictions']['compact']==r['label'])-int(r['predictions'][reference]==r['label'])) for r in q];quality=paired_interval(diffs,dict(seed=2026112607,bootstrap_samples=20000),tail)
   comparisons[reference]=dict(timing=timing,accuracy_delta_pp=quality,passed=timing['adoption_lower']>=5 and quality['adoption_lower']>=-1)
  results[level]=dict(comparisons=comparisons,metrics={k:{m:statistics.mean(s['metrics'][k][m] for s in subset) for m in ('accuracy_pct','mean_encoder_ms','mean_wait_ms','p95_wait_ms','p95_response_ms','mean_service_ms')} for k in ACTIONS})
 report=dict(results=results,all_loads_passed=all(v['passed'] for r in results.values() for v in r['comparisons'].values()),family_tail=tail,accuracy_pct={k:100*statistics.mean(r['predictions'][k]==r['label'] for r in scored) for k in ACTIONS},fresh_examples=len(scored),live_requests=len(scored)*3,lock_sha256=experiment.digest(out/'lock.json'),evidence_sha256={str(f.relative_to(out)):experiment.digest(f) for f in out.rglob('*.json')},limitations=['One CPU host; finite 96-request overload sessions, duration affects queue latency.','Fresh IDs exclude saved prior trace/held-out sets; historical aggregate test exposure remains.','Frozen pretrained architecture comparison, not width-only causality.'])
 (out/'summary.json').write_text(json.dumps(report,indent=2));print('FINISHED',report['all_loads_passed'],report['accuracy_pct'],flush=True)
if __name__=='__main__':main()
