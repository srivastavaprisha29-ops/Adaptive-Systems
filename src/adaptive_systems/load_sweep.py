"""Calibrate a machine-specific fixed comparator, then run a frozen-policy load sweep."""
import argparse
import itertools
import json
import platform
import random
import statistics
import time
from pathlib import Path

from . import live_burst


def bootstrap_bounds(values,samples,seed,lower_probability):
    if len(values)<2: raise ValueError('Repeated paired sessions required')
    rng=random.Random(seed)
    draws=[statistics.mean(rng.choices(values,k=len(values))) for _ in range(samples)]
    return dict(mean=statistics.mean(values),interval_95=[live_burst.quantile(draws,.025),live_burst.quantile(draws,.975)],
                adoption_lower=live_burst.quantile(draws,lower_probability))


def analyze(sessions,settings):
    output={}
    tail=settings['family_alpha']/settings['planned_adoption_bounds']
    for level in settings['nominal_loads']:
        records=[s for s in sessions if s['level']==level]
        if len(records)!=settings['sessions_per_load']: raise ValueError('Missing repeated load sessions')
        comparisons={}
        for candidate in ('bounded_queue','always_active'):
            for reference in ('best_fixed','always_active'):
                if candidate==reference:continue
                gains=[];deltas=[]
                for r in records:
                    a,b=(r['metrics'][n] for n in (candidate,reference))
                    gains.append(100*(1-a['p95_response_ms']/b['p95_response_ms']))
                    deltas.append(a['accuracy_pct']-b['accuracy_pct'])
                comparisons[f'{candidate}_vs_{reference}']=dict(
                    latency=bootstrap_bounds(gains,settings['bootstrap_samples'],settings['seed'],tail),
                    quality=bootstrap_bounds(deltas,settings['bootstrap_samples'],settings['seed'],tail),
                    per_session_gains_pct=gains,per_session_accuracy_delta_pp=deltas)
        best=comparisons['bounded_queue_vs_best_fixed'];gate=comparisons['bounded_queue_vs_always_active']
        best_pass=best['latency']['adoption_lower']>=settings['p95_target_pct'] and best['quality']['adoption_lower']>=-settings['quality_tolerance_pp']
        gate_pass=gate['latency']['adoption_lower']>0 and gate['quality']['adoption_lower']>=-settings['quality_tolerance_pp']
        output[level]=dict(comparisons=comparisons,beats_best_fixed_target=best_pass,
                           adds_value_over_always_active=gate_pass,claimed_win=best_pass and gate_pass)
    return output


def make_plans(settings,service_ms,available):
    if service_ms<=0 or settings['sessions_per_load']!=6: raise ValueError('Need positive capacity estimate and six balanced sessions')
    levels=list(settings['nominal_loads']);names=['best_fixed','always_active','bounded_queue']
    if len(levels)!=3 or any(v<=0 for v in settings['nominal_loads'].values()):raise ValueError('Need three positive offered loads')
    needed=3*6*settings['requests']
    if len(set(available))!=len(available) or len(available)<needed:raise ValueError('Need disjoint fresh panels')
    pool=list(available);rng=random.Random(settings['seed']);rng.shuffle(pool)
    policy_orders={level:list(itertools.permutations(names)) for level in levels}
    for orders in policy_orders.values():rng.shuffle(orders)
    load_orders=list(itertools.permutations(levels));rng.shuffle(load_orders)
    plans=[];offset=0
    for session in range(6):
        for level in load_orders[session]:
            ids=pool[offset:offset+settings['requests']];offset+=len(ids)
            gap_ns=round(service_ms/settings['nominal_loads'][level]*1e6)
            plans.append(dict(session=session,level=level,nominal_load=settings['nominal_loads'][level],gap_ns=gap_ns,
                              order=policy_orders[level][session],trace=[dict(request_id=i,image_id=image,arrival_ns=i*gap_ns) for i,image in enumerate(ids)]))
    return plans


def main():
    import torch
    from torchvision.datasets import CIFAR10
    from . import confidence_exit,early_exit,experiment
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('artifacts','head','data','feature_cache','prior_run','output'):
        p.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    p.add_argument('--settings',type=Path,default=Path('configs/load-sweep.json'))
    args=p.parse_args()
    if args.output.exists():p.error('Use a fresh output directory')
    settings=json.loads(args.settings.read_text())
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    experiment.verify_preparation(args.artifacts,args.data)
    config=experiment.configuration(args.artifacts/'config.json')
    splits=json.loads((args.artifacts/'splits.json').read_text())
    prior=json.loads((args.prior_run/'lock.json').read_text())
    head=torch.load(args.head,weights_only=True,map_location='cpu')
    provenance={n:experiment.digest(args.artifacts/n) for n in ('manifest.json','config.json','splits.json','states.pt','encoder.pt')}
    expected=dict(provenance,early_exit_source=experiment.digest(early_exit.__file__),experiment_source=experiment.digest(experiment.__file__))
    if head['provenance']!=expected or prior['preparation']!=provenance or prior['head_sha256']!=experiment.digest(args.head):
        raise ValueError('Model and prior inputs mismatch')
    saved=torch.load(args.feature_cache/'calibration.pt',weights_only=True,map_location='cpu')
    if saved['signature']['preparation']!=provenance or saved['signature']['indices']!=splits['calibration']:
        raise ValueError('Calibration cache mismatch')
    x,teacher,labels=saved['values'];logits=x@head['weight']+head['bias']
    margins=confidence_exit.margin(logits);early_correct=logits.argmax(1).eq(labels);full_correct=teacher.argmax(1).eq(labels)
    full,early,transform=early_exit.load_models(args.artifacts)
    baseline=torch.load(args.artifacts/'states.pt',weights_only=True)['512_fp32']
    train_data=CIFAR10(args.data,train=True)
    test_data=CIFAR10(args.data,train=False)
    @torch.inference_mode()
    def predict(image,threshold):
        cpu=time.thread_time_ns();x=transform(image).unsqueeze(0)
        for layer in (full.conv1,full.bn1,full.relu,full.maxpool,full.layer1,full.layer2,full.layer3):x=layer(x)
        z=early[8](early[7](x))@head['weight']+head['bias']
        margin=confidence_exit.margin(z).item();pred=z.argmax(1).item();prob=z.softmax(-1).max().item()
        exited=threshold is not None and margin>=threshold
        answer=pred if exited else experiment.classify(torch.flatten(full.avgpool(full.layer4(x)),1),baseline).argmax(1).item()
        return answer,exited,dict(margin=margin,top_probability=prob,early_prediction=pred,inference_thread_cpu_ns=time.thread_time_ns()-cpu)
    args.output.mkdir(parents=True)
    def timed(trace,order,policies,dataset,folder):
        folder.mkdir(parents=True)
        images=[dataset[r['image_id']][0] for r in trace]
        raw={}
        for name in order:
            for i in range(settings['warmup']):predict(images[i%len(images)],0.);predict(images[i%len(images)],None)
            raw[name]=live_burst.run_fifo(trace,images,policies[name],predict)
            live_burst.validate_records(trace,raw[name])
            (folder/f'{name}.json').write_text(json.dumps(raw[name],indent=2,allow_nan=False))
        return raw
    # Only training-calibration data enters the fixed-policy quality filter and timing selection.
    default=prior['threshold'];limit=prior['relaxation_limit'];depth=prior['policies']['safeguarded_queue']['depth']
    thresholds=sorted({prior['policies']['best_fixed']['thresholds'][0],default,*[float(v) for v in torch.quantile(margins,torch.tensor([.7,.8,.9]))]})+[None]
    candidates=[]
    for i,threshold in enumerate(thresholds):
        correct=full_correct if threshold is None else torch.where(margins>=threshold,early_correct,full_correct)
        delta,ci=experiment.paired_quality_interval(full_correct,correct,config['bootstrap_samples'],config['seed'])
        candidates.append(dict(name=f'fixed-{i}',threshold=threshold,calibration_delta_pp=delta,
                               calibration_interval_95_pp=ci,quality_feasible=ci[0]>=-settings['quality_tolerance_pp']))
    qualified={c['name']:dict(thresholds=[c['threshold'],c['threshold']],depth=1) for c in candidates if c['quality_feasible']}
    calibration_ids=random.Random(settings['seed']).sample(splits['calibration'],settings['calibration_profile_requests'])
    profile=[]
    for repeat in range(settings['calibration_profile_repeats']):
        ids=list(calibration_ids);random.Random(settings['seed']+repeat).shuffle(ids)
        # One-nanosecond gaps keep the server backlogged, measuring real fixed-policy service capacity.
        trace=[dict(request_id=i,image_id=image,arrival_ns=i) for i,image in enumerate(ids)]
        order=list(qualified);random.Random(settings['seed']+repeat).shuffle(order)
        raw=timed(trace,order,qualified,train_data,args.output/'calibration-profile'/str(repeat))
        profile.append(dict(repeat=repeat,trace=trace,order=order,
                            metrics={n:live_burst.summarize(trace,rows,train_data.targets) for n,rows in raw.items()}))
    for candidate in candidates:
        if candidate['quality_feasible']:
            candidate['profile_mean_service_ms']=[r['metrics'][candidate['name']]['mean_service_ms'] for r in profile]
            candidate['selection_service_ms']=statistics.median(candidate['profile_mean_service_ms'])
    selected=min((c for c in candidates if c['quality_feasible']),key=lambda c:(c['selection_service_ms'],c['name']))
    policies=dict(best_fixed=qualified[selected['name']],
                  always_active=dict(thresholds=[default,default],depth=1),
                  bounded_queue=dict(thresholds=[default,limit],depth=depth))
    capacity_service=selected['selection_service_ms']
    plans=make_plans(settings,capacity_service,list(range(len(test_data))))
    lock=dict(settings=settings,prior_lock_sha256=experiment.digest(args.prior_run/'lock.json'),
              preparation=provenance,head_sha256=experiment.digest(args.head),policies=policies,
              fixed_candidates=candidates,fixed_selection=selected,profile=profile,
              capacity_service_ms=capacity_service,capacity_requests_per_second=1000/capacity_service,
              default_threshold=default,relaxation_limit=limit,plans=plans,
              dataset=dict(evaluation='CIFAR-10 official test',train=False,
                           test_batch_sha256=experiment.digest(args.data/'cifar-10-batches-py/test_batch')),
              sources={n:experiment.digest(m.__file__) for n,m in [('sweep',__import__(__name__,fromlist=['main'])),('fifo',live_burst),('experiment',experiment),('early_exit',early_exit),('confidence',confidence_exit)]},
              platform=platform.platform(),torch_version=str(torch.__version__),threads=1,
              family='3 load ranges x 4 one-sided endpoints; Bonferroni across all 12 for any adoption claim')
    (args.output/'lock.json').write_text(json.dumps(lock,indent=2,allow_nan=False))
    (args.output/'prior-lock.json').write_bytes((args.prior_run/'lock.json').read_bytes())
    (args.output/'evaluation-frozen.json').write_text(json.dumps(dict(lock_sha256=experiment.digest(args.output/'lock.json'),calibration_complete=True),indent=2))
    print('Frozen selection',json.dumps(selected),'capacity',1000/capacity_service,flush=True)
    sessions=[]
    for plan in plans:
        folder=args.output/f"session-{plan['session']}"/plan['level']
        raw=timed(plan['trace'],plan['order'],policies,test_data,folder)
        images=[test_data[r['image_id']][0] for r in plan['trace']]
        # Counterfactual full predictions and labels enter only after all three timed trials.
        reference=[predict(image,None)[0] for image in images]
        for rows in raw.values():
            for row,full_prediction in zip(rows,reference):
                row['full_prediction']=full_prediction;row['label']=int(test_data.targets[row['image_id']])
                expected=row['confidence']['early_prediction'] if row['early_exit'] else full_prediction
                if row['prediction']!=expected:raise ValueError('Prediction branch mismatch')
        metrics={n:live_burst.summarize(plan['trace'],rows,test_data.targets) for n,rows in raw.items()}
        for name,rows in raw.items():
            metrics[name]['observed_offered_load']=metrics[name]['mean_service_ms']/(plan['gap_ns']/1e6)
            (folder/f'{name}.json').write_text(json.dumps(rows,indent=2,allow_nan=False))
        sessions.append(dict(session=plan['session'],level=plan['level'],metrics=metrics))
        (args.output/'sessions.json').write_text(json.dumps(sessions,indent=2,allow_nan=False))
        print('Completed',plan['session'],plan['level'],flush=True)
    results=analyze(sessions,settings)
    wins=[name for name,r in results.items() if r['claimed_win']]
    report=dict(status='frozen_policy_fresh_trace_load_sweep',research_validated=False,results=results,
                claimed_load_wins=wins,next_direction='confirm winning ranges on new hardware/traces' if wins else 'carry machine-selected best fixed forward; shift research to width/precision adaptation',
                lock_sha256=experiment.digest(args.output/'lock.json'),
                evidence_sha256={str(f.relative_to(args.output)):experiment.digest(f) for f in args.output.rglob('*.json')},
                limitations=['Six paired sessions per range on one shared host; percentile bootstrap is approximate.',
                             'Official test data have prior static aggregate results, but these dynamic traces and image panels are fresh.',
                             'Nominal load is defined by calibration capacity; observed service utilization is also reported.',
                             'Best fixed means fastest quality-feasible member of the declared calibration grid, not all possible policies.',
                             'Confidence-gate parameters are copied unchanged from the prior lock.',
                             'No energy, peak-memory or actual width/precision adaptation claim.'])
    (args.output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print('Claimed winning ranges',wins,flush=True)


if __name__=='__main__':main()
