"""Matched full/PCA FP32 representations on live FIFO traces with component timing."""
import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import platform
import random
import statistics
import time

from . import live_burst

_worker_inputs=None


def _extract(ids):
    from . import experiment
    model,transform,dataset=_worker_inputs
    return experiment.features(model,transform,dataset,ids).numpy()


def collect(model,transform,dataset,indices,workers):
    import torch
    from . import experiment
    if workers<=1:
        return experiment.features(model,transform,dataset,indices)
    global _worker_inputs
    _worker_inputs=(model,transform,dataset)
    shards=[indices[i:i+320] for i in range(0,len(indices),320)]
    with mp.get_context('fork').Pool(min(workers,len(shards),len(os.sched_getaffinity(0)))) as pool:
        parts=[]
        for i,array in enumerate(pool.imap(_extract,shards),1):
            parts.append(torch.from_numpy(array))
            if i%5==0 or i==len(shards):print('Extracted',min(i*320,len(indices)),'/',len(indices),flush=True)
    return torch.cat(parts)


def fit_states(x,labels,widths,ridge):
    import torch
    from . import experiment
    mean=x.mean(0)
    _,_,vh=torch.linalg.svd(x-mean,full_matrices=False)
    states={}
    for width in [x.shape[1],*widths]:
        projection=None if width==x.shape[1] else vh[:width].T.contiguous()
        features=x if projection is None else (x-mean)@projection
        weight,bias=experiment.ridge_head(features,labels,ridge)
        states[width]=dict(width=width,mean=mean,projection=projection,weight=weight,bias=bias)
    return states


def logits_for(features,state):
    z=features if state['projection'] is None else (features-state['mean'])@state['projection']
    return z@state['weight']+state['bias']


def select_width(candidates):
    qualified=[c for c in candidates if c['quality_pass']]
    if qualified:return min(qualified,key=lambda c:c['width']),True
    # Preserve the ability to diagnose compression even when the quality gate fails.
    return max(candidates,key=lambda c:(c['accuracy_pct'],c['width'])),False


def timing_comparison(sessions,settings):
    gains=[100*(1-s['metrics']['pca_fp32']['p95_response_ms']/s['metrics']['full_fp32']['p95_response_ms']) for s in sessions]
    rng=random.Random(settings['seed'])
    draws=[statistics.mean(rng.choices(gains,k=len(gains))) for _ in range(settings['bootstrap_samples'])]
    ci=[live_burst.quantile(draws,.025),live_burst.quantile(draws,.975)]
    return dict(mean_p95_reduction_pct=statistics.mean(gains),interval_95_pct=ci,
                per_session_reduction_pct=gains,latency_target_pass=ci[0]>=settings['p95_target_pct'],
                interval_within_plus_minus_5pct=ci[0]>=-5 and ci[1]<=5)


def main():
    import torch
    from torchvision.datasets import CIFAR10
    from . import early_exit,experiment
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('artifacts','data','prior_sweep','output'):
        p.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    p.add_argument('--settings',type=Path,default=Path('configs/live-pca.json'))
    args=p.parse_args()
    if args.output.exists():p.error('Use a fresh output directory')
    settings=json.loads(args.settings.read_text())
    if settings['sessions']!=6 or any(w<=0 or w>=512 for w in settings['candidate_widths']):
        p.error('Use six sessions and widths strictly between zero and 512')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    experiment.verify_preparation(args.artifacts,args.data)
    config=experiment.configuration(args.artifacts/'config.json')
    splits=json.loads((args.artifacts/'splits.json').read_text())
    train_data=CIFAR10(args.data,train=True);test_data=CIFAR10(args.data,train=False)
    model,_,transform=early_exit.load_models(args.artifacts)
    args.output.mkdir(parents=True)
    rng=random.Random(settings['seed'])
    train_indices=[]
    for label in range(10):
        group=[i for i in splits['train'] if train_data.targets[i]==label]
        train_indices+=rng.sample(group,settings['training_per_class'])
    train_indices=sorted(train_indices)
    if set(train_indices)&set(splits['calibration']):raise ValueError('Training/calibration overlap')
    print('Extracting matched training features',len(train_indices),flush=True)
    train=collect(model,transform,train_data,train_indices,settings['preparation_workers'])
    train_labels=torch.tensor(train_data.targets)[train_indices]
    states=fit_states(train,train_labels,settings['candidate_widths'],config['ridge'])
    print('Extracting calibration features',flush=True)
    cal=collect(model,transform,train_data,splits['calibration'],settings['preparation_workers'])
    cal_labels=torch.tensor(train_data.targets)[splits['calibration']]
    full_correct=logits_for(cal,states[512]).argmax(1).eq(cal_labels)
    candidates=[]
    for width in settings['candidate_widths']:
        correct=logits_for(cal,states[width]).argmax(1).eq(cal_labels)
        delta,ci=experiment.paired_quality_interval(full_correct,correct,config['bootstrap_samples'],config['seed'])
        candidates.append(dict(width=width,accuracy_pct=100*correct.double().mean().item(),delta_pp=delta,
                               interval_95_pp=ci,quality_pass=ci[0]>=-settings['quality_tolerance_pp']))
    chosen,calibration_qualified=select_width(candidates)
    selected=chosen['width']
    checkpoint=args.output/'heads.pt'
    torch.save({512:states[512],selected:states[selected]},checkpoint)
    (args.output/'calibration.json').write_text(json.dumps(dict(training_indices=train_indices,
        calibration_indices=splits['calibration'],full_accuracy_pct=100*full_correct.double().mean().item(),
        candidates=candidates,selected=selected,calibration_qualified=calibration_qualified),indent=2))
    print('Selected width',selected,'qualified',calibration_qualified,flush=True)
    @torch.inference_mode()
    def predict(image,action):
        state=states[512 if action=='full_fp32' else selected]
        t0=time.perf_counter_ns();inputs=transform(image).unsqueeze(0);t1=time.perf_counter_ns()
        features=model(inputs);t2=time.perf_counter_ns()
        if state['projection'] is None:
            z=features;t3=t2
        else:
            z=(features-state['mean'])@state['projection'];t3=time.perf_counter_ns()
        pred=(z@state['weight']+state['bias']).argmax(1).item();t4=time.perf_counter_ns()
        if z.dtype!=torch.float32:raise ValueError('Expected true FP32 feature tensors')
        return pred,False,dict(preprocess_ns=t1-t0,backbone_ns=t2-t1,
                              projection_ns=t3-t2,classifier_ns=t4-t3,
                              feature_bytes=z.numel()*z.element_size(),backbone_feature_bytes=features.numel()*features.element_size())
    # Calibration-only live full-path service profile determines the shared arrival schedule.
    profile_ids=rng.sample(splits['calibration'],settings['profile_examples'])
    profile_images=[train_data[i][0] for i in profile_ids]
    for i in range(settings['warmup']):predict(profile_images[i%len(profile_images)],'full_fp32')
    service=[]
    for image in profile_images:
        start=time.perf_counter_ns();predict(image,'full_fp32');service.append((time.perf_counter_ns()-start)/1e6)
    service_ms=statistics.median(service)
    gap_ns=round(service_ms/settings['nominal_load']*1e6)
    prior=json.loads((args.prior_sweep/'lock.json').read_text())
    used={r['image_id'] for plan in prior['plans'] for r in plan['trace']}
    available=[i for i in range(len(test_data)) if i not in used]
    test_indices=rng.sample(available,settings['held_out_examples'])
    if settings['sessions']*settings['requests']>len(test_indices):raise ValueError('Too few held-out examples for disjoint live panels')
    orders=[['full_fp32','pca_fp32']]*3+[['pca_fp32','full_fp32']]*3;rng.shuffle(orders)
    plans=[]
    for session in range(settings['sessions']):
        panel=test_indices[session*settings['requests']:(session+1)*settings['requests']]
        plans.append(dict(session=session,order=orders[session],trace=[dict(request_id=i,image_id=image,arrival_ns=i*gap_ns) for i,image in enumerate(panel)]))
    checkpoint_hash=experiment.digest(checkpoint)
    lock=dict(settings=settings,selected_width=selected,calibration_qualified=calibration_qualified,
              checkpoint_sha256=checkpoint_hash,calibration_sha256=experiment.digest(args.output/'calibration.json'),
              training_examples=len(train_indices),held_out_indices=test_indices,plans=plans,
              prior_sweep_lock_sha256=experiment.digest(args.prior_sweep/'lock.json'),
              test_namespace='CIFAR-10 official test; prior dynamic sweep panels excluded',
              test_batch_sha256=experiment.digest(args.data/'cifar-10-batches-py/test_batch'),
              preparation_sha256={n:experiment.digest(args.artifacts/n) for n in ('encoder.pt','manifest.json','splits.json','config.json')},
              sources={n:experiment.digest(m.__file__) for n,m in [('study',__import__(__name__,fromlist=['main'])),('fifo',live_burst),('experiment',experiment),('early_exit',early_exit)]},
              calibration_service_ms=service,arrival_gap_ns=gap_ns,platform=platform.platform(),torch_version=str(torch.__version__),threads=1,
              feature_bytes=dict(full_fp32=512*4,pca_fp32=selected*4),
              pca_parameter_bytes=states[selected]['projection'].numel()*4+states[selected]['mean'].numel()*4)
    (args.output/'lock.json').write_text(json.dumps(lock,indent=2,allow_nan=False))
    (args.output/'prior-sweep-lock.json').write_bytes((args.prior_sweep/'lock.json').read_bytes())
    print('Frozen live comparison; arrival gap ms',gap_ns/1e6,flush=True)
    print('Extracting held-out accuracy features',flush=True)
    held_out=collect(model,transform,test_data,test_indices,settings['preparation_workers'])
    labels=torch.tensor(test_data.targets)[test_indices]
    reference=logits_for(held_out,states[512]).argmax(1)
    reduced=logits_for(held_out,states[selected]).argmax(1)
    delta,ci=experiment.paired_quality_interval(reference.eq(labels),reduced.eq(labels),config['bootstrap_samples'],config['seed'])
    quality=dict(examples=len(test_indices),full_accuracy_pct=100*reference.eq(labels).double().mean().item(),
                 pca_accuracy_pct=100*reduced.eq(labels).double().mean().item(),delta_pp=delta,interval_95_pp=ci,
                 quality_pass=ci[0]>=-settings['quality_tolerance_pp'])
    # Retain exact per-image predictions for independent scoring and post-run live/batch checks.
    scored=[dict(image_id=image,label=int(labels[i]),full_prediction=int(reference[i]),pca_prediction=int(reduced[i])) for i,image in enumerate(test_indices)]
    (args.output/'held-out-predictions.json').write_text(json.dumps(scored,indent=2))
    lookup={r['image_id']:r for r in scored}
    sessions=[]
    actions={name:dict(thresholds=[name,name],depth=1) for name in ('full_fp32','pca_fp32')}
    for plan in plans:
        folder=args.output/f"session-{plan['session']}";folder.mkdir()
        images=[test_data[r['image_id']][0] for r in plan['trace']];raw={};metrics={}
        for name in plan['order']:
            for i in range(settings['warmup']):predict(images[i%len(images)],name)
            rows=live_burst.run_fifo(plan['trace'],images,actions[name],predict)
            live_burst.validate_records(plan['trace'],rows)
            for row in rows:
                diagnostics=row.pop('confidence');row['stages']=diagnostics;row['action']=row.pop('threshold')
                row['label']=lookup[row['image_id']]['label']
                if row['prediction']!=lookup[row['image_id']]['full_prediction' if name=='full_fp32' else 'pca_prediction']:
                    raise ValueError('Live/batch classification mismatch')
            (folder/f'{name}.json').write_text(json.dumps(rows,indent=2,allow_nan=False))
            value=live_burst.summarize(plan['trace'],rows,test_data.targets)
            value['stages_mean_ms']={key:statistics.mean(r['stages'][key]/1e6 for r in rows) for key in
                                     ('preprocess_ns','backbone_ns','projection_ns','classifier_ns')}
            value['feature_bytes']=rows[0]['stages']['feature_bytes']
            metrics[name]=value
        sessions.append(dict(session=plan['session'],metrics=metrics))
        (args.output/'sessions.json').write_text(json.dumps(sessions,indent=2,allow_nan=False))
        print('Completed live session',plan['session'],flush=True)
    timing=timing_comparison(sessions,settings)
    report=dict(status='matched_PCA_FP32_live_comparison',research_validated=False,quality=quality,timing=timing,
                selected_width=selected,calibration_qualified=calibration_qualified,
                feature_bytes=lock['feature_bytes'],feature_reduction_pct=100*(1-selected/512),
                lock_sha256=experiment.digest(args.output/'lock.json'),
                evidence_sha256={str(f.relative_to(args.output)):experiment.digest(f) for f in args.output.rglob('*') if f.is_file()},
                next_candidate='reduce actual encoder work; post-pooling PCA leaves the backbone unchanged' if not timing['latency_target_pass'] else 'confirm PCA latency result on additional independent traces',
                limitations=['Both heads and PCA fit the same 4,000-image training subset, not the previous 44,000-image heads.',
                             'Calibration selects width; 2,000 separate official-test images estimate held-out quality.',
                             'Prior static aggregate test results exist; these dynamic panels exclude the latest load sweep.',
                             'One shared CPU host, six paired sessions, no energy or peak-memory measurement.',
                             'Feature bytes describe the selected vector, not total RAM: the 512D backbone output and PCA parameters still exist.',
                             'Preparation uses parallel extraction; timed inference is single-thread, batch one, with no feature cache.'])
    (args.output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(dict(quality=quality,timing=timing,feature_bytes=report['feature_bytes'])),flush=True)


if __name__=='__main__':main()
