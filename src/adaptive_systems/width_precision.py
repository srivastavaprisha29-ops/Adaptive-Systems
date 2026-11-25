"""Four fixed FP32/INT8-storage configurations with calibration-frozen scales."""
import argparse
import json
import platform
import random
import statistics
import time
from pathlib import Path

from . import live_burst,live_pca

ACTIONS=('full_fp32','reduced_fp32','full_int8_storage','reduced_int8_storage')


def quantize_storage(z,scale):
    import torch
    if scale<=0:raise ValueError('Scale must be positive')
    return (z/scale).round().clamp(-127,127).to(torch.int8)


def reconstruct(q,scale):
    import torch
    if q.dtype!=torch.int8:raise ValueError('Storage tensor must be INT8')
    return q.float()*scale


def projected(features,state):
    return features if state['projection'] is None else (features-state['mean'])@state['projection']


def paired_interval(values,settings,tail=.025):
    rng=random.Random(settings['seed'])
    draws=[statistics.mean(rng.choices(values,k=len(values))) for _ in range(settings['bootstrap_samples'])]
    return dict(mean=statistics.mean(values),interval_95=[live_burst.quantile(draws,.025),live_burst.quantile(draws,.975)],
                adoption_lower=live_burst.quantile(draws,tail))


def analyze(sessions,quality,baseline,settings):
    result={};tail=settings['family_alpha']/(len(settings['loads'])*len(ACTIONS)*(len(ACTIONS)-1)*2)
    for level in settings['loads']:
        rows=[r for r in sessions if r['level']==level]
        if len(rows)!=settings['repeats']:raise ValueError('Missing load repetitions')
        comparisons={};supported=[]
        for candidate in ACTIONS:
            for reference in ACTIONS:
                if candidate==reference:continue
                gains=[100*(1-r['metrics'][candidate]['p95_response_ms']/r['metrics'][reference]['p95_response_ms']) for r in rows]
                delta=quality[candidate]['accuracy_pct']-quality[reference]['accuracy_pct']
                # Paired image uncertainty is computed once from the complete held-out panel.
                q=quality[candidate]['pairwise'][reference]
                latency=paired_interval(gains,settings,tail)
                passed=latency['adoption_lower']>=settings['p95_target_pct'] and q['simultaneous_lower_pp']>=-settings['quality_tolerance_pp']
                comparisons[f'{candidate}_vs_{reference}']=dict(latency=latency,quality_delta_pp=delta,
                    quality_interval_95_pp=q['interval_95_pp'],quality_adoption_lower_pp=q['simultaneous_lower_pp'],passes_latency_quality=passed)
            if candidate!=baseline and comparisons[f'{candidate}_vs_{baseline}']['passes_latency_quality']:
                supported.append(candidate)
        point_fastest=min(ACTIONS,key=lambda n:statistics.mean(r['metrics'][n]['p95_response_ms'] for r in rows))
        result[level]=dict(comparisons=comparisons,supported_vs_calibration_fixed=supported,point_fastest=point_fastest)
    crossovers=[]
    for i,left in enumerate(settings['loads']):
        for right in list(settings['loads'])[i+1:]:
            for a in ACTIONS:
                for b in ACTIONS:
                    if a!=b and result[left]['comparisons'][f'{a}_vs_{b}']['passes_latency_quality'] and result[right]['comparisons'][f'{b}_vs_{a}']['passes_latency_quality']:
                        crossovers.append(dict(load_a=left,action_a=a,load_b=right,action_b=b))
    universal=[a for a in ACTIONS if all(all(b==a or result[level]['comparisons'][f'{a}_vs_{b}']['passes_latency_quality'] for b in ACTIONS) for level in settings['loads'])]
    return dict(loads=result,supported_crossovers=crossovers,supported_universal_fixed=universal)


def main():
    import numpy as np
    import torch
    from torchvision.datasets import CIFAR10
    from . import early_exit,experiment
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('artifacts','data','prior_pca','output'):p.add_argument('--'+n.replace('_','-'),type=Path,required=True)
    p.add_argument('--settings',type=Path,default=Path('configs/width-precision-grid.json'))
    args=p.parse_args()
    if args.output.exists():p.error('Use a fresh output directory')
    settings=json.loads(args.settings.read_text())
    if settings['repeats']!=4:p.error('The four-position balanced protocol requires four repetitions')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    experiment.verify_preparation(args.artifacts,args.data)
    prior=json.loads((args.prior_pca/'lock.json').read_text())
    if experiment.digest(args.prior_pca/'heads.pt')!=prior['checkpoint_sha256']:raise ValueError('Prior heads changed')
    for name,expected in prior['preparation_sha256'].items():
        if experiment.digest(args.artifacts/name)!=expected:raise ValueError('Backbone/preparation changed')
    states=torch.load(args.prior_pca/'heads.pt',weights_only=True,map_location='cpu');width=prior['selected_width']
    train=CIFAR10(args.data,train=True);test=CIFAR10(args.data,train=False)
    calibration=json.loads((args.prior_pca/'calibration.json').read_text())
    model,_,transform=early_exit.load_models(args.artifacts)
    args.output.mkdir(parents=True)
    print('Extracting calibration features',flush=True)
    cal=live_pca.collect(model,transform,train,calibration['calibration_indices'],settings['preparation_workers'])
    scales={d:max(float(projected(cal,states[d]).abs().max())/127,1e-8) for d in (512,width)}
    def state_for(action):return states[width if action.startswith('reduced') else 512]
    def logits(features,action):
        state=state_for(action);z=projected(features,state)
        if 'int8' in action:z=reconstruct(quantize_storage(z,scales[state['width']]),scales[state['width']])
        return z@state['weight']+state['bias']
    cal_labels=torch.tensor(train.targets)[calibration['calibration_indices']]
    cal_full=logits(cal,'full_fp32').argmax(1).eq(cal_labels);cal_quality={}
    for action in ACTIONS:
        correct=logits(cal,action).argmax(1).eq(cal_labels)
        delta,ci=experiment.paired_quality_interval(cal_full,correct,2000,settings['seed'])
        cal_quality[action]=dict(accuracy_pct=100*correct.double().mean().item(),delta_pp=delta,interval_95_pp=ci,
                                 quality_pass=ci[0]>=-settings['quality_tolerance_pp'])
    @torch.inference_mode()
    def predict(image,action):
        state=state_for(action);t0=time.perf_counter_ns();x=transform(image).unsqueeze(0);t1=time.perf_counter_ns()
        f=model(x);t2=time.perf_counter_ns();z=projected(f,state);t3=time.perf_counter_ns()
        if 'int8' in action:
            q=quantize_storage(z,scales[state['width']]);t4=time.perf_counter_ns()
            decoded=reconstruct(q,scales[state['width']]);t5=time.perf_counter_ns();payload=q.numel()*q.element_size();metadata=4
        else:
            decoded=z;t4=t3;t5=t3;payload=z.numel()*z.element_size();metadata=0
        pred=(decoded@state['weight']+state['bias']).argmax(1).item();t6=time.perf_counter_ns()
        if decoded.dtype!=torch.float32:raise ValueError('Classifier input must be FP32')
        return pred,False,dict(preprocess_ns=t1-t0,backbone_ns=t2-t1,projection_ns=t3-t2,
            quantize_ns=t4-t3,dequantize_ns=t5-t4,classifier_ns=t6-t5,
            feature_bytes=payload,scale_bytes=metadata,classifier_input_bytes=decoded.numel()*decoded.element_size())
    rng=random.Random(settings['seed']);profile_ids=rng.sample(calibration['calibration_indices'],settings['profile_examples'])
    profile_trace=[dict(request_id=i,image_id=image,arrival_ns=i) for i,image in enumerate(profile_ids)]
    actions={n:dict(thresholds=[n,n],depth=1) for n in ACTIONS}
    def timed(trace,order,dataset,folder):
        folder.mkdir(parents=True);images=[dataset[r['image_id']][0] for r in trace];raw={}
        for name in order:
            for i in range(settings['warmup']):predict(images[i%len(images)],name)
            rows=live_burst.run_fifo(trace,images,actions[name],predict)
            for row in rows:
                row['stages']=row.pop('confidence');row['action']=row.pop('threshold');row['label']=dataset.targets[row['image_id']]
            live_burst.validate_records(trace,rows)
            (folder/f'{name}.json').write_text(json.dumps(rows,indent=2,allow_nan=False));raw[name]=rows
        return raw
    profiles=[]
    for repeat in range(settings['profile_repeats']):
        order=list(ACTIONS);rng.shuffle(order)
        raw=timed(profile_trace,order,train,args.output/'calibration-profile'/str(repeat))
        profiles.append(dict(repeat=repeat,order=order,trace=profile_trace,
                             metrics={a:live_burst.summarize(profile_trace,r,train.targets) for a,r in raw.items()}))
    profile_cost={a:statistics.median(p['metrics'][a]['mean_service_ms'] for p in profiles) for a in ACTIONS}
    baseline=min((a for a in ACTIONS if cal_quality[a]['quality_pass']),key=lambda a:(profile_cost[a],a))
    base_service=profile_cost['full_fp32'];plans=[]
    # First four prior PCA image panels are replayed at three loads. Within each load all four traces match.
    base_orders=[list(ACTIONS[i:]+ACTIONS[:i]) for i in range(4)];rng.shuffle(base_orders)
    for repeat in range(4):
        levels=list(settings['loads']);rng.shuffle(levels)
        ids=[r['image_id'] for r in prior['plans'][repeat]['trace']]
        for level in levels:
            gap=round(base_service/settings['loads'][level]*1e6)
            plans.append(dict(session=repeat,level=level,order=base_orders[repeat],gap_ns=gap,
                trace=[dict(request_id=i,image_id=image,arrival_ns=i*gap) for i,image in enumerate(ids)]))
    (args.output/'heads.pt').write_bytes((args.prior_pca/'heads.pt').read_bytes())
    (args.output/'prior-pca-lock.json').write_bytes((args.prior_pca/'lock.json').read_bytes())
    lock=dict(settings=settings,selected_width=width,scales=scales,scale_kind='one symmetric calibration-max FP32 scale per width, zero point 0',
        calibration_quality=cal_quality,profile=profiles,profile_cost_ms=profile_cost,baseline=baseline,plans=plans,
        held_out_indices=prior['held_out_indices'],prior_lock_sha256=experiment.digest(args.prior_pca/'lock.json'),
        checkpoint_sha256=prior['checkpoint_sha256'],preparation=prior['preparation_sha256'],
        sources={n:experiment.digest(m.__file__) for n,m in [('grid',__import__(__name__,fromlist=['main'])),('fifo',live_burst),('pca',live_pca),('experiment',experiment),('early_exit',early_exit)]},
        platform=platform.platform(),torch_version=str(torch.__version__),threads=1,
        arithmetic='FP32 backbone, projection and classifier; INT8 storage only with timed FP32 reconstruction')
    (args.output/'lock.json').write_text(json.dumps(lock,indent=2,allow_nan=False))
    print('Frozen grid, baseline',baseline,'profile',profile_cost,flush=True)
    print('Extracting held-out features',flush=True)
    features=live_pca.collect(model,transform,test,prior['held_out_indices'],settings['preparation_workers'])
    labels=torch.tensor(test.targets)[prior['held_out_indices']]
    predictions={a:logits(features,a).argmax(1) for a in ACTIONS};quality={}
    tail=settings['family_alpha']/(3*4*3*2)
    for a in ACTIONS:
        quality[a]=dict(accuracy_pct=100*predictions[a].eq(labels).double().mean().item(),pairwise={})
        for b in ACTIONS:
            if a==b:continue
            difference=(predictions[a].eq(labels).float()-predictions[b].eq(labels).float()).numpy()*100
            gen=np.random.default_rng(settings['seed']);draws=[float(gen.choice(difference,len(difference),replace=True).mean()) for _ in range(settings['bootstrap_samples'])]
            quality[a]['pairwise'][b]=dict(delta_pp=float(difference.mean()),interval_95_pp=np.quantile(draws,[.025,.975]).tolist(),simultaneous_lower_pp=float(np.quantile(draws,tail)))
    scored=[dict(image_id=image,label=int(labels[i]),predictions={a:int(predictions[a][i]) for a in ACTIONS}) for i,image in enumerate(prior['held_out_indices'])]
    (args.output/'held-out-predictions.json').write_text(json.dumps(scored,indent=2));lookup={r['image_id']:r for r in scored}
    sessions=[]
    for plan in plans:
        folder=args.output/f"session-{plan['session']}"/plan['level'];raw=timed(plan['trace'],plan['order'],test,folder)
        metrics={}
        for a,rows in raw.items():
            for row in rows:
                if row['prediction']!=lookup[row['image_id']]['predictions'][a]:raise ValueError('Live/batch prediction mismatch')
            m=live_burst.summarize(plan['trace'],rows,test.targets)
            m['stages_mean_ms']={k:statistics.mean(r['stages'][k]/1e6 for r in rows) for k in ('preprocess_ns','backbone_ns','projection_ns','quantize_ns','dequantize_ns','classifier_ns')}
            m['feature_bytes']=rows[0]['stages']['feature_bytes'];m['scale_bytes']=rows[0]['stages']['scale_bytes'];metrics[a]=m
        sessions.append(dict(session=plan['session'],level=plan['level'],metrics=metrics))
        (args.output/'sessions.json').write_text(json.dumps(sessions,indent=2,allow_nan=False))
        print('Completed',plan['session'],plan['level'],flush=True)
    results=analyze(sessions,quality,baseline,settings)
    report=dict(status='fixed_width_precision_storage_grid',research_validated=False,selected_width=width,quality=quality,
        baseline=baseline,results=results,controller_case=bool(results['supported_crossovers']),
        decision=('carry the supported universal fixed configuration forward' if results['supported_universal_fixed'] else
                          'build calibration-profile controller and test on fresh traces' if results['supported_crossovers'] else
                          'test a narrower encoder before the final embedding'),
        lock_sha256=experiment.digest(args.output/'lock.json'),
        evidence_sha256={str(f.relative_to(args.output)):experiment.digest(f) for f in args.output.rglob('*') if f.is_file()},
        limitations=['INT8 is compressed feature storage followed by FP32 classification, not INT8 acceleration.',
                     'Four paired repetitions per load on one shared host; bootstrap bounds are approximate.',
                     'Same 2000 test images, heads and PCA width as the previous experiment; no retuning on test accuracy.',
                     'First four prior live panels repeat across loads, so loads are not independent image sets.',
                     'Calibration selects the strongest fixed reference before evaluation.',
                     'All ordered-pair/load claims use a conservative family adjustment over 72 endpoints.'])
    (args.output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print('Controller crossovers',results['supported_crossovers'],flush=True)


if __name__=='__main__':main()
