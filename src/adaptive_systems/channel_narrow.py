"""Structurally narrow ResNet-18 layer4 inner channels; retain a 512D output."""
import argparse
import copy
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import random
import statistics
import time

from . import live_burst
_cache_inputs=None


def narrow_layer4(original,width):
    import torch
    from torch import nn
    if type(width) is not int or not 0<width<=512:raise ValueError('Width must be an integer in (0, 512]')
    stage=copy.deepcopy(original)
    kept={}
    if width==512:return stage,kept
    for i,block in enumerate(stage):
        a,b,bn=block.conv1,block.conv2,block.bn1
        # Data-free structured selection: keep channels with largest outgoing weight energy.
        scores=b.weight.detach().square().sum((0,2,3))
        indices=scores.topk(width).indices.sort().values
        ca=nn.Conv2d(a.in_channels,width,a.kernel_size,a.stride,a.padding,a.dilation,1,False,a.padding_mode)
        cb=nn.Conv2d(width,b.out_channels,b.kernel_size,b.stride,b.padding,b.dilation,1,False,b.padding_mode)
        norm=nn.BatchNorm2d(width,eps=bn.eps,momentum=bn.momentum)
        with torch.no_grad():
            ca.weight.copy_(a.weight[indices]);cb.weight.copy_(b.weight[:,indices])
            norm.weight.copy_(bn.weight[indices]);norm.bias.copy_(bn.bias[indices])
            norm.running_mean.copy_(bn.running_mean[indices]);norm.running_var.copy_(bn.running_var[indices])
            norm.num_batches_tracked.copy_(bn.num_batches_tracked)
        block.conv1=ca;block.conv2=cb;block.bn1=norm
        kept[str(i)]=indices.tolist()
    return stage,kept


def _cache_shard(task):
    import numpy as np
    import torch
    prefix,transform,data,path,total=_cache_inputs
    start,indices=task
    output=np.memmap(path,dtype='float32',mode='r+',shape=(total,256,14,14))
    with torch.inference_mode():
        for offset in range(0,len(indices),16):
            ids=indices[offset:offset+16]
            values=prefix(torch.stack([transform(data[i][0]) for i in ids]))
            if values.shape[1:]!=(256,14,14):raise ValueError('Unexpected prefix tensor shape')
            output[start+offset:start+offset+len(ids)]=values.numpy()
    output.flush()
    return len(indices)


def cache_prefix(prefix,transform,data,indices,path,workers):
    import numpy as np
    global _cache_inputs
    output=np.memmap(path,dtype='float32',mode='w+',shape=(len(indices),256,14,14));output.flush();del output
    _cache_inputs=(prefix,transform,data,str(path),len(indices))
    tasks=[(i,indices[i:i+128]) for i in range(0,len(indices),128)]
    with mp.get_context('fork').Pool(min(workers,len(tasks),len(os.sched_getaffinity(0)))) as pool:
        count=0
        for value in pool.imap(_cache_shard,tasks):
            count+=value
            if count%1024==0 or count==len(indices):print('Cached prefix',count,'/',len(indices),flush=True)
    return dict(path=str(path),examples=len(indices),shape=[len(indices),256,14,14])


def batches(cache,batch_size=32):
    import numpy as np
    import torch
    values=np.memmap(cache['path'],dtype='float32',mode='r',shape=tuple(cache['shape']))
    for start in range(0,cache['examples'],batch_size):
        yield torch.from_numpy(values[start:start+batch_size].copy())


def recalibrate(stage,cache):
    import torch
    from torch import nn
    stage.eval()
    for module in stage.modules():
        if isinstance(module,nn.BatchNorm2d):
            module.reset_running_stats();module.momentum=None;module.train()
    with torch.inference_mode():
        for x in batches(cache):stage(x)
    stage.eval()


def pooled(stage,cache):
    import torch
    with torch.inference_mode():
        return torch.cat([stage(x).mean((2,3)) for x in batches(cache)])


def width_selection(candidates):
    passing=[c for c in candidates if c['quality_pass']]
    if passing:return min(passing,key=lambda c:c['width']),True
    return max(candidates,key=lambda c:(c['accuracy_pct'],c['width'])),False


def graph_inventory(model):
    import torch
    from torch import nn
    records=[];hooks=[]
    def record(name):
        def hook(module,inputs,output):
            records.append(dict(name=name,weight_shape=list(module.weight.shape),output_shape=list(output.shape),
                                convolution_macs=int(output.numel()*(module.in_channels//module.groups)*module.kernel_size[0]*module.kernel_size[1])))
        return hook
    for name,module in model.named_modules():
        if isinstance(module,nn.Conv2d):hooks.append(module.register_forward_hook(record(name)))
    with torch.inference_mode():model(torch.zeros(1,3,224,224))
    for hook in hooks:hook.remove()
    return dict(encoder_parameters=sum(p.numel() for p in model.parameters()),
                convolution_macs=sum(r['convolution_macs'] for r in records),convolutions=records)


def timing_result(sessions,settings):
    gains=[100*(1-s['metrics']['narrow']['p95_response_ms']/s['metrics']['full']['p95_response_ms']) for s in sessions]
    rng=random.Random(settings['seed']);draws=[statistics.mean(rng.choices(gains,k=len(gains))) for _ in range(settings['bootstrap_samples'])]
    ci=[live_burst.quantile(draws,.025),live_burst.quantile(draws,.975)]
    return dict(mean_p95_reduction_pct=statistics.mean(gains),interval_95_pct=ci,per_session_gains_pct=gains,
                latency_pass=ci[0]>=settings['p95_target_pct'])


def main():
    import torch
    from torch import nn
    from torchvision.datasets import CIFAR10
    from . import early_exit,experiment
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('artifacts','data','prior_pca','prior_sweep','output'):p.add_argument('--'+n.replace('_','-'),required=True,type=Path)
    p.add_argument('--settings',type=Path,default=Path('configs/channel-narrow.json'))
    args=p.parse_args()
    if args.output.exists():p.error('Use a fresh output directory')
    settings=json.loads(args.settings.read_text());torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    experiment.verify_preparation(args.artifacts,args.data)
    prior=json.loads((args.prior_pca/'lock.json').read_text());split=json.loads((args.prior_pca/'calibration.json').read_text())
    for name,value in prior['preparation_sha256'].items():
        if experiment.digest(args.artifacts/name)!=value:raise ValueError('Prior encoder/data preparation changed')
    config=experiment.configuration(args.artifacts/'config.json')
    train_data=CIFAR10(args.data,train=True);test_data=CIFAR10(args.data,train=False)
    train_ids=split['training_indices'];cal_ids=split['calibration_indices']
    if set(train_ids)&set(cal_ids):raise ValueError('Fitting/calibration overlap')
    base,early,transform=early_exit.load_models(args.artifacts);prefix=early[:7].eval()
    for parameter in base.parameters():parameter.requires_grad_(False)
    args.output.mkdir(parents=True);cache_dir=args.output/'cache';cache_dir.mkdir()
    print('Caching fitting prefix activations',flush=True)
    fitting=cache_prefix(prefix,transform,train_data,train_ids,cache_dir/'train.f32',settings['preparation_workers'])
    print('Caching calibration prefix activations',flush=True)
    calibration=cache_prefix(prefix,transform,train_data,cal_ids,cache_dir/'calibration.f32',settings['preparation_workers'])
    train_labels=torch.tensor(train_data.targets)[train_ids];cal_labels=torch.tensor(train_data.targets)[cal_ids]
    reference_stage,unused=narrow_layer4(base.layer4,512);recalibrate(reference_stage,fitting)
    reference_train=pooled(reference_stage,fitting);weight,bias=experiment.ridge_head(reference_train,train_labels,config['ridge'])
    reference_head=dict(weight=weight,bias=bias)
    reference_cal=pooled(reference_stage,calibration);reference_logits=reference_cal@weight+bias
    reference_correct=reference_logits.argmax(1).eq(cal_labels)
    candidates=[];stages={};heads={};kept={}
    for width in settings['inner_widths']:
        print('Fitting structurally narrowed layer4',width,flush=True)
        stage,indices=narrow_layer4(base.layer4,width);recalibrate(stage,fitting)
        features=pooled(stage,fitting);w,b=experiment.ridge_head(features,train_labels,config['ridge'])
        values=pooled(stage,calibration);pred=(values@w+b).argmax(1);correct=pred.eq(cal_labels)
        delta,ci=experiment.paired_quality_interval(reference_correct,correct,config['bootstrap_samples'],config['seed'])
        candidates.append(dict(width=width,accuracy_pct=100*correct.double().mean().item(),delta_pp=delta,interval_95_pp=ci,
                               quality_pass=ci[0]>=-settings['quality_tolerance_pp']))
        stages[width]=stage;heads[width]=dict(weight=w,bias=b);kept[width]=indices
        print('Calibration',candidates[-1],flush=True)
    selected,qualified=width_selection(candidates);width=selected['width']
    reference=copy.deepcopy(base);reference.layer4=reference_stage;reference.eval()
    narrow=copy.deepcopy(base);narrow.layer4=stages[width];narrow.eval()
    # Full-width convolution weights are unchanged. Store only its adapted BN state, plus the selected narrow stage.
    bn_state={name:copy.deepcopy(module.state_dict()) for name,module in reference_stage.named_modules() if isinstance(module,nn.BatchNorm2d)}
    torch.save(dict(width=width,narrow_layer4=stages[width].state_dict(),reference_layer4_bn=bn_state,
                    reference_head=reference_head,narrow_head=heads[width],kept_channels=kept[width]),args.output/'checkpoint.pt')
    @torch.inference_mode()
    def predict(image,action):
        model=reference if action=='full' else narrow;head=reference_head if action=='full' else heads[width]
        t0=time.perf_counter_ns();x=transform(image).unsqueeze(0);t1=time.perf_counter_ns()
        z=model(x);t2=time.perf_counter_ns();answer=(z@head['weight']+head['bias']).argmax(1).item();t3=time.perf_counter_ns()
        if z.shape!=(1,512) or z.dtype!=torch.float32:raise ValueError('Expected common 512D FP32 output')
        return answer,False,dict(preprocess_ns=t1-t0,encoder_ns=t2-t1,classifier_ns=t3-t2,feature_bytes=z.numel()*z.element_size())
    rng=random.Random(settings['seed']);profile_ids=rng.sample(cal_ids,settings['profile_examples'])
    images=[train_data[i][0] for i in profile_ids]
    for i in range(settings['warmup']):predict(images[i%len(images)],'full')
    profile=[]
    for image in images:
        start=time.perf_counter_ns();predict(image,'full');profile.append((time.perf_counter_ns()-start)/1e6)
    gap=round(statistics.median(profile)/settings['nominal_load']*1e6)
    sweep=json.loads((args.prior_sweep/'lock.json').read_text())
    used=set(prior['held_out_indices'])|{r['image_id'] for p in sweep['plans'] for r in p['trace']}
    available=[i for i in range(len(test_data)) if i not in used]
    held_out=rng.sample(available,settings['held_out_examples'])
    orders=[['full','narrow']]*3+[['narrow','full']]*3;rng.shuffle(orders)
    plans=[]
    for session in range(settings['sessions']):
        panel=held_out[session*settings['requests']:(session+1)*settings['requests']]
        plans.append(dict(session=session,order=orders[session],trace=[dict(request_id=i,image_id=image,arrival_ns=i*gap) for i,image in enumerate(panel)]))
    lock=dict(settings=settings,selected_inner_width=width,calibration_qualified=qualified,calibration_candidates=candidates,
        reference_calibration_accuracy_pct=100*reference_correct.double().mean().item(),
        training_indices=train_ids,calibration_indices=cal_ids,held_out_indices=held_out,plans=plans,
        input_encoder_sha256=experiment.digest(args.artifacts/'encoder.pt'),prior_pca_lock_sha256=experiment.digest(args.prior_pca/'lock.json'),
        prior_sweep_lock_sha256=experiment.digest(args.prior_sweep/'lock.json'),checkpoint_sha256=experiment.digest(args.output/'checkpoint.pt'),
        reference_graph=graph_inventory(reference),narrow_graph=graph_inventory(narrow),kept_channels=kept[width],
        adaptation='No gradient fine-tuning. Both layer4 BN statistics recalibrated on fitting data only; both ridge heads fit same 4000 examples.',
        source_sha256=experiment.digest(__file__),dependencies={n:experiment.digest(m.__file__) for n,m in [('fifo',live_burst),('experiment',experiment),('early_exit',early_exit)]},
        platform=platform.platform(),torch_version=str(torch.__version__),threads=1,profile_service_ms=profile,arrival_gap_ns=gap)
    (args.output/'lock.json').write_text(json.dumps(lock,indent=2,allow_nan=False))
    (args.output/'prior-pca-lock.json').write_bytes((args.prior_pca/'lock.json').read_bytes());(args.output/'prior-sweep-lock.json').write_bytes((args.prior_sweep/'lock.json').read_bytes())
    print('Frozen selected width',width,'calibration pass',qualified,flush=True)
    print('Caching fresh held-out prefix activations',flush=True)
    test_cache=cache_prefix(prefix,transform,test_data,held_out,cache_dir/'test.f32',settings['preparation_workers'])
    full_features=pooled(reference_stage,test_cache);narrow_features=pooled(stages[width],test_cache)
    ref_pred=(full_features@weight+bias).argmax(1);nar_pred=(narrow_features@heads[width]['weight']+heads[width]['bias']).argmax(1)
    labels=torch.tensor(test_data.targets)[held_out]
    delta,ci=experiment.paired_quality_interval(ref_pred.eq(labels),nar_pred.eq(labels),config['bootstrap_samples'],config['seed'])
    quality=dict(examples=len(held_out),reference_accuracy_pct=100*ref_pred.eq(labels).double().mean().item(),
                 narrow_accuracy_pct=100*nar_pred.eq(labels).double().mean().item(),delta_pp=delta,interval_95_pp=ci,quality_pass=ci[0]>=-settings['quality_tolerance_pp'])
    records=[dict(image_id=image,label=int(labels[i]),full_prediction=int(ref_pred[i]),narrow_prediction=int(nar_pred[i])) for i,image in enumerate(held_out)]
    (args.output/'held-out-predictions.json').write_text(json.dumps(records,indent=2));lookup={r['image_id']:r for r in records}
    sessions=[];policies={a:dict(thresholds=[a,a],depth=1) for a in ('full','narrow')}
    print('Starting live paired sessions',flush=True)
    for plan in plans:
        folder=args.output/f"session-{plan['session']}";folder.mkdir();metrics={};images=[test_data[r['image_id']][0] for r in plan['trace']]
        for action in plan['order']:
            for i in range(settings['warmup']):predict(images[i%len(images)],action)
            rows=live_burst.run_fifo(plan['trace'],images,policies[action],predict)
            for row in rows:
                row['stages']=row.pop('confidence');row['action']=row.pop('threshold');row['label']=lookup[row['image_id']]['label']
                if row['prediction']!=lookup[row['image_id']][f'{action}_prediction']:raise ValueError('Live/batch prediction mismatch')
            (folder/f'{action}.json').write_text(json.dumps(rows,indent=2,allow_nan=False))
            m=live_burst.summarize(plan['trace'],rows,test_data.targets)
            m['encoder_mean_ms']=statistics.mean(r['stages']['encoder_ns']/1e6 for r in rows);m['feature_bytes']=2048;metrics[action]=m
        sessions.append(dict(session=plan['session'],metrics=metrics))
        (args.output/'sessions.json').write_text(json.dumps(sessions,indent=2,allow_nan=False));print('Completed live session',plan['session'],flush=True)
    timing=timing_result(sessions,settings)
    report=dict(status='ResNet18_structurally_narrowed_layer4_FP32',research_validated=False,selected_inner_width=width,
        calibration_qualified=qualified,quality=quality,timing=timing,matched_pilot_gate_passed=qualified and quality['quality_pass'] and timing['latency_pass'],
        adopted=False,lock_sha256=experiment.digest(args.output/'lock.json'),
        evidence_sha256={str(f.relative_to(args.output)):experiment.digest(f) for f in args.output.rglob('*') if f.is_file() and 'cache' not in f.relative_to(args.output).parts},
        limitations=['Only layer4 internal channels are narrowed; the output embedding stays 512D FP32.',
                     'Structured selection uses pretrained outgoing-filter energy, with training-only BN-statistics adaptation; no gradient fine-tuning.',
                     'Both models use matching 4000 fitting examples and BN/ridge procedure; this differs from the older frozen-BN checkpoint.',
                     'Fresh test panels exclude previous PCA and load-sweep panels, but historical static aggregates used the official test dataset.',
                     'One shared host and six paired sessions; no precision/storage change, energy or peak-memory claim.'])
    (args.output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False));print(json.dumps(dict(quality=quality,timing=timing)),flush=True)


if __name__=='__main__':main()
