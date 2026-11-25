"""FP32 compact-encoder control after the fixed storage-grid feasibility gate."""
import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import random
import statistics
import time

from . import live_burst
_inputs=None


def _shard(indices):
    from . import experiment
    model,transform,data,width=_inputs
    return experiment.features(model,transform,data,indices,width).numpy()


def collect(model,transform,data,indices,width,workers=8):
    import torch
    global _inputs
    _inputs=(model,transform,data,width)
    shards=[indices[i:i+320] for i in range(0,len(indices),320)]
    with mp.get_context('fork').Pool(min(workers,len(shards),len(os.sched_getaffinity(0)))) as pool:
        parts=[]
        for i,array in enumerate(pool.imap(_shard,shards),1):
            parts.append(torch.from_numpy(array))
            if i%5==0 or i==len(shards):print('Extracted',min(i*320,len(indices)),'/',len(indices),flush=True)
    return torch.cat(parts)


def main():
    import torch
    from torchvision import models
    from torchvision.datasets import CIFAR10
    from . import early_exit,experiment
    from .width_precision import projected,quantize_storage,reconstruct
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('artifacts','data','prior_pca','grid','output'):p.add_argument('--'+n.replace('_','-'),required=True,type=Path)
    args=p.parse_args()
    if args.output.exists():p.error('Use a fresh output directory')
    grid=json.loads((args.grid/'summary.json').read_text())
    if grid['results']['supported_crossovers'] or grid['results']['supported_universal_fixed']:
        raise ValueError('The fixed-grid decision requires its supported controller/fixed branch first')
    grid_lock=json.loads((args.grid/'lock.json').read_text())
    grid_action=grid['baseline']
    prior=json.loads((args.prior_pca/'lock.json').read_text());calibration=json.loads((args.prior_pca/'calibration.json').read_text())
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    experiment.verify_preparation(args.artifacts,args.data)
    if experiment.digest(args.prior_pca/'heads.pt')!=prior['checkpoint_sha256']:raise ValueError('Prior heads changed')
    for name,value in prior['preparation_sha256'].items():
        if experiment.digest(args.artifacts/name)!=value:raise ValueError('Backbone/preparation changed')
    args.output.mkdir(parents=True)
    resnet,_,transform=early_exit.load_models(args.artifacts)
    # A trained compact architecture control, with smaller early channels and depthwise operators.
    # This changes architecture as well as channel widths: it is not an isolated ResNet width ablation.
    weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    compact=models.mobilenet_v3_small(weights=weights,progress=False).eval()
    compact.classifier=torch.nn.Identity()
    for parameter in compact.parameters():parameter.requires_grad_(False)
    if weights.transforms().crop_size!=transform.crop_size or weights.transforms().mean!=transform.mean:
        raise ValueError('Common input preprocessing is not compatible')
    train=CIFAR10(args.data,train=True);test=CIFAR10(args.data,train=False)
    train_ids=calibration['training_indices'];cal_ids=calibration['calibration_indices'];test_ids=prior['held_out_indices']
    print('Preparing compact encoder head',flush=True)
    features=collect(compact,transform,train,train_ids,576)
    weight,bias=experiment.ridge_head(features,torch.tensor(train.targets)[train_ids],1.)
    torch.save(dict(encoder=compact.state_dict(),weight=weight,bias=bias),args.output/'compact.pt')
    all_heads=torch.load(args.prior_pca/'heads.pt',weights_only=True)
    reference_head=all_heads[512]
    def compact_predictions(dataset,indices):
        x=collect(compact,transform,dataset,indices,576)
        return (x@weight+bias).argmax(1)
    cal_pred=compact_predictions(train,cal_ids)
    # Full-reference calibration predictions are preserved in the matched PCA calibration features only as summaries;
    # report compact calibration accuracy without selecting any new hyperparameter.
    cal_accuracy=100*cal_pred.eq(torch.tensor(train.targets)[cal_ids]).double().mean().item()
    lock=dict(architecture='torchvision MobileNetV3-Small IMAGENET1K_V1, classifier replaced with matched ridge head',
              architecture_control=True,isolated_resnet_width_ablation=False,weights_url=weights.url,
              prior_pca_lock_sha256=experiment.digest(args.prior_pca/'lock.json'),grid_summary_sha256=experiment.digest(args.grid/'summary.json'),
              compact_checkpoint_sha256=experiment.digest(args.output/'compact.pt'),
              reference_checkpoint_sha256=prior['checkpoint_sha256'],plans=prior['plans'][:4],
              train_indices=train_ids,calibration_indices=cal_ids,held_out_indices=test_ids,
              calibration_accuracy_pct=cal_accuracy,compact_embedding_width=576,reference_embedding_width=512,
              encoder_parameters=dict(resnet18=sum(p.numel() for p in resnet.parameters()),compact=sum(p.numel() for p in compact.parameters())),
              grid_fixed_action=grid_action,source_sha256=experiment.digest(__file__),platform=platform.platform(),torch_version=str(torch.__version__),threads=1)
    (args.output/'lock.json').write_text(json.dumps(lock,indent=2,allow_nan=False))
    test_pred=compact_predictions(test,test_ids)
    baseline=json.loads((args.prior_pca/'held-out-predictions.json').read_text());by_id={r['image_id']:r for r in baseline}
    labels=torch.tensor(test.targets)[test_ids];reference=torch.tensor([by_id[i]['full_prediction'] for i in test_ids])
    delta,ci=experiment.paired_quality_interval(reference.eq(labels),test_pred.eq(labels),2000,29)
    grid_scored={r['image_id']:r for r in json.loads((args.grid/'held-out-predictions.json').read_text())}
    grid_predictions=torch.tensor([grid_scored[i]['predictions'][grid_action] for i in test_ids])
    gd,gci=experiment.paired_quality_interval(grid_predictions.eq(labels),test_pred.eq(labels),2000,29)
    grid_quality=dict(reference_accuracy_pct=100*grid_predictions.eq(labels).double().mean().item(),delta_pp=gd,interval_95_pp=gci,quality_pass=gci[0]>=-1.)
    quality=dict(examples=len(test_ids),reference_accuracy_pct=100*reference.eq(labels).double().mean().item(),
                 compact_accuracy_pct=100*test_pred.eq(labels).double().mean().item(),delta_pp=delta,interval_95_pp=ci,quality_pass=ci[0]>=-1.)
    scored=[dict(image_id=image,label=int(labels[i]),reference_prediction=int(reference[i]),compact_prediction=int(test_pred[i]),grid_prediction=int(grid_predictions[i])) for i,image in enumerate(test_ids)]
    (args.output/'predictions.json').write_text(json.dumps(scored,indent=2));lookup={r['image_id']:r for r in scored}
    @torch.inference_mode()
    def predict(image,action):
        t0=time.perf_counter_ns();x=transform(image).unsqueeze(0);t1=time.perf_counter_ns()
        z=(compact if action=='compact' else resnet)(x);t2=time.perf_counter_ns();payload=z.numel()*z.element_size()
        if action=='grid_fixed':
            state=all_heads[grid_lock['selected_width'] if grid_action.startswith('reduced') else 512]
            z=projected(z,state)
            if 'int8' in grid_action:
                scale=grid_lock['scales'][str(state['width'])]
                q=quantize_storage(z,scale);payload=q.numel()*q.element_size();z=reconstruct(q,scale)
            if 'int8' not in grid_action:payload=z.numel()*z.element_size()
            w,b=state['weight'],state['bias']
        else:w,b=(reference_head['weight'],reference_head['bias']) if action=='resnet18' else (weight,bias)
        ta=time.perf_counter_ns()
        pred=(z@w+b).argmax(1).item();t3=time.perf_counter_ns()
        return pred,False,dict(preprocess_ns=t1-t0,backbone_ns=t2-t1,adapter_ns=ta-t2,classifier_ns=t3-ta,feature_bytes=payload,classifier_input_bytes=z.numel()*z.element_size())
    sessions=[]
    for plan in lock['plans']:
        # Exact prior input/arrival trace; pair order alternates for the two architecture controls.
        session=plan['session'];order=[['resnet18','grid_fixed','compact'],['grid_fixed','compact','resnet18'],['compact','resnet18','grid_fixed'],['compact','grid_fixed','resnet18']][session]
        folder=args.output/f'session-{session}';folder.mkdir();images=[test[r['image_id']][0] for r in plan['trace']];metrics={}
        for action in order:
            for i in range(3):predict(images[i%len(images)],action)
            rows=live_burst.run_fifo(plan['trace'],images,dict(thresholds=[action,action],depth=1),predict)
            for row in rows:
                row['stages']=row.pop('confidence');row['action']=row.pop('threshold');row['label']=lookup[row['image_id']]['label']
                expected=lookup[row['image_id']][{'resnet18':'reference_prediction','grid_fixed':'grid_prediction','compact':'compact_prediction'}[action]]
                if row['prediction']!=expected:raise ValueError('Live/batch prediction mismatch')
            (folder/f'{action}.json').write_text(json.dumps(rows,indent=2,allow_nan=False))
            m=live_burst.summarize(plan['trace'],rows,test.targets)
            m['stages_mean_ms']={k:statistics.mean(r['stages'][k]/1e6 for r in rows) for k in ('preprocess_ns','backbone_ns','adapter_ns','classifier_ns')}
            m['feature_bytes']=rows[0]['stages']['feature_bytes'];metrics[action]=m
        sessions.append(dict(session=session,metrics=metrics))
        (args.output/'sessions.json').write_text(json.dumps(sessions,indent=2));print('Completed compact session',session,flush=True)
    from .width_precision import paired_interval
    gains=[100*(1-s['metrics']['compact']['p95_response_ms']/s['metrics']['resnet18']['p95_response_ms']) for s in sessions]
    timing=paired_interval(gains,dict(seed=20261129,bootstrap_samples=8000))
    grid_gains=[100*(1-s['metrics']['compact']['p95_response_ms']/s['metrics']['grid_fixed']['p95_response_ms']) for s in sessions]
    grid_timing=paired_interval(grid_gains,dict(seed=20261129,bootstrap_samples=8000))
    report=dict(status='compact_encoder_architecture_control',research_validated=False,quality=quality,timing=timing,
                timing_vs_grid_fixed=grid_timing,quality_vs_grid_fixed=grid_quality,grid_fixed_action=grid_action,
                latency_pass=timing['interval_95'][0]>=5.,adopted=False,
                matched_pilot_gate_passed=timing['interval_95'][0]>=5. and quality['quality_pass'] and grid_timing['interval_95'][0]>=5. and grid_quality['quality_pass'],
                encoder_parameters=lock['encoder_parameters'],feature_bytes=dict(resnet18=2048,compact=2304),
                lock_sha256=experiment.digest(args.output/'lock.json'),
                evidence_sha256={str(f.relative_to(args.output)):experiment.digest(f) for f in args.output.rglob('*') if f.is_file()},
                limitations=['Different pretrained architectures; not a causal isolation of ResNet channel-width reduction.',
                             'Frozen pretrained bodies and same 4000 fitting images/ridge parameter for the two heads.',
                             'Four paired sessions at the prior single load, one host; broader deployment is unvalidated.',
                             'This test changes encoder work in FP32 and does not assert any INT8 arithmetic acceleration.'])
    (args.output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False));print(json.dumps(report),flush=True)


if __name__=='__main__':main()
