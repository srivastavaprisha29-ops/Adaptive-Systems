"""Calibrate confidence, freeze policies, and compare repeated live queue traces."""
import argparse
from bisect import bisect_right
import itertools
import json
from pathlib import Path
import platform
import random
import statistics

from . import live_burst


def paired_session_interval(values, samples, seed):
    if len(values) < 2:
        raise ValueError('Need repeated paired sessions')
    rng = random.Random(seed)
    draws = [statistics.mean(rng.choices(values, k=len(values))) for _ in range(samples)]
    return dict(mean=statistics.mean(values), interval_95=[live_burst.quantile(draws,.025), live_burst.quantile(draws,.975)])


def error_groups(rows, margin_bins, queue_bins):
    groups = {}
    for row in rows:
        key = f"margin-bin-{bisect_right(margin_bins,row['confidence']['margin'])-1}/queue-bin-{bisect_right(queue_bins,row['pending_depth'])-1}"
        group = groups.setdefault(key, dict(requests=0, early_wrong_full_right=0,
                                            harmful_actual_exits=0, actual_exits=0))
        harmful = row['confidence']['early_prediction'] != row['label'] and row['full_prediction'] == row['label']
        group['requests'] += 1
        group['early_wrong_full_right'] += int(harmful)
        group['harmful_actual_exits'] += int(harmful and row['early_exit'])
        group['actual_exits'] += int(row['early_exit'])
    return groups


def compare(sessions, settings):
    result = {}
    for workload in ('burst', 'steady'):
        result[workload] = {}
        for candidate in ('safeguarded_queue', 'fixed_confidence'):
            for reference in ('best_fixed', 'fixed_confidence'):
                if candidate == reference:
                    continue
                gain, delta = [], []
                for session in sessions:
                    a,b = (session['metrics'][workload][name] for name in (candidate,reference))
                    gain.append(100*(1-a['p95_response_ms']/b['p95_response_ms']))
                    delta.append(a['accuracy_pct']-b['accuracy_pct'])
                latency = paired_session_interval(gain, settings['bootstrap_samples'], settings['seed'])
                quality = paired_session_interval(delta, settings['bootstrap_samples'], settings['seed'])
                result[workload][f'{candidate}_vs_{reference}'] = dict(
                    latency_reduction_pct=latency, accuracy_delta_pp=quality,
                    per_session_gain_pct=gain, per_session_accuracy_delta_pp=delta,
                    joint_lower_bound_pass=latency['interval_95'][0] >= settings['p95_target_pct']
                    and quality['interval_95'][0] >= -settings['quality_tolerance_pp'])
    return result


def main():
    import torch
    from torchvision.datasets import CIFAR10
    from . import confidence_exit, early_exit, experiment

    p = argparse.ArgumentParser(description=__doc__)
    for name in ('artifacts','head','data','feature_cache','output'):
        p.add_argument('--'+name.replace('_','-'), type=Path, required=True)
    p.add_argument('--settings',type=Path,default=Path('configs/safeguard-study.json'))
    p.add_argument('--base-protocol',type=Path,default=Path('configs/burst-smoke.json'))
    args=p.parse_args()
    if args.output.exists(): p.error('Use a new output directory')
    settings=json.loads(args.settings.read_text())
    if settings['sessions'] != 6 or settings['requests'] < 1:
        p.error('This balanced protocol requires six sessions and positive request counts')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    experiment.verify_preparation(args.artifacts,args.data)
    config=experiment.configuration(args.artifacts/'config.json')
    splits=json.loads((args.artifacts/'splits.json').read_text())
    base=json.loads(args.base_protocol.read_text())
    head=torch.load(args.head,weights_only=True,map_location='cpu')
    provenance={n:experiment.digest(args.artifacts/n) for n in
                ('manifest.json','config.json','splits.json','states.pt','encoder.pt')}
    expected=dict(provenance,early_exit_source=experiment.digest(early_exit.__file__),
                  experiment_source=experiment.digest(experiment.__file__))
    if head['provenance'] != expected or experiment.digest(args.head) != base['identity']['head']:
        raise ValueError('Head does not match preparation or historical policy')
    def cached(split):
        saved=torch.load(args.feature_cache/f'{split}.pt',weights_only=True,map_location='cpu')
        if saved['signature']['preparation'] != provenance or saved['signature']['indices'] != splits[split]:
            raise ValueError('Feature cache preparation or split mismatch')
        x,teacher,labels=saved['values']
        logits=x@head['weight']+head['bias']
        return logits,teacher,labels
    # Calibration data only. Freeze the maximum-coverage margin cutoff passing the existing quality lower bound.
    cal,cal_full,cal_labels=cached('calibration')
    tau,calibration=confidence_exit.calibrate(confidence_exit.margin(cal),cal.argmax(1).eq(cal_labels),
                                            cal_full.argmax(1).eq(cal_labels),config)
    if tau == float('inf'): raise ValueError('No nontrivial safe early exit on calibration; use full-model fallback')
    strict=base['policies']['best_fixed']['thresholds'][0]
    policies=dict(best_fixed=dict(thresholds=[strict,strict],depth=1),
                  fixed_confidence=dict(thresholds=[tau,tau],depth=1),
                  safeguarded_queue=dict(thresholds=[None,tau],depth=settings['queue_depth']))
    # None means use the full path when there is no queue pressure. The same floor applies whenever exit is tried.
    names=list(policies)
    orders=list(itertools.permutations(names)); random.Random(settings['seed']).shuffle(orders)
    pool=list(splits['validation']); random.Random(settings['seed']).shuffle(pool)
    if len(pool) < settings['requests']*settings['sessions']: raise ValueError('Too few disjoint evaluation images')
    plans=[]
    for session in range(settings['sessions']):
        ids=pool[session*settings['requests']:(session+1)*settings['requests']]
        trace_protocol=dict(base,seed=settings['seed']+session,requests=len(ids))
        burst=live_burst.make_trace(trace_protocol,ids)
        steady=[dict(r,arrival_ns=i*round(settings['steady_gap_ms']*1e6)) for i,r in enumerate(burst)]
        plans.append(dict(session=session,orders=dict(burst=orders[session],steady=orders[5-session]),
                          traces=dict(burst=burst,steady=steady)))
    args.output.mkdir(parents=True)
    lock=dict(settings=settings,policies=policies,calibration=calibration,threshold=tau,plans=plans,
              preparation=provenance,head_sha256=experiment.digest(args.head),
              sources={n:experiment.digest(m.__file__) for n,m in [('study',__import__(__name__,fromlist=['main'])),
                        ('fifo',live_burst),('experiment',experiment),('early_exit',early_exit),('confidence',confidence_exit)]},
              feature_cache_sha256={s:experiment.digest(args.feature_cache/f'{s}.pt') for s in ('calibration','validation')},
              platform=platform.platform(),torch_version=str(torch.__version__),threads=1,
              primary='burst safeguarded_queue vs best_fixed; both one-sided 97.5% lower bounds must meet targets',
              policy_semantics='Queue-aware uses full path below backlog depth and calibrated confidence gate above it')
    (args.output/'lock.json').write_text(json.dumps(lock,indent=2,allow_nan=False))
    full,early,transform=early_exit.load_models(args.artifacts)
    baseline=torch.load(args.artifacts/'states.pt',weights_only=True)['512_fp32']
    data=CIFAR10(args.data,train=True)
    @torch.inference_mode()
    def predict(image,threshold):
        x=transform(image).unsqueeze(0)
        for layer in (full.conv1,full.bn1,full.relu,full.maxpool,full.layer1,full.layer2,full.layer3): x=layer(x)
        logits=early[8](early[7](x))@head['weight']+head['bias']
        margin=confidence_exit.margin(logits).item()
        early_prediction=logits.argmax(1).item()
        probability=logits.softmax(-1).max().item()
        exited=threshold is not None and margin>=threshold
        prediction=early_prediction if exited else experiment.classify(torch.flatten(full.avgpool(full.layer4(x)),1),baseline).argmax(1).item()
        return prediction,exited,dict(margin=margin,top_probability=probability,early_prediction=early_prediction)
    def timed(trace,order,folder):
        images=[data[r['image_id']][0] for r in trace]
        raw={}
        folder.mkdir(parents=True)
        for name in order:
            for i in range(settings['warmup']):
                predict(images[i%len(images)],0.);predict(images[i%len(images)],None)
            raw[name]=live_burst.run_fifo(trace,images,policies[name],predict)
            live_burst.validate_records(trace,raw[name])
            (folder/f'{name}.json').write_text(json.dumps(raw[name],indent=2,allow_nan=False))
        return raw
    # One live calibration sanity run per policy, using calibration images only; policies remain unchanged.
    cal_protocol=dict(base,seed=settings['seed'],requests=settings['calibration_requests'])
    cal_trace=live_burst.make_trace(cal_protocol,splits['calibration'])
    timed(cal_trace,names,args.output/'calibration-live')
    (args.output/'evaluation-frozen.json').write_text(json.dumps(dict(lock_sha256=experiment.digest(args.output/'lock.json'),
        live_calibration_completed=True,no_threshold_changes=True),indent=2))
    sessions=[]
    for plan in plans:
        raw_by_workload={}
        for workload in ('burst','steady'):
            raw_by_workload[workload]=timed(plan['traces'][workload],plan['orders'][workload],args.output/f"session-{plan['session']}"/workload)
        # Labels and counterfactual full outputs are joined only after every timed run in this session.
        logits,teacher,labels=cached('validation')
        lookup={image:i for i,image in enumerate(splits['validation'])}
        record=dict(session=plan['session'],metrics={},error_groups={})
        for workload,raw in raw_by_workload.items():
            record['metrics'][workload]={};record['error_groups'][workload]={}
            for name,rows in raw.items():
                for row in rows:
                    i=lookup[row['image_id']]
                    row['label']=int(labels[i]);row['full_prediction']=int(teacher[i].argmax())
                    expected_early=int(logits[i].argmax())
                    if expected_early != row['confidence']['early_prediction']: raise ValueError('Live/cache early prediction mismatch')
                    if row['prediction'] != (expected_early if row['early_exit'] else row['full_prediction']):
                        raise ValueError('Live/cache final prediction mismatch')
                folder=args.output/f"session-{plan['session']}"/workload
                (folder/f'{name}.json').write_text(json.dumps(rows,indent=2,allow_nan=False))
                record['metrics'][workload][name]=live_burst.summarize(plan['traces'][workload],rows,data.targets)
                record['error_groups'][workload][name]=error_groups(rows,settings['confidence_bins_margin'],settings['queue_bins'])
        sessions.append(record)
        (args.output/'sessions.json').write_text(json.dumps(sessions,indent=2,allow_nan=False))
        print('Completed session',plan['session'],json.dumps(record['metrics']),flush=True)
    comparisons=compare(sessions,settings)
    report=dict(status='six_session_exploratory_study',research_validated=False,comparisons=comparisons,
                frozen_threshold=tau,calibration=calibration,lock_sha256=experiment.digest(args.output/'lock.json'),
                evidence_sha256={str(f.relative_to(args.output)):experiment.digest(f) for f in args.output.rglob('*.json')},
                limitations=['Six balanced sessions on one host; bootstrap uncertainty remains unstable at this sample size.',
                             'Same images reused for burst and steady within a session; panels disjoint across sessions.',
                             'Cached features used only for offline calibration and post-run diagnostics, never timed inference.',
                             'Top softmax probability from ridge scores is logged but is not a calibrated probability.',
                             'Best fixed is historical calibration-selected; no exhaustive current-machine reoptimization.',
                             'Only the burst queue-vs-best-fixed comparison is primary; other intervals are descriptive.',
                             'The primary two endpoints use one-sided 97.5% lower bounds (Bonferroni); bootstrap is approximate.',
                             'Timing includes confidence diagnostics and scalar switching; no format switching/energy claim.',
                             'Gating only under pressure can save no more per-image compute than the same fixed confidence gate.'])
    (args.output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(comparisons),flush=True)


if __name__=='__main__': main()
