"""Controlled replay with an always-active default and calibration-bounded relaxation."""
import argparse
from bisect import bisect_right
import itertools
import json
from pathlib import Path
import platform
import random
import statistics
import time
import resource

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


def calibrate_limit(logits,full_logits,labels,default,best_fixed,config):
    import torch
    from . import confidence_exit,experiment
    margins=confidence_exit.margin(logits)
    early=logits.argmax(1).eq(labels)
    full=full_logits.argmax(1).eq(labels)
    default_exits=margins>=default
    reference=torch.where(margins>=best_fixed,early,full)
    grid=sorted({default,best_fixed,*[float(v) for v in torch.quantile(margins,torch.linspace(0,1,41)) if v<=default]})
    candidates=[]
    for cutoff in grid:
        if cutoff>default: continue
        band=(margins>=cutoff)&~default_exits
        # Offline adverse case: new exits occur only where early is wrong and full is right.
        adverse_exits=default_exits|(band&~early&full)
        candidate=torch.where(adverse_exits,early,full)
        delta,ci=experiment.paired_quality_interval(reference,candidate,config['bootstrap_samples'],config['seed'])
        candidates.append(dict(cutoff=cutoff,newly_eligible=int(band.sum()),
            harmful_newly_eligible=int((band&~early&full).sum()),delta_pp=delta,interval_95_pp=ci,
            qualified=ci[0]>=-config['quality_tolerance_pp']))
    qualified=[c for c in candidates if c['qualified']]
    if not qualified: raise ValueError('Even the default gate fails calibration relative to best fixed')
    return qualified[0]['cutoff'],candidates


def adoption_bounds(sessions,settings):
    primary=compare(sessions,settings)['burst']
    bounds={}
    for reference in ('best_fixed','fixed_confidence'):
        row=primary[f'safeguarded_queue_vs_{reference}']
        for field,values in [('gain',row['per_session_gain_pct']),('quality',row['per_session_accuracy_delta_pp'])]:
            rng=random.Random(settings['seed'])
            draws=[statistics.mean(rng.choices(values,k=len(values))) for _ in range(settings['bootstrap_samples'])]
            bounds[f'{reference}_{field}_lower']=live_burst.quantile(draws,.0125)
    passed=(bounds['best_fixed_gain_lower']>=settings['p95_target_pct'] and
            bounds['fixed_confidence_gain_lower']>0 and
            bounds['best_fixed_quality_lower']>=-settings['quality_tolerance_pp'] and
            bounds['fixed_confidence_quality_lower']>=-settings['quality_tolerance_pp'])
    return dict(one_sided_confidence_pct=98.75,lower_bounds=bounds,queue_awareness_supported=passed)


def main():
    import torch
    from torchvision.datasets import CIFAR10
    from . import confidence_exit, early_exit, experiment

    p = argparse.ArgumentParser(description=__doc__)
    for name in ('artifacts','head','data','feature_cache','prior_run','output'):
        p.add_argument('--'+name.replace('_','-'), type=Path, required=True)
    p.add_argument('--settings',type=Path,default=Path('configs/relaxation-study.json'))
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
    limit,candidates=calibrate_limit(cal,cal_full,cal_labels,tau,strict,config)
    policies=dict(best_fixed=dict(thresholds=[strict,strict],depth=1),
                  fixed_confidence=dict(thresholds=[tau,tau],depth=1),
                  safeguarded_queue=dict(thresholds=[tau,limit],depth=settings['queue_depth']))
    names=list(policies)
    prior=json.loads((args.prior_run/'lock.json').read_text())
    if prior['preparation'] != provenance or prior['head_sha256'] != experiment.digest(args.head):
        raise ValueError('Prior study uses different model inputs')
    plans=prior['plans']
    if len(plans)!=settings['sessions'] or any(len(p['traces']['burst'])!=settings['requests'] for p in plans):
        raise ValueError('Prior traces do not match the fixed replay settings')
    args.output.mkdir(parents=True)
    lock=dict(settings=settings,policies=policies,calibration=calibration,threshold=tau,relaxation_limit=limit,calibration_candidates=candidates,plans=plans,
              prior_lock_sha256=experiment.digest(args.prior_run/'lock.json'),
              preparation=provenance,head_sha256=experiment.digest(args.head),
              sources={n:experiment.digest(m.__file__) for n,m in [('study',__import__(__name__,fromlist=['main'])),
                        ('fifo',live_burst),('experiment',experiment),('early_exit',early_exit),('confidence',confidence_exit)]},
              feature_cache_sha256={s:experiment.digest(args.feature_cache/f'{s}.pt') for s in ('calibration','validation')},
              platform=platform.platform(),torch_version=str(torch.__version__),threads=1,
              primary='Burst adoption requires four one-sided 98.75% bounds: gains >=5% vs best fixed and >0% vs always-active gate, each quality delta >=-1pp',
              policy_semantics='Always-active default threshold, relaxed only to the calibration-bounded limit under queue pressure')
    (args.output/'lock.json').write_text(json.dumps(lock,indent=2,allow_nan=False))
    full,early,transform=early_exit.load_models(args.artifacts)
    baseline=torch.load(args.artifacts/'states.pt',weights_only=True)['512_fp32']
    data=CIFAR10(args.data,train=True)
    @torch.inference_mode()
    def predict(image,threshold):
        cpu_start=time.thread_time_ns()
        x=transform(image).unsqueeze(0)
        for layer in (full.conv1,full.bn1,full.relu,full.maxpool,full.layer1,full.layer2,full.layer3): x=layer(x)
        logits=early[8](early[7](x))@head['weight']+head['bias']
        margin=confidence_exit.margin(logits).item()
        early_prediction=logits.argmax(1).item()
        probability=logits.softmax(-1).max().item()
        exited=threshold is not None and margin>=threshold
        prediction=early_prediction if exited else experiment.classify(torch.flatten(full.avgpool(full.layer4(x)),1),baseline).argmax(1).item()
        return prediction,exited,dict(margin=margin,top_probability=probability,early_prediction=early_prediction,inference_thread_cpu_ns=time.thread_time_ns()-cpu_start)
    def timed(trace,order,folder,selected_policies=None):
        selected_policies=policies if selected_policies is None else selected_policies
        images=[data[r['image_id']][0] for r in trace]
        raw={}
        folder.mkdir(parents=True)
        for name in order:
            for i in range(settings['warmup']):
                predict(images[i%len(images)],0.);predict(images[i%len(images)],None)
            before=resource.getrusage(resource.RUSAGE_SELF)
            raw[name]=live_burst.run_fifo(trace,images,selected_policies[name],predict)
            after=resource.getrusage(resource.RUSAGE_SELF)
            (folder/f'{name}-host.json').write_text(json.dumps(dict(
                voluntary_context_switches=after.ru_nvcsw-before.ru_nvcsw,
                involuntary_context_switches=after.ru_nivcsw-before.ru_nivcsw),indent=2))
            live_burst.validate_records(trace,raw[name])
            (folder/f'{name}.json').write_text(json.dumps(raw[name],indent=2,allow_nan=False))
        return raw
    # Calibration-only live sanity runs precede the frozen evaluation; no evaluation-driven tuning.
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
    # Rerun the exact anomalous steady panel with the ORIGINAL three policies to isolate recurrence.
    anomaly_plan=prior['plans'][settings['anomaly_session']]
    anomaly_trace=anomaly_plan['traces']['steady']
    original_raw=timed(anomaly_trace,anomaly_plan['orders']['steady'],args.output/'anomaly-original-replay',prior['policies'])
    anomaly_metrics={n:live_burst.summarize(anomaly_trace,rows,data.targets) for n,rows in original_raw.items()}
    (args.output/'anomaly-summary.json').write_text(json.dumps(dict(session=settings['anomaly_session'],
        original_policy_replay=True,policies=prior['policies'],metrics=anomaly_metrics),indent=2))
    comparisons=compare(sessions,settings)
    adoption=adoption_bounds(sessions,settings)
    report=dict(status='six_session_controlled_relaxation_replay',research_validated=False,comparisons=comparisons,adoption=adoption,anomaly_replay=anomaly_metrics,
                frozen_threshold=tau,relaxation_limit=limit,calibration=calibration,lock_sha256=experiment.digest(args.output/'lock.json'),
                evidence_sha256={str(f.relative_to(args.output)):experiment.digest(f) for f in args.output.rglob('*.json')},
                limitations=['Six balanced sessions on one host; bootstrap uncertainty remains unstable at this sample size.',
                             'Same images reused for burst and steady within a session; panels disjoint across sessions.',
                             'Cached features used only for offline calibration and post-run diagnostics, never timed inference.',
                             'Top softmax probability from ridge scores is logged but is not a calibrated probability.',
                             'Best fixed is historical calibration-selected; no exhaustive current-machine reoptimization.',
                             'All evaluation images and arrivals deliberately replay the previously studied panels.',
                             'Adoption uses four Bonferroni one-sided 98.75% bounds; six-session bootstrap is approximate.',
                             'Timing includes confidence diagnostics and scalar switching; no format switching/energy claim.',
                             'The calibration limit uses an adverse routing subset relative to best fixed, not a formal out-of-sample guarantee.'])
    (args.output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(comparisons),flush=True)


if __name__=='__main__': main()
