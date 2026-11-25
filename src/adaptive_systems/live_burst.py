"""Single-server live FIFO burst benchmark with locked policies and arrival traces."""
import argparse
from bisect import bisect_right
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import statistics
import time


def sha_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def quantile(values, probability):
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def validate_protocol(p):
    for key in ('requests', 'period_requests', 'fast_requests', 'warmup'):
        if type(p[key]) is not int or p[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if not 0 < p['fast_requests'] < p['period_requests']:
        raise ValueError('A period needs both burst and recovery arrivals')
    for key in ('fast_gap_ms', 'slow_gap_ms'):
        if not math.isfinite(p[key]) or p[key] <= 0:
            raise ValueError('Arrival gaps must be finite and positive')
    if set(p['policies']) != {'best_fixed', 'queue_aware'}:
        raise ValueError('Expected paired fixed and queue-aware policies')
    for policy in p['policies'].values():
        if (type(policy['depth']) is not int or policy['depth'] < 1
                or len(policy['thresholds']) != 2
                or any(not math.isfinite(v) or v < 0 for v in policy['thresholds'])):
            raise ValueError('Invalid policy parameters')
    fixed = p['policies']['best_fixed']['thresholds']
    dynamic = p['policies']['queue_aware']['thresholds']
    if fixed[0] != fixed[1] or dynamic[0] != fixed[0] or dynamic[1] > dynamic[0]:
        raise ValueError('Require a fixed comparator and a relaxed-on-backlog controller')


def make_trace(p, indices):
    validate_protocol(p)
    if len(set(indices)) != len(indices) or len(indices) < p['requests']:
        raise ValueError('Need enough unique evaluation images')
    image_ids = random.Random(p['seed']).sample(list(indices), p['requests'])
    arrivals = [0]
    for i in range(1, len(image_ids)):
        gap = p['fast_gap_ms'] if (i-1) % p['period_requests'] < p['fast_requests'] else p['slow_gap_ms']
        arrivals.append(arrivals[-1] + round(gap * 1_000_000))
    return [dict(request_id=i, image_id=image_id, arrival_ns=arrival)
            for i, (image_id, arrival) in enumerate(zip(image_ids, arrivals))]


def run_fifo(trace, images, policy, predict, *, clock=time.perf_counter_ns, sleep=time.sleep):
    """Policy sees only arrived pending work. Labels are unavailable in this path."""
    arrivals = [r['arrival_ns'] for r in trace]
    if (len(images) != len(trace) or not trace or arrivals[0] != 0
            or any(b <= a for a, b in zip(arrivals, arrivals[1:]))):
        raise ValueError('Need aligned images and strictly increasing relative arrivals')
    rows = []
    origin = clock()
    previous_mode = 0  # Policy state reset between trials.
    for i, (request, image) in enumerate(zip(trace, images)):
        scheduled = origin + request['arrival_ns']
        now = clock()
        while now < scheduled:
            sleep((scheduled - now) / 1e9)
            now = clock()
        start = now
        monitor_start = clock()
        waiting = bisect_right(arrivals, monitor_start - origin, lo=i+1) - i - 1
        monitor_end = clock()
        mode = int(policy['thresholds'][0] != policy['thresholds'][1] and waiting >= policy['depth'])
        threshold = policy['thresholds'][mode]
        switch_start = clock()
        changed = mode != previous_mode
        previous_mode = mode
        decision_end = clock()
        result = predict(image, threshold)
        prediction, exited = result[:2]
        diagnostics = result[2] if len(result) == 3 else {}
        prediction = int(prediction)
        exited = bool(exited)
        finish = clock()  # Completed scalar prediction, before logging or label lookup.
        rows.append(dict(**request, start_ns=start-origin, finish_ns=finish-origin,
                         wait_ns=start-scheduled, service_ns=finish-start,
                         response_ns=finish-scheduled, pending_depth=waiting,
                         monitor_ns=monitor_end-monitor_start,
                         decision_ns=decision_end-monitor_end,
                         state_update_ns=decision_end-switch_start,
                         threshold=threshold, relaxed=bool(mode), state_changed=changed,
                         prediction=prediction, early_exit=exited, confidence=diagnostics))
    return rows


def validate_records(trace, rows):
    if len(trace) != len(rows):
        raise ValueError('Missing or extra request records')
    last_finish = 0
    for request, row in zip(trace, rows):
        if any(row[k] != request[k] for k in ('request_id', 'image_id', 'arrival_ns')):
            raise ValueError('Trace identity mismatch')
        if not row['arrival_ns'] <= row['start_ns'] <= row['finish_ns'] or row['start_ns'] < last_finish:
            raise ValueError('Invalid FIFO timing')
        if (row['wait_ns'] != row['start_ns']-row['arrival_ns']
                or row['service_ns'] != row['finish_ns']-row['start_ns']
                or row['response_ns'] != row['wait_ns']+row['service_ns']):
            raise ValueError('Response boundary mismatch')
        if min(row[k] for k in ('pending_depth', 'monitor_ns', 'decision_ns', 'state_update_ns')) < 0:
            raise ValueError('Negative diagnostic')
        last_finish = row['finish_ns']


def summarize(trace, rows, labels):
    validate_records(trace, rows)
    correct = [row['prediction'] == labels[row['image_id']] for row in rows]
    return dict(requests=len(rows), accuracy_pct=100*statistics.mean(correct),
                p95_response_ms=quantile([r['response_ns']/1e6 for r in rows], .95),
                p50_response_ms=quantile([r['response_ns']/1e6 for r in rows], .5),
                p95_wait_ms=quantile([r['wait_ns']/1e6 for r in rows], .95),
                mean_service_ms=statistics.mean(r['service_ns']/1e6 for r in rows),
                mean_monitor_us=statistics.mean(r['monitor_ns']/1e3 for r in rows),
                mean_decision_us=statistics.mean(r['decision_ns']/1e3 for r in rows),
                max_pending_depth=max(r['pending_depth'] for r in rows),
                policy_state_changes=sum(r['state_changed'] for r in rows),
                exit_pct=100*statistics.mean(r['early_exit'] for r in rows))


def main():
    import torch
    from torchvision.datasets import CIFAR10
    from . import confidence_exit, early_exit, experiment

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path, required=True)
    parser.add_argument('--head', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, default=Path('configs/burst-smoke.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh output directory')
    protocol = json.loads(args.protocol.read_text())
    validate_protocol(protocol)
    config = experiment.configuration(args.artifacts/'config.json')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    experiment.verify_preparation(args.artifacts, args.data)
    head = torch.load(args.head, weights_only=True, map_location='cpu')
    provenance = {n: experiment.digest(args.artifacts/n) for n in
                  ('manifest.json', 'config.json', 'splits.json', 'states.pt', 'encoder.pt')}
    provenance.update(early_exit_source=experiment.digest(early_exit.__file__),
                      experiment_source=experiment.digest(experiment.__file__))
    if head['provenance'] != provenance:
        raise ValueError('Head preparation or source mismatch')
    if (protocol['identity']['preparation'] != provenance['manifest.json']
            or protocol['identity']['head'] != experiment.digest(args.head)):
        raise ValueError('Historical policy lock does not match supplied checkpoints')
    splits = json.loads((args.artifacts/'splits.json').read_text())
    trace = make_trace(protocol, splits['validation'])
    if set(r['image_id'] for r in trace).intersection(splits['calibration']):
        raise ValueError('Evaluation images overlap calibration')
    full, early, transform = early_exit.load_models(args.artifacts)
    baseline = torch.load(args.artifacts/'states.pt', weights_only=True)['512_fp32']
    dataset = CIFAR10(args.data, train=True)
    images = [dataset[r['image_id']][0] for r in trace]
    order = list(protocol['policies'])
    random.Random(protocol['seed']).shuffle(order)
    args.output.mkdir(parents=True)
    lock = dict(protocol=protocol, trace=trace, order=order, preparation=provenance,
                source_sha256=experiment.digest(__file__),
                prediction_source_sha256=experiment.digest(confidence_exit.__file__),
                head_sha256=experiment.digest(args.head),
                platform=platform.platform(), processor=platform.processor(),
                torch_version=str(torch.__version__), threads=1)
    (args.output/'lock.json').write_text(json.dumps(lock, indent=2, allow_nan=False))
    raw = {}
    with torch.inference_mode():
        def predict(image, threshold):
            pred, exited = confidence_exit.predict_gated(full, early, transform, image, head, baseline, threshold)
            return pred.item(), exited
        for name in order:
            policy = protocol['policies'][name]
            for i in range(protocol['warmup']):
                # Warm both the early and fallback branches immediately before each trial.
                predict(images[i % len(images)], 0.)
                predict(images[i % len(images)], float('inf'))
            raw[name] = run_fifo(trace, images, policy, predict)
            validate_records(trace, raw[name])
            (args.output/f'{name}.json').write_text(json.dumps(raw[name], indent=2, allow_nan=False))
            print('Completed', name, len(raw[name]), flush=True)
        # Reference predictions and all scoring occur after both timed policy runs.
        reference = [early_exit.predict(full, transform, image, baseline['weight'], baseline['bias']).item()
                     for image in images]
    labels = dataset.targets
    metrics = {name: summarize(trace, rows, labels) for name, rows in raw.items()}
    fixed, adaptive = metrics['best_fixed'], metrics['queue_aware']
    delta = adaptive['accuracy_pct'] - fixed['accuracy_pct']
    reduction = 100*(1-adaptive['p95_response_ms']/fixed['p95_response_ms'])
    full_accuracy = 100*statistics.mean(pred == labels[r['image_id']] for pred, r in zip(reference, trace))
    report = dict(status='single_session_benchmark_check', research_validated=False,
                  trace_sha256=sha_json(trace), lock_sha256=experiment.digest(args.output/'lock.json'),
                  raw_sha256={n: experiment.digest(args.output/f'{n}.json') for n in raw},
                  summaries=metrics, full_reference_accuracy_pct=full_accuracy,
                  comparison=dict(p95_reduction_pct=reduction, accuracy_delta_vs_fixed_pp=delta,
                                  accuracy_delta_vs_full_pp=adaptive['accuracy_pct']-full_accuracy,
                                  point_target_met=reduction >= protocol['p95_target_pct'] and delta >= -protocol['quality_tolerance_pp']),
                  full_reference_predictions=reference, labels_in_trace_order=[labels[r['image_id']] for r in trace],
                  limitations=['One paired session, no independent-session latency confidence interval.',
                               'Historical calibration-selected fixed comparator, not newly optimized for this hardware.',
                               'Validation images have been used in previous studies.',
                               'Input images preloaded; image decoding, disk and network excluded. Preprocessing is timed.',
                               'Switching changes a scalar threshold; no model-loading or format-switching claim.',
                               'Sequential policy runs in randomized order; host contention can differ.',
                               'No drops or queue capacity limit; drain every scheduled request.',
                               'Reconstructed workload and runner, not an exact rerun of the missing historical generator.'])
    (args.output/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report['comparison']), flush=True)


if __name__ == '__main__':
    main()
