"""Verify saved live-burst records and recompute metrics without ML inference."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from adaptive_systems.live_burst import make_trace, sha_json, summarize


def verify(directory):
    lock = json.loads((directory/'lock.json').read_text())
    report = json.loads((directory/'summary.json').read_text())
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    if digest(directory/'lock.json') != report['lock_sha256'] or sha_json(lock['trace']) != report['trace_sha256']:
        raise ValueError('Lock or trace checksum mismatch')
    trace = lock['trace']
    ids = [row['image_id'] for row in trace]
    if len(ids) != len(set(ids)) or len(trace) != lock['protocol']['requests']:
        raise ValueError('Duplicate or missing image IDs')
    expected_trace = make_trace(lock['protocol'], list(range(len(trace))))
    if ([r['arrival_ns'] for r in trace] != [r['arrival_ns'] for r in expected_trace]
            or [r['request_id'] for r in trace] != list(range(len(trace)))):
        raise ValueError('Arrival waveform or request IDs do not match protocol')
    labels = dict(zip(ids, report['labels_in_trace_order']))
    if len(labels) != len(trace) or len(report['full_reference_predictions']) != len(trace):
        raise ValueError('Unaligned scoring data')
    for name, expected in report['summaries'].items():
        path = directory/f'{name}.json'
        if digest(path) != report['raw_sha256'][name]:
            raise ValueError('Raw record checksum mismatch')
        rows = json.loads(path.read_text())
        actual = summarize(trace, rows, labels)
        policy = lock['protocol']['policies'][name]
        previous = False
        for row in rows:
            relaxed = policy['thresholds'][0] != policy['thresholds'][1] and row['pending_depth'] >= policy['depth']
            if (row['threshold'] != policy['thresholds'][int(relaxed)] or row['relaxed'] != relaxed
                    or row['state_changed'] != (relaxed != previous)):
                raise ValueError('Policy decision mismatch')
            if row['monitor_ns']+row['decision_ns'] > row['service_ns'] or row['state_update_ns'] > row['decision_ns']:
                raise ValueError('Invalid nested overhead accounting')
            previous = relaxed
        for key, value in actual.items():
            if not math.isclose(value, expected[key], abs_tol=1e-9, rel_tol=1e-12):
                raise ValueError(f'Summary mismatch: {name}/{key}')
    fixed, adaptive = (report['summaries'][name] for name in ('best_fixed', 'queue_aware'))
    comparison = report['comparison']
    expected_gain = 100*(1-adaptive['p95_response_ms']/fixed['p95_response_ms'])
    if not math.isclose(comparison['p95_reduction_pct'], expected_gain, abs_tol=1e-9):
        raise ValueError('Comparison mismatch')
    reference = 100*statistics.mean(pred == labels[r['image_id']] for pred,r in zip(report['full_reference_predictions'],trace))
    if reference != report['full_reference_accuracy_pct']:
        raise ValueError('Full reference mismatch')
    delta = adaptive['accuracy_pct']-fixed['accuracy_pct']
    if (comparison['accuracy_delta_vs_fixed_pp'] != delta
            or comparison['accuracy_delta_vs_full_pp'] != adaptive['accuracy_pct']-reference):
        raise ValueError('Accuracy comparison mismatch')
    point = expected_gain >= lock['protocol']['p95_target_pct'] and delta >= -lock['protocol']['quality_tolerance_pp']
    if comparison['point_target_met'] != point:
        raise ValueError('Point target flag mismatch')
    if report['research_validated']:
        raise ValueError('A smoke session cannot validate research superiority')
    return dict(verified_requests=2*len(trace), trace_sha256=report['trace_sha256'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    print(json.dumps(verify(parser.parse_args().run)))
