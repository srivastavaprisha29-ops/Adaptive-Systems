"""Audit legacy controller evidence without interpreting request timings as IID trials."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import zipfile

from verify_queue_results import quantile, verify_run


def paired_accuracy(reference, candidate, samples=2000, seed=29):
    if not reference or len(reference) != len(candidate) or samples < 1:
        raise ValueError('Need equal nonempty paired panels and positive resamples')
    differences = [int(b) - int(a) for a, b in zip(reference, candidate)]
    rng = random.Random(seed)
    draws = [100 * sum(rng.choices(differences, k=len(differences))) / len(differences)
             for _ in range(samples)]
    return dict(delta_pp=100 * statistics.mean(differences),
                interval_95_pp=[quantile(draws, .025), quantile(draws, .975)])


def audit(path, samples=2000, seed=29, tolerance=1.0):
    if not math.isfinite(tolerance) or tolerance < 0 or samples < 1:
        raise ValueError('Use a finite nonnegative tolerance and positive resamples')
    seen = set()
    rows = []
    with zipfile.ZipFile(path) as archive:
        lock_bytes = archive.read('lock.json')
        lock = json.loads(lock_bytes)
        lock_sha = hashlib.sha256(lock_bytes).hexdigest()
        for session in range(3):
            trace = json.loads(archive.read(f'session-{session}.json'))
            if trace['lock_sha256'] != lock_sha or trace['session'] != session:
                raise ValueError('Session identity or policy lock checksum mismatch')
            indices = trace['indices']
            if not indices or len(set(indices)) != len(indices) or seen.intersection(indices):
                raise ValueError('Image panels overlap or contain duplicate indices')
            seen.update(indices)
            if set(trace['results']) != set(lock['workloads']):
                raise ValueError('Workload panel does not match lock')
            for workload, runs in sorted(trace['results'].items()):
                metrics = {name: verify_run(run, len(indices), f'{session}/{workload}/{name}')
                           for name, run in runs.items()}
                full, adaptive, fixed = (runs[name] for name in ('full', 'queue_aware', 'best_fixed'))
                comparisons = {}
                for name in ('full', 'best_fixed'):
                    comparisons[name] = paired_accuracy([r[0] for r in runs[name]['requests']],
                                                       [r[0] for r in adaptive['requests']], samples, seed)
                fixed_p95 = metrics['best_fixed']['p95_ms']
                if fixed_p95 <= 0:
                    raise ValueError('Cannot compare nonpositive fixed-policy p95')
                delta = comparisons['full']['delta_pp']
                lower = comparisons['full']['interval_95_pp'][0]
                rows.append(dict(session=session, workload=workload, examples=len(indices),
                                 paired_accuracy=comparisons,
                                 point_quality_pass=delta >= -tolerance - 1e-10,
                                 interval_quality_pass=lower >= -tolerance - 1e-10,
                                 p95_reduction_vs_saved_best_fixed_pct=100*(1-metrics['queue_aware']['p95_ms']/fixed_p95),
                                 summaries=metrics,
                                 fixed_record_alias=runs.get('matched_fixed', {}).get('requests') == fixed['requests']))
    bursts = [r for r in rows if r['workload'].startswith('bursty')]
    return dict(schema_version=1, status='broader_controller_not_validated',
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                matrix_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                lock_sha256=lock_sha, seed=seed, bootstrap_samples=samples,
                quality_tolerance_pp=tolerance, unique_image_examples=len(seen),
                sessions=3, workloads_per_session=len(lock['workloads']),
                burst_summary=dict(runs=len(bursts),
                    positive_p95_point_estimates=sum(r['p95_reduction_vs_saved_best_fixed_pct'] > 0 for r in bursts),
                    point_quality_passes=sum(r['point_quality_pass'] for r in bursts),
                    interval_quality_passes=sum(r['interval_quality_pass'] for r in bursts)),
                assumptions=['Request row order is assumed to follow the shared session image-index list.',
                             'Image bootstrap describes conditional prediction uncertainty, not dependent queue latency.',
                             'No independent request-level latency bootstrap is reported.',
                             'Accuracy intervals are per comparison and unadjusted for multiple comparisons.',
                             'Workloads reuse images within each session and must not be pooled as new images.',
                             'Saved best_fixed is the historical calibration-selected comparator; global optimality is not established.'],
                unresolved=['Original live queue generator source is missing.',
                            'Legacy records do not identify monitoring, decision and switching timing boundaries.',
                            'Three sessions do not establish robust cross-machine latency uncertainty.',
                            'No held-out dynamic-controller test with a predeclared joint quality/latency gate.',
                            'Width-plus-precision adaptation, energy and peak memory claims remain unvalidated.'],
                results=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output path')
    report = audit(args.matrix)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as output:
        json.dump(report, output, indent=2, allow_nan=False)
    print(json.dumps(report['burst_summary']))
    print(report['status'])


if __name__ == '__main__':
    main()
