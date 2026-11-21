"""Recompute stored FIFO metrics from request records with no ML dependencies."""

import argparse
import json
import math
import statistics
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def quantile(values, probability):
    values = sorted(values)
    location = (len(values) - 1) * probability
    low = math.floor(location)
    high = math.ceil(location)
    return values[low] + (values[high] - values[low]) * (location - low)


def verify_run(run, expected_count, label):
    rows = run['requests']
    if len(rows) != expected_count:
        raise ValueError(f'{label}: request count mismatch')
    for row in rows:
        if (len(row) != 5 or row[0] not in (0, 1) or row[1] not in (0, 1)
                or row[2] < 0 or row[2] != int(row[2])
                or any(not math.isfinite(value) or value < 0 for value in row[3:])):
            raise ValueError(f'{label}: invalid request record')
    metrics = dict(
        accuracy_pct=statistics.mean(float(row[0]) for row in rows) * 100,
        exit_pct=statistics.mean(float(row[1]) for row in rows) * 100,
        p95_ms=quantile([row[3] + row[4] for row in rows], .95),
        p95_wait_ms=quantile([row[3] for row in rows], .95),
    )
    for name, actual in metrics.items():
        if not math.isclose(actual, run[name], rel_tol=1e-10, abs_tol=1e-8):
            raise ValueError(f'{label}: {name} mismatch')
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    evidence = parser.parse_args().evidence
    live = json.loads((evidence / 'Adaptive_Systems_Queue_Live.json').read_text())
    verified = 0
    for workload, trace in live['results'].items():
        for name, run in trace['runs'].items():
            verify_run(run, len(trace['indices']), f'initial/{workload}/{name}')
            verified += 1

    summary = []
    seen = set()
    with zipfile.ZipFile(evidence / 'Adaptive_Systems_Queue_Matrix.zip') as archive:
        for session in range(3):
            trace = json.loads(archive.read(f'session-{session}.json'))
            indices = trace['indices']
            if len(set(indices)) != len(indices) or seen.intersection(indices):
                raise ValueError('Matrix image panels overlap or contain duplicates')
            seen.update(indices)
            for workload, runs in sorted(trace['results'].items()):
                values = {name: verify_run(run, len(indices), f'{session}/{workload}/{name}')
                          for name, run in runs.items()}
                verified += len(values)
                adaptive, fixed, full = (values[name] for name in ('queue_aware', 'best_fixed', 'full'))
                summary.append(dict(session=session, workload=workload,
                                    p95_reduction_pct=100 * (1 - adaptive['p95_ms'] / fixed['p95_ms']),
                                    accuracy_delta_pp=adaptive['accuracy_pct'] - full['accuracy_pct'],
                                    accuracy_pass=adaptive['accuracy_pct'] - full['accuracy_pct'] >= -1 - 1e-8))
    bursts = [row for row in summary if row['workload'].startswith('bursty')]
    print(f'Verified {verified} stored policy/workload summaries.')
    print(f'Bursty p95 improved in {sum(row["p95_reduction_pct"] > 0 for row in bursts)}/{len(bursts)} runs.')
    print(f'Bursty accuracy budget met in {sum(row["accuracy_pass"] for row in bursts)}/{len(bursts)} runs.')
    print('Duplicate best-fixed and matched-fixed records count as one execution when interpreting runs.')


if __name__ == '__main__':
    main()
