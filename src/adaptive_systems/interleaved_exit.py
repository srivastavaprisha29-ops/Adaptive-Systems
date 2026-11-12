"""Time a frozen confidence gate with randomized, interleaved image pairs."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

import torch
from torchvision.datasets import CIFAR10

from . import confidence_exit, early_exit, experiment


def benchmark(full, early, transform, dataset, indices, head, baseline, threshold, config, seed, *, gated_predict=confidence_exit.predict_gated):
    rng = np.random.default_rng(seed)
    indices = np.random.default_rng(config['seed']).permutation(indices)[:config['timed_examples']]
    if len(indices) != config['timed_examples']:
        raise ValueError('Too few validation images for timing')
    images = [dataset[int(index)][0] for index in indices]
    paths = {
        'full': lambda image: early_exit.predict(full, transform, image, baseline['weight'], baseline['bias']),
        'gated': lambda image: gated_predict(full, early, transform, image, head, baseline, threshold),
    }
    wall = {name: [] for name in paths}
    cpu = {name: [] for name in paths}
    exits = 0
    for _ in range(config['repeats']):
        for name in paths:
            wall[name].append([])
            cpu[name].append([])
            for j in range(config['warmup']):
                paths[name](images[j % len(images)])
        order = rng.permutation(len(images))
        full_first = np.arange(len(images)) % 2 == 0
        rng.shuffle(full_first)
        for j, full_goes_first in zip(order, full_first):
            image = images[j]
            names = ('full', 'gated') if full_goes_first else ('gated', 'full')
            for name in names:
                predict = paths[name]
                wall_start = time.perf_counter_ns()
                cpu_start = time.thread_time_ns()
                result = predict(image)
                cpu_elapsed = time.thread_time_ns() - cpu_start
                wall_elapsed = time.perf_counter_ns() - wall_start
                cpu[name][-1].append(cpu_elapsed / 1e6)
                wall[name][-1].append(wall_elapsed / 1e6)
                if name == 'gated':
                    exits += int(result[1])
    report = {'seed': seed, 'indices': indices.tolist(),
              'timed_exit_pct': 100 * exits / (len(images) * config['repeats']),
              'protocol': 'balanced randomized order within image pairs; no samples removed'}
    for name, timings in [('wall', wall), ('thread_cpu', cpu)]:
        means = {key: np.mean(groups, axis=1).tolist() for key, groups in timings.items()}
        gain, interval = experiment.paired_latency_interval(
            means['full'], means['gated'], config['bootstrap_samples'], seed,
        )
        report[name] = {'p50_ms': {key: float(np.median(groups)) for key, groups in timings.items()},
                        'repeat_means_ms': means, 'mean_reduction_pct': gain,
                        'latency_95ci_pct': interval, 'timings_ms': timings}
    report['latency_pass'] = report['wall']['latency_95ci_pct'][0] >= config['minimum_latency_reduction_pct']
    return report


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path, required=True)
    parser.add_argument('--head', type=Path, required=True)
    parser.add_argument('--validation', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=Path('data'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=33)
    args = parser.parse_args()
    locked = json.loads(args.validation.read_text())
    config = experiment.configuration(args.artifacts / 'config.json')
    torch.set_num_threads(config['threads'])
    torch.use_deterministic_algorithms(True)
    experiment.verify_preparation(args.artifacts, args.data)
    provenance = {name: experiment.digest(args.artifacts / name)
                  for name in ('manifest.json', 'config.json', 'splits.json', 'states.pt', 'encoder.pt')}
    provenance.update(early_exit_source=experiment.digest(early_exit.__file__),
                      experiment_source=experiment.digest(experiment.__file__))
    head = torch.load(args.head, weights_only=True)
    if (provenance != locked['provenance'] or provenance != head['provenance']
            or locked['head_sha256'] != experiment.digest(args.head)
            or locked['gate_source_sha256'] != experiment.digest(confidence_exit.__file__)):
        raise ValueError('Frozen validation report does not match the experiment inputs or code')
    threshold = locked['threshold']
    full, early, transform = early_exit.load_models(args.artifacts)
    baseline = torch.load(args.artifacts / 'states.pt', weights_only=True)['512_fp32']
    dataset = CIFAR10(args.data, train=True)
    splits = json.loads((args.artifacts / 'splits.json').read_text())
    result = benchmark(full, early, transform, dataset, splits['validation'], head, baseline,
                       float('inf') if threshold is None else threshold, config, args.seed)
    report = dict(threshold=threshold, validation_sha256=experiment.digest(args.validation),
                  benchmark_source_sha256=experiment.digest(__file__), result=result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as output:
        json.dump(report, output, indent=2)
    print(f'Timing report saved to {args.output}')


if __name__ == '__main__':
    main()
