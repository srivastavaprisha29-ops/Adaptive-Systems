"""Compare a static ResNet-18 stage-three exit with the full FP32 baseline."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torchvision import datasets, models

from . import experiment


def load_models(artifacts):
    full = models.resnet18(weights=None).eval()
    full.fc = torch.nn.Identity()
    full.load_state_dict(torch.load(artifacts / 'encoder.pt', weights_only=True))
    early = torch.nn.Sequential(
        full.conv1, full.bn1, full.relu, full.maxpool,
        full.layer1, full.layer2, full.layer3, full.avgpool, torch.nn.Flatten(1),
    ).eval()
    transform = models.ResNet18_Weights.IMAGENET1K_V1.transforms()
    return full, early, transform


def prepare(args, config, dataset, splits, full, early, transform, baseline, provenance):
    labels = torch.tensor(dataset.targets)
    train = experiment.features(early, transform, dataset, splits['train'], 256)
    weight, bias = experiment.ridge_head(train, labels[splits['train']], config['ridge'])
    del train
    indices = splits['validation']
    reference = experiment.features(full, transform, dataset, indices)
    candidate = experiment.features(early, transform, dataset, indices, 256)
    correct_full = experiment.classify(reference, baseline).argmax(1).eq(labels[indices])
    correct_early = (candidate @ weight + bias).argmax(1).eq(labels[indices])
    delta, interval = experiment.paired_quality_interval(
        correct_full, correct_early, config['bootstrap_samples'], config['seed'],
    )
    quality = dict(examples=len(indices), full_accuracy_pct=correct_full.double().mean().item() * 100,
                   early_accuracy_pct=correct_early.double().mean().item() * 100,
                   delta_pp=delta, delta_95ci_pp=interval,
                   quality_pass=interval[0] >= -config['quality_tolerance_pp'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as output:
        torch.save(dict(weight=weight, bias=bias, quality=quality, provenance=provenance), output)
    print(json.dumps(quality, indent=2), flush=True)


def predict(model, transform, image, weight, bias):
    return (model(transform(image).unsqueeze(0)) @ weight + bias).argmax(1)


def benchmark(args, config, dataset, splits, full, early, transform, baseline, provenance):
    prepared = torch.load(args.head, weights_only=True)
    if prepared['provenance'] != provenance:
        raise ValueError('Early-exit head was prepared with different inputs or source')
    actions = {'full': (full, baseline['weight'], baseline['bias']),
               'stage3': (early, prepared['weight'], prepared['bias'])}
    sample_rng = np.random.default_rng(config['seed'])
    indices = sample_rng.permutation(splits['validation'])[:config['timed_examples']]
    if len(indices) != config['timed_examples']:
        raise ValueError('Too few validation examples for the timing configuration')
    images = [dataset[int(index)][0] for index in indices]
    order_rng = np.random.default_rng(args.order_seed)
    timings = {name: [] for name in actions}
    for _ in range(config['repeats']):
        order = order_rng.permutation(len(images))
        for name in order_rng.permutation(list(actions)):
            model, weight, bias = actions[name]
            for j in range(config['warmup']):
                predict(model, transform, images[order[j % len(images)]], weight, bias)
            times = []
            for j in order:
                start = time.perf_counter_ns()
                predict(model, transform, images[j], weight, bias)
                times.append((time.perf_counter_ns() - start) / 1e6)
            timings[name].append(times)
    means = {name: np.mean(values, axis=1).tolist() for name, values in timings.items()}
    gain, interval = experiment.paired_latency_interval(
        means['full'], means['stage3'], config['bootstrap_samples'], args.order_seed,
    )
    report = dict(quality=prepared['quality'], p50_ms={k: float(np.median(v)) for k, v in timings.items()},
                  repeat_means_ms=means, mean_latency_reduction_pct=gain, latency_95ci_pct=interval,
                  latency_pass=interval[0] >= config['minimum_latency_reduction_pct'],
                  timings_ms=timings, example_indices=indices.tolist(), order_seed=args.order_seed,
                  provenance=provenance, head_sha256=experiment.digest(args.head))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as output:
        json.dump(report, output, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k not in ('timings_ms', 'example_indices')}, indent=2))


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'benchmark'))
    parser.add_argument('--artifacts', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=Path('data'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--head', type=Path)
    parser.add_argument('--order-seed', type=int, default=29)
    args = parser.parse_args()
    if args.command == 'benchmark' and args.head is None:
        parser.error('benchmark requires --head')
    config = experiment.configuration(args.artifacts / 'config.json')
    torch.manual_seed(config['seed'])
    torch.set_num_threads(config['threads'])
    torch.use_deterministic_algorithms(True)
    experiment.verify_preparation(args.artifacts, args.data)
    provenance = {name: experiment.digest(args.artifacts / name)
                  for name in ('manifest.json', 'config.json', 'splits.json', 'states.pt', 'encoder.pt')}
    provenance['early_exit_source'] = experiment.digest(__file__)
    provenance['experiment_source'] = experiment.digest(experiment.__file__)
    full, early, transform = load_models(args.artifacts)
    splits = json.loads((args.artifacts / 'splits.json').read_text())
    dataset = datasets.CIFAR10(args.data, train=True)
    baseline = torch.load(args.artifacts / 'states.pt', weights_only=True)['512_fp32']
    action = prepare if args.command == 'prepare' else benchmark
    action(args, config, dataset, splits, full, early, transform, baseline, provenance)


if __name__ == '__main__':
    main()
