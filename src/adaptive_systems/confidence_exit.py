"""Calibrate and evaluate a confidence-gated ResNet-18 stage-three exit."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torchvision import datasets

from . import early_exit, experiment


def margin(logits):
    top = logits.topk(2, dim=-1).values
    return top[..., 0] - top[..., 1]


def predictions(full, early, transform, dataset, indices, head, baseline):
    early_features = experiment.features(early, transform, dataset, indices, 256)
    early_logits = early_features @ head['weight'] + head['bias']
    full_features = experiment.features(full, transform, dataset, indices)
    full_pred = experiment.classify(full_features, baseline).argmax(1)
    labels = torch.tensor(dataset.targets)[indices]
    return margin(early_logits), early_logits.argmax(1).eq(labels), full_pred.eq(labels)


def quality(scores, early_correct, full_correct, threshold, config):
    exits = scores >= threshold
    correct = torch.where(exits, early_correct, full_correct)
    delta, interval = experiment.paired_quality_interval(
        full_correct, correct, config['bootstrap_samples'], config['seed'],
    )
    return dict(examples=len(scores), exit_pct=100 * exits.double().mean().item(),
                accuracy_pct=100 * correct.double().mean().item(),
                baseline_accuracy_pct=100 * full_correct.double().mean().item(),
                delta_pp=delta, delta_95ci_pp=interval,
                quality_pass=interval[0] >= -config['quality_tolerance_pp'])


def calibrate(scores, early_correct, full_correct, config):
    # Inspect a fixed coverage grid using calibration labels only.
    thresholds = torch.quantile(scores, torch.linspace(0, 1, 21)).unique().tolist()
    thresholds.append(float('inf'))
    candidates = [(threshold, quality(scores, early_correct, full_correct, threshold, config))
                  for threshold in thresholds]
    return max((item for item in candidates if item[1]['quality_pass']),
               key=lambda item: item[1]['exit_pct'])


def predict_gated(full, early, transform, image, head, baseline, threshold):
    x = transform(image).unsqueeze(0)
    for layer in (full.conv1, full.bn1, full.relu, full.maxpool, full.layer1, full.layer2, full.layer3):
        x = layer(x)
    logits = early[8](early[7](x)) @ head['weight'] + head['bias']
    if margin(logits).item() >= threshold:
        return logits.argmax(1), True
    pooled = torch.flatten(full.avgpool(full.layer4(x)), 1)
    return experiment.classify(pooled, baseline).argmax(1), False


def benchmark(full, early, transform, dataset, indices, head, baseline, threshold, config, seed):
    rng = np.random.default_rng(seed)
    indices = np.random.default_rng(config['seed']).permutation(indices)[:config['timed_examples']]
    if len(indices) != config['timed_examples']:
        raise ValueError('Too few validation images for timing')
    images = [dataset[int(index)][0] for index in indices]
    paths = {
        'full': lambda image: early_exit.predict(full, transform, image, baseline['weight'], baseline['bias']),
        'gated': lambda image: predict_gated(full, early, transform, image, head, baseline, threshold),
    }
    timings = {name: [] for name in paths}
    exits = []
    for _ in range(config['repeats']):
        order = rng.permutation(len(images))
        for name in rng.permutation(list(paths)):
            predict = paths[name]
            for j in range(config['warmup']):
                predict(images[order[j % len(images)]])
            times = []
            for j in order:
                image = images[j]
                start = time.perf_counter_ns()
                result = predict(image)
                times.append((time.perf_counter_ns() - start) / 1e6)
                if name == 'gated':
                    exits.append(result[1])
            timings[name].append(times)
    means = {name: np.mean(values, axis=1).tolist() for name, values in timings.items()}
    gain, interval = experiment.paired_latency_interval(
        means['full'], means['gated'], config['bootstrap_samples'], seed,
    )
    return dict(seed=seed, timed_exit_pct=100 * np.mean(exits), indices=indices.tolist(),
                p50_ms={name: float(np.median(times)) for name, times in timings.items()},
                mean_reduction_pct=gain, latency_95ci_pct=interval,
                latency_pass=interval[0] >= config['minimum_latency_reduction_pct'],
                timings_ms=timings)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path, required=True)
    parser.add_argument('--head', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=Path('data'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = experiment.configuration(args.artifacts / 'config.json')
    torch.set_num_threads(config['threads'])
    torch.use_deterministic_algorithms(True)
    experiment.verify_preparation(args.artifacts, args.data)
    head = torch.load(args.head, weights_only=True)
    provenance = {name: experiment.digest(args.artifacts / name)
                  for name in ('manifest.json', 'config.json', 'splits.json', 'states.pt', 'encoder.pt')}
    provenance.update(early_exit_source=experiment.digest(early_exit.__file__),
                      experiment_source=experiment.digest(experiment.__file__))
    if head['provenance'] != provenance:
        raise ValueError('The early-exit head does not match the inputs or source')
    full, early, transform = early_exit.load_models(args.artifacts)
    dataset = datasets.CIFAR10(args.data, train=True)
    splits = json.loads((args.artifacts / 'splits.json').read_text())
    baseline = torch.load(args.artifacts / 'states.pt', weights_only=True)['512_fp32']
    threshold, calibration = calibrate(
        *predictions(full, early, transform, dataset, splits['calibration'], head, baseline), config,
    )
    print('Calibration:', calibration, flush=True)
    validation = quality(
        *predictions(full, early, transform, dataset, splits['validation'], head, baseline), threshold, config,
    )
    print('Validation:', validation, flush=True)
    timing = benchmark(full, early, transform, dataset, splits['validation'], head, baseline,
                       threshold, config, config['seed'])
    report = dict(threshold=threshold if np.isfinite(threshold) else None,
                  calibration=calibration, validation=validation, timing=timing,
                  pass_both=validation['quality_pass'] and timing['latency_pass'],
                  provenance=provenance, head_sha256=experiment.digest(args.head),
                  gate_source_sha256=experiment.digest(__file__))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as output:
        json.dump(report, output, indent=2)


if __name__ == '__main__':
    main()
