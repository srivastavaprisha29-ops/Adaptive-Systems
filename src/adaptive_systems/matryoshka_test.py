"""Evaluate the frozen 64-dimensional Matryoshka pilot on CIFAR-10 test data."""

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from torchvision.datasets import CIFAR10

from . import early_exit, experiment, matryoshka


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path, required=True)
    parser.add_argument('--pilot', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=Path('data'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('--output already exists')
    config = experiment.configuration(args.artifacts / 'config.json')
    torch.set_num_threads(config['threads'])
    torch.use_deterministic_algorithms(True)
    experiment.verify_preparation(args.artifacts, args.data)
    validation = args.pilot / 'validation.json'
    weights = args.pilot / 'projection.pt'
    locked = json.loads(validation.read_text())
    if (locked['projection_sha256'] != experiment.digest(weights)
            or locked['preparation_sha256'] != experiment.digest(args.artifacts / 'manifest.json')
            or not locked['results']['nested_64']['quality_pass']):
        raise ValueError('The 64-dimensional checkpoint does not match the qualifying pilot')
    model = matryoshka.NestedProjection(512, locked['widths']).eval()
    model.load_state_dict(torch.load(weights, weights_only=True, map_location='cpu'))
    full, _, transform = early_exit.load_models(args.artifacts)
    dataset = CIFAR10(args.data, train=False)
    labels = torch.tensor(dataset.targets)
    features = experiment.features(full, transform, dataset, list(range(len(dataset))))
    prefix = F.linear(features, model.projection.weight[:64], model.projection.bias[:64])
    full_projection = model.projection(features)
    torch.testing.assert_close(prefix, full_projection[:, :64])
    baseline = torch.load(args.artifacts / 'states.pt', weights_only=True, map_location='cpu')['512_fp32']
    outputs = {
        'prefix_64': model.heads['64'](prefix),
        'projected_512': model.heads['512'](full_projection),
        'ridge_512': experiment.classify(features, baseline),
    }
    correct = {name: logits.argmax(1).eq(labels) for name, logits in outputs.items()}
    comparisons = {}
    for name in ('projected_512', 'ridge_512'):
        delta, interval = experiment.paired_quality_interval(
            correct[name], correct['prefix_64'], config['bootstrap_samples'], config['seed'],
        )
        comparisons[name] = dict(delta_pp=delta, delta_95ci_pp=interval,
                                 quality_pass=interval[0] >= -config['quality_tolerance_pp'])
    report = dict(split='official CIFAR-10 test', examples=len(dataset), selected_width=64,
                  accuracy_pct={k: 100 * v.double().mean().item() for k, v in correct.items()},
                  prefix_comparisons=comparisons, quality_tolerance_pp=config['quality_tolerance_pp'],
                  raw_feature_bytes={'prefix_64': 256, 'projected_512': 2048, 'ridge_512': 2048},
                  validation_sha256=experiment.digest(validation), projection_sha256=experiment.digest(weights),
                  preparation_sha256=locked['preparation_sha256'], source_sha256=experiment.digest(__file__))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
