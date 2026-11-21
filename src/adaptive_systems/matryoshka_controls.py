"""Evaluate jointly trained embedding prefixes on frozen ResNet-18 features."""

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torchvision.datasets import CIFAR10

from . import early_exit, experiment


class NestedProjection(nn.Module):
    def __init__(self, input_width, widths):
        super().__init__()
        self.projection = nn.Linear(input_width, max(widths))
        self.heads = nn.ModuleDict({str(width): nn.Linear(width, 10) for width in widths})

    def forward(self, features):
        embedding = self.projection(features)
        return {width: head(embedding[:, :int(width)]) for width, head in self.heads.items()}


def fit(train, labels, widths, epochs, seed, model=None):
    if model is None:
        model = NestedProjection(train.shape[1], widths)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    rng = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        for indices in torch.randperm(len(train), generator=rng).split(256):
            logits = model(train[indices])
            loss = sum(nn.functional.cross_entropy(value, labels[indices]) for value in logits.values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model.eval()


def matched_controls(train, labels, validation, validation_labels, widths, epochs, seeds, samples):
    results = []
    for seed in seeds:
        torch.manual_seed(seed)
        nested = NestedProjection(train.shape[1], widths)
        controls = {}
        for width in (64, 128):
            control = NestedProjection(train.shape[1], (width,))
            with torch.no_grad():
                control.projection.weight.copy_(nested.projection.weight[:width])
                control.projection.bias.copy_(nested.projection.bias[:width])
            control.heads[str(width)].load_state_dict(nested.heads[str(width)].state_dict())
            controls[width] = control
        fit(train, labels, widths, epochs, seed, nested)
        with torch.inference_mode():
            nested_predictions = {key: logits.argmax(1).eq(validation_labels)
                                  for key, logits in nested(validation).items()}
        for width, control in controls.items():
            fit(train, labels, (width,), epochs, seed, control)
            with torch.inference_mode():
                key = str(width)
                single = control(validation)[key].argmax(1).eq(validation_labels)
                joint = nested_predictions[key]
                delta, interval = experiment.paired_quality_interval(single, joint, samples, seed)
                results.append(dict(seed=seed, width=width, single_accuracy_pct=100 * single.double().mean().item(),
                                    joint_accuracy_pct=100 * joint.double().mean().item(),
                                    joint_minus_single_pp=delta, delta_95ci_pp=interval))
        print(f'Completed matched training seed {seed}', flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=Path('data'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--control-seeds', type=int, nargs='+', help='Compare joint and single-width training')
    args = parser.parse_args()
    if args.control_seeds and (min(args.control_seeds) < 0 or len(set(args.control_seeds)) != len(args.control_seeds)):
        parser.error('--control-seeds must be distinct nonnegative integers')
    if args.epochs < 1:
        parser.error('--epochs must be positive')
    if args.output.exists():
        parser.error('--output already exists')
    config = experiment.configuration(args.artifacts / 'config.json')
    seed = config['seed']
    torch.manual_seed(seed)
    torch.set_num_threads(config['threads'])
    torch.use_deterministic_algorithms(True)
    experiment.verify_preparation(args.artifacts, args.data)
    full, _, transform = early_exit.load_models(args.artifacts)
    dataset = CIFAR10(args.data, train=True)
    splits = json.loads((args.artifacts / 'splits.json').read_text())
    labels = torch.tensor(dataset.targets)
    train = experiment.features(full, transform, dataset, splits['train'])
    validation = experiment.features(full, transform, dataset, splits['validation'])
    train_labels, validation_labels = labels[splits['train']], labels[splits['validation']]
    widths = (64, 128, 256, 512)
    if args.control_seeds:
        results = matched_controls(train, train_labels, validation, validation_labels,
                                   widths, args.epochs, args.control_seeds, config['bootstrap_samples'])
        report = dict(epochs=args.epochs, widths=widths, train_examples=len(train),
                      validation_examples=len(validation), results=results,
                      preparation_sha256=experiment.digest(args.artifacts / 'manifest.json'),
                      source_sha256=experiment.digest(__file__))
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / 'matched-controls.json').write_text(json.dumps(report, indent=2))
        return
    nested = fit(train, train_labels, widths, args.epochs, seed)
    states = torch.load(args.artifacts / 'states.pt', weights_only=True)
    with torch.inference_mode():
        reference = experiment.classify(validation, states['512_fp32']).argmax(1).eq(validation_labels)
        outputs = {f'nested_{width}': logits for width, logits in nested(validation).items()}
        outputs['pca_128_fp32'] = experiment.classify(validation, states['128_fp32'])
        weight, bias = experiment.ridge_head(train[:, :128], train_labels, config['ridge'])
        outputs['raw_128_fp32'] = validation[:, :128] @ weight + bias
        results = {}
        for name, logits in outputs.items():
            correct = logits.argmax(1).eq(validation_labels)
            delta, interval = experiment.paired_quality_interval(
                reference, correct, config['bootstrap_samples'], seed,
            )
            results[name] = dict(accuracy_pct=100 * correct.double().mean().item(),
                                 delta_pp=delta, delta_95ci_pp=interval,
                                 quality_pass=interval[0] >= -config['quality_tolerance_pp'])
    args.output.mkdir(parents=True, exist_ok=False)
    weights = args.output / 'projection.pt'
    torch.save(nested.state_dict(), weights)
    report = dict(paper='https://arxiv.org/abs/2205.13147', widths=widths, epochs=args.epochs,
                  seed=seed, train_examples=len(train), validation_examples=len(validation),
                  reference_accuracy_pct=100 * reference.double().mean().item(),
                  results=results, projection_sha256=experiment.digest(weights),
                  preparation_sha256=experiment.digest(args.artifacts / 'manifest.json'),
                  source_sha256=experiment.digest(__file__))
    (args.output / 'validation.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
