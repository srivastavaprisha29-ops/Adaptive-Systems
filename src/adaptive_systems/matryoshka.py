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


def fit(train, labels, widths, epochs, seed):
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=Path('data'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=20)
    args = parser.parse_args()
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
