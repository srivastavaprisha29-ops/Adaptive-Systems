"""Train a linear gate to estimate the accuracy benefit of stage four."""

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torchvision.datasets import CIFAR10

from . import confidence_exit, early_exit, experiment


def logit_features(logits):
    log_probs = logits.log_softmax(dim=-1)
    entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
    return torch.cat((logits, confidence_exit.margin(logits).unsqueeze(-1),
                      entropy.unsqueeze(-1)), dim=-1)


def fit_gate(features, early_correct, full_correct, ridge):
    mean = features.mean(0)
    scale = features.std(0, unbiased=False).clamp_min(1e-6)
    design = (features - mean) / scale
    target = full_correct.float() - early_correct.float()
    penalty = ridge * torch.eye(design.shape[1], dtype=design.dtype)
    weight = torch.linalg.solve(design.T @ design + penalty,
                                design.T @ (target - target.mean()))
    return dict(mean=mean, scale=scale, weight=weight, bias=target.mean())


def exit_score(features, gate):
    return -(((features - gate['mean']) / gate['scale']) @ gate['weight'] + gate['bias'])


class PairedFeatures(nn.Module):
    def __init__(self, full, early):
        super().__init__()
        self.prefix = early[:7]
        self.final_stage = full.layer4
        self.pool = full.avgpool

    def forward(self, images):
        intermediate = self.prefix(images)
        early = self.pool(intermediate).flatten(1)
        full = self.pool(self.final_stage(intermediate)).flatten(1)
        return torch.cat((early, full), dim=1)


def examples(model, transform, dataset, indices, early_head, full_head):
    features = experiment.features(model, transform, dataset, indices, 768)
    logits = features[:, :256] @ early_head['weight'] + early_head['bias']
    labels = torch.tensor(dataset.targets)[indices]
    early_correct = logits.argmax(1).eq(labels)
    full_correct = experiment.classify(features[:, 256:].contiguous(), full_head).argmax(1).eq(labels)
    return logits, early_correct, full_correct


def score_curve(scores, early_correct, full_correct, config):
    thresholds = torch.quantile(scores, torch.linspace(0, 1, 21)).unique().tolist()
    return [dict(threshold=threshold, **confidence_exit.quality(
        scores, early_correct, full_correct, threshold, config,
    )) for threshold in thresholds]


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', required=True, type=Path)
    parser.add_argument('--head', required=True, type=Path)
    parser.add_argument('--data', default=Path('data'), type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--train-per-class', default=400, type=int)
    args = parser.parse_args()
    config = experiment.configuration(args.artifacts / 'config.json')
    if args.train_per_class < 1 or args.output.exists():
        parser.error('Use a positive training count and a new output directory')
    torch.set_num_threads(config['threads'])
    torch.use_deterministic_algorithms(True)
    experiment.verify_preparation(args.artifacts, args.data)
    early_head = torch.load(args.head, weights_only=True, map_location='cpu')
    provenance = {name: experiment.digest(args.artifacts / name)
                  for name in ('manifest.json', 'config.json', 'splits.json', 'states.pt', 'encoder.pt')}
    provenance.update(early_exit_source=experiment.digest(early_exit.__file__),
                      experiment_source=experiment.digest(experiment.__file__))
    if provenance != early_head['provenance']:
        raise ValueError('Early head does not match the prepared artifacts')
    full_head = torch.load(args.artifacts / 'states.pt', weights_only=True)['512_fp32']
    full, early, transform = early_exit.load_models(args.artifacts)
    model = PairedFeatures(full, early).eval()
    dataset = CIFAR10(args.data, train=True)
    splits = json.loads((args.artifacts / 'splits.json').read_text())
    rng = torch.Generator().manual_seed(config['seed'])
    train = torch.tensor(splits['train'])
    targets = torch.tensor(dataset.targets)
    train_groups = [train[targets[train].eq(label)] for label in range(10)]
    if args.train_per_class > min(map(len, train_groups)):
        parser.error('--train-per-class exceeds the available examples')
    train_indices = sorted(torch.cat([
        group[torch.randperm(len(group), generator=rng)[:args.train_per_class]]
        for group in train_groups
    ]).tolist())

    def collect(indices):
        return examples(model, transform, dataset, indices, early_head, full_head)

    train_logits, train_early, train_full = collect(train_indices)
    gate = fit_gate(logit_features(train_logits), train_early, train_full, config['ridge'])
    cal_logits, cal_early, cal_full = collect(splits['calibration'])
    val_logits, val_early, val_full = collect(splits['validation'])
    results = {}
    for name, cal, val in (
        ('margin', confidence_exit.margin(cal_logits), confidence_exit.margin(val_logits)),
        ('learned', exit_score(logit_features(cal_logits), gate), exit_score(logit_features(val_logits), gate)),
    ):
        threshold, calibration = confidence_exit.calibrate(cal, cal_early, cal_full, config)
        validation = confidence_exit.quality(val, val_early, val_full, threshold, config)
        results[name] = dict(threshold=threshold if threshold != float('inf') else None,
                             calibration=calibration, validation=validation,
                             validation_curve=score_curve(val, val_early, val_full, config))
        print(name, validation, flush=True)
    args.output.mkdir(parents=True, exist_ok=False)
    checkpoint = args.output / 'gate.pt'
    torch.save(gate, checkpoint)
    report = dict(results=results, training_indices=train_indices, provenance=provenance,
                  early_head_sha256=experiment.digest(args.head), gate_sha256=experiment.digest(checkpoint),
                  source_sha256=experiment.digest(__file__))
    (args.output / 'validation.json').write_text(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
