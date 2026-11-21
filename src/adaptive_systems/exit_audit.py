"""Audit the relaxed margin band and calibrate a conservative cutoff."""

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision.datasets import CIFAR10

from . import confidence_exit, early_exit, experiment


class PairedFeatures(nn.Module):
    def __init__(self, full, early):
        super().__init__()
        self.prefix = early[:7]
        self.stage_four = full.layer4
        self.pool = full.avgpool

    def forward(self, images):
        x = self.prefix(images)
        return torch.cat((self.pool(x).flatten(1), self.pool(self.stage_four(x)).flatten(1)), dim=1)


def summarize(exits, early, full, config):
    output = torch.where(exits, early, full)
    delta, interval = experiment.paired_quality_interval(
        full, output, config['bootstrap_samples'], config['seed'],
    )
    return dict(exits=int(exits.sum()), accuracy_pct=100 * output.double().mean().item(),
                delta_pp=delta, interval_95_pp=interval)


def band_counts(band, early, full):
    return dict(requests=int(band.sum()), corrected_by_stage_four=int((band & ~early & full).sum()),
                harmed_by_stage_four=int((band & early & ~full).sum()))


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('artifacts', 'head', 'matrix', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--data', default=Path('data'), type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('--output already exists')
    config = experiment.configuration(args.artifacts / 'config.json')
    torch.set_num_threads(config['threads'])
    torch.use_deterministic_algorithms(True)
    experiment.verify_preparation(args.artifacts, args.data)
    with zipfile.ZipFile(args.matrix) as archive:
        lock = json.loads(archive.read('lock.json'))
        sessions = [json.loads(archive.read(f'session-{i}.json')) for i in range(3)]
    if (lock['identity']['preparation'] != experiment.digest(args.artifacts / 'manifest.json')
            or lock['identity']['head'] != experiment.digest(args.head)):
        raise ValueError('Matrix and model artifacts do not match')
    policy = lock['policies']['queue_aware']
    strict, relaxed = policy['thresholds']
    depth = policy['depth']
    full_model, early_model, transform = early_exit.load_models(args.artifacts)
    paired = PairedFeatures(full_model, early_model).eval()
    dataset = CIFAR10(args.data, train=True)
    splits = json.loads((args.artifacts / 'splits.json').read_text())
    early_head = torch.load(args.head, weights_only=True)
    baseline = torch.load(args.artifacts / 'states.pt', weights_only=True)['512_fp32']
    labels = torch.tensor(dataset.targets)

    def score(indices):
        features = experiment.features(paired, transform, dataset, indices, 768)
        logits = features[:, :256] @ early_head['weight'] + early_head['bias']
        full_logits = experiment.classify(features[:, 256:].contiguous(), baseline)
        return confidence_exit.margin(logits), logits.argmax(1).eq(labels[indices]), full_logits.argmax(1).eq(labels[indices])

    margins, early, full = score(splits['calibration'])
    thresholds = sorted({strict, relaxed, *(
        float(t) for t in torch.quantile(margins, torch.linspace(0, 1, 21)) if relaxed < t < strict
    )})
    fixed = margins >= strict
    candidates = []
    for cutoff in thresholds:
        band = (margins >= cutoff) & ~fixed
        # Adverse case: only the newly eligible exits that cost accuracy are admitted.
        harmful = band & ~early & full
        adverse = summarize(fixed | harmful, early, full, config)
        candidates.append(dict(cutoff=cutoff, **band_counts(band, early, full), adverse=adverse))
    qualifying = [c for c in candidates if c['adverse']['interval_95_pp'][0] >= -config['quality_tolerance_pp']]
    chosen = qualifying[0]['cutoff'] if qualifying else strict
    print('Frozen cutoff', chosen, 'calibration candidates', len(candidates), flush=True)

    old_indices = sorted({index for session in sessions for index in session['indices']})
    margins, early, full = score(old_indices)
    lookup = {index: offset for offset, index in enumerate(old_indices)}
    prior = {}
    for session in sessions:
        offsets = [lookup[index] for index in session['indices']]
        for workload, runs in session['results'].items():
            if not workload.startswith('bursty'):
                continue
            records = runs['queue_aware']['requests']
            busy_exits = torch.tensor([r[1] and r[2] >= depth for r in records])
            band = busy_exits & (margins[offsets] < strict)
            prior[f"{session['session']}/{workload}"] = band_counts(band, early[offsets], full[offsets])
    used = set(old_indices)
    unused = [i for i in splits['validation'] if i not in used]
    margins, early, full = score(unused)
    fixed = margins >= strict
    band = (margins >= chosen) & ~fixed
    validation = dict(examples=len(unused), full_accuracy_pct=100 * full.double().mean().item(),
                      fixed=summarize(fixed, early, full, config),
                      all_eligible=summarize(fixed | band, early, full, config),
                      adverse=summarize(fixed | (band & ~early & full), early, full, config),
                      band=band_counts(band, early, full))
    report = dict(strict=strict, original_relaxed=relaxed, conservative=chosen, queue_depth=depth,
                  calibration_qualified=bool(qualifying), calibration=candidates, prior_live_audit=prior,
                  unused_validation=validation, matrix_sha256=experiment.digest(args.matrix),
                  source_sha256=experiment.digest(__file__))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(prior=prior, validation=validation), indent=2))


if __name__ == '__main__':
    main()
