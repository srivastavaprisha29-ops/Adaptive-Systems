"""Hinton et al. (2015) distillation adapted to a frozen stage-three exit."""
import argparse
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.datasets import CIFAR10

from . import confidence_exit, early_exit, experiment
from .learned_exit import PairedFeatures


def distillation_loss(student, teacher, labels, temperature=4.0, alpha=0.7):
    """Hard-label CE plus T-squared KL(teacher || student), averaged by batch."""
    if not math.isfinite(temperature) or temperature <= 0 or not 0 <= alpha <= 1:
        raise ValueError('Temperature must be positive and alpha in [0, 1]')
    hard = F.cross_entropy(student, labels)
    soft = F.kl_div(F.log_softmax(student / temperature, dim=-1),
                    F.softmax(teacher.detach() / temperature, dim=-1),
                    reduction='batchmean') * temperature ** 2
    return (1 - alpha) * hard + alpha * soft


def fit(x, teacher, labels, seed, epochs, temperature, alpha, *, history=None, cosine=False):
    torch.manual_seed(seed)
    head = nn.Linear(x.shape[1], teacher.shape[1])
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.001, weight_decay=0.01)
    rng = torch.Generator().manual_seed(seed)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs) if cosine else None
    for epoch in range(epochs):
        total_loss = 0.0
        for ids in torch.randperm(len(x), generator=rng).split(256):
            optimizer.zero_grad()
            loss = distillation_loss(head(x[ids]), teacher[ids], labels[ids], temperature, alpha)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(ids)
        if history is not None and (epoch == 0 or (epoch + 1) % 5 == 0 or epoch + 1 == epochs):
            with torch.no_grad():
                correct = sum(head(x[ids]).argmax(1).eq(labels[ids]).sum().item()
                              for ids in torch.arange(len(x)).split(1024))
            record = dict(epoch=epoch + 1, loss=total_loss / len(x),
                          train_accuracy_pct=100 * correct / len(x), lr=optimizer.param_groups[0]['lr'])
            history.append(record)
            print('Training', seed, alpha, record, flush=True)
        if scheduler is not None:
            scheduler.step()
    return dict(weight=head.weight.detach().T.contiguous(), bias=head.bias.detach())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=Path('data'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--train-per-class', type=int, default=400)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--cosine', action='store_true', help='Decay learning rate over the fixed training budget')
    parser.add_argument('--feature-cache', type=Path, help='Reuse verified frozen features across training runs')
    args = parser.parse_args()
    if args.output.exists() or args.train_per_class < 1 or args.epochs < 1:
        parser.error('Use positive training counts and a new output directory')
    config = experiment.configuration(args.artifacts / 'config.json')
    torch.set_num_threads(config['threads'])
    torch.use_deterministic_algorithms(True)
    experiment.verify_preparation(args.artifacts, args.data)
    splits = json.loads((args.artifacts / 'splits.json').read_text())
    dataset = CIFAR10(args.data, train=True)
    labels = torch.tensor(dataset.targets)
    train = torch.tensor(splits['train'])
    rng = torch.Generator().manual_seed(config['seed'])
    groups = [train[labels[train].eq(c)] for c in range(10)]
    if args.train_per_class > min(map(len, groups)):
        parser.error('Training count exceeds available split')
    indices = sorted(torch.cat([g[torch.randperm(len(g), generator=rng)[:args.train_per_class]]
                                for g in groups]).tolist())
    full, early, transform = early_exit.load_models(args.artifacts)
    model = PairedFeatures(full, early).eval()
    teacher = torch.load(args.artifacts / 'states.pt', weights_only=True)['512_fp32']
    provenance = {n: experiment.digest(args.artifacts / n) for n in
                  ('encoder.pt', 'states.pt', 'splits.json', 'config.json', 'manifest.json')}
    dependencies = {name: experiment.digest(module.__file__) for name, module in
                    [('experiment', experiment), ('early_exit', early_exit)]}
    dependencies['paired_features'] = experiment.digest(__import__('inspect').getfile(PairedFeatures))
    collected = {}
    for name, ids in [('train', indices), ('calibration', splits['calibration']),
                      ('validation', splits['validation'])]:
        signature = dict(preparation=provenance, dependencies=dependencies, indices=ids)
        cache = args.feature_cache / f'{name}.pt' if args.feature_cache else None
        if cache is not None and cache.exists():
            saved = torch.load(cache, weights_only=True, map_location='cpu')
            if saved['signature'] != signature:
                raise ValueError(f'Frozen-feature cache does not match inputs: {cache}')
            collected[name] = saved['values']
        else:
            with torch.no_grad():
                features = experiment.features(model, transform, dataset, ids, 768)
                collected[name] = (features[:, :256].clone(),
                                   experiment.classify(features[:, 256:].contiguous(), teacher), labels[ids])
            if cache is not None:
                cache.parent.mkdir(parents=True, exist_ok=True)
                temporary = cache.with_suffix('.tmp')
                torch.save(dict(signature=signature, values=collected[name]), temporary)
                temporary.replace(cache)
        print('Collected', name, len(ids), flush=True)
    args.output.mkdir(parents=True)
    results = []
    checkpoints = {}
    for seed in (29, 30, 31):
        predictions = {}
        for name, alpha in [('supervised', 0.0), ('distilled', 0.7)]:
            history = []
            state = fit(*collected['train'], seed, args.epochs, 4.0, alpha, history=history, cosine=args.cosine)
            checkpoints[f'{seed}_{name}'] = state
            evaluations = {}
            for split in ('calibration', 'validation'):
                x, logits, target = collected[split]
                student = x @ state['weight'] + state['bias']
                evaluations[split] = (confidence_exit.margin(student), student.argmax(1).eq(target),
                                      logits.argmax(1).eq(target))
            threshold, calibration = confidence_exit.calibrate(*evaluations['calibration'], config)
            validation = confidence_exit.quality(*evaluations['validation'], threshold, config)
            predictions[name] = evaluations['validation'][1]
            result = dict(seed=seed, method=name, student_accuracy_pct=100 * predictions[name].double().mean().item(),
                          threshold=threshold if math.isfinite(threshold) else None,
                          calibration=calibration, validation=validation, training_history=history)
            results.append(result)
            print(result, flush=True)
        delta, interval = experiment.paired_quality_interval(predictions['supervised'], predictions['distilled'],
                                                            config['bootstrap_samples'], seed)
        results.append(dict(seed=seed, distilled_minus_supervised_pp=delta, paired_95ci_pp=interval))
    torch.save(checkpoints, args.output / 'heads.pt')
    report = dict(paper='https://arxiv.org/abs/1503.02531', adaptation='Frozen ImageNet ResNet-18 features, stage-four ridge teacher and stage-three linear student',
                  training_indices=indices, epochs=args.epochs, seeds=[29,30,31], temperature=4.0, alpha=0.7,
                  optimizer=dict(name='AdamW', lr=0.001, weight_decay=0.01, batch_size=256, schedule='cosine' if args.cosine else 'constant'),
                  results=results, source_sha256=experiment.digest(__file__),
                  preparation_sha256=provenance, dependency_sha256=dependencies,
                  checkpoint_sha256=experiment.digest(args.output / 'heads.pt'),
                  limitations=(['Pilot training subset'] if args.train_per_class < min(map(len, groups)) else []) + ['No official test evaluation', 'No latency measurement',
                               'Temperature and alpha fixed in advance, no tuning'])
    (args.output / 'validation.json').write_text(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
