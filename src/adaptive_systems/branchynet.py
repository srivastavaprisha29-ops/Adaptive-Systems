"""Single-branch adaptation of BranchyNet inference (arXiv:1709.01686)."""

import argparse
import json
from pathlib import Path

import torch
from torchvision.datasets import CIFAR10

from . import confidence_exit, early_exit, experiment, interleaved_exit


def negative_entropy(logits):
    """Higher scores indicate a more concentrated softmax distribution."""
    log_probs = logits.log_softmax(dim=-1)
    return (log_probs.exp() * log_probs).sum(dim=-1)


def predict(full, early, transform, image, head, baseline, threshold):
    x = transform(image).unsqueeze(0)
    for layer in (full.conv1, full.bn1, full.relu, full.maxpool, full.layer1, full.layer2, full.layer3):
        x = layer(x)
    logits = early[8](early[7](x)) @ head['weight'] + head['bias']
    if negative_entropy(logits).item() >= threshold:
        return logits.argmax(1), True
    pooled = torch.flatten(full.avgpool(full.layer4(x)), 1)
    return experiment.classify(pooled, baseline).argmax(1), False


def split_scores(full, early, transform, dataset, indices, head, baseline):
    features = experiment.features(early, transform, dataset, indices, 256)
    early_logits = features @ head['weight'] + head['bias']
    full_features = experiment.features(full, transform, dataset, indices)
    labels = torch.tensor(dataset.targets)[indices]
    scores = {'entropy': negative_entropy(early_logits), 'margin': confidence_exit.margin(early_logits)}
    return scores, early_logits.argmax(1).eq(labels), experiment.classify(full_features, baseline).argmax(1).eq(labels)


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
    provenance = {name: experiment.digest(args.artifacts / name)
                  for name in ('manifest.json', 'config.json', 'splits.json', 'states.pt', 'encoder.pt')}
    provenance.update(early_exit_source=experiment.digest(early_exit.__file__),
                      experiment_source=experiment.digest(experiment.__file__))
    head = torch.load(args.head, weights_only=True)
    if provenance != head['provenance']:
        raise ValueError('Early-exit head does not match the inputs or code')
    full, early, transform = early_exit.load_models(args.artifacts)
    baseline = torch.load(args.artifacts / 'states.pt', weights_only=True)['512_fp32']
    splits = json.loads((args.artifacts / 'splits.json').read_text())
    dataset = CIFAR10(args.data, train=True)
    context = full, early, transform, dataset
    scores, early_correct, full_correct = split_scores(*context, splits['calibration'], head, baseline)
    thresholds = {name: confidence_exit.calibrate(values, early_correct, full_correct, config)
                  for name, values in scores.items()}
    print('Calibration:', thresholds, flush=True)
    scores, early_correct, full_correct = split_scores(*context, splits['validation'], head, baseline)
    results = {}
    for name, predictor in [('entropy', predict), ('margin', confidence_exit.predict_gated)]:
        threshold, calibration = thresholds[name]
        validation = confidence_exit.quality(scores[name], early_correct, full_correct, threshold, config)
        print(f'{name} validation:', validation, flush=True)
        timing = interleaved_exit.benchmark(
            *context, splits['validation'], head, baseline, threshold, config, config['seed'],
            gated_predict=predictor,
        )
        results[name] = dict(score_threshold=threshold if threshold != float('inf') else None,
                             calibration=calibration, validation=validation, timing=timing)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(paper='https://arxiv.org/abs/1709.01686', provenance=provenance,
                  head_sha256=experiment.digest(args.head),
                  branchynet_source_sha256=experiment.digest(__file__),
                  timing_source_sha256=experiment.digest(interleaved_exit.__file__), results=results)
    with args.output.open('x') as output:
        json.dump(report, output, indent=2)
    print(f'Report saved to {args.output}')


if __name__ == '__main__':
    main()
