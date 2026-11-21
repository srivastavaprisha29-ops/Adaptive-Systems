"""Time the locked 64- and 512-dimensional paths from one checkpoint."""

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from torchvision.datasets import CIFAR10

from . import early_exit, experiment, matryoshka


def measure(paths, examples, config, seed):
    rng = np.random.default_rng(seed)
    timings = {width: [] for width in paths}
    for _ in range(config['repeats']):
        for width, predict in paths.items():
            timings[width].append([])
            for index in range(config['warmup']):
                predict(examples[index % len(examples)])
        order = rng.permutation(len(examples))
        compact_first = np.arange(len(examples)) % 2 == 0
        rng.shuffle(compact_first)
        for index, first in zip(order, compact_first):
            for width in ((64, 512) if first else (512, 64)):
                predict = paths[width]
                example = examples[index]
                start = time.perf_counter_ns()
                predict(example)
                timings[width][-1].append((time.perf_counter_ns() - start) / 1e6)
    gain, interval = experiment.paired_latency_interval(
        np.mean(timings[512], axis=1), np.mean(timings[64], axis=1),
        config['bootstrap_samples'], seed,
    )
    return dict(p50_ms={width: float(np.median(values)) for width, values in timings.items()},
                mean_reduction_pct=gain, interval_95_pct=interval, raw_ms=timings)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path, required=True)
    parser.add_argument('--pilot', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=Path('data'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=34)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('--output already exists')
    config = experiment.configuration(args.artifacts / 'config.json')
    torch.set_num_threads(config['threads'])
    torch.use_deterministic_algorithms(True)
    experiment.verify_preparation(args.artifacts, args.data)
    weights = args.pilot / 'projection.pt'
    report = json.loads((args.pilot / 'validation.json').read_text())
    held_out = json.loads((args.pilot / 'held-out-test.json').read_text())
    if (experiment.digest(weights) != report['projection_sha256']
            or experiment.digest(args.artifacts / 'manifest.json') != report['preparation_sha256']
            or experiment.digest(args.pilot / 'validation.json') != held_out['validation_sha256']
            or held_out['projection_sha256'] != report['projection_sha256']
            or held_out['selected_width'] != 64):
        raise ValueError('Pilot inputs do not match the frozen held-out evaluation')
    model = matryoshka.NestedProjection(512, report['widths']).eval()
    model.load_state_dict(torch.load(weights, weights_only=True, map_location='cpu'))
    encoder, _, transform = early_exit.load_models(args.artifacts)
    dataset = CIFAR10(args.data, train=True)
    split = json.loads((args.artifacts / 'splits.json').read_text())['validation']
    indices = np.random.default_rng(config['seed']).permutation(split)[:config['timed_examples']]
    if len(indices) != config['timed_examples']:
        raise ValueError('Too few validation images')
    images = [dataset[int(index)][0] for index in indices]

    def make_head(width):
        weight = model.projection.weight[:width]
        bias = model.projection.bias[:width]
        head = model.heads[str(width)]

        def predict(features):
            return head(torch.nn.functional.linear(features, weight, bias)).argmax(1)

        return predict

    def make_pipeline(head):
        return lambda image: head(encoder(transform(image).unsqueeze(0)))

    heads = {width: make_head(width) for width in (64, 512)}
    end_to_end = measure({width: make_pipeline(head) for width, head in heads.items()},
                         images, config, args.seed)
    features = [encoder(transform(image).unsqueeze(0)) for image in images]
    downstream = measure(heads, features, config, args.seed)
    result = dict(seed=args.seed, platform=platform.platform(), threads=config['threads'],
                  indices=indices.tolist(), checkpoint_sha256=report['projection_sha256'],
                  source_sha256=experiment.digest(__file__), end_to_end=end_to_end,
                  downstream=downstream,
                  latency_pass=end_to_end['interval_95_pct'][0] >= config['minimum_latency_reduction_pct'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
    print({key: {k: v for k, v in result[key].items() if k != 'raw_ms'}
           for key in ('end_to_end', 'downstream')}, flush=True)


if __name__ == '__main__':
    main()
