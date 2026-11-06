"""CPU-only static pilot: frozen ResNet features, PCA, ridge heads, INT8 storage.

INT8 is explicitly quantize/dequantize storage, not accelerated integer inference.
"""
import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import psutil
import torch
from torchvision import datasets, models


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def partition(labels, seed, validation_per_class, calibration_per_class):
    rng = np.random.default_rng(seed)
    groups = [[], [], []]
    for label in sorted(set(labels)):
        indices = rng.permutation(np.flatnonzero(np.asarray(labels) == label))
        n, m = validation_per_class, calibration_per_class
        if n < 1 or m < 1 or n + m >= len(indices):
            raise ValueError('Each class needs disjoint nonempty train/calibration/validation sets')
        for target, values in zip(groups, (indices[n+m:], indices[n:n+m], indices[:n])):
            target.extend(values.tolist())
    return [sorted(g) for g in groups]


def encode(x, precision, scale):
    if precision == 'fp32':
        return x
    if precision != 'int8_storage':
        raise ValueError('Unsupported precision')
    return torch.clamp(torch.round(x / scale), -127, 127).to(torch.int8)


def decode(x, precision, scale):
    return x if precision == 'fp32' else x.float() * scale


def project(x, state):
    return x if state['projection'] is None else (x - state['mean']) @ state['projection']


def classify(x, state):
    z = project(x, state)
    z = decode(encode(z, state['precision'], state['scale']), state['precision'], state['scale'])
    return z @ state['weight'] + state['bias']


def fit_states(train, labels, calibration, width, ridge):
    full = train.shape[1]
    if not 0 < width < min(full, len(train)) or ridge <= 0:
        raise ValueError('Require 0 < reduced_width < full width and positive ridge')
    mean = train.mean(0)
    # Fit PCA on training features only. No arbitrary feature truncation.
    _, _, vh = torch.linalg.svd(train - mean, full_matrices=False)
    projection = vh[:width].T.contiguous()
    states = {}
    for d in (full, width):
        p = None if d == full else projection
        x = train if p is None else (train - mean) @ p
        cal = calibration if p is None else (calibration - mean) @ p
        scale = max(float(cal.abs().max()) / 127, 1e-8)
        for precision in ('fp32', 'int8_storage'):
            z = decode(encode(x, precision, scale), precision, scale)
            augmented = torch.cat((z.double(), torch.ones(len(z), 1, dtype=torch.float64)), 1)
            targets = torch.nn.functional.one_hot(labels, 10).double()
            penalty = torch.eye(d + 1, dtype=torch.float64) * ridge
            penalty[-1, -1] = 0
            weights = torch.linalg.solve(augmented.T @ augmented + penalty, augmented.T @ targets).float()
            name = f'{d}_{precision}'
            states[name] = dict(width=d, precision=precision, scale=scale, mean=mean,
                                projection=p, weight=weights[:-1], bias=weights[-1])
    return states


def paired_quality_interval(reference, candidate, samples, seed):
    # Paired example bootstrap uses the same examples for every action.
    delta = (candidate.float() - reference.float()).numpy() * 100
    rng = np.random.default_rng(seed)
    means = [rng.choice(delta, len(delta), replace=True).mean() for _ in range(samples)]
    return float(delta.mean()), np.quantile(means, [0.025, 0.975]).tolist()


def configuration(path):
    c = json.loads(Path(path).read_text())
    if c['device'] != 'cpu':
        raise ValueError('Pilot supports CPU only; adding another backend requires timing validation')
    for key in ('threads', 'warmup', 'timed_examples', 'repeats', 'bootstrap_samples'):
        if not isinstance(c[key], int) or c[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if c['quality_tolerance_pp'] < 0 or not 0 <= c['minimum_latency_reduction_pct'] < 100:
        raise ValueError('Invalid acceptance threshold')
    return c


def backbone():
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights).eval()
    model.fc = torch.nn.Identity()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, weights.transforms()


@torch.inference_mode()
def features(model, transform, dataset, indices):
    rows = []
    for start in range(0, len(indices), 32):
        batch = torch.stack([transform(dataset[i][0]) for i in indices[start:start+32]])
        rows.append(model(batch))
    result = torch.cat(rows)
    if result.shape[1] != 512:
        raise ValueError(f'Unexpected feature shape: {result.shape}')
    return result


def manifest(root, config, artifacts):
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = 'unavailable', None
    import torchvision
    import PIL
    return dict(config=config, git_commit=commit, git_dirty=dirty, python=sys.version,
                platform=platform.platform(), processor=platform.processor(),
                cpu_count=psutil.cpu_count(), ram_bytes=psutil.virtual_memory().total,
                torch=torch.__version__, torchvision=torchvision.__version__,
                numpy=np.__version__, pillow=PIL.__version__,
                artifacts={str(p): digest(p) for p in artifacts},
                config_sha256=hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
                energy_joules=None, boundary='PIL image in host memory -> preprocessing -> prediction; batch=1')


def prepare(args, c):
    out = Path(args.artifacts)
    out.mkdir(parents=True, exist_ok=False)
    dataset = datasets.CIFAR10(args.data, train=True, download=args.download)
    splits = partition(dataset.targets, c['seed'], c['validation_per_class'], c['calibration_per_class'])
    (out / 'splits.json').write_text(json.dumps(dict(zip(('train', 'calibration', 'validation'), splits))))
    model, transform = backbone()
    train = features(model, transform, dataset, splits[0])
    cal = features(model, transform, dataset, splits[1])
    labels = torch.tensor(dataset.targets)[splits[0]]
    states = fit_states(train, labels, cal, c['reduced_width'], c['ridge'])
    torch.save(states, out / 'states.pt')
    torch.save(model.state_dict(), out / 'encoder.pt')
    (out / 'config.json').write_text(json.dumps(c, indent=2))
    files = [out / p for p in ('states.pt', 'encoder.pt', 'splits.json', 'config.json')]
    files += sorted((Path(args.data) / 'cifar-10-batches-py').glob('*'))
    files = [p for p in files if p.is_file()]
    (out / 'manifest.json').write_text(json.dumps(manifest(out, c, files), indent=2))
    print(f'Prepared {len(states)} static actions in {out}')


@torch.inference_mode()
def benchmark(args, c):
    artifacts = Path(args.artifacts)
    saved = json.loads((artifacts / 'config.json').read_text())
    if c != saved:
        raise ValueError('Benchmark config differs from preparation config; prepare fresh artifacts')
    provenance = json.loads((artifacts / 'manifest.json').read_text())
    for name in ('states.pt', 'encoder.pt', 'splits.json', 'config.json'):
        path = artifacts / name
        if provenance['artifacts'].get(str(path)) != digest(path):
            raise ValueError(f'Artifact checksum mismatch: {path}')
    dataset = datasets.CIFAR10(args.data, train=args.split == 'validation', download=args.download)
    # Verify the actual local data, not only the artifact files.
    for path, expected in provenance['artifacts'].items():
        if 'cifar-10-batches-py' in path and digest(path) != expected:
            raise ValueError(f'Dataset checksum mismatch: {path}')
    splits = json.loads((artifacts / 'splits.json').read_text())
    indices = splits['validation'] if args.split == 'validation' else list(range(len(dataset)))
    states = torch.load(artifacts / 'states.pt', weights_only=True, map_location='cpu')
    model = models.resnet18(weights=None).eval()
    model.fc = torch.nn.Identity()
    model.load_state_dict(torch.load(artifacts / 'encoder.pt', weights_only=True, map_location='cpu'))
    transform = models.ResNet18_Weights.IMAGENET1K_V1.transforms()
    run = Path(args.output) / (args.split + '-' + uuid.uuid4().hex[:12])
    run.mkdir(parents=True, exist_ok=False)
    x = features(model, transform, dataset, indices)
    labels = torch.tensor(dataset.targets)[indices]
    correct = {name: classify(x, state).argmax(1).eq(labels) for name, state in states.items()}
    torch.save(correct, run / 'correctness.pt')
    rng = np.random.default_rng(c['seed'])
    timed = rng.permutation(indices)[:c['timed_examples']].tolist()
    images = [(i, dataset[i][0]) for i in timed]  # Decode/loading intentionally outside timing.
    records = []
    process = psutil.Process()
    for repeat in range(c['repeats']):
        for name in rng.permutation(list(states)):
            state = states[name]
            for j in range(c['warmup']):
                classify(model(transform(images[j % len(images)][1]).unsqueeze(0)), state)
            for index, image in images:
                t0 = time.perf_counter_ns()
                tensor = transform(image).unsqueeze(0)
                t1 = time.perf_counter_ns()
                embedding = model(tensor)
                t2 = time.perf_counter_ns()
                z = project(embedding, state)
                t3 = time.perf_counter_ns()
                packed = encode(z, state['precision'], state['scale'])
                restored = decode(packed, state['precision'], state['scale'])
                t4 = time.perf_counter_ns()
                prediction = (restored @ state['weight'] + state['bias']).argmax(1)
                t5 = time.perf_counter_ns()
                records.append(dict(config=name, repeat=repeat, example=index,
                    preprocess_ms=(t1-t0)/1e6, encoder_ms=(t2-t1)/1e6,
                    projection_ms=(t3-t2)/1e6, conversion_ms=(t4-t3)/1e6,
                    head_ms=(t5-t4)/1e6, total_ms=(t5-t0)/1e6,
                    feature_payload_bytes=packed.numel()*packed.element_size(),
                    rss_after_inference_bytes=process.memory_info().rss,
                    prediction=int(prediction.item())))
    with open(run / 'timings.jsonl', 'w') as f:
        for record in records:
            f.write(json.dumps(record) + '\n')
    reference = f'{x.shape[1]}_fp32'
    summary = {}
    for name in states:
        delta, interval = paired_quality_interval(correct[reference], correct[name], c['bootstrap_samples'], c['seed'])
        rows = [r for r in records if r['config'] == name]
        times = [r['total_ms'] for r in rows]
        summary[name] = dict(accuracy_pct=float(correct[name].float().mean())*100,
            delta_accuracy_pp=delta, delta_accuracy_95ci_pp=interval,
            quality_feasible=interval[0] >= -c['quality_tolerance_pp'],
            p50_ms=float(np.median(times)), p95_ms=float(np.quantile(times, .95)),
            throughput_per_s=1000/float(np.mean(times)),
            repeat_mean_ms=[float(np.mean([r['total_ms'] for r in rows if r['repeat']==j])) for j in range(c['repeats'])],
            feature_payload_bytes=rows[0]['feature_payload_bytes'],
            max_observed_rss_bytes=max(r['rss_after_inference_bytes'] for r in rows))
    if args.split == 'validation':
        eligible = [name for name in summary if summary[name]['quality_feasible']]
        best = min(eligible, key=lambda name: summary[name]['p50_ms'])
        selection = dict(configuration=best, criterion='lowest validation p50 under paired quality CI floor',
                         states_sha256=digest(artifacts / 'states.pt'), config_sha256=digest(artifacts / 'config.json'))
        (run / 'selection.json').write_text(json.dumps(selection, indent=2))
    else:
        selection = json.loads(Path(args.selection).read_text())
        if selection['states_sha256'] != digest(artifacts / 'states.pt') or selection['config_sha256'] != digest(artifacts / 'config.json'):
            raise ValueError('Selection does not match prepared artifacts')
        best = selection['configuration']
        if best not in summary:
            raise ValueError('Selected action is absent')
    gain = 100 * (1-summary[best]['p50_ms']/summary[reference]['p50_ms'])
    report = dict(split=args.split, reference=reference, selected_static=best, metrics=summary,
                  selected_latency_reduction_pct=gain,
                  pilot_pass=best != reference and summary[best]['quality_feasible'] and gain >= c['minimum_latency_reduction_pct'],
                  limitation='RSS is sampled after inference with all heads resident, not per-action peak memory; energy unmeasured')
    (run / 'summary.json').write_text(json.dumps(report, indent=2))
    (run / 'manifest.json').write_text(json.dumps(manifest(run, c, [artifacts / 'manifest.json', run / 'timings.jsonl', run / 'correctness.pt']), indent=2))
    print(json.dumps(report, indent=2))
    print(f'Artifacts: {run}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'benchmark'])
    parser.add_argument('--config', default='configs/static.json')
    parser.add_argument('--data', default='data')
    parser.add_argument('--artifacts', default='runs/prepared')
    parser.add_argument('--output', default='runs/benchmarks')
    parser.add_argument('--download', action='store_true', help='Explicitly allow CIFAR/checkpoint acquisition; pretrained weights may also be fetched by prepare')
    parser.add_argument('--split', choices=['validation', 'test'], default='validation')
    parser.add_argument('--selection', help='Validation selection.json; required for test, never retune on test')
    args = parser.parse_args()
    if args.command == 'benchmark' and args.split == 'test' and not args.selection:
        parser.error('--selection is required for held-out test evaluation')
    c = configuration(args.config)
    torch.manual_seed(c['seed'])
    torch.set_num_threads(c['threads'])
    torch.use_deterministic_algorithms(True)
    prepare(args, c) if args.command == 'prepare' else benchmark(args, c)


if __name__ == '__main__':
    main()
