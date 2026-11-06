import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from adaptive_systems.experiment import (benchmark, classify, configuration, decode,
    encode, fit_states, paired_quality_interval, partition, prepare)


class FakeCIFAR:
    def __init__(self, root, train=True, download=False):
        self.targets = list(range(10)) * (5 if train else 2)
    def __getitem__(self, index):
        image = Image.fromarray(np.full((8, 8, 3), index % 10 * 20, dtype=np.uint8))
        return image, self.targets[index]
    def __len__(self):
        return len(self.targets)


def transform(image):
    return torch.tensor(np.array(image).copy()).permute(2, 0, 1).float() / 255


class TinyEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('basis', torch.linspace(0, 1, 512))
        self.fc = torch.nn.Identity()
    def forward(self, x):
        return x.mean((1, 2, 3))[:, None] * self.basis[None, :]


class StaticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_split_is_disjoint_complete_reproducible(self):
        labels = list(range(10)) * 10
        first = partition(labels, 29, 2, 1)
        self.assertEqual(first, partition(labels, 29, 2, 1))
        self.assertEqual(sorted(sum(first, [])), list(range(100)))
        self.assertEqual(len(set(first[0]) & set(first[1])), 0)
        with self.assertRaises(ValueError):
            partition(labels, 29, 9, 1)

    def test_quantization_error_and_real_payload(self):
        x = torch.tensor([[-1., -.31, 0., .31, 1.]])
        packed = encode(x, 'int8_storage', 1 / 127)
        self.assertEqual(packed.dtype, torch.int8)
        self.assertEqual(packed.numel() * packed.element_size(), 5)
        self.assertLessEqual(float((decode(packed, 'int8_storage', 1/127)-x).abs().max()), 1/254+1e-6)
        with self.assertRaises(ValueError):
            encode(x, 'unknown', 1)

    def test_joint_grid_and_paired_quality(self):
        torch.manual_seed(29)
        train, cal = torch.randn(40, 8), torch.randn(10, 8)
        states = fit_states(train, torch.arange(40) % 10, cal, 3, 1)
        self.assertEqual(len(states), 4)
        for state in states.values():
            result = classify(cal, state)
            self.assertEqual(result.shape, (10, 10))
            self.assertTrue(torch.isfinite(result).all())
        reference = torch.ones(20, dtype=torch.bool)
        delta, ci = paired_quality_interval(reference, ~reference, 100, 29)
        self.assertEqual(delta, -100)
        self.assertEqual(ci, [-100, -100])

    def test_prepare_benchmark_and_heldout_selection(self):
        # Synthetic wiring test only: never use these values as research evidence.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c = dict(seed=29, device='cpu', threads=1, reduced_width=2, ridge=1.,
                validation_per_class=1, calibration_per_class=1, quality_tolerance_pp=1.,
                minimum_latency_reduction_pct=5., warmup=1, timed_examples=2,
                repeats=2, bootstrap_samples=20)
            config = root / 'config.json'
            config.write_text(json.dumps(c))
            self.assertEqual(configuration(config), c)
            args = SimpleNamespace(artifacts=str(root/'prepared'), data=str(root/'data'),
                output=str(root/'benchmarks'), download=False, split='validation', selection=None)
            with patch('adaptive_systems.experiment.datasets.CIFAR10', FakeCIFAR), \
                 patch('adaptive_systems.experiment.backbone', lambda: (TinyEncoder(), transform)), \
                 patch('adaptive_systems.experiment.models.resnet18', lambda **kw: TinyEncoder()), \
                 patch('adaptive_systems.experiment.models.ResNet18_Weights') as weights, \
                 contextlib.redirect_stdout(io.StringIO()):
                weights.IMAGENET1K_V1.transforms.return_value = transform
                prepare(args, c)
                benchmark(args, c)
                validation = next((root/'benchmarks').glob('validation-*'))
                args.selection = str(validation/'selection.json')
                args.split = 'test'
                benchmark(args, c)
                test = next((root/'benchmarks').glob('test-*'))
                summary = json.loads((test/'summary.json').read_text())
                selection = json.loads(Path(args.selection).read_text())
                self.assertEqual(summary['selected_static'], selection['configuration'])
                self.assertEqual(len(summary['metrics']), 4)
                self.assertEqual(len((test/'timings.jsonl').read_text().splitlines()), 16)
                # Mutating a fitted artifact must fail before evaluation.
                with open(root/'prepared'/'states.pt', 'ab') as f:
                    f.write(b'changed')
                with self.assertRaisesRegex(ValueError, 'checksum'):
                    benchmark(args, c)


if __name__ == '__main__':
    unittest.main()
