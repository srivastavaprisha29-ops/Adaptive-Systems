import unittest

import torch
from torch.nn import functional as F

from adaptive_systems.distillation import distillation_loss, fit


class DistillationTests(unittest.TestCase):
    def test_soft_gradient_has_temperature_scaling_and_detaches_teacher(self):
        student = torch.tensor([[1., -2., 3.], [0., 2., -1.]], requires_grad=True)
        teacher = torch.tensor([[2., 0., -1.], [-1., 1., 3.]], requires_grad=True)
        labels = torch.tensor([0, 2])
        loss = distillation_loss(student, teacher, labels, temperature=4., alpha=1.)
        loss.backward()
        expected = 4. / 2 * ((student.detach() / 4).softmax(-1) - (teacher.detach() / 4).softmax(-1))
        torch.testing.assert_close(student.grad, expected)
        self.assertIsNone(teacher.grad)

    def test_zero_alpha_matches_supervised_cross_entropy(self):
        student = torch.randn(4, 3)
        target = torch.tensor([0, 1, 2, 0])
        torch.testing.assert_close(distillation_loss(student, torch.randn(4, 3), target, alpha=0),
                                   F.cross_entropy(student, target))

    def test_invalid_hyperparameters_fail(self):
        for temperature, alpha in [(0, .7), (-1, .7), (float('nan'), .7), (4, 1.1), (4, -.1)]:
            with self.assertRaises(ValueError):
                distillation_loss(torch.zeros(2, 3), torch.zeros(2, 3), torch.zeros(2, dtype=torch.long),
                                  temperature, alpha)

    def test_matched_seed_repeats_training(self):
        torch.manual_seed(5)
        x, teacher = torch.randn(8, 4), torch.randn(8, 3)
        target = torch.arange(8) % 3
        a = fit(x, teacher, target, 29, 2, 4., .7)
        b = fit(x, teacher, target, 29, 2, 4., .7)
        for key in a:
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)

    def test_convergence_trace_and_cosine_schedule(self):
        torch.manual_seed(5)
        x, teacher = torch.randn(16, 4), torch.randn(16, 3)
        target = torch.arange(16) % 3
        history = []
        traced = fit(x, teacher, target, 29, 6, 4., .7, history=history, cosine=True)
        plain = fit(x, teacher, target, 29, 6, 4., .7, cosine=True)
        self.assertEqual([row['epoch'] for row in history], [1, 5, 6])
        self.assertGreater(history[0]['lr'], history[-1]['lr'])
        for row in history:
            self.assertTrue(torch.isfinite(torch.tensor(row['loss'])))
            self.assertTrue(0 <= row['train_accuracy_pct'] <= 100)
        for key in traced:
            torch.testing.assert_close(traced[key], plain[key], rtol=0, atol=0)
