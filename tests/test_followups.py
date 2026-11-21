import unittest

import torch
from torch.nn import functional as F

from adaptive_systems.learned_exit import exit_score, fit_gate, logit_features
from adaptive_systems.matryoshka import NestedProjection


class FollowupTests(unittest.TestCase):
    def test_compact_projection_matches_full_prefix(self):
        torch.manual_seed(29)
        model = NestedProjection(8, (2, 4)).eval()
        features = torch.randn(5, 8)
        compact = F.linear(features, model.projection.weight[:2], model.projection.bias[:2])
        torch.testing.assert_close(model.heads['2'](compact), model(features)['2'])

    def test_gate_exits_when_continuing_has_negative_benefit(self):
        features = torch.tensor([[-1.], [0.], [1.]])
        early = torch.tensor([True, True, False])
        full = torch.tensor([False, True, True])
        scores = exit_score(features, fit_gate(features, early, full, 1))
        self.assertGreater(scores[0].item(), scores[1].item())
        self.assertGreater(scores[1].item(), scores[2].item())
        self.assertEqual(logit_features(torch.zeros(3, 10)).shape, (3, 12))

    def test_joint_loss_updates_every_prefix_and_roundtrips_weights(self):
        import io

        torch.manual_seed(29)
        model = NestedProjection(8, (2, 4))
        features = torch.randn(6, 8)
        labels = torch.tensor([0, 1, 2, 3, 4, 5])
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        outputs = model(features)
        loss = sum(F.cross_entropy(logits, labels) for logits in outputs.values())
        loss.backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(parameter.grad.abs().sum().item(), 0, name)
        before = model.projection.weight.detach().clone()
        optimizer.step()
        self.assertFalse(torch.equal(before, model.projection.weight))

        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        restored = NestedProjection(8, (2, 4))
        restored.load_state_dict(torch.load(buffer, weights_only=True))
        for width, logits in model(features).items():
            torch.testing.assert_close(restored(features)[width], logits)

    def test_constant_gate_features_remain_finite(self):
        features = torch.zeros(6, 12)
        correct = torch.ones(6, dtype=torch.bool)
        gate = fit_gate(features, correct, correct, 1)
        scores = exit_score(features, gate)
        self.assertTrue(torch.isfinite(scores).all())
        torch.testing.assert_close(scores, torch.zeros(6))
