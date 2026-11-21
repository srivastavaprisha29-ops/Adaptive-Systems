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
