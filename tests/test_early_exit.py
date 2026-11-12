import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import torch
from torchvision import models

from adaptive_systems.early_exit import load_models


class EarlyExitTests(unittest.TestCase):
    def test_stage_three_exit_skips_layer_four(self):
        with tempfile.TemporaryDirectory() as directory:
            artifacts = Path(directory)
            original = models.resnet18(weights=None).eval()
            original.fc = torch.nn.Identity()
            torch.save(original.state_dict(), artifacts / 'encoder.pt')
            full, early, _ = load_models(artifacts)
            hook = Mock(return_value=None)
            image = torch.zeros(1, 3, 32, 32)
            with full.layer4.register_forward_hook(hook), torch.inference_mode():
                self.assertEqual(early(image).shape, (1, 256))
                hook.assert_not_called()
                torch.testing.assert_close(full(image), original(image))
                hook.assert_called_once()
