import unittest

import torch
from torchvision.models import resnet18

from adaptive_systems.channel_narrow import narrow_layer4,width_selection


class ChannelNarrowTests(unittest.TestCase):
    def test_convolutions_are_structurally_smaller_and_residual_output_is_preserved(self):
        torch.manual_seed(29)
        original=resnet18(weights=None).eval().layer4
        reduced,kept=narrow_layer4(original,384);reduced.eval()
        self.assertEqual(tuple(reduced[0].conv1.weight.shape),(384,256,3,3))
        self.assertEqual(tuple(reduced[0].conv2.weight.shape),(512,384,3,3))
        self.assertEqual(tuple(reduced[1].conv1.weight.shape),(384,512,3,3))
        self.assertEqual(tuple(reduced[1].conv2.weight.shape),(512,384,3,3))
        self.assertEqual(original[0].conv1.out_channels,512)
        self.assertLess(sum(p.numel() for p in reduced.parameters()),sum(p.numel() for p in original.parameters()))
        self.assertTrue(all(p.dtype==torch.float32 for p in reduced.parameters()))
        with torch.inference_mode():
            self.assertEqual(tuple(reduced(torch.randn(1,256,14,14)).shape),(1,512,7,7))
        for i in range(2):
            ids=torch.tensor(kept[str(i)])
            torch.testing.assert_close(reduced[i].conv1.weight,original[i].conv1.weight[ids])
            torch.testing.assert_close(reduced[i].conv2.weight,original[i].conv2.weight[:,ids])

    def test_full_width_copy_preserves_output_and_rejects_invalid_shapes(self):
        original=resnet18(weights=None).eval().layer4
        reference,kept=narrow_layer4(original,512)
        self.assertEqual(kept,{})
        x=torch.randn(1,256,14,14)
        with torch.inference_mode():torch.testing.assert_close(original(x),reference(x),rtol=0,atol=0)
        for width in (0,513,1.5):
            with self.assertRaises(ValueError):narrow_layer4(original,width)

    def test_calibration_selects_smallest_qualified_or_flags_failure(self):
        values=[dict(width=256,accuracy_pct=80.,quality_pass=False),
                dict(width=384,accuracy_pct=82.,quality_pass=True),dict(width=448,accuracy_pct=83.,quality_pass=True)]
        selected,qualified=width_selection(values)
        self.assertEqual(selected['width'],384);self.assertTrue(qualified)
        selected,qualified=width_selection([dict(v,quality_pass=False) for v in values])
        self.assertEqual(selected['width'],448);self.assertFalse(qualified)
