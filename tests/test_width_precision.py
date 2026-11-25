import unittest
import torch

from adaptive_systems.width_precision import ACTIONS,analyze,quantize_storage,reconstruct


class WidthPrecisionTests(unittest.TestCase):
    def test_storage_roundtrip_is_int8_then_float32_and_counts_actual_payload(self):
        x=torch.tensor([[-1.,-.25,0.,.25,1.]],dtype=torch.float32)
        scale=1/127
        q=quantize_storage(x,scale);decoded=reconstruct(q,scale)
        self.assertEqual(q.dtype,torch.int8)
        self.assertEqual(decoded.dtype,torch.float32)
        self.assertEqual(q.numel()*q.element_size(),5)
        self.assertLessEqual((decoded-x).abs().max().item(),scale/2+1e-7)
        self.assertTrue(torch.isfinite(decoded).all())
        with self.assertRaises(ValueError):quantize_storage(x,0)
        with self.assertRaises(ValueError):reconstruct(x,scale)

    def test_clipping_is_symmetric_and_does_not_refit_scale_to_test_values(self):
        q=quantize_storage(torch.tensor([[-100.,100.]]),.1)
        self.assertEqual(q.tolist(),[[-127,127]])
        torch.testing.assert_close(reconstruct(q,.1),torch.tensor([[-12.7,12.7]]))

    def fixture(self,costs):
        settings=dict(seed=29,repeats=4,loads=dict(below=.65,near=.95,above=1.15),
                      bootstrap_samples=100,family_alpha=.05,p95_target_pct=5,quality_tolerance_pp=1)
        quality={a:dict(accuracy_pct=90.,pairwise={b:dict(interval_95_pp=[0.,0.],simultaneous_lower_pp=0.)
                   for b in ACTIONS if b!=a}) for a in ACTIONS}
        sessions=[dict(level=level,metrics={a:dict(p95_response_ms=costs[level][a]) for a in ACTIONS})
                  for level in settings['loads'] for _ in range(4)]
        return sessions,quality,settings

    def test_universal_fixed_winner_does_not_require_a_controller(self):
        costs={level:{a:(50 if a=='full_fp32' else 100) for a in ACTIONS} for level in ['below','near','above']}
        sessions,quality,settings=self.fixture(costs)
        result=analyze(sessions,quality,'full_fp32',settings)
        self.assertEqual(result['supported_universal_fixed'],['full_fp32'])
        self.assertFalse(result['supported_crossovers'])

    def test_crossover_requires_latency_and_quality_both(self):
        costs={level:{a:(50 if a==('full_fp32' if level=='below' else 'full_int8_storage') else 100) for a in ACTIONS}
               for level in ['below','near','above']}
        sessions,quality,settings=self.fixture(costs)
        self.assertTrue(analyze(sessions,quality,'full_fp32',settings)['supported_crossovers'])
        for a in ACTIONS:
            for b in ACTIONS:
                if a!=b:quality[a]['pairwise'][b]['simultaneous_lower_pp']=-2.
        result=analyze(sessions,quality,'full_fp32',settings)
        self.assertFalse(result['supported_crossovers'])
        self.assertFalse(result['supported_universal_fixed'])
