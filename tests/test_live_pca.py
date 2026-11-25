import unittest

import torch

from adaptive_systems.live_pca import fit_states,logits_for,select_width,timing_comparison


class LivePCATests(unittest.TestCase):
    def test_matched_heads_use_orthonormal_training_pca_prefixes(self):
        torch.manual_seed(29)
        x=torch.randn(24,8);labels=torch.arange(24)%10
        states=fit_states(x,labels,[2,4],1.)
        self.assertIsNone(states[8]['projection'])
        torch.testing.assert_close(states[2]['mean'],x.mean(0))
        torch.testing.assert_close(states[4]['projection'].T@states[4]['projection'],torch.eye(4),atol=1e-6,rtol=1e-5)
        torch.testing.assert_close(states[2]['projection'],states[4]['projection'][:,:2])
        for width,state in states.items():
            self.assertEqual(state['weight'].shape,(width,10))
            self.assertEqual(logits_for(x,state).shape,(24,10))
            self.assertEqual(logits_for(x,state).dtype,torch.float32)

    def test_selection_prefers_smallest_feasible_and_flags_infeasible_diagnostic(self):
        choices=[dict(width=64,quality_pass=False,accuracy_pct=70),
                 dict(width=128,quality_pass=True,accuracy_pct=80),
                 dict(width=256,quality_pass=True,accuracy_pct=81)]
        chosen,qualified=select_width(choices)
        self.assertEqual(chosen['width'],128);self.assertTrue(qualified)
        failed=[dict(c,quality_pass=False) for c in choices]
        chosen,qualified=select_width(failed)
        self.assertEqual(chosen['width'],256);self.assertFalse(qualified)

    def test_equivalence_and_useful_latency_gain_are_separate(self):
        settings=dict(seed=29,bootstrap_samples=100,p95_target_pct=5)
        sessions=[dict(metrics=dict(full_fp32=dict(p95_response_ms=100),pca_fp32=dict(p95_response_ms=100))) for _ in range(6)]
        result=timing_comparison(sessions,settings)
        self.assertFalse(result['latency_target_pass'])
        self.assertTrue(result['interval_within_plus_minus_5pct'])
        for s in sessions:s['metrics']['pca_fp32']['p95_response_ms']=90
        result=timing_comparison(sessions,settings)
        self.assertTrue(result['latency_target_pass'])
        self.assertFalse(result['interval_within_plus_minus_5pct'])
