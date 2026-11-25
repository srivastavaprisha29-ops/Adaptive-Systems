import unittest

from adaptive_systems.live_burst import run_fifo
from adaptive_systems.safeguard_study import compare, error_groups, paired_session_interval


class SafeguardStudyTests(unittest.TestCase):
    def test_error_groups_count_only_early_errors_the_full_model_corrects(self):
        rows=[dict(confidence=dict(margin=.15,early_prediction=1),pending_depth=0,label=0,full_prediction=0,early_exit=True),
              dict(confidence=dict(margin=.15,early_prediction=1),pending_depth=3,label=0,full_prediction=0,early_exit=False),
              dict(confidence=dict(margin=.4,early_prediction=1),pending_depth=8,label=0,full_prediction=2,early_exit=True)]
        groups=error_groups(rows,[0.,.1,.2,.3,.5],[0,1,3,8])
        self.assertEqual(groups['margin-bin-1/queue-bin-0']['harmful_actual_exits'],1)
        self.assertEqual(groups['margin-bin-1/queue-bin-2']['early_wrong_full_right'],1)
        self.assertEqual(groups['margin-bin-1/queue-bin-2']['harmful_actual_exits'],0)
        self.assertEqual(groups['margin-bin-3/queue-bin-3']['early_wrong_full_right'],0)

    def test_paired_session_constant_effect_has_exact_bounds(self):
        result=paired_session_interval([7.]*6,100,29)
        self.assertEqual(result['mean'],7.)
        self.assertEqual(result['interval_95'],[7.,7.])
        with self.assertRaises(ValueError): paired_session_interval([7.],100,29)

    def test_latency_cannot_override_accuracy_endpoint(self):
        metrics=dict(best_fixed=dict(p95_response_ms=100,accuracy_pct=90),
                     fixed_confidence=dict(p95_response_ms=95,accuracy_pct=91),
                     safeguarded_queue=dict(p95_response_ms=80,accuracy_pct=88))
        sessions=[dict(metrics=dict(burst=metrics,steady=metrics)) for _ in range(6)]
        report=compare(sessions,dict(bootstrap_samples=100,seed=29,p95_target_pct=5,quality_tolerance_pp=1))
        primary=report['burst']['safeguarded_queue_vs_best_fixed']
        self.assertGreater(primary['latency_reduction_pct']['mean'],5)
        self.assertFalse(primary['joint_lower_bound_pass'])

    def test_none_threshold_falls_back_and_confidence_is_recorded(self):
        now=[0]
        def predict(image,threshold):
            now[0]+=10_000_000
            margin=.1
            exited=threshold is not None and margin>=threshold
            return 1 if exited else 0,exited,dict(margin=margin,early_prediction=1,top_probability=.4)
        trace=[dict(request_id=i,image_id=i,arrival_ns=i*1_000_000) for i in range(3)]
        rows=run_fifo(trace,[0,1,2],dict(thresholds=[None,.2],depth=1),predict,
                      clock=lambda:now[0],sleep=lambda _:None)
        self.assertEqual([r['threshold'] for r in rows],[None,.2,None])
        self.assertTrue(all(not r['early_exit'] for r in rows))
        self.assertTrue(all(r['confidence']['margin']==.1 for r in rows))
