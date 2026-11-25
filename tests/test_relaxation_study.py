import unittest
from unittest.mock import patch

import torch

from adaptive_systems.live_burst import run_fifo
from adaptive_systems.relaxation_study import adoption_bounds, calibrate_limit


class RelaxationStudyTests(unittest.TestCase):
    def test_default_gate_stays_active_and_pressure_cannot_bypass_floor(self):
        now=[0]
        margins=[.5,.1,.5]
        def predict(image,threshold):
            now[0]+=10_000_000
            return image,margins[image]>=threshold,dict(margin=margins[image])
        trace=[dict(request_id=i,image_id=i,arrival_ns=i*1_000_000) for i in range(3)]
        rows=run_fifo(trace,list(range(3)),dict(thresholds=[.4,.2],depth=1),predict,
                      clock=lambda:now[0],sleep=lambda _:None)
        self.assertEqual([r['threshold'] for r in rows],[.4,.2,.4])
        self.assertEqual([r['early_exit'] for r in rows],[True,False,True])

    def test_calibration_rejects_overly_harmful_relaxation(self):
        logits=torch.tensor([[.1,0],[.2,0],[.3,0],[.4,0]])
        labels=torch.tensor([1,0,1,0])
        full=torch.nn.functional.one_hot(labels,2).float()
        def exact(reference,candidate,*args):
            delta=100*(candidate.double()-reference.double()).mean().item()
            return delta,[delta,delta]
        config=dict(quality_tolerance_pp=10,bootstrap_samples=20,seed=29)
        with patch('adaptive_systems.experiment.paired_quality_interval',side_effect=exact):
            limit,candidates=calibrate_limit(logits,full,labels,.35,.25,config)
        self.assertGreater(limit,.1)
        self.assertLessEqual(limit,.2)
        self.assertTrue(all(c['interval_95_pp'][0]>=-10 for c in candidates if c['qualified']))
        self.assertFalse(candidates[0]['qualified'])

    def test_adoption_requires_improvement_over_always_active_gate(self):
        metrics=dict(best_fixed=dict(p95_response_ms=100,accuracy_pct=90),
                     fixed_confidence=dict(p95_response_ms=80,accuracy_pct=90),
                     safeguarded_queue=dict(p95_response_ms=90,accuracy_pct=90))
        sessions=[dict(metrics=dict(burst=metrics,steady=metrics)) for _ in range(6)]
        settings=dict(bootstrap_samples=100,seed=29,p95_target_pct=5,quality_tolerance_pp=1)
        result=adoption_bounds(sessions,settings)
        self.assertGreater(result['lower_bounds']['best_fixed_gain_lower'],5)
        self.assertLess(result['lower_bounds']['fixed_confidence_gain_lower'],0)
        self.assertFalse(result['queue_awareness_supported'])
