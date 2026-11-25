import unittest

from adaptive_systems.load_sweep import analyze,make_plans


class LoadSweepTests(unittest.TestCase):
    def settings(self):
        return dict(seed=7,sessions_per_load=6,requests=4,
                    nominal_loads=dict(below=.65,near=.95,above=1.15),bootstrap_samples=100,
                    quality_tolerance_pp=1,p95_target_pct=5,family_alpha=.05,planned_adoption_bounds=12)

    def test_fresh_disjoint_panels_balanced_orders_and_capacity_gaps(self):
        settings=self.settings()
        plans=make_plans(settings,10.,list(range(100)))
        self.assertEqual(plans,make_plans(settings,10.,list(range(100))))
        ids=[r['image_id'] for p in plans for r in p['trace']]
        self.assertEqual(len(ids),72)
        self.assertEqual(len(set(ids)),72)
        for level,load in settings['nominal_loads'].items():
            selected=[p for p in plans if p['level']==level]
            self.assertEqual(len({tuple(p['order']) for p in selected}),6)
            self.assertTrue(all(p['gap_ns']==round(10./load*1e6) for p in selected))
        with self.assertRaises(ValueError):make_plans(settings,10.,[0]*100)

    def test_latency_win_cannot_override_quality_loss_in_any_range(self):
        settings=self.settings();records=[]
        for level in settings['nominal_loads']:
            for i in range(6):
                records.append(dict(level=level,metrics=dict(
                    best_fixed=dict(p95_response_ms=100,accuracy_pct=90),
                    always_active=dict(p95_response_ms=110,accuracy_pct=90),
                    bounded_queue=dict(p95_response_ms=80,accuracy_pct=88))))
        result=analyze(records,settings)
        self.assertTrue(all(not row['claimed_win'] for row in result.values()))

    def test_claims_are_specific_to_load_range_and_require_both_comparators(self):
        settings=self.settings();records=[]
        for level in settings['nominal_loads']:
            for i in range(6):
                records.append(dict(level=level,metrics=dict(
                    best_fixed=dict(p95_response_ms=100,accuracy_pct=90),
                    always_active=dict(p95_response_ms=70 if level=='below' else 110,accuracy_pct=90),
                    bounded_queue=dict(p95_response_ms=80,accuracy_pct=90))))
        result=analyze(records,settings)
        self.assertFalse(result['below']['claimed_win'])
        self.assertTrue(result['near']['claimed_win'])
        self.assertTrue(result['above']['claimed_win'])
        with self.assertRaises(ValueError):analyze(records[:-1],settings)
