import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from audit_controller_validation import audit, paired_accuracy


class ControllerValidationTests(unittest.TestCase):
    def matrix(self, path, mutation=None):
        lock = json.dumps({'workloads': ['bursty-test']}).encode()
        full = dict(requests=[[True, False, 0, 0., 10.]] * 2,
                    accuracy_pct=100., exit_pct=0., p95_ms=10., p95_wait_ms=0.)
        adaptive = dict(requests=[[True, True, 0, 0., 8.], [False, True, 0, 0., 8.]],
                        accuracy_pct=50., exit_pct=100., p95_ms=8., p95_wait_ms=0.)
        with zipfile.ZipFile(path, 'w') as z:
            z.writestr('lock.json', lock)
            for i in range(3):
                trace = dict(session=i, lock_sha256=hashlib.sha256(lock).hexdigest(),
                             indices=[2*i, 2*i+1], results={'bursty-test': dict(
                                 full=copy.deepcopy(full), best_fixed=copy.deepcopy(full),
                                 matched_fixed=copy.deepcopy(full), queue_aware=copy.deepcopy(adaptive))})
                if mutation: mutation(trace)
                z.writestr(f'session-{i}.json', json.dumps(trace))

    def test_latency_gain_cannot_override_quality_failure(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'matrix.zip'
            self.matrix(p)
            r = audit(p, samples=20)
            self.assertEqual(r['burst_summary']['positive_p95_point_estimates'], 3)
            self.assertEqual(r['burst_summary']['interval_quality_passes'], 0)
            self.assertEqual(r['status'], 'broader_controller_not_validated')
            self.assertEqual(r['unique_image_examples'], 6)
            self.assertTrue(all(row['fixed_record_alias'] for row in r['results']))

    def test_overlapping_panels_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'matrix.zip'
            self.matrix(p, lambda t: t.update(indices=[0, 1]))
            with self.assertRaisesRegex(ValueError, 'overlap'):
                audit(p, samples=20)

    def test_tampered_summary_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'matrix.zip'
            self.matrix(p, lambda t: t['results']['bursty-test']['queue_aware'].update(accuracy_pct=100))
            with self.assertRaisesRegex(ValueError, 'accuracy_pct mismatch'):
                audit(p, samples=20)

    def test_paired_difference_handles_identical_predictions(self):
        r = paired_accuracy([True, False, True], [True, False, True], samples=20)
        self.assertEqual(r['delta_pp'], 0)
        self.assertEqual(r['interval_95_pp'], [0., 0.])
        with self.assertRaises(ValueError):
            paired_accuracy([True], [])
