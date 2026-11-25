import copy
import unittest

from adaptive_systems.live_burst import make_trace, run_fifo, summarize, validate_records


class Clock:
    def __init__(self): self.now = 0
    def clock(self): return self.now
    def sleep(self, seconds): self.now += round(seconds * 1e9)


class LiveBurstTests(unittest.TestCase):
    def test_fifo_uses_arrived_backlog_and_includes_queue_wait(self):
        clock = Clock()
        trace = [dict(request_id=i, image_id=i, arrival_ns=i*1_000_000) for i in range(3)]
        thresholds = []
        def predict(image, threshold):
            thresholds.append(threshold)
            clock.now += 10_000_000
            return image, threshold < .5
        rows = run_fifo(trace, [0, 1, 2], dict(thresholds=[.5, .1], depth=1), predict,
                        clock=clock.clock, sleep=clock.sleep)
        self.assertEqual(thresholds, [.5, .1, .5])
        self.assertEqual([r['pending_depth'] for r in rows], [0, 1, 0])
        self.assertEqual([r['response_ns'] for r in rows], [10_000_000, 19_000_000, 28_000_000])
        validate_records(trace, rows)
        self.assertEqual(summarize(trace, rows, [0, 1, 2])['accuracy_pct'], 100)
        bad = copy.deepcopy(rows)
        bad[1]['response_ns'] = bad[1]['service_ns']
        with self.assertRaisesRegex(ValueError, 'boundary'):
            validate_records(trace, bad)

    def test_idle_server_waits_for_scheduled_arrival_and_drains_last(self):
        clock = Clock()
        trace = [dict(request_id=i, image_id=i, arrival_ns=i*20_000_000) for i in range(2)]
        def predict(image, threshold):
            clock.now += 5_000_000
            return image, True
        rows = run_fifo(trace, [0, 1], dict(thresholds=[.5, .5], depth=1), predict,
                        clock=clock.clock, sleep=clock.sleep)
        self.assertEqual([r['wait_ns'] for r in rows], [0, 0])
        self.assertEqual(rows[-1]['finish_ns'], 25_000_000)
        self.assertEqual(len(rows), len(trace))
        self.assertEqual(sum(r['state_changed'] for r in rows), 0)
        bad = copy.deepcopy(rows)
        bad[-1]['image_id'] = 3
        with self.assertRaisesRegex(ValueError, 'identity'):
            validate_records(trace, bad)

    def test_trace_is_deterministic_and_has_burst_and_recovery(self):
        protocol = dict(seed=29, requests=6, warmup=1, period_requests=3, fast_requests=2,
                        fast_gap_ms=1., slow_gap_ms=5., policies={
                            'best_fixed': dict(thresholds=[.5, .5], depth=1),
                            'queue_aware': dict(thresholds=[.5, .1], depth=1)})
        trace = make_trace(protocol, list(range(10)))
        self.assertEqual(trace, make_trace(protocol, list(range(10))))
        self.assertEqual([r['arrival_ns'] for r in trace], [0, 1_000_000, 2_000_000, 7_000_000, 8_000_000, 9_000_000])
        self.assertEqual(len({r['image_id'] for r in trace}), 6)
        with self.assertRaises(ValueError):
            make_trace(protocol, [0]*10)
