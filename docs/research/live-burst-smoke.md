# Reproducible live burst benchmark

The replacement FIFO benchmark runs real ResNet inference against a scheduled burst trace. Its first checked session validates the measurement machinery. It does not validate the broader adaptive-controller hypothesis.

## Locked workload and policies

`configs/burst-smoke.json` freezes 160 distinct validation images selected with seed 20261125, the arrival schedule and both policies. Every 20 interarrival gaps comprise twelve 12 ms burst gaps and eight 96 ms recovery gaps. The first arrival is at time zero. Both policies receive identical ordered image IDs and relative arrival timestamps. Policy execution order is randomized once using the locked seed; in this session queue-aware ran first.

The fixed comparator is the saved historical calibration-selected best fixed threshold, 0.21725761890411377. Queue-aware uses that strict threshold until at least three already-arrived requests wait behind the current request, then uses 0.09145951271057129. These settings are copied from the archived policy lock and are not selected using this run's labels or timings. The runner verifies checkpoint identities against that lock. The comparator was not newly optimized for this machine, and its historical feasibility used point estimates. Calling it best fixed refers to that saved calibration selection, not an exhaustive current-hardware optimum.

## Measurement boundary

A single worker handles requests in FIFO order using a monotonic nanosecond clock. The scheduled arrival list defines an implicit queue: a request is counted as arrived only after its scheduled time has elapsed. No independent producer thread or inferred service-time simulation is used. There is no finite queue capacity, dropping or truncation; the worker drains all scheduled requests.

Response time is `completed scalar prediction - scheduled arrival`. Waiting is `service start - scheduled arrival`. Service includes arrival-backlog observation, threshold selection/state updates, image preprocessing, real early/fallback network execution and scalar prediction conversion. The input images are decoded and preloaded before timing, so disk, network and image decoding are excluded. Both policies use identical instrumentation, including the queue observation on the fixed baseline.

Nested monitoring, decision and state-update timers are diagnostics. They are already inside service time and must not be added again. Switching here changes a scalar threshold; it does not load models or switch feature formats. Both early and fallback paths receive five warmup inferences immediately before each policy trial. Labels and full-reference scoring are accessed after both timed runs.

The runner saves explicit request/image IDs, relative arrival/start/finish timestamps, queue depth, threshold, predicted class, exit decision and nested timing. Trace and checkpoint/source hashes are saved in a lock before policy execution. Raw request JSON is written after each complete trial. Logging between requests can affect subsequent waiting and is part of the observed benchmark execution, while the current request's boundary ends at prediction completion.

## Checked session result

| Metric | Historical best fixed | Queue-aware |
|---|---:|---:|
| Completed requests | 160 | 160 |
| p95 response time | 899.42 ms | 655.21 ms |
| p95 queue waiting | 852.37 ms | 612.66 ms |
| Mean service time | 49.39 ms | 47.58 ms |
| Accuracy | 83.750% | 84.375% |
| Early exits | 55.625% | 76.875% |
| Maximum pending queue | 19 | 15 |
| Actual threshold-state changes | 0 | 12 |
| Mean monitoring cost | 3.26 us | 3.03 us |
| Mean decision cost | 1.67 us | 1.88 us |

The p95 point estimate falls **27.15%** and accuracy rises **0.625 percentage points** relative to fixed. The full reference also achieves 83.750% on these images. The checked session meets the 5%/one-point target by point estimate. The summary explicitly records `research_validated: false` because one short paired session cannot establish session-level uncertainty or robustness. In particular, near queue saturation small service-time differences can produce much larger response-time differences.

An initial diagnostic execution reported a 47.48% p95 reduction. Checks found that its fixed-policy state counter counted queue-mode transitions despite an unchanged threshold. The corrected runner pins fixed policy state and was rerun on the same trace, giving the result above. This is a bookkeeping correction and a technical rerun, not two independent confirmatory sessions. Both observed latency estimates also illustrate sensitivity to execution conditions. The corrected implementation and its raw evidence are the committed artifact.

## Reproduce and verify

Install the pinned project dependencies and obtain the frozen inputs and original CIFAR-10 data as described in [reproducing follow-ups](../reproducing-followups.md). Use the package's `frozen/prepared` directory and `frozen/head.pt`; their digests must match the supplied protocol.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m adaptive_systems.live_burst \
  --artifacts /path/to/package/frozen/prepared \
  --head /path/to/package/frozen/head.pt --data /path/to/data \
  --protocol configs/burst-smoke.json --output runs/live-burst-new

python scripts/verify_live_burst.py --run runs/live-burst-new
python scripts/verify_live_burst.py --run evidence/live-burst-smoke-v1
python -m unittest discover -s tests -v
```

The output directory must be new. Reproduction fixes inputs, policies and arrivals; machine load and timings naturally vary. `lock.json` records machine platform, PyTorch version and single-thread configuration. The verifier recomputes stored metrics, checks hashes, trace identities, FIFO/nonnegative timing, exact response decomposition and policy decisions. It can run without PyTorch.

All **24 tests pass**, including simulated-clock tests for burst backlog, no access to future arrivals, idle waiting, complete queue draining, trace determinism and response-boundary corruption. All 320 live policy predictions and 160 full-reference predictions were additionally checked against independently extracted frozen validation features after timing, with zero mismatches. That cache is used only for this post-run check, never as a substitute for timed inference. The validation images have appeared in earlier research and are not a new untouched test set.

## Next research gate

Freeze a multi-session protocol before further evaluation. Run independent paired sessions with new locked image panels/seeds and balanced policy execution order. Compare p95 improvements at the paired-session level, preserving within-trace dependence, and check paired accuracy loss with the predeclared tolerance and multiplicity handling. A stronger current-machine fixed-policy calibration, multiple workloads and hardware, and measured resource/format switching remain separate requirements of the broader study. The target remains at least 5% p95 improvement with at most one percentage point accuracy loss; this smoke result alone cannot answer that research question.
