# Run the static pilot

Use Python 3.10–3.12 on the target CPU. The commands below start from the repository root. The direct scientific dependencies are pinned in `pyproject.toml`; each run records installed versions. Preserve a full environment freeze for archival reproduction.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m pip freeze > environment-freeze.txt
python -m unittest discover -s tests -v
adaptive-static prepare --config configs/static.json --artifacts runs/prepared --download
adaptive-static benchmark --config configs/static.json --artifacts runs/prepared
```

Preparation downloads CIFAR-10 and ImageNet-pretrained ResNet-18 weights if needed, extracts training/calibration features, fits PCA and four classifiers, and saves provenance. The 224px pretrained input transform is intentionally used for all actions. Full CPU preparation can take substantial time; this is a feasibility study, not a promise of a quick benchmark. Feature caches during fitting also consume RAM. Use an otherwise idle machine, record its power/thermal conditions, and keep the config unchanged between preparation and benchmarking.

Benchmark prints a unique validation run directory containing `briefing.md`, `summary.json`, `selection.json`, `timings.jsonl`, `correctness.pt`, and `manifest.json`. Review its quality floor, conversion cost, encoder share and repeat variability. To evaluate the **locked validation selection** once on the official test set, replace the path below with the printed validation directory:

```bash
adaptive-static benchmark --config configs/static.json --artifacts runs/prepared --split test --selection runs/benchmarks/validation-RUN_ID/selection.json
```

Do not change the config or refit after seeing test results. New candidate development requires a new development/evaluation protocol. Prepared artifacts and datasets can move: verification resolves relative filenames beneath the current `--artifacts` and `--data` locations. Manifest schema 2 is required; regenerate artifacts created by the original scaffold. Back up `runs/` and data outside Git; they are intentionally ignored. `environment-freeze.txt` may be included with a reviewed experiment artifact, not mistaken for a universal cross-platform lock.

## Interpreting output

- `accuracy_pct`, `delta_accuracy_pp`, and `delta_accuracy_95ci_pp`: quality and paired uncertainty relative to full FP32.
- `quality_feasible`: lower interval bound is within the configured degradation margin.
- `p50_ms`, `p95_ms`, `repeat_mean_ms`, `throughput_per_s`: complete declared inference boundary. Throughput is sequential, not server concurrency throughput.
- `feature_payload_bytes`: actual adapted tensor storage, excluding metadata and other allocations.
- `max_observed_rss_bytes`: largest post-inference RSS sample, with all configurations resident; not peak or per-action memory.
- `selected_mean_latency_reduction_95ci_pct`: paired repeat-bootstrap interval; the pilot requires its lower bound to meet the configured useful improvement. Repeats within one session do not establish cross-session reproducibility.
- `baseline_rss_bytes`, `resident_tensor_storage_bytes`, `action_tensor_storage_bytes`: baseline process observation and unique backing tensor-storage inventory. They do not replace peak memory profiling.
- `pilot_pass`: provisional quality and uncertainty-aware latency gate; not a validated research claim.
- `diagnostic_total_ms` and stage timings: separate diagnostic inference; headline `total_ms` uses a single outer timer. Diagnostic work is excluded from headline throughput.

Energy and controller performance remain unmeasured. Synthetic tests validate plumbing only and produce no publishable accuracy or efficiency evidence. See [the contract](experiment-contract.md) before interpreting any result.

## Stage-three early-exit pilot

This follow-up compares the original full-width FP32 head with a separate FP32 ridge head on the globally pooled 256-channel output of ResNet-18's third residual stage. The candidate skips the fourth stage. Both paths use the same pretrained encoder weights, input transform, and CIFAR-10 splits as the static pilot.

Reuse the checksummed static preparation artifacts. The early-exit preparation fits its head on the training split and records paired validation accuracy. Each fresh benchmark process then measures both paths on the same examples in a randomized order.

```bash
python -m adaptive_systems.early_exit prepare --artifacts runs/prepared --output runs/early-exit/head.pt
for seed in 29 30 31; do
  python -m adaptive_systems.early_exit benchmark --artifacts runs/prepared \
    --head runs/early-exit/head.pt --output "runs/early-exit/session-${seed}.json" \
    --order-seed "$seed" || break
done
```

Use the paired validation accuracy interval and the paired repeat-mean latency intervals to assess the configured gates. Fresh processes on one CPU check session-to-session consistency. This exploratory command evaluates only the validation split.

## Confidence-gated stage-three exit

The gate uses the margin between the two largest stage-three logits. It selects
the highest-coverage threshold that meets the paired accuracy criterion on the
calibration split. The threshold is then fixed for validation and timing.
Uncertain inputs continue from the stage-three activation through stage four.

```bash
python -m adaptive_systems.confidence_exit --artifacts runs/prepared \
  --head runs/early-exit/head.pt --output runs/confidence-exit/validation.json
```

The JSON records calibration and validation accuracy, early-exit coverage, and
paired latency measurements. The latency interval covers repeats within one
process. Re-run in fresh sessions before drawing a cross-session conclusion.

In the held-out CIFAR-10 run, the frozen threshold exited early on 41.21% of
10,000 images. Accuracy was 84.63% gated and 84.52% full, a paired difference of
+0.11 percentage points (95% interval −0.07 to +0.29). In three fresh
single-thread Intel Xeon Platinum 8573C sessions, median validation latency
fell by 6.7% to 14.9%. Only one session met the 5% mean-latency reduction
target at the lower confidence bound. The held-out accuracy target was met,
and the latency target remains unconfirmed across sessions.
