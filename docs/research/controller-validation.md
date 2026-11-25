# Adaptive-controller validation status

**The broader controller is not validated.** Fixed early-exit and representation experiments establish useful components. Distillation improves its matched supervised control but remains weaker than the existing ridge early head. Those results do not establish resource-conditioned width/precision selection or net controller benefit.

A [replacement reproducible live burst benchmark](live-burst-smoke.md) now completes one checked paired session with explicit arrival-to-prediction timing. This repairs the ability to run a new experiment; the missing historical generator is not reconstructed exactly, and independent-session research validation remains outstanding.

A [six-session confidence-safeguard study](confidence-safeguard.md) now tests prediction failures, a calibration-frozen confidence floor, a fixed confidence-gate ablation and steady load. The guarded controller fails the joint target: burst p95 is 21.16% worse on average than the historical best fixed comparator, with unchanged mean accuracy. It is not promoted.

A [controlled relaxation replay](calibration-bounded-relaxation.md) now keeps the default gate active and permits a calibration-bounded relaxation under pressure. It restores exits but fails the adoption target: burst p95 is 3.16% worse than best fixed. The original anomalous steady panel was also replayed with its original policies.

The [fresh-trace frozen load sweep](frozen-load-sweep.md) reselects the fixed baseline on the current machine and tests below/near/above calibration capacity. No load range clears the uncertainty-aware target. The selected fixed gate is retained, and the research direction returns to width/precision action profiling.

## Audit of the saved live queue matrix

The executable audit verifies the policy-lock checksum, disjoint session image panels, stored request counts and summary arithmetic. It recomputes paired accuracy intervals relative to both the full model and the saved calibration-selected fixed comparator. It identifies aliased matched-fixed records so they are not counted as independent executions.

| Session | Burst workload | p95 reduction vs saved best fixed | Accuracy delta vs full (pp) | Paired 95% interval (pp) |
|---|---|---:|---:|---|
| 0 | bursty-65 | 3.80% | +0.75 | [-1.00, 2.75] |
| 0 | bursty-90 | 36.06% | +1.00 | [-1.00, 3.00] |
| 1 | bursty-65 | 13.61% | -1.50 | [-3.50, 0.50] |
| 1 | bursty-90 | 10.87% | -1.50 | [-3.50, 0.50] |
| 2 | bursty-65 | 8.99% | -2.50 | [-4.50, -0.75] |
| 2 | bursty-90 | 18.56% | -2.50 | [-4.50, -0.75] |

Six of six burst p95 point estimates improve. Only two of six meet the one-percentage-point quality tolerance, both by point estimate and by the interval lower bound. The 2,000-resample percentile intervals use a fixed seed of 29 and image-paired correctness. These are conditional, unadjusted intervals; they are not simultaneous guarantees across workloads. In-session workloads reuse the same 400 images, so the matrix contains 1,200 distinct images rather than 4,800 independent image observations. Per-policy row order is assumed to follow the shared image-index list because individual records lack explicit image IDs.

Queue latencies are temporally dependent. The audit does not bootstrap individual request latencies as independent trials or infer a reliable latency confidence interval from three sessions. The named best-fixed comparator was selected historically during calibration; this audit does not establish its global optimality.

## Reproduce the audit

Extract the existing [reproducibility evidence package](https://chatgpt.com/library?file_id=libfile_3cefc90f4ef48191a84cc47c5f061aad&account_id=personal), then run from the repository root:

```bash
python scripts/audit_controller_validation.py \
  --matrix /path/to/evidence/Adaptive_Systems_Queue_Matrix.zip \
  --output runs/controller-validation-audit.json
python -m unittest discover -s tests -v
```

The committed `controller-validation-audit.json` contains all twelve session/workload comparisons and input/source hashes. This audit recomputes existing records. It does not rerun inference or recreate the missing original queue generator.

## Next live validation protocol

The following is a concrete proposed protocol for the replacement runner. Its execution is still outstanding.

1. **Freeze implementation and policy selection.** Recover or reconstruct the live FIFO runner in version control, with request IDs, arrival/dequeue/finish timestamps, image IDs, action, confidence, queue depth, policy state and checkpoint/source hashes. Record calibration-selected fixed and dynamic settings before evaluation. Keep test labels inaccessible to policy decisions. Use the strongest existing early head as a starting candidate.
2. **Measure the full boundary.** Response time runs from scheduled arrival to completed prediction and includes waiting, telemetry, decision, preprocessing, network execution and switching. Record nested timings for attribution and an outer timer for net comparisons. Record missed arrivals, timeouts, dropped requests and failures; do not silently exclude them. Downstream feature caches may support training but cannot substitute for inference in this benchmark.
3. **Match comparators.** Run the full model, calibration-selected feasible fixed action, calibrated fixed margin gate, threshold controller and profile-driven controller on identical image/arrival traces. Randomize policy execution order within paired sessions. Hysteresis-on/off and monitoring-only ablations isolate oscillation and overhead. Hold memory and numeric-format claims for actions that have measured compatible implementations.
4. **Separate selection and evaluation.** Choose thresholds, dwell times, profiles and workload rates using calibration runs only. Lock evaluation seeds and trace generation before seeing evaluation labels or timings. Historical CIFAR test results have already been inspected for fixed policies, so describe the dataset as previously studied rather than globally untouched. New controller sequences must remain excluded from tuning.
5. **Stress changes and negative controls.** Include steady, bursty, gradual ramp, rapid oscillation, measured memory pressure and a low-benefit regime. Observe real queue/resource signals. Do not supply future arrivals or future resource states to the controller. Log infeasible actions and explicit fallback outcomes.
6. **Predeclare joint acceptance.** Use the existing pilot thresholds of at most one accuracy point loss and at least 5% p95 response-time improvement over the calibration-selected feasible fixed comparator. These thresholds are research assumptions. Require a quality lower bound above the tolerance and a paired-session latency lower bound above the improvement target in every claimed regime. Decide multiplicity handling before running; report failed regimes and steady-state regressions. A favorable point estimate alone cannot pass.
7. **Use independent experimental units.** Plan at least ten independent paired sessions per claimed workload on a named hardware configuration. Estimate uncertainty by resampling paired sessions or complete independent traces, preserving request order within each trace. Report sensitivity to session count; ten sessions alone do not ensure adequate statistical power. Validate on a second hardware configuration before making portability claims.

Energy in joules and peak memory require separate instrumentation. Until those measurements and actual width/precision controller trials exist, report queue-aware early-exit findings under their narrower scope.
