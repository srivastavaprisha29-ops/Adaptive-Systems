# Frozen compact encoder across load levels

This follow-up freezes the MobileNetV3-Small FP32 body and ridge classifier from the architecture pilot. Its references are the original full-width ResNet-18 FP32 pipeline and the frozen grid winner, reduced-width INT8 feature storage with reconstruction to FP32 before classification. No models, scales, projections or classifiers are fitted during this study. All three retain the original common input preprocessing and training split.

This is an architecture comparison, not an isolated ResNet channel-width ablation. The compact model produces a **576D FP32 / 2304-byte embedding**, versus ResNet-18's **512D FP32 / 2048 bytes**. A latency improvement therefore cannot be attributed to shrinking the final vector.

## Protocol

Only original calibration images establish the current host's service capacity: 24 images, three passes, six warmup requests, median wall service time. Arrival schedules cover nominal utilizations 0.65, 0.95 and 1.15 relative to ResNet-18, plus 0.95 and 1.15 relative to the compact encoder. Labels never enter the timed prediction path. Realized service times and queue waits reveal whether a nominal load actually saturated a pipeline.

Each of five load levels has six matched three-model sessions of 96 requests. The six model execution permutations occur once per level, and a seeded shuffle interleaves load/session jobs. Every pipeline receives the exact same image IDs and relative scheduled arrival times within a session. Queues start empty and drain to completion. There are 2880 unique fresh trace images, excluding saved held-out and trace IDs from the earlier live-PCA, load-sweep and channel-narrow studies. These are fresh request panels, not a claim that the public CIFAR-10 test set has never been inspected in project history.

The live runner measures scheduled arrival to completed scalar prediction. Queue wait, preprocessing, encoder, feature adapter, classifier, monitor and dispatch costs are included. Image bytes are preloaded equally; disk/network ingestion and log serialization are outside the inference boundary. Encoders execute for every timed request. INT8 quantization and FP32 reconstruction remain inside the grid comparator's service time.

## Decision rule

For each load and each reference, bootstrap paired session p95 reductions over the six sessions. Quality uses paired correct/incorrect outcomes on that load's 576 unique images. Report ordinary 95% intervals and conservative one-sided adoption bounds, using a 0.00125 tail probability across ten latency and ten quality comparisons. The compact model must have at least 5% lower p95 and no more than one percentage point of accuracy loss at every tested load against both references to replace the study's fixed baseline. An inconclusive quality bound is a failure to establish noninferiority, not proof of inferior accuracy. Aggregate accuracy is descriptive and cannot override a failed per-load gate.

## Reproduction

From the repository root, with the pilot's run directories and frozen preparation restored:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src python -m adaptive_systems.compact_load_sweep --output runs/compact-load-sweep-new
python scripts/verify_compact_load_sweep.py --run runs/compact-load-sweep-new
```

The runner expects `runs/compact-encoder`, `runs/live-pca`, `runs/width-precision`, `runs/load-sweep` and `runs/channel-narrow`, plus `../repro-package/frozen/prepared` and the original CIFAR-10 data under `data`. The evidence bundle preserves frozen inputs and run records. Checkpoints and exclusions must match before rerunning. The protocol file documents constants implemented in the runner; it is not a second source of runtime settings.

## Limits

This is one CPU host with one inference thread. Sessions repeat within the same process; they are not independent machines, deployments or days. Above-capacity p95 depends on the finite 96-request duration and empty starting queue. Changing session length can change overload rankings. Confidence bounds are empirical bootstrap bounds with only six timing clusters per load. No production, hardware-wide, energy or native INT8 arithmetic claim follows.

## Results — 26 November 2026

Thirty original matched sessions cover 2880 unique images and 8640 live requests. A separately identified low-load coverage check adds six sessions, 288 fresh images and 864 requests. All original latency comparisons clear the conservative 5% bound, but the original per-load quality family does **not** establish the one-point noninferiority limit. The compact encoder is **not promoted** and no switching controller is built.

The original overall held-out accuracies are **81.67% full ResNet-18**, **81.98% grid fixed**, and **83.78% compact**. Compact point accuracy meets the tolerance at every original load, including its largest observed loss of 0.52 points at reference-near. The failed gate reflects uncertainty: for example, its simultaneous lower quality bound versus the grid is −5.38 points at reference-near. The data do not establish a load-dependent accuracy failure; each load contains a different sampled image panel.

The table reports means of session p95s and means of paired relative reductions. These are different aggregation rules; do not compute paired gains by taking the ratio of the p95 means. Encoder time is the mean per-request encoder stage time averaged across the six sessions.

| Load | Full p95 ms | Grid p95 ms | Compact p95 ms | Compact encoder ms | Compact p95 wait ms | Paired gain vs grid | Compact accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| reference_below | 1427.30 | 577.23 | 26.45 | 9.31 | 6.57 | 81.79% | 83.33% |
| reference_near | 321.00 | 507.57 | 16.00 | 8.21 | 0.49 | 95.41% | 81.60% |
| reference_above | 3746.71 | 706.71 | 13.28 | 6.97 | 0.43 | 97.66% | 84.72% |
| compact_near | 4852.80 | 3574.05 | 62.67 | 6.96 | 53.66 | 98.09% | 83.68% |
| compact_above | 4264.68 | 4152.14 | 228.18 | 7.78 | 219.07 | 95.84% | 85.59% |
| verified_low | 84.25 | 85.86 | 350.69 | 18.84 | 289.95 | -327.04% | 86.11% |

### Capacity coverage and host variability

Original calibration median service times were 44.46 ms for full ResNet, 44.46 ms for the grid, and 8.28 ms for compact. Nominal reference gaps were 68.40, 46.80 and 38.66 ms; compact gaps were 8.72 and 7.20 ms. Live reference service times varied substantially. The nominal below-capacity full ResNet mean service time reached 73.71 ms, above its 68.40 ms arrival gap, and developed queues. Therefore the nominal labels alone do not prove actual underload.

The **supplemental verified-low** check was added after detecting that coverage issue. It freezes all checkpoints, excludes all 2880 original IDs, uses 250 ms arrival spacing and 48 requests per session, and repeats all six execution orders. It is reported separately, never pooled into the original adoption family or used to reverse the original decision. Shorter sessions make overload comparisons inappropriate; this supplement checks low-load coverage only.

| Verified-low model | Mean service ms | Mean encoder ms | Mean wait ms | Mean session p95 wait ms |
|---|---:|---:|---:|---:|
| resnet18 | 61.59 | 58.85 | 0.20 | 0.55 |
| grid_fixed | 57.55 | 54.31 | 0.29 | 0.65 |
| compact | 22.37 | 18.84 | 42.85 | 289.95 |

Large and nonmonotonic reference timing outliers remain visible; balanced order and repeated sessions do not eliminate host scheduling variance. The compact encoder’s much lower directly measured encoder time and consistent paired latency advantage in the original family support an architecture-level cost advantage on this host. Its larger 2304-byte final embedding confirms that this is not a final-feature compression result.

### Carry-forward decision

Keep the existing adopted fixed policy unchanged. The compact architecture remains the next fixed candidate. All original latency comparisons favor compact, but a supplemental low-load outlier prevents a reliable across-load latency claim. That outlier alone is not evidence for a switching controller. Before promotion, use a larger independently reserved quality panel with a prespecified noninferiority analysis, and repeat timing across independent host sessions or days. Do not treat inconclusive per-load bounds as proof that the model should switch by load.

Supplement reproduction uses the same runner with `--low-load-check --output runs/compact-low-load-new`, after preserving the original `runs/compact-load-sweep` lock used for exclusions. Each run preserves its as-executed `runner.py`; the original source snapshot predates the supplemental CLI addition. The verifier checks the snapshot against the lock instead of assuming the working runner is unchanged.

### Low-load anomaly and diagnostic replay

The supplemental compact session 0 has p95 response **1984.80 ms**, versus 82.17 ms for the grid and 65.81 ms for full ResNet. Request 21 spent **2439.78 ms inside the encoder**, causing later requests to wait. The other five compact sessions have p95s between 16.61 and 36.01 ms. The slow session remains in every summary; no outlier is discarded. Wall-only original records cannot distinguish computation from scheduling or other pauses.

A single exact-trace diagnostic replay retains all checkpoints, images, arrivals and original model order, while adding encoder thread CPU timing. It is not an independent confirmation and is not pooled into adoption.

| Replay model | p95 response ms | Maximum encoder wall ms | Thread CPU ms for that request |
|---|---:|---:|---:|
| resnet18 | 52.90 | 51.40 | 51.40 |
| grid_fixed | 99.76 | 172.98 | 172.93 |
| compact | 15.56 | 19.09 | 19.09 |

The replay cannot explain the original event retrospectively. If the spike disappears, its cause remains unresolved. If it recurs, wall and thread CPU time narrow the investigation but do not identify a specific scheduler/backend cause alone. Stable independent low-load replication is needed before promoting a baseline or interpreting a load-dependent crossover. Replay command: add `--low-load-check --replay-low-anomaly --output runs/compact-low-replay-new`, retaining the original `runs/compact-low-load/lock.json`.

[Download frozen checkpoints, all live records, source snapshots and diagnostic replay](https://chatgpt.com/library?file_id=libfile_bf768acd0a60819198201af68384888a&account_id=personal).
