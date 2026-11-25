# Frozen-policy load sweep with a machine-selected fixed baseline

## Protocol

This experiment first selects the fixed comparator on calibration data on the current machine, then freezes every policy and offered-load level before evaluating fresh dynamic traces. The model, encoder, early head and full head remain unchanged.

The 1,000 training-calibration images provide fixed-policy quality estimates. The declared fixed grid contains the historical threshold, the existing default, the 70th/80th/90th calibration margin quantiles and full inference. A candidate is eligible only when its paired calibration accuracy-delta lower 95% bound relative to the full model is at least -1 percentage point. This is an empirical eligibility screen, not a simultaneous or formal out-of-sample quality guarantee.

Every eligible candidate is profiled on the same 64 calibration images in three shuffled-order live trials. One-nanosecond scheduled gaps keep these trials backlogged. Select the candidate with the smallest median of its three mean service times. This is the fastest eligible member of the declared grid, not a claim of global optimality over every possible policy. Quality and timings are selected without evaluation images or labels.

The selected threshold is **0.2750557065010071**, with median mean service **53.1268 ms**, an estimated **18.8229 requests/second**. Its three calibration mean service times are 52.8347, 53.1268 and 61.3910 ms. The historical threshold 0.21725761890411377 is excluded because its calibration lower accuracy bound fails the one-point screen. This differs from its older point-estimate feasibility rule and is reported explicitly.

| Policy | Default threshold | Under at least three pending requests |
|---|---|---|
| Machine-selected best fixed | 0.2750557065010071 | Unchanged |
| Always-active confidence gate | 0.2750557065010071 | Unchanged |
| Bounded queue-aware gate | 0.2750557065010071 | 0.20296822488307953 |

The two confidence policies are copied unchanged from the prior relaxation lock. No threshold is retuned by load. The selected best fixed is identical to the always-active gate in this experiment. Both are timed independently as requested, but they represent the same algorithm and their execution noise does not count as independent algorithmic evidence. Their per-image predictions and exit decisions must match.

## Load levels and fresh traces

The load labels refer to offered rates relative to the selected fixed policy's measured calibration capacity:

| Nominal range | Offered rate / calibration capacity | Arrival gap | Approximate requests/second |
|---|---:|---:|---:|
| Below | 0.65 | 81.734 ms | 12.235 |
| Near | 0.95 | 55.923 ms | 17.882 |
| Above | 1.15 | 46.197 ms | 21.646 |

There are six paired sessions per level and 128 images per session. All 18 panels are disjoint, giving **2,304 distinct official-test images** and **6,912 timed evaluation requests**. Image panels, order and arrival timestamps match across policies within a block. These dynamic-controller panels are fresh and use the official CIFAR-10 test namespace rather than prior training-validation panels. The official test set has appeared in earlier static aggregate evaluations, so it is not globally untouched research data.

Interarrivals are constant within a load block, with new rates derived exclusively from calibration. Policy execution uses all six permutations once per level. Load execution also uses all six permutations across sessions, balancing time/order effects. The fixed random seed and all traces are saved before evaluation. The study is a controlled steady-arrival load sweep; it does not establish behavior under arbitrary stochastic arrival patterns or distribution shifts.

Every block is a finite 128-request observation followed by a full queue drain. Above-capacity queues can keep growing, so these p95 values are finite-horizon measurements, not stationary tail-latency guarantees. Actual service demand divided by arrival gap is reported for every trial. It can differ from nominal load if the machine slows down or the policy changes service cost. A nominal label alone is not proof that the live server remained below or above saturation.

## Measurements and claims

Response time runs from scheduled arrival to completed scalar prediction. Service includes monitoring, dispatch, scalar state changes, preprocessing, actual network execution and confidence diagnostics. Queue waiting is measured separately. All requests are drained without drops or truncation. Predecoded images are preloaded; decoding, disk and network are excluded. Full-model counterfactual predictions and labels are joined after all three timed policy runs in a block. No cached test inference substitutes for actual execution.

Report early-exit rate, service time, p95 wait, p95 response, accuracy and observed offered load. Per-request thread CPU diagnostics remain available to inspect execution outliers. Nested controller/monitor timings must not be added to service time again.

A claimed winning load range must have at least 5% lower p95 response time and at most one percentage point accuracy loss against machine-selected best fixed, and positive p95 improvement without exceeding the accuracy tolerance against the always-active gate. The bootstrap resamples the six complete paired sessions at each load, not individual dependent request latencies. It uses 12,000 draws. For any load-specific adoption claim, the one-sided error budget is 0.05/12 across three loads and four endpoints. The ordinary 95% intervals are descriptive; adoption uses the more conservative family-adjusted lower bounds.

Six sessions on a shared host remain a limited sample. The bootstrap family adjustment is approximate and does not establish independence or portability. The artifact keeps `research_validated: false` even if a pilot load range passes its stated numerical gate. If no range passes, retain the selected fixed configuration and shift the research focus to width and precision adaptation, rather than tuning queue thresholds further on the evaluated traces.

## Reproduction and independent audit

Use the frozen model/head/data and calibration feature cache linked in the [bounded relaxation study](calibration-bounded-relaxation.md). Extract its [raw evidence](https://chatgpt.com/library?file_id=libfile_206c9a2306cc8191badeff99a453848c&account_id=personal) for `--prior-run`. That lock supplies the frozen confidence-policy parameters. Use a fresh output directory.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m adaptive_systems.load_sweep \
  --artifacts /path/to/frozen/prepared --head /path/to/frozen/head.pt \
  --data /path/to/data --feature-cache /path/to/feature-cache \
  --prior-run /path/to/bounded-relaxation \
  --settings configs/load-sweep.json --output runs/load-sweep-new

python scripts/export_load_sweep_calibration.py \
  --cache /path/to/feature-cache/calibration.pt \
  --head /path/to/frozen/head.pt --run runs/load-sweep-new
python scripts/verify_load_sweep.py --run runs/load-sweep-new
python -m unittest discover -s tests -v
```

The calibration export is post-run audit data and cannot affect the already-frozen selection. It exposes calibration margins, labels and predictions so the verifier can recompute eligibility point estimates and all live profiling summaries. The verifier checks the recorded bootstrap-bound eligibility decision; it does not independently reproduce NumPy's calibration bootstrap distribution. It also verifies hashes, fixed-policy selection, unmodified confidence policies, fresh disjoint panels and namespaces, all dispatch decisions, timing decomposition, evaluation metrics, paired-session intervals and load-specific claims. Once the export exists, this audit requires no ML inference.

## Observed results and decision

All 18 paired blocks completed. The verifier checked **6,912 evaluation requests**, **960 calibration-profile requests**, 2,304 distinct test images and identical predictions/exit flags for the two equivalent fixed gates. Frozen-policy violations: zero. All **34 tests pass**.

| Load | Queue p95 reduction vs selected fixed | Descriptive 95% session interval | Accuracy delta | Family-adjusted latency lower bound | Family-adjusted accuracy lower bound | Claimed win |
|---|---:|---|---:|---:|---:|---|
| below | +3.00% | [-2.90%, +9.02%] | +0.000 pp | -4.76% | +0.000 pp | No |
| near | -1.73% | [-36.50%, +26.57%] | +0.000 pp | -48.20% | -0.521 pp | No |
| above | +5.33% | [-11.45%, +21.26%] | -0.521 pp | -16.22% | -1.172 pp | No |

**No load range clears the evidence bar.** Above capacity meets the target by point estimate (+5.33% p95 improvement, -0.521 pp accuracy), but its latency interval includes regression and its accuracy lower bound exceeds the tolerated loss. Near capacity does not show a benefit even by mean point estimate (-1.73% reduction). The queue signal has no supported saturation-specific win in this sweep.

### Measured service, waiting and accuracy

Values are means of six session metrics per load; mean session p95 is not a pooled request p95. Observed load is mean service divided by scheduled gap.

| Load | Policy | Exit rate | Mean service | Mean session p95 wait | Mean session p95 response | Accuracy | Observed load |
|---|---|---:|---:|---:|---:|---:|---:|
| below | best_fixed | 41.93% | 57.71 ms | 0.28 ms | 70.91 ms | 83.854% | 0.706 |
| below | always_active | 41.93% | 57.12 ms | 0.43 ms | 70.49 ms | 83.854% | 0.699 |
| below | bounded_queue | 41.93% | 57.31 ms | 0.20 ms | 68.33 ms | 83.854% | 0.701 |
| near | best_fixed | 40.23% | 57.38 ms | 224.62 ms | 281.80 ms | 83.333% | 1.026 |
| near | always_active | 40.23% | 56.13 ms | 150.23 ms | 206.70 ms | 83.333% | 1.004 |
| near | bounded_queue | 42.84% | 56.99 ms | 197.79 ms | 254.74 ms | 83.333% | 1.019 |
| above | best_fixed | 41.28% | 56.19 ms | 1215.00 ms | 1270.43 ms | 85.417% | 1.216 |
| above | always_active | 41.28% | 56.94 ms | 1310.03 ms | 1364.61 ms | 85.417% | 1.232 |
| above | bounded_queue | 52.08% | 55.33 ms | 1099.62 ms | 1155.86 ms | 84.896% | 1.198 |

### Where the queue signal changed work

| Load | Different exit decisions vs always-active gate | Observed fixed load range across sessions |
|---|---:|---|
| below | 0 / 768 | [0.641, 0.738] |
| near | 20 / 768 | [0.993, 1.063] |
| above | 83 / 768 | [1.157, 1.289] |

Nominal near-capacity trials sometimes run above their calibrated capacity because actual service changes. Timing differences between the equivalent fixed policies are execution variability, not algorithmic differences. In particular, the always-active gate's large near-load point difference versus best fixed cannot be credited as a new method: their per-image outputs and exits are identical. Six short finite-horizon trials on a shared host leave substantial uncertainty. The previously unexplained steady spikes are not considered resolved by this sweep.

### Carry-forward configuration and research transition

Retain the machine-selected fixed margin gate in `configs/carry-forward-fixed.json`, with threshold 0.2750557065010071, the same frozen head/encoder and one CPU thread. The configuration records its calibration lock and preparation hashes. Recalibrate for a different checkpoint or machine rather than treating the measured capacity as a portable constant. No queue-aware policy is promoted.

The next research phase is [width and precision action profiling](width-precision-phase.md). The controller study now supplies a fixed reference and measurement harness; it does not establish an advantage for queue adaptation. Complete numeric results and means are in `load-sweep-results.json` and `load-sweep-analysis.json`. The full raw package preserves calibration profiles, calibration scores, all fresh test traces and request records, policy/source/data hashes and the carry-forward configuration.

[Download the complete frozen load-sweep evidence](https://chatgpt.com/library?file_id=libfile_100d72146b508191b0012cd80508d18f&account_id=personal). It includes calibration scores, all raw trials and the carry-forward configuration.
