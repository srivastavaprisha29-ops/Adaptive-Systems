# Always-active confidence gate with bounded queue relaxation

## Controlled question

The previous safeguarded controller turned off early exits at low queue depth. This replay tests the loss of exits directly: use an always-active default confidence gate, then allow queue pressure to lower its threshold only as far as calibration permits. Compare it with both the unchanged historical best fixed policy and the always-active default gate.

The model checkpoints, six 160-image panels, image order, relative arrivals, burst/steady schedules and balanced policy execution orders are copied byte-for-value from the previous study lock. These are controlled replays of already studied validation traces, not new independent confirmation data. The previous result's inputs are preserved with the new evidence. All runtime measurements use real inference, with the scheduled-arrival-to-prediction boundary unchanged.

## Calibration and policies

The default margin threshold remains **0.2750557065010071**, selected using only the original 1,000 calibration images. A fixed calibration grid (41 margin quantiles plus default and historical fixed thresholds) considers lower cutoffs. For each cutoff, form the adverse case in which additional early exits occur only where the early head is wrong and the full model is right. Compare that adverse policy with the historical best fixed policy on calibration correctness, using the existing paired image-bootstrap interval.

Choose the most permissive cutoff whose lower accuracy-delta bound is at least -1 point. The chosen limit is **0.20296822488307953**. It admits 125 additional calibration images relative to default; 9 are harmful potential exits. The adverse-case delta relative to best fixed is -0.50 points, with interval [-1.00, -0.10]. This is a calibration estimate, including grid selection, not a formal out-of-sample guarantee. It deliberately compares against the user-requested best fixed reference rather than claiming an absolute quality guarantee against full inference.

| Policy | Low queue depth | At least three waiting requests |
|---|---|---|
| Historical best fixed | Margin >= 0.21725761890411377 | Same |
| Always-active confidence gate | Margin >= 0.2750557065010071 | Same |
| Bounded queue-aware gate | Margin >= 0.2750557065010071 | Margin >= 0.20296822488307953 |

Low-confidence predictions always finish stage four. The queue policy never disables its default early gate or relaxes below the calibration limit. All policies log early prediction, margin, top softmax score, early-exit decision, queue depth, service time and scheduled/start/finish timestamps. The softmax score is diagnostic and is not a calibrated probability. Labels and cached counterfactual full outputs are joined only after timed evaluation.

Three live calibration sanity trials precede evaluation, with no subsequent threshold tuning. The frozen lock records both thresholds, all calibration candidates, traces, execution orders, inputs and source hashes. Live timing never substitutes cached features for inference. Warmup, preloaded-image boundary and confidence-logging overhead are consistent across policies.

## Acceptance rule

The primary operational target remains at least **5% lower p95 response time** with **at most one percentage point accuracy loss** versus historical best fixed. Queue awareness must additionally demonstrate positive p95 improvement versus the always-active confidence gate, without exceeding that accuracy-loss tolerance.

The adoption check uses four one-sided 98.75% lower bounds, allocating a 5% family error budget by Bonferroni across these four endpoints. Bounds come from 8,000 paired-session bootstrap draws over six sessions. Ordinary 95% descriptive intervals are also reported. These are approximate small-sample bootstrap calculations on a shared host, not a guarantee of independence, portability or broader controller validation. Steady-load checks are descriptive and reported separately.

## Steady anomaly diagnostic

The original steady session 5 had safeguarded-policy p95 waiting of **126.003 ms**, versus **0.222 ms** for best fixed. Its largest safeguarded service times were **225.90 ms** and **212.74 ms**, while best fixed's maximum was **82.22 ms**. With a 72 ms arrival gap, those long services can accumulate waiting. Original records contain no thread CPU measurements, so they cannot identify why those outliers occurred.

In addition to the new six-session comparison, the runner replays the exact session-5 steady trace with the **original three policies**, including the old pressure-only safeguarded gate. That diagnostic distinguishes a policy rewrite from recurrence of the original spike. New records add per-inference thread CPU time and per-trial process context-switch counts. CPU and context-switch diagnostics may help describe scheduling effects; they cannot establish the cause of a historical outlier by themselves. They are not added to wall service time.

## Reproduce

Obtain the frozen model/head/data and feature cache using the links in the [confidence safeguard study](confidence-safeguard.md). Extract the [previous raw study](https://chatgpt.com/library?file_id=libfile_63110e47bae08191b5aa16e4ff188892&account_id=personal) as the prior-run directory.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m adaptive_systems.relaxation_study \
  --artifacts /path/to/frozen/prepared --head /path/to/frozen/head.pt \
  --data /path/to/data --feature-cache /path/to/feature-cache \
  --prior-run /path/to/prior-study --settings configs/relaxation-study.json \
  --output runs/relaxation-study-new
```

Use a fresh output path and the pinned dependencies. The resource-counter diagnostic uses the Unix `resource` module, matching the tested Linux environment. Store the prior lock beside the new evidence for independent verification:

```bash
cp /path/to/prior-study/lock.json runs/relaxation-study-new/prior-lock.json
python scripts/verify_relaxation_study.py --run runs/relaxation-study-new
python -m unittest discover -s tests -v
```

The verifier checks unchanged prior traces, model-policy evidence hashes, calibration qualification, bounded decisions, prediction branches, FIFO timing, error tables, all session comparisons, the four-endpoint adoption decision and the original-policy anomaly replay. It runs without PyTorch. The new raw records include early-exit rate, service time, waiting and accuracy for every trial.

## Observed results

All saved model identities and six trace plans match the prior lock. The two fixed comparators' 3,840 request predictions, exit flags and image/arrival records are unchanged from the prior study. Timing is newly measured. All 31 tests pass. The independent verifier checks 6,240 timed evaluation/replay requests, with zero relaxation-limit violations. There are another 240 calibration requests.

### Burst metrics

Values below are arithmetic means of the six per-session metrics. Mean session p95 is not a pooled request p95.

| Policy | Early exits | Mean service | Mean session p95 queue wait | Mean session p95 response | Accuracy |
|---|---:|---:|---:|---:|---:|
| best_fixed | 49.69% | 54.54 ms | 1578.02 ms | 1633.99 ms | 83.438% |
| fixed_confidence | 42.08% | 55.88 ms | 1784.31 ms | 1838.78 ms | 83.438% |
| safeguarded_queue | 52.19% | 54.89 ms | 1623.12 ms | 1679.17 ms | 83.438% |

| Burst comparison | Mean paired p95 reduction | Descriptive 95% session interval | Mean accuracy delta |
|---|---:|---|---:|
| safeguarded_queue_vs_best_fixed | -3.16% | [-8.96%, +2.84%] | +0.000 pp |
| safeguarded_queue_vs_fixed_confidence | +7.81% | [+0.20%, +16.45%] | +0.000 pp |
| fixed_confidence_vs_best_fixed | -12.70% | [-20.65%, -5.45%] | +0.000 pp |

Queue awareness restores the lost exits, but it does not meet the target relative to best fixed. Its mean p95 reduction is -3.16%, with descriptive interval [-8.96%, +2.84%], and mean accuracy delta is zero. Relative to the always-active confidence gate, its point estimate improves p95 by 7.81%, with descriptive interval [+0.20%, +16.45%]. That descriptive result is not the predeclared four-endpoint adoption test.

The adoption-adjusted one-sided gain lower bounds are **-10.74% vs best fixed** and **-0.45% vs always-active confidence**. Accuracy lower bounds are -0.521 and -0.833 pp respectively. The joint adoption rule therefore fails. Fixed-confidence itself is also slower than historical best fixed on these burst trials. Retain fixed gating as the simpler approach and the historical best fixed as the performance reference; do not promote either new candidate as superior.

The mean reduction is the mean of six within-session relative p95 changes, not the ratio of the table's mean p95 values. These replays address the proposed mechanism but cannot assign the difference from the prior -21.16% result solely to restored exits: machine execution conditions also changed between studies.

### Steady-load metrics

| Policy | Early exits | Mean service | Mean session p95 queue wait | Mean session p95 response | Accuracy |
|---|---:|---:|---:|---:|---:|
| best_fixed | 49.69% | 54.86 ms | 0.19 ms | 66.02 ms | 83.438% |
| fixed_confidence | 42.08% | 55.88 ms | 0.19 ms | 66.53 ms | 83.438% |
| safeguarded_queue | 42.19% | 57.42 ms | 35.06 ms | 103.29 ms | 83.438% |

The new queue policy continues early exits at low pressure, rather than making zero steady exits as before. However, large timing outliers remain. In new steady session 0 its p95 wait is 58.82 ms; in session 3 it is 150.86 ms. Session 3 request 138 takes 337.56 ms of service and 337.51 ms of inference thread CPU time. This observation is not consistent with treating the entire delay as passive queue waiting or attributing it simply to a brief controller decision. Thread CPU time alone does not identify the CPU/kernel/hardware cause. Further profiling and timing stability are needed before a small p95 improvement can support adoption.

### Exact original anomaly replay

The original session-5 trace was additionally rerun with its original policies, including the pressure-only safeguard.

| Original policy replay | p95 queue wait | Maximum service | Maximum inference thread CPU |
|---|---:|---:|---:|
| best_fixed | 0.122 ms | 69.07 ms | 67.41 ms |
| fixed_confidence | 0.262 ms | 89.36 ms | 89.26 ms |
| safeguarded_queue | 0.107 ms | 82.77 ms | 82.70 ms |

The old safeguard's 126.003 ms p95 wait does not recur in this exact-policy replay: it is 0.107 ms, versus best fixed's 0.122 ms. This does not prove the old spike was fixed or identify its historical cause. It coexists with the new steady outliers documented above, so timing instability is reported rather than discarded.

Complete numeric comparisons and diagnostic means are committed in `relaxation-results.json` and `relaxation-analysis.json`. The raw evidence includes every request, thread CPU diagnostic, context-switch count, calibration candidate, original-policy replay and lock.

[Download the full controlled-replay evidence](https://chatgpt.com/library?file_id=libfile_206c9a2306cc8191badeff99a453848c&account_id=personal). It includes the prior lock required by the verifier.
