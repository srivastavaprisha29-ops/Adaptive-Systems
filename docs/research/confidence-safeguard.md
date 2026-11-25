# Confidence safeguard and value of queue awareness

## Frozen design

This study extends the live burst benchmark with a confidence safeguard, per-request error attribution, six paired burst sessions and six steady-load checks. It is an exploratory study on one host, using one model and dataset. It does not by itself validate the broader width/precision controller.

Calibration uses only the prepared 1,000-image calibration split. On a fixed quantile grid, choose the maximum-coverage margin threshold whose paired image-bootstrap lower accuracy bound stays within one percentage point of the full model. The calibrated floor is **0.2750557065010071**, with 40.0% early exits, calibration accuracy 83.4% versus full 83.7%, and an accuracy-delta 95% interval of [-0.90, +0.20] points. Three live 80-image calibration trials then exercise the policies on calibration images. No threshold is changed after those trials. A frozen-lock record is written before evaluation. The calibration grid search is not multiplicity-adjusted, so this is an empirical safeguard rather than a formal quality guarantee, especially after queue-conditioned selection.

| Policy | When an early exit is permitted |
|---|---|
| Historical best fixed | Margin at least 0.21725761890411377 on every request |
| Fixed confidence gate | Margin at least the calibrated floor on every request |
| Safeguarded queue-aware | At least three requests are waiting behind the current request **and** margin meets the same calibrated floor; otherwise finish the full model |

Queue pressure controls whether the safeguarded controller tries an exit. It never lowers the confidence floor. This is a new guarded policy, rather than a claim that the historical aggressive queue controller was safe. The fixed comparator retains its historical calibration selection; it is not an exhaustive new optimum for the current machine.

The fixed confidence gate is a crucial ablation. With this particular design it can skip stage four on every image the queue-aware policy can skip, plus eligible images encountered without pressure. Therefore queue awareness has no intrinsic per-image compute advantage over that fixed gate. It may change accuracy and response-time behavior through routing and queue dynamics; the measurements must establish whether any such trade-off is useful.

## Replication and timing

`configs/safeguard-study.json` fixes six sessions of 160 images each. Evaluation panels are disjoint across sessions and exclude calibration. Burst and steady workloads reuse the same panel within a session. Each policy receives exactly the same image order and scheduled arrival timestamps within a workload. The six permutations of three policies each occur once per workload, balancing execution order. Policy state and timing origin reset before every trial.

Burst gaps follow the existing 12 ms/96 ms schedule. The steady check uses 72 ms gaps. There are 960 distinct evaluation images and 5,760 timed evaluation requests, plus 240 timed calibration requests. Historical validation data have been studied before and are not an untouched confirmatory test set. Separate panels and balanced order improve replication, but serial execution on a shared host does not guarantee statistically independent host conditions.

Response time runs from scheduled arrival to completed scalar prediction. It includes queueing, backlog observation, controller selection, scalar state changes, image preprocessing and actual model execution. Images are decoded before timing. Monitoring and decision diagnostics are nested in service time and are not added again. All policies compute and log early-head confidence, including full-path requests, so that diagnostic overhead is included consistently. Inputs are never served from the feature cache during timing.

## Prediction diagnostics

Every evaluated request stores queue depth, early predicted class, early exit, top-two logit margin, top softmax probability, final class, scheduled/start/finish timestamps and timing decomposition. Labels and counterfactual full predictions are joined only after all timed trials for the session. Frozen validation features provide those counterfactual predictions and a post-run check of every live early and final prediction. The raw ridge-score softmax value is not a calibrated probability; the guard uses the margin threshold calibrated above.

Error tables count both **early-head wrong / full-model right** cases and the subset actually admitted as harmful exits. Grouping separates model weakness from a bad routing decision.

| Margin bin | Range |
|---|---|
| 0 | [0, 0.1) |
| 1 | [0.1, 0.2) |
| 2 | [0.2, 0.3) |
| 3 | [0.3, 0.5) |
| 4 | [0.5, infinity) |

| Queue bin | Pending requests behind the current request |
|---|---|
| 0 | 0 |
| 1 | 1–2 |
| 2 | 3–7 |
| 3 | 8 or more |

Binned counts are request-level diagnostics. They must not be treated as independent image samples when the same panel is reused across policies and workloads.

## Statistical decision

The primary comparison is **burst safeguarded queue-aware versus historical best fixed**. The estimand is the mean of the six within-session relative p95 reductions, alongside the mean paired accuracy delta. The bootstrap resamples complete paired sessions (8,000 draws), preserving request order and dependence within each trace. Requests are not resampled as independent timing observations.

The primary joint gate requires the one-sided 97.5% lower latency bound to be at least 5%, and the one-sided 97.5% lower accuracy-delta bound to be at least -1 percentage point. The two endpoint error budgets use Bonferroni allocation, with approximate percentile-bootstrap bounds. Six sessions remain a small sample, and this procedure is not a finite-sample guarantee. Fixed-confidence comparisons and steady-load intervals are descriptive secondary checks without a simultaneous superiority claim.

The saved summary records this bounded result separately from the broader validation status. Host portability, more workloads, distribution shifts, actual representation/format switching, energy and peak memory remain outside this experiment.

## Reproduction

Use the frozen inputs in the [reproducibility package](https://chatgpt.com/library?file_id=libfile_3cefc90f4ef48191a84cc47c5f061aad&account_id=personal). The verified calibration/validation feature cache is in the [full-training distillation evidence package](https://chatgpt.com/library?file_id=libfile_83c5217ccc388191b265c600cdd68bd3&account_id=personal), under `feature-cache`. The cache is for calibration and diagnostics only.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m adaptive_systems.safeguard_study \
  --artifacts /path/to/frozen/prepared --head /path/to/frozen/head.pt \
  --data /path/to/data --feature-cache /path/to/feature-cache \
  --settings configs/safeguard-study.json --output runs/safeguard-study-new
python scripts/verify_safeguard_study.py --run runs/safeguard-study-new
python -m unittest discover -s tests -v
```

Use a fresh output directory. The lock saves all traces, policy orders, thresholds, input/source hashes and calibration results before evaluation. Raw files and summary hashes preserve the complete evidence. The standard-library verifier recomputes metric/error tables, checks the confidence safeguard and branches, and reproduces session-level comparisons without executing a neural network.

## Observed results

All 5,760 evaluated request records passed the independent verifier, with zero exits below the confidence floor. All live early and final predictions matched the frozen-feature post-run checks. All 28 unit tests pass.

| Burst session | Best fixed p95 | Fixed confidence p95 | Safeguarded queue p95 | Queue reduction vs best fixed | Accuracy delta vs best fixed |
|---|---:|---:|---:|---:|---:|
| 0 | 1191.00 ms | 1204.12 ms | 1282.43 ms | -7.68% | +1.250 pp |
| 1 | 2051.38 ms | 1932.14 ms | 2144.95 ms | -4.56% | +0.000 pp |
| 2 | 1402.59 ms | 1161.52 ms | 1886.05 ms | -34.47% | +0.000 pp |
| 3 | 1614.49 ms | 2022.13 ms | 1801.48 ms | -11.58% | +0.625 pp |
| 4 | 963.78 ms | 1606.80 ms | 1470.30 ms | -52.56% | -0.625 pp |
| 5 | 1190.96 ms | 1313.58 ms | 1382.71 ms | -16.10% | -1.250 pp |

The safeguarded controller is slower in all six burst sessions relative to historical best fixed. Its mean paired p95 reduction is **-21.16%** (95% session-bootstrap interval **[-35.88%, -8.82%]**), meaning a 21.16% mean increase in the within-session p95 ratio. Mean accuracy delta is **0.00 pp**, with interval **[-0.625, +0.625] pp**. Accuracy meets the tolerance for this sample, but the latency endpoint fails, so the joint target fails. No controller replacement is justified.

| Comparison | Burst mean p95 reduction (95% interval) | Steady mean p95 reduction (95% interval) |
|---|---|---|
| Safeguarded queue vs best fixed | -21.16% [-35.88%, -8.82%] | -30.75% [-99.07%, +5.87%] |
| Safeguarded queue vs fixed confidence | -10.96% [-33.04%, +4.36%] | -36.08% [-109.51%, +5.90%] |
| Fixed confidence vs best fixed | -13.39% [-36.42%, +6.13%] | +2.89% [-3.19%, +8.71%] |

Secondary comparisons are noisy and do not establish superiority of either guarded policy. The queue policy makes zero early exits in the steady check, where its pending depth never reaches three. It therefore incurs full-path work while the fixed confidence gate can still exit. Queue awareness adds no demonstrated value under this tested policy and workload. This does not rule out a different controller on different operating regimes.

Across the 960 distinct burst images, the early head is wrong while the full model is right on **125** images. Best fixed admits **7** of those as harmful exits; each guarded policy admits **2**. But beneficial exits where the early head is right and the full model is wrong also fall from **9 to 4**, explaining the unchanged aggregate accuracy. A confidence safeguard reduces a particular failure mode without guaranteeing a net accuracy gain or a latency improvement.

### Burst error attribution by margin and queue depth

Each cell is **potential early errors corrected by full / harmful errors actually exited**. The same images appear once per policy; queue depth is policy-dependent, so these cells are descriptive routing diagnostics.

**best_fixed**

| Margin | Queue 0 | Queue 1–2 | Queue 3–7 | Queue 8+ |
|---|---:|---:|---:|---:|
| [0, 0.1) | 2 / 0 | 1 / 0 | 19 / 0 | 60 / 0 |
| [0.1, 0.2) | 0 / 0 | 1 / 0 | 8 / 0 | 24 / 0 |
| [0.2, 0.3) | 1 / 1 | 1 / 0 | 0 / 0 | 7 / 5 |
| [0.3, 0.5) | 0 / 0 | 0 / 0 | 0 / 0 | 1 / 1 |
| [0.5, infinity) | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |

**fixed_confidence**

| Margin | Queue 0 | Queue 1–2 | Queue 3–7 | Queue 8+ |
|---|---:|---:|---:|---:|
| [0, 0.1) | 2 / 0 | 1 / 0 | 12 / 0 | 67 / 0 |
| [0.1, 0.2) | 0 / 0 | 1 / 0 | 8 / 0 | 24 / 0 |
| [0.2, 0.3) | 1 / 0 | 1 / 0 | 0 / 0 | 7 / 1 |
| [0.3, 0.5) | 0 / 0 | 0 / 0 | 0 / 0 | 1 / 1 |
| [0.5, infinity) | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |

**safeguarded_queue**

| Margin | Queue 0 | Queue 1–2 | Queue 3–7 | Queue 8+ |
|---|---:|---:|---:|---:|
| [0, 0.1) | 2 / 0 | 1 / 0 | 12 / 0 | 67 / 0 |
| [0.1, 0.2) | 0 / 0 | 0 / 0 | 9 / 0 | 24 / 0 |
| [0.2, 0.3) | 1 / 0 | 1 / 0 | 0 / 0 | 7 / 1 |
| [0.3, 0.5) | 0 / 0 | 0 / 0 | 0 / 0 | 1 / 1 |
| [0.5, infinity) | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |

The committed `safeguard-errors.json` provides complete burst/steady group counts and denominators. `safeguard-results.json` preserves session comparisons, intervals and evidence hashes. The full evidence package also contains the frozen traces, per-request confidence and predictions, calibration trials and per-session metrics.

[Download the complete raw study evidence](https://chatgpt.com/library?file_id=libfile_63110e47bae08191b5aa16e4ff188892&account_id=personal). Extract it and pass that directory to `scripts/verify_safeguard_study.py --run`.
