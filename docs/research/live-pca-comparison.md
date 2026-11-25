# Full-width FP32 versus calibration-selected PCA FP32

## Controlled design

Both paths run the same frozen ImageNet ResNet-18 encoder, through its final pooling layer, producing a 512-dimensional FP32 vector. The full path classifies that vector directly. The reduced path subtracts the training mean, applies the selected PCA projection, then runs a dimension-compatible FP32 classifier.

Both ridge classifier heads and PCA are fit using the **same 4,000 training images**, sampled at 400 per class from the prepared 44,000-image fitting split. The ridge parameter is 1.0. This matched training budget isolates the representation comparison. These are newly fitted matched heads, not the earlier classifier trained on all 44,000 images; their accuracy should not be presented as an update to that earlier checkpoint.

PCA mean and basis use training features only. On the separate 1,000-image calibration split, evaluate 64/128/256/384 dimensions and choose the smallest width whose paired accuracy-delta lower 95% bound relative to full width is at least -1 point. Width selection is complete before held-out evaluation. Calibration is an empirical grid-based screen, not a formal generalization guarantee.

| Width | Calibration accuracy | Delta vs full | Paired 95% interval | Passes screen |
|---|---:|---:|---|---|
| Full 512 | 81.10% | 0 | Reference | Reference |
| PCA 64 | 79.40% | -1.70 pp | [-3.60, +0.30] pp | No |
| PCA 128 | 81.90% | +0.80 pp | [-1.10, +2.70] pp | No |
| PCA 256 | 83.30% | +2.20 pp | [+0.70, +3.70] pp | Yes, selected |
| PCA 384 | 81.60% | +0.50 pp | [-0.70, +1.70] pp | Yes |

The selected representation is **256D FP32**, with **1,024 bytes per vector**, versus 512D FP32 at **2,048 bytes**. The full 512D backbone output still exists in the PCA path. PCA also stores a 512-by-256 projection matrix and a 512-element mean, adding 526,336 bytes of FP32 parameters before accounting for head weights. A smaller vector does not imply lower total RAM or peak memory.

## Held-out data and live traces

A frozen random sample of 2,000 official CIFAR-10 test images estimates accuracy. The latest load-sweep panels are excluded. The official test set has previously appeared in static aggregate evaluations, so these images are not claimed to be globally untouched research data. Training, calibration and test namespaces remain distinct.

Six paired live sessions use disjoint 128-image panels drawn from that held-out sample. Both actions receive identical image IDs, order and scheduled arrivals within each session. Full-first and PCA-first orders each occur three times. A calibration-only full-path service profile sets the shared interarrival gap at nominal 80% load. Both paths use a single inference thread, batch one, and execute the actual encoder; no feature cache substitutes for timed inference. Parallel feature extraction is used only for fitting and offline accuracy evaluation.

Timing begins at scheduled arrival and ends at scalar prediction completion, including queue waiting and dispatch overhead. Images are decoded before timing; preprocessing is timed. Nested wall timers separately measure preprocessing, the backbone, projection and classifier. Full-width projection time is exactly zero. Component means are attribution diagnostics inside service time; they must not be added again or used to sum unrelated component p95 values.

The selected width and heads are frozen before testing. Every live prediction must match the corresponding offline prediction for the same frozen checkpoint. Per-request records store action, feature bytes, full-backbone feature bytes, component times and arrival/start/finish timestamps.

## Interpretation rule

A useful latency result retains the existing 5% lower p95 target with paired-session uncertainty. An interval that includes zero is not proof of unchanged latency; the artifact separately records whether the complete 95% interval lies inside +/-5%. Quality is reported on the 2,000-image held-out panel with paired image-bootstrap uncertainty. A calibration pass does not establish held-out quality preservation.

If only vector bytes improve while end-to-end latency has no supported useful gain, the next candidate must reduce computation inside the encoder. Post-pooling PCA cannot remove backbone work. Encoder-side depth/width/compute-precision changes need their own matched training and static quality/cost feasibility checks before adaptation.

## Reproduce

Use the original frozen encoder/data/preparation described in [reproducing follow-ups](../reproducing-followups.md). Obtain the previous sweep lock from the [frozen load-sweep evidence](https://chatgpt.com/library?file_id=libfile_100d72146b508191b0012cd80508d18f&account_id=personal). Its test panels are excluded from the new sample.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m adaptive_systems.live_pca \
  --artifacts /path/to/frozen/prepared --data /path/to/data \
  --prior-sweep /path/to/load-sweep --settings configs/live-pca.json \
  --output runs/live-pca-new
python scripts/verify_live_pca.py --run runs/live-pca-new
python -m unittest discover -s tests -v
```

Use a new output directory. Default preparation uses Unix `fork` workers; set `preparation_workers` to 1 for serial extraction if needed. Exact source/preparation hashes, the selected heads, training/calibration indices, test sample, all live traces and accuracy predictions are retained. The standard-library verifier recomputes held-out accuracy points, selection/quality-gate logic, raw feature bytes, FIFO timing, component sums and paired-session timing intervals. It verifies the recorded image-bootstrap quality gate rather than independently reproducing NumPy's bootstrap distribution.

## Observed results

The standard-library verifier checked all 2,000 held-out labels/prediction pairs and **1,536 live requests**, with matched traces, unchanged action checkpoints, valid timing decomposition and exact feature-byte counts. All **37 tests pass**.

| Metric | Full 512D FP32 | PCA 256D FP32 |
|---|---:|---:|
| Held-out accuracy (2,000 images) | 82.50% | 81.85% |
| Selected vector bytes | 2,048 | 1,024 |
| Mean service | 43.391 ms | 46.836 ms |
| Mean session p95 queue wait | 2.200 ms | 54.582 ms |
| Mean session p95 response | 56.253 ms | 109.984 ms |

PCA reduces vector bytes by **50%**. Held-out accuracy delta is **-0.65 pp**, with paired 95% interval **[-1.90, +0.50] pp**. The point loss is under one point, but the uncertainty bound fails the quality gate. The favorable calibration result did not establish held-out quality preservation. Width is not retuned after this result.

Live p95 is worse in all six paired sessions. Mean paired p95 reduction is **-92.75%**, with session-bootstrap interval **[-235.80%, -7.71%]**. This large relative regression is strongly influenced by a queueing outlier: session 0 has PCA p95 response 314.58 ms versus full 58.49 ms. The result is **not unchanged latency**, and the interval does not support equivalence within +/-5%. It is a failed latency candidate with substantial execution/queue variability. Mean session p95s in the table and the mean of paired relative changes use different aggregation rules.

### Component attribution

These are mean nested wall times across the six sessions. All remain inside the outer service timer.

| Component | Full FP32 | PCA FP32 |
|---|---:|---:|
| preprocess | 1.485050 ms | 1.534311 ms |
| backbone | 41.803900 ms | 45.138893 ms |
| projection | 0.000000 ms | 0.066483 ms |
| classifier | 0.042652 ms | 0.029737 ms |

The backbone contributes **96.34%** of full-path service and **96.38%** of PCA-path service. Both execute the identical backbone graph. The projection adds approximately **66.5 us**, while reducing the classifier saves approximately **12.9 us**.

Observed backbone times also differ across trials even though its graph is unchanged. The large response regressions cannot be assigned entirely to the small measured projection time. CPU/cache/runtime conditions and queue amplification require additional profiling; this experiment does not identify their cause. No observations are dropped to make the result appear unchanged.

### Decision

Retain full width as the reference for this matched-head experiment. The earlier production carry-forward fixed-confidence checkpoint is unchanged. The PCA checkpoint is not promoted: quality uncertainty and live latency both fail their gates. The only established compression result is the selected vector's 50% byte reduction; total RAM, peak memory and energy were not measured.

The next candidate is a **trained narrower encoder in FP32**, with actual convolution/channel work removed and compatible heads. It must pass a new static quality/cost check before adaptive switching or precision changes. This targets the measured dominant component rather than reducing a post-pooling vector while leaving all encoder operations intact. The next implementation has not yet been run.

[Download checkpoints and complete live PCA evidence](https://chatgpt.com/library?file_id=libfile_33a6b3d31e508191a7809ea4b531ef15&account_id=personal).
