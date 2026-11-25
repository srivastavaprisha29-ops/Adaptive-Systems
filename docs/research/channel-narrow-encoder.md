# Structurally narrowed ResNet-18 FP32 encoder

## Model and fitting protocol

This is a same-family ResNet-18 channel experiment. Both encoders retain stages 1–3 and a 512D final embedding. The reduced model narrows the **internal channels of both final-stage residual blocks**, before pooling. In each block, conv1 produces fewer channels and conv2 consumes that smaller tensor while retaining 512 output channels for the residual addition.

The calibration grid is 256, 384 and 448 internal channels, compared with 512 in full width. Selection retains channels with the largest outgoing conv2 weight energy, using pretrained weights only. New Conv2d and BatchNorm modules allocate physically smaller tensors. This is not activation masking, post-pooling PCA or a different network architecture. Convolution shape inventories and forward-hook MAC counts are saved for both graphs. MAC counts are a convolution-only work proxy, not measured time or total floating-point operation counts.

Both encoders use the same 4,000 fitting images and 1,000 calibration images from the previous matched-head study. For both the full and narrowed final stage, batch-normalization running statistics are reset and recalibrated on fitting images only, using cumulative minibatch statistics. Both compatible 512D ridge heads are then fit on those same 4,000 images with ridge parameter 1.0. No gradient fine-tuning occurs. This changes full-reference BN statistics from the older frozen-BN experiment, so new accuracy should not be presented as a direct update to that old checkpoint.

The width is selected solely on calibration accuracy: choose the smallest reduced width whose paired accuracy-delta lower 95% bound versus the matched full reference stays within one percentage point. If none qualifies, the highest-calibration-accuracy reduced model is retained only as a flagged diagnostic. Width is never retuned after held-out results.

## Data separation and live measurement

Training/calibration indices are identical to the prior PCA study. The new 2,000-image official-test panel excludes the 2,000 PCA panels and the earlier 2,304 load-sweep panels. Historical static aggregate results exist on the official test set; these are fresh dynamic-study panels rather than globally untouched research data.

Stages 1–3 are cached only to make fitting and offline scoring efficient. Both compared models share those unchanged prefix weights. The caches contain FP32 spatial activations at stage 3, not final embeddings. **Live requests execute the entire encoder**, including stages 1–4, on every request. No cached activation substitutes for timed inference.

Six paired sessions use disjoint 128-image panels, with identical image order and scheduled arrivals for the two models. Full-first and narrow-first orders each occur three times. A calibration-only full-reference profile sets a constant interarrival gap at nominal 80% load. All requests drain through the single-server FIFO; none are dropped or truncated.

Response time runs from scheduled arrival to completed scalar prediction. Separate nested timers cover preprocessing, complete encoder work and classifier. Queue waiting is measured separately. Inputs are decoded before timing, preprocessing is inside it, and inference is batch one/single-thread CPU. The recorded output vector remains **512D FP32 / 2,048 bytes** for both models. INT8 feature storage is not part of this experiment.

## Acceptance and scope

The retained target is at least **5% lower p95 response time**, with a paired-session 95% lower improvement bound above the target, and at most **one percentage point held-out accuracy loss**, checked by the paired image-bootstrap lower bound. Six sessions on one shared host are a bounded pilot; uncertainty, timing outliers and the train-only BN adaptation are reported. Calibration width selection is not a formal simultaneous quality guarantee.

The artifact reports whether this matched pilot gate passed, while keeping production adoption and broader controller validation false. A passing result motivates full-budget/fresh-load confirmation; a failing result is not repaired by changing the held-out width or accuracy tolerance.

## Reproduce

Use the frozen model/data preparation described in [reproducing follow-ups](../reproducing-followups.md), the [live PCA input artifact](https://chatgpt.com/library?file_id=libfile_33a6b3d31e508191a7809ea4b531ef15&account_id=personal), and the [load-sweep artifact](https://chatgpt.com/library?file_id=libfile_100d72146b508191b0012cd80508d18f&account_id=personal). Use a fresh output directory.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m adaptive_systems.channel_narrow \
  --artifacts /path/to/frozen/prepared --data /path/to/data \
  --prior-pca /path/to/live-pca --prior-sweep /path/to/load-sweep \
  --settings configs/channel-narrow.json --output runs/channel-narrow-new
python scripts/verify_channel_narrow.py --run runs/channel-narrow-new
python -m unittest discover -s tests -v
```

Preparation uses Linux fork workers and temporary memory-mapped prefix activations. The caches are reproducible intermediates and are excluded from the durable evidence package. The checkpoint preserves the selected narrowed final stage, full-reference adapted BN state, both ridge heads and kept-channel indices. The original frozen encoder checkpoint is required to reconstruct the shared prefix and unchanged full convolution weights. The full and reduced convolution shapes, parameter counts, MAC estimates, calibration results, held-out predictions, live request records and hashes are retained.

The standard-library verifier recomputes quality point estimates, width selection, recorded quality-gate logic, structural shape checks, FIFO timing, encoder means, paired-session timing intervals and the joint pilot decision. It verifies the recorded image-bootstrap quality bound rather than independently regenerating NumPy's bootstrap distribution.

## Results

All three reduced widths failed the calibration lower-bound screen. Width 448 had the highest calibration accuracy and was frozen only as the predeclared diagnostic candidate. It was not treated as quality-qualified.

| Layer4 internal width | Calibration accuracy | Delta vs full | Paired 95% interval | Qualified |
|---|---:|---:|---|---|
| 256 | 74.80% | -5.70 pp | [-8.20, -3.40] pp | No |
| 384 | 78.40% | -2.10 pp | [-4.10, -0.10] pp | No |
| 448 | 80.30% | -0.20 pp | [-2.00, +1.50] pp | No |

Matched full-reference calibration accuracy was **80.50%**.

### Structural proof

| Convolution | Full weight shape | Narrow weight shape |
|---|---|---|
| layer4.0.conv1 | [512, 256, 3, 3] | [448, 256, 3, 3] |
| layer4.0.conv2 | [512, 512, 3, 3] | [512, 448, 3, 3] |
| layer4.1.conv1 | [512, 512, 3, 3] | [448, 512, 3, 3] |
| layer4.1.conv2 | [512, 512, 3, 3] | [512, 448, 3, 3] |

Encoder parameters fall from **11,176,512** to **10,144,064** (9.24%). Convolution-only MAC estimates fall from **1,813,561,344** to **1,762,983,936** (2.79%). Both final embeddings remain 512D FP32 / 2,048 bytes. The change occurs inside the encoder, but narrowing only late, low-spatial-resolution channels removes much less total convolution work than parameter count alone suggests.

### Held-out quality and live timings

| Metric | Matched full width | Diagnostic narrow 448 |
|---|---:|---:|
| Held-out accuracy (2,000 fresh-panel images) | 83.00% | 82.35% |
| Mean encoder time | 57.582 ms | 56.540 ms |
| Mean service time | 59.737 ms | 58.723 ms |
| Mean session p95 queue wait | 1086.958 ms | 1036.278 ms |
| Mean session p95 response | 1147.730 ms | 1097.282 ms |

Held-out accuracy delta is **-0.65 pp**, with paired 95% interval **[-1.80, +0.60] pp**. The one-point preservation bound fails. Mean encoder time falls about **1.81%**, while the paired-session mean p95 reduction is **-8.86%**, with a wide 95% interval **[-57.28%, +38.11%]**. The 5% latency target is not established.

The p95 row averages per-session p95s, while the paired reduction averages within-session ratios. They are different aggregations and should not be substituted for one another. Large timing/queue variability remains even with preparation and auxiliary execution kept outside the live window. All six sessions are retained. Encoder-time reduction and smaller convolution tensors alone do not establish a response-time win.

**Decision: do not adopt this candidate.** It fails calibration qualification and held-out quality/latency gates. This is a negative result for pretrained outgoing-weight-energy pruning with BN/ridge adaptation, not evidence that all channel-reduced ResNet models fail. There was no gradient fine-tuning to recover pruned-channel accuracy. A future reduced-work candidate needs accuracy-recovery training with a matched full-reference training procedure, followed by calibration selection and fresh held-out evaluation. More aggressive late-stage widths remove more work but failed quality here; parameter counts must not substitute for complete operation/time measurements.

The older production carry-forward checkpoint remains unchanged. INT8 storage remains separate. No controller, width switching or production replacement is introduced by this diagnostic.

## Verification

All **44 unit tests pass**, including actual smaller Conv2d tensor allocation, preserved residual output shape, unmodified full reference, weight-slice mapping and calibration fallback status. The independent verifier checked **2,000 held-out predictions** and **1,536 live request records**, structural graph shapes/MAC reduction, exact vector bytes, fresh panels, timing decomposition and the paired-session decision. The fitted checkpoint is retained without the reproducible large activation caches.

[Download the fitted channel-narrow checkpoint and complete evidence](https://chatgpt.com/library?file_id=libfile_fa70081fd2c48191887e3f7bf9d5fd08&account_id=personal). Reconstruct the shared prefix/full convolution weights from the original encoder checkpoint whose hash is recorded in the lock.
