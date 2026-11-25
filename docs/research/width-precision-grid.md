# Four fixed width/precision configurations and an encoder-work control

## Frozen four-configuration experiment

The grid reuses the exact ResNet-18 checkpoint, matched 4,000-image ridge heads and calibration-selected PCA 256D projection from the live PCA comparison. No head is retrained for INT8. The same 2,000 held-out images estimate accuracy. Four paired 128-request panels replay the first four prior live image/order traces at below/near/above calibration-derived loads. Within each load, all four configurations receive identical image IDs and scheduled arrivals. Four Latin-square action orders balance each configuration across execution positions once per load. Panels repeat across loads, so the three loads do not provide independent image samples.

INT8 is **storage only**: symmetric rounding/clipping to [-127, 127], with one calibration-maximum scale per width, then reconstruction to FP32 before the unchanged FP32 classifier. The backbone and PCA arithmetic remain FP32. Quantization and reconstruction are both inside service timing. Test data never refit scales.

| Configuration | Held-out accuracy | Raw vector bytes | Classifier-input bytes after conversion |
|---|---:|---:|---:|
| full_fp32 | 82.50% | 2048 | 2048 |
| reduced_fp32 | 81.85% | 1024 | 1024 |
| full_int8_storage | 82.05% | 512 | 2048 |
| reduced_int8_storage | 82.05% | 256 | 1024 |

INT8 scale metadata has a defined 4-byte FP32 serialized-format allowance per configured scalar, amortizable across vectors. This is not a measurement of Python-object RAM or an implemented network transfer. JSON stores the calibration scale as a number. Raw payload byte counts come from actual INT8/FP32 tensor sizes. The original backbone vector, PCA matrix and reconstructed FP32 working buffer still exist. No total-RAM, peak-memory or INT8-kernel acceleration claim follows.

Three live 24-image calibration profiles per configuration select the fastest calibration-quality-feasible fixed reference by median mean service. The selected reference is `reduced_int8_storage`. Its profile score is 45.18 ms versus 45.92 ms for full FP32, 46.61 ms for reduced FP32 and 47.41 ms for full INT8 storage. This small calibration timing selection is not proof that INT8 intrinsically accelerates the model. Held-out quality uncertainty is still reported independently, and this experimental reference does not replace the earlier production carry-forward checkpoint.

All test policies/scales are frozen before evaluation. Loads are 0.65/0.95/1.15 times full-FP32 calibration service capacity. Timing covers scheduled arrival to completed prediction, including waiting, preprocessing, unchanged backbone, projection/adapter dispatch, quantization, reconstruction and classifier. Full width's adapter timer includes a no-op function dispatch. Images are decoded before timing. Component means are nested attribution and must not be added again to response time.

## Measured live metrics

Each value is the arithmetic mean of four session metrics, not a pooled request p95. Accuracy above uses all 2,000 held-out images; live timing panels use 512 of those images. All 6,144 live predictions match the frozen offline predictions.

| Load | Configuration | Mean service | Mean session p95 wait | Mean session p95 response | Quantization | FP32 reconstruction |
|---|---|---:|---:|---:|---:|---:|
| below | full_fp32 | 51.204 ms | 114.876 ms | 183.942 ms | 0.00 us | 0.00 us |
| below | reduced_fp32 | 51.473 ms | 277.540 ms | 342.321 ms | 0.00 us | 0.00 us |
| below | full_int8_storage | 54.541 ms | 260.484 ms | 325.534 ms | 64.57 us | 24.58 us |
| below | reduced_int8_storage | 50.135 ms | 63.871 ms | 129.502 ms | 51.64 us | 24.70 us |
| near | full_fp32 | 73.651 ms | 3343.294 ms | 3394.680 ms | 0.00 us | 0.00 us |
| near | reduced_fp32 | 46.656 ms | 125.038 ms | 177.148 ms | 0.00 us | 0.00 us |
| near | full_int8_storage | 48.256 ms | 164.998 ms | 227.231 ms | 48.30 us | 23.14 us |
| near | reduced_int8_storage | 88.272 ms | 5186.898 ms | 5235.574 ms | 111.27 us | 31.23 us |
| above | full_fp32 | 64.555 ms | 3126.596 ms | 3170.136 ms | 0.00 us | 0.00 us |
| above | reduced_fp32 | 45.212 ms | 639.967 ms | 686.887 ms | 0.00 us | 0.00 us |
| above | full_int8_storage | 44.773 ms | 542.081 ms | 591.055 ms | 38.88 us | 16.05 us |
| above | reduced_int8_storage | 45.345 ms | 664.631 ms | 711.226 ms | 42.93 us | 18.07 us |

The trial set has severe backbone/runtime stalls, including multi-second service outliers that amplify queue waiting. The raw records retain them. Small format-conversion costs cannot by themselves explain these large p95 swings. Point fastest configurations differ by load, but that does not establish a stable switching case.

The analysis bootstraps complete paired sessions (8,000 draws), not individual dependent request latencies. Ordered-pair/load adoption checks require at least 5% p95 improvement and an accuracy-loss lower bound within one point, with a conservative family adjustment across 72 endpoints. Accuracy uncertainty uses the full 2,000-image paired panel once per ordered pair; repeating a load does not create new images. Four paired trials are a limited pilot, not a finite-sample confidence guarantee.

Reduced FP32 passes a near-load comparison against the calibration-selected reduced INT8 reference: mean paired p95 reduction 70.17%, descriptive interval [44.23%, 96.12%], accuracy delta -0.20 pp. That comparison is retained. However, no reverse ranking at a different load satisfies both gates, and no configuration establishes a supported universal win. **There is no supported crossover on which to build a width/precision controller.** Runtime instability and image-quality uncertainty also prevent a broad format-adoption claim. No controller is built.

## Conditional encoder-work test: executed

The next control changes actual encoder work in FP32. It uses the official Torchvision 0.20 ImageNet-pretrained **MobileNetV3-Small**, replacing its ImageNet classifier with a ridge head fit on the same 4,000 images as the ResNet comparison. Both use the same 224px input preprocessing, 1,000 calibration images, 2,000 test images and four prior matched live traces. It is also compared to the exact strongest calibration-selected grid action on those traces.

This changes architecture and operators as well as early channel widths; it is a **compact-encoder architecture control**, not an isolated width multiplier or pruning ablation of ResNet-18. The pretrained model and transforms are documented by [Torchvision 0.20](https://docs.pytorch.org/vision/0.20/models/generated/torchvision.models.mobilenet_v3_small.html). The fitted encoder checkpoint and weights URL/hash are retained. No encoder fine-tuning or INT8 arithmetic is used.

| Control | Encoder parameters | Embedding bytes | Held-out accuracy | Mean service | Mean session p95 response | Mean backbone time |
|---|---:|---:|---:|---:|---:|---:|
| ResNet-18 FP32 | 11,176,512 | 2048 | 82.50% | 51.530 ms | 777.873 ms | 49.860 ms |
| Selected grid fixed | 11,176,512 | 256 | 82.05% | 45.717 ms | 85.747 ms | 44.036 ms |
| MobileNetV3-Small FP32 | 927,008 | 2304 | 83.65% | 9.336 ms | 15.095 ms | 7.532 ms |

The compact control has a **larger final vector** (576D/2,304 bytes) than ResNet's 512D/2,048 bytes, yet is much faster. This distinguishes encoder work reduction from post-pooling feature storage compression.

- Versus matched ResNet FP32: **85.28% mean paired p95 reduction**, 95% interval **[76.23%, 94.39%]**. Accuracy delta **+1.15 pp**, interval **[-0.70, +2.95] pp**.
- Versus selected grid fixed: **80.04% mean paired p95 reduction**, interval **[75.83%, 86.55%]**. Accuracy delta **+1.60 pp**, interval **[-0.20, +3.35] pp**.

It passes the 5%/one-point gate against both references in this matched pilot. The architecture-control intervals are ordinary unadjusted 95% intervals on already studied test data, not a confirmatory family-adjusted deployment claim. Mean session p95s and mean paired relative improvements are different aggregation rules; ResNet timing outliers also influence the response means. The compact backbone averages 7.53 ms versus 49.86 ms in the full-ResNet control, providing direct component evidence that the changed encoder is doing cheaper work. This does not isolate which architecture feature caused the difference.

**Decision:** continue with the compact FP32 encoder as the next fixed candidate. Do not deploy it or build a controller yet. It needs a full-training-budget comparison and fresh multi-load confirmation against the strongest fixed model. Current results compare matched 4,000-image heads, not the earlier 44,000-image production head. The earlier production carry-forward remains unchanged.

## Verification and reproduction

All **41 unit tests pass**. The two independent verifiers check **6,144 grid requests** plus **1,536 compact-control requests**, along with all held-out accuracy points, frozen checkpoints, byte accounting, conversion timing, FIFO boundaries, paired-session timing calculations and recorded adoption-gate logic. NumPy's image-bootstrap distributions are preserved rather than independently regenerated by the standard-library verifiers.

Use the inputs and prior artifact linked in [the live PCA comparison](live-pca-comparison.md). Use fresh output directories.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m adaptive_systems.width_precision   --artifacts /path/to/frozen/prepared --data /path/to/data   --prior-pca /path/to/live-pca --output runs/width-precision-new
python scripts/verify_width_precision.py --run runs/width-precision-new

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m adaptive_systems.compact_encoder   --artifacts /path/to/frozen/prepared --data /path/to/data   --prior-pca /path/to/live-pca --grid runs/width-precision-new   --output runs/compact-encoder-new
python scripts/verify_compact_encoder.py --run runs/compact-encoder-new
python -m unittest discover -s tests -v
```

The compact runner deliberately rejects a grid with a supported universal fixed winner or a supported crossover, because those results call for a different branch. Weight download uses Torchvision's official URL. Preparation uses Unix fork workers; timed inference remains single-thread and batch one. The evidence preserves full fitted checkpoints and request records. All findings are exploratory on one shared host and do not establish controller superiority, portability, energy savings or peak-memory savings.

[Download the complete grid and compact-encoder evidence](https://chatgpt.com/library?file_id=libfile_1a275f44dac481919c751bfd45cb6e28&account_id=personal). The archive has `grid` and `compact-encoder` folders with fitted checkpoints, locks and raw records.
