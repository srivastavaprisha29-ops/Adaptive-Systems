# Static feasibility pilot — provisional contract

Status: implementation defaults, **not jointly confirmed research choices**. No measured research results are committed. Confirm the tolerance, target hardware and minimum useful benefit before a confirmatory run. Keep a copied configuration and artifact directory for each experiment; do not overwrite a completed run.

| Field | Implemented pilot | Remaining decision |
|---|---|---|
| Task | CIFAR-10, one image to one of ten classes | Confirm this task fits the research question |
| Model | Frozen torchvision ResNet-18 `IMAGENET1K_V1`; pooled 512D features; fitted ridge classifier | Confirm checkpoint/task fit; compare original task-trained models later |
| Dataset | Official torchvision CIFAR-10 loader checks archive integrity; manifest records local data hashes | Confirm dataset licensing/use; preserve downloaded data and hashes |
| Partitions | Seeded class-balanced split of official training set: 44,000 fit / 1,000 calibration / 5,000 validation; official 10,000 test held out | Review protocol before final evaluation |
| Device | CPU only, one thread, batch one, eager FP32 encoder/head | Record actual machine from run manifest; CPU model may need manual annotation if OS omits it |
| State | Steady sequential requests, decoded PIL images already resident; no injected contention | Record background processes, power mode and thermal conditions manually |
| Actions | Full 512D / PCA 128D crossed with FP32 / symmetric INT8 **feature storage** | CPU implementation verified by synthetic checks; hardware-specific profiling still needed |
| Quality | Top-1 accuracy; provisional margin 1.0 absolute percentage point; paired bootstrap 95% interval for candidate minus reference | Confirm margin before inspecting confirmatory results |
| Objective | Lowest validation median end-to-end latency among quality-feasible actions; provisional useful reduction 5% | Confirm minimum practically useful reduction |
| Resources | RSS observations, no enforced memory/latency cap | Set caps before resource-constrained controller evaluation |
| Energy | Unmeasured, null in manifest | Choose supported instrumentation before any joules claim |

## Representation and fitting

The encoder remains unchanged across all four configurations. Fit PCA on training features only. Fit each configuration's ridge classifier on its own adapted training features with the same regularization and training examples. Full width bypasses PCA. Quantizer scale is the calibration-set maximum absolute adapted feature divided by 127; clipping uses [-127,127]. The INT8 payload is a real integer tensor and is restored to FP32 before the head. Conversion, reconstruction and any clipping are part of the measured inference path. This tests lossy feature storage, **not INT8 arithmetic acceleration**. Payload bytes exclude quantizer metadata, the full encoder output, projections, all heads and temporary tensors.

This implementation does not train Matryoshka, Slimmable, Any-Precision or NestDNN models. Those remain literature directions. Learned/nested representations and hardware-accelerated low-precision kernels are later alternatives, after the static feasibility evidence.

## Measurement boundary

Decoded host PIL image -> preprocessing -> encoder -> optional projection -> encode/decode -> classifier -> class argmax. Disk I/O, image decoding and training are excluded and must not be described as request-to-response latency. CPU execution is synchronous. Warm up each action, randomize action order within each repeat, and use identical timed validation examples for all actions. Record raw per-example timings, repeat means, p50/p95 and sequential throughput. Headline latency uses one outer timer around the complete pipeline. Stage timers come from a separate diagnostic inference, so their sum is not substituted for headline latency. Repeats share a process and machine; they are paired blocks, not independent machine/session trials. Raw telemetry is outside timed intervals. Release the quality feature cache before timing. Record baseline RSS, post-inference RSS, and unique backing tensor storage (count shared views once). RSS is sampled after each inference with every head resident: it is **not peak allocation or isolated per-action memory**. Process isolation and peak-memory instrumentation are deferred; do not claim memory-budget compliance from these observations.

Quality evaluation may cache encoder features because it measures predictions, not latency. Timed inference always executes the encoder and preprocessing again. No controller runs here: controller, telemetry and switching costs are absent rather than estimated as zero-cost adaptive operation.

## Selection and held-out evaluation

The paired example-bootstrap interval's lower endpoint must be at least minus the prespecified margin. This is a pilot screening rule with an ordinary two-sided 95% interval, not a multiplicity-adjusted confirmatory guarantee. The full FP32 reference is always eligible. Choose the feasible action with lowest validation p50, save `selection.json`, and pass that exact selection to held-out evaluation. Never choose a new winner from test metrics. Test results for other actions are descriptive; repeatedly tuning after reading test output invalidates its held-out status.

Pilot pass: the selected action differs from reference, passes the quality screen, and has a paired repeat-bootstrap 95% latency-reduction interval whose lower endpoint meets the configured useful reduction. Bootstrap repeat means together to preserve reference/candidate pairing. The selection criterion remains validation p50; the acceptance gate uses mean-latency uncertainty and reports both metrics. Inspect repeat variability and raw records before making a speed claim. A controller needs at least two useful actions with different resource-conditioned trade-offs; a single steady-state winner does not establish a reason to adapt.

## Reproducibility and stop conditions

Prepared artifacts use manifest schema 2 with relative artifact/data checksums, permitting directory relocation; older artifacts require fresh preparation. Store config/hash, split indices, fitted heads/PCA/scale, encoder state, file hashes, source commit/dirty state, versions, installed package inventory, source-file hash, actual thread count, actual environment and raw outcomes. The held-out selection must match encoder, split, action and configuration hashes and declare validation as its source; check it before model evaluation. Only load trusted local `.pt` files; loading uses `weights_only=True`. Artifact hashes are checked before benchmarking. Preparation refuses an existing output directory. Data and model downloads require network access; checkpoint acquisition may occur when preparing if it is not cached.

Stop or revise the adaptation point if the encoder dominates, projection/conversion removes the saving, no reduced action meets quality tolerance, or differences are within timing noise. Defer controller, oscillating workloads, bandits, physical-layout changes and novelty claims until evidence justifies them.
