# Full-training distillation follow-up

## Question and fixed protocol

The first pilot used 4,000 training images for 20 epochs and produced weak student heads. This follow-up asks whether a larger, better optimized training run produces a useful early classifier, and whether teacher soft targets outperform a matched supervised control under that budget.

Both methods use all 44,000 training images from the existing preparation. There are three seeds (29, 30, 31), 60 epochs, batches of 256, AdamW at initial learning rate 0.001 with weight decay 0.01, and cosine learning-rate decay over the fixed budget. Initialization and minibatch permutations match within each seed. Teacher and encoder weights remain fixed. Distillation temperature stays 4 and alpha stays 0.7. The supervised control uses alpha=0.

Training loss and training accuracy are recorded at epoch 1, every fifth epoch, and the final epoch. These diagnostics do not select checkpoints. The final checkpoint is always evaluated. A falling training objective is evidence of optimization progress, not proof of global convergence.

Calibration uses the existing 1,000-image split to freeze each gate's margin threshold. Validation uses the existing 5,000-image split with paired image-bootstrap uncertainty. The official test set is untouched. The validation set has been used in previous experiments, so these findings remain exploratory.

This changes training size, number of updates and learning-rate schedule together relative to the pilot. The within-run comparison isolates the distillation objective. The pilot-to-full comparison cannot determine which training change caused a gain.

## Reproduce

```bash
python -m adaptive_systems.distillation \
  --artifacts runs/prepared-validation --data data \
  --output runs/distillation-full --train-per-class 4400 \
  --epochs 60 --cosine --feature-cache runs/distillation-features
```

The optional feature cache stores stage-three features, teacher logits and labels with the exact split indices, preparation checksums and feature-extraction source checksums. A mismatching cache is rejected. Preparation and data verification still run before cache reuse. Cached features accelerate offline training and are excluded from inference timing claims.

Outputs are six student heads in `heads.pt` and a `validation.json` containing fixed hyperparameters, training histories, split indices, calibrated gates, validation results, paired intervals and provenance hashes. Source and result hashes allow the saved evidence to be tied to this implementation.

No default inference-policy change is made by this runner. A stronger student needs an independent held-out evaluation and interleaved end-to-end latency measurement before replacing the current candidate.

## Results

| Seed | Supervised accuracy | Distilled accuracy | Gain (pp) | Paired 95% interval (pp) | Distilled exit coverage | Gated accuracy |
|---|---|---|---|---|---|---|
| 29 | 71.70% | 75.14% | 3.44 | [2.60, 4.28] | 26.66% | 84.64% |
| 30 | 71.68% | 75.20% | 3.52 | [2.68, 4.36] | 21.70% | 84.70% |
| 31 | 71.74% | 75.06% | 3.32 | [2.46, 4.16] | 26.66% | 84.66% |

Mean student accuracy is **71.71% supervised** and **75.13% distilled**, a **3.43 percentage-point gain**. Each seed's paired image-bootstrap interval excludes zero. These intervals quantify image sampling uncertainty for the fitted checkpoints, not a population-level confidence interval across training seeds.

Training objectives decrease for every run. Final supervised training accuracy is 71.90–71.93% and distilled training accuracy is 74.89–74.93%. Training and validation accuracy are similar, which does not suggest a large generalization gap for these settings. The fixed cosine schedule drives the last updates toward zero, so a flat final trace alone does not establish convergence to the best achievable classifier. More data, more updates and a changed schedule all contribute to the pilot-to-full change and are not separately identified.

The existing 44,000-image ridge early head remains stronger at 78.30% validation accuracy. Its existing margin gate exits 42.04% of validation images at 84.74% gated accuracy, while the distilled gates here exit only 21.70–26.66% at 84.64–84.70%. The full teacher remains at 84.72%. The distillation objective helps the matched neural-head control, but these checkpoints do not improve the existing system candidate. There is no latency result or official-test result for this follow-up, and no default policy is replaced.

The teacher uses ridge scores passed through a temperature softmax. Those scores were not trained as calibrated probabilistic logits. This is a limitation of the adaptation and should inform further teacher/student experiments.

The complete report and fitted checkpoints are retained with frozen feature caches. The compact companion `distillation-full.json` preserves all results, training histories and provenance while replacing the 44,000 training-index list with its count and SHA-256 hash. Saved checkpoints were reloaded and their validation accuracies independently recomputed against the report. All 17 unit tests pass.
