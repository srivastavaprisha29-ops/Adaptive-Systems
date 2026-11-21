# Knowledge distillation for an early inference exit

## Research question

Can a stage-three classifier learn useful class relationships from the full stage-four classifier, improving its accuracy or the fraction of requests that can exit early?

Reference: Geoffrey Hinton, Oriol Vinyals, Jeff Dean. **Distilling the Knowledge in a Neural Network** (2015), https://arxiv.org/abs/1503.02531. Section 2 defines temperature-softened targets and the temperature-squared gradient correction.

## Fundamentals

A hard label identifies one correct class. A teacher's distribution also expresses similarities among alternative classes. Softmax at temperature T divides each logit by T before normalization. Increasing T softens the distribution. The student matches that distribution during training and predicts normally at temperature 1.

The implemented objective is `(1-alpha) CE(labels, student) + alpha T² KL(teacher_T || student_T)`, using batch-mean KL and detached teacher predictions. KL differs from the paper's soft-target cross entropy by teacher entropy, which is constant with respect to student parameters, so the student gradients are equivalent.

## Adaptation to Adaptive Systems

The teacher is the existing 512-feature ResNet-18 stage-four ridge classifier. The student is a 256-feature stage-three linear classifier. The ImageNet encoder stays frozen. This isolates head training and does not reproduce the paper's MNIST or speech experiments or train an independent compressed backbone.

The initial pilot uses 400 randomly selected training images per class, 1,000 separate calibration images and 5,000 validation images. The official CIFAR-10 test set is untouched by this experiment. Training uses three seeds (29–31), 20 epochs, AdamW, learning rate 0.001, weight decay 0.01 and batches of 256. T=4 and alpha=0.7 are fixed before evaluation. The supervised control has alpha=0 with identical initialization, minibatch permutations, architecture and optimizer. This controls for the benefit of training a new head.

Calibration chooses the highest early-exit coverage on a fixed quantile grid whose paired 95% bootstrap lower bound stays within one percentage point of the teacher. Validation freezes that threshold. Each seed reports raw student accuracy, paired distilled-minus-supervised accuracy uncertainty and gated quality and coverage. Image-bootstrap intervals do not measure variation across training seeds. No latency improvement can be inferred from exit coverage alone.

## Reproduce

From the repository root, after preparing the existing static artifacts:

```bash
python -m adaptive_systems.distillation \
  --artifacts runs/prepared-validation --data data \
  --output runs/distillation-pilot --train-per-class 400 --epochs 20
python -m unittest discover -s tests -v
```

Use a new output directory. For a full-training experiment set `--train-per-class 4400`, with the same prepared split. The runner verifies preparation checksums, saves training indices and hyperparameters, and hashes the preparation, source and saved heads. The saved `heads.pt` uses the same weight/bias layout as the existing early-exit prediction code. It is an experimental checkpoint and does not change the default gate.

Outputs include `heads.pt` and `validation.json`. The initial pilot's evidence is recorded in `docs/research/distillation-pilot.json` when complete. A larger run and independent latency measurements are required before deployment claims.

## Initial pilot results

| Seed | Supervised accuracy | Distilled accuracy | Difference (pp) | Paired 95% interval (pp) |
|---|---|---|---|---|
| 29 | 58.72% | 58.74% | 0.02 | [-0.7799999713897705, 0.8199999928474426] |
| 30 | 56.66% | 58.28% | 1.62 | [0.8794999957084656, 2.319999933242798] |
| 31 | 58.14% | 59.66% | 1.52 | [0.9399999976158142, 2.140000104904175] |

Mean student accuracy is 57.84% for supervised training and 58.89% for distillation. Two of three image-bootstrap intervals exclude zero. This is a small-subset pilot with fixed optimization settings, not evidence of a broadly reliable gain. Both student variants are far below the existing 44,000-image ridge early head (78.30% validation accuracy), so the current production candidate is retained. The comparisons use different training budgets and do not isolate the reason for that gap.

Distilled gated validation accuracy ranges from 84.60% to 84.64%, with only 4.98–5.94% early exits. The teacher achieves 84.72%. Coverage is insufficient to claim improved system efficiency and latency was not measured. The next experiment should address training convergence on the full training split while retaining a matched supervised control.
