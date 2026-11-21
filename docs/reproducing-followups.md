# Completed experiments and reproduction

The experiments use a frozen ImageNet-pretrained ResNet-18 on CIFAR-10.
The training split contains 44,000 fitting, 1,000 calibration and 5,000
validation images. Held-out evaluation uses the official 10,000-image test set.
The pilot quality tolerance is one percentage point and the latency target
is a 5% mean reduction at the lower confidence bound.

## Findings

| Method | Result | Scope |
| --- | --- | --- |
| Fixed stage-three margin gate | 84.63% test accuracy versus 84.52% full. Mean latency fell 10.54–11.11% in three interleaved AMD sessions. | One model and dataset, single-thread CPU, batch one. |
| Learned 64-dimensional prefix | 85.56% test accuracy versus 85.42% for the same checkpoint's 512D head. Feature payload fell 87.5%. | End-to-end latency benefit was unconfirmed in three Intel sessions. |
| Matched prefix training controls | Joint minus single-width validation accuracy averaged −0.048 points at 64D and −0.004 at 128D. | Five training seeds, all ten paired image intervals included zero. |
| Learned exit score | 31.88% validation exits versus 42.04% for margin, with both within the accuracy budget. | Linear regression on early logits, margin and entropy. |
| Queue-aware routing | Lower bursty p95 in 6/6 runs, with point-estimate accuracy budget met in 2/6. | Saved live request records. Original queue generator source remains missing. |

The Matryoshka implementation trains a projection on frozen features and sums
cross-entropy across prefix heads. The paper trains the backbone as well.
Independent supervised controls use matching prefix initialization and batch order.

## Frozen inputs and saved evidence

Download and extract [Adaptive_Systems_Reproducibility.zip](https://chatgpt.com/library?file_id=libfile_3cefc90f4ef48191a84cc47c5f061aad&account_id=personal).
Its REPRODUCIBILITY.md documents checksums, original and regenerated checkpoints,
data acquisition, and the limits of exact reproduction. Regenerated Matryoshka
weights reproduce the reported validation and held-out metrics but have a
different file hash. Preserve each checkpoint's own validation report.

Set the paths below to the extracted package and original CIFAR-10 data.
Run commands from the repository root after installing with pip install -e .

```bash
python -m adaptive_systems.matryoshka --artifacts /path/to/package/frozen/prepared \
  --data /path/to/data --output runs/matryoshka

python -m adaptive_systems.matryoshka_controls --artifacts /path/to/package/frozen/prepared \
  --data /path/to/data --control-seeds 29 30 31 32 33 --output runs/matryoshka-controls

python -m adaptive_systems.matryoshka_test --artifacts /path/to/package/frozen/prepared \
  --data /path/to/data --pilot runs/matryoshka --output runs/matryoshka/held-out-test.json

python -m adaptive_systems.matryoshka_latency --artifacts /path/to/package/frozen/prepared \
  --data /path/to/data --pilot runs/matryoshka --output runs/matryoshka/latency-34.json --seed 34

python -m adaptive_systems.learned_exit --artifacts /path/to/package/frozen/prepared \
  --data /path/to/data --head /path/to/package/frozen/head.pt --output runs/learned-exit

python -m adaptive_systems.exit_audit --artifacts /path/to/package/frozen/prepared \
  --data /path/to/data --head /path/to/package/frozen/head.pt \
  --matrix /path/to/package/evidence/Adaptive_Systems_Queue_Matrix.zip --output runs/exit-audit.json

python scripts/verify_queue_results.py --evidence /path/to/package/evidence
```

Use a new output path for each run. Feature extraction can take substantial CPU
time. The queue verifier recomputes 64 saved summaries with the standard library.
Identical fixed-policy records reuse one live execution. It does not rerun model
inference or estimate independent latency uncertainty from those shared records.

The reconstructed follow-up sources match the source hashes in their historical
reports. The matched-controls source is preserved as matryoshka_controls.py to
retain both the original pilot implementation and its later controlled variant.
