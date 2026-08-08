# Preliminary development results

**Status:** single-seed model-selection evidence only  
**Protocol:** PTB-XL 1.0.3, 100 Hz, folds 1–7 train, fold 8 model selection  
**Seed:** 2026  
**Labels:** NORM, MI, STTC, CD, HYP

These numbers are not fold-10 test results and are not final research claims.
They establish that the pipeline is credible enough to proceed to equal-budget
tuning and multiple seeds. Fold 9 and fold 10 were not accessed by either run.

## Matched-capacity comparison

| Model | Parameters | Best epoch | Completed epochs | Macro AUROC | Macro AUPRC | Brier | ECE | Peak allocated VRAM | Train time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1D ResNet | 8,739,973 | 12 | 22 | 0.9287 | 0.8261 | 0.0841 | 0.0379 | 457.9 MiB | 389.6 s |
| ECG transformer | 8,726,833 | 24 | 34 | 0.9237 | 0.8050 | 0.0879 | 0.0301 | 868.0 MiB | 618.1 s |

The capacity gap is 0.1503%, well inside the preregistered ±15% tolerance.
Both runs used batch 128, BF16, AdamW, the same warmup-cosine schedule, the same
train-only normalization, and the same early-stopping rule. Mean observed
training throughput was approximately 852 records/s for the ResNet and 830
records/s for the transformer; first-epoch initialization is included.

![Matched-capacity development training curves](figures/development/matched_seed2026_training_curves.png)

## Per-label metrics at each model's selected epoch

| Model | Label | AUROC | AUPRC | Brier | ECE |
|---|---|---:|---:|---:|---:|
| ResNet | NORM | 0.95 | 0.93 | 0.09 | 0.04 |
| ResNet | MI | 0.93 | 0.85 | 0.09 | 0.04 |
| ResNet | STTC | 0.93 | 0.82 | 0.09 | 0.03 |
| ResNet | CD | 0.93 | 0.85 | 0.08 | 0.03 |
| ResNet | HYP | 0.91 | 0.67 | 0.07 | 0.04 |
| Transformer | NORM | 0.95 | 0.93 | 0.09 | 0.04 |
| Transformer | MI | 0.92 | 0.83 | 0.09 | 0.02 |
| Transformer | STTC | 0.93 | 0.79 | 0.09 | 0.02 |
| Transformer | CD | 0.91 | 0.81 | 0.09 | 0.03 |
| Transformer | HYP | 0.91 | 0.66 | 0.07 | 0.04 |

Rounded per-label values are for orientation; machine-readable histories and
checkpoints under `runs/development/` retain full precision and complete
provenance. The apparent ResNet advantage is not yet a conclusion: the next
gate requires equal-budget tuning, at least three seeds, and paired
patient-cluster confidence intervals.

## Selected checkpoint SHA-256

- ResNet: `9051b829d556fd5824f1f52463654f620ca77a075a0653c6407012bece72f8dd`
- Transformer: `9199dc9243cce5f348c1bcd7ffd3d88bc259a74085ab2f7663b9707d5e1fa09a`

The checkpoints are local, gitignored artifacts. These digests identify the
exact files used for any later development-fold export or frozen refit.
