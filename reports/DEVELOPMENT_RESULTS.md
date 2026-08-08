# Development results

**Status:** equal-budget single-seed model-selection evidence; multi-seed confirmation pending

**Protocol:** PTB-XL 1.0.3, 100 Hz, folds 1–7 train, fold 8 model selection  
**Seed:** 2026  
**Labels:** NORM, MI, STTC, CD, HYP

These numbers are not fold-10 test results and are not final research claims.
They establish that the pipeline is credible enough to proceed to paired
multi-seed confirmation. Fold 9 and fold 10 were not accessed by any run.

## Equal-budget paired sweep

Both architectures completed the same immutable 12-row Latin-hypercube
candidate plan with a fixed seed of 2026, a 30-epoch ceiling, identical data
roles, no pruning, and alternating first-model order. A machine-verified sweep
summary confirms 12 unique complete candidates for each architecture. One
ResNet attempt was interrupted by the terminal transport, marked failed, and
retried with the identical candidate and seed; it reproduced the partial
history exactly and did not consume the candidate budget.

| Model | Winning candidate | Best epoch | Completed epochs | Fold-8 macro AUROC | Peak allocated VRAM | Runtime | Batch | Learning rate | Weight decay |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1D ResNet | 11 | 12 | 22 | 0.931852 | 455.4 MiB | 382.6 s | 128 | 0.00242420 | 0.0000147610 |
| ECG transformer | 6 | 21 | 30 | 0.923927 | 1,204.6 MiB | 793.6 s | 192 | 0.00123597 | 0.00141241 |

The seed-2026 development difference is 0.007925 macro-AUROC in favor of the
ResNet. This is not yet the architecture decision: the preregistered rule uses
the mean over paired seeds 2026, 2027, and 2028 with a practical margin of
0.005. Both architectures proceed through all later stages regardless of that
decision.

The final sweep summary is identified by SHA-256
`04574c17773dea894efd84bf2ed5fa5be685ce3a6687deb3524a373ea6d8df6b`.
Its shared candidate-plan hash is
`fd1c140be07f8f99d1655c21e8986bac6759c1838be438e22fd18c475a51f8f7`.
The exact winner checkpoints are:

- ResNet candidate 11: `5eaa84fa5f47a66cbd4f9ccc3bb5fea75f7abf2a6ff31bff288d90ba9089cd97`
- Transformer candidate 6: `004cf4be1d423fbdb4828f098467d0747b73c25df6c86678ced1186d2b87a476`

The verified sweep-winner training-curve figure is generated at
`artifacts/figures/sweep_winner_training_curves.png`.

## Initial matched-capacity comparison

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
provenance. The initial comparison motivated the matched sweep above. The
apparent ResNet advantage is still not a final conclusion: the next gate
requires all three paired seeds, and the final architecture comparison
requires paired patient-cluster confidence intervals on the single authorized
fold-10 batch.

## Selected checkpoint SHA-256

- ResNet: `9051b829d556fd5824f1f52463654f620ca77a075a0653c6407012bece72f8dd`
- Transformer: `9199dc9243cce5f348c1bcd7ffd3d88bc259a74085ab2f7663b9707d5e1fa09a`

The checkpoints are local, gitignored artifacts. These digests identify the
exact files used for any later development-fold export or frozen refit.
