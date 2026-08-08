# ECG trust project: build order

This is the operational order for the PTB-XL five-superclass project. A stage
does not advance until its exit gate passes; fold 10 remains sealed until the
final configuration is frozen.

## Infrastructure

- **Machine:** Ryzen 9 8940HX, 16 cores / 32 threads, about 31 GB RAM.
- **GPU:** NVIDIA RTX 5070 Ti Laptop GPU, 12 GB VRAM, Blackwell `sm_120`.
- **Storage:** PTB-XL 100 Hz plus environments and artifacts require only a few
  gigabytes; keep at least 20 GB free for checkpoints, sweeps, and reports.
- **Runtime:** project-local Python 3.12.13 managed by `uv`.
- **Training stack:** PyTorch 2.13.0 + CUDA 13.0, cuDNN 9.2, BF16 enabled after
  the verified GPU forward/backward test.
- **Data/science stack:** WFDB, NumPy, pandas, SciPy, scikit-learn, PyArrow.
- **Research stack:** Optuna, TensorBoard, Captum, Matplotlib, Seaborn.
- **Quality stack:** pytest, Ruff, strict mypy, Git, deterministic run metadata.

The machine has enough capacity. The main advantage of the GPU is fast
iteration and equal-budget sweeps, not fitting an otherwise oversized model.

## Ordered stages

1. **Lock the scientific contract — complete.** Use PTB-XL 1.0.3, 100 Hz,
   canonical leads, multi-hot labels `[NORM, MI, STTC, CD, HYP]`, folds 1–7 for
   training, fold 8 for model selection, fold 9 for calibration/gating, and
   fold 10 for one-time final evaluation.

2. **Acquire and prove the data — complete.** The 100 Hz waveform set and all
   selected metadata pass the official PhysioNet SHA-256 inventory. The
   deterministic task manifest contains 21,388 labeled ECGs from 18,617
   patients and passes record, label, file, fold, and patient-isolation gates.

3. **Freeze preprocessing — complete.** Load representative real records, verify exact
   `[12, 1000]` float32 signals and lead order, compute per-lead mean/std from
   folds 1–7 only, save provenance and hashes, and create dataset overview
   figures.

4. **Prove the pipeline cheaply — complete.** Run the training-prevalence baseline, then
   overfit 64 records. A model that cannot overfit this subset indicates a
   wiring, target, optimizer, or normalization problem.

5. **Reproduce a convolutional benchmark — complete.** Train the 1D ResNet at 100 Hz,
   tune only on fold 8, and investigate any large gap from the published
   roughly 0.92–0.93 macro-AUROC range before adding complexity.

6. **Run the fair architecture comparison — active.** Train the matched-capacity
   ResNet and ECG transformer under the same data, tuning budget, optimizer
   budget, augmentations, and seeds. Record parameters, throughput, peak VRAM,
   wall time, and fold-8 metrics.

7. **Freeze choices, then calibrate.** Refit the two chosen configurations on
   folds 1–8 with frozen epoch budgets. Fit temperature scaling, thresholds,
   and abstention cutoffs only on fold 9. Save the resolved configuration and
   protocol hashes before final testing.

8. **Open fold 10 once.** Generate immutable logits/probabilities and report
   per-label and macro AUROC/AUPRC, Brier score, log loss, ECE, thresholded
   metrics, risk–coverage curves, and paired patient-cluster bootstrap
   intervals. Do not tune after inspecting these results.

9. **Audit trustworthiness.** Measure subgroup performance and gate coverage;
   run baseline-wander, noise, amplitude, time-shift, and lead corruptions;
   compare Grad-CAM, Integrated Gradients, and occlusion with deletion,
   ablation, stability, and parameter-randomization checks.

10. **Package the result.** Build the local research viewer, add data/model
    cards and limitations, export reproducible tables/figures, and present the
    system as research software—not a diagnostic or medical device.

## Immediate command sequence

Run from the project root:

```powershell
uv run ecg-verify
uv run python scripts/smoke_dataset.py
uv run python scripts/sweep.py preflight
uv run python scripts/sweep.py status
uv run python scripts/sweep.py run
uv run pytest
uv run ruff check src tests scripts
uv run mypy
```

Use `scripts/sweep.py run --resume` only for the exact persisted candidate
plan after an interruption. No fold-9 or fold-10 command belongs in routine
development instructions.
