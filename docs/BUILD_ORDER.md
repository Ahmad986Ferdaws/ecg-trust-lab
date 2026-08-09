# ECG trust project: build order

This is the completed operational order for the PTB-XL five-superclass
project. Each stage advanced only after its exit gate passed; fold 10 remained
sealed until the final configuration and one-time ledger were frozen.

## Infrastructure

- **Machine:** Ryzen 9 8940HX, 16 cores / 32 threads, about 31 GB RAM.
- **GPU:** NVIDIA RTX 5070 Ti Laptop GPU, 12 GB VRAM, Blackwell `sm_120`.
- **Storage:** the completed data, environment, artifact, and run trees occupy
  about 12 GB; keep at least 20 GB free for checkpoints, sweeps, and reports.
- **Runtime:** project-local Python 3.12.13 managed by `uv`.
- **Training stack:** PyTorch 2.13.0 + CUDA 13.0, cuDNN 9.2, BF16 enabled after
  the verified GPU forward/backward test.
- **Data/science stack:** WFDB, NumPy, pandas, SciPy, scikit-learn, PyArrow.
- **Research stack:** Optuna, TensorBoard, Captum, Matplotlib, Seaborn.
- **Demo stack:** FastAPI, Uvicorn, Jinja2, Plotly, multipart WFDB uploads.
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

6. **Run the fair architecture comparison — complete.** Train the matched-capacity
   ResNet and ECG transformer under the same data, tuning budget, optimizer
   budget, augmentations, and seeds. Record parameters, throughput, peak VRAM,
   wall time, and fold-8 metrics. The paired 12-candidate sweep completed with
   fold-8 leaders of 0.931852 for ResNet and 0.923927 for the transformer.

7. **Freeze choices, then calibrate — complete.** Paired seeds 2026, 2027, and
   2028, the architecture decision, median epoch budgets, all six folds-1–8
   refits, label-free subgroups, the CUDA/runtime/report specification, six
   fold-9 prediction pairs, and six independent temperature/threshold/gate
   policies are frozen and sealed.

8. **Open fold 10 once — complete.** The global spec-keyed exact-six ledgered
   batch completed on August 9, 2026 UTC. It published six immutable prediction
   pairs, six member reports, two architecture summaries, and three paired
   patient-cluster bootstrap comparisons. Exact resumes recovered operational
   hash-representation failures without overwriting predictions or repeating a
   scientific query. No post-test tuning was performed.

9. **Audit trustworthiness — complete.** The immutable r3 audit includes raw
   and calibrated reliability, dense risk-coverage, error detection, subgroup
   performance and coverage, 246 controlled-corruption member-cases, and 900
   explanation-control evaluations using Grad-CAM, Integrated Gradients, and
   temporal occlusion. These are post-evaluation descriptive analyses.

10. **Package the result — complete.** Reproducible tables and figures, the r3
    report, post-evaluation run log, model card, frozen demo binding, and local
    research viewer are present. The browser-level CSP nonce, SVG waveform
    portability correction, and successful isolated-Chromium retest are
    tracked in
    `reports/POST_EVALUATION_RUN_LOG.md`; this operational retest does not
    change the scientific package.

## Completion record

Do not rerun or modify the sealed fold-10 release to reproduce a prettier
result. The authoritative identities are:

- final-evaluation specification:
  `sha256:1f73c021a544ffeb119ffe8e490a16e32ec84247e30bce1ffd895fcffed6c762`;
- final batch:
  `sha256:a4da85d5272b1634baaf953496c3d9efd8917777ad8de13b1b0c6dc754699e62`;
- completed opening ledger:
  `sha256:3bb83554a08832212989ea8f3ea212f6af42c08460edbde9ba130065b1115a57`;
- r3 post-evaluation specification:
  `sha256:5727858f0c22b6311152749e0d9a3d20b3c14f4ee1c72ef9d1cf6e1943434200`;
  and
- r3 derived manifest:
  `sha256:fae6df30090ee59425a347034a7f4272cac5b799582a5742fff9c62b92a092f8`.

Use `docs/REPRODUCIBILITY.md` for the executable workflow and
`reports/POST_EVALUATION_RUN_LOG.md` for the immutable r1/r2/r3 recovery
history. A completed `run-final --resume` is verification-only; it must never
be used as an invitation to regenerate or tune the sealed evaluation.
