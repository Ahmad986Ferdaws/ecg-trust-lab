# Research blueprint: trustworthy 12-lead ECG classification on PTB-XL

**Research snapshot:** August 8, 2026  
**Dataset version:** PTB-XL 1.0.3  
**Primary task:** five diagnostic superclasses, multi-label  
**Models:** 1D ResNet and patch-based ECG transformer  
**Core contribution:** calibrated selective prediction with subgroup, robustness, and explanation audits

## 1. Executive decision

This is a strong, feasible portfolio project for the available machine, but the strongest version is no longer simply “train a ResNet and a transformer and compare AUROC.” A July 2026 preprint already performed a closely overlapping three-architecture comparison on PTB-XL and reported a 1D ResNet macro-AUROC of 0.9241 alongside age- and sex-stratified results. Repeating only that experiment would be useful engineering practice, not a distinctive research contribution.

The project should instead ask:

> When two ECG architectures have similar discrimination on PTB-XL, which one produces the most reliable system after calibration, selective abstention, subgroup auditing, corruption testing, and explanation validation?

The intended claim is deliberately narrower than “clinical-grade ECG AI”:

> On the public PTB-XL benchmark, we built a reproducible pipeline that compares architecture, calibrated probability quality, selective risk, subgroup coverage, robustness, and explanation fidelity under a leakage-resistant patient-level protocol.

That is coherent with a “trustworthy ML in high-stakes domains” research identity and is supportable by the dataset. It does not imply clinical deployment readiness.

## 2. Corrections to the initial project description

Several details matter for scientific accuracy:

1. The current PTB-XL 1.0.3 release contains **21,799 ECG records from 18,869 patients**, not 21,837 records. Version 1.0.2 removed 36 duplicate waveforms and 1.0.3 removed two more.
2. PTB-XL has **71 total ECG statements**, not 71 diagnostic labels. The benchmark paper describes 44 diagnostic statements, 19 form statements, and 12 rhythm statements; some categories overlap. Diagnostic labels also map to 24 subclasses and five superclasses.
3. The five superclasses are **NORM, MI, STTC, CD, and HYP**. This is better described as diagnostic ECG statement classification than pure arrhythmia classification. Rhythm prediction is a separate 12-statement task.
4. The target is **multi-label**, not mutually exclusive five-class classification. One ECG can have more than one positive superclass, so outputs use independent sigmoid probabilities and binary targets.
5. The original benchmark's directly comparable protocol uses **100 Hz signals**, patient-respecting folds 1–8 for training, fold 9 for validation, and fold 10 for testing. It reported 0.930 macro-AUROC for the Wang 1D ResNet on the superclass task; the accompanying leaderboard and later work put strong models in roughly the 0.92–0.93 range.
6. “Patient-level cross-validation” must not be implemented by inventing a new random split. PTB-XL already supplies patient-respecting `strat_fold` assignments. Direct literature comparison requires preserving them.

## 3. Dataset dossier

### 3.1 What is available

PTB-XL 1.0.3 provides:

- 21,799 ten-second, clinical 12-lead ECGs from 18,869 patients.
- Standard leads I, II, III, aVL, aVR, aVF, and V1–V6.
- Waveforms at 500 Hz and downsampled copies at 100 Hz.
- WFDB-formatted 16-bit signals at 1 µV/LSB resolution.
- Demographic and acquisition metadata, including age, sex, height, weight, site, device, and date.
- SCP-ECG statements with likelihood values, diagnostic hierarchy, report strings, and signal-quality metadata.
- Patient-respecting, stratified folds and higher-quality human-reviewed labels in folds 9 and 10.
- CC BY 4.0 licensing and open access.
- A 1.7 GB download and approximately 3.0 GB uncompressed footprint.

Superclass record counts overlap because the task is multi-label: NORM 9,514; MI 5,469; STTC 5,235; CD 4,898; HYP 2,649.

### 3.2 Dataset limits that constrain the claims

- The ECGs were acquired on Schiller devices in a historical German clinical cohort between 1989 and 1996. This is a single-source, retrospective dataset, not evidence of present-day multi-site generalization.
- Labels originate from clinical reports standardized into SCP-ECG statements. They are useful benchmark targets but are not a prospective adjudication study.
- Age and sex are available, but race/ethnicity is not suitable for a comprehensive demographic fairness claim.
- Multiple conditions co-occur, so aggregate superclass performance can hide low-performing diagnostic subtypes.
- Fold 10 is a public benchmark test fold. Repeatedly inspecting it during development would turn it into another validation set.
- Synthetic corruptions measure controlled sensitivity, not real clinical distribution shift.

## 4. Literature positioning and novelty

### 4.1 Reproduction anchor

The 2021 PTB-XL benchmark established the standard evaluation pattern and found convolutional models, especially ResNet- and Inception-style architectures, strongest across tasks. For the five diagnostic superclasses at 100 Hz, reported macro-AUROCs included 0.930 for `resnet1d_wang` and 0.921 for `inception1d`; the paper also analyzed hidden stratification, ensemble uncertainty, attribution, split leakage, and 100 Hz versus 500 Hz sampling.

The accompanying public repository fixes folds 1–8/9/10 as train/validation/test and exposes prediction artifacts and bootstrapped evaluation. Reproducing one published baseline within its confidence interval is therefore the first engineering gate.

### 4.2 The 2026 overlap that changes this project

Rehman and Nazir's July 2026 medRxiv preprint trained a 1D ResNet, CNN–BiLSTM, and CNN–BiLSTM–transformer on the same five-superclass task. All models were near 0.92 macro-AUROC; the 1D ResNet was marginally best at 0.9241. They reported lower performance for female patients and for patients aged 80 or older.

Therefore the following are **not sufficient novelty by themselves**:

- comparing a ResNet with a transformer;
- reporting approximately 0.92 macro-AUROC;
- showing only age- and sex-stratified AUROC;
- claiming that a larger architecture does not necessarily win.

### 4.3 Defensible differentiation

The differentiating research package should combine:

1. **Calibration:** compare raw sigmoid outputs with post-hoc calibrators fitted without test leakage.
2. **Selective prediction:** measure risk–coverage behavior and preregister gates that defer uncertain or poor-quality cases.
3. **Subgroup-aware abstention:** test whether a gate improves accepted-case performance by achieving unequal or unsafe coverage across age, sex, device, signal-quality, or pathology groups.
4. **Robustness:** evaluate controlled baseline wander, powerline/noise injection, amplitude scaling, time shift, lead dropout, and lead permutation checks.
5. **Explanation faithfulness:** compare at least Grad-CAM and Integrated Gradients/occlusion, then run parameter-randomization, label-randomization, deletion, stability, and lead-ablation checks.
6. **Reproducibility:** preserve exact folds, version every split/configuration, save test predictions, and report patient-cluster bootstrap intervals.

The work becomes an audit of two model families as decision-support components, not a model leaderboard chase.

## 5. Preregistered research questions and hypotheses

### RQ1 — discrimination

Do a parameter-matched 1D ResNet and patch-based transformer differ meaningfully in test macro-AUROC?

- **H1:** Both will reach approximately 0.91–0.93 macro-AUROC at 100 Hz, and their patient-bootstrap 95% confidence intervals will substantially overlap.
- **Decision rule:** do not call one architecture better based only on a higher point estimate. Report the paired bootstrap difference and interval.

### RQ2 — calibration

Which architecture produces better probability estimates before and after post-hoc calibration?

- **H2:** raw neural probabilities will be measurably miscalibrated; regularized classwise sigmoid scaling will improve macro Brier score and log loss without changing AUROC materially.
- **Decision rule:** a calibrator is retained only if it improves a proper scoring rule on the calibration fold and does not create severe per-label degradation.

### RQ3 — abstention

Can calibrated uncertainty identify cases for which deferral reduces error?

- **H3:** deep-ensemble uncertainty or calibrated predictive entropy will dominate maximum-probability confidence on area under the risk–coverage curve.
- **Decision rule:** report accepted-case performance at fixed target coverages of 95%, 90%, 80%, and 70%, plus realized coverage on the untouched test fold.

### RQ4 — subgroup coverage

Does abstention improve overall selective risk while disproportionately rejecting particular subgroups?

- **H4:** oldest-patient and low-signal-quality groups will have both lower discrimination and lower gate coverage.
- **Decision rule:** report subgroup sample size, prevalence, AUROC/AUPRC, Brier score, sensitivity, and coverage with uncertainty. Avoid ranking small subgroups without adequate intervals.

### RQ5 — robustness and explanations

Are explanations stable and faithful under clinically plausible perturbations, and does apparent visual plausibility predict faithfulness?

- **H5:** visually smooth attributions will not always pass randomization and deletion tests. Grad-CAM may localize useful regions for the CNN, while occlusion or Integrated Gradients will provide a more architecture-comparable audit.
- **Decision rule:** explanations are described as model attributions, never as causal evidence or proof that a waveform segment is clinically diagnostic.

## 6. Leakage-resistant experimental protocol

### 6.1 Two reporting tracks

**Track A — published benchmark reproduction**

- PTB-XL 1.0.3, 100 Hz.
- Train: folds 1–8.
- Validation/early stopping: fold 9.
- Test: fold 10, evaluated once after the pipeline is frozen.
- Purpose: direct comparability with the official benchmark.

**Track B — trustworthy probability protocol**

- Development training: folds 1–7.
- Model/hyperparameter selection and epoch-budget selection: fold 8.
- Final refit: folds 1–8 using the frozen configuration and training budget.
- Calibration, threshold selection, and abstention cutoff selection: fold 9 only.
- Final discrimination, calibration, selective-risk, subgroup, robustness, and attribution reports: fold 10 only.

Track B prevents the same cases from simultaneously driving architecture choice and probability calibration. It also reserves fold 10 for a genuinely final report. All operations must assert disjoint `patient_id` sets.

### 6.2 Test-set governance

- Generate a hash of the frozen experiment configuration before final testing.
- Keep fold-10 labels inaccessible to training and calibration code paths.
- Permit one formal final evaluation per preregistered model/configuration.
- Save logits, calibrated probabilities, targets, patient IDs, labels, and configuration hash.
- Any change motivated by test results creates a new explicitly labeled exploratory experiment, not a replacement confirmatory result.

### 6.3 Data and preprocessing

- Begin at 100 Hz for benchmark parity: shape `[12, 1000]` per record.
- Run 500 Hz only as a preregistered ablation after the 100 Hz pipeline is frozen.
- Preserve physical amplitude information. Default normalization uses training-fold, per-lead statistics; do not normalize each ECG independently without an ablation because amplitude carries diagnostic information.
- Perform no test-time filtering based on label knowledge.
- Record every transform in the run configuration.
- Check units, lead order, duration, finite values, patient overlap, target construction, and class counts before training.
- Treat lead order as a contract and add a failing unit test for permutations.

### 6.4 Target construction

- Parse `scp_codes` safely as data, not executable text.
- Join with `scp_statements.csv` and retain statements flagged as diagnostic.
- Map positives to the five `diagnostic_class` values.
- Use multi-hot targets in a fixed order: `[NORM, MI, STTC, CD, HYP]`.
- Include ECGs that have at least one superclass label, matching the selected benchmark protocol.
- Persist a label-manifest file with dataset version, mappings, counts, and hashes.

### 6.5 Loss and imbalance

Start with unweighted `BCEWithLogitsLoss` as the calibration-friendly reference. Compare it with a bounded positive-class weighting strategy. Focal loss is optional and should not be the default because it changes probability behavior and complicates calibration.

For every loss, separate three questions:

- ranking quality: AUROC/AUPRC;
- thresholded decisions: F1, sensitivity, specificity;
- probability quality: Brier score, log loss, calibration curves.

Do not assume that the loss with the best AUROC yields the best calibrated probabilities.

## 7. Model plan

### 7.1 Baselines

1. **Prevalence predictor:** emits training prevalence for each label. This anchors Brier score and log loss.
2. **Small 1D CNN:** catches pipeline errors and provides a low-capacity reference.
3. **Published-style 1D ResNet:** reproduction anchor, ideally matching `resnet1d_wang` or XResNet behavior.

### 7.2 Primary 1D ResNet

Recommended initial search space:

- four residual stages;
- channels approximately `[64, 128, 256, 512]` with a smaller variant for parameter matching;
- kernel sizes in `{5, 7, 15}`;
- strided convolution or pooling between stages;
- GroupNorm or BatchNorm as a declared ablation;
- global average pooling and five sigmoid logits;
- stochastic depth/dropout only if validation evidence supports it.

Grad-CAM is computed from the final temporal convolution block. Integrated Gradients and temporal occlusion provide architecture-independent comparators.

### 7.3 Primary ECG transformer

A pure sample-token transformer is wasteful even at 100 Hz. Use a convolutional stem or non-overlapping temporal patches:

- input `[B, 12, 1000]`;
- patch width initially 20 samples (200 ms at 100 Hz), giving 50 temporal tokens;
- patch projection that jointly observes all 12 leads, plus an ablation using lead-aware tokens;
- learned class token or attention pooling;
- 4–8 encoder layers, 4–8 heads, embedding dimension 128–256;
- positional encoding, pre-norm blocks, GELU MLP, and dropout;
- five independent sigmoid logits.

Patch width is clinically consequential: coarse patches can hide short waveform events, while very fine patches increase attention cost. Search `{10, 20, 25, 50}` samples under a fixed budget.

### 7.4 Fair comparison rules

- Match parameter counts within roughly ±15%, or report both a matched-capacity comparison and each model's best practical configuration.
- Use the same input resolution, folds, augmentations, optimizer budget, evaluation code, and tuning budget.
- Tune each family separately within the same number of trials.
- Report parameters, peak VRAM, examples/second, training time, inference latency, and energy proxy if available.
- Use at least three seeds for the final two configurations. Prefer paired patient bootstrap intervals over declaring a winner from seed means alone.

## 8. Training and sweep strategy

### 8.1 Reference settings

- Optimizer: AdamW.
- Mixed precision: BF16 if verified stable on the installed PyTorch build; otherwise FP16 with gradient scaling.
- Schedule: cosine decay with short warmup.
- Epoch budget: determine on fold 8, then freeze for final refit.
- Gradient clipping: transformer only if needed.
- Checkpoint criterion: validation macro-AUROC, with proper scores logged concurrently.
- Seeds and deterministic settings: record Python, NumPy, and PyTorch seeds; enable deterministic algorithms for the reproducibility run and document any unsupported operation.

### 8.2 Sweep order

1. Overfit a 64-record subset to validate model/data wiring.
2. Train the small CNN for a pipeline sanity result.
3. Reproduce the ResNet benchmark.
4. Tune ResNet and transformer with a small, equal-budget search.
5. Freeze architecture and optimization choices.
6. Run three final seeds.
7. Fit calibration and gating on fold 9.
8. Run fold-10 evaluation once.
9. Run robustness and explanation audits on frozen models.

Suggested initial tuning budget: 12–20 trials per primary family using a pruning scheduler. A huge sweep is not scientifically useful if the test protocol, uncertainty method, or attribution validation remains weak.

## 9. Evaluation contract

### 9.1 Discrimination

Primary:

- macro-AUROC across the five labels;
- per-label AUROC;
- patient-cluster bootstrap 95% confidence intervals;
- paired bootstrap interval for model differences.

Secondary:

- macro- and per-label AUPRC;
- sample-centric Fmax for benchmark compatibility;
- sensitivity, specificity, precision, and F1 at thresholds chosen only on fold 9;
- label prevalence alongside every AUPRC.

Accuracy is not a headline metric for this imbalanced, multi-label task.

### 9.2 Calibration

Report before and after calibration:

- macro and per-label Brier score;
- binary cross-entropy/log loss;
- equal-mass reliability diagrams;
- adaptive or classwise ECE with bin counts and estimator definition;
- calibration slope and intercept per label where sample size permits;
- uncertainty intervals from patient-level bootstrap.

ECE alone is insufficient because its value depends strongly on binning. Proper scoring rules and diagrams are required.

Calibrators fitted on fold 9:

1. global temperature scaling as the low-variance baseline;
2. regularized classwise sigmoid scaling, `sigmoid(z_k / T_k + b_k)`;
3. isotonic regression only as an exploratory comparator because HYP and rare probability regions may be data-limited.

### 9.3 Selective prediction and abstention

Candidate uncertainty scores:

- mean label entropy after calibration;
- smallest distance to any selected decision threshold;
- deep-ensemble predictive entropy;
- ensemble mutual information or variance;
- signal-quality score;
- combined uncertainty plus signal-quality gate.

Select all gates and cutoffs on fold 9. On fold 10 report:

- realized coverage;
- selective macro-F1 and labelwise error;
- selective Brier score and log loss;
- risk–coverage curve and area under it;
- error-detection AUROC/AUPRC for the uncertainty score;
- accepted versus rejected prevalence and subgroup composition;
- subgroup coverage gaps.

For a multi-label record, explicitly define the sample loss used by the risk–coverage curve. Recommended primary risk is mean per-label binary log loss; thresholded Hamming loss is secondary. A record-level “correct/incorrect” flag is too coarse.

### 9.4 Subgroup and hidden-stratification audit

Predeclare:

- sex;
- age bands `<40`, `40–59`, `60–79`, and `80+`, with sensitivity analyses for alternate bins;
- device and site when counts permit;
- signal-quality annotations;
- single-label versus multi-label records;
- presence/absence of NORM co-label;
- diagnostic subclasses nested under each superclass.

For each subgroup report sample and patient counts, prevalence, AUROC, AUPRC, Brier score, sensitivity, specificity, and gate coverage. Use bootstrap intervals and state when a subgroup is too small for a stable claim. This is an audit of observed disparities, not proof of fairness or discrimination.

### 9.5 Robustness matrix

Apply perturbations to frozen fold-10 inputs without retraining:

- baseline wander at multiple amplitudes/frequencies;
- powerline or broadband noise at controlled signal-to-noise ratios;
- amplitude scaling and DC offsets;
- small temporal shifts;
- random contiguous masking;
- single-lead dropout and clinically grouped lead dropout;
- explicit lead permutation as a negative control;
- 100 Hz versus 500 Hz as a separate trained-model ablation.

For each severity, report changes in macro-AUROC, Brier score, calibration error, gate coverage, and uncertainty. A useful uncertainty method should generally become less confident as corruption worsens.

## 10. Explanation plan: visualization plus validation

### 10.1 Methods

- ResNet: 1D Grad-CAM from the last temporal convolution layer.
- Both models: Integrated Gradients and temporal/lead occlusion.
- Transformer: attention rollout only as an auxiliary visualization, never as the sole explanation.

Render signed and absolute attribution separately over the 12 traces. Preserve lead names, voltage scale, and time axis. Allow the user to select the target label because multi-label attribution is class-specific.

### 10.2 Faithfulness checks

1. **Model-parameter randomization:** attribution should materially change as learned layers are randomized.
2. **Label randomization:** a model trained on shuffled labels should not retain the same structured explanations.
3. **Deletion/insertion:** masking the most-attributed time windows should change the target logit more than masking random windows of equal size.
4. **Lead ablation:** attribution concentration on a lead should predict sensitivity to ablating that lead better than chance.
5. **Stability:** small non-semantic perturbations should not cause arbitrary attribution changes.
6. **Cross-method agreement:** report agreement, but do not treat agreement itself as correctness.

Published ECG attribution evaluation found Grad-CAM strong on localization-style metrics, while broader saliency research shows visually appealing maps can fail model/data randomization tests. Both lessons belong in the implementation.

## 11. Demo specification

The demo is a transparent research viewer, not an automated diagnostic tool.

### Inputs

- select a known PTB-XL record; or
- upload a compatible WFDB header/data pair;
- optional later support for a documented CSV schema with exactly 12 named leads.

### Outputs

- 12-lead waveform plot with standard lead names and units;
- raw and calibrated probabilities for all five superclasses;
- per-label threshold and predicted state;
- uncertainty score, gate threshold, and `predict`/`defer` status;
- target-selectable attribution overlay;
- data-quality warnings and detected schema problems;
- model card, dataset scope, and conspicuous research-only disclaimer.

### Safety language

Use “model output,” “class probability,” and “defer.” Do not use “diagnosis,” “patient is normal,” “safe,” or treatment recommendations. NORM is a dataset label, not proof that a patient has no clinically important condition.

## 12. Local infrastructure audit

Audit performed August 8, 2026.

| Component | Observed system | Assessment |
|---|---|---|
| Computer | ASUS ROG Strix G16 G614PR | Suitable mobile workstation |
| CPU | AMD Ryzen 9 8940HX, 16 cores / 32 threads | Excellent for WFDB loading, preprocessing, and parallel data workers |
| RAM | 31.21 GB installed; 13.85 GB free during audit | Sufficient for 100 Hz and 500 Hz streaming; avoid unnecessary full-data copies |
| GPU | NVIDIA GeForce RTX 5070 Ti Laptop GPU | Excellent for both proposed model families |
| VRAM | 12,227 MiB total; about 9,986 MiB free during audit | Ample; tune batch size empirically |
| GPU power | 105 W limit observed | Strong, but below some maximum-TGP variants; plug in and use performance mode for benchmarks |
| Compute capability | 12.0 (`sm_120`, Blackwell) | Requires Blackwell-capable framework binaries |
| Driver | 596.49; `nvidia-smi` reports CUDA capability up to 13.2 | Modern and compatible with older bundled CUDA runtimes |
| Local CUDA toolkit | 12.4 (`nvcc` 12.4.99) | Too old to compile native Blackwell kernels; not needed for standard PyTorch wheels |
| Storage | 952.96 GB total, 252.03 GB free | More than adequate for the 3 GB dataset, environments, runs, and checkpoints |
| Python | Not installed/discoverable | Blocking setup item |
| `uv` | 0.11.24 installed | Preferred environment/package manager |
| Git | 2.53.0; Git LFS 3.7.1 | Ready |
| WSL | Not installed | Not required; native Windows is sufficient |
| Docker | Not installed | Not required for the initial project |

### 12.1 Critical CUDA conclusion

`nvidia-smi`'s “CUDA 13.2” is the maximum runtime level supported by the driver; it is not the version used automatically by PyTorch. The installed CUDA 12.4 toolkit predates Blackwell native compilation support. NVIDIA documents Blackwell support beginning with CUDA 12.8, and PyTorch introduced Blackwell plus CUDA 12.8 wheels in PyTorch 2.7.

Recommended environment:

- native Windows;
- a `uv`-managed Python 3.12 environment inside this project;
- current stable PyTorch using an official Windows CUDA wheel that includes Blackwell support (CUDA 12.8 or newer, selected from PyTorch's current installer);
- no manual dependence on the system CUDA 12.4 toolkit;
- verify `torch.cuda.is_available()`, device name, compute capability, BF16 support, and an actual forward/backward step before downloading the data.

Do not downgrade the working driver. Updating or removing the 12.4 toolkit is optional unless custom CUDA extensions are later compiled.

### 12.2 Capacity expectations

At 100 Hz, one float32 ECG is about 48 KB (`12 × 1000 × 4` bytes); the full signal tensor is approximately 1.05 GB before Python/container overhead. At 500 Hz it is about 5.23 GB. Both fit in system RAM, but memory-mapped or streaming WFDB loading is cleaner and reduces copies.

Twelve GB of VRAM is far beyond the minimum for these models. Practical batch sizes depend on architecture, patching, precision, and dataloader behavior; an initial target is 128–512 records at 100 Hz and 32–128 at 500 Hz. These are starting ranges, not promises. The first benchmark script must measure peak allocated VRAM, throughput, and thermal throttling.

The GPU's main value is iteration speed. However, a research-grade set of equal-budget sweeps, three seeds, ensembles, corruption tests, bootstraps, and attributions can still consume many GPU-hours. “Two weekends” is realistic for a strong MVP and initial results, not for every confirmatory audit in this blueprint.

## 13. Proposed repository architecture

```text
.
├── README.md
├── pyproject.toml
├── uv.lock
├── configs/
│   ├── data/
│   ├── model/
│   ├── experiment/
│   └── sweep/
├── data/
│   ├── raw/                 # gitignored PTB-XL 1.0.3
│   ├── interim/
│   └── manifests/
├── docs/
│   ├── RESEARCH_BLUEPRINT.md
│   ├── MODEL_CARD.md
│   └── DATA_CARD.md
├── src/ecg_trust/
│   ├── data/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   ├── calibration/
│   ├── selective/
│   ├── explain/
│   └── demo/
├── tests/
│   ├── unit/
│   └── integration/
├── scripts/
│   ├── verify_system.py
│   ├── download_ptbxl.py
│   ├── build_manifest.py
│   ├── train.py
│   ├── evaluate.py
│   └── launch_demo.py
├── artifacts/               # gitignored models/predictions
└── reports/                 # committed summary tables/figures
```

Configuration should be declarative and every run should save the resolved config, Git commit, environment versions, random seed, dataset manifest hash, fold IDs, metrics, and prediction files.

## 14. Execution roadmap

### Phase 0 — environment and data integrity (half day)

- Create Python environment and install a Blackwell-capable PyTorch wheel.
- Run CUDA forward/backward and mixed-precision smoke tests.
- Download PTB-XL 1.0.3 and verify checksums.
- Build data/label manifest and patient-leakage tests.
- Plot representative records and label distributions.

**Exit gate:** device test passes; exact record/patient counts match 1.0.3; folds have disjoint patient IDs; lead order and targets are verified.

### Phase 1 — reproducible baseline (days 1–2)

- Prevalence and small-CNN baselines.
- Published-style 1D ResNet at 100 Hz.
- Save fold-9 predictions and reproduce macro-AUROC near the published interval.

**Exit gate:** pipeline result is credible before transformer work begins.

### Phase 2 — fair architecture study (days 3–5)

- Implement transformer and parameter-matched ResNet.
- Equal-budget sweeps on development folds.
- Freeze two primary configurations and run three seeds.

**Exit gate:** training curves, speed, VRAM, and validation metrics are reproducible.

### Phase 3 — calibration and abstention (days 6–7)

- Final refit on folds 1–8.
- Fit calibrators and selection gates on fold 9.
- Freeze configuration and produce formal fold-10 predictions.

**Exit gate:** test artifacts are immutable and all primary metrics have patient-bootstrap intervals.

### Phase 4 — audit suite (additional 1–2 weeks for research-grade work)

- Subgroup/hidden-stratification analysis.
- Corruption and lead-drop robustness.
- Grad-CAM, Integrated Gradients, and occlusion.
- Attribution randomization, deletion, ablation, and stability tests.

### Phase 5 — demo and research package (2–3 days)

- Build local viewer.
- Add model/data cards, limitations, methods, and reproducibility commands.
- Export publication-quality tables and figures.

## 15. Acceptance criteria

The project is complete only when:

- PTB-XL 1.0.3 counts, checksums, folds, labels, and patient disjointness are automatically verified.
- The baseline reproduces a literature-comparable 100 Hz result or the discrepancy is explained.
- ResNet and transformer are compared under an equal and documented tuning budget.
- Final results include three seeds plus paired patient-bootstrap confidence intervals.
- Calibration uses no fold-10 information and reports Brier, log loss, reliability diagrams, and ECE definition.
- Abstention reports full risk–coverage curves and subgroup coverage, not only a hand-picked operating point.
- Robustness tests demonstrate how both error and uncertainty change with perturbation severity.
- Attribution maps pass documented sanity/faithfulness tests or failures are reported plainly.
- The demo rejects malformed inputs, preserves units/lead order, and displays a research-only limitation notice.
- No clinical-safety, causal-explanation, or real-world-generalization claim exceeds the evidence.

## 16. Main risks and stop rules

| Risk | Consequence | Mitigation / stop rule |
|---|---|---|
| Blackwell-incompatible PyTorch build | GPU errors or CPU fallback | Require verified CUDA backward pass before any training |
| Leakage through patient or fold handling | Inflated results | Automated patient-disjoint assertions; exact official folds |
| Repeated test inspection | Invalid confirmatory claim | Freeze config and hash before fold-10 evaluation |
| Miscasting task as single-label | Wrong loss and metrics | Fixed multi-hot target tests and sigmoid outputs |
| Per-record normalization erases amplitude | Misleading comparison | Training-fold statistics; normalization ablation |
| Class weighting harms calibration | Confident but biased probabilities | Compare proper scores and post-hoc calibration |
| Gate hides poor subgroup performance | Unsafe selective system | Always report subgroup coverage and rejected-case composition |
| Attractive but unfaithful saliency | False interpretability claim | Randomization, deletion, stability, and ablation checks |
| Transformer sweep consumes schedule | Demo never ships | Stop after equal trial budget; prioritize audit over marginal AUROC |
| “Clinical-grade” framing | Unsupported high-stakes claim | Research-only interface and explicit external-validation limits |

## 17. Suggested paper/report structure

1. Introduction: discrimination is not enough for ECG decision support.
2. Related work: PTB-XL benchmarks, 2026 demographic audit, calibration, selective prediction, and ECG attribution.
3. Data and label construction.
4. Leakage-resistant experimental design.
5. Architecture and compute comparison.
6. Discrimination and calibration results.
7. Selective-risk and subgroup-coverage results.
8. Robustness and attribution-faithfulness results.
9. Limitations and clinical-scope statement.
10. Reproducibility checklist and artifact links.

## 18. Primary sources and implementation references

### Dataset and benchmark

- [PTB-XL 1.0.3 on PhysioNet](https://physionet.org/content/ptb-xl/1.0.3/) — canonical current dataset page, files, license, folds, release notes, and counts.
- [PTB-XL: A Large Publicly Available ECG Dataset](https://doi.org/10.1038/s41597-020-0495-6) — original Scientific Data paper.
- [Deep Learning for ECG Analysis: Benchmarks and Insights from PTB-XL](https://arxiv.org/abs/2004.13701) — benchmark tasks, metrics, model results, hidden stratification, uncertainty, interpretability, and sampling/split studies.
- [Official PTB-XL benchmarking repository](https://github.com/helme/ecg_ptbxl_benchmarking) — reference code, leaderboard, folds, and output conventions.

### Closely overlapping current work

- [Accurate overall, uneven by patient: a benchmark and demographic audit of deep learning for 12 lead ECG classification on PTB-XL](https://doi.org/10.64898/2026.07.09.26357670) — July 2026 preprint motivating the shift beyond a basic architecture/fairness comparison.

### Calibration, selective prediction, and explanations

- [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599) — temperature scaling baseline.
- [Measuring Calibration in Deep Learning](https://arxiv.org/abs/1904.01685) — pitfalls and design choices in calibration measurement.
- [Selective Classification for Deep Neural Networks](https://papers.neurips.cc/paper_files/paper/2017/file/4a8423d5e91fda00bb7e46540e2b0cf1-Paper.pdf) — risk–coverage framing.
- [Evaluating Feature Attribution Methods for Electrocardiogram](https://arxiv.org/abs/2211.12702) — ECG-specific attribution evaluation.
- [Sanity Checks for Saliency Maps](https://arxiv.org/abs/1810.03292) — model- and data-randomization requirements.
- [Captum model-understanding tutorial](https://docs.pytorch.org/tutorials/beginner/introyt/captumyt.html) — PyTorch implementation path for Integrated Gradients and related methods.

### Local GPU compatibility and reproducibility

- [PyTorch 2.7 release: Blackwell and CUDA 12.8 support](https://pytorch.org/blog/pytorch-2-7/).
- [PyTorch official local installation selector](https://pytorch.org/get-started/locally/).
- [NVIDIA Blackwell compatibility guide](https://docs.nvidia.com/cuda/archive/12.8.2/blackwell-compatibility-guide/index.html).
- [NVIDIA CUDA driver/toolkit compatibility matrix](https://docs.nvidia.com/datacenter/tesla/drivers/cuda-toolkit-driver-and-architecture-matrix.html).
- [PyTorch reproducibility notes](https://docs.pytorch.org/docs/stable/notes/randomness.html).

## 19. Immediate next action

Build Phase 0 only: create the local `uv` environment with a current Blackwell-capable PyTorch build, add the system verification script and tests, download PTB-XL 1.0.3, and prove the folds/labels are correct before implementing either primary model.
