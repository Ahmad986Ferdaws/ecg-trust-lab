# Model card template: PTB-XL trustworthy superclass classifier

> **Template, not a result.** Copy this file to a versioned model-card path and
> replace bracketed fields only from integrity-checked artifacts. Leave a field
> as `Not evaluated` when evidence does not exist. Never convert a fold-8
> development metric into a fold-10 result.

## 1. Card and evidence status

| Field | Value |
|---|---|
| Model-card version | `[MODEL_CARD_VERSION]` |
| Model ID | `[MODEL_ID]` |
| Architecture | `[resnet1d / ecg_transformer / comparison card]` |
| Release date | `[YYYY-MM-DD]` |
| Owners | `[NAMES_OR_TEAM]` |
| Repository revision | `[GIT_COMMIT; include dirty state]` |
| Protocol ID | `ptbxl-superclass-trust-v1` |
| Protocol hash | `sha256:ebfdb588615bfa22eedc6d936d7b0155a33702878cbe0258ebb84aaa88567e09` |
| Dataset | `PTB-XL 1.0.3, canonical 100 Hz five-superclass manifest` |
| Evidence status | `[DEVELOPMENT ONLY / CALIBRATED, NOT TESTED / FINAL TEST COMPLETE / EXPLORATORY]` |
| Preregistered seeds | `[SEED_LIST]` |
| Completed final seeds | `[SEED_LIST_OR_NONE]` |
| Final-report hashes | `[ONE_PER_MODEL_AND_SEED_OR_NOT_EVALUATED]` |

### Evidence-status declaration

`[Write one plain-language paragraph stating exactly which folds have been
accessed. If only run_metadata.json from development exists, say that evidence
is limited to model-selection fold 8. If fold 9 has been used, say it was used
for calibration and decision fitting, not final generalization. State whether
fold 10 has ever been opened and why.]`

## 2. Summary

`[Describe the model as a research classifier that emits five independent
diagnostic-superclass scores from a ten-second, 12-lead ECG. State the frozen
architecture, seed set, calibration method, abstention method, and evidence
status in two or three sentences. Do not use diagnosis, clinical-grade, safe,
or deployment-ready language.]`

## 3. Intended use

This model is intended for:

- reproducible research on PTB-XL diagnostic-superclass classification;
- comparison of a capacity-matched 1D ResNet and ECG transformer;
- study of discrimination, calibration, selective abstention, subgroup
  behavior, robustness, and model attribution methods;
- a research-only viewer for compatible ECGs, with probabilities, a defer
  decision, and target-specific attribution overlays.

This model is **not** intended for:

- diagnosis, treatment, triage, emergency decisions, or patient management;
- use as a medical device or autonomous clinical decision maker;
- declaring a patient normal because the NORM score is high;
- use on incomplete, nonstandard, differently sampled, or unknown-unit ECGs;
- claims of external, prospective, multi-site, or present-day clinical
  generalization;
- causal interpretation of saliency or attribution maps.

Required interface notice:

> Research-only prototype. It is not a medical device and must not be used for
> diagnosis, treatment, triage, or emergency decisions.

## 4. Task and output contract

| Field | Frozen value |
|---|---|
| Task | Multi-label diagnostic-superclass classification |
| Input | Complete ten-second, 100 Hz, 12-lead ECG in physical millivolts |
| Tensor shape | `[12, 1000]` before batching |
| Lead order | `I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6` |
| Label/output order | `NORM, MI, STTC, CD, HYP` |
| Raw output | Five logits; independent sigmoid probabilities, not softmax |
| Calibration | One global temperature fitted on fold 9 |
| Decisions | Five per-label thresholds optimized for F1 on fold 9 |
| Abstention | Mean normalized binary entropy with fold-9 coverage cutoffs |
| Default coverage targets | `1.0, 0.9, 0.8, 0.7, 0.5` |

`NORM` is a PTB-XL target label, not proof that no clinically important
condition is present. Multiple labels may be positive for one ECG.

## 5. Data and split governance

The source release contains 21,799 ECGs from 18,869 patients. The canonical
five-superclass manifest excludes 411 records with no mapped diagnostic
superclass and therefore contains 21,388 ECGs from 18,617 patients.

| Role | Folds | Use in this release |
|---|---:|---|
| Parameter training | 1-7 | `[CONFIRM_OR_EXPLAIN]` |
| Model/epoch selection | 8 | `[CONFIRM_OR_EXPLAIN]` |
| Frozen refit | 1-8 | `[CONFIRM_OR_EXPLAIN]` |
| Calibration/threshold/gate fitting | 9 | `[CONFIRM_OR_NOT_EVALUATED]` |
| One-time final report | 10 | `[PURPOSE_AND_DATE_OR_NOT_OPENED]` |

Patient IDs are constrained to one official fold. Normalization uses only folds
1-7 even when the selected model is refit on folds 1-8.

### Data provenance

| Artifact | SHA-256 |
|---|---|
| Manifest Parquet | `[MANIFEST_SHA256]` |
| Manifest summary | `[MANIFEST_SUMMARY_SHA256]` |
| Folds-1-7 normalization file | `[NORMALIZATION_FILE_SHA256]` |
| Normalization selected-row content | `[NORMALIZATION_MANIFEST_SHA256]` |
| Subgroup definition JSON | `[SUBGROUP_JSON_SHA256_OR_NOT_EVALUATED]` |

### Label prevalence in the source release

These counts overlap because the target is multi-label.

| Label | Positive ECGs |
|---|---:|
| NORM | 9,514 |
| MI | 5,469 |
| STTC | 5,235 |
| CD | 4,898 |
| HYP | 2,649 |

## 6. Capacity-matched architectures

Both checked primary configs select preset
`ptbxl_100hz_matched_capacity_v1`.

| Property | 1D ResNet | ECG transformer |
|---|---|---|
| Trainable parameters | 8,739,973 | 8,726,833 |
| Main representation | Temporal residual convolutions | Temporal patch tokens plus class token |
| Widths / embedding | Stages `64, 128, 256, 512` | Embedding dimension `320` |
| Depth | `2, 2, 2, 2` residual blocks | `7` transformer blocks |
| Temporal kernel / patch | Stem 15; block 7 | Patch 20, stride 20; 50 patches |
| Downsampling | Stem stride 2, max pool 2, stage stride 2 | Non-overlapping patch projection |
| Attention heads | Not applicable | 8 |
| MLP ratio | Not applicable | 4.0 |
| Feature dropout | Block 0.10; classifier 0.20 | 0.10; attention 0.05 |
| Lead front end | Joint 12-lead convolution | Depthwise per-lead kernel 7 |
| Readout | Global average pool | Class token |
| Output | Five logits | Five logits |

The transformer/ResNet parameter ratio is approximately `0.9984966`; the
absolute count gap is approximately `0.1503%`. The preset also carries a 15%
tolerance and fails on exact expected-count drift.

### Selected model provenance

| Model | Seed | Source config hash | Resolved config hash | Development best-checkpoint SHA-256 | Refit final-checkpoint SHA-256 |
|---|---:|---|---|---|---|
| ResNet | `[SEED]` | `[HASH]` | `[HASH]` | `[HASH]` | `[HASH_OR_NOT_REFIT]` |
| Transformer | `[SEED]` | `[HASH]` | `[HASH]` | `[HASH]` | `[HASH_OR_NOT_REFIT]` |

Add one row per seed. Do not summarize omitted or failed seeds away; list them
in the run-accounting section with a reason.

## 7. Training procedure

The checked matched configs use:

| Setting | Value |
|---|---|
| Loss | `BCEWithLogitsLoss` |
| Optimizer | AdamW |
| Batch size | 128 |
| Learning rate | 0.001 |
| Weight decay | 0.01 |
| Schedule | Five-epoch warmup then cosine decay to 0.01 of initial LR |
| Maximum development epochs | 50 |
| Gradient clipping | Norm 1.0 |
| Development selection | Fold-8 macro AUROC |
| Early stopping | Patience 10; minimum delta 0.0001 |
| Precision | BF16 autocast on supported CUDA hardware |
| Determinism | Seeded Python/NumPy/PyTorch/CUDA/loaders; deterministic algorithms enabled |
| Frozen refit | Folds 1-8 for selected best epoch + 1; no early stopping |

### Run accounting

| Architecture | Seed | Run name | Status | Inclusion decision | Reason if excluded |
|---|---:|---|---|---|---|
| `[MODEL]` | `[SEED]` | `[RUN_NAME]` | `[complete/failed/running]` | `[included/excluded]` | `[REASON]` |

The checked repository currently provides configs only for seed 2026.
Additional seeds require explicit config copies with unique run names. A
single seed must never be described as a multi-seed study.

## 8. Calibration and decision policy

Fill this section only from an integrity-checked fold-9 decision artifact.

| Model | Seed | Fold-9 prediction artifact SHA-256 | Decision artifact SHA-256 | Temperature | NLL before | NLL after | Fit status |
|---|---:|---|---|---:|---:|---:|---|
| `[MODEL]` | `[SEED]` | `[HASH]` | `[HASH]` | `[VALUE]` | `[VALUE_OR_NA]` | `[VALUE_OR_NA]` | `[STATUS]` |

### Per-label thresholds

| Model | Seed | NORM | MI | STTC | CD | HYP | Objective/source |
|---|---:|---:|---:|---:|---:|---:|---|
| `[MODEL]` | `[SEED]` | `[T]` | `[T]` | `[T]` | `[T]` | `[T]` | `F1, fold 9` |

### Frozen entropy gates on fold 9

| Model | Seed | Target coverage | Entropy cutoff | Realized calibration coverage | Selected / total |
|---|---:|---:|---:|---:|---:|
| `[MODEL]` | `[SEED]` | `[TARGET]` | `[CUTOFF]` | `[COVERAGE]` | `[N/TOTAL]` |

Temperature-fit NLL values describe fold-9 calibration fitting. They are not
final-test log-loss measurements. The current final-report artifact does not
emit final-set log loss.

## 9. Development results — fold 8 only

> **Development model-selection evidence; not a held-out test result.** These
> values may explain why a configuration and epoch budget were selected. They
> must not be used in the abstract or final table as test performance.

| Architecture | Seed | Best epoch (zero-based) | Frozen epoch count | Fold-8 macro AUROC | Run metadata hash/path |
|---|---:|---:|---:|---:|---|
| ResNet | `[SEED]` | `[EPOCH]` | `[EPOCH_PLUS_ONE]` | `[VALUE]` | `[PATH_OR_HASH]` |
| Transformer | `[SEED]` | `[EPOCH]` | `[EPOCH_PLUS_ONE]` | `[VALUE]` | `[PATH_OR_HASH]` |

If multiple development seeds exist, report all rows. A descriptive mean and
standard deviation may be added below, but keep `Fold 8 development` in the
column title.

| Architecture | Completed declared seeds | Fold-8 development macro AUROC, mean +/- SD |
|---|---:|---:|
| ResNet | `[N]` | `[MEAN +/- SD OR NOT ENOUGH SEEDS]` |
| Transformer | `[N]` | `[MEAN +/- SD OR NOT ENOUGH SEEDS]` |

## 10. Final results — fold 10 only

> Leave this entire section as `Not evaluated` until each row can be traced to
> an integrity-verified `ecg_trust.final_evaluation_report` with fold role
> `final_test`, fold `[10]`, matching frozen fold-9 decisions, and a valid
> `report_sha256`. Every report must also bind the exact final-evaluation
> specification and required protocol-deviation disclosure.

### Per-seed macro results

Report the point estimate and that same report's patient-cluster percentile
interval. Do not average interval endpoints across seeds.

| Architecture | Seed | Report SHA-256 | Samples / patients | Macro AUROC (95% patient CI) | Macro AP (95% patient CI) | Macro Brier (95% patient CI) | Macro ECE (95% patient CI) | Bootstrap status |
|---|---:|---|---:|---:|---:|---:|---:|---|
| ResNet | `[SEED]` | `[HASH]` | `[N/P]` | `[EST (LOW, HIGH)]` | `[EST (LOW, HIGH)]` | `[EST (LOW, HIGH)]` | `[EST (LOW, HIGH)]` | `[STATUS; VALID/INVALID]` |
| Transformer | `[SEED]` | `[HASH]` | `[N/P]` | `[EST (LOW, HIGH)]` | `[EST (LOW, HIGH)]` | `[EST (LOW, HIGH)]` | `[EST (LOW, HIGH)]` | `[STATUS; VALID/INVALID]` |

### Cross-seed descriptive summary

This table is permitted only after **all preregistered seeds** have a complete
final report. Mean +/- SD summarizes variability among seed point estimates;
it is not a replacement for patient-bootstrap uncertainty.

| Architecture | Completed / declared seeds | Macro AUROC, mean +/- SD | Macro AP, mean +/- SD | Macro Brier, mean +/- SD | Macro ECE, mean +/- SD |
|---|---:|---:|---:|---:|---:|
| ResNet | `[N/N]` | `[VALUE]` | `[VALUE]` | `[VALUE]` | `[VALUE]` |
| Transformer | `[N/N]` | `[VALUE]` | `[VALUE]` | `[VALUE]` | `[VALUE]` |

Populate this table only from the sealed `ecg_trust.final_architecture_aggregate`
artifacts created by the exact-six final batch, retaining their batch, source,
final-specification, and deviation-log hashes.

### Per-label final results

Create one table per architecture/seed or a clearly identified pooled display.
Never hide degenerate labels or invalid bootstrap replicates.

| Model / seed | Label | Positives / negatives | Prevalence | AUROC (95% patient CI) | AP (95% patient CI) | Brier (95% patient CI) | ECE (95% patient CI) | Status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `[MODEL/SEED]` | NORM | `[P/N]` | `[VALUE]` | `[VALUE]` | `[VALUE]` | `[VALUE]` | `[VALUE]` | `[STATUS]` |
| `[MODEL/SEED]` | MI | `[P/N]` | `[VALUE]` | `[VALUE]` | `[VALUE]` | `[VALUE]` | `[VALUE]` | `[STATUS]` |
| `[MODEL/SEED]` | STTC | `[P/N]` | `[VALUE]` | `[VALUE]` | `[VALUE]` | `[VALUE]` | `[VALUE]` | `[STATUS]` |
| `[MODEL/SEED]` | CD | `[P/N]` | `[VALUE]` | `[VALUE]` | `[VALUE]` | `[VALUE]` | `[VALUE]` | `[STATUS]` |
| `[MODEL/SEED]` | HYP | `[P/N]` | `[VALUE]` | `[VALUE]` | `[VALUE]` | `[VALUE]` | `[VALUE]` | `[STATUS]` |

### Architecture difference

Do not infer superiority from point estimates or overlapping/non-overlapping
single-model intervals. Align model artifacts on `ecg_id`, `patient_id`, fold,
targets, manifest, protocol, and label order, then use shared patient draws.

| Seed | Difference direction | Metric | Estimate (95% paired patient CI) | Resamples valid / invalid | Artifact/code provenance |
|---:|---|---|---:|---:|---|
| `[SEED]` | `Transformer minus ResNet` | Macro AUROC | `[VALUE]` | `[V/I]` | `[HASH_OR_NOT_EVALUATED]` |

Use the sealed per-seed paired-bootstrap artifacts and their manifest from the
exact-six final batch. The implemented comparison direction is
`ecg_transformer minus resnet1d`; do not reverse signs silently when presenting
the table.

## 11. Selective prediction

Fill from `selective_prediction` in each final report. These are realized
fold-10 values after applying fold-9 entropy cutoffs; do not retune the cutoffs
to hit prettier final coverage.

| Model / seed | Target coverage | Realized coverage | Selected / abstained | Hamming risk | Exact-match accuracy | Mean selected uncertainty |
|---|---:|---:|---:|---:|---:|---:|
| `[MODEL/SEED]` | `[TARGET]` | `[REALIZED]` | `[N/N]` | `[VALUE_OR_NA]` | `[VALUE_OR_NA]` | `[VALUE_OR_NA]` |

An accepted output can still be wrong. Abstention is not a safety guarantee.

## 12. Subgroup audit

Report every preregistered attribute/group, including small or empty selected
groups. Preserve the artifact's status such as
`small_group_descriptive_only` or `no_selected_samples`.

| Model / seed | Attribute | Group | Samples / patients | Status | Macro AUROC | Macro AP | Macro Brier | Coverage at target `[X]` | Hamming risk |
|---|---|---|---:|---|---:|---:|---:|---:|---:|
| `[MODEL/SEED]` | `[ATTRIBUTE]` | `[GROUP]` | `[N/P]` | `[STATUS]` | `[VALUE]` | `[VALUE]` | `[VALUE]` | `[VALUE]` | `[VALUE_OR_NA]` |

This is an observed-disparity audit on available PTB-XL metadata, not proof of
fairness or discrimination. PTB-XL does not support a comprehensive demographic
fairness claim.

## 13. Robustness and explanation evidence

The locked final batch does not generate robustness or
explanation-faithfulness artifacts. Do not mark these as complete based on a
waveform visualization alone.

### Robustness

| Audit | Severities | Frozen model/checkpoint | Result artifact | Status / main finding |
|---|---|---|---|---|
| Baseline wander | `[VALUES]` | `[HASH]` | `[PATH/HASH]` | `[NOT EVALUATED OR RESULT]` |
| Broadband noise | `[VALUES]` | `[HASH]` | `[PATH/HASH]` | `[NOT EVALUATED OR RESULT]` |
| Amplitude scale | `[VALUES]` | `[HASH]` | `[PATH/HASH]` | `[NOT EVALUATED OR RESULT]` |
| Time shift | `[VALUES]` | `[HASH]` | `[PATH/HASH]` | `[NOT EVALUATED OR RESULT]` |
| Lead dropout | `[VALUES]` | `[HASH]` | `[PATH/HASH]` | `[NOT EVALUATED OR RESULT]` |
| Lead permutation negative control | `[VALUES]` | `[HASH]` | `[PATH/HASH]` | `[NOT EVALUATED OR RESULT]` |

### Attributions

The implemented explanation methods are 1D Grad-CAM for the ResNet and
Integrated Gradients for both model families. Attributions show model
sensitivity, not physiological causality or clinician reasoning.

| Method | Architecture | Target labels | Faithfulness/sanity tests | Result artifact | Status |
|---|---|---|---|---|---|
| Grad-CAM | ResNet | `[LABELS]` | `[DELETION/RANDOMIZATION/STABILITY]` | `[PATH/HASH]` | `[NOT EVALUATED OR RESULT]` |
| Integrated Gradients | Both | `[LABELS]` | `[DELETION/RANDOMIZATION/STABILITY]` | `[PATH/HASH]` | `[NOT EVALUATED OR RESULT]` |

## 14. Compute and efficiency

Fill only from the synthetic benchmark JSON and completed run metadata. Keep
synthetic train-step throughput distinct from real dataset throughput.

| Architecture | Parameters | Device / precision | Batch | Synthetic samples/s | Median / p95 step ms | Peak allocated / reserved VRAM | Real training wall time |
|---|---:|---|---:|---:|---:|---:|---:|
| ResNet | 8,739,973 | `[DEVICE/BF16]` | `[B]` | `[VALUE]` | `[VALUE/VALUE]` | `[VALUE/VALUE]` | `[VALUE]` |
| Transformer | 8,726,833 | `[DEVICE/BF16]` | `[B]` | `[VALUE]` | `[VALUE/VALUE]` | `[VALUE/VALUE]` | `[VALUE]` |

There is no current minimum-throughput or maximum-VRAM quality threshold.

## 15. Limitations

- PTB-XL is a retrospective, historical, single-source cohort; this card does
  not establish external, prospective, contemporary, or multi-device
  generalization.
- Targets derive from clinical reports mapped to broad diagnostic
  superclasses. They are benchmark labels, not prospective adjudicated
  diagnoses.
- The five superclasses hide diagnostic subclasses and co-occurrence patterns;
  aggregate macro metrics can hide poor subgroup or subtype behavior.
- The system supports only complete ten-second, 100 Hz, canonical 12-lead
  signals in physical millivolts. It does not silently repair missing leads,
  duration, sampling rate, units, order, or non-finite values.
- Calibration uses one global temperature. Fixed-bin ECE depends on binning;
  the current report uses 15 bins and does not emit final-set log loss.
- Thresholds optimize labelwise F1 on fold 9. They are not clinical operating
  points and have no validated cost model.
- Entropy gates are selected on fold 9. Accepted cases can be wrong, and
  subgroup coverage can be unequal.
- Small subgroup estimates may be unstable or descriptive only. Available
  metadata cannot support a complete fairness analysis.
- Synthetic corruptions are controlled sensitivity tests, not evidence of
  real-world distribution-shift robustness.
- Attribution maps are model-specific sensitivity maps, not causal evidence or
  proof of clinically meaningful reasoning.
- The repository currently lacks cross-seed aggregation and paired-comparison
  CLIs. Any added analysis must preserve and document the same fold and hash
  contracts.
- Public fold 10 can become another validation set if inspected repeatedly.
  Any change motivated by final results must be labeled exploratory rather
  than replacing the confirmatory result.

## 16. Ethical and safety considerations

`[Describe how the research-only notice is displayed, how malformed inputs are
rejected, who can access uploaded ECG data, retention behavior, and whether any
protected health information is accepted. For a local demo, state that no
network transmission occurs only if that has been verified.]`

The interface should use `model output`, `class probability`, `accept`, and
`abstain/defer`. It should not say that the model diagnosed a condition, that a
patient is normal or safe, or that a highlighted segment is causally
diagnostic.

## 17. Reproduction references

- Protocol: `configs/protocol.yaml`
- Matched development configs: `configs/train_resnet_matched.yaml` and
  `configs/train_transformer_matched.yaml`
- Frozen-refit configs: `configs/refit_resnet_frozen.yaml` and
  `configs/refit_transformer_frozen.yaml`
- Full operational record: `docs/REPRODUCIBILITY.md`
- Research framing and literature context: `docs/RESEARCH_BLUEPRINT.md`

### Sign-off

| Review | Name | Date | Evidence checked |
|---|---|---|---|
| Data/protocol | `[NAME]` | `[DATE]` | `[HASHES_AND_FOLD_GATES]` |
| Modeling | `[NAME]` | `[DATE]` | `[CONFIGS_CHECKPOINTS_SEEDS]` |
| Calibration/final report | `[NAME]` | `[DATE]` | `[DECISION_AND_REPORT_HASHES]` |
| Safety/claims | `[NAME]` | `[DATE]` | `[SCOPE_AND_LIMITATIONS]` |
