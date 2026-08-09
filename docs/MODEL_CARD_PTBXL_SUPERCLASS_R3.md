# Model card: PTB-XL five-superclass comparison, r3

## Card and evidence status

| Field | Value |
|---|---|
| Model ID | `ptbxl_matched_equal_budget_v1__audit-r3` |
| Card version | `r3-final-2026-08-09` |
| Architectures | Capacity-matched 1D ResNet and ECG transformer |
| Operational demo default | ResNet, seed 2026, frozen fold-9 0.8-coverage gate |
| Dataset | PTB-XL 1.0.3, canonical 100 Hz five-superclass manifest |
| Evidence status | Sealed fold-10 comparison complete; post-evaluation descriptive audits complete |
| Preregistered/completed seeds | `2026, 2027, 2028` / `2026, 2027, 2028` for both architectures |
| Sealed-evaluation Git revision | `b0334730d8ab287a364f5978003dbe961770867c` |
| r3 analysis Git revision | `731753796e3c26d68bfff072583c5ea54851e8c1` |
| Protocol hash | `sha256:ebfdb588615bfa22eedc6d936d7b0155a33702878cbe0258ebb84aaa88567e09` |

Fold 9 was used only to fit one temperature, five F1 thresholds, and entropy
coverage cutoffs per frozen architecture/seed member. Fold 10 was opened on
August 9, 2026 UTC for the single preregistered exact-six model-evaluation batch.
Exact resumes recovered from operational hash-representation failures without
overwriting a prediction or repeating a scientific query. Post-evaluation r3
then analyzed only the frozen models, decisions, and immutable predictions.

[DEV-001](../reports/PROTOCOL_DEVIATIONS.md) must accompany every scientific
claim: before model evaluation, a bounded metadata search accidentally showed
some raw fold-10 rows containing SCP codes. No waveform, prediction, model
metric, aggregate label result, or exposed row value was used to make a
modeling or reporting choice. Strict operator-level outcome-label blindness
was nevertheless breached, so this is not a completely blind test.

## Summary

This research system maps a complete ten-second, 100 Hz, canonical 12-lead ECG
to five independent PTB-XL diagnostic-superclass probabilities: NORM, MI,
STTC, CD, and HYP. It compares nearly equal-capacity convolutional and
transformer models under the same development budget, applies policies fitted
only on fold 9, and reports a sealed fold-10 evaluation followed by descriptive
calibration, selective-prediction, subgroup, corruption, and attribution
audits.

The ResNet was stronger than the transformer on sealed macro AUROC, average
precision, and Brier score for all three paired seeds. The ECE difference was
not resolved, and important subgroup-coverage, corruption-sensitivity, and
explanation limitations prevent any clinical-safety or fairness claim.

## Intended use and prohibited use

Intended uses are reproducible PTB-XL benchmark research, controlled
architecture comparison, study of uncertainty and abstention behavior, and a
local research viewer for compatible ECGs. The viewer may display model-output
probabilities, an accept/defer state, and target-specific sensitivity overlays.

The system is not intended for diagnosis, treatment, triage, emergency
decisions, patient management, autonomous decision-making, or use as a medical
device. It must not be used to declare a patient normal or safe, and it has no
validated operating point or clinical cost model. It does not support missing,
duplicate, unnamed, or unexpected leads, unknown units, other sampling rates
or durations, or non-finite signals. Named WFDB leads may arrive in any order;
the loader validates their identities and canonicalizes them internally.

Required interface notice:

> Research-only prototype. It is not a medical device and must not be used for
> diagnosis, treatment, triage, or emergency decisions.

## Task, data, and split governance

| Field | Frozen value |
|---|---|
| Task | Multi-label diagnostic-superclass classification |
| Input | Ten seconds, 100 Hz, 12 leads, physical millivolts |
| Tensor shape | `[12, 1000]` before batching |
| Lead order | `I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6` |
| Output order | `NORM, MI, STTC, CD, HYP` |
| Output semantics | Five logits and independent sigmoids; not softmax |
| Source release | 21,799 ECGs from 18,869 patients |
| Canonical labeled manifest | 21,388 ECGs from 18,617 patients |
| Final cohort | 2,158 ECGs from 1,877 patients |

| Role | Folds | Actual use |
|---|---:|---|
| Parameter training | 1-7 | Model fitting and normalization statistics |
| Model/epoch selection | 8 | Equal-budget sweep and paired three-seed confirmation |
| Frozen refit | 1-8 | Fresh fixed-epoch refit, with folds-1-7 normalization retained |
| Calibration and decisions | 9 | Temperature, thresholds, and entropy gates only |
| One-time final evaluation | 10 | Sealed exact-six batch and preregistered reports |

The manifest Parquet hash is
`sha256:563a2b715cc6f6657b04c2f67d813fd7c30a696210740f97c55a070f157579a0`.
The folds-1-7 normalization file hash is
`sha256:4a6cb489098361d8221403c14871c242672c346975af3a07f731ceac97264363`,
and its selected-row provenance hash is
`55dd86001dee2006cb241ff8b4f3970d8fcbb1ae9ecd430f5dd61478673ce235`.
Patient IDs are isolated to one official fold.

`NORM` is a dataset target, not proof that no clinically important condition
is present. The task is multi-label, so more than one superclass may be
positive for one ECG.

## Models and training

| Property | 1D ResNet | ECG transformer |
|---|---:|---:|
| Trainable parameters | 8,739,973 | 8,726,833 |
| Core representation | Temporal residual convolutions | 20-sample temporal patches and class token |
| Development winner, fold-8 macro AUROC | 0.931852 | 0.923927 |
| Three-seed fold-8 mean | 0.932173 | 0.917785 |
| Frozen refit epochs | 12 | 22 |

The parameter-count gap is approximately 0.1503%. Both architectures used the
same task, split, normalization, BCE-with-logits objective, matched 12-candidate
search budget, fixed confirmation seeds, and deterministic BF16 CUDA training
policy. Fold-8 values explain the frozen choice; they are not held-out test
results. The architecture rule selected the ResNet as the development primary
while retaining and evaluating all six members.

## Calibration, thresholds, and abstention

Each member has one global temperature, five labelwise thresholds optimized
for F1, and mean-normalized-binary-entropy gates for target coverages 1.0, 0.9,
0.8, 0.7, and 0.5. All were fitted on fold 9 and applied unchanged on fold 10.
The demo default is ResNet seed 2026 at the nominal 0.8 gate because it was
frozen as an operational default, not because fold 10 selected it.

An accepted output can be wrong. The gate is a descriptive workload-risk
tradeoff, not a safety guarantee. At nominal 0.8 coverage, realized global
coverage was about 0.79-0.81; Hamming risk was about 0.09-0.10 for the ResNet
and about 0.11 for the transformer.

## Sealed fold-10 results

Values are descriptive means and sample standard deviations over the three
preregistered seed point estimates.

| Architecture | Macro AUROC | Macro AP | Brier | ECE |
|---|---:|---:|---:|---:|
| 1D ResNet | `0.921921 +/- 0.000913` | `0.810248 +/- 0.003327` | `0.085040 +/- 0.000718` | `0.022744 +/- 0.002984` |
| ECG transformer | `0.897420 +/- 0.003270` | `0.765270 +/- 0.007739` | `0.096807 +/- 0.001968` | `0.025646 +/- 0.001163` |

Paired differences are transformer minus ResNet, using shared
patient-cluster draws on aligned patients.

| Seed | Macro AUROC, 95% CI | Macro AP, 95% CI | Brier, 95% CI | ECE, 95% CI |
|---:|---:|---:|---:|---:|
| 2026 | `-0.0280 [-0.0342, -0.0222]` | `-0.0515 [-0.0638, -0.0390]` | `+0.0143 [+0.0113, +0.0174]` | `+0.0048 [-0.0006, +0.0108]` |
| 2027 | `-0.0244 [-0.0304, -0.0185]` | `-0.0467 [-0.0590, -0.0340]` | `+0.0116 [+0.0085, +0.0144]` | `-0.0001 [-0.0056, +0.0055]` |
| 2028 | `-0.0211 [-0.0272, -0.0150]` | `-0.0367 [-0.0483, -0.0247]` | `+0.0094 [+0.0063, +0.0124]` | `+0.0040 [-0.0024, +0.0080]` |

The paired AUROC and AP intervals favor the ResNet for every seed. Brier is
lower for the ResNet for every seed. All ECE intervals cross zero, so no ECE
advantage is established. Patient resampling also duplicates rows while the
fixed 15-bin ECE estimator repartitions them; ECE bootstrap intervals can be
biased and may not contain their point estimate. Treat them as sensitivity
summaries, not conventional inferential guarantees.

## Subgroup observations

Age and sex audits use the frozen metadata definitions and are descriptive.
At the nominal 0.8 gate, age-80+ coverage was approximately 0.60-0.65, compared
with approximately 0.93-0.95 below age 40. Age 80+ also had the lowest
age-band AUROC for all six members: about 0.881-0.885 for the ResNet and
0.834-0.857 for the transformer. These gaps make global coverage unsuitable as
a fairness claim. Small groups, broad age bins, limited metadata, internal
sampling, and absence of external validation constrain interpretation.

## Robustness audit

The r3 grid verified exact clean-logit equivalence for all six members and then
evaluated 240 non-clean cases, for 246 member-cases total. The most severe
observations were:

- macro-AUROC delta `-0.2939` and macro-Brier delta `+0.2213` for transformer
  seed 2027 under full lead reversal;
- Hamming-AURC delta `+0.3857` for ResNet seed 2026 under full lead reversal;
  and
- absolute frozen-gate coverage delta `0.5714` for ResNet seed 2027 after all
  precordial leads were dropped.

The matrix includes baseline wander, broadband noise, powerline interference,
DC offset, amplitude scaling, time shifts, contiguous masking, lead dropout,
and lead-permutation controls. These are controlled sensitivity tests, not
evidence of robustness to real deployment shifts.

## Explanation-control audit

The one-ECG-per-patient cohort contains 60 ECGs balanced across label/status
cells. Grad-CAM 1D was evaluated for the ResNet; Integrated Gradients and
temporal occlusion were evaluated for both architectures. Across 15 method
artifacts and 900 method-ECG evaluations, repeats were exact, mean 40 dB
stability cosine was `0.9929`, mean parameter-randomization cosine was
`-0.0067`, and mean random-minus-guided deletion AUC was `0.0495`; a positive
difference favors the guided ranking because lower guided deletion AUC is
better. Cross-method agreement was limited: mean cosine `0.5636` and mean
Spearman `0.2012`, with all 720 comparisons valid. The largest
FP32-attribution versus sealed-BF16 cohort raw-logit drift was `0.0913756`.

The cohort has no clinical localization ground truth. These methods show model
sensitivity under a signed correct-status score; they do not show causality,
physiological relevance, or clinician reasoning.

## Audit lineage and reproducibility anchors

The complete post-evaluation package is r3. r1 failed because decimal case IDs
collided during path construction. r2 regenerated all branch artifacts but
failed final loading because it compared an unordered JSON mapping in sequence
order. Both trees remain immutable, each successor binds the prior tree, and
r3 reused no derived artifact.

| Anchor | SHA-256 |
|---|---|
| Final-evaluation specification | `sha256:1f73c021a544ffeb119ffe8e490a16e32ec84247e30bce1ffd895fcffed6c762` |
| Final batch | `sha256:a4da85d5272b1634baaf953496c3d9efd8917777ad8de13b1b0c6dc754699e62` |
| Completed opening ledger | `sha256:3bb83554a08832212989ea8f3ea212f6af42c08460edbde9ba130065b1115a57` |
| r1 specification | `sha256:3a86035ecfaebafc0bfa8da4713fde90cb62345f151996f5ff79e8a68fa3043e` |
| r2 specification | `sha256:c0b07012265b5bd63ffb2c08438b64db7d74877e88050673e3192e354d9a0073` |
| r3 specification | `sha256:5727858f0c22b6311152749e0d9a3d20b3c14f4ee1c72ef9d1cf6e1943434200` |
| r3 probability audit | `sha256:e5f5261b2ff95dfe11596e544227effad5d5e98a4e547251fa5246d32ad26528` |
| r3 robustness manifest | `sha256:be9ff204f85f2ca11ed78ed93d4bf066029fd5c68f7c61fcfb0e1d9cfa6d906a` |
| r3 explanation manifest | `sha256:0803fe333b8f736d8c61babc2c50fe26194d3a75a7e06d123640ee2e1fc9733d` |
| r3 demo binding | `sha256:c8f7417875f646d1cabe4af4cb420813b2bde6cf6c201a0e1f1442ce133edb31` |
| r3 derived manifest | `sha256:fae6df30090ee59425a347034a7f4272cac5b799582a5742fff9c62b92a092f8` |

The rendered r3 report is
`runs/post_evaluation/ptbxl_matched_equal_budget_v1__audit-r3/reports/FINAL_RESULTS.md`
with file SHA-256
`c4363d89a43c59b7e7185a01b13a22377a90e35eb356e8a640bf7517fe721d0c`.
See the [post-evaluation run log](../reports/POST_EVALUATION_RUN_LOG.md) for the
full immutable lineage and the
[final-evaluation run log](../reports/FINAL_EVALUATION_RUN_LOG.md) for the
one-time opening and recovery history.

## Limitations and safety considerations

- PTB-XL is a retrospective, historical, single-source cohort; no external,
  prospective, contemporary, multi-site, or multi-device validation was done.
- Targets are broad report-derived benchmark labels, not prospective
  adjudicated diagnoses. Macro metrics can hide label, subtype, and subgroup
  failures.
- DEV-001 prevents a claim of complete operator blindness.
- Fixed-bin ECE is bin-dependent, and its patient-bootstrap behavior has the
  caveat described above.
- Fold-9 F1 thresholds are not clinical operating points. Entropy abstention
  does not make accepted cases safe and creates unequal subgroup coverage.
- Controlled corruptions do not represent the distribution or frequency of
  real acquisition failures.
- Attribution maps are sensitivity visualizations without clinical
  localization ground truth.
- Public fold 10 must not be reused for iterative tuning. Any change motivated
  by these results is exploratory and cannot replace this confirmatory record.
- The final August 9 repository gate passed all 434 tests, Ruff, and strict
  mypy. Pytest emitted one upstream Starlette/httpx deprecation warning.
- The r3 demo binding is materialized. Isolated Chromium verified the model
  health state, five label-free examples, ordinary inference, calibrated and
  raw probabilities, the frozen gate, all 12 rendered waveform leads, and a
  Grad-CAM overlay without console errors or failed requests; see the
  [post-evaluation run log](../reports/POST_EVALUATION_RUN_LOG.md).

Uploaded ECGs may contain sensitive health information. The viewer should be
run locally, should not receive unnecessary identifiers, and must not claim
that data remain local unless network behavior has been independently
verified. Malformed shape, duration, sampling-rate, lead-identity, unit, and
finite-value contracts must fail closed; valid named WFDB leads are reordered
to the canonical sequence.

## Reproduction references

- [Protocol](../configs/protocol.yaml)
- [Data card](DATA_CARD.md)
- [Environment record](ENVIRONMENT.md)
- [Reproducibility protocol](REPRODUCIBILITY.md)
- [Historical development results](../reports/DEVELOPMENT_RESULTS.md)
- [Final-evaluation run log](../reports/FINAL_EVALUATION_RUN_LOG.md)
- [Post-evaluation run log](../reports/POST_EVALUATION_RUN_LOG.md)
- [Required protocol deviations](../reports/PROTOCOL_DEVIATIONS.md)
