# Methods and results draft

Prepared on 2026-09-16 for issue #5, following the
[paper outline](PAPER_OUTLINE.md). This is a writing draft from the sealed
[model card](../docs/MODEL_CARD_PTBXL_SUPERCLASS_R3.md),
[public result snapshot](../reports/FINAL_RESULTS_PUBLIC.md), and linked
aggregate artifacts. It does not supersede either immutable study package.
No models were trained, metrics recalculated, or sealed cohorts reopened.

## Methods

### Study scope and evidence hierarchy

The study compared a capacity-matched 1D ResNet and ECG transformer for
multi-label PTB-XL diagnostic-superclass classification. The sealed fold-10
architecture comparison and preregistered within-seed paired comparisons are
confirmatory. The subsequent r3 calibration, selective-prediction, subgroup,
corruption, and attribution analyses are descriptive audits of frozen models
and decisions. A later SPH experiment is reported separately as exploratory
retrospective external transport. These evidence levels follow the
[public snapshot](../reports/FINAL_RESULTS_PUBLIC.md#interpretation-boundary).

### Data, inputs, and labels

PTB-XL 1.0.3 contains 21,799 ECGs from 18,869 patients. The project's canonical
labeled manifest contains 21,388 ECGs from 18,617 patients; the final fold-10
cohort contains 2,158 ECGs from 1,877 patients. Inputs are ten-second recordings
sampled at 100 Hz in physical millivolts, with tensor shape `[12, 1000]` and
lead order `I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6`.

Targets are `NORM`, `MI`, `STTC`, `CD`, and `HYP`. Each model emits five logits
with independent sigmoid probabilities, using a BCE-with-logits objective.
The task permits multiple positive labels; `NORM` is an ontology target and
does not establish that an ECG or patient is clinically normal. Counts, input
semantics, and manifest provenance are documented in the
[model card](../docs/MODEL_CARD_PTBXL_SUPERCLASS_R3.md#task-data-and-split-governance).

### Split governance and model selection

Patients were isolated to one official fold. Folds 1–7 supplied training and
normalization statistics; fold 8 supplied model and epoch selection through
a matched 12-candidate search and paired three-seed confirmation. Fresh
fixed-epoch refits used folds 1–8 while retaining folds-1–7 normalization.
Fold 9 supplied calibration and decision policies. Fold 10 was opened on
2026-08-09 UTC for the single preregistered batch of six frozen members:
two architectures at seeds 2026, 2027, and 2028.

The ResNet has 8,739,973 trainable parameters and temporal residual
convolutions; the transformer has 8,726,833 parameters, 20-sample temporal
patches, and a class token. The parameter-count gap is approximately 0.1503%.
Frozen refits used 12 and 22 epochs respectively, under the shared task,
normalization, objective, search budget, and deterministic BF16 CUDA policy.
These are the frozen design facts in the
[model card](../docs/MODEL_CARD_PTBXL_SUPERCLASS_R3.md#models-and-training),
not a new compute-efficiency measurement.

### Calibration, decisions, and uncertainty

For each frozen member, fold 9 fitted one global temperature, five labelwise
F1 thresholds, and mean-normalized-binary-entropy cutoffs at target coverages
1.0, 0.9, 0.8, 0.7, and 0.5. They were applied unchanged on fold 10. The
ResNet seed-2026 nominal-0.8 gate is a frozen demo default, not a choice made
from fold-10 performance. Accepted predictions can still be incorrect.

Architecture summaries report means and sample standard deviations across
the three seed point estimates. Paired differences are transformer minus
ResNet, with shared patient-cluster draws on aligned patients. These
within-seed patient-bootstrap intervals are distinct from between-seed SD.
The fixed 15-bin ECE estimator has resampling/binning sensitivity; its
bootstrap intervals are treated as sensitivity summaries, not conventional
inferential guarantees. See the
[calibration protocol and interpretation](../docs/MODEL_CARD_PTBXL_SUPERCLASS_R3.md#calibration-thresholds-and-abstention)
and [paired public table](results/tables/paired_deltas.csv).

### Post-evaluation r3 audits

The completed r3 audits examined frozen risk–coverage behavior and demographic
subgroups, a controlled-corruption grid, and attribution controls. The
corruption grid contains 246 verified member-cases: six clean-equivalence
cases and 240 non-clean corruptions. The fixed explanation cohort contains
60 ECGs and produced 15 method artifacts and 900 method-ECG evaluations
across six members. Grad-CAM, Integrated Gradients, and temporal occlusion
were assessed through repetition, perturbation, randomization, deletion,
and cross-method controls. These are audits of model sensitivity, without
clinical localization ground truth. The completed design and scope are in
the [model card](../docs/MODEL_CARD_PTBXL_SUPERCLASS_R3.md) and
[public audit summary](../reports/FINAL_RESULTS_PUBLIC.md#post-evaluation-findings).

### Separate exploratory SPH transport

After r3 completion, all six frozen models were applied once to SPH without
target-domain training, selection, preprocessing adaptation, recalibration,
threshold changes, or gate changes. A conservative AHA-to-PTB-superclass
mapping defined the primary cohort and two sensitivity cohorts. The mapping
was not clinically adjudicated. The broad cohort operationally treats 3,144
records without a direct mapping as all-zero; these are unknown, not verified
negatives. This limitation prevents treating cohort differences as a simple
cross-dataset performance ranking.

The [frozen SPH report](external_transport_sph_r2/FINAL_RESULTS.md),
[cohort summary](external_transport_sph_r2/cohort_summary.json), and
[protocol](../configs/external_transport_sph_frozen_r2.yaml) define this
separate retrospective experiment.

### Protocol deviation and operational recovery

[DEV-001](../reports/PROTOCOL_DEVIATIONS.md) records accidental pre-evaluation
exposure to a bounded set of raw fold-10 label-bearing metadata rows. No
exposed value informed a model, calibration, threshold, gate, subgroup, or
reporting choice, and no fold-10 prediction or metric was seen at that time.
Nevertheless, strict operator-level outcome-label blindness was breached;
the study does not claim a completely blind test.

Representation-only SHA-256 comparison failures required exact operational
resumes of the sealed batch. Existing immutable predictions were revalidated
and adopted without overwrite or a repeated scientific query. The
[evaluation run log](../reports/FINAL_EVALUATION_RUN_LOG.md) preserves this
recovery record. Neither this draft nor the engineering checks repeat those
scientific operations.

## Results

### Confirmatory sealed PTB-XL fold-10 comparison

Values below are mean ± sample SD across the three frozen seeds, copied at
six-decimal presentation precision from the
[architecture table](results/tables/architecture_metrics.csv).

| Architecture | Macro AUROC | Macro AP | Brier | ECE |
|---|---:|---:|---:|---:|
| 1D ResNet | 0.921921 ± 0.000913 | 0.810248 ± 0.003327 | 0.085040 ± 0.000718 | 0.022744 ± 0.002984 |
| ECG transformer | 0.897420 ± 0.003270 | 0.765270 ± 0.007739 | 0.096807 ± 0.001968 | 0.025646 ± 0.001163 |

The preregistered paired AUROC differences are shown below at the public
report's four-decimal precision. Negative differences favor the ResNet.

| Seed | Transformer minus ResNet macro AUROC | Paired 95% CI |
|---:|---:|---:|
| 2026 | -0.0280 | [-0.0342, -0.0222] |
| 2027 | -0.0244 | [-0.0304, -0.0185] |
| 2028 | -0.0211 | [-0.0272, -0.0150] |

Within every seed, paired intervals favored the ResNet for macro AUROC,
average precision, and Brier score. Every paired ECE interval crossed zero;
no comparative ECE advantage is established. AUROC measures ranking across
labels and must not be described as diagnostic accuracy. The complete paired
comparisons are in the [public CSV](results/tables/paired_deltas.csv) and
[model card](../docs/MODEL_CARD_PTBXL_SUPERCLASS_R3.md#sealed-fold-10-results).

### Descriptive selective prediction and subgroup coverage

At nominal 0.8 coverage, realized global coverage was approximately 0.79–0.81.
Hamming risk was approximately 0.09–0.10 for the ResNet and 0.11 for the
transformer. Coverage for age 80+ was approximately 0.60–0.65, compared with
0.93–0.95 below age 40. Thus a global gate produced materially unequal
retention across age groups, and retained cases still contained errors.
These are internal-cohort descriptive observations, not fairness or safety
guarantees. The rounded ranges are retained from the
[model card](../docs/MODEL_CARD_PTBXL_SUPERCLASS_R3.md); exact aggregate
values are available in the [risk–coverage](results/tables/mean_seed_risk_coverage.csv)
and [subgroup coverage](results/tables/subgroup_coverage.csv) tables.

### Descriptive corruption and attribution controls

The adverse controlled-corruption findings below are copied from the
[public audit summary](../reports/FINAL_RESULTS_PUBLIC.md#controlled-corruption-audit).

| Statistic | Worst observed delta | Member and case |
|---|---:|---|
| Macro AUROC | -0.2939 | Transformer seed 2027, reverse all leads |
| Macro Brier | +0.2213 | Transformer seed 2027, reverse all leads |
| Hamming-AURC | +0.3857 | ResNet seed 2026, reverse all leads |
| Absolute gate-coverage change | 0.5714 | ResNet seed 2027, drop precordial leads |

The explanation controls produced a minimum method-level mean repeat cosine
of 1.0000, mean stability cosine of 0.9929 at 40 dB, and mean parameter-
randomization cosine of -0.0067. Mean random-minus-guided deletion AUC was
0.0495 (positive favors guided ranking). Mean cross-method cosine was 0.5636
and Spearman correlation was 0.2012, each with 720/720 valid comparisons.
The maximum FP32-attribution versus sealed-BF16 raw-logit drift was 0.0913756.
These [published controls](../reports/FINAL_RESULTS_PUBLIC.md#explanation-control-audit)
show repeatability alongside disagreement and precision sensitivity; they
do not establish physiological causality or clinical explanation validity.

### Exploratory SPH transport results

The SPH cohort sizes are copied from the
[public cohort summary](external_transport_sph_r2/cohort_summary.json).

| Cohort | ECGs | Patients |
|---|---:|---:|
| Broad exact-10-second sensitivity | 18,842 | 18,157 |
| Primary mapped | 15,698 | 15,193 |
| No-ambiguous mapped sensitivity | 15,563 | 15,066 |

Primary-cohort calibrated results are mean ± sample SD across the three
frozen seeds, as reported in the
[SPH result table](external_transport_sph_r2/FINAL_RESULTS.md).

| Architecture | Macro AUROC | Macro AP | Brier | ECE |
|---|---:|---:|---:|---:|
| 1D ResNet | 0.930912 ± 0.000964 | 0.698955 ± 0.006752 | 0.061301 ± 0.000248 | 0.052477 ± 0.000877 |
| ECG transformer | 0.924088 ± 0.001231 | 0.657838 ± 0.007557 | 0.064153 ± 0.003962 | 0.061480 ± 0.006313 |

Each seed's paired patient-bootstrap AUROC interval favored the ResNet.
The primary cohort contains only 138 MI-positive ECGs from 131 patients and
113 HYP-positive ECGs from 110 patients. Rare endpoints, the unadjudicated
ontology bridge, and retrospective sampling limit interpretation. The
[full SPH report](external_transport_sph_r2/FINAL_RESULTS.md) preserves both
sensitivity cohorts. These findings are no-adaptation external transport
evidence and do not establish clinical validity or prospective utility.

### OOD evidence boundary

The separate, completed OOD-v1 source-support experiment retained 440 of 465
known-source ECGs. Record-level false rejection was 5.3763%, and its one-sided
95% patient-cluster upper bound was 7.2961%, exceeding the preregistered
5.0000% maximum. The result is integrity-valid but unfavorable, with
`research_bundle_eligible=false`. No OOD-positive cohort was evaluated, so
OOD recall, AUROC, and average precision remain unestablished. These facts
are preserved from the [sealed completion result](../docs/TRUST_SENTINEL_OOD_COMPLETION_RESULT.md);
they are not new outcomes from this draft.

### Interpretation and limits

The sealed comparison supports a ResNet advantage for discrimination and
Brier score within this protocol. It leaves comparative ECE unresolved and
shows age-related under-coverage and sensitivity to controlled corruptions.
The separate SPH result adds limited retrospective transport evidence.
DEV-001 remains a required disclosure alongside every scientific claim.
Neither study, the attribution controls, nor the later Sentinel software
establishes diagnostic safety, medical-device readiness, or fitness for
patient care. See the [model-card scope](../docs/MODEL_CARD_PTBXL_SUPERCLASS_R3.md#intended-use-and-prohibited-use).

## Artifact checks for editorial review

- Check fold-10 numbers against the [architecture](results/tables/architecture_metrics.csv)
  and [paired-difference](results/tables/paired_deltas.csv) CSVs.
- Check SPH numbers against the [separate report](external_transport_sph_r2/FINAL_RESULTS.md)
  and [cohort JSON](external_transport_sph_r2/cohort_summary.json).
- Keep the [PTB-XL](results/SHA256SUMS.txt) and
  [SPH](external_transport_sph_r2/SHA256SUMS.txt) inventories unchanged.
- Retain [DEV-001](../reports/PROTOCOL_DEVIATIONS.md), age 80+ under-coverage,
  unresolved ECE, unadjudicated SPH labels, and the
  [OOD-v1 miss](../docs/TRUST_SENTINEL_OOD_COMPLETION_RESULT.md) in subsequent edits.
- Follow [reproducibility guidance](../docs/REPRODUCIBILITY.md) for engineering
  checks; no new scientific execution is needed to review this draft.
