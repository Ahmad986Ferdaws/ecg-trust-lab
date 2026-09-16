# Paper outline: ECG discrimination, calibration, and selective prediction

Draft prepared from existing evidence on 2026-09-16 for issue #4. The ten
sections follow [blueprint section 17](../docs/RESEARCH_BLUEPRINT.md#17-suggested-paperreport-structure).
This outline introduces no experiment or new analysis. The companion
[methods and results draft](METHODS_AND_RESULTS.md) develops sections 3–8.

## 1. Introduction: discrimination is not enough for ECG decision support

Frame the question as a controlled comparison of convolutional and transformer
models, followed by audits of probabilities, abstention, subgroup coverage,
corruption sensitivity, and attribution. Describe the contribution as
retrospective research evidence. Distinguish the completed scientific work
from the later Sentinel engineering system and its unvalidated capabilities.

Evidence: [model card](../docs/MODEL_CARD_PTBXL_SUPERCLASS_R3.md),
[case study](../docs/PORTFOLIO_CASE_STUDY.md), and
[public result snapshot](../reports/FINAL_RESULTS_PUBLIC.md).

## 2. Related work: benchmarks, demographic audits, and trust measurements

Organize the existing bibliography around PTB-XL benchmarking, the demographic
audit identified in the blueprint, temperature scaling, selective prediction,
and ECG attribution controls. Use these as context for the study design;
avoid cross-paper numerical rankings with unmatched protocols. This outline
does not claim a new literature search or priority over the cited work.

Evidence: [blueprint bibliography](../docs/RESEARCH_BLUEPRINT.md#18-primary-sources-and-implementation-references)
and [case-study research framing](../docs/PORTFOLIO_CASE_STUDY.md).

## 3. Data and label construction

Describe PTB-XL 1.0.3, the canonical labeled manifest, patient isolation,
100 Hz ten-second twelve-lead inputs in millivolts, and five independent
diagnostic-superclass targets. Explain the separate SPH cohorts and conservative,
unadjudicated ontology bridge. Absent broad-cohort mappings are unknown rather
than verified negatives; the primary MI and HYP endpoints have few positives.

Evidence: [model-card data/splits](../docs/MODEL_CARD_PTBXL_SUPERCLASS_R3.md#task-data-and-split-governance),
[SPH report](external_transport_sph_r2/FINAL_RESULTS.md), and
[SPH cohort summary](external_transport_sph_r2/cohort_summary.json).

## 4. Leakage-resistant experimental design

Trace folds 1–7 for fitting and normalization, fold 8 for model/epoch selection,
fresh refits on folds 1–8, fold 9 for temperatures/thresholds/entropy gates,
and the sealed exact-six-member fold-10 evaluation. Disclose DEV-001 alongside
this design: bounded pre-evaluation exposure of label-bearing metadata means
complete operator blindness cannot be claimed. Explain exact operational
resumes without repeated scientific queries. SPH remains a later, separate,
no-adaptation exploratory transport study.

Evidence: [model card](../docs/MODEL_CARD_PTBXL_SUPERCLASS_R3.md),
[DEV-001](../reports/PROTOCOL_DEVIATIONS.md),
[evaluation run log](../reports/FINAL_EVALUATION_RUN_LOG.md), and
[SPH protocol](../configs/external_transport_sph_frozen_r2.yaml).

## 5. Architecture and compute comparison

Describe the capacity-matched 1D ResNet and temporal-patch transformer,
matched search budget, paired confirmation seeds, frozen refit lengths, and
shared objective and training policy. Separate development selection metrics
from final results; parameter matching alone does not imply equal latency or
deployment cost. Do not invent a new speed or compute benchmark.

Evidence: [model-card architecture table](../docs/MODEL_CARD_PTBXL_SUPERCLASS_R3.md#models-and-training),
[ResNet configuration](../configs/train_resnet_matched.yaml), and
[transformer configuration](../configs/train_transformer_matched.yaml).

## 6. Discrimination and calibration results

Present the sealed fold-10 mean and sample SD across three seeds, then the
within-seed paired patient-bootstrap differences. ResNet is favored for AUROC,
AP, and Brier; every ECE interval crosses zero. Keep the ECE resampling/binning
caveat. Put SPH primary and sensitivity findings in a separate exploratory
subsection, with no clinical-validation interpretation.

Evidence: [architecture table](results/tables/architecture_metrics.csv),
[paired deltas](results/tables/paired_deltas.csv),
[public snapshot](../reports/FINAL_RESULTS_PUBLIC.md), and
[SPH results](external_transport_sph_r2/FINAL_RESULTS.md).

## 7. Selective-risk and subgroup-coverage results

Show the workload-risk tradeoff from unchanged fold-9 gates. At nominal 0.8
coverage, age 80+ coverage is approximately 0.60–0.65, compared with 0.93–0.95
below age 40. Retained predictions still contain errors. Treat these as
descriptive internal-cohort observations, without fairness or safety guarantees.

Evidence: [mean-seed risk–coverage](results/tables/mean_seed_risk_coverage.csv),
[subgroup coverage](results/tables/subgroup_coverage.csv),
[subgroup metrics](results/tables/subgroup_metrics.csv), and
[model-card interpretation](../docs/MODEL_CARD_PTBXL_SUPERCLASS_R3.md).

## 8. Robustness and attribution-faithfulness results

Summarize the completed controlled-corruption and explanation-control audits,
including adverse results, parameter randomization, deletion controls,
cross-method disagreement, and attribution precision drift. Controlled
perturbations are not estimates of natural distribution shift, and attribution
has no clinical localization ground truth.

Evidence: [public audit aggregates](../reports/FINAL_RESULTS_PUBLIC.md#post-evaluation-findings)
and [model card](../docs/MODEL_CARD_PTBXL_SUPERCLASS_R3.md).

## 9. Limitations and clinical scope

Retain DEV-001, inconclusive ECE, age 80+ under-coverage, rare SPH endpoints,
unadjudicated mapping, and retrospective scope. State explicitly that OOD-v1
completed but missed its source-support gate: the upper bound exceeded the
frozen maximum, and OOD-positive detection was not evaluated. Do not imply that
later engineering fixes or frozen OOD-v2.1 plans establish detection performance.
The system is not validated for diagnosis, triage, treatment, or patient care.

Evidence: [protocol deviations](../reports/PROTOCOL_DEVIATIONS.md),
[model-card limits](../docs/MODEL_CARD_PTBXL_SUPERCLASS_R3.md),
[SPH limits](external_transport_sph_r2/FINAL_RESULTS.md), and
[OOD-v1 unfavorable result](../docs/TRUST_SENTINEL_OOD_COMPLETION_RESULT.md).

## 10. Reproducibility checklist and artifact links

- Identify the sealed model/evaluation revisions and protocol hashes from the
  [model card](../docs/MODEL_CARD_PTBXL_SUPERCLASS_R3.md).
- Link the [PTB-XL publication inventory](results/SHA256SUMS.txt) and
  [SPH publication inventory](external_transport_sph_r2/SHA256SUMS.txt).
- Describe environment and CPU engineering checks using
  [reproducibility instructions](../docs/REPRODUCIBILITY.md); these checks
  do not certify a frozen Windows/CUDA scientific run.
- Preserve the [run log](../reports/FINAL_EVALUATION_RUN_LOG.md) and
  [DEV-001 disclosure](../reports/PROTOCOL_DEVIATIONS.md).
- Publish only the existing aggregate assets. Private identifiers, waveforms,
  predictions, and checkpoints remain outside this writing deliverable.

Completion criterion: every section above points to existing artifacts;
drafting does not require reopening a sealed cohort or adding a model.
