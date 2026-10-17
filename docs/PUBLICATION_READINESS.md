# Publication readiness and claim review

Use this checklist when preparing a manuscript, preprint, talk or public project
announcement. It connects the existing research evidence to the author work
still needed for publication. A checked engineering test does not establish a
completed reporting checklist or clinical validity.

## Choose the report's scope

The primary study compares capacity-matched architectures on PTB-XL and audits
calibration, abstention, subgroup behavior, corruption sensitivity and explanation
faithfulness. The SPH experiment is a separate retrospective transport stress
test. Trust Sentinel and its OOD work must be described as distinct development
work, with the missed source-support target retained.

For a model development/evaluation paper, start with the official
[TRIPOD+AI statement and checklist](https://www.tripod-statement.org/), which
covers prediction-model reporting across regression and machine-learning methods.
If framing a study as diagnostic test accuracy, assess the applicability of
[STARD-AI](https://www.nature.com/articles/s41591-025-03953-8).
[DECIDE-AI](https://www.nature.com/articles/s41591-022-01772-9) addresses early
live clinical evaluation; the retrospective experiments here do not satisfy
that study scope. These are reporting guidelines, not regulatory approvals.
Guideline scope reviewed 2026-10-11.

## Map the manuscript to existing evidence

| Manuscript topic | Repository source | Author review still needed |
|---|---|---|
| Dataset identity and inclusion | [Data card](DATA_CARD.md), [research blueprint](RESEARCH_BLUEPRINT.md) | Match all counts to the exact task/cohort, include selection flow and exclusions |
| Patient isolation and split roles | [Reproducibility guide](REPRODUCIBILITY.md) | Explain training, selection, calibration and sealed evaluation separately |
| Model selection and compute budget | [Model card](MODEL_CARD_PTBXL_SUPERCLASS_R3.md) | Identify matched settings, selection rule and seeds; avoid implying exhaustive architecture comparison |
| Discrimination, calibration and uncertainty | [Public results](../reports/FINAL_RESULTS_PUBLIC.md) | Distinguish across-seed SD from patient-bootstrap intervals and report estimands and units |
| Abstention and subgroups | [Model card](MODEL_CARD_PTBXL_SUPERCLASS_R3.md) | Include the age 80+ retention problem and the gate/workload tradeoff |
| Explainability and corruptions | [Post-evaluation log](../reports/POST_EVALUATION_RUN_LOG.md) | Describe controlled probes, not causal physiology or clinical shift validation |
| External transport | [SPH protocol](EXTERNAL_TRANSPORT_SPH_R2.md), [result](../publication/external_transport_sph_r2/FINAL_RESULTS.md) | State cohort selection, rare labels and unadjudicated ontology mapping |
| Deviations and failures | [Protocol deviations](../reports/PROTOCOL_DEVIATIONS.md), [OOD result](TRUST_SENTINEL_OOD_COMPLETION_RESULT.md) | Disclose DEV-001 and missed OOD gate wherever their associated result is presented |
| Code and artifact availability | [Contribution guide](../CONTRIBUTING.md), [reproducibility guide](REPRODUCIBILITY.md) | Name the exact release commit and what a source-only clone omits |

This mapping is not a completed TRIPOD+AI or STARD-AI checklist. Authors should
complete the selected official checklist with manuscript page references and
explicit reasons for non-applicable items. Resolve missing details from source
records; do not fill gaps with assumed experiments or approvals.

## Check every quantitative claim

- Name the dataset, cohort, task, preprocessing and evaluation role beside the
  result. Do not combine PTB-XL and SPH sample counts as one validation cohort.
- Keep AUROC, average precision, Brier score and ECE distinct. A macro AUROC of
  0.9219 is not 92.19% accuracy, sensitivity or diagnostic correctness.
- State whether uncertainty is SD across three seeds or a patient-cluster
  interval. Three seeds are not three independent clinical sites.
- Retain the absence of a demonstrated comparative ECE advantage. Fixed-bin
  ECE also has the documented binning caveat.
- Report abstention retention and error on accepted cases together. Withholding
  a result does not prove safety, fairness or successful OOD detection.
- Treat the ontology bridge and all descriptive post-evaluation analyses as
  scoped limitations, not new prespecified confirmatory evidence.

## Prepare a reviewable release

1. Pin the intended source commit and retain immutable research artifacts.
2. Reproduce the public-evidence parity checks and relevant engineering gates;
   record platform, dependency versions, skips and failures.
3. Verify figure captions, table headings, sample counts and source attribution
   against the sealed aggregate artifacts. Do not regenerate evidence to improve
   the appearance of a result.
4. Check all links and ensure private identifiers, row-level predictions,
   waveforms and credentials are absent from the release bundle.
5. Confirm author contributions, funding, competing interests, applicable
   ethics/data-use statements and citation metadata from actual records.
6. Resolve the code-license decision and distribution rights for each included
   asset. A dataset license does not automatically license original code.
7. Have a reviewer trace every abstract claim to a methods/results passage.
   Record reviewer, date, unresolved items and final source revision.

Submission, DOI registration and public announcements are separate owner
actions. This checklist does not claim peer review, journal acceptance,
clinical validation or approval to run another experiment.
