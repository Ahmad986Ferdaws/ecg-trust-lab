# Public result snapshot: PTB-XL comparison and SPH transport stress test

> **Publication copy, not a new analysis.** The PTB-XL sections were prepared
> from the already finalized r3 report, and the separate SPH section was copied
> from the successfully audited frozen r2 transport output. This file is not
> part of either immutable derived manifest and does not supersede either local
> report.

## Interpretation boundary

Architecture metrics and preregistered within-seed paired patient-bootstrap
comparisons are the confirmatory sealed fold-10 results. Calibration views,
risk-coverage analysis, subgroup coverage, corruptions, explanations, and the
demo are post-evaluation descriptive audits of frozen models and policies.

This is research-only evidence. The later SPH experiment adds limited
retrospective external-transport evidence, but it does not establish clinical
validity, diagnostic safety, medical-device performance, fairness, prospective
utility, or fitness for patient care.

## Confirmatory fold-10 result

Values are mean +/- sample standard deviation across the three fixed seeds.

| Architecture | Macro AUROC | Macro AP | Brier | ECE |
|---|---:|---:|---:|---:|
| 1D ResNet | `0.921921 +/- 0.000913` | `0.810248 +/- 0.003327` | `0.085040 +/- 0.000718` | `0.022744 +/- 0.002984` |
| ECG transformer | `0.897420 +/- 0.003270` | `0.765270 +/- 0.007739` | `0.096807 +/- 0.001968` | `0.025646 +/- 0.001163` |

Paired differences are ECG transformer minus ResNet on aligned patients.
Negative AUROC and AP differences and positive Brier differences favor the
ResNet.

| Seed | Macro AUROC difference | Paired 95% CI |
|---:|---:|---:|
| 2026 | `-0.0280` | `[-0.0342, -0.0222]` |
| 2027 | `-0.0244` | `[-0.0304, -0.0185]` |
| 2028 | `-0.0211` | `[-0.0272, -0.0150]` |

Within every seed, paired patient-bootstrap intervals supported the ResNet for
macro AUROC, average precision, and Brier score. Every paired ECE interval
crossed zero. Fixed-bin ECE also has a resampling/binning caveat, so no
comparative ECE advantage is claimed.

## Exploratory SPH external transport result

After PTB-XL r3 was complete, the same six frozen members were applied once to
SPH under a pre-specified no-adaptation protocol. The broad exact-10-second
cohort contained 18,842 ECGs from 18,157 patients. The primary conservatively
mapped cohort contained 15,698 ECGs from 15,193 patients, and the
no-ambiguous sensitivity cohort contained 15,563 ECGs from 15,066 patients.

Primary calibrated values are mean +/- sample standard deviation across the
three frozen seeds.

| Architecture | Macro AUROC | Macro AP | Brier | ECE |
|---|---:|---:|---:|---:|
| 1D ResNet | `0.930912 +/- 0.000964` | `0.698955 +/- 0.006752` | `0.061301 +/- 0.000248` | `0.052477 +/- 0.000877` |
| ECG transformer | `0.924088 +/- 0.001231` | `0.657838 +/- 0.007557` | `0.064153 +/- 0.003962` | `0.061480 +/- 0.006313` |

Within each seed, the paired patient-bootstrap macro-AUROC interval favored the
ResNet. This result is an **exploratory external transport stress test** with
**no tuning or recalibration on SPH**; it is **not clinical validation**. The
AHA-to-PTB-superclass bridge was not clinically adjudicated, MI and HYP have
few positive patients, and the study is retrospective. The complete sanitized
tables and required limits are in the
[SPH result snapshot](../publication/external_transport_sph_r2/FINAL_RESULTS.md),
with the evidence review in the
[SPH external-transport audit](SPH_EXTERNAL_TRANSPORT_AUDIT.md).

## Post-evaluation findings

### Selective prediction and subgroups

Only fold-9-fitted temperatures, label thresholds, and entropy gates were used.
At nominal 0.8 coverage, realized coverage was roughly 0.79-0.81. Coverage for
age 80+ was approximately 0.60-0.65, versus about 0.93-0.95 below age 40.
Accepted cases still contained errors. These are internal-cohort observations,
not a clinical safety or fairness claim.

### Controlled-corruption audit

The grid contains 246 verified member-cases: six clean-equivalence cases and
240 non-clean corruptions.

| Statistic | Worst observed delta | Member / case |
|---|---:|---|
| Minimum macro AUROC delta | `-0.2939` | transformer seed 2027 / reverse all leads |
| Maximum macro Brier delta | `+0.2213` | transformer seed 2027 / reverse all leads |
| Maximum Hamming-AURC delta | `+0.3857` | ResNet seed 2026 / reverse all leads |
| Maximum absolute gate-coverage delta | `0.5714` | ResNet seed 2027 / drop precordial leads |

These deliberately controlled perturbations expose sensitivity; they are not
estimates of real-world distribution shift.

### Explanation-control audit

The fixed 60-ECG cohort produced 15 method artifacts and 900 method-ECG
evaluations across six model members.

| Control | Verified aggregate |
|---|---:|
| Minimum method-level mean repeat cosine | `1.0000` |
| Mean stability cosine at 40 dB | `0.9929` |
| Mean parameter-randomization cosine | `-0.0067` |
| Mean random-minus-guided deletion AUC | `0.0495` (positive favors guided ranking) |
| Mean cross-method cosine | `0.5636` (`720/720` valid) |
| Mean cross-method Spearman | `0.2012` (`720/720` valid) |
| Maximum FP32-attribution versus sealed-BF16 raw-logit drift | `0.0913756` |

There is no clinical localization ground truth. Grad-CAM, Integrated
Gradients, and temporal occlusion characterize model sensitivity, not
physiological causality or clinician reasoning.

## Published result assets

- [Architecture/member metrics](../publication/results/figures/architecture_member_metrics.png)
- [Paired differences and confidence intervals](../publication/results/figures/paired_deltas_ci.png)
- [Mean-seed risk-coverage](../publication/results/figures/mean_seed_risk_coverage.png)
- [Raw versus calibrated reliability](../publication/results/figures/raw_vs_calibrated_reliability_seed2026.png)
- [Subgroup performance and coverage](../publication/results/figures/subgroup_performance_coverage.png)
- [Identifier-free CSV tables](../publication/results/tables/)
- [SHA-256 inventory](../publication/results/SHA256SUMS.txt)
- [Sanitized SPH external-transport result](../publication/external_transport_sph_r2/FINAL_RESULTS.md)
- [SPH external-transport artifact audit](SPH_EXTERNAL_TRANSPORT_AUDIT.md)

The 12 published figure/table files contain aggregate results only. Raw ECGs,
patient identifiers, record identifiers, predictions, checkpoints, and the raw
post-evaluation specification are intentionally excluded.

## Required disclosure

[DEV-001](PROTOCOL_DEVIATIONS.md) records accidental pre-evaluation exposure of
a bounded set of raw fold-10 label-bearing metadata rows. No exposed value was
used for model, calibration, threshold, gate, subgroup, or reporting choices,
and no fold-10 prediction or metric was seen at that time. Strict
operator-level outcome-label blindness was nevertheless breached, so this work
does not claim a completely blind test.

The sealed batch also required exact resumes after representation-only SHA-256
comparison failures. Existing immutable predictions were revalidated and
adopted without overwrite or a repeated scientific query. The full recovery
record is in the [final-evaluation run log](FINAL_EVALUATION_RUN_LOG.md).

## Provenance anchors

- Protocol: `sha256:ebfdb588615bfa22eedc6d936d7b0155a33702878cbe0258ebb84aaa88567e09`
- Final-evaluation specification: `sha256:1f73c021a544ffeb119ffe8e490a16e32ec84247e30bce1ffd895fcffed6c762`
- Final batch: `sha256:a4da85d5272b1634baaf953496c3d9efd8917777ad8de13b1b0c6dc754699e62`
- r3 post-evaluation specification: `sha256:5727858f0c22b6311152749e0d9a3d20b3c14f4ee1c72ef9d1cf6e1943434200`
- r3 probability audit: `sha256:e5f5261b2ff95dfe11596e544227effad5d5e98a4e547251fa5246d32ad26528`
- r3 robustness manifest: `sha256:be9ff204f85f2ca11ed78ed93d4bf066029fd5c68f7c61fcfb0e1d9cfa6d906a`
- r3 explanation manifest: `sha256:0803fe333b8f736d8c61babc2c50fe26194d3a75a7e06d123640ee2e1fc9733d`
- r3 derived-manifest artifact: `sha256:fae6df30090ee59425a347034a7f4272cac5b799582a5742fff9c62b92a092f8`
- Sealed local final-report file: `sha256:c4363d89a43c59b7e7185a01b13a22377a90e35eb356e8a640bf7517fe721d0c`
- Frozen SPH r2 protocol: `sha256:840acb758d50dbf1a04bf704b16a58d4d29d668370ce7ef91ef7a44860bf311b`
- SPH public manifest: `sha256:eb333e255f41beece3cd5fa413e9a605c017bf63b0c977207c28e5ac1373fc0f`

The sealed local specification, manifest, and report remain in the ignored run
tree and are verified by the reproducibility pipeline. This public snapshot
contains their result-level evidence without publishing identifier-bearing or
machine-specific artifacts.
