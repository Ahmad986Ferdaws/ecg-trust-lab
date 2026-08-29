# Trust Sentinel source-support completion result

- **Protocol:** `trust-sentinel-ood-completion-v1`
- **Completed:** 2026-08-29 UTC
- **Status:** `SOURCE_SUPPORT_GATE_TARGET_MISSED`
- **Research-bundle eligible:** no
- **Scope:** retrospective PTB-XL source-domain development only

## Result in one sentence

The one-shot evaluation retained 440 of 465 known-source PTB-XL ECGs
(94.62% source-support coverage), but its 5.38% false-rejection rate and 7.30%
one-sided 95% patient-cluster upper bound did not satisfy the preregistered
maximum of 5%.

This is a completed, integrity-valid unfavorable result. It is not an
execution failure, and the threshold was not tuned or retried after the
source-validation cohort was opened.

## Frozen design

The experiment kept all three roles patient-disjoint:

| Role | Purpose | PTB-XL folds | ECGs | Patients |
|---|---|---:|---:|---:|
| R | Estimate the source-reference mean and covariance from frozen 512-dimensional embeddings | 1–8 | 17,084 | 14,823 |
| B | Fix the source-support threshold by the preregistered order statistic | 9 | 834 | 757 |
| C | One-shot source-validation evaluation | 9 | 465 | 409 |

The detector used frozen ResNet embeddings and shrinkage Mahalanobis distance.
R estimated the reference mean and shrinkage covariance/precision without
altering the representation. B selected the 794th score in ascending order,
using rank `ceil((834 + 1) × 0.95)`. Rejection is strict only above that
threshold, so ties are retained. The rule was fixed before any C waveform
access. Uncertainty was estimated with 10,000 patient-cluster bootstrap
replicates using the frozen seed `20260829`.

## Aggregate outcomes

| Outcome | Value |
|---|---:|
| Source-validation ECGs retained | 440 / 465 |
| Source-validation ECGs rejected | 25 / 465 |
| Source-support coverage | 94.6237% |
| Record-level false rejection | 5.3763% |
| Patient-equalized false rejection | 4.6659% |
| Patients with any rejected ECG | 22 / 409 (5.3790%) |
| Two-sided 95% patient-cluster interval | 3.2396%–7.7253% |
| One-sided 95% patient-cluster upper bound | 7.2961% |
| Preregistered maximum upper bound | 5.0000% |
| Validation observations tied at the threshold | 0 |

Because 7.2961% is greater than 5.0000%, the frozen eligibility rule requires
`research_bundle_eligible=false`. The result was retained exactly as observed.

## What this does and does not establish

This experiment estimates how often the frozen source-support rule retains
known-source PTB-XL ECGs under its declared development split. It does not
measure whether unfamiliar ECGs are detected.

The following outcomes remain exactly `NOT_EVALUATED`:

- semantic OOD recall;
- severe OOD recall;
- OOD AUROC;
- OOD average precision; and
- unseen-site or unseen-device performance.

Accordingly, this result cannot support claims of validated OOD detection,
unknown-disease discovery, external generalization, clinical validity,
diagnostic safety, or medical-device readiness.

## Integrity and independent verification

The implementation and frozen protocol were committed before the one-shot run.
Three independent read-only audits then:

- accepted the complete immutable evidence bundle;
- reproduced the detector mean/precision fit, threshold, aggregates, and seeded
  10,000-replicate bootstrap exactly;
- confirmed patient separation, waveform and model-state hashes, runtime
  provenance, claim binding, and a clean source revision; and
- confirmed that the research-eligibility verifier rejects this bundle for the
  intended scientific reason.

Public verification identifiers:

| Item | Identifier type | Identifier |
|---|---|---|
| Frozen configuration | Canonical configuration SHA-256 | `5d12a71e8cd11350580a6d88b3656ca416392bedd3209d558ba116a90d536070` |
| Source revision | Git commit (SHA-1 repository format) | `fa85ddc727eeb892a23c9ec9023fa216fe214a1e` |
| Aggregate result | Logical artifact SHA-256 identity | `dd76258b30c95a3ac8f865da54973a42a93d0135caba09da0c6412267f041b53` |
| Success manifest | Logical manifest SHA-256 identity | `6f97e0697d661372e62f4aee9245f26014312e6a1d681615314bc9fcb77c5732` |

Private row-level embeddings, identifiers, and distribution-policy contents
remain local and are not publication artifacts.

## Interpretation

The scientifically correct conclusion is not that the software failed. The
software completed the preregistered experiment and preserved the result. The
source-support gate missed its target, so this version must not be promoted as
a research-eligible unfamiliar-input detector. A future protocol may introduce
new OOD-positive cohorts or a newly preregistered method, but it must not tune
against or relabel this one-shot result.
