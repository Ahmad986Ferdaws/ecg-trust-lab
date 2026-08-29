# Trust Sentinel external OOD v2 parent protocol

**Protocol:** `trust-sentinel-ood-external-v2-parent`
**Status:** frozen parent preregistration, before download and execution

**Frozen:** 2026-08-29 17:36:32 UTC

**Scope:** retrospective external acquisition and population stress testing
**Clinical use:** prohibited

The machine-readable parent contract is
[`configs/trust_sentinel_ood_external_v2.yaml`](../configs/trust_sentinel_ood_external_v2.yaml).
It freezes the scientific selection and decision rules. A child execution
contract must later bind the exact downloaded file inventory, selected record
counts, ordered cohort identities, runtime, code revision, and output paths
before any waveform is decoded.

## Design history and v1 boundary

This protocol was written **after** the aggregate result of
`trust-sentinel-ood-completion-v1` was known. It is therefore not an independent
replication or a retry of v1. The v1 source-validation cohort C rejected 25 of
465 records, producing a 5.3763% source false-rejection rate and a 7.2961%
one-sided 95% patient-cluster upper bound. That missed the preregistered 5%
maximum and remains immutable unfavorable evidence.

No C waveform, identifier, embedding, score, subgroup, or error analysis may
enter this study. C cannot be used to fit a detector, select an operating point,
choose a method, or explain a v2 result. PTB-XL fold 10 and SPH are also
previously observed and excluded. The only permitted use of v1's aggregate
result is transparent historical context.

The external study uses the exact verified v1 distribution policy:

- artifact identity:
  `sha256:d544c28ad18b764e3e30cc316b092a41d75125a8334f1d41ed58c31ec37568db`;
- physical file SHA-256:
  `sha256:817d6e5c4a3058c064cdc7bdceafb774c7ea4bb0b6cf725be1b8f12c7aae9c1c`;
- method: frozen 512-dimensional ResNet embedding with shrinkage Mahalanobis
  distance;
- threshold: `270.9668613705653`;
- rejection rule: score strictly greater than the threshold.

There is no candidate comparison, refit, recalibration, threshold adjustment,
ensemble, or target-site adaptation. Threshold ties remain supported. The
whole v1 success bundle must verify before the policy is loaded, but its known
`research_bundle_eligible=false` status must also be preserved and reported.

## Frozen external sources

### PhysioNet Challenge 2011 Set A

The first source is Set A from [PhysioNet Challenge 2011 version
1.0.0](https://physionet.org/content/challenge-2011/1.0.0/), using the official
[`set-a`](https://physionet.org/content/challenge-2011/1.0.0/set-a/) WFDB files.
The files are open under the Open Data Commons Attribution License 1.0.

Set A contains 1,000 ten-second, simultaneous 12-lead ECGs sampled at 500 Hz
with 16-bit resolution. Although the original Challenge concerned mobile
collection, Set A was acquired with conventional ECG machines. Every official
Set A WFDB header/data pair is selected exactly once. The text/CSV
representation is ignored so the same signal is not counted twice. No
demographic or quality filtering is allowed before quality evaluation.

Set A is the primary natural technical-quality cohort because it provides
reference assessments based on blinded review by 3–18 annotators:

- Group 1: acceptable;
- Group 2: indeterminate and excluded from primary binary quality metrics;
- Group 3: unacceptable.

The source does not provide a trustworthy patient-clustering key for this
protocol. Record identifiers must not be relabeled as patients, so Challenge
uncertainty uses a record bootstrap and no patient-level claim is allowed.

### ZZU pediatric ECG v1

The second source is [ZZU pECG version
1](https://doi.org/10.6084/m9.figshare.27078763.v1), published on figshare under
CC BY 4.0. The upstream release describes 14,190 ECGs from 11,643 hospitalized
children aged 0–14: 12,334 are 12-lead and 1,856 are 9-lead, sampled at 500 Hz
with durations from 5 to 120 seconds.

The frozen selection includes all and only records that:

1. contain exactly the 12 canonical named leads;
2. contain at least 5,000 samples at 500 Hz; and
3. are at least ten seconds long.

Nine-lead records and recordings shorter than ten seconds are excluded from
header metadata before waveform decoding. Missing leads are never derived,
and short recordings are never padded. Exact selected record and patient
counts, plus an ordered identity digest, must be frozen in the child execution
contract before waveform access. All eligible records and all records from a
selected patient are retained; disease, age, sex, score, or quality cannot
select cases.

ZZU disease labels are not used to define OOD, choose records, fit the policy,
or assess diagnostic performance. Dataset-scoped patient IDs are used only for
role integrity and cluster resampling.

## Canonical signal path

Both sources follow one preprocessing path:

1. Decode physical millivolts and validate finite numeric values, unique named
   leads, source rate, and sufficient duration.
2. Reorder by explicit lead names to
   `[I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6]`. No lead is inferred,
   repaired, or relabeled.
3. Select the first 5,000 samples—the first ten seconds—without a random,
   quality-based, or model-based window choice.
4. Resample from 500 Hz to 100 Hz with
   `scipy.signal.resample_poly(up=1, down=5, axis=time,
   window=("kaiser", 5.0), padtype="constant")`.
5. Require a contiguous finite `float32` array of shape `[12, 1000]`.
6. Run the frozen quality policy in physical-mV space.
7. Only after a quality pass, apply the unchanged PTB-XL folds-1–7
   normalization and extract the v1 embedding twice in deterministic CUDA
   inference mode. The aligned embedding hashes must match exactly.

There is no silent rescaling, clipping, imputation, padding, target-site
normalization, or adaptive preprocessing.

## Technical quality is not distribution support

The fail-closed decision order remains:

1. `INVALID_INPUT` — file, metadata, units, lead, shape, or processing contract
   failure;
2. `REACQUIRE` — the recording is technically unusable under the frozen quality
   policy;
3. `UNSUPPORTED_INPUT` — the valid, quality-passing signal exceeds the exact v1
   distribution threshold;
4. `ABSTAIN` — supported input with insufficient prediction certainty; and
5. `PREDICTION_ALLOWED` — every preceding gate passed.

Only quality-status `PASS` records enter distribution-support metrics. Catching
an unacceptable recording is technical-quality success, never OOD-detector
success. Conversely, a high distribution score cannot excuse a missed quality
failure.

## Operational OOD-positive definition

Every selected, contract-valid, quality-passing external record is an
**operational acquisition/population OOD positive** relative to the PTB-XL-fitted
policy:

- Challenge Set A represents a separately acquired conventional-machine
  population and acquisition domain;
- ZZU represents a pediatric hospital population and separate acquisition
  domain.

This definition does not inspect disease labels. It does not say that a record
contains an unknown disease, that the model has discovered a condition, or that
the score is a probability of being OOD. The study evaluates external-domain
rejection at one frozen operating point. It has no external in-distribution
control, so OOD AUROC and average precision are explicitly not evaluated.

## Four co-primary gates

All four endpoints are fixed before waveform access:

| Endpoint | Statistic | Required lower bound |
|---|---|---:|
| Natural technical-quality sensitivity | Fraction of Challenge Group-3 records receiving quality status `REACQUIRE` | 95% |
| Acceptable-record quality pass rate | Fraction of Challenge Group-1 records receiving quality status `PASS` | 90% |
| Challenge external distribution recall | Fraction of all quality-passing selected Set A records rejected by the exact v1 threshold | 90% |
| ZZU external distribution recall | Record-weighted fraction of quality-passing selected records rejected by the exact v1 threshold | 90% |

The family-wise one-sided alpha is 0.05. Bonferroni correction allocates 0.0125
to each endpoint, so each gate uses a one-sided 98.75% lower confidence bound.
All four lower bounds must meet their target. Challenge uses a 10,000-replicate
record bootstrap with PCG64 seed `20260901`; ZZU uses a 10,000-replicate patient
cluster bootstrap with seed `20260902`, drawing patients with replacement and
including all records from each draw. Replicates retain canonical ordering and
use `numpy.quantile(..., method="linear")`. Empty, undefined, or nonfinite
replicates fail closed.

Hard gates additionally require:

- complete Challenge reference-label alignment and zero `INVALID_INPUT` Set A
  records, so malformed inputs cannot count as successful quality blocking;
- zero Challenge Group-3 cases ending in `PREDICTION_ALLOWED`;
- zero skipped selected records;
- no external fitting or adaptation;
- exact v1 policy bytes before and after execution;
- exact input-inventory verification before and after execution;
- matching repeated embedding hashes;
- a valid immutable success bundle with no failure receipt; and
- aggregate-only public evidence.

Group 2, quality reason codes, score quantiles, patient-equalized ZZU recall,
patient-any ZZU recall, route counts, and exclusion counts are secondary and
cannot rescue a failed primary gate. Diagnostic performance, pediatric disease
performance, unknown-disease recall, OOD AUROC, and OOD average precision are
not evaluated.

## One-shot access and immutable evidence

Before the access boundary, execution may download and checksum files, parse
headers and non-waveform metadata, read the Challenge reference quality groups,
and build the exact selected identity inventory. It may not decode a waveform
sample, run quality logic, extract an embedding, calculate a distribution score,
or inspect an endpoint.

After the parent and child contracts are frozen on a clean committed revision,
the implementation must durably create `external-access-armed.json`. It then
atomically creates, without overwrite, the permanent adjacent claim:

`artifacts/trust_sentinel/.ood_external_v2.one-shot-claim.json`

Claim publication precedes the first selected waveform-sample decode and covers
both external sources as one experiment. Retry, resume, or a second inference
pass beyond the frozen deterministic repeat is forbidden. Post-claim failure
preserves existing evidence, retains the claim and marker, emits a sanitized
failure receipt when possible, and requires a new protocol and output root.

The private bundle may contain alignments, patient keys, embeddings, scores,
quality decisions, route decisions, and bootstrap replicates. Public output is
created through an aggregate allowlist and may never contain record or patient
IDs, paths, waveforms, embeddings, row scores, logits, probabilities, raw
demographics, native disease labels, or replicate arrays. Publication is never
automatic.

## Eligibility and integration

Passing every adjusted endpoint and hard gate yields only
`EXTERNAL_OOD_EVIDENCE_COMPLETE`. A miss yields
`EXTERNAL_OOD_TARGET_MISSED`; it is still completed evidence and cannot be
discarded or retuned.

This external study cannot by itself authorize Sentinel integration. v1's
source-support evidence is currently ineligible, so integration remains closed
even if every external gate passes. Integration would additionally require a
distinct future source-retention protocol that succeeds without retrying or
reusing C, plus whole-release compatibility and integrity verification.

## Claim boundary

The strongest permitted statement is that the exact, untuned v1 distribution
policy and frozen quality policy were retrospectively stress-tested on the two
declared external acquisition/population domains, with technical quality kept
separate from distribution support.

This is research-only evidence. It is not clinical validation, pediatric
diagnostic validation, proof of unknown-disease detection, an OOD probability,
evidence of clinical safety, a treatment recommendation, a medical device, or
a deployment-ready Trust Sentinel release.
