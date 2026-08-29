# Trust Sentinel OOD-completion protocol

**Protocol:** `trust-sentinel-ood-completion-v1`

**Status:** frozen before execution

**Frozen:** 2026-08-29 08:53:21 UTC
**Scope:** retrospective PTB-XL source-domain research only

This is a preregistration, not a completed detector, validated OOD study, or
release. No embedding result is created by this document. The machine-readable
contract is
[`configs/trust_sentinel_ood_completion_v1.yaml`](../configs/trust_sentinel_ood_completion_v1.yaml).

## Why this is a separate protocol

The completed source-calibration artifact is intentionally immutable. It
records its unfamiliar-input component as `PENDING` and must stay that way.
This protocol will create separate evidence that references the exact source
artifact; it will never replace its status, fields, or bytes. An unfavorable
result remains valid completed evidence and cannot be overwritten or silently
rerun under the same protocol.

## Frozen data roles

| Role | Source | ECGs | Patients | Permitted use |
|---|---|---:|---:|---|
| R / `REFERENCE` | PTB-XL task-manifest folds 1–8 | 17,084 | 14,823 | Embedding mean and covariance only |
| B / `THRESHOLD_FIT` | Reserved fold-9 hash range `[0.4,0.8)` | 834 | 757 | Source-support threshold only |
| C / `SOURCE_VALIDATION` | Reserved fold-9 hash range `[0.8,1.0)` | 465 | 409 | One evaluation after detector sealing |

Patients must be disjoint across R, B, and C. Records are weighted equally for
the reference estimator and threshold. C includes every canonical selected
record; no signal-quality filter, label, score, or manual review may remove a
case. Fold-9 decision-fit A, PTB-XL fold 10, SPH, observed external sites, and
future lockboxes are forbidden.

The exact ordered input identities are frozen as
`sha256:4aec3498193b962a0f9434e2032f5050e6f7daf4a8ddb44f87f54721efb72ae8`
for R and
`sha256:f5b06b01ca347e33068b128d93a0ed6bc3cd0e1f2e85f931cae5b35834612707`
for the complete fold-9 pool. These are distinct from the private embedding
artifact alignment hashes, which do not contain `record_path`.

Their byte contract is exact. Rows are sorted by strictly increasing `ecg_id`.
`ecg_id`, `patient_id`, and `strat_fold` are encoded as base-10 JSON integers;
booleans and missing values are invalid. `record_path` must already be
normalized project-relative POSIX text with no backslash, absolute root, `.`,
or `..` component. The payload is:

```json
{"algorithm":"ordered_role_input_identity_v1","records":[{"ecg_id":1,"patient_id":15709,"record_path":"records100/00000/00001_lr","strat_fold":3}],"schema_version":1}
```

The displayed record is illustrative; the real array contains every selected
record. Serialization uses Python `json.dumps` with `allow_nan=False`,
`ensure_ascii=True`, `separators=(",", ":")`, and `sort_keys=True`, encoded as
UTF-8 with no trailing newline. The exact digest input is the UTF-8 domain
prefix `ecg_trust.ordered_role_input_identity.v1`, one zero byte, then those
canonical JSON bytes:

```python
domain = b"ecg_trust.ordered_role_input_identity.v1\x00"
canonical = json.dumps(
    payload,
    allow_nan=False,
    ensure_ascii=True,
    separators=(",", ":"),
    sort_keys=True,
).encode("utf-8")
identity = "sha256:" + hashlib.sha256(domain + canonical).hexdigest()
```

### Official checksum-subset lineage

A second digest binds the exact official PTB-XL files selected for inference.
It is separate from both cohort identity and embedding alignment. The selected
base paths are the 17,084 R records plus the 834 B and 465 C records; fold-9 A
is excluded. Each base path must have exactly one `.dat` and one `.hea` entry
in the bound official `SHA256SUMS.txt`, producing 18,383 selected records and
36,766 file pairs. Before inference, each actual file hash must match its
official entry.

Each pair is represented as `{"relative_path": path, "sha256": digest}`.
Paths must be normalized dataset-relative POSIX text. Digests are lowercase
64-character hexadecimal text without a `sha256:` prefix. Duplicate paths,
missing pairs, unexpected pairs, unsafe paths, and hash mismatches fail the
run. Pairs are sorted by the UTF-8 bytes of `relative_path`. The canonical
payload is:

```json
{"algorithm":"official_checksum_subset_v1","files":[{"relative_path":"records100/00000/00001_lr.dat","sha256":"..."}],"schema_version":1}
```

It uses the same exact JSON settings and no trailing newline. Its digest input
is the UTF-8 domain prefix `ecg_trust.official_checksum_subset.v1`, one zero
byte, and the canonical payload:

```python
domain = b"ecg_trust.official_checksum_subset.v1\x00"
canonical = json.dumps(
    payload,
    allow_nan=False,
    ensure_ascii=True,
    separators=(",", ":"),
    sort_keys=True,
).encode("utf-8")
checksum_subset_sha256 = "sha256:" + hashlib.sha256(domain + canonical).hexdigest()
```

The protocol freezes this encoding, not an observed digest. The observed
digest is produced only during execution after all selected bytes verify. Only
the digest and aggregate selected-record/file counts may enter public lineage;
the full path/hash inventory remains private.

## Frozen model and embedding

Every ECG must already satisfy the canonical input contract: simultaneous 12
leads in the order I, II, III, aVR, aVL, aVF, V1–V6; physical mV; 100 Hz; ten
seconds; and 1,000 samples per lead. It is normalized with the existing
folds-1–7 per-lead statistics. Resampling, augmentation, and silent repair are
forbidden.

For normalized input \(x_i\), the 512-value representation is

\[
h_i=\operatorname{AdaptiveAvgPool1d}
\left(\operatorname{ResNetStages}(\operatorname{Stem}(x_i))\right).
\]

This is the final temporal feature map after stage four and global average
pooling, before classifier dropout and the classifier. The single frozen
ResNet checkpoint is evaluated under `torch.inference_mode()` and `eval()`.

Embedding inference is bound to `cuda:0` on the RTX 5070 Ti in FP32. Inputs,
weights, and exported embeddings remain FP32; autocast, TF32, and
`torch.compile` are disabled. The runtime is bound to Python `3.12.13`, NVIDIA
driver `596.49`, and cuDNN human version `9.20.0`, with
`torch.backends.cudnn.version()` returning integer `92000`. Deterministic
algorithms are required, with a batch size of 128, four workers, fixed ascending
ECG order, no shuffle, and no dropped final batch. Every role is extracted twice
in full. Its ordered
alignment and embedding-tensor SHA-256 values must match exactly across passes
or the run fails.

The historical demo policy and demo binding may be read only to satisfy the
existing checkpoint-loading lineage checks. Their temperature, classification
thresholds, and entropy decision cannot fit or select any OOD component.

## Frozen detector mathematics

Detector fitting and scoring use CPU float64. With reference count \(n_R\),
embedding dimension \(d=512\), mean \(\mu\), and sample covariance \(S\):

\[
\mu=\frac{1}{n_R}\sum_i h_i,
\qquad
S=\frac{1}{n_R-1}\sum_i(h_i-\mu)(h_i-\mu)^T,
\]

\[
\Sigma=0.9S+0.1\frac{\operatorname{tr}(S)}{d}I+10^{-6}I,
\qquad P=\Sigma^{-1},
\]

\[
D_i^2=\max\left(0,(h_i-\mu)^TP(h_i-\mu)\right).
\]

Higher scores mean less similar to the source reference; they are not
probabilities. Mean and covariance use R only. After they are fixed, B scores
select the 95% inlier order statistic:

\[
k=\left\lceil(834+1)\times0.95\right\rceil=794,
\qquad \tau=D^2_{(794)}.
\]

A record is supported when \(D_i^2\le\tau\) and unsupported only when
\(D_i^2>\tau\). All exact threshold ties are retained. This is a record-level
source threshold, not an individual or patient-level coverage guarantee.

## One-time source validation

Before the distribution policy containing \(\mu\), \(P\), and \(\tau\) is
sealed, only C's preregistered label-free identity and official checksum
metadata may be accessed, solely for partition, alignment, and provenance
integrity. C waveform bytes or waveform decoding, embeddings, scores, and
metrics remain forbidden until after sealing. C cannot influence a parameter
or selection.

The metrics-bearing source-calibration config and result are treated as opaque,
hash-bound files during initial preflight. Their schemas, historical C metrics,
and nested source-prediction bindings are decoded and cross-checked only after
the newly fitted distribution policy has been written, reloaded, and verified.
The patient split used to fit R and B comes from the independently frozen salt
already declared in this OOD protocol; the post-seal check must match it back
to the historical source-calibration lineage before C waveform access.

The primary outcome is record-level source false rejection. The report also
contains the required `source_record_support_coverage`, defined as accepted
records divided by all C records and exactly one minus the record false-
rejection rate, plus retained/rejected counts, patient-equalized and
patient-any rejection, score quantiles, and threshold ties.

Uncertainty uses a patient-cluster percentile bootstrap with 10,000 resamples
and seed `20260829`. Unique numeric patient IDs are sorted ascending and passed
to one `numpy.random.Generator(numpy.random.PCG64(seed))`. Each replicate draws
exactly 409 patient positions with replacement. Every record belonging to each
drawn patient is included once for that draw, so drawing a patient twice
repeats all of that patient's records twice. Records within a patient remain in
ascending `ecg_id` order. The replicate statistic is total rejected records
divided by total represented records; it is therefore record-weighted while
resampling whole patient clusters.

After all 10,000 finite replicate statistics are collected in generation
order, the two-sided 95% percentile interval is
`numpy.quantile(values, [0.025, 0.975], method="linear")`. The one-sided 95%
upper bound is `numpy.quantile(values, 0.95, method="linear")`. Empty or
nonfinite replicates fail the run. The evidence is eligible for a research
bundle only if that upper bound is at most 0.05. Eligible evidence uses status
`SOURCE_SUPPORT_GATE_COMPLETE`; otherwise it uses
`SOURCE_SUPPORT_GATE_TARGET_MISSED`. Both statuses describe only this
source-support gate, never validated OOD detection or release. Failing the rule
does not erase the result: it produces completed but non-eligible evidence.
Changing the threshold and reusing C is forbidden.

No OOD-positive cohort is used. The public result must contain the five exact
fields `semantic_ood_recall`, `severe_ood_recall`, `ood_auroc`,
`ood_average_precision`, and `unseen_site_or_device_performance`, each with the
exact value `NOT_EVALUATED`.

## Integrity bindings

The frozen configuration binds the exact source-calibration result and config,
PTB-XL task manifest and official `SHA256SUMS.txt`, refit completion and
checkpoint, resolved config and its inner hash, normalization, historical demo
lineage files, canonical experiment protocol, `pyproject.toml`, and `uv.lock`.
Every byte hash must verify before its file is decoded.

### One-shot C access and retained failure evidence

The execution output root is
`artifacts/trust_sentinel/ood_completion_v1` and must not already exist. Before
any C waveform byte is hashed, read, or decoded, execution first atomically
creates and durably flushes the sanitized staging marker
`source-validation-access-armed.json`. The marker contains a fresh 256-bit
owner nonce and binds the prospective canonical external-claim file SHA-256.
It contains no row identity, embedding, score, logit, probability, waveform,
or filesystem path.

Only after that marker is durable may execution atomically create, without
overwrite, the adjacent fixed-name claim
`artifacts/trust_sentinel/.ood_completion_v1.source-validation-one-shot-claim.json`.
Claim publication is the sole C-access boundary. The marker and claim must
contain the same owner nonce and their hashes must agree before any C waveform
access. The claim is permanent. If it already exists, concurrent access and
every retry or resume are forbidden. This ordering ensures that every durable
claim already has a durable matching marker.

A crash after the marker is durable but before claim publication proves no C
access; nevertheless, automatic cleanup or retry is forbidden and the orphaned
armed staging state requires manual forensic review under a new protocol. The
armed marker is retained in the output on both success and any failure after
the one-shot claim.

After the external claim exists, an exception must not delete already-created
evidence. The staging evidence and marker are preserved, the external claim is
never removed, and an atomic sanitized `failure-receipt.json` is emitted when
possible. A post-claim failure cannot have a success manifest, and the run
cannot be retried or resumed under this protocol.

### Success finalization and bundle verification

The output-directory commit is provisional until final verification. After
commit, every expected file is reopened and checked, and existing files may
not be mutated. The self-hashed `success-manifest.json` is then atomically
created without overwrite as the last success-path write. It is excluded from
its own inventory. The manifest inventories exactly these nine files in
ascending relative-path UTF-8 byte order:

1. `distribution-policy.json`
2. `ood-completion-result.json`
3. `private/reference-embeddings.json`
4. `private/reference-embeddings.npz`
5. `private/source-validation-embeddings.json`
6. `private/source-validation-embeddings.npz`
7. `private/threshold-fit-embeddings.json`
8. `private/threshold-fit-embeddings.npz`
9. `source-validation-access-armed.json`

Each inventory entry contains only `relative_path`, `file_sha256`, and
`size_bytes`. The manifest's `artifact_sha256` is SHA-256 over canonical JSON
excluding `artifact_sha256`, using `allow_nan=False`, `ensure_ascii=True`,
`separators=(",", ":")`, and `sort_keys=True`, with no newline in the hashed
body; the stored JSON has exactly one trailing newline. Once this last file is
written, the successful output root is immutable.

Bundle verification requires both the valid success manifest and the adjacent
one-shot claim. The external claim's file SHA-256 must match the value bound by
the retained marker. Verification fails on any `failure-receipt.json`, a
missing or extra expected output artifact, a file-size or SHA-256 mismatch, or
a manifest self-hash mismatch.

Three local private NPZ files and strict sidecars are retained for R, B, and C.
Their exact NPZ keys are `ecg_id`, `patient_id`, `strat_fold`, and singular
`embedding`. They contain row-level identities and embeddings and must never be
published. The distribution policy also remains private. The completion result
exposes aggregate counts, hashes, metrics, intervals, and claim boundaries
only—never identifiers, embeddings, row scores, logits, probabilities,
waveform data, or filesystem paths. Publication and release are never
automatic.

## Claim boundary

If completed, this protocol may support only the statement that a
Mahalanobis-based source-support threshold was fitted and evaluated on its
declared, patient-separated PTB-XL development roles. It cannot establish
validated OOD detection, unknown-disease discovery, an OOD probability,
external generalization, clinical validity or safety, diagnostic or treatment
utility, medical-device status, or a complete Trust Sentinel vNext release.
The exact machine-readable claim scope is
`retrospective_ptbxl_source_domain_development_only`.
