# Trust Sentinel external OOD v2.1 successor protocol

**Protocol:** `trust-sentinel-ood-external-v2-1-parent`
**Status:** frozen successor preregistration, before waveform or model access
**Scope:** retrospective external acquisition/population stress testing
**Clinical use:** prohibited

The machine-readable authority is
[`configs/trust_sentinel_ood_external_v2_1.yaml`](../configs/trust_sentinel_ood_external_v2_1.yaml).
This document explains that contract; if prose and YAML differ, execution must
fail closed.

## Why v2.1 exists

The original v2 parent was frozen before download. Its metadata-only preflight
found that selected ZZU headers use `AVR`, `AVL`, and `AVF`, while the parent
required the exact mixed-case names `aVR`, `aVL`, and `aVF` and forbade
relabeling. Applying a name map would have been an unregistered preprocessing
change. The original run was therefore terminated before a claim, output root,
waveform decode, quality decision, embedding, score, logit, or metric existed.
That decision is preserved in the
[v2 termination note](TRUST_SENTINEL_OOD_EXTERNAL_V2_INFEASIBILITY.md).

This successor is knowingly written after that metadata-only feasibility
access. It permits exactly three dataset-specific, case-only ZZU aliases:
`AVR→aVR`, `AVL→aVL`, and `AVF→aVF`. Challenge records must retain the original
mixed-case names. No generic case folding is allowed. The map changes names
only; it cannot infer, swap, sign-flip, rescale, or otherwise alter a signal.

## Evidence boundary and v1 history

The aggregate v1 source-support result was already known when both external
protocols were designed. On 465 records from 409 patients, v1 rejected 25
records: 5.3763% false rejection with a one-sided 95% upper bound of 7.2961%,
above its frozen 5% maximum. That unfavorable result remains immutable.

The successor reuses exactly one v1 distribution policy:

- method: shrinkage Mahalanobis distance in the 512-dimensional frozen ResNet
  embedding;
- policy artifact:
  `sha256:d544c28ad18b764e3e30cc316b092a41d75125a8334f1d41ed58c31ec37568db`;
- physical policy file:
  `sha256:817d6e5c4a3058c064cdc7bdceafb774c7ea4bb0b6cf725be1b8f12c7aae9c1c`;
- threshold: `270.9668613705653`; and
- rejection rule: score strictly greater than the threshold.

The named whole-bundle verifier must authenticate v1 before use. It may parse
private v1 members solely to verify immutable bytes; no C identity, waveform,
embedding, score, row, subgroup, or error output may be exposed to this study.
There is no refit, recalibration, candidate comparison, threshold change,
target-site adaptation, or post-result method selection.

Even a perfect external result cannot authorize system integration because
the independent source-support gate remains failed. Integration stays false
until a distinct future source-retention study succeeds without reusing C.

## External cohorts and metadata-only observations

### PhysioNet Challenge 2011 Set A

Every official Set A WFDB header/data pair is selected exactly once: 1,000
ten-second, 12-lead, 500 Hz records. Official quality roles contain 773
acceptable, 225 unacceptable, and 2 indeterminate records. The source has no
trustworthy patient key for this protocol, so every inventory patient field is
null and uncertainty resampling is by record.

The downloaded archive is 107,909,323 bytes with SHA-256
`3e9d079fbc65eda8700d11549853b8ab97bb5103967df319e5c507d82129eb99`.
The exact `RECORDS`, acceptable, and unacceptable list hashes are frozen in the
YAML. Those lists are reparsed before and after evaluation to rederive every
role; the inventory cannot self-assert its labels.

### ZZU pediatric ECG v1

The release contains 14,190 ECGs from 11,643 hospitalized children. Header and
role metadata select all and only records with 12 admitted source-specific
lead names, 500 Hz sampling, and at least 5,000 samples/ten seconds. The frozen
metadata result is 12,328 selected records from 10,350 patients. Exclusions are
exactly 1,856 non-12-lead records and 6 recordings under ten seconds, with no
other exclusion reason.

The two split waveform archive parts and four supporting release files are
bound by exact size, SHA-256, and expected MD5 in the YAML. Before and after
evaluation, `AttributesDictionary.csv` is reparsed to rederive the exact
Filename–ECG_ID–Patient_ID relationship, every candidate header is accounted
for, and all-and-only selection plus exclusion counts are compared with the
inventory. Disease labels never select records, define OOD, fit a component,
or support a diagnostic claim.

Archive identity alone is insufficient. Before the child freeze, a byte-only
closure manifest proves that every evaluated extracted file is the matching
member of the exact frozen archive. Challenge closes all 3,004 regular release
members: 1,000 headers, 1,000 data files, 1,000 redundant text files, the
three reference lists, and `HEADER.shtml`. Only headers, data, and reference
lists are operational inputs; the other 1,001 files are ignored for analysis
but remain inside the byte-integrity closure. ZZU
validates a safe exact split-archive member list, tests the bound archive,
extracts it once into a fresh isolated temporary directory, and compares every
candidate path, size, and SHA-256 with the evaluated tree. Traversal, links,
duplicate, extra, or missing file members fail closed. The domain-separated
closure hash and aggregate member counts are bound in the child/public
projection; no samples are decoded.

## Successor inventory and signal path

The v2.1 inventory uses a fresh, no-overwrite namespace and cannot reuse the
v2 feasibility inventory. Its private record contract binds exact dataset and
record roles, patient/quality role or null, header and data hashes/sizes,
sampling rate, source sample count, 12 raw physical-unit strings, raw lead
order, canonical lead order, and all 12 local WFDB `.dat` leaf references.
Only an aggregate projection is committed publicly.

For every claimed record the runtime must:

1. verify the record is inside the exact frozen extraction root and that no
   source path, file, header reference, or Git binding enters PTB-XL, fold 10,
   SPH, the sealed C bundle, or a broad ancestor directory;
2. require one local `<record>.dat` leaf for all 12 WFDB channels, with no path
   separator or alternate data file;
3. preserve raw and canonical lead names separately and apply only the exact
   dataset-specific alias map;
4. decode physical millivolts, require exact 12×`mV` raw units, finite values,
   500 Hz, and sufficient samples;
5. select the first 5,000 source samples before resampling;
6. use SciPy `resample_poly(up=1, down=5, axis=time,
   window=("kaiser", 5.0), padtype="constant")` to produce contiguous
   float32 `[12,1000]`; and
7. run quality in physical-mV space, then reuse the exact bound PTB float32
   normalization path only for quality-passing model inputs.

An adapter, parser, path, unit, lead-name, header, or provenance contract error
is an execution-integrity failure. It is never converted into a completed
`INVALID_INPUT` row. `INVALID_INPUT` is reserved for a successfully adapted
canonical signal that the frozen natural technical-quality policy classifies
as invalid; its zero-per-dataset hard gate may therefore be evaluated honestly.

The four inventory/freeze/evaluate/verify CLIs start only through the exact project interpreter
with `-I -S -B -X pycache_prefix=<fresh-owned-root>/pycache`. A random
256-bit handoff binds a single-use runtime root under
`artifacts/trust_sentinel`; the working directory is the exact project root,
`site`/`.pth`/`sitecustomize`/`usercustomize` startup is forbidden, and the
launcher manually appends only the exact venv `site-packages` and project
`src` roots in the frozen order. Caller environment variables are discarded.
Ephemeral `APPDATA`, `LOCALAPPDATA`, `TEMP`, `TMP`, and `USERPROFILE` values
must resolve to the exact owned runtime-root layout but are represented by
stable semantic role labels in the child hash. Pycache, temporary, and
application-cache leaves must be empty immediately before terminal success and
again after the independent terminal reread, before success is returned.
`CUDA_CACHE_DISABLE=1` is exact and mandatory. `TORCHINDUCTOR_CACHE_DIR` is
bound directly to the existing isolated runtime-temp role so eager PyTorch
initialization cannot create an unbound `torchinductor_*` child. A dirty-scratch failure must
still be able to publish a sanitized failure receipt; receipt retention is not
conditioned on scratch emptiness.

Before the inventory builder reads any official archive, metadata, header, or
inventory byte, a metadata-only preflight requires the exact frozen parent,
clean pushed implementation revision X, live remote, protected-path history,
tracked parent and complete source blobs, isolated runtime/Git identity,
bound `__main__`, and absent successor claim/output. Before it reports success,
a postflight repeats that proof and strict-reloads the exact canonical private
inventory and public projection, matching physical and logical hashes to the
in-memory build.

The child path-neutrally binds every regular file and directory in the full
resolved CPython 3.12.13 base tree and the full venv `site-packages` tree,
including distribution metadata, licenses, `.pth`, bytecode, and native
libraries; there are no exclusions from those two complete trees. NumPy
`2.5.1`, SciPy `1.18.0`, WFDB `4.3.1`, Torch `2.13.0+cu130`, Pydantic
`2.13.4`, pydantic-core `2.46.4`, and PyYAML `6.0.3` also have an additional
import-root projection inside the complete site tree. Every file-backed
`sys.modules` origin, namespace search location, built-in/frozen loader, the
exact `__main__`, and every loaded native image is audited. A `.pyc`, shadow
module, unbound namespace, or non-OS native image outside the bound CPython or
site tree fails closed. On the uv Windows runtime, `sys.executable` is the
separately hash-bound venv redirector while PSAPI must observe the exact bound
base-tree `python.exe` as the process image; the redirector is not falsely
required to remain loaded.

Every tracked `src/ecg_trust/**/*.py` file and all four inventory/freeze/
evaluate/verify entrypoints are bound by exact path, size, SHA-256, worktree
bytes, and X/Y Git blobs. Git itself is the exact 2.53.0 executable and full
`mingw64` runtime tree, run with a sanitized environment, explicit Git/worktree
directories, disabled replacement/config redirection, and a strict local-
config allowlist. The NVIDIA query binds exact `nvidia-smi.exe`, `nvml.dll`,
and `nvcuda.dll` bytes and driver 596.49. The exact 7-Zip 26.02 executable and
library are copied into a fresh two-file application directory for every
call; a separate empty cwd and minimal `PATH` prevent adjacent codecs,
formats, plugins, caller-directory DLLs, or caller environment from executing.
All runtime trees and tools are recomputed at child freeze, immediately before
the claim, after evaluation, and during terminal semantic verification.
Absolute user paths are never published.

## Five-state decision path

Every selected record follows this order:

1. `INVALID_INPUT` only for a successfully adapted signal whose frozen natural
   technical-quality status is exactly invalid;
2. `REACQUIRE` for a non-pass result from the frozen technical-quality policy;
3. `UNSUPPORTED_INPUT` when a quality-pass embedding score is strictly above
   the frozen v1 threshold;
4. `ABSTAIN` when a supported input fails either frozen uncertainty gate; or
5. `PREDICTION_ALLOWED` only when every preceding gate passes.

The uncertainty path uses temperature `1.319620052379425`, maximum normalized
binary entropy `0.5975748221759414`, and the five exact labelwise split-
conformal thresholds bound in the YAML. Quality status, distribution support,
and uncertainty remain separate. A technical-quality success cannot count as
OOD detection, and an unsupported-input result cannot excuse a missed quality
failure.

## Operational OOD definition and four co-primary endpoints

An OOD positive means only a valid, quality-passing record from a preregistered
external acquisition or population domain relative to the PTB-XL-fitted
policy. It does not mean an unknown disease, discovered condition, or
probability of being OOD. There is no external in-distribution control, so OOD
AUROC and average precision are not evaluated.

All four one-sided 98.75% lower bounds must pass:

| Endpoint | Required lower bound |
|---|---:|
| Challenge unacceptable records with quality status exactly `REACQUIRE` | 95% |
| Challenge acceptable records with quality status `PASS` | 90% |
| Quality-passing Challenge records rejected by the v1 threshold | 90% |
| Quality-passing ZZU records rejected by the v1 threshold | 90% |

Bonferroni correction assigns one-sided alpha 0.0125 to each endpoint from a
family-wise alpha of 0.05. Every endpoint uses 10,000 PCG64 percentile-
bootstrap replicates and NumPy linear quantiles. Challenge draws exactly *n*
record indices with replacement using seed `20260901`; a binomial shortcut is
forbidden. Each Challenge endpoint reinitializes its own PCG64 generator with
that same seed; endpoints do not share one advancing stream. ZZU draws patients
with replacement using seed `20260902`, includes
all their quality-passing records, and computes record-weighted recall.

Hard gates additionally require zero invalid inputs for both datasets, zero
skipped records, zero Challenge Group-3 records reaching
`PREDICTION_ALLOWED`, at least one Challenge quality-pass record, and at least
80% of selected ZZU records and 80% of
selected ZZU patients represented in the quality-pass denominator. They also
require exact role rederivation, input and policy bytes unchanged before and
after, package/runtime agreement, two bit-exact embedding passes, and an
aggregate-only immutable bundle. The 80% coverage checks are deterministic
denominator-integrity gates, not additional inferential endpoints.

## One-shot execution and failure semantics

The parent and exact child must be committed and pushed before access. The
implementation revision and child-freeze execution revision are consecutive:
the execution revision has the implementation revision as its sole first
parent, and exactly one commit lies in `X..Y`. That commit is additive-only and
contains exactly the tracked child plus its aggregate public inventory
projection; the private inventory remains ignored and untracked. The only Git
remote is `origin`, its fetch and push URL are exactly
`https://github.com/Ahmad986Ferdaws/ecg-trust-lab.git`, and
`refs/remotes/origin/main` must equal X at child freeze and Y immediately
before the claim and after evaluation. A live `git ls-remote --symref` against
the exact HTTPS URL—not a symbolic local remote—must return exactly the HEAD
symref plus HEAD and `refs/heads/main` at that revision, with no other
advertised ref. The local tracking ref alone is not accepted as proof of push.
Output and claim paths must be absent.

At child freeze, immediately before the claim, and after evaluation, the
successor parent must be a tracked file whose exact `git show <revision>:<path>`
blob equals both the worktree bytes and the frozen SHA-256. At those same
boundaries, `git log --all --reflog --format=%H -- <exact protected glob
pathspecs>` must return empty output for the entire external raw-data tree,
both protocols' complete private-preflight trees, both output/claim namespaces,
all retained staging namespaces, and the isolated runtime-root namespace. This
proves absence of those protected pathnames from local reachable refs and
reflogs. It does not—and Git cannot—prove the content absence of unreachable
objects that are not named by those refs/reflogs; no stronger repository-wide
purge claim is made.

The implementation durably creates the armed marker, including parent-
directory persistence, and only then atomically creates the permanent adjacent
claim. A visibility witness consumes the one-shot immediately after the claim
entry appears; durable completion is a separate gate. The first waveform
sample may be decoded only after the claim entry and durability checks pass.

Visibility and durability are tracked separately. If the claim directory
entry becomes visible and a later temporary-entry removal or directory flush
fails, the one-shot is consumed and a postclaim failure receipt is required.
On POSIX the implementation uses file flush plus directory `fsync`; on Windows
it uses `FlushFileBuffers` on the file and a writable backup-semantics
directory handle. A filesystem that cannot establish the required guarantee
fails closed rather than being described as POSIX-durable.

The exact direct `artifacts/trust_sentinel` namespace-parent device/file
identity is captured before any protocol write and rechecked before and after
staging creation, claim link, output-root rename, terminal/failure link, and
each corresponding durability flush. The staging directory's device/file
identity is also captured before the claim.
These controls target accidental drift, persistent substitution, and ordinary
filesystem races under a trusted local operator. They do not claim resistance
to an active privileged local attacker capable of syscall-timed swap/restore
or modifying the self-checking code itself; no handle-relative ACL security
claim is made.
Immediately before and after the staging-root rename, terminal-manifest link,
or failure-receipt link, the implementation rechecks that same identity and
the exact owner-nonce marker. Existence of an output pathname is never treated
as ownership, and a foreign or substituted directory is never written. The
verifier records stable file IDs, sizes, timestamps, and SHA-256s for the
complete exact tree and adjacent claim before deep semantic checks, then
performs a second complete enumeration and hash afterward; any change or extra
member fails closed.

There is no retry or resume. A post-claim exception preserves all available
evidence and produces a sanitized failure receipt when possible; a later run
would need another protocol and namespace. On success, the complete
pre-manifest tree is verified and committed to its final output root first.
`success-manifest.json` is then atomically created as the terminal success
write and the complete bundle is independently reverified. That verifier is
not hash-only: it requires the exact live project root and bound 7-Zip tool,
reloads the canonical frozen parent and child, rechecks X/Y Git blobs and
module origins, rebuilds raw-source and archive/metadata closure, and re-runs
every selected raw adapter to require bit-exact equality with the private
canonical-signal shards before replaying quality, backbone, routing, endpoints,
and bootstrap evidence. A moved or self-consistently rebundled directory cannot
substitute for the canonical live lineage.

If the candidate manifest entry becomes visible but its directory durability
or the independent terminal reread fails, the run is
`AMBIGUOUS_TERMINAL_COMMIT`, not success. A sanitized failure receipt is kept
beside the candidate manifest, and the verifier rejects any root containing
both. A successful entrypoint returns only the independently reloaded result.

## Reporting boundary

Private evidence may contain record and patient roles, paths, scores,
embeddings, decisions, and bootstrap arrays. Public evidence is an explicit
aggregate allowlist and cannot contain identifiers, paths, waveforms, row
scores, embeddings, logits, probabilities, demographics, disease labels, or
replicate arrays. Unfavorable, insufficient, or infeasible evidence must be
retained.

Private audit evidence uses a fixed scalable layout. Complete quality reports
are split by inventory order into 53 canonical JSON shards of at most 256
records and 8 MiB each, with one self-hashed index binding contiguous ranges,
logical hashes, file hashes, and sizes. `record-evidence.json` contains only
the domain-separated report hash. Every adapter-success waveform is stored
privately as float32 `[12,1000]` in a 53-chunk, inventory-aligned, uncompressed
NPZ plus a self-hashed sidecar binding exact indices, record references, signal
hashes, and tensor hashes. Duplicate/case-colliding ZIP names, compression,
unsafe paths, excessive members, and expansion are rejected. The verifier
loads one bounded chunk at a time, re-runs the frozen quality policy, and
compares every metric, issue, boundary, status, and reason.

The bundle also contains manifest-covered exact copies of the parent, child,
sealed v1 aggregate result, v1 distribution policy, checkpoint, resolved model
config, and exact PTB normalization artifact. The public historical source
summary must equal, field for field, the source gate rederived from that copied
sealed v1 result; a merely self-consistent replacement point estimate,
interval, rate, count, or assignment digest is rejected. First and
repeated embeddings and logits are stored separately and must match. The
premanifest verifier first streams every raw record through its bound adapter a
second time, rechecks exact provenance and raw source hashes/sizes/official MD5,
and requires each float32 signal to be array-equal to the corresponding stored
inventory-index shard. It then reconstructs all quality-pass signals in exact
inventory order, applies the manifest-covered normalization through the same
Torch float32 path, and performs two full-backbone passes on the exact bound
CUDA runtime and batch-128 path. Both results must be array-equal to the stored
first and repeated embeddings and the model state must remain unchanged.

The declared two full embedding passes are the only stored scientific passes.
Raw-adapter reruns and any staged, preterminal, or postpublication full-model
replays are integrity checks under the same already-consumed one-shot claim:
they must reproduce the stored tensors and cannot create, replace, select, or
publish a new scientific result. They are not retries and do not create a new
claim. If no record passes quality, the bundle stores and verifies canonical
empty embedding, logit, probability, score, and decision arrays without
invoking CUDA inference on an empty batch.

The terminal verifier also reconstructs the frozen classifier head and replays
both embedding tensors through the same bound CUDA path. Both float64 transport
logit arrays must be exactly array-equal to that replay; CPU replay and numeric
tolerances are forbidden. It then recomputes scores, probabilities, entropy,
conformal decisions, routes, all
endpoint counts/rates/gates, and every bootstrap array and published quantile.
No exploratory secondary reports are produced: public aggregates are limited
to the four co-primary endpoints, hard gates, and all five route counts in the
frozen order with explicit zeros. The two-record Challenge indeterminate group
is not separately published.

`EXTERNAL_OOD_EVIDENCE_COMPLETE` requires all four defined endpoints and every
gate. A fully defined run that misses any target is retained as
`EXTERNAL_OOD_TARGET_MISSED`. A valid completed run with an empty endpoint
denominator omits only that undefined endpoint and its bootstrap array and is
terminally sealed as `EXTERNAL_OOD_INSUFFICIENT_EVIDENCE`; an undefined 0/0
rate is never converted to a success or ordinary target miss. Input, source-
hash, runtime, code-integrity, or implementation failures instead retain a
failure receipt and forbid a success manifest.

The strongest allowed claim is an untuned, retrospective stress test of one
frozen distribution policy and one frozen quality policy on two declared
external acquisition/population domains. It is research-only—not independent
replication, diagnostic or pediatric validation, unknown-disease detection,
clinical validation, evidence of safety, a medical device, or a deployment-
ready Trust Sentinel release.
