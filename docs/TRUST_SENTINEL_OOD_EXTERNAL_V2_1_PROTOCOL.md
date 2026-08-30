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

The first frozen successor implementation (`85b55d0f358e12052b23c8afa7468f2285342181`)
then stopped at its remote preflight because GitHub also advertised the
preexisting encrypted-v1-evidence backup tag. That refusal occurred before any
new-successor external-source archive, header/record metadata, waveform,
trained-checkpoint, or inference byte was accessed and created no successor
inventory, claim, or output. The preflight did reread already-frozen predecessor
metadata/inventory evidence plus Git/runtime metadata and imported project
source code. This amended parent permits the existing tag associated with the
separately documented, hash-pinned encrypted backup asset, using only its exact
tag name and pinned revision; every other unexpected remote ref remains
forbidden. This Git-ref check neither fetches nor authenticates the GitHub
release asset; asset integrity remains governed separately by
`TRUST_SENTINEL_PRIVATE_BACKUP.md` and is not an execution dependency.

The resulting tag-amended revision
`b5727c47dc719a8ec3d51deacad9936fd9df2a50` then refused twice at the same
pre-inventory remote boundary. The restricted execution could not reach
GitHub, and the network-enabled execution correctly rejected the private
repository when the sanitized Git process had no explicit credential helper.
Both refusals preceded any new-successor external-source archive,
header/record metadata, waveform, trained-checkpoint, or inference access and
created no successor inventory, child, claim, or output. They did reread the
already-frozen predecessor metadata/inventory evidence, Git/runtime metadata,
and imported project source code.

Before this second amendment was frozen, infrastructure-only probes established
that the exact four expected Git refs can be read with the already-bound Git
Credential Manager under a hardened, noninteractive command. Those probes
observed Git metadata only; they did not fetch or authenticate the release
asset. Git and GCM necessarily hold the credential in helper-process memory and
the HTTPS exchange. The launcher never supplies the secret through the URL,
argv, environment, Python inputs, configuration, logs, hashes, evidence, or
artifacts. Child stdout and stderr are treated as untrusted: their bytes may
enter Python memory before validation, but are checked against fixed size
limits, discarded, and never included in application errors, logs, hashes,
evidence, or artifacts.
Credential contents and authorization scope are operator-managed and
intentionally unbound; the code cannot prove least-privilege token scope.

The resulting private-auth revision
`6b6ddfd0e26c2c65265e7c128bafb3a13c0bf9a6`, whose parent YAML hash is
`sha256:9b0358be1d4a12ca1771c57d8387c1b332bbef5698e01d3da2707f59157a586c`
and whose parent was frozen at `2026-08-30T00:50:40Z`, then made one official
inventory-builder attempt. It refused before reading even the successor parent
protocol or any official external-source byte: the isolated runtime exposed
only `C:\Windows\System32` on `PATH`, while Git discovery still used
`shutil.which("git")`. No inventory, inventory-build authorization marker,
external-access armed marker, external one-shot claim, or successor output root
was created, and no trained checkpoint, waveform, inference, or scientific
result was accessed. X4 is a sole-parent operational amendment of that revision.
It fixes the exact Git resolver and strengthens the inventory-build boundary;
it does not change any cohort, preprocessing rule, model, threshold, endpoint,
gate, or scientific claim.

Frozen X4 revision `6b04c5c6308cfddd9a3b2b06f1ebbe24acc961e9`, whose
parent YAML hash is
`sha256:ac3653cd3a83d8d963531e54566487749c0faf03b5bc816ae66bdbde7f21927c`
and whose parent was frozen at `2026-08-30T02:00:56Z`, then entered its
production inventory command once. It refused inside runtime provenance before
the X4 authorization marker, any official source-content read, inventory,
projection, child, external claim, or output root existed. The first rejection
was caused by comparing legitimate frozen-module aliases against registry keys
instead of their canonical `FrozenImporter` specifications. Controls-only
engineering triage also exposed an empty `six.moves` dynamic namespace,
relative `torch.ops` and `torch.classes` namespace placeholders, native CPython
paths reported through the already-verified project alias junction, and two
host-injected security modules that required explicit binding. X4
authorization `x4_inventory_build_attempt_1` is therefore permanently retired
as `RETIRED_UNCONSUMED`; its marker path must remain absent.

X5 is a direct-child operational amendment. It validates canonical frozen
module identity and only three exact aliases, verifies dynamic namespaces
through a bound file-backed owner (including relative PyTorch placeholders),
maps the verified CPython alias to its exact resolved target for native-image
checks, and requires exact path, size, and
SHA-256 for the observed Norton and Defender security modules. It also adds a
repeatable `--preflight-only` branch that runs the exact shared
pre-consumption controls but cannot consume authorization, read official source
content, build an inventory, or write protocol artifacts. X5 changes no
scientific decision.

Frozen X5 revision `ff7c821e8b01e48e7e96fc29ddcec6e515286ddb`, whose
parent YAML hash is
`sha256:d4c3145985219fd65c9a5a4800773427cecd1f099b9e7ab75958596b7a995c61`
and whose parent was frozen at `2026-08-30T03:08:59Z`, then ran the
controls-only preflight. The isolated child completed the shared controls and
emitted `OOD_V2_INVENTORY_PREFLIGHT_VERIFIED`, stage `complete`, with
`authorization_consumed`, `official_source_content_accessed`, and
`protocol_artifact_written` all false. The complete launcher nevertheless
refused during outer runtime-root cleanup: exact bound GCM
`2.7.3+5fa7116896c82164996a609accd1c5ad90fe730a` had left one direct,
empty `temp/system-commandline-sentinel-files` directory. The verified inner
line did not make the overall nonzero launcher invocation successful. Neither
the X4 nor X5 authorization marker exists; no official external-source content,
trained checkpoint, inference, inventory, public projection, child contract,
armed marker, external claim, or successor output was created or accessed.
X5 authorization `x5_inventory_build_attempt_1` is therefore permanently
retired as `RETIRED_UNCONSUMED`, and its marker path must remain absent beside
the already-retired X4 path.

X6 is the sole direct-child operational amendment. Immediately after each
bound GCM-version or authenticated-remote Job runner returns and reports zero
active processes,
it opens only the case-exact sentinel with `CreateFileW`, directory backup and
open-reparse-point flags, delete/list/read-attributes access, and read sharing
only. While that restrictive handle remains open, X6 uses
`GetFileInformationByHandleEx` with `FileAttributeTagInfo` and 128-bit
`FileIdInfo` to verify a non-reparse directory, its stable volume/file
identity, and emptiness, then marks that same handle for deletion with
`SetFileInformationByHandle(FileDispositionInfo)`, closes it, and requires
temporary scratch to be exactly empty. A bounded-runner exception cannot
assert a returned process-closure result, so inner scratch cleanup is deferred
to the launcher after isolated-child exit. All four isolated launchers
independently apply the same handle-bound fallback after child exit. X6 issues
the new unconsumed authorization
`x6_inventory_build_attempt_1`; it changes no cohort, preprocessing rule,
model, policy, threshold, endpoint, hard gate, or scientific claim.

Frozen X6 revision `62f18d2ab4a20d8b588e97d8b6f93b95387996ca`, whose
parent YAML hash is
`sha256:5457ef7e773825523446d15e4f9f688f7c7006364c7843cd2d624dc2514fe11a`
and whose parent was frozen at `2026-08-30T04:40:56Z`, first passed the complete
controls-only preflight. That invocation consumed no authorization, accessed
no official source content, wrote no protocol artifact, and removed its owned
runtime root. The production invocation then durably created and consumed the
retained marker
`artifacts/trust_sentinel/.ood_external_v2_1.x6-inventory-build-attempt.json`.
Its file SHA-256 is
`sha256:4e3e968a2dc9f0c7f552bc05f8d70ef6afc99d97b5b81a60c2920e064efbe9e8`
and its logical artifact SHA-256 is
`sha256:88fb0a119f5c550f352cc2dca6f181567e0dd660449eb0ddd3c6247a7884cf93`.
The invocation failed after marker consumption and before creation of the
output parent. It created no private inventory, public projection, child
contract, armed marker, external one-shot claim, or output. It performed no
waveform sample decode, quality-policy execution, model inference, embedding,
distribution score, logit, probability, endpoint, or subgroup metric, and its
runtime cleanup succeeded.

Bounded post-failure forensic checks are disclosed as operational observations,
not scientific results: all 10 frozen source size, SHA-256, and declared MD5
bindings matched; Challenge contained 1,000 records; and ZZU contained 14,190
candidates, of which 12,328 were selected and 1,862 excluded. No waveform
sample was decoded. Absence of the output parent narrowed the failure to
archive closure or a later prewrite in-memory stage, but X6's generic stderr
made the exact stage unrecoverable. The X6 authorization is therefore
`CONSUMED_FAILED_RETAINED`; its exact ignored, untracked marker must remain
present permanently and cannot authorize a retry, resume, or reuse.

X7 is the sole direct-child operational amendment. It requires that exact X6
marker, adds an ordered stage tracker and an immutable sanitized failure
receipt, and issues one new authorization, `x7_inventory_build_attempt_1`, at
`artifacts/trust_sentinel/.ood_external_v2_1.x7-inventory-build-attempt.json`.
Any post-consumption failure attempts exactly one durable create-new write at
`artifacts/trust_sentinel/.ood_external_v2_1.x7-inventory-build-failure.json`.
That ignored, untracked receipt can disclose only frozen provenance and state
booleans, one allowlisted stage and its ordinal, one allowlisted output state,
and its logical self-hash; it contains no exception-derived data, timestamp,
path, external source or record identifier, waveform, or model output. Receipt
write failure does not restore authorization. X7 permits one consumption only;
any further attempt requires a future frozen amendment and a new authorization
ID. This operational amendment changes no scientific decision.

Frozen X7 revision `207fd1568697adb56991baeccee29ded38d3caf1`, whose
parent YAML hash is
`sha256:1da505b37d64dec804f147fa8cfd43a5029fe2ee7d92d1666177d490ea7016e1`
and whose parent was frozen at `2026-08-30T06:34:49Z`, then consumed its one
production authorization. The retained X7 marker has file SHA-256
`sha256:8255b58e5c63a4e18ae2a0b7715109106e4f1a949cb68204b66cde9f1fd4af01`
and logical artifact SHA-256
`sha256:ec01f1554da7733a4d298161a3d67818dc1edd2fefb663760277070359354830`.
The immutable failure receipt has file SHA-256
`sha256:af6995828daad64a6606dfd1875a2ced6daa9ac390e328152488270f5dcffac6`
and logical artifact SHA-256
`sha256:02c2d212a1ff4108c9dd10bd67095a94727d5aabb68e0c1e94a2ed9d4304d7d3`.
It reports stage `zzu_archive_listing`, zero-based ordinal 8, official source
content accessed, and output state `NONE`. Runtime cleanup succeeded. No
private inventory, public projection, child, external-access marker, external
one-shot claim, waveform sample, quality result, embedding, distribution
score, logit, probability, endpoint, or subgroup metric was created or
observed. Both X7 artifacts are permanently retained, ignored, untracked, and
cannot authorize a retry, resume, or reuse.

Post-failure review used only frozen code and source-free synthetic archives.
It found that the already-bound ZZU terminal `.zip` operand remained project-relative
when the isolated 7-Zip runner changed to its separate fresh working directory;
the relative operand was therefore interpreted from the wrong directory. This
is an operational path-presentation defect, not evidence about archive members.
With exact 7-Zip 26.02, a standards-compliant synthetic two-volume archive with
14,190 synthetic records, 28,380 regular files, and 42,586 total listed entries
produced 12,957,423 bytes of listing output (19.31% of the 64 MiB bound) in
0.34 seconds and parsed in 1.04 seconds with empty stderr. A small synthetic
split archive emitted the exact `Multivolume = +` and `Volumes = 2` markers.
These are bounded engineering observations only; they accessed no official
source or identifier and support no scientific claim.

X8 is the sole direct-child operational amendment. After both exact direct ZZU
split-archive paths have already been bound, it converts the terminal `.zip`
operand used by every isolated 7-Zip listing, test, and extraction command to
that same file's exact absolute direct path before process creation. Direct
ancestry, regular-file status, adjacency, stable identity, and pre/post hashes remain
required; the isolated empty working directory, two-file tool replica, minimal
environment, safe member parser, and every archive/extraction comparison remain
unchanged. No absolute user path is serialized or published. X8 requires the
exact retained X6 marker plus the exact X7 marker and failure receipt, and
issues `x8_inventory_build_attempt_1` at
`artifacts/trust_sentinel/.ood_external_v2_1.x8-inventory-build-attempt.json`.
Any consumed X8 failure attempts one immutable sanitized receipt at
`artifacts/trust_sentinel/.ood_external_v2_1.x8-inventory-build-failure.json`.
The X8 authorization permits one consumption only. Once its marker first becomes
visible, success, failure, or receipt-write failure permanently terminates it,
and any further build requires a new frozen amendment and authorization ID. X8
changes no archive byte, member,
role, cohort, selection, preprocessing rule, model, policy, threshold,
endpoint, hard gate, or scientific claim.

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
metadata result is 12,328 selected records from 10,350 patients. The exact,
ordered, mutually exclusive exclusion vector is: 1,856 with the upstream
`pediatric_12_lead` flag false, 6 under ten seconds, 0 with a non-500 Hz rate,
0 with a header lead count other than 12 after the flag check, and 0 with a
noncanonical lead set. These counts sum with the selected records to all
14,190 candidates. The flag-false records are not counted again under a later
lead-count reason.

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
projection; no samples are decoded. At the Windows `7z -slt` presentation
boundary only, an all-backslash member name is normalized to forward slashes
before the shared canonical path validator runs. Mixed separators, absolute,
rooted, drive, UNC, device, traversal, dot, empty, trailing, reserved-name, or
control-character components and exact or case-folded collisions remain
forbidden. Stored archive and evidence paths remain canonical POSIX paths; the
normalization does not broaden any other path contract.

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
The only recognized GCM scratch side effect is the case-exact relative
directory `temp/system-commandline-sentinel-files`. Immediately after the
bound GCM-version Job runner and each authenticated private-remote Job runner
return and report zero active processes, absence is accepted; otherwise the
pipeline opens that
exact path with `CreateFileW`. Desired access is exactly `DELETE |
FILE_LIST_DIRECTORY | FILE_READ_ATTRIBUTES`, disposition is `OPEN_EXISTING`,
and flags are exactly `FILE_FLAG_BACKUP_SEMANTICS |
FILE_FLAG_OPEN_REPARSE_POINT`. Share mode is `FILE_SHARE_READ` only: omission
of `FILE_SHARE_WRITE` and `FILE_SHARE_DELETE` denies concurrent mutation,
deletion, or rename of the opened sentinel object while the handle is held. No
ancestor-immutability claim is made; direct ancestry is instead revalidated
under that lock.

`GetFileInformationByHandleEx(FileAttributeTagInfo)` must report
`FILE_ATTRIBUTE_DIRECTORY` and must not report
`FILE_ATTRIBUTE_REPARSE_POINT`.
`GetFileInformationByHandleEx(FileIdInfo)` records the 64-bit
`VolumeSerialNumber` and exact 128-bit `FileId.Identifier`. The pipeline
revalidates direct ancestry and verifies sentinel emptiness while the main
handle remains open. It then opens an identity-only pathname witness with
`FILE_LIST_DIRECTORY | FILE_READ_ATTRIBUTES`, all three share flags,
`OPEN_EXISTING`, and the same backup/open-reparse-point flags. The witness must
have exactly the initial main handle's attributes and volume/file identity and
is closed before the main handle is queried again; that final main result must
also match exactly.

The pipeline calls `SetFileInformationByHandle` with `FileDispositionInfo` and
`DeleteFile` true on the main handle that performed the locked verification;
pathname reopening for deletion is forbidden. It then calls `CloseHandle`,
requires the sentinel pathname to be absent and non-indirect, and requires all
temporary scratch to be exactly empty. A failed API call, nonempty, reparse,
differently named, additional, identity-changed, or raced entry fails closed
and retains the runtime root; the runtime scratch verifier has no sentinel
allowlist. If a bounded runner raises instead of returning its proven
process-closure result, inner deletion is not attempted and cleanup is deferred
until the isolated child has exited. After isolated-child exit, each of the
inventory, freeze, evaluate,
and verify launchers performs the same handle-bound fallback for abrupt failure
paths. Recursive or wildcard deletion is forbidden. Overall launcher success
requires both child exit code zero and complete removal of the owned runtime
root; an inner verified report alone is insufficient. An outer cleanup attempt
is never retried: the `finally` fallback is reserved for a child-launch path
that exited before any outer cleanup attempt began, so a cleanup refusal
retains its runtime root.
`CUDA_CACHE_DISABLE=1` is exact and mandatory. `TORCHINDUCTOR_CACHE_DIR` is
bound directly to the existing isolated runtime-temp role so eager PyTorch
initialization cannot create an unbound `torchinductor_*` child. A dirty-scratch failure must
still be able to publish a sanitized failure receipt; receipt retention is not
conditioned on scratch emptiness.

Before the inventory builder reads any official archive, metadata, header, or
inventory byte, a metadata-only preflight requires the exact frozen parent,
clean pushed implementation revision X8, live remote, protected-path history,
tracked parent and complete source blobs, isolated runtime/Git identity,
bound `__main__`, and absent successor claim/output. It also binds the exact
parent schema, cohort and exclusion-count invariants, canonical dataset and raw
source paths/keysets, declared raw size/SHA-256/MD5 values, and exact 7-Zip
tool. X4 and X5 marker paths must remain absent; the retained X6 marker must
match its exact canonical bytes, file hash, and logical hash. The retained X7
marker and failure receipt must likewise match their exact canonical bytes,
file hashes, logical hashes, stage 8, source-access truth, and `NONE` output
state. The X8 marker, X8 failure receipt, and private/public destinations must
be absent. After those checks but before the first official source byte, the
builder must durably create, without overwrite,
`artifacts/trust_sentinel/.ood_external_v2_1.x8-inventory-build-attempt.json`.
That marker authorizes exactly one fresh X8 preclaim inventory build, remains
ignored and untracked, contains no source identifiers or model output, and is
distinct from the consumed X6 and X7 markers and the later waveform one-shot
claim. The first permitted raw action is again to hash every official source
against the frozen parent. Before reporting success, a postflight repeats the
preflight, all three marker proofs, the X7 failure-receipt proof, and strict-
reloads the exact canonical private inventory and public projection, matching
physical and logical hashes to the in-memory build.

The same exact argument contract can first be run with `--preflight-only`.
That repeatable branch validates the runtime, Git, parent, namespace, schema,
raw-path metadata, 7-Zip binding, absent X4/X5 markers, the exact retained X6
marker, the exact retained X7 marker and failure receipt, absent X8 marker and
failure receipt, and absent private/public destinations, then returns fixed
path-free JSON. It never calls authorization consumption, official-source
hashing, inventory construction, writers, or postflight. Production reuses the
same shared control function and immediately rechecks every marker and
destination condition before atomically consuming X8. Its inner report counts
as an overall pass only when the outer process returns zero and removes the
runtime root.

Beginning with X8 authorization publication, the builder records entry to
exactly these stages in this order: `authorization_publication`,
`raw_source_binding_verification`, `expectation_materialization`,
`challenge_inventory`, `zzu_metadata_parse_and_counts`,
`zzu_header_selection_and_counts`, `challenge_archive_closure`,
`zzu_tool_resolution`, `zzu_archive_listing`, `zzu_archive_test`,
`zzu_evaluated_tree_snapshot`, `zzu_isolated_extraction`,
`zzu_archive_comparison`, `archive_closure_role_validation`,
`inventory_assembly_and_reverification`,
`public_projection_build_and_verify`, `canonical_serialization`,
`precommit_inventory_reverify`, `output_transaction`,
`output_reload_and_verify`, and `postflight`. A post-consumption failure writes
exactly one schema-1 receipt with state `PRECLAIM_INVENTORY_BUILD_FAILED` and
the zero-based ordinal of the last entered allowlisted stage. Its `output_state`
is exactly one of `NONE`, `PRIVATE_ONLY`, `PUBLIC_ONLY`, `BOTH`, or
`UNVERIFIABLE`.

The authorization-publication stage is entered before the create-new marker
operation. A separate witness turns `official_source_content_accessed` true
only immediately after the first official source file is successfully opened
and before its first content read for hashing; publication or open failures
before that witness therefore remain factually source-free in the immutable
receipt.

The inventory output transaction creates each missing directory one level at
a time and durably flushes the directory that owns each new entry. It then
flushes every exact leaf parent after destination-link publication and
temporary-entry removal; rollback deletions receive the same durability gate.

The receipt's exact top-level fields are `artifact_type`,
`authorization_consumed`, `authorization_id`,
`authorization_marker_artifact_sha256`, `authorization_marker_file_sha256`,
`contains_external_source_bytes_or_identifiers`,
`contains_model_outputs_embeddings_or_scores`,
`external_one_shot_claim_consumed`,
`failure_requires_new_frozen_amendment_and_authorization_id`, `failure_stage`,
`failure_stage_ordinal`, `implementation_revision`,
`official_source_content_accessed`, `output_state`,
`parent_config_file_sha256`, `protocol_id`,
`quality_model_score_logit_probability_or_metric_observed`,
`retry_resume_or_reuse_authorized`, `schema_version`, `state`,
`waveform_sample_decode_occurred`, and `artifact_sha256`. Canonical serialization
and the logical self-hash make it immutable. Exception class, message,
traceback, errno, subprocess output, timestamps, paths, archive/member/file/
record/patient identifiers, source bytes, waveform observations, and model or
metric outputs are forbidden. Failure to create the receipt is itself reported
only as a boolean and never permits the X8 authorization to be reused.

The child binds the X6, X7, and X8 markers' exact canonical paths, file SHA-256
values, and logical self-hashes, plus the X7 failure receipt's exact path and
physical and logical hashes. At Y, the verifier reconstructs its expected X8
bytes from the parent hash, child-bound X8/source/Python/Git identities, and
frozen source-bound authorization constants without requiring HEAD to return
to X8. It also requires the disclosed exact X6 and X7 markers and X7 receipt
and rechecks all four proofs during initial input verification, immediately
before the external claim, after evaluation, and during terminal bundle
verification.

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
site tree fails closed. Frozen entries are verified through their canonical
`FrozenImporter` specification and canonical registry identity; only
`importlib._bootstrap→_frozen_importlib`,
`importlib._bootstrap_external→_frozen_importlib_external`, and
`os.path→ntpath` may use alias keys. An empty dynamic namespace is accepted
only when a bound file-backed owner such as the exact installed `six.py`
authenticates it. On the uv Windows runtime, `sys.executable` is the
separately hash-bound venv redirector while PSAPI must observe the exact bound
base-tree `python.exe` as the process image; the redirector is not falsely
required to remain loaded. Native paths reported through the one verified
CPython alias junction are mapped to the corresponding exact resolved-target
file before ancestry checks. The loaded host-security set must also contain
exactly the bound Norton `aswAMSI.dll` and Defender `MpOav.dll` paths, sizes,
and SHA-256 hashes; an absent, relocated, changed, or additional unbound
injected image fails closed.

Every tracked `src/ecg_trust/**/*.py` file and all four inventory/freeze/
evaluate/verify entrypoints are bound by exact path, size, SHA-256, worktree
bytes, and X8/Y Git blobs. Git resolves only from the exact direct Windows
installation root `C:\Program Files\Git`: the `cmd\git.exe` launcher and
`mingw64\bin\git.exe` binary are derived from that root without `PATH` lookup,
then their bytes, direct ancestry, and the full `mingw64` runtime tree are
verified. It is run with a sanitized environment, explicit Git/worktree
directories, disabled replacement/config redirection, and a strict local-
config allowlist. An anonymous query first clears every credential helper and
must return code 128, empty stdout, and the exact 40-byte ASCII Git denial
`fatal: unable to get password from user\n`; the stderr is byte-compared and
discarded without decoding or disclosure. DNS, TLS, transport, and other Git
failures therefore cannot substitute for credential denial. The authenticated
query must then succeed against the same hardcoded URL and return the exact
frozen refs. This denial-then-success pair is the runtime proof that the
endpoint is not publicly Git-readable. Only the authenticated query may invoke the exact
`git-credential-manager.exe` 2.7.3 binary already inside the bound tree. Its
command-local configuration first clears all helpers, then pins GCM, Windows
Credential Manager storage, the default `git` namespace, GitHub provider,
public account name, noninteractive/no-GUI operation, HTTPS certificate
verification, no redirects, and disabled trace/secret-trace/debug settings.
An exact nonsecret GCM environment redundantly pins those security-critical
settings above registry defaults. Standard input is `NUL`; stdout must be the
exact raw ref advertisement and stderr must be empty. There is no authentication
fallback. Each GCM or remote process is created with raw Windows
`CreateProcessW` plus `STARTUPINFOEX`: `PROC_THREAD_ATTRIBUTE_JOB_LIST`
atomically places it in an unnamed Job Object before it can run, while
`PROC_THREAD_ATTRIBUTE_HANDLE_LIST` permits exactly `NUL`, the stdout-pipe
writer, and the stderr-pipe writer to cross the process boundary. The Job uses
`JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`, forbids descendant breakaway, and must
report zero active processes after cleanup. Concurrent `ReadFile` drains cap
each Git-remote stream at 4,096 bytes and each GCM-version stream at 256 bytes.
The GCM and remote caller timeouts are 30 and 60 seconds; timeout, overflow, or
reader failure terminates the whole Job tree, verifies cleanup within 10
seconds, overwrites the mutable capture bytearrays, and raises a generic
context-free error. Normally completed stream bytes are released after exact
semantic validation without serialization. An interrupt such as `Ctrl+C` or
any other `BaseException` is normalized only after Job cleanup and buffer
erasure into the same fresh generic integrity error. No temporary capture files
are used. Windows Credential Manager and the CLR are explicitly trusted OS
secret brokers, not falsely claimed as part of the path-free `mingw64` hash
closure. The NVIDIA query binds
exact `nvidia-smi.exe`, `nvml.dll`, and `nvcuda.dll` bytes and driver 596.49. The
exact 7-Zip 26.02 executable and library are copied into a fresh two-file
application directory for every call; a separate empty cwd and minimal `PATH`
prevent adjacent codecs, formats, plugins, caller-directory DLLs, or caller
environment from executing. Before process creation, X8 converts the already-
bound direct ZZU terminal `.zip` operand to its exact absolute direct path and
passes that same normalized operand to listing, archive test, and isolated extraction.
The archive operand is rechecked as a stable regular file under its exact direct
ancestry; no path is stored in public evidence. All runtime trees and tools are
recomputed at child freeze, immediately before the claim, after evaluation, and
during terminal semantic verification. Absolute user paths are never published.

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

The parent and exact child must be committed and pushed before access. The first
frozen successor revision `85b55d0...`, tag-amended revision `b5727c4...`,
private-auth revision `6b6ddfd...`, X4, and X5 form four consecutive sole-parent
amendments; X6 extends that chain to five, X7 to six, and X8 to seven. The
first two amendments each modify the same seven frozen paths.
Historical X4 has sole parent `6b6ddfd0e26c2c65265e7c128bafb3a13c0bf9a6`
and modifies all and only these eleven existing paths, each with Git status
`M`:

- `configs/trust_sentinel_ood_external_v2_1.yaml`;
- `docs/TRUST_SENTINEL_OOD_EXTERNAL_V2_1_PROTOCOL.md`;
- `scripts/build_trust_sentinel_ood_v2_inventory.py`;
- `src/ecg_trust/ood_v2/inventory.py`;
- `src/ecg_trust/ood_v2/models.py`;
- `src/ecg_trust/ood_v2/pipeline.py`;
- `tests/unit/test_ood_v2_inventory.py`;
- `tests/unit/test_ood_v2_inventory_cli.py`;
- `tests/unit/test_ood_v2_models.py`;
- `tests/unit/test_ood_v2_pipeline.py`; and
- `tests/unit/test_ood_v2_protocol_closure.py`.

X5 has exact sole parent `6b04c5c6308cfddd9a3b2b06f1ebbe24acc961e9`
and modifies all and only these nine existing paths, also with Git status
`M`:

- `configs/trust_sentinel_ood_external_v2_1.yaml`;
- `docs/TRUST_SENTINEL_OOD_EXTERNAL_V2_1_PROTOCOL.md`;
- `scripts/build_trust_sentinel_ood_v2_inventory.py`;
- `src/ecg_trust/ood_v2/models.py`;
- `src/ecg_trust/ood_v2/pipeline.py`;
- `tests/unit/test_ood_v2_inventory_cli.py`;
- `tests/unit/test_ood_v2_models.py`;
- `tests/unit/test_ood_v2_pipeline.py`; and
- `tests/unit/test_ood_v2_protocol_closure.py`.

X6 has exact sole parent `ff7c821e8b01e48e7e96fc29ddcec6e515286ddb`,
whose parent YAML hash is
`sha256:d4c3145985219fd65c9a5a4800773427cecd1f099b9e7ab75958596b7a995c61`
and whose parent was frozen at `2026-08-30T03:08:59Z`. It modifies all and only
these thirteen existing paths, each with Git status `M`:

- `configs/trust_sentinel_ood_external_v2_1.yaml`;
- `docs/TRUST_SENTINEL_OOD_EXTERNAL_V2_1_PROTOCOL.md`;
- `scripts/build_trust_sentinel_ood_v2_inventory.py`;
- `scripts/evaluate_trust_sentinel_ood_external_v2.py`;
- `scripts/freeze_trust_sentinel_ood_external_v2.py`;
- `scripts/verify_trust_sentinel_ood_external_v2.py`;
- `src/ecg_trust/ood_v2/models.py`;
- `src/ecg_trust/ood_v2/pipeline.py`;
- `tests/unit/test_ood_v2_cli.py`;
- `tests/unit/test_ood_v2_inventory_cli.py`;
- `tests/unit/test_ood_v2_models.py`;
- `tests/unit/test_ood_v2_pipeline.py`; and
- `tests/unit/test_ood_v2_protocol_closure.py`.

X7 has exact sole parent `62f18d2ab4a20d8b588e97d8b6f93b95387996ca`,
whose parent YAML hash is
`sha256:5457ef7e773825523446d15e4f9f688f7c7006364c7843cd2d624dc2514fe11a`
and whose parent was frozen at `2026-08-30T04:40:56Z`. It modifies all and only
these eleven existing paths, each with Git status `M`:

- `configs/trust_sentinel_ood_external_v2_1.yaml`;
- `docs/TRUST_SENTINEL_OOD_EXTERNAL_V2_1_PROTOCOL.md`;
- `scripts/build_trust_sentinel_ood_v2_inventory.py`;
- `src/ecg_trust/ood_v2/inventory.py`;
- `src/ecg_trust/ood_v2/models.py`;
- `src/ecg_trust/ood_v2/pipeline.py`;
- `tests/unit/test_ood_v2_inventory.py`;
- `tests/unit/test_ood_v2_inventory_cli.py`;
- `tests/unit/test_ood_v2_models.py`;
- `tests/unit/test_ood_v2_pipeline.py`; and
- `tests/unit/test_ood_v2_protocol_closure.py`.

X8 has exact sole parent `207fd1568697adb56991baeccee29ded38d3caf1`,
whose parent YAML hash is
`sha256:1da505b37d64dec804f147fa8cfd43a5029fe2ee7d92d1666177d490ea7016e1`
and whose parent was frozen at `2026-08-30T06:34:49Z`. It modifies all and only
these nine existing paths, each with Git status `M`:

- `configs/trust_sentinel_ood_external_v2_1.yaml`;
- `docs/TRUST_SENTINEL_OOD_EXTERNAL_V2_1_PROTOCOL.md`;
- `src/ecg_trust/ood_v2/inventory.py`;
- `src/ecg_trust/ood_v2/models.py`;
- `src/ecg_trust/ood_v2/pipeline.py`;
- `tests/unit/test_ood_v2_inventory.py`;
- `tests/unit/test_ood_v2_models.py`;
- `tests/unit/test_ood_v2_pipeline.py`; and
- `tests/unit/test_ood_v2_protocol_closure.py`.

Every historical parent byte and exact diff is independently verified. X8 and
child-freeze execution revision Y are also consecutive: Y has X8 as its sole
parent, exactly one commit lies in `X8..Y`, and that commit adds only the
tracked child plus its aggregate public inventory projection. The private
inventory and durable X6, X7, and X8 inventory-build markers remain ignored and
untracked; the exact X7 failure receipt remains present, the X8 failure receipt
must be absent on success, and the retired X4 and X5 markers remain absent.
The only Git remote is `origin`, its fetch and push URL are exactly
`https://github.com/Ahmad986Ferdaws/ecg-trust-lab.git`, and
`refs/remotes/origin/main` must equal X8 at child freeze and Y immediately
before the claim and after evaluation. A live `git ls-remote --symref` against
the exact HTTPS URL—not a symbolic local remote—must return exactly the HEAD
symref, HEAD and `refs/heads/main` at that revision, plus the required pinned
`private-evidence-backup-v1-2026-08-29` tag at
`a88ef86e8e0b28dd6f162cda88e16b4159d195d8`. Every other advertised ref is
forbidden. The tag must remain a lightweight direct-commit ref with no peeled
line; its local object must be a commit and an ancestor of the current X or Y,
so it adds no protected-history reachability beyond `main`. The local tracking
ref alone is not accepted as proof of push. Because the repository is private,
the authenticated query uses only the frozen noninteractive GCM/WinCred
boundary above. Its explicit `credential.useHttpPath=false` lookup is
host/account-scoped even though the network target is the one exact repository
URL; credential contents and authorization scope remain operator-managed and
runtime-unverifiable. The preceding visibility probe and all other Git
operations remain credentialless. Output and claim paths must be absent.

Shallow repository state (`.git/shallow` or `.git/shallow.lock`), object
alternates, grafts, replacement refs, sparse checkout, or linked-worktree
configuration are forbidden because they can hide or reinterpret reachable
history.

At child freeze, immediately before the claim, and after evaluation, the
successor parent must be a tracked file whose exact `git show <revision>:<path>`
blob equals both the worktree bytes and the frozen SHA-256. At those same
boundaries, `git log --full-history --all --reflog --format=%H -- <exact protected glob
pathspecs>` must return empty output for the entire external raw-data tree,
both protocols' complete private-preflight trees, both output/claim namespaces,
the retired X4 and X5, retained X6 and X7 artifacts, current X8 inventory-build
marker and X8 failure-receipt paths, all retained staging namespaces, and
the isolated runtime-root namespace. This
proves absence of those protected pathnames from local reachable refs and
reflogs. It does not—and Git cannot—prove the content absence of unreachable
objects that are not named by those refs/reflogs; no stronger repository-wide
purge claim is made.

No durable inventory-build authorization is the external one-shot claim: each
is consumed before metadata inventory source-byte access and none authorizes
waveform decoding, quality, model, score, or endpoint access. X6 and X7 are
consumed-failed and permanently retained; no retry, resume, or marker/receipt
reuse is allowed. X8 authorizes one fresh build only. Once X8 is consumed,
success or a sanitized immutable failure receipt terminates that authorization;
receipt write failure also leaves it consumed. Any further build requires a
future frozen amendment and new authorization ID. The X4 and X5 authorizations
were never consumed and are permanently retired; they cannot be revived or
substituted for X6, X7, or X8.

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

After the external claim first becomes visible, the no-retry rule is absolute:
there is no operator, failure-mode, resume, reuse, or second-scientific-pass
exception. A post-claim exception preserves all available evidence and
produces a sanitized failure receipt when possible; a later run would need
another protocol and namespace. The preregistered deterministic integrity
replays remain verification inside that same consumed claim, not retries. On
success, the complete
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
