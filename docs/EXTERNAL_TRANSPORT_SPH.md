# Frozen SPH external-transport protocol

**Protocol ID:** `sph-external-transport-v1`
**Frozen configuration:** `configs/external_transport_sph_frozen.yaml`
**Status:** frozen before the first SPH model inference
**Scope:** exploratory external transport stress test; **not clinical validation**

## 1. Question and interpretation boundary

This experiment asks a narrow question: how do the six already-frozen PTB-XL
models behave when transported, without adaptation, to a different hospital's
ECGs after a conservative cross-ontology label mapping?

It does not retrain a model, choose a model, adjust a probability, fit a
threshold, tune a coverage gate, or change a reporting rule using SPH. It is a
retrospective engineering stress test of transport behavior. It is not a
prospective study, a medical-device evaluation, a safety study, or evidence
that the system is fit for diagnosis or patient care.

All three analysis cohorts below are selected from one model-inference pass.
Their masks and expected counts are frozen before inference so no observed SPH
model result can determine which cohort is reported.

## 2. Primary-source provenance and license

SPH is the Shandong Provincial Hospital dataset described by Liu et al. in
Scientific Data. The authors report 25,770 clinical 12-lead ECGs from 24,666
patients, collected from August 2019 through August 2020, at 500 Hz with record
lengths of 10-60 seconds. The signals are stored in millivolts, and the fixed
lead order is `I, II, III, aVR, aVL, aVF, V1-V6`.

Primary sources:

- [Scientific Data paper](https://doi.org/10.1038/s41597-022-01403-5)
- [Figshare collection](https://doi.org/10.6084/m9.figshare.c.5779802.v1)
- [Metadata item](https://doi.org/10.6084/m9.figshare.17912441.v1)
- [ECG-record item](https://doi.org/10.6084/m9.figshare.17912444.v1)
- [Diagnostic-dictionary item](https://doi.org/10.6084/m9.figshare.17912507.v1)
- [Translation-rule item](https://doi.org/10.6084/m9.figshare.19738468.v1)

Each Figshare item is marked **CC0**. Citation remains scientifically
appropriate even though CC0 does not require attribution. CC0 does not turn
this retrospective experiment into clinical validation and does not remove
the need for ethical, safety, regulatory, or institutional review for any
future clinical use.

The raw files remain under the Git-ignored
`data/raw/sph/figshare-v1/` directory and are not added to the repository.

| Official file | Bytes | Figshare MD5 | Local SHA-256 |
|---|---:|---|---|
| `metadata.csv` | 999,219 | `8ef8e8789059a585c89c5302c7891909` | `c6f8dc5197758a3970aae554f8b6ee884d96656b121879bc0eb1adf367ea6d86` |
| `code.csv` | 1,839 | `9fafec6cc6e1ef8117e2964116654d05` | `4d22759fd19d37133c78d439a5726f9e0bc412cc17483c424926ce9abb33c8fc` |
| `rule.pdf` | 47,063 | `8e84772f206d50f0e661c787cd4c905a` | `3991b608fa1f9ab470a934a6437352ebd0080c9dd066dc5085cb155208dfec68` |
| `records.tar.gz` | 2,281,799,680 | `0beef976f54cf0a95fac9f45bb157950` | `cebad89d7d25663272eeb45545712d76302d334bb107cdc23fd15390c9194e55` |

The local archive audit found 25,770 unique `Axxxxx.h5` members, rejected path
traversal, and extracted 25,770 files. The extracted filename set matches all
25,770 `ECG_ID` values in `metadata.csv`, with no missing or extra records.

## 3. Frozen exact-10-second cohort flow

The official paper establishes 500 Hz sampling. Therefore `N == 5000` in the
official metadata is exactly 10 seconds; duration is not estimated from a
signal or rounded.

Counts below were reproduced independently from `metadata.csv` using exact
integer AHA tokens. Record and patient counts overlap across labels because the
task is multi-label.

| Frozen stage | ECGs | Patients | Meaning |
|---|---:|---:|---|
| Complete SPH metadata | 25,770 | 24,666 | Official source population |
| `broad_exact10` | 18,842 | 18,157 | Every row with `N == 5000` |
| `primary_mapped` | 15,698 | 15,193 | Exact 10 seconds and at least one direct target mapping |
| `no_ambiguous_mapped` | 15,563 | 15,066 | Primary cohort after removing any record carrying a frozen ambiguous primary code |

The 3,144 broad-cohort records with no direct target mapping are not treated as
verified negatives in the primary analysis. Their AHA statements are outside
the frozen five-class map, so their five targets are unknown under this
cross-ontology contract. They receive operational all-zero targets only in the
pre-specified `broad_exact10` mapping-sensitivity analysis, where this
limitation must accompany every result.

The primary positive counts are:

| Target | Positive ECGs | Positive patients |
|---|---:|---:|
| NORM | 11,172 | 10,874 |
| MI | 138 | 131 |
| STTC | 3,030 | 2,947 |
| CD | 1,510 | 1,453 |
| HYP | 113 | 110 |

The no-ambiguous sensitivity removes 135 ECGs from 134 patients. Its positive
counts are NORM 11,172/10,874 patients, MI 131/124, STTC 2,981/2,899, CD
1,470/1,417, and HYP 64/63. The large relative change for HYP is itself a
warning that ontology choices materially affect that rare endpoint.

Exact-token parsing finds zero rows with both NORM and a directly mapped
abnormal superclass. The source value `1;1` occurs once and is a duplicated
NORM statement, not a NORM-abnormal conflict. A nonzero conflict count at run
time is therefore an integrity failure, not a post-hoc invitation to relabel.

## 4. Frozen direct label map

Semicolons separate diagnostic statements. Within each statement, split the
exact integer tokens on `+`: a non-modifier is a code below 200 or at least
500, while a modifier is in `[200, 500)`. Exactly one non-modifier must be
present, independent of token order; otherwise parsing fails closed. Only that
non-modifier can map to a target. Modifiers and duplicate statements remain in
private provenance. Set-membership mapping is idempotent, so a duplicate such
as `1;1` does not change the target vector.

| PTB-XL target | Direct SPH AHA primary codes | Official SPH descriptions |
|---|---|---|
| NORM | `1` | Normal ECG |
| MI | `160, 161, 165, 166` | Anterior, inferior, anteroseptal, or extensive anterior MI |
| STTC | `145, 146, 147, 148` | ST deviation; ST deviation with T-wave change; T-wave abnormality; prolonged QT |
| CD | `83-88, 101, 102, 104-106, 108` | Second/advanced/complete AV block; fascicular or bundle-branch block; ventricular preexcitation |
| HYP | `140, 142, 143` | Left atrial enlargement; left or right ventricular hypertrophy |

This is intentionally a small semantic intersection, not an attempt to
translate all 44 SPH primary statements into PTB-XL. The official PTB-XL
statement dictionary places long QT in STTC and left atrial
overload/enlargement in HYP, which anchors the otherwise non-obvious mappings
of SPH codes 148 and 140.

The following primary codes are explicitly ambiguous and unmapped:

| Code | SPH description | Reason not mapped |
|---:|---|---|
| 80 | Short PR interval | Interval observation is not treated as a PTB CD diagnosis |
| 81 | AV conduction ratio N:D | Context-dependent conduction description |
| 82 | Prolonged PR interval | No unreviewed inference to first-degree AV block |
| 152 | TU fusion | Does not cleanly identify one frozen superclass |
| 153 | ST-T change due to ventricular hypertrophy | Compound etiology would require adjudicating STTC versus HYP |
| 155 | Early repolarization | No direct frozen PTB superclass equivalence |

Every other unlisted SPH code is also unmapped. Absence from the map means
“outside this transport label contract,” not “disease absent.”

### No new expert adjudication

The SPH publication describes cardiologist review of the original dataset and
its Chinese-to-AHA translation rules. That source review must not be confused
with review of our new AHA-to-PTB-superclass bridge. No cardiologist or other
clinical expert reviewed or adjudicated this cross-ontology map. No SPH label
will be changed after model outputs are observed. Any future expert-reviewed
map must be a separately versioned experiment, not a revision of this run.

## 5. Signal and preprocessing contract

Each selected HDF5 file must contain a root dataset named `ecg`, with a finite
numeric array of shape exactly `[12, 5000]`. Automatic transposition, lead
reordering, missing-lead imputation, record cropping, or amplitude rescaling is
forbidden. The official millivolt values and lead order are taken unchanged.

The only domain-format conversion is deterministic resampling with
`scipy.signal.resample_poly(up=1, down=5, axis=1)`, producing a contiguous
`float32` array of shape `[12, 1000]` at 100 Hz.

After resampling, the exact PTB training-fold normalization is applied:

- path: `artifacts/preprocessing/ptbxl_v1.0.3_train_folds_1-7_normalization.json`
- SHA-256: `4a6cb489098361d8221403c14871c242672c346975af3a07f731ceac97264363`

SPH statistics are never used to center, scale, normalize, recalibrate, or
otherwise adapt the inputs or outputs. This deliberately measures transport
under the original PTB preprocessing contract rather than hiding a domain
shift through target-domain preprocessing.

Before model inference or outcome calculation, perform a descriptive scale
smoke check on 256 primary-cohort records. Sort by `ECG_ID` and select indices
`numpy.linspace(0, n - 1, 256, dtype=int)`. The rounded absolute-amplitude
references from that sample are median 0.042047 mV, p95 0.311201 mV, p99
0.864611 mV, and maximum 58.506 mV. These values document the observed unit
scale; they are not acceptance limits. Finite outliers are retained, with no
clipping, rejection, or rescaling. A later full-cohort aggregate signal audit
may add descriptive statistics to the run report, but cannot change cohort
eligibility, preprocessing, or any frozen analysis choice.

### Completed full-waveform QC before inference

The pre-inference audit read all 18,842 `broad_exact10` records. All 18,842
source arrays were numeric, finite `float16` with shape `[12, 5000]`; all
18,842 adapter outputs were finite, contiguous `float32` with shape
`[12, 1000]`. Thus 18,842/18,842 passed the frozen shape/dtype/finiteness
contract.

The amplitude summaries below use the resampled 100 Hz arrays in physical
millivolts, before PTB normalization. They are observed descriptive evidence,
not acceptance limits.

| Absolute-amplitude statistic | Global samples (mV) | Per-record maximum (mV) |
|---|---:|---:|
| q50 | 0.042204175144433975 | 1.682507574558258 |
| q95 | 0.3094510748982424 | 2.915231156349182 |
| q99 | 0.8505193614959694 | 16.064358968734766 |
| q99.9 | 1.7757529041766844 | 57.53314697647054 |
| q99.99 | 10.919973223976712 | not pre-specified |
| Maximum | 717.4902954101562 | 717.4902954101562 |

Per-record maximum absolute amplitude exceeded 5 mV for 354 records, 10 mV
for 260, 20 mV for 96, and 50 mV for 24. These extreme but finite values may
reflect artifacts and/or source-domain shift. Every one is retained: there is
no amplitude-based exclusion, clipping, rejection, rescaling, cohort change,
model choice, or tuning rule. Their effect belongs in the transport result
rather than being removed after inspection.

## 6. Frozen models: no tuning or recalibration

All six released members are evaluated: ResNet1D and ECG Transformer for seeds
2026, 2027, and 2028. The run binds the existing refit bundle and calibration
bundle by both file and artifact hashes.

Loading those members through `load_completed_audit_runtime` also requires the
completed-release authorization lineage below. These are read-only inputs:

| Runtime-lineage input | File SHA-256 | Bound internal hash |
|---|---|---|
| `configs/protocol.yaml` | `d630ccb99569513082ccaaafe1b0117f5fe1567a505c600add9c0a79b64c51c8` | protocol `ebfdb588615bfa22eedc6d936d7b0155a33702878cbe0258ebb84aaa88567e09` |
| `runs/release/ptbxl_matched_equal_budget_v1/final_evaluation_spec.json` | `143770b44871129b6d00efba9e67c3c16f58dd9117d9b4cd3eef5d3f896d621d` | artifact `1f73c021a544ffeb119ffe8e490a16e32ec84247e30bce1ffd895fcffed6c762` |
| `runs/release/.final-test-openings/1f73c021a544ffeb119ffe8e490a16e32ec84247e30bce1ffd895fcffed6c762.opening-ledger.json` | `73b6ccc7076f2d01621b7c442fe7ee7f249479a4447e3cb9c716bb4074293e53` | ledger `3bb83554a08832212989ea8f3ea212f6af42c08460edbde9ba130065b1115a57` |

This binding proves that external inference reloads the already-completed
authorized model runtime. It does not reopen fold 10, authorize another
fold-10 query, or permit any edit to the protocol, specification, ledger, or
sealed release artifacts.

### Frozen one-pass external runtime

SPH inference is one pass over `broad_exact10` on exactly `cuda:0`, with BF16
required. Automatic device selection and CPU fallback are forbidden. Batch
size is 128, the minimum of the sealed member loader plans (ResNet1D used 128;
ECG Transformer used 192). The loader uses four workers, pinned memory,
persistent workers, no shuffle, and no dropped final batch.

Unavailable CUDA/BF16 support or an out-of-memory condition fails closed. It
does not authorize a smaller batch, CPU execution, another precision, or any
other in-place runtime change. A changed runtime requires a newly frozen
protocol and a separate immutable output root.

For each member, the checkpoint, global fold-9 temperature, five fold-9 label
thresholds, and fold-9 entropy gates remain unchanged. SPH cannot be used for:

- model or seed selection;
- training or fine-tuning;
- probability recalibration;
- threshold or entropy-gate fitting;
- choosing exclusions or label mappings; or
- selecting a favorable analysis cohort.

Raw-sigmoid and frozen-temperature probability views are both retained. Any
SPH-specific adaptation would answer a different question and requires a new
protocol and a new, untouched SPH evaluation partition.

## 7. Metrics and uncertainty

For every member and every frozen cohort, report prevalence plus per-label and
macro AUROC, average precision, Brier score, and 15-bin ECE. Report Hamming
risk and exact-match accuracy using only the member's frozen fold-9 thresholds.
At the frozen fold-9 entropy gates for nominal coverages 1.0, 0.9, 0.8, 0.7,
and 0.5, report observed SPH coverage, Hamming risk, and exact-match accuracy.
Nominal coverage is a PTB fold-9 target, not a promise about SPH coverage.

Report all six members individually. For each architecture, report the three
seed values, mean, and sample standard deviation. Within each seed, report the
paired difference `ECG Transformer - ResNet1D` on identical patients. Do not
pool predictions across seeds as though they were independent patients.

Uncertainty uses a patient-cluster percentile bootstrap:

- 1,000 resamples and 95% intervals;
- all ECGs for a sampled patient travel together, including repeat records;
- minimum 500 valid resamples for an interval;
- base seed `20260816`, plus the frozen cohort offset and model seed; and
- identical patient draws for the paired architecture comparison within a
  seed and cohort.

MI and HYP have few positive patients. Their point estimates and percentile
intervals may be unstable, and an unestimable bootstrap replicate is omitted,
not coerced to a favorable value. No multiplicity correction or confirmatory
hypothesis claim is made because the experiment is exploratory.

## 8. Gates

Hard gates are integrity gates only. The run stops before inference or
publication if a source size/hash changes, archive paths are unsafe, extracted
records do not match metadata, cohort or positive counts differ, a signal
violates its shape/unit/lead contract, a frozen bundle or normalization hash
changes, any completed-runtime lineage file or internal hash changes, the
external runtime differs from its frozen values, the execution revision is not
recorded cleanly, or the output root is already populated.

There is deliberately **no scientific performance pass/fail gate**. A high
AUROC does not erase mapping uncertainty or establish clinical validity; a low
AUROC is still an informative transport result. The three frozen cohort views
must all be reported, even when one is less favorable.

## 9. Immutable outputs

Outputs belong under `runs/external_transport/sph_figshare_v1/`. The run is
fail-closed against overwrite and must include a byte-for-byte protocol
snapshot, source inventory, cohort manifest and summary, member predictions
and reports, architecture summaries, paired bootstrap reports, final Markdown
report, run log, and a derived-artifact manifest hashing every generated file
and bound input.

Partial or failed output is preserved. A corrected attempt uses a new
content-addressed output root and records why it supersedes the earlier one;
it never edits a published run in place. The sealed PTB fold-10 release and
the r1/r2/r3 post-evaluation trees are read-only inputs and must not be changed.

The raw run root is Git-ignored and local-private. `ECG_ID`, `Patient_ID`,
record paths, raw AHA strings, and parsed primary/modifier/ambiguous code lists
must remain in the private cohort/alignment artifact. Public artifacts may
contain sanitized aggregates or identifier-free aligned target/probability
arrays, and bind their row alignment only through the SHA-256 of the private
artifact. A public snapshot is a separate sanitization step; it is never an
automatic copy of the raw run tree.

Identifier-free public rows use the frozen permutation seed `2026081601` so
the same private alignment produces the same public ordering on every run.
This permutation is a reproducibility convention, not a confidentiality,
de-identification, anonymization, or security mechanism. Public artifacts
remain acceptable only because identities, paths, codes, and row-level source
metadata are omitted under the privacy contract above.

## 10. Required final wording

Every result summary must say that this was an **exploratory external transport
stress test**, conducted with **no tuning or recalibration on SPH**, and that it
is **not clinical validation**. It must remain labeled research-only and must
not claim diagnostic safety, deployment readiness, or medical-device status.
