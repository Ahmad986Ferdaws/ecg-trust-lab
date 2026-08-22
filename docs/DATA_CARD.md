# Data card: PTB-XL five-superclass cohort

**Snapshot:** PTB-XL version 1.0.3, 100 Hz waveforms  
**Project task:** multi-label classification in the fixed output order
`[NORM, MI, STTC, CD, HYP]`  
**Protocol:** `ptbxl-superclass-trust-v1`  
**Status:** local source files, manifest, and training-only normalization verified

> This dataset and every model built from it are for research and education.
> This project is **not a medical device**. Outputs are not diagnoses, triage
> decisions, treatment recommendations, or evidence of clinical safety.

## Identity, provenance, and license

| Item | Value |
|---|---|
| Upstream dataset | [PTB-XL, version 1.0.3](https://physionet.org/content/ptb-xl/1.0.3/) |
| Version release | November 9, 2022 |
| Version DOI | [10.13026/kfzx-aw45](https://doi.org/10.13026/kfzx-aw45) |
| Dataset authors | Patrick Wagner, Nils Strodthoff, Ralf-Dieter Bousseljot, Wojciech Samek, Tobias Schaeffter |
| Original data descriptor | Wagner et al., *Scientific Data* 7, 154 (2020), [10.1038/s41597-020-0495-6](https://doi.org/10.1038/s41597-020-0495-6) |
| License | [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/) (CC BY 4.0) |
| Project subset | Official downsampled `records100/` WFDB files plus release metadata; the 500 Hz files are not acquired |
| Acquisition path | [`scripts/download_ptbxl.py`](../scripts/download_ptbxl.py), using PhysioNet's public S3 mirror and the release's authoritative `SHA256SUMS.txt` |

Users must preserve attribution and comply with CC BY 4.0. The downloaded copy
of the license is at `data/raw/ptb-xl/1.0.3/LICENSE.txt`. Version 1.0.3 matters:
it removed two further duplicate records and reports consensus labels for
retained duplicate records. Results must not be described as using another
PTB-XL release.

The canonical upstream release contains **21,799 ECG records from 18,869
patients**. Every selected metadata, header, and waveform file was checked
against the official release checksum inventory before downstream artifacts
were built.

## Project cohort and target construction

The manifest builder parses `scp_codes`, keeps only statements marked as
diagnostic in `scp_statements.csv`, and aggregates their `diagnostic_class`
values into five binary targets. A mapped code's presence creates the target;
the source likelihood value is retained in `scp_codes` but is not used as a
target weight or threshold. Rhythm and form statements outside these five
diagnostic superclasses are not prediction targets.

| Cohort quantity | Count |
|---|---:|
| Upstream records | 21,799 |
| Upstream patients | 18,869 |
| Included records with at least one target | **21,388** |
| Included patients | **18,617** |
| Positive superclass assignments | 27,765 |
| Excluded records with no mapped superclass | **411** |
| Patients represented among excluded records | 329 |
| Excluded-only patients absent from the task cohort | 252 |

The 411 excluded records are not necessarily annotation-free. They have no
mapped target among this task's five diagnostic superclasses and may contain
other SCP-ECG rhythm or form statements. Seventy-seven of their patients also
have at least one included ECG; the other 252 patients disappear from the task
cohort. No included row has an all-zero target.

### Labels

The order below is the model-output and artifact contract. It must never be
alphabetically reordered.

| Position | Label | Meaning | Positive records | Mapped SCP-ECG diagnostic statements |
|---:|---|---|---:|---|
| 0 | NORM | Normal ECG | 9,514 | `NORM` |
| 1 | MI | Myocardial infarction | 5,469 | `ALMI`, `AMI`, `ASMI`, `ILMI`, `IMI`, `INJAL`, `INJAS`, `INJIL`, `INJIN`, `INJLA`, `IPLMI`, `IPMI`, `LMI`, `PMI` |
| 2 | STTC | ST/T change | 5,235 | `ANEUR`, `DIG`, `EL`, `ISCAL`, `ISCAN`, `ISCAS`, `ISCIL`, `ISCIN`, `ISCLA`, `ISC_`, `LNGQT`, `NDT`, `NST_` |
| 3 | CD | Conduction disturbance | 4,898 | `1AVB`, `2AVB`, `3AVB`, `CLBBB`, `CRBBB`, `ILBBB`, `IRBBB`, `IVCD`, `LAFB`, `LPFB`, `WPW` |
| 4 | HYP | Hypertrophy | 2,649 | `LAO/LAE`, `LVH`, `RAO/RAE`, `RVH`, `SEHYP` |

These counts overlap. The sum of class positives must therefore not be compared
with the number of ECGs as if this were a single-label task.

| Positive labels on one record | Records |
|---:|---:|
| 1 | 16,244 |
| 2 | 4,068 |
| 3 | 919 |
| 4 | 157 |
| 5 | 0 |

In total, **5,144 records (24.05%) are multi-label**. Pairwise intersections
are shown below; the diagonal is the per-label total.

| | NORM | MI | STTC | CD | HYP |
|---|---:|---:|---:|---:|---:|
| NORM | 9,514 | 1 | 33 | 415 | 5 |
| MI | 1 | 5,469 | 1,339 | 1,794 | 818 |
| STTC | 33 | 1,339 | 5,235 | 1,066 | 1,509 |
| CD | 415 | 1,794 | 1,066 | 4,898 | 787 |
| HYP | 5 | 818 | 1,509 | 787 | 2,649 |

The authoritative mapping and source counts are serialized in
`data/manifests/ptbxl_superclasses_v1.0.3.summary.json`.

## Patient-safe folds and permitted roles

PTB-XL's `strat_fold` assignment keeps all records from a patient in one fold.
The manifest builder checks this invariant before writing artifacts. This
project intentionally uses a stricter four-role protocol than the upstream
three-way benchmark convention.

| Fold(s) | Project role | Records | Patients | Permitted use |
|---|---|---:|---:|---|
| 1–7 | Development training | 14,955 | 12,969 | Fit models and training-only preprocessing |
| 8 | Model selection | 2,129 | 1,854 | Architecture and hyperparameter selection; early stopping |
| 9 | Calibration | 2,146 | 1,917 | Fit calibration, thresholds, and abstention policy after model choice is frozen |
| 10 | **Sealed final test** | 2,158 | 1,877 | One-time final evaluation only, after configuration and policy freeze |
| 1–10 | Included cohort | 21,388 | 18,617 | — |

The individual included-fold counts are:

| Fold | Records | Patients | Fold | Records | Patients |
|---:|---:|---:|---:|---:|---:|
| 1 | 2,139 | 1,854 | 6 | 2,129 | 1,860 |
| 2 | 2,138 | 1,822 | 7 | 2,134 | 1,846 |
| 3 | 2,150 | 1,864 | 8 | 2,129 | 1,854 |
| 4 | 2,130 | 1,858 | 9 | 2,146 | 1,917 |
| 5 | 2,135 | 1,865 | 10 | 2,158 | 1,877 |

Fold 10 is **sealed in code**, not merely reserved by convention. Ordinary
dataset access is rejected unless an explicit final-test token is bound to the
resolved protocol. It must never be inspected during preprocessing, debugging,
model selection, calibration, threshold fitting, abstention tuning, or error
analysis. The routine smoke test deliberately loads folds 1, 8, and 9 only.

The resolved protocol is [`configs/protocol.yaml`](../configs/protocol.yaml):

- protocol ID: `ptbxl-superclass-trust-v1`
- protocol SHA-256: `sha256:ebfdb588615bfa22eedc6d936d7b0155a33702878cbe0258ebb84aaa88567e09`

After model selection is frozen, the chosen configurations may be refit on
folds 1–8 (17,084 records; 14,823 patients) with a frozen epoch and optimizer
budget. Fold 8 must not drive any new choice during refit, and normalization
remains the fold-1–7 artifact documented below.

## Waveform and loader contract

| Property | Contract |
|---|---|
| Container | WFDB `.hea` + `.dat` records from `records100/` |
| Signal representation | WFDB physical signal decoded as finite `float32`; source headers use mV with 1,000 ADC units/mV |
| Sampling rate | Exactly 100 Hz |
| Duration | Exactly 10 seconds |
| Samples | Exactly 1,000 per lead |
| Tensor shape | `[12, 1000]` in `[lead, time]` order |
| Lead order | `[I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6]` |
| Target shape | `[5]` multi-hot `float32` in `[NORM, MI, STTC, CD, HYP]` order |

The loader validates the WFDB sampling frequency, sample count, lead names,
lead uniqueness, shape, and finite values, then reorders case-insensitive lead
names into the canonical contract. It fails closed on missing, duplicated, or
unexpected leads. It performs **no silent resampling, padding, truncation,
imputation, or per-record normalization**. See
[`src/ecg_trust/data/dataset.py`](../src/ecg_trust/data/dataset.py).

## Training-only normalization

Per-lead statistics were computed from the **14,955 records in folds 1–7 only**:
14,955,000 time samples per lead. The implementation streams one ECG at a time,
uses mergeable Welford updates with `float64` accumulators, and reports the
population standard deviation (`M2 / N`). At load time, the physical `float32`
signal is transformed independently per lead as `(x - mean) / std`.

| Lead | Mean (mV) | Population std (mV) |
|---|---:|---:|
| I | -0.0018307274 | 0.1610751489 |
| II | -0.0015488698 | 0.1596513185 |
| III | 0.0002817755 | 0.1609899659 |
| aVR | 0.0016720850 | 0.1387131065 |
| aVL | -0.0010021137 | 0.1398463762 |
| aVF | -0.0006112175 | 0.1385036882 |
| V1 | 0.0002298929 | 0.2335947866 |
| V2 | -0.0009616068 | 0.3339088760 |
| V3 | -0.0015730588 | 0.3286838056 |
| V4 | -0.0013690209 | 0.2928534079 |
| V5 | -0.0007703355 | 0.2723298360 |
| V6 | -0.0025978538 | 0.2871410955 |

The full-precision values and provenance are stored in
`artifacts/preprocessing/ptbxl_v1.0.3_train_folds_1-7_normalization.json`.

| Normalization provenance field | Value |
|---|---|
| Schema version | 1 |
| Dataset version | 1.0.3 |
| Training folds | `[1, 2, 3, 4, 5, 6, 7]` |
| Records / samples per lead | 14,955 / 14,955,000 |
| Stable selected-row manifest fingerprint | `55dd86001dee2006cb241ff8b4f3970d8fcbb1ae9ecd430f5dd61478673ce235` |
| Normalization JSON file SHA-256 | `4a6cb489098361d8221403c14871c242672c346975af3a07f731ceac97264363` |

The selected-row fingerprint hashes normalized manifest content used by the
statistics routine; it is deliberately different from the byte hash of either
the CSV or Parquet container.

## Demographic and audit metadata

Counts below describe the 21,388-record task manifest, not the full upstream
release.

| Field | Present | Missing | Observed summary and cautions |
|---|---:|---:|---|
| `age` | 21,388 | 0 | Median 62; 285 values are the HIPAA-protective sentinel `300`; non-sentinel range 2–89 |
| `sex` | 21,388 | 0 | Binary source codes: `0` = 11,111 (51.95%), `1` = 10,277 (48.05%); the manifest preserves the source encoding |
| `height` | 6,918 | 14,470 (67.65%) | Median of present values 166; strong missingness and implausible outliers require an explicit audit policy |
| `weight` | 9,366 | 12,022 (56.21%) | Median of present values 70; strong missingness and implausible outliers require an explicit audit policy |
| `recording_date` | 21,388 | 0 | Pseudonymized by a random patient-specific date shift; not a valid calendar-time covariate |
| `validated_by_human` | 21,388 | 0 | True for 15,895 (74.32%); all included fold-9 and fold-10 records are true |

The upstream documentation states that apparent ages above 89 are represented
in the range of 300 for HIPAA compliance and that recording dates were shifted
by a random offset per patient. The value `300` must be treated as censored,
not as a literal age. The current manifest does not include race, ethnicity,
geography, language, socioeconomic status, medications, outcomes, or care
setting, so it cannot support fairness claims along those axes.

## Integrity fingerprints

Generated data and artifacts are intentionally ignored by Git and must be
recreated locally. The canonical local snapshot has these hashes:

| Artifact | SHA-256 |
|---|---|
| Source `ptbxl_database.csv` | `7600de9c1b27d181d850b3c6038a35d7c3ddb6bb33b702e3a20252a6859d216b` |
| Source `scp_statements.csv` | `ad05b0b1fcae83bb1230755ad9cfc7c96f303feddc08a4a9ad5bdc9ca63bac8f` |
| Manifest CSV | `ff771ec783dd1665c8e59f497be0f624ed521fd34a73c9e70e2a9783b44ec49c` |
| Manifest Parquet | `563a2b715cc6f6657b04c2f67d813fd7c30a696210740f97c55a070f157579a0` |
| Manifest summary JSON | `7e7199e0378a213bc29b5fe6f1ae3f9e0eda0350d5e07c56fd0d90aeda19b8a6` |
| Training-only normalization JSON | `4a6cb489098361d8221403c14871c242672c346975af3a07f731ceac97264363` |

The three manifest hashes are also recorded in
`data/manifests/ptbxl_superclasses_v1.0.3.sha256`.

## Relationship to the SPH transport dataset

This card's training, selection, calibration, and sealed-test contracts remain
PTB-XL-only. After those stages and the r3 audit were complete, the same six
frozen members were applied once to SPH as a separate exploratory retrospective
external-transport stress test. SPH was not incorporated into the PTB-XL
manifest and was not used for training, model or seed selection, normalization,
fine-tuning, recalibration, threshold fitting, gate fitting, or post-result
cohort selection.

SPH is the Shandong Provincial Hospital dataset described by Liu et al.
([Scientific Data paper](https://doi.org/10.1038/s41597-022-01403-5),
[Figshare collection](https://doi.org/10.6084/m9.figshare.c.5779802.v1)). The
Figshare items are marked CC0; citation remains appropriate. Raw source files
remain Git-ignored and are not redistributed.

| Frozen SPH view | ECGs | Patients | Role |
|---|---:|---:|---|
| `broad_exact10` | 18,842 | 18,157 | Every exact-10-second source record; mapping sensitivity only |
| `primary_mapped` | 15,698 | 15,193 | At least one conservative direct superclass mapping; primary analysis |
| `no_ambiguous_mapped` | 15,563 | 15,066 | Pre-specified sensitivity excluding ambiguous primary codes |

The new AHA-to-PTB-superclass bridge was not clinically adjudicated. The
primary cohort contains only 138 MI-positive and 113 HYP-positive ECGs, so
those endpoints are particularly sparse. The [frozen protocol](EXTERNAL_TRANSPORT_SPH_R2.md),
[sanitized result](../publication/external_transport_sph_r2/FINAL_RESULTS.md),
and [artifact audit](../reports/SPH_EXTERNAL_TRANSPORT_AUDIT.md) preserve the
separate source, cohort, label-map, and no-adaptation provenance. This is not
prospective or clinical validation.

## Known limitations and risks

- **Historical, single-origin cohort.** Signals were collected at PTB from
  Schiller AG devices between October 1989 and June 1996. Performance on
  contemporary populations, workflows, or acquisition hardware is unknown.
- **Single-vendor device domain.** The upstream `device` column contains 11
  device/software string variants, but all recordings come from Schiller AG.
  The project manifest does not currently propagate `device`, so device-wise
  audit requires a provenance-preserving join to raw metadata.
- **Downsampled signal only.** This project uses 100 Hz waveforms, not the
  upstream 500 Hz files. Conclusions that depend on higher-frequency morphology
  are out of scope.
- **Coarse and incomplete target space.** PTB-XL contains 71 SCP-ECG statements
  across diagnostic, rhythm, and form categories. This task predicts only five
  aggregated diagnostic superclasses; it is not a 71-label diagnostic system
  and does not model rhythms or forms outside the mapping above.
- **Label uncertainty.** Records were annotated by up to two cardiologists and
  human validation is not universal outside folds 9 and 10. Aggregation hides
  within-superclass heterogeneity, and source likelihoods are not modeled.
- **Multi-label dependence.** Superclasses overlap substantially, including
  some `NORM` co-occurrence. Independent sigmoid outputs do not make labels
  clinically independent or mutually exclusive.
- **Metadata constraints.** Height and weight are heavily missing, age is
  censored above 89, sex is binary-coded, and dates are shifted. Missing or
  absent attributes cannot be treated as evidence of subgroup safety.
- **Limited retrospective transport evidence; no clinical validation.** The
  held-out PTB-XL fold estimates internal benchmark performance. The separate
  frozen SPH stress test probes transport without adaptation, but its
  unadjudicated ontology bridge, sparse MI/HYP labels, retrospective design,
  and single external source do not establish general transportability,
  clinical benefit, prospective safety, or deployment readiness.

## Intended and prohibited uses

Intended uses are reproducible research on five-superclass discrimination,
calibration, selective abstention, robustness, subgroup behavior, explanation
faithfulness, and a local research demonstration using contract-compatible
ECGs.

Project policy prohibits using this cohort or derived models for autonomous or
assisted clinical diagnosis, patient triage, treatment selection, emergency
decision-making, real-world screening, or any other medical-device function.
It also prohibits claims of demographic fairness, device portability, or
clinical generalization without appropriate independent prospective clinical
evidence; the one exploratory SPH stress test does not satisfy that standard;
attempts to re-identify people; and any tuning or selection after fold-10
results have been inspected.

## Reproduction and verification

Run from the repository root in PowerShell. These are the checked project CLI
entry points; no fold-10 access appears in the routine sequence.

```powershell
# Recreate the locked environment.
uv sync --frozen --all-groups

# Acquire the official 100 Hz release subset and verify every selected file.
uv run --frozen python scripts/download_ptbxl.py --workers 64
uv run --frozen python scripts/download_ptbxl.py --verify-only

# Rebuild deterministic task artifacts and training-only statistics.
uv run --frozen python scripts/build_manifest.py
uv run --frozen python scripts/compute_normalization.py

# Exercise only development, selection, and calibration folds.
uv run --frozen python scripts/smoke_dataset.py --folds 1 8 9

# Verify the contracts and leakage guards.
uv run --frozen pytest tests/unit/test_manifest.py tests/unit/test_dataset.py tests/unit/test_protocol.py
```

A correct rebuild must recover the counts and hashes listed above. The smoke
loader has been run successfully on real records from folds 1, 8, and 9,
returning finite tensors of shape `[12, 1000]` and targets of shape `[5]`.
Opening fold 10 is a separate, explicit final-evaluation event after all model,
calibration, threshold, abstention, and reporting choices are frozen.
Reproducing the later SPH transport experiment is separately governed by the
[immutable r2 protocol](EXTERNAL_TRANSPORT_SPH_R2.md); an existing output root
must never be overwritten or reused.
