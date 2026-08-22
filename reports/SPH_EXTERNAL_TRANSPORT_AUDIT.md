# SPH external-transport r2 audit record

This record audits the completed SPH r2 run against the
[frozen r2 protocol](../docs/EXTERNAL_TRANSPORT_SPH_R2.md) and its
[frozen configuration](../configs/external_transport_sph_frozen_r2.yaml). The run is an
**exploratory external transport stress test**, performed with **no tuning or
recalibration on SPH**. It is **not clinical validation** and is **research only**.

## Run identity and completion

- Execution commit: `215b26e0aafe893e081a13db7117c303c7dbb438` on a clean worktree.
- Scientific parent commit: `724a510b03eb539eda2add6de359855f3ffaf2b5`.
- Completed: 2026-08-16 UTC; the final derived manifest was written at
  21:01:28 UTC.
- Immutable local run root: `runs/external_transport/sph_figshare_v1__attempt-r2/`.
- Completion state: exactly 61 regular files and no failure receipt. The public
  manifest covers the exact 49-file public payload; the derived manifest covers
  the exact 60-file pre-manifest root. Every listed size and byte hash matches.
- All 48 JSON artifacts pass their canonical self-hash check. The attempt marker
  uses `attempt_sha256`; the other 47 use `artifact_sha256`.
- All nine JSON/NPZ pairs have matching sidecar names, byte sizes, SHA-256 values,
  array names, shapes, and dtypes.
- All 11 frozen input files still match the pre-inference size/hash snapshot.

## Cryptographic bindings

All values below are SHA-256. “Self” means canonical finite JSON after removing
the named self-hash field; “file” means the exact bytes on disk.

| Binding | SHA-256 |
|---|---|
| Frozen protocol/config file | `840acb758d50dbf1a04bf704b16a58d4d29d668370ce7ef91ef7a44860bf311b` |
| Attempt marker self | `bb5cfdafeee2145f6ae4b18b6e3f8a138b9516a81f9e84db369125f5c57584c7` |
| Attempt marker file | `3b64134110d2ca879bbfee3ba6b955d13d2d78d06afe6af98aeda55d3e3a7189` |
| Bound-input set | `7c47b3e4acf6d8619168627c588a72ce80b50369f60edcaeee9d342467b5b854` |
| Bound-input snapshot self / file | `473c6cb717838b5ce84e148f2e3aabc7fcd4268956ddf5e4a76ccbdf47bca0b6` / `3813a4110300b9f1256f25354fe13522d2e6a81a43e719e8366d143894d3e88d` |
| Source-inventory content | `3b1cd0f23b0d223bc41af6ac92acd7a6148f9a39c021b890159b297dbb1384f7` |
| Source-inventory wrapper self / file | `fef1bb20c24de2764117792cc5ed261bfb67c79b0d75e112c87950e6c228ba64` / `91fc22007f312ee1f37b7e20f8cd9f501287a788470b80739f67659982065bbe` |
| Public manifest self / file | `eb333e255f41beece3cd5fa413e9a605c017bf63b0c977207c28e5ac1373fc0f` / `bf0eee05a49de54119bd920ca41c60feab562330efba9381ee78869d34f6b476` |
| Derived manifest self / file | `96cc02d2a9cd5d940ef2cd4fc06a0dccd16a8495610fd443d70d7f436bbb81c2` / `c87a103cf5741fff6896a81e96f9a269dbc4b8dbc0ce4e150ba1b5a6b5ac788f` |
| Private alignment file / logical broad alignment | `ab3d8dc16c74690bb3ccd7f35cb6990c651e54ea8cb6e51f85782e2ce8f3d3f4` / `3c3b67479fbcaefab2e2367445a8915ef0a3a455608864438e626e961d45904e` |
| Physical resampled signal | `f0888fa70573084c346dc373680d6e47c8ed830390b96a40f2de62201f8ffeec` |

The source audit also bound 25,771 safe archive members, all 25,770 expected
HDF5 records, exact metadata membership, byte-identical extraction, and content
tree SHA-256
`b4382ef49d0ee9cccb7f8a372023815ad31932cb9497152ba8115263b8f5159d`.

## Frozen inference and cohorts

The private checkpoint records one protocol-level SPH inference pass over the
broad cohort and exactly six frozen members: ResNet1D and ECG Transformer for
seeds 2026, 2027, and 2028. It contains six raw-logit matrices of shape
`[18,842, 5]`; no second SPH inference pass was performed. Before that pass, all
six reloaded members reproduced their sealed PTB fold-10 logits exactly: 10,790
logits per member, zero mismatches, and zero maximum absolute error.

| Frozen cohort | ECG records | Patients | Operational all-zero rows |
|---|---:|---:|---:|
| `primary_mapped` | 15,698 | 15,193 | 0 |
| `broad_exact10` | 18,842 | 18,157 | 3,144 |
| `no_ambiguous_mapped` | 15,563 | 15,066 | 0 |

Positive counts below are `ECG records / patients` and are multi-label, so
columns need not sum to the cohort total.

| Cohort | NORM | MI | STTC | CD | HYP |
|---|---:|---:|---:|---:|---:|
| `primary_mapped` | 11,172 / 10,874 | 138 / 131 | 3,030 / 2,947 | 1,510 / 1,453 | 113 / 110 |
| `broad_exact10` | 11,172 / 10,874 | 138 / 131 | 3,030 / 2,947 | 1,510 / 1,453 | 113 / 110 |
| `no_ambiguous_mapped` | 11,172 / 10,874 | 131 / 124 | 2,981 / 2,899 | 1,470 / 1,417 | 64 / 63 |

## Independent result recomputation

A read-only audit reloaded the identifier-omitted prediction arrays and
recomputed both probability views, frozen threshold decisions, and frozen
entropy-gate decisions for all 18 member/cohort combinations. It completed 84
member checks, recomputed all 432 paired point estimates, and rebuilt all six
architecture summaries. The largest absolute floating-point difference was
`1.942890293094024e-16`, attributable to summation after the frozen public row
permutation.

Across the 54 bootstrap result blocks, requested and completed resamples were
both exactly 1,000. A recursive audit enumerated exactly 1,476 reported
intervals; every interval had status `ok`, 1,000 valid resamples, and zero
invalid resamples. The final Markdown report was regenerated byte-for-byte from
the stored reports. Its SHA-256 is
`9a315ca3194eed9a055ae4a8f30543f64f72cf6823d7f8b103ac34b3f85f223b`.

## Aggregate results

Values are calibrated macro mean +/- sample standard deviation across the three
frozen seeds. Lower Brier and ECE are better; higher AUROC and AP are better.

| Cohort | Architecture | AUROC | AP | Brier | ECE |
|---|---|---:|---:|---:|---:|
| `primary_mapped` | ResNet1D | 0.930912 +/- 0.000964 | 0.698955 +/- 0.006752 | 0.061301 +/- 0.000248 | 0.052477 +/- 0.000877 |
| `primary_mapped` | ECG Transformer | 0.924088 +/- 0.001231 | 0.657838 +/- 0.007557 | 0.064153 +/- 0.003962 | 0.061480 +/- 0.006313 |
| `broad_exact10` sensitivity | ResNet1D | 0.904714 +/- 0.001118 | 0.642155 +/- 0.006472 | 0.076688 +/- 0.000146 | 0.071016 +/- 0.000858 |
| `broad_exact10` sensitivity | ECG Transformer | 0.898652 +/- 0.001028 | 0.601741 +/- 0.008373 | 0.078817 +/- 0.003018 | 0.078034 +/- 0.004649 |
| `no_ambiguous_mapped` sensitivity | ResNet1D | 0.928405 +/- 0.001382 | 0.666044 +/- 0.005249 | 0.060795 +/- 0.000245 | 0.052181 +/- 0.000772 |
| `no_ambiguous_mapped` sensitivity | ECG Transformer | 0.920511 +/- 0.000971 | 0.627274 +/- 0.003484 | 0.063547 +/- 0.003912 | 0.061313 +/- 0.006297 |

The conservative conclusion is that ResNet1D had stronger aggregate point
estimates in all three frozen cohort views. On the primary cohort, paired
Transformer-minus-ResNet AUROC and AP 95% intervals were below zero for all
three seeds. Macro Brier differences for two seeds included zero, so this is
not evidence of uniform superiority on every metric, label, population, or
deployment setting. MI and HYP have few positive patients, the label bridge was
not clinically adjudicated, and the broad all-zero analysis treats missing
direct mappings as operational zeros even though they are unknown rather than
verified negatives. There was no scientific performance pass/fail gate.

## Privacy and publication boundary

The intended tracked output is the
[publication-safe aggregate snapshot](../publication/external_transport_sph_r2/README.md),
not the ignored raw run root. The privacy audit used an exact 35-artifact
whitelist: `FINAL_RESULTS.md`, `cohort_summary.json`, six architecture summaries,
18 member aggregate reports, and nine paired aggregate reports. Those 35 files
were copied byte-for-byte; the bundle adds only its README and checksum index.

The snapshot deliberately excludes the raw public manifest, all six
row-aligned member-prediction JSON/NPZ pairs, both signal-QC files, every NumPy
archive, and the entire private directory. It therefore excludes ECG and patient
identifiers, record paths, AHA codes, deterministic row alignment, private
cohort tables, the inference checkpoint, source ECGs, and model weights. The
deterministic permutation in the local run is a reproducibility convention, not
an anonymization or confidentiality mechanism.

These results do not establish diagnostic safety, treatment utility,
deployment readiness, prospective generalization, or medical-device status.
They must not be used for diagnosis, treatment, triage, or any other clinical
decision.
