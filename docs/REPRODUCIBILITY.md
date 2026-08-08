# Reproducibility protocol

This document is the executable record for the current PTB-XL five-superclass
pipeline. Run commands from the repository root in PowerShell. The matched
12-by-2 sweep is complete; the fixed multi-seed confirmation and downstream
release machinery are implemented, while their scientific results remain
pending until each gated command actually completes.

The scientific task is multi-label prediction of `[NORM, MI, STTC, CD, HYP]`
from a ten-second, 100 Hz, 12-lead ECG. It is diagnostic-superclass
classification, not a mutually exclusive five-class task and not a general
arrhythmia diagnosis system.

## 1. Frozen contract

The authoritative protocol is `configs/protocol.yaml`.

| Role | PTB-XL folds | Permitted use |
|---|---:|---|
| Training | 1-7 | Fit model parameters and folds-1-7 normalization |
| Model selection | 8 | Early stopping, architecture/configuration comparison, and epoch-budget selection |
| Calibration | 9 | Fit one global temperature, per-label F1 thresholds, and entropy coverage gates |
| Final test | 10 | One-time reporting with all choices frozen |

The current protocol hash is:

```text
sha256:ebfdb588615bfa22eedc6d936d7b0155a33702878cbe0258ebb84aaa88567e09
```

Recompute it rather than copying it into a new run:

```powershell
$env:UV_CACHE_DIR = Join-Path (Get-Location) ".uv-cache"
uv run python -c "from ecg_trust.protocol import load_protocol; print(load_protocol('configs/protocol.yaml').protocol_hash)"
```

The code enforces the following boundaries:

- development training must use folds 1-7 and validation must use fold 8;
- normalization must remain fitted on folds 1-7, including during refit;
- frozen refit must use exactly folds 1-8 and has no early stopping or model
  selection;
- calibration artifacts must contain fold 9 only;
- fold 10 cannot be opened without a process-local protocol token issued after
  the exact confirmation `I understand fold 10 is the one-time final test
  set.`;
- calibration and final predictions must match on model name, seed, protocol
  hash, resolved config hash, manifest hash, and label order.

The low-level exporter token is deliberate friction. The formal
`release_pipeline.py run-final` path adds the authoritative control: a
canonical opening marker bound to the exact refit/calibration bundle, an
exclusive writer lock, rejection of pre-ledger outputs, and a persistent
one-time ledger containing the purpose, operator, timestamps, and artifact
hashes. Final reporting must use that release path, not direct ad hoc exports.

Fold 8 is development evidence. Fold 9 is post-processing evidence. Neither is
a held-out final-test result.

## 2. Environment bootstrap

The lockfile and `.python-version` define a Python 3.12 environment. The
project sources PyTorch from the CUDA 13.0 wheel index in `pyproject.toml`.

```powershell
$env:UV_CACHE_DIR = Join-Path (Get-Location) ".uv-cache"
uv sync --frozen --all-groups
uv run python --version
uv run ecg-verify
```

`ecg-verify` passes only after a real CUDA autocast forward/backward loop has
finite losses and gradients. It refuses to silently pass on CPU. The JSON
output must end with `"status": "PASS"`; archive that output with the run
record.

Do not replace the locked CUDA build with a generic CPU-only PyTorch package.
The training runners use deterministic algorithms, disable cuDNN benchmarking,
and seed Python, NumPy, PyTorch, CUDA, DataLoader workers, and the loader
generator.

The runners record Python, platform, PyTorch, CUDA runtime, device, BF16 status,
GPU memory, Git, source-tree, dependency-lock, manifest, and normalization
provenance. The multi-seed planner requires a completely clean downstream
commit B, proves that the sweep commit A is its ancestor, and requires a
byte-empty A-to-B diff over the fixed scientific kernel. It also records the
exact allowed non-kernel changes. A dirty or unaddressable tree cannot create
a confirmation plan.

## 3. Quality gate

Run the complete code-quality gate before data work and again before freezing
each confirmatory configuration:

```powershell
uv run pytest -q
uv run ruff check src tests scripts
uv run mypy
uv run ecg-verify
```

All four commands must exit zero. Test count is not a stable contract; record
the count and date rather than hard-coding it into a result claim.

## 4. Acquire and verify PTB-XL

Download the official PTB-XL 1.0.3 root metadata and 100 Hz WFDB records:

```powershell
uv run python scripts/download_ptbxl.py --workers 16
uv run python scripts/download_ptbxl.py --verify-only
```

The destination is `data/raw/ptb-xl/1.0.3/`. Downloads use resumable `.part`
files. Success requires 21,799 unique `filename_lr` records and verification of
every selected file represented in the official `SHA256SUMS.txt` inventory.

For the canonical research dataset, do not use `--allow-missing-checksums`.
`--force` is a recovery option that redownloads all selected files, not a normal
reproduction step.

## 5. Build and verify the manifest

```powershell
uv run python scripts/build_manifest.py
```

The canonical invocation deliberately omits fixture/development flags:

- do not use `--skip-file-checks`;
- do not use `--allow-noncanonical-counts`;
- do not use `--include-unlabeled`.

The strict builder verifies 21,799 source records, 18,869 source patients,
folds 1-10, patient-fold isolation, all waveform files, and these overlapping
source label counts: NORM 9,514; MI 5,469; STTC 5,235; CD 4,898; HYP 2,649.
After excluding 411 records with no mapped diagnostic superclass, the current
canonical manifest contains 21,388 records from 18,617 patients.

Artifacts are written to `data/manifests/`:

| Artifact | Current local SHA-256 |
|---|---|
| `ptbxl_superclasses_v1.0.3.csv` | `ff771ec783dd1665c8e59f497be0f624ed521fd34a73c9e70e2a9783b44ec49c` |
| `ptbxl_superclasses_v1.0.3.parquet` | `563a2b715cc6f6657b04c2f67d813fd7c30a696210740f97c55a070f157579a0` |
| `ptbxl_superclasses_v1.0.3.summary.json` | `7e7199e0378a213bc29b5fe6f1ae3f9e0eda0350d5e07c56fd0d90aeda19b8a6` |
| `ptbxl_superclasses_v1.0.3.sha256` | inventory of the three hashes above |

The summary also binds the source metadata:

- `ptbxl_database.csv`:
  `7600de9c1b27d181d850b3c6038a35d7c3ddb6bb33b702e3a20252a6859d216b`;
- `scp_statements.csv`:
  `ad05b0b1fcae83bb1230755ad9cfc7c96f303feddc08a4a9ad5bdc9ca63bac8f`.

Inspect the inventory and independently calculate the three file hashes:

```powershell
Get-Content data\manifests\ptbxl_superclasses_v1.0.3.sha256
Get-FileHash @(
  "data\manifests\ptbxl_superclasses_v1.0.3.csv",
  "data\manifests\ptbxl_superclasses_v1.0.3.parquet",
  "data\manifests\ptbxl_superclasses_v1.0.3.summary.json"
) -Algorithm SHA256
```

Hash differences require investigation. Do not silently update the model card
to accept a new manifest.

## 6. Prove the dataset contract and freeze normalization

Load real records from representative non-test roles:

```powershell
uv run python scripts/smoke_dataset.py --folds 1 8 9
```

Each load must produce finite float32 tensors shaped `[12, 1000]` and targets
shaped `[5]`. The lead order is fixed as:

```text
I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
```

Compute population per-lead mean and standard deviation from folds 1-7 only:

```powershell
uv run python scripts/compute_normalization.py
Get-FileHash artifacts\preprocessing\ptbxl_v1.0.3_train_folds_1-7_normalization.json -Algorithm SHA256
```

The expected normalization provenance is:

| Field | Value |
|---|---|
| Training folds | 1-7 |
| Records | 14,955 |
| Samples | 14,955,000 |
| Sampling frequency | 100 Hz |
| Samples per record | 1,000 |
| Selected-row manifest hash | `55dd86001dee2006cb241ff8b4f3970d8fcbb1ae9ecd430f5dd61478673ce235` |
| Current normalization file SHA-256 | `4a6cb489098361d8221403c14871c242672c346975af3a07f731ceac97264363` |

The selected-row hash is not the Parquet file hash. It hashes normalized fold,
record-path, and target content for the exact folds-1-7 rows. Development and
refit runners recompute it and fail if it no longer matches normalization
provenance.

Optional descriptive figures use development records only:

```powershell
uv run python scripts/plot_dataset_overview.py
```

## 7. Synthetic capacity and throughput benchmark

First prove the benchmark path without GPU dependence:

```powershell
uv run python scripts/benchmark_models.py --cpu-smoke `
  --output artifacts\benchmarks\models_cpu_smoke.json
```

Then measure the matched models under one equal workload:

```powershell
uv run python scripts/benchmark_models.py `
  --models matched_resnet matched_transformer `
  --device cuda `
  --precision bf16 `
  --batch-size 128 `
  --warmup-steps 3 `
  --steps 10 `
  --probe-max-batch 512 `
  --seed 2026 `
  --output artifacts\benchmarks\matched_bf16_seed2026.json
```

The benchmark uses synthetic `[batch, 12, 1000]` inputs and does not access
PTB-XL. Its JSON records parameter counts, step latencies, throughput, peak
allocated/reserved VRAM, workload, seed, and environment. A valid run requires
finite loss and successful train steps. There is no checked-in minimum
throughput or maximum-VRAM acceptance threshold, so these measurements are
descriptive, not a performance pass/fail claim.

The matched-capacity code does enforce exact parameter counts:

| Model | Trainable parameters |
|---|---:|
| 1D ResNet | 8,739,973 |
| ECG transformer | 8,726,833 |

The transformer/ResNet ratio is approximately `0.9984966`, an absolute gap of
approximately `0.1503%`. The preset's configured tolerance is 15%; exact count
drift also raises an error.

## 8. Training flow

### 8.1 Smoke training

```powershell
uv run python scripts/train.py --config configs/train_smoke.yaml `
  --protocol configs/protocol.yaml
```

The smoke config trains a small ResNet for at most two epochs on at most 1,024
training records. It is a wiring check, not benchmark evidence.

Training run directories are immutable by convention and the runner refuses to
reuse an existing path. The checked config writes to
`runs/development/smoke_resnet_seed2026/`. If that directory already exists,
copy the YAML, assign a new `run_name`, and run the copy. Preserve the old run;
do not delete it merely to reuse a name.

### 8.2 Matched development runs

```powershell
uv run python scripts/train.py --config configs/train_resnet_matched.yaml `
  --protocol configs/protocol.yaml

uv run python scripts/train.py --config configs/train_transformer_matched.yaml `
  --protocol configs/protocol.yaml
```

Both configs use batch 128, AdamW, BCE-with-logits loss, learning rate 0.001,
weight decay 0.01, five warmup epochs, warmup-cosine scheduling, gradient norm
1.0, at most 50 epochs, and fold-8 macro AUROC early stopping with patience 10
and minimum delta 0.0001. Both checked configs currently contain seed 2026.

The primary equal-budget search is one paired schema-v2 comparison. Inspect its
completed state without writing study data:

```powershell
uv run python scripts/sweep.py preflight
uv run python scripts/sweep.py status
```

The plan contains 12 Latin-hypercube candidates. Both architectures received
the exact same candidate rows and fixed development seed `2026`; candidate
order alternates which architecture runs first. The objective is uncalibrated
fold-8 macro ROC-AUC with all five labels defined, Optuna pruning is disabled,
and a failed attempt does not consume the completion budget. All 24 required
candidates are complete. Do not start a second sweep for this comparison.

The runner holds an exclusive comparison lock, rejects source/input/config
drift, verifies run metadata, histories, resolved configs, protocol files, and
both checkpoints before counting a candidate complete, and releases winners
only after both architectures have 12 verified completions. The immutable
summary is
`runs/sweeps/ptbxl_matched_equal_budget_v1/sweep_summary.json`, SHA-256
`04574c17773dea894efd84bf2ed5fa5be685ce3a6687deb3524a373ea6d8df6b`.

Each completed development run contains:

```text
runs/development/<run_name>/
  best.ckpt
  last.ckpt
  history.jsonl
  protocol.json
  resolved_config.json
  run_metadata.json
```

Only a run whose `run_metadata.json` has `status: complete` is reportable.
`best_validation_macro_auroc` and `best_epoch` are fold-8 development values.
They must never be copied into a final-test table.

The run metadata records source and resolved config hashes, protocol hash,
Parquet manifest hash, normalization file hash and provenance, seed, runtime,
timings, throughput, and VRAM. Checkpoints bind the protocol hash, resolved
config hash, manifest hash, epoch, model/optimizer state, and early-stopping
state.

### 8.3 Freeze the epoch budget and refit

The fixed confirmation seeds are `2026`, `2027`, and `2028`. Plan creation is
allowed only from a clean downstream commit B whose scientific kernel is
byte-identical to sweep commit A. Seed 2026 reuses each verified sweep winner;
only the four architecture/seed combinations for 2027 and 2028 train anew.
All six members receive fresh fold-8 prediction exports:

```powershell
git status --short
uv run python scripts/multiseed.py plan
uv run python scripts/multiseed.py status
uv run python scripts/multiseed.py run
```

The first command must print nothing. After an interruption, use `run --resume`
only against the exact persisted plan. The runner uses immutable attempt
directories, a three-attempt infrastructure-failure ceiling, an exclusive
writer lock, and complete source/hash verification before publishing any
`member_completion.json`.

After all six receipts verify, freeze them explicitly; do not discover a
subset by score or directory order:

```powershell
uv run python scripts/freeze_multiseed.py `
  --completion runs\confirmation\ptbxl_matched_equal_budget_v1\members\resnet1d\seed2026\member_completion.json `
  --completion runs\confirmation\ptbxl_matched_equal_budget_v1\members\resnet1d\seed2027\member_completion.json `
  --completion runs\confirmation\ptbxl_matched_equal_budget_v1\members\resnet1d\seed2028\member_completion.json `
  --completion runs\confirmation\ptbxl_matched_equal_budget_v1\members\ecg_transformer\seed2026\member_completion.json `
  --completion runs\confirmation\ptbxl_matched_equal_budget_v1\members\ecg_transformer\seed2027\member_completion.json `
  --completion runs\confirmation\ptbxl_matched_equal_budget_v1\members\ecg_transformer\seed2028\member_completion.json
```

The freeze applies the preregistered 0.005 mean-score margin, retains both
architectures, and derives one median selected-epoch budget per architecture.
It transactionally publishes exactly six immutable JSON recipes and commits
the freeze artifact last. Each prediction-recomputed fold-8 macro-AUROC must
agree with its training receipt within an absolute `1e-6`; both values and the
delta are preserved, and the architecture decision uses the recomputed values.
Run every generated recipe:

```powershell
Get-ChildItem `
  runs\confirmation\ptbxl_matched_equal_budget_v1\refit_recipes\*-refit.json | `
  ForEach-Object {
    uv run python scripts/refit.py --config $_.FullName `
      --protocol configs/protocol.yaml
  }
```

Each successful refit writes:

```text
runs/refit/ptbxl_matched_equal_budget_v1/<run_name>/attemptNN/
  final.ckpt                   # authoritative frozen-epoch checkpoint
  last.ckpt                    # crash-recovery checkpoint
  best_training_loss.ckpt      # diagnostic only; never model selection
  refit_history.jsonl
  protocol.json
  resolved_refit_config.json
  refit_metadata.json
  attempt_identity.json
  refit_completion.json        # downstream release authority
```

Each retry receives a new immutable attempt directory. The completion receipt
binds the freeze, recipe, source confirmation member, downstream commit and
lock, final checkpoint, resolved config, metadata, protocol, manifest, and
normalization. A refit uses folds 1-8, retains folds-1-7 normalization, starts
from fresh weights, and performs no validation or early stopping.

## 9. Export immutable fold predictions

`scripts/predict.py` validates the completed run metadata, resolved-config
wrapper and hash, checkpoint schema/config/protocol/manifest lineage,
normalization file hash and folds-1-7 provenance, exact model metadata/state,
patient-fold isolation, and requested fold role before constructing a dataset.

At downstream commit B, this low-level exporter still supports the legacy
single-run refit schema only. It is intentionally kept byte-identical to sweep
commit A while the four confirmation runs and six refits execute. Do not use
the legacy fold-9/fold-10 examples below for the post-sweep release. After all
six refit receipts exist, a separately recorded commit C must add and test
schema-v2 completion support; only then may `release_pipeline.py` cross the
six-refit bundle gate into fold 9.

A selected development checkpoint may export fold 8 only:

```powershell
uv run python scripts/predict.py `
  --checkpoint runs\development\resnet1d_matched_seed2026\best.ckpt `
  --resolved-config runs\development\resnet1d_matched_seed2026\resolved_config.json `
  --role model_selection `
  --output artifacts\predictions\resnet1d_seed2026_fold8.npz `
  --device cuda `
  --bf16
```

The authoritative frozen-refit checkpoint may export fold 9:

```powershell
uv run python scripts/predict.py `
  --checkpoint runs\refit\resnet1d_refit_folds1-8_seed2026\final.ckpt `
  --resolved-config runs\refit\resnet1d_refit_folds1-8_seed2026\resolved_refit_config.json `
  --role calibration `
  --output artifacts\predictions\resnet1d_seed2026_fold9.npz `
  --device cuda `
  --bf16
```

The exporter derives the sibling `run_metadata.json` or `refit_metadata.json`
and the data paths from the resolved config. Use `--run-metadata`, `--manifest`,
`--dataset-root`, or `--normalization` only to relocate an artifact; the stored
hashes and provenance must still match. `--batch-size` and `--num-workers` may
override inference throughput settings without changing the model lineage.

The exporter collects and aligns:

- unique `ecg_id`;
- `patient_id`;
- `strat_fold`;
- binary targets shaped `[n, 5]`;
- raw logits shaped `[n, 5]`;
- model name and seed;
- protocol, resolved refit config, and manifest hashes;
- exactly the fold assigned to `model_selection`, `calibration`, or
  token-gated `final_test`.

Saving creates an immutable `.npz` plus same-stem `.json` sidecar. The sidecar
records array shapes/dtypes, source metadata, row-alignment hash, NPZ size and
SHA-256, and a canonical artifact SHA-256. Existing pairs are never
overwritten. A second export to the same stem fails.

Fold 10 remains explicit and purpose-bound:

```powershell
uv run python scripts/predict.py `
  --checkpoint runs\refit\resnet1d_refit_folds1-8_seed2026\final.ckpt `
  --resolved-config runs\refit\resnet1d_refit_folds1-8_seed2026\resolved_refit_config.json `
  --role final_test `
  --output artifacts\predictions\resnet1d_seed2026_fold10.npz `
  --device cuda `
  --bf16 `
  --final-test-purpose "Confirmatory fold-10 predictions for frozen ResNet seed 2026" `
  --final-test-confirmation "I understand fold 10 is the one-time final test set."
```

Use a stable naming convention:

```text
artifacts/predictions/<model>_seed<seed>_fold9.npz
artifacts/predictions/<model>_seed<seed>_fold9.json
artifacts/predictions/<model>_seed<seed>_fold10.npz
artifacts/predictions/<model>_seed<seed>_fold10.json
```

## 10. Calibration and locked final report

The following commands consume the immutable prediction artifacts produced
above.

### 10.1 Fit fold-9 decisions

```powershell
uv run python -m ecg_trust.calibration_cli fit `
  --predictions artifacts\predictions\resnet1d_seed2026_fold9.npz `
  --protocol configs\protocol.yaml `
  --output artifacts\decisions\resnet1d_seed2026_fold9_decisions.json
```

Omitting `--coverage` uses the frozen default targets
`[1.0, 0.9, 0.8, 0.7, 0.5]`. If different targets are desired, declare them
before opening fold 10 and repeat `--coverage` for each value.

The non-overwriting decision JSON binds the fold-9 prediction hash and row
alignment, model/seed, protocol/config/manifest hashes, global temperature,
per-label F1 thresholds, normalized binary-entropy cutoffs, software versions,
and its own artifact hash.

### 10.2 Prepare subgroup metadata

The final-report command requires a JSON object with exactly this shape:

```json
{
  "ecg_id": [10001, 10002],
  "attributes": {
    "sex": [0, 1],
    "age_band": ["60-79", "80+"]
  }
}
```

`ecg_id` must exactly match the sorted fold-10 prediction order, and every
attribute must have one value per ECG. Choose subgroup definitions before
opening fold 10. The current repository does not include a subgroup-JSON
builder CLI.

### 10.3 Open fold 10 once

Only after all model, seed, calibration, threshold, gate, subgroup, bootstrap,
and reporting choices are frozen:

```powershell
uv run python -m ecg_trust.calibration_cli final-report `
  --decisions artifacts\decisions\resnet1d_seed2026_fold9_decisions.json `
  --predictions artifacts\predictions\resnet1d_seed2026_fold10.npz `
  --subgroups artifacts\subgroups\ptbxl_fold10_subgroups.json `
  --protocol configs\protocol.yaml `
  --output artifacts\final\resnet1d_seed2026_fold10_report.json `
  --final-test-purpose "Confirmatory report for frozen ResNet seed 2026" `
  --final-test-confirmation "I understand fold 10 is the one-time final test set." `
  --bootstrap-resamples 1000 `
  --bootstrap-seed 20260808 `
  --minimum-group-samples 30 `
  --minimum-group-patients 20
```

Run one formal command per preregistered model/seed. The report applies fold-9
choices without fitting anything. It contains:

- per-label and macro AUROC, average precision, Brier score, and 15-bin ECE;
- fixed-gate realized coverage, Hamming risk, exact-match accuracy, and
  per-label error rates;
- subgroup metrics and subgroup coverage under the shared global gates;
- patient-cluster percentile-bootstrap intervals, including valid/invalid
  replicate counts;
- source hashes and a non-overwriting report SHA-256.

The current final report does not emit final-set log loss, robustness results,
explanation-faithfulness results, cross-seed aggregation, or paired model
difference intervals. Those must not appear as if produced by this command.
The library does provide `paired_model_difference_intervals()` and prediction
alignment checks, but no paired-comparison CLI is currently checked in.

## 11. Artifact layout and provenance ledger

Defaults and recommended generated paths are:

```text
data/raw/ptb-xl/1.0.3/             official source files and 100 Hz WFDB records
data/manifests/                    CSV, Parquet, summary, and SHA inventory
artifacts/preprocessing/           folds-1-7 normalization JSON
artifacts/benchmarks/              synthetic benchmark JSON
artifacts/predictions/             immutable NPZ/JSON prediction pairs
artifacts/decisions/               immutable fold-9 decision JSON
artifacts/subgroups/               preregistered aligned subgroup JSON
artifacts/final/                   immutable fold-10 final reports
runs/development/<run_name>/       development checkpoints and metadata
runs/refit/<run_name>/             frozen-refit checkpoints and metadata
reports/figures/data/              generated descriptive figures
```

`data/raw/`, `data/manifests/`, `artifacts/`, `runs/`, and checkpoint files are
Git-ignored. Back them up independently; a Git commit alone cannot reproduce or
recover them.

For every reported row, retain this provenance chain:

```text
Git revision / dirty-state record
  -> protocol hash
  -> source config hash
  -> resolved config hash
  -> manifest Parquet hash
  -> folds-1-7 normalization file and selected-row hashes
  -> development best-checkpoint hash and selected epoch
  -> resolved refit config and final-checkpoint hash
  -> fold-9 prediction artifact hash
  -> calibration-decision artifact hash
  -> fold-10 prediction artifact hash
  -> final-report hash
```

Do not mix prefixed and unprefixed hashes silently. Protocol and canonical
config/prediction artifact hashes use `sha256:<digest>` in several APIs, while
manifest and file hashes in run metadata are stored as the 64-character digest.

## 12. Stage gates

| Stage | Pass condition |
|---|---|
| Environment | Locked sync succeeds and `ecg-verify` reports PASS on CUDA |
| Source data | `download_ptbxl.py --verify-only` exits zero against official SHA-256 values |
| Manifest | Strict canonical counts, waveform existence, and patient-fold isolation pass; hash inventory matches |
| Signal contract | Real folds 1/8/9 load as finite float32 `[12, 1000]` in canonical lead order |
| Normalization | Provenance says folds 1-7, 14,955 records, and the selected-row hash matches |
| Code quality | pytest, Ruff, strict mypy, and CUDA verification all exit zero |
| Capacity benchmark | Finite train steps; exact matched parameter counts; descriptive compute JSON saved |
| Smoke training | Unique run completes and writes valid best/last checkpoints; no scientific claim attached |
| Development | `status: complete`; fold-8 selection only; all seeds/configs use equal declared budgets |
| Frozen refit | Selected epoch + 1 verified; no early stopping; `final.ckpt` is authoritative |
| Calibration | Integrity-bound fold-9-only prediction and decision artifacts |
| Final | Frozen choices, explicit token, fold-10-only prediction, non-overwriting report, hashes archived |

The literature-scale macro-AUROC near 0.92-0.93 is a contextual sanity check,
not an automated quality gate in the current code. A discrepancy should trigger
an investigation; it must not trigger repeated fold-10 tuning.

## 13. Reproduction record checklist

Before publishing or filling the model card, capture:

- date, operator, machine, and Git revision/dirty state;
- `uv.lock` hash and full `ecg-verify` JSON;
- protocol, source config, resolved config, manifest, and normalization hashes;
- every seed and run name, including failed or excluded runs with reasons;
- development and refit metadata plus checkpoint hashes;
- immutable fold-9/fold-10 prediction sidecars;
- calibration-decision and final-report hashes;
- exact bootstrap settings and subgroup JSON hash;
- all code-quality command outputs;
- a statement that fold 10 was opened only after freezing choices, or an
  explicit exploratory label if that condition was violated.
