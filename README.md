# Trustworthy 12-lead ECG classification on PTB-XL

This project compares a 1D residual network with a patch-based ECG transformer on PTB-XL, then evaluates what matters beyond headline discrimination: probability calibration, selective abstention, subgroup behavior, robustness, and explanation faithfulness.

The canonical task is **multi-label prediction of the five PTB-XL diagnostic superclasses** (NORM, MI, STTC, CD, and HYP). The intended demo is a research-only interface that loads a compatible 12-lead ECG, shows calibrated probabilities and an abstention decision, and highlights waveform regions associated with each output.

Start with the concise [build order](docs/BUILD_ORDER.md) or the full
[research and implementation blueprint](docs/RESEARCH_BLUEPRINT.md). The
[data card](docs/DATA_CARD.md) records the verified dataset contract, and the
[preliminary development report](reports/DEVELOPMENT_RESULTS.md) separates
current fold-8 evidence from future final-test claims.

## Current status

- PTB-XL 1.0.3 100 Hz is downloaded: 21,799 records and 43,605 selected files
  pass the official SHA-256 inventory.
- The immutable task manifest contains 21,388 labeled ECGs from 18,617
  patients; all official source, label, fold, file, and patient-isolation gates
  pass.
- Train-only normalization is frozen from 14,955 records in folds 1–7 with
  provenance hash `55dd86001dee2006cb241ff8b4f3970d8fcbb1ae9ecd430f5dd61478673ce235`.
- Project-local Python 3.12.13, PyTorch 2.13.0 + CUDA 13.0, cuDNN 9.2, and BF16
  pass the real RTX 5070 Ti forward/backward check.
- Matched-capacity single-seed development runs are complete. On fold 8, the
  8.74M-parameter ResNet reached 0.9287 macro-AUROC and the 8.73M-parameter
  transformer reached 0.9237. These are model-selection results, not fold-10
  test results.
- Calibration, abstention, patient-bootstrap, subgroup, corruption,
  attribution, immutable prediction, refit, final-report, and demo-backend
  layers are implemented and tested.
- The schema-v2 paired sweep executor is ready: one deterministic 12-row
  Latin-hypercube plan is shared by both architectures, the HPO seed is fixed,
  execution alternates model order by candidate, and resume verifies immutable
  artifacts before counting a trial complete.
- Repository gate: 206 tests pass; Ruff and strict mypy are clean.
- Fold 9 and fold 10 have not been used for the reported development results.

## Development commands

Run these commands from the project directory:

```powershell
uv sync --all-groups
uv run ecg-verify
uv run python scripts/smoke_dataset.py
uv run python scripts/benchmark_models.py --help
uv run python scripts/train.py --config configs/train_smoke.yaml
uv run python scripts/sweep.py preflight
uv run python scripts/sweep.py status
uv run python scripts/refit.py --help
uv run python scripts/predict.py --help
uv run python -m ecg_trust.calibration_cli --help
uv run pytest
uv run ruff check src tests scripts
uv run mypy
```

See the [verified environment record](docs/ENVIRONMENT.md) for exact versions
and CUDA details, and [reproducibility guide](docs/REPRODUCIBILITY.md) for the
complete artifact flow.

## Scope statement

This is a research and educational system, not a medical device. Its outputs must not be described as diagnoses, clinical recommendations, or evidence of real-world clinical safety without external, prospective validation.
