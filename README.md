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
- The matched equal-budget sweep is complete: both architectures ran the same
  12-candidate plan. The fold-8 leaders were 0.931852 macro-AUROC for ResNet
  candidate 11 and 0.923927 for transformer candidate 6. These are
  model-selection results, not fold-10 test results.
- Calibration, abstention, patient-bootstrap, subgroup, corruption,
  attribution, immutable prediction, refit, final-report, and demo-backend
  layers are implemented and tested.
- The fixed seeds 2026/2027/2028 confirmation, immutable architecture freeze,
  six folds-1-8 refits, fold-9 release gate, and one-time fold-10 ledger are
  implemented with end-to-end provenance and adversarial integrity checks.
- A local FastAPI/Plotly research viewer is implemented for compatible WFDB
  uploads and curated examples; it will be bound to the final frozen model and
  calibration artifacts after the one-time evaluation.
- Repository gate on August 8, 2026: 250 tests pass; Ruff and strict mypy are
  clean across 35 source modules.
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
uv run python scripts/multiseed.py --help
uv run python scripts/freeze_multiseed.py --help
uv run python scripts/refit.py --help
uv run python scripts/release_pipeline.py --help
uv run python scripts/predict.py --help
uv run python -m ecg_trust.calibration_cli --help
uv run python scripts/demo_server.py --help
uv run pytest
uv run ruff check src tests scripts
uv run mypy
```

See the [verified environment record](docs/ENVIRONMENT.md) for exact versions
and CUDA details, and [reproducibility guide](docs/REPRODUCIBILITY.md) for the
complete artifact flow.

## Scope statement

This is a research and educational system, not a medical device. Its outputs must not be described as diagnoses, clinical recommendations, or evidence of real-world clinical safety without external, prospective validation.
