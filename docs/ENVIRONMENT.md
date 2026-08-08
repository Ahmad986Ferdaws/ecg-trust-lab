# Verified development environment

**Verified:** August 8, 2026  
**Status:** Verified for CUDA training, immutable release stages, and the local research viewer

## Installed project environment

The environment is self-contained inside the project and does not modify the system Python:

- Python runtime: `.python/` — CPython 3.12.13
- Virtual environment: `.venv/`
- Download/build cache: `.uv-cache/`
- Reproducible lockfile: `uv.lock`

The runtime, virtual environment, and cache are intentionally ignored by version control. `pyproject.toml`, `.python-version`, and `uv.lock` define the reproducible environment.

## Core package versions

| Package | Installed version |
|---|---:|
| Python | 3.12.13 |
| PyTorch | 2.13.0+cu130 |
| CUDA runtime bundled with PyTorch | 13.0 |
| cuDNN | 9.2.0 |
| NumPy | 2.5.1 |
| pandas | 3.0.5 |
| SciPy | 1.18.0 |
| scikit-learn | 1.9.0 |
| WFDB | 4.3.1 |
| Captum | 0.9.0 |
| Optuna | 4.9.0 |
| TensorBoard | 2.21.0 |
| FastAPI | 0.141.1 |
| Uvicorn | 0.52.1 |
| Plotly | 6.9.0 |
| Jinja2 | 3.1.6 |
| python-multipart | 0.0.32 |
| pytest | 9.1.1 |
| Ruff | 0.16.2 |
| mypy | 2.3.0 |

## GPU verification

`uv run ecg-verify` ran a three-step BF16 training loop through Conv1D and transformer layers using inputs shaped like a 100 Hz PTB-XL batch.

| Check | Result |
|---|---:|
| CUDA available | Yes |
| Device | NVIDIA GeForce RTX 5070 Ti Laptop GPU |
| Compute capability | 12.0 (`sm_120`) |
| Reported VRAM | 11.94 GiB |
| BF16 supported | Yes |
| Test batch | `[16, 12, 1000]` |
| Test output | `[16, 5]` |
| Training steps | 3 |
| Gradients finite | Yes |
| Peak allocated VRAM | 86.6 MiB |
| Timed loop | 0.537 seconds |
| Final status | PASS |

This smoke result verifies device execution and numerical viability; it is not a model-performance or production-throughput benchmark.

The system CUDA 12.4 toolkit remains installed but is not used by this environment. PyTorch carries its CUDA 13.0 runtime, which is compatible with the installed NVIDIA driver and Blackwell GPU.

## Quality checks

The integrated package gate on August 8, 2026 passed:

- `uv run pytest -q` — 250 tests passed (one upstream Starlette/httpx
  deprecation warning);
- `uv run ruff check src tests scripts` — all checks passed;
- `uv run mypy` — no issues in 35 source files;
- `uv run ecg-verify` — real CUDA BF16 training passed.

## Commands

```powershell
# Reproduce or update the environment from the lockfile
uv sync --all-groups

# Verify that PyTorch is using the GPU, not silently falling back to CPU
uv run ecg-verify

# Run project checks
uv run pytest
uv run ruff check src tests scripts
uv run mypy
```

Do not install a generic CPU-only `torch` package over this environment. `pyproject.toml` explicitly sources PyTorch from the official CUDA 13.0 wheel index.
