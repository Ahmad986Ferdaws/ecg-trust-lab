"""Trustworthy ECG classification research package."""

from __future__ import annotations

import os

# cuBLAS reads this contract when its first CUDA context is created.  Defining
# it at package import keeps deterministic training/inference valid even when a
# release preflight queries CUDA before the model runner seeds PyTorch.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

__version__ = "0.1.0"
