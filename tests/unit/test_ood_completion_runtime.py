from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import ecg_trust.ood_completion.runtime as runtime_module
from ecg_trust.ood_completion.models import canonical_sha256
from ecg_trust.ood_completion.runtime import (
    OODDeterminismError,
    OODRuntimeError,
    RepeatedEmbeddingExtraction,
    configure_deterministic_cuda,
)


def _runtime_arguments() -> dict[str, object]:
    return {
        "expected_device_name": "NVIDIA GeForce RTX 5070 Ti Laptop GPU",
        "expected_compute_capability": (12, 0),
        "expected_python_version": "3.12.13",
        "expected_torch_version": "2.13.0+cu130",
        "expected_cuda_runtime": "13.0",
        "expected_cudnn_version": 92_000,
        "expected_nvidia_driver_version": "596.49",
    }


def _patch_frozen_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime_torch = runtime_module.torch  # type: ignore[attr-defined]
    runtime_random = runtime_module.random  # type: ignore[attr-defined]
    runtime_numpy = runtime_module.np  # type: ignore[attr-defined]
    runtime_platform = runtime_module.platform  # type: ignore[attr-defined]
    fake_cuda = SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 1,
        set_device=lambda _device: None,
        get_device_name=lambda _device: "NVIDIA GeForce RTX 5070 Ti Laptop GPU",
        get_device_capability=lambda _device: (12, 0),
        manual_seed_all=lambda _seed: None,
    )
    fake_cudnn = SimpleNamespace(
        version=lambda: 92_000,
        deterministic=False,
        benchmark=True,
        allow_tf32=True,
    )
    fake_backends = SimpleNamespace(
        cudnn=fake_cudnn,
        cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
    )
    monkeypatch.setattr(runtime_torch, "cuda", fake_cuda)
    monkeypatch.setattr(runtime_torch, "backends", fake_backends)
    monkeypatch.setattr(runtime_torch, "version", SimpleNamespace(cuda="13.0"))
    monkeypatch.setattr(runtime_torch, "__version__", "2.13.0+cu130")
    monkeypatch.setattr(runtime_torch, "manual_seed", lambda _seed: None)
    monkeypatch.setattr(
        runtime_torch,
        "use_deterministic_algorithms",
        lambda _enabled: None,
    )
    monkeypatch.setattr(
        runtime_torch,
        "are_deterministic_algorithms_enabled",
        lambda: True,
    )
    monkeypatch.setattr(
        runtime_torch,
        "set_float32_matmul_precision",
        lambda _precision: None,
    )
    monkeypatch.setattr(runtime_random, "seed", lambda _seed: None)
    monkeypatch.setattr(runtime_numpy.random, "seed", lambda _seed: None)
    monkeypatch.setattr(runtime_platform, "python_version", lambda: "3.12.13")
    monkeypatch.setattr(runtime_module, "_nvidia_driver_version", lambda: "596.49")
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)


def test_repeated_embeddings_use_irreversibly_read_only_bytes() -> None:
    first = np.arange(2 * 512, dtype=np.float32).reshape(2, 512)
    repeated = first.copy()

    extraction = RepeatedEmbeddingExtraction(first=first, repeated=repeated)

    assert first.flags.writeable
    assert repeated.flags.writeable
    assert extraction.first.flags.c_contiguous
    assert extraction.repeated.flags.c_contiguous
    for array in (extraction.first, extraction.repeated):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.setflags(write=True)
    first[0, 0] = -1.0
    repeated[0, 0] = -1.0
    assert extraction.first[0, 0] == 0.0
    assert extraction.repeated[0, 0] == 0.0


def test_repeat_mismatch_has_a_distinct_determinism_failure_type() -> None:
    first = np.zeros((1, 512), dtype=np.float32)
    repeated = first.copy()
    repeated[0, 0] = 1.0

    with pytest.raises(OODDeterminismError, match="exact comparison"):
        RepeatedEmbeddingExtraction(first=first, repeated=repeated)


def test_runtime_summary_uses_validated_observed_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_frozen_cuda(monkeypatch)

    resolved = configure_deterministic_cuda(**_runtime_arguments())  # type: ignore[arg-type]

    summary = resolved.summary
    assert summary.resolved_device == "cuda:0"
    assert summary.device_type == "cuda"
    assert summary.device_name == "NVIDIA GeForce RTX 5070 Ti Laptop GPU"
    assert summary.compute_capability == "12.0"
    assert summary.python_version == "3.12.13"
    assert summary.torch_version == "2.13.0+cu130"
    assert summary.cuda_runtime_version == "13.0"
    assert summary.cudnn_version == 92_000
    assert summary.nvidia_driver_version == "596.49"
    assert resolved.runtime_sha256 == canonical_sha256(summary.model_dump(mode="json"))


def test_runtime_rejects_nonfrozen_expectations_before_cuda_access() -> None:
    arguments = _runtime_arguments()
    arguments["expected_python_version"] = "3.12.12"

    with pytest.raises(OODRuntimeError, match="expectations differ"):
        configure_deterministic_cuda(**arguments)  # type: ignore[arg-type]
