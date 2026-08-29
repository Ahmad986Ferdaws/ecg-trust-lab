"""Deterministic CUDA embedding execution for OOD completion v1."""

from __future__ import annotations

import os
import platform
import random
import subprocess
from collections.abc import Sized
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from ecg_trust.models.resnet1d import ResNet1D
from ecg_trust.ood_completion.models import (
    EmbeddingRuntimeSummary,
    canonical_sha256,
)

Float32Array = NDArray[np.float32]

_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
_BATCH_SIZE = 128
_NUM_WORKERS = 4
_SEED = 2026


class OODRuntimeError(RuntimeError):
    """Raised when the frozen embedding runtime cannot be satisfied."""


class OODDeterminismError(OODRuntimeError):
    """Raised when repeated frozen inference is not bit-exact."""


@dataclass(frozen=True, slots=True)
class DeterministicCUDARuntime:
    """Resolved CUDA runtime plus its path-free canonical identity."""

    device: torch.device
    summary: EmbeddingRuntimeSummary
    runtime_sha256: str


@dataclass(frozen=True, slots=True)
class RepeatedEmbeddingExtraction:
    """Two exact full passes; only the first array is retained downstream."""

    first: Float32Array
    repeated: Float32Array

    def __post_init__(self) -> None:
        first = self.first
        repeated = self.repeated
        if first.dtype != np.dtype(np.float32) or repeated.dtype != np.dtype(np.float32):
            raise OODRuntimeError("embedding passes must remain float32")
        if first.ndim != 2 or first.shape != repeated.shape:
            raise OODRuntimeError("embedding passes must have the same two-dimensional shape")
        if first.shape[0] == 0 or first.shape[1] != 512:
            raise OODRuntimeError("embedding passes must contain [records, 512]")
        if not np.all(np.isfinite(first)) or not np.all(np.isfinite(repeated)):
            raise OODRuntimeError("embedding passes must contain only finite values")
        if not np.array_equal(first, repeated):
            raise OODDeterminismError("embedding passes differ under exact comparison")
        object.__setattr__(self, "first", _immutable_float32_copy(first))
        object.__setattr__(self, "repeated", _immutable_float32_copy(repeated))


def configure_deterministic_cuda(
    *,
    expected_device_name: str,
    expected_compute_capability: tuple[int, int],
    expected_python_version: str,
    expected_torch_version: str,
    expected_cuda_runtime: str,
    expected_cudnn_version: int,
    expected_nvidia_driver_version: str,
    nvidia_smi_executable: str | Path | None = None,
) -> DeterministicCUDARuntime:
    """Resolve and fail-closed configure the exact preregistered CUDA runtime."""

    if (
        expected_device_name != "NVIDIA GeForce RTX 5070 Ti Laptop GPU"
        or expected_compute_capability != (12, 0)
        or expected_python_version != "3.12.13"
        or expected_torch_version != "2.13.0+cu130"
        or expected_cuda_runtime != "13.0"
        or expected_cudnn_version != 92_000
        or expected_nvidia_driver_version != "596.49"
    ):
        raise OODRuntimeError("runtime expectations differ from the frozen v1 contract")
    prior_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if prior_workspace not in {None, _CUBLAS_WORKSPACE_CONFIG}:
        raise OODRuntimeError("CUBLAS_WORKSPACE_CONFIG conflicts with the frozen runtime")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = _CUBLAS_WORKSPACE_CONFIG
    if not torch.cuda.is_available() or torch.cuda.device_count() <= 0:
        raise OODRuntimeError("the frozen cuda:0 runtime is unavailable")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    observed_name = torch.cuda.get_device_name(device)
    observed_capability = torch.cuda.get_device_capability(device)
    observed_python = platform.python_version()
    observed_torch = str(torch.__version__)
    observed_cuda = torch.version.cuda
    observed_cudnn = torch.backends.cudnn.version()  # type: ignore[no-untyped-call]
    observed_driver = (
        _nvidia_driver_version()
        if nvidia_smi_executable is None
        else _nvidia_driver_version(nvidia_smi_executable)
    )
    resolved_device = str(device)
    observed_device_type = device.type
    observed_capability_text = f"{observed_capability[0]}.{observed_capability[1]}"
    if observed_name != expected_device_name:
        raise OODRuntimeError("CUDA device name differs from the frozen runtime")
    if observed_capability != expected_compute_capability:
        raise OODRuntimeError("CUDA compute capability differs from the frozen runtime")
    if observed_python != expected_python_version:
        raise OODRuntimeError("Python version differs from the frozen runtime")
    if observed_torch != expected_torch_version:
        raise OODRuntimeError("PyTorch version differs from the frozen runtime")
    if observed_cuda != expected_cuda_runtime:
        raise OODRuntimeError("CUDA runtime version differs from the frozen runtime")
    if observed_cudnn != expected_cudnn_version:
        raise OODRuntimeError("cuDNN version differs from the frozen runtime")
    if observed_driver != expected_nvidia_driver_version:
        raise OODRuntimeError("NVIDIA driver version differs from the frozen runtime")
    if resolved_device != "cuda:0" or observed_device_type != "cuda":
        raise OODRuntimeError("resolved CUDA device differs from the frozen runtime")
    if observed_capability_text != "12.0":
        raise OODRuntimeError("CUDA capability text differs from the frozen runtime")

    # These assertions narrow observed runtime values to the exact Literal fields
    # in EmbeddingRuntimeSummary. The emitted evidence remains the observed value.
    assert observed_name == "NVIDIA GeForce RTX 5070 Ti Laptop GPU"
    assert observed_capability_text == "12.0"
    assert observed_python == "3.12.13"
    assert observed_torch == "2.13.0+cu130"
    assert observed_cuda == "13.0"
    assert observed_cudnn == 92_000
    assert observed_driver == "596.49"
    assert resolved_device == "cuda:0"
    assert observed_device_type == "cuda"
    narrowed_resolved_device = cast(Literal["cuda:0"], resolved_device)
    narrowed_device_type = cast(Literal["cuda"], observed_device_type)
    narrowed_device_name = cast(
        Literal["NVIDIA GeForce RTX 5070 Ti Laptop GPU"], observed_name
    )
    narrowed_capability = cast(Literal["12.0"], observed_capability_text)
    narrowed_python = cast(Literal["3.12.13"], observed_python)
    narrowed_torch = cast(Literal["2.13.0+cu130"], observed_torch)
    narrowed_cuda = cast(Literal["13.0"], observed_cuda)
    narrowed_driver = cast(Literal["596.49"], observed_driver)

    random.seed(_SEED)
    np.random.seed(_SEED)
    torch.manual_seed(_SEED)
    torch.cuda.manual_seed_all(_SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    if not torch.are_deterministic_algorithms_enabled():
        raise OODRuntimeError("PyTorch deterministic algorithms could not be enabled")
    if torch.backends.cudnn.benchmark or not torch.backends.cudnn.deterministic:
        raise OODRuntimeError("cuDNN deterministic settings were not retained")
    if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
        raise OODRuntimeError("TF32 could not be disabled")

    summary = EmbeddingRuntimeSummary(
        requested_device="cuda:0",
        resolved_device=narrowed_resolved_device,
        device_type=narrowed_device_type,
        device_name=narrowed_device_name,
        compute_capability=narrowed_capability,
        python_version=narrowed_python,
        torch_version=narrowed_torch,
        cuda_runtime_version=narrowed_cuda,
        cudnn_version=observed_cudnn,
        nvidia_driver_version=narrowed_driver,
        tensor_precision="float32",
        autocast_enabled=False,
        bf16_enabled=False,
        tf32_enabled=False,
        deterministic_algorithms=True,
        cudnn_deterministic=True,
        cudnn_benchmark=False,
        cublas_workspace_config=":4096:8",
        inference_mode=True,
        model_eval_mode=True,
        shuffled=False,
        batch_size=128,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        extraction_passes=2,
        torch_compile_enabled=False,
        seed=2026,
    )
    return DeterministicCUDARuntime(
        device=device,
        summary=summary,
        runtime_sha256=canonical_sha256(summary.model_dump(mode="json")),
    )


def _immutable_float32_copy(value: Float32Array) -> Float32Array:
    """Copy a matrix onto immutable bytes so NumPy cannot re-enable writes."""

    contiguous = np.ascontiguousarray(value, dtype=np.float32)
    payload = contiguous.tobytes(order="C")
    immutable = np.frombuffer(payload, dtype=np.float32).reshape(contiguous.shape)
    return cast(Float32Array, immutable)


def _nvidia_driver_version(executable: str | Path = "nvidia-smi") -> str:
    requested: str | Path
    environment: dict[str, str] | None = None
    working_directory: Path | None = None
    if isinstance(executable, Path):
        try:
            requested = executable.resolve(strict=True)
        except OSError as error:
            raise OODRuntimeError("bound NVIDIA driver tool is unavailable") from error
        junction = getattr(requested, "is_junction", None)
        if (
            not requested.is_absolute()
            or requested.is_symlink()
            or bool(junction is not None and junction())
            or not requested.is_file()
        ):
            raise OODRuntimeError("bound NVIDIA driver tool is indirect")
        working_directory = requested.parent
        for name in ("ProgramFiles", "ProgramW6432"):
            if os.environ.get(name) != r"C:\Program Files":
                raise OODRuntimeError(
                    "bound NVIDIA driver environment is not canonical"
                )
        environment = {
            name: value
            for name in (
                "COMSPEC",
                "ProgramFiles",
                "ProgramW6432",
                "SYSTEMROOT",
                "TEMP",
                "TMP",
                "WINDIR",
            )
            if (value := os.environ.get(name))
        }
        environment["PATH"] = str(requested.parent)
    else:
        requested = executable
    try:
        completed = subprocess.run(
            [
                requested,
                "--id=0",
                "--query-gpu=driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            cwd=working_directory,
            env=environment,
            errors="strict",
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError, UnicodeError) as error:
        raise OODRuntimeError("NVIDIA driver version could not be queried") from error
    values = tuple(line.strip() for line in completed.stdout.splitlines() if line.strip())
    if completed.returncode != 0 or completed.stderr or len(values) != 1:
        raise OODRuntimeError("NVIDIA driver version query was not canonical")
    return values[0]


def prepare_resnet_for_embedding(
    model: nn.Module,
    *,
    runtime: DeterministicCUDARuntime,
) -> ResNet1D:
    """Move the exact frozen ResNet to CUDA without changing its state."""

    if type(model) is not ResNet1D:
        raise OODRuntimeError("OOD completion requires the exact ResNet1D architecture")
    if hasattr(model, "_orig_mod"):
        raise OODRuntimeError("torch.compile wrappers are forbidden")
    resnet = model
    resnet.requires_grad_(False)
    resnet.to(device=runtime.device, dtype=torch.float32)
    resnet.eval()
    if resnet.training:
        raise OODRuntimeError("embedding model must remain in evaluation mode")
    if any(parameter.dtype is not torch.float32 for parameter in resnet.parameters()):
        raise OODRuntimeError("all model parameters must remain float32")
    if any(parameter.device != runtime.device for parameter in resnet.parameters()):
        raise OODRuntimeError("all model parameters must reside on cuda:0")
    return resnet


def extract_embeddings_twice(
    model: ResNet1D,
    dataset: Dataset[tuple[Tensor, Tensor]],
    *,
    runtime: DeterministicCUDARuntime,
) -> RepeatedEmbeddingExtraction:
    """Run two complete ordered passes and require bit-exact representations."""

    if type(model) is not ResNet1D or model.training:
        raise OODRuntimeError("embedding extraction requires the exact eval-mode ResNet1D")
    record_count = len(cast(Sized, dataset))
    if record_count <= 0:
        raise OODRuntimeError("embedding dataset must not be empty")
    loader = DataLoader(
        dataset,
        batch_size=_BATCH_SIZE,
        shuffle=False,
        num_workers=_NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True,
    )
    try:
        first = _extract_single_pass(model, loader=loader, runtime=runtime)
        repeated = _extract_single_pass(model, loader=loader, runtime=runtime)
        if first.shape[0] != record_count:
            raise OODRuntimeError("embedding extraction did not return every dataset record")
        return RepeatedEmbeddingExtraction(first=first, repeated=repeated)
    finally:
        _shutdown_persistent_embedding_loader(loader)


def _shutdown_persistent_embedding_loader(
    loader: DataLoader[tuple[Tensor, Tensor]],
) -> None:
    """Synchronously close the frozen persistent-worker loader on every exit path."""

    if str(torch.__version__) != "2.13.0+cu130":
        raise OODRuntimeError("embedding loader cleanup requires the frozen PyTorch")
    if (
        loader.num_workers != _NUM_WORKERS
        or loader.pin_memory is not True
        or loader.persistent_workers is not True
    ):
        raise OODRuntimeError("embedding loader cleanup contract differs")

    # PyTorch has no public eager-shutdown API for a persistent DataLoader.  The
    # frozen 2.13.0 runtime exposes this exact iterator method; relying on the
    # iterator destructor is unsafe when an exception traceback retains frames.
    iterator = getattr(loader, "_iterator", None)
    if iterator is None:
        return
    shutdown_workers = getattr(iterator, "_shutdown_workers", None)
    workers = getattr(iterator, "_workers", None)
    pin_memory_thread = getattr(iterator, "_pin_memory_thread", None)
    if (
        not callable(shutdown_workers)
        or not isinstance(workers, list)
        or len(workers) != _NUM_WORKERS
        or getattr(iterator, "_persistent_workers", None) is not True
    ):
        raise OODRuntimeError("frozen embedding loader cleanup API is unavailable")

    cleanup_error: Exception | None = None
    try:
        shutdown_workers()
    except Exception as error:  # pragma: no cover - defensive frozen-runtime guard
        cleanup_error = error
    finally:
        # Break the loader-to-iterator reference even when the extraction
        # traceback itself is retained by the post-claim failure path.
        loader._iterator = None

    worker_state_invalid = False
    for worker in workers:
        is_alive = getattr(worker, "is_alive", None)
        if not callable(is_alive) or bool(is_alive()):
            worker_state_invalid = True
    pin_thread_alive = False
    if pin_memory_thread is not None:
        pin_is_alive = getattr(pin_memory_thread, "is_alive", None)
        pin_thread_alive = not callable(pin_is_alive) or bool(pin_is_alive())
    if (
        cleanup_error is not None
        or getattr(iterator, "_shutdown", None) is not True
        or worker_state_invalid
        or pin_thread_alive
    ):
        raise OODRuntimeError("embedding loader workers could not be closed") from cleanup_error


def _extract_single_pass(
    model: ResNet1D,
    *,
    loader: DataLoader[tuple[Tensor, Tensor]],
    runtime: DeterministicCUDARuntime,
) -> Float32Array:
    batches: list[Float32Array] = []
    with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
        for signals, _targets in loader:
            if signals.dtype is not torch.float32 or signals.ndim != 3:
                raise OODRuntimeError("dataset produced an invalid float32 ECG batch")
            inputs = signals.to(runtime.device, dtype=torch.float32, non_blocking=True)
            embeddings = model.forward_embedding(inputs)
            _validate_embedding_batch(
                embeddings,
                expected_batch=int(signals.shape[0]),
                expected_device=runtime.device,
            )
            batches.append(embeddings.detach().cpu().contiguous().numpy())
    if not batches:
        raise OODRuntimeError("embedding loader produced no batches")
    result = np.ascontiguousarray(np.concatenate(batches, axis=0), dtype=np.float32)
    if result.ndim != 2 or result.shape[1] != 512 or not np.all(np.isfinite(result)):
        raise OODRuntimeError("concatenated embedding output is invalid")
    return result


def _validate_embedding_batch(
    embeddings: Tensor,
    *,
    expected_batch: int,
    expected_device: torch.device,
) -> None:
    if embeddings.shape != (expected_batch, 512):
        raise OODRuntimeError("ResNet embedding batch has an invalid shape")
    if embeddings.dtype is not torch.float32 or embeddings.device != expected_device:
        raise OODRuntimeError("ResNet embeddings violated the CUDA float32 contract")
    if embeddings.requires_grad:
        raise OODRuntimeError("ResNet embeddings must be produced under inference mode")
    if not torch.isfinite(embeddings).all().item():
        raise OODRuntimeError("ResNet embeddings contain non-finite values")


__all__ = [
    "DeterministicCUDARuntime",
    "OODDeterminismError",
    "OODRuntimeError",
    "RepeatedEmbeddingExtraction",
    "configure_deterministic_cuda",
    "extract_embeddings_twice",
    "prepare_resnet_for_embedding",
]
