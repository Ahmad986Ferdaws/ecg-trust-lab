"""Synthetic, dataset-independent training benchmarks for ECG models."""

from __future__ import annotations

import gc
import math
import platform
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Literal

import torch
from torch import Tensor, nn

from ecg_trust.constants import LEADS, SUPERCLASSES
from ecg_trust.models import (
    MATCHED_CAPACITY_PRESET,
    ECGTransformer,
    ECGTransformerConfig,
    ResNet1D,
    ResNet1DConfig,
    count_parameters,
)

Precision = Literal["bf16", "fp32"]


@dataclass(frozen=True, slots=True)
class BenchmarkModelSpec:
    """Named architecture configuration included in benchmark JSON."""

    name: str
    architecture: Literal["resnet1d", "ecg_transformer"]
    comparison_group: Literal["practical_default", "matched_capacity"]
    config: ResNet1DConfig | ECGTransformerConfig

    def build(self) -> nn.Module:
        if self.architecture == "resnet1d":
            if not isinstance(self.config, ResNet1DConfig):
                raise TypeError("resnet1d spec requires ResNet1DConfig")
            return ResNet1D(self.config)
        if not isinstance(self.config, ECGTransformerConfig):
            raise TypeError("ecg_transformer spec requires ECGTransformerConfig")
        return ECGTransformer(self.config)

    def metadata(self) -> dict[str, object]:
        return {
            "name": self.name,
            "architecture": self.architecture,
            "comparison_group": self.comparison_group,
            "config": asdict(self.config),
        }


MODEL_SPECS: dict[str, BenchmarkModelSpec] = {
    "default_resnet": BenchmarkModelSpec(
        name="default_resnet",
        architecture="resnet1d",
        comparison_group="practical_default",
        config=ResNet1DConfig(),
    ),
    "default_transformer": BenchmarkModelSpec(
        name="default_transformer",
        architecture="ecg_transformer",
        comparison_group="practical_default",
        config=ECGTransformerConfig(),
    ),
    "matched_resnet": BenchmarkModelSpec(
        name="matched_resnet",
        architecture="resnet1d",
        comparison_group="matched_capacity",
        config=MATCHED_CAPACITY_PRESET.resnet_config,
    ),
    "matched_transformer": BenchmarkModelSpec(
        name="matched_transformer",
        architecture="ecg_transformer",
        comparison_group="matched_capacity",
        config=MATCHED_CAPACITY_PRESET.transformer_config,
    ),
}


def percentile(values: Sequence[float], quantile: float) -> float:
    """Compute a linearly interpolated percentile without a NumPy dependency."""

    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must lie in [0, 1]")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


@dataclass(frozen=True, slots=True)
class TrainBenchmarkResult:
    """One model/batch training benchmark with JSON-safe serialization."""

    model_name: str
    batch_size: int
    warmup_steps: int
    measured_steps: int
    precision: Precision
    trainable_parameters: int
    total_parameters: int
    step_latencies_ms: tuple[float, ...]
    median_step_latency_ms: float
    p95_step_latency_ms: float
    throughput_samples_per_second: float
    peak_allocated_vram_mib: float | None
    peak_reserved_vram_mib: float | None
    final_loss: float

    def to_dict(self) -> dict[str, object]:
        return {
            "model_name": self.model_name,
            "batch_size": self.batch_size,
            "warmup_steps": self.warmup_steps,
            "measured_steps": self.measured_steps,
            "precision": self.precision,
            "parameter_counts": {
                "trainable": self.trainable_parameters,
                "total": self.total_parameters,
            },
            "step_latencies_ms": list(self.step_latencies_ms),
            "median_step_latency_ms": self.median_step_latency_ms,
            "p95_step_latency_ms": self.p95_step_latency_ms,
            "throughput_samples_per_second": self.throughput_samples_per_second,
            "peak_allocated_vram_mib": self.peak_allocated_vram_mib,
            "peak_reserved_vram_mib": self.peak_reserved_vram_mib,
            "final_loss": self.final_loss,
        }


def _validate_benchmark_request(
    device: torch.device,
    precision: Precision,
    batch_size: int,
    warmup_steps: int,
    measured_steps: int,
) -> None:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if warmup_steps < 0:
        raise ValueError("warmup_steps cannot be negative")
    if measured_steps < 1:
        raise ValueError("measured_steps must be positive")
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("benchmark device must be CPU or CUDA")
    if precision == "bf16" and device.type != "cuda":
        raise ValueError("BF16 benchmark precision is supported only on CUDA")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("the selected CUDA device does not support BF16")


def _train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    inputs: Tensor,
    targets: Tensor,
    *,
    precision: Precision,
) -> Tensor:
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(
        device_type=inputs.device.type,
        dtype=torch.bfloat16,
        enabled=precision == "bf16",
    ):
        logits = model(inputs)
        if not isinstance(logits, Tensor) or logits.shape != targets.shape:
            observed = (
                type(logits).__name__ if not isinstance(logits, Tensor) else tuple(logits.shape)
            )
            raise ValueError(
                f"model must return raw logits shaped {tuple(targets.shape)}, got {observed}"
            )
        loss = nn.functional.binary_cross_entropy_with_logits(logits, targets)
    loss.backward()  # type: ignore[no-untyped-call]
    optimizer.step()
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite benchmark loss")
    return loss.detach()


def _timed_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    inputs: Tensor,
    targets: Tensor,
    *,
    precision: Precision,
) -> tuple[Tensor, float]:
    if inputs.device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
        end = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
        start.record()
        loss = _train_step(model, optimizer, inputs, targets, precision=precision)
        end.record()
        end.synchronize()
        return loss, float(start.elapsed_time(end))
    start_time = time.perf_counter()
    loss = _train_step(model, optimizer, inputs, targets, precision=precision)
    return loss, (time.perf_counter() - start_time) * 1000.0


def benchmark_train_steps(
    model: nn.Module,
    *,
    model_name: str,
    device: torch.device,
    batch_size: int = 16,
    warmup_steps: int = 3,
    measured_steps: int = 10,
    precision: Precision = "bf16",
    seed: int = 2026,
    learning_rate: float = 1e-3,
) -> TrainBenchmarkResult:
    """Measure synthetic forward/backward/optimizer training-step performance."""

    _validate_benchmark_request(device, precision, batch_size, warmup_steps, measured_steps)
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = model.to(device).train()
    inputs = torch.randn(batch_size, len(LEADS), 1000, device=device)
    targets = torch.randint(
        0,
        2,
        (batch_size, len(SUPERCLASSES)),
        device=device,
        dtype=torch.int64,
    ).to(torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for _ in range(warmup_steps):
        _train_step(model, optimizer, inputs, targets, precision=precision)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    latencies: list[float] = []
    loss = torch.tensor(float("nan"), device=device)
    for _ in range(measured_steps):
        loss, elapsed_ms = _timed_step(
            model,
            optimizer,
            inputs,
            targets,
            precision=precision,
        )
        latencies.append(elapsed_ms)

    elapsed_seconds = sum(latencies) / 1000.0
    peak_allocated: float | None = None
    peak_reserved: float | None = None
    if device.type == "cuda":
        peak_allocated = torch.cuda.max_memory_allocated(device) / (1024**2)
        peak_reserved = torch.cuda.max_memory_reserved(device) / (1024**2)

    return TrainBenchmarkResult(
        model_name=model_name,
        batch_size=batch_size,
        warmup_steps=warmup_steps,
        measured_steps=measured_steps,
        precision=precision,
        trainable_parameters=count_parameters(model),
        total_parameters=count_parameters(model, trainable_only=False),
        step_latencies_ms=tuple(latencies),
        median_step_latency_ms=statistics.median(latencies),
        p95_step_latency_ms=percentile(latencies, 0.95),
        throughput_samples_per_second=batch_size * measured_steps / elapsed_seconds,
        peak_allocated_vram_mib=peak_allocated,
        peak_reserved_vram_mib=peak_reserved,
        final_loss=float(loss.cpu()),
    )


@dataclass(frozen=True, slots=True)
class BatchProbeAttempt:
    batch_size: int
    succeeded: bool
    peak_allocated_vram_mib: float | None
    error: str | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class BatchProbeResult:
    maximum_requested_batch: int
    maximum_successful_batch: int | None
    first_oom_batch: int | None
    attempts: tuple[BatchProbeAttempt, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "maximum_requested_batch": self.maximum_requested_batch,
            "maximum_successful_batch": self.maximum_successful_batch,
            "first_oom_batch": self.first_oom_batch,
            "attempts": [attempt.to_dict() for attempt in self.attempts],
        }


def batch_probe_candidates(maximum_batch_size: int) -> tuple[int, ...]:
    """Return powers of two through a user-specified inclusive maximum."""

    if maximum_batch_size < 1:
        raise ValueError("maximum_batch_size must be positive")
    candidates: list[int] = []
    value = 1
    while value <= maximum_batch_size:
        candidates.append(value)
        value *= 2
    if candidates[-1] != maximum_batch_size:
        candidates.append(maximum_batch_size)
    return tuple(candidates)


def _is_cuda_oom(error: BaseException) -> bool:
    return isinstance(error, torch.OutOfMemoryError) or (
        isinstance(error, RuntimeError) and "out of memory" in str(error).lower()
    )


def probe_safe_batch_size(
    model_factory: Callable[[], nn.Module],
    *,
    model_name: str,
    device: torch.device,
    maximum_batch_size: int,
    precision: Precision = "bf16",
    seed: int = 2026,
) -> BatchProbeResult:
    """Probe increasing CUDA batches, recovering cleanly after the first OOM."""

    if device.type != "cuda":
        raise ValueError("safe batch probing is available only for CUDA")
    candidates = batch_probe_candidates(maximum_batch_size)
    attempts: list[BatchProbeAttempt] = []
    maximum_successful: int | None = None
    first_oom: int | None = None

    for batch_size in candidates:
        model: nn.Module | None = None
        try:
            model = model_factory()
            result = benchmark_train_steps(
                model,
                model_name=model_name,
                device=device,
                batch_size=batch_size,
                warmup_steps=0,
                measured_steps=1,
                precision=precision,
                seed=seed,
            )
            maximum_successful = batch_size
            attempts.append(
                BatchProbeAttempt(
                    batch_size=batch_size,
                    succeeded=True,
                    peak_allocated_vram_mib=result.peak_allocated_vram_mib,
                    error=None,
                )
            )
        except BaseException as error:
            if not _is_cuda_oom(error):
                raise
            first_oom = batch_size
            attempts.append(
                BatchProbeAttempt(
                    batch_size=batch_size,
                    succeeded=False,
                    peak_allocated_vram_mib=None,
                    error="CUDA out of memory",
                )
            )
            break
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()

    return BatchProbeResult(
        maximum_requested_batch=maximum_batch_size,
        maximum_successful_batch=maximum_successful,
        first_oom_batch=first_oom,
        attempts=tuple(attempts),
    )


def environment_metadata(device: torch.device) -> dict[str, object]:
    """Collect runtime metadata needed to interpret benchmark results."""

    metadata: dict[str, object] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),  # type: ignore[no-untyped-call]
        "device": str(device),
        "cpu_threads": torch.get_num_threads(),
    }
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        metadata["gpu"] = {
            "index": index,
            "name": properties.name,
            "compute_capability": f"{properties.major}.{properties.minor}",
            "total_vram_mib": properties.total_memory / (1024**2),
            "multiprocessors": properties.multi_processor_count,
            "bf16_supported": torch.cuda.is_bf16_supported(),
        }
    else:
        metadata["gpu"] = None
    return metadata


def selected_model_specs(names: Sequence[str]) -> tuple[BenchmarkModelSpec, ...]:
    """Resolve validated model names while preserving CLI order."""

    unknown = sorted(set(names).difference(MODEL_SPECS))
    if unknown:
        raise ValueError(f"unknown benchmark models: {unknown}")
    if not names:
        raise ValueError("at least one model must be selected")
    return tuple(MODEL_SPECS[name] for name in names)


__all__ = [
    "MODEL_SPECS",
    "BatchProbeAttempt",
    "BatchProbeResult",
    "BenchmarkModelSpec",
    "Precision",
    "TrainBenchmarkResult",
    "batch_probe_candidates",
    "benchmark_train_steps",
    "environment_metadata",
    "percentile",
    "probe_safe_batch_size",
    "selected_model_specs",
]
