"""Deterministic, explicitly parameterized ECG corruption transforms.

These transforms support controlled sensitivity audits. They are not models of
clinical deployment shift and must not be interpreted as such.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import cast

import torch
from torch import Tensor

from ecg_trust.constants import LEADS


class CorruptionValidationError(ValueError):
    """Raised when a robustness transform violates the signal contract."""


class CorruptionKind(StrEnum):
    """Supported controlled corruptions."""

    BASELINE_WANDER = "baseline_wander"
    POWERLINE = "powerline"
    GAUSSIAN_NOISE = "gaussian_noise"
    AMPLITUDE_SCALE = "amplitude_scale"
    TIME_SHIFT = "time_shift"
    LEAD_DROPOUT = "lead_dropout"
    LEAD_PERMUTATION = "lead_permutation"


@dataclass(frozen=True, slots=True)
class CorruptionSpec:
    """Serializable corruption request for an audit matrix."""

    kind: CorruptionKind
    severity: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.severity) or self.severity < 0.0:
            raise CorruptionValidationError("severity must be finite and non-negative")

    def to_dict(self) -> dict[str, object]:
        return {"kind": self.kind.value, "severity": self.severity}


def _validate_waveforms(waveforms: Tensor) -> None:
    if not isinstance(waveforms, Tensor):
        raise TypeError("waveforms must be a torch.Tensor")
    if waveforms.ndim != 3 or waveforms.shape[1] != len(LEADS):
        raise CorruptionValidationError(f"waveforms must have shape [batch, {len(LEADS)}, time]")
    if waveforms.shape[0] < 1 or waveforms.shape[2] < 2:
        raise CorruptionValidationError("waveforms must contain a batch and at least two samples")
    if not waveforms.is_floating_point():
        raise CorruptionValidationError("waveforms must use a floating-point dtype")
    if not torch.isfinite(waveforms).all().item():
        raise CorruptionValidationError("waveforms must contain only finite values")


def _positive_finite(value: float, name: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise CorruptionValidationError(f"{name} must be finite and positive")
    return parsed


def _nonnegative_finite(value: float, name: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise CorruptionValidationError(f"{name} must be finite and non-negative")
    return parsed


def _example_rms(waveforms: Tensor) -> Tensor:
    return waveforms.square().mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-12)


def baseline_wander(
    waveforms: Tensor,
    *,
    amplitude_fraction: float,
    frequency_hz: float = 0.33,
    sampling_frequency_hz: float = 100.0,
    phase_radians: float = 0.0,
) -> Tensor:
    """Add a shared low-frequency sinusoid scaled by each record's RMS."""

    _validate_waveforms(waveforms)
    fraction = _nonnegative_finite(amplitude_fraction, "amplitude_fraction")
    frequency = _positive_finite(frequency_hz, "frequency_hz")
    sampling = _positive_finite(sampling_frequency_hz, "sampling_frequency_hz")
    if frequency >= sampling / 2.0:
        raise CorruptionValidationError("frequency_hz must be below Nyquist")
    if not math.isfinite(phase_radians):
        raise CorruptionValidationError("phase_radians must be finite")
    time = (
        torch.arange(waveforms.shape[2], device=waveforms.device, dtype=waveforms.dtype) / sampling
    )
    oscillation = torch.sin(2.0 * math.pi * frequency * time + phase_radians).view(1, 1, -1)
    return waveforms + fraction * _example_rms(waveforms) * oscillation


def powerline_interference(
    waveforms: Tensor,
    *,
    amplitude_fraction: float,
    frequency_hz: float = 50.0,
    sampling_frequency_hz: float = 100.0,
    phase_radians: float = math.pi / 2.0,
) -> Tensor:
    """Add a sinusoidal mains component, allowing the exact Nyquist frequency."""

    _validate_waveforms(waveforms)
    fraction = _nonnegative_finite(amplitude_fraction, "amplitude_fraction")
    frequency = _positive_finite(frequency_hz, "frequency_hz")
    sampling = _positive_finite(sampling_frequency_hz, "sampling_frequency_hz")
    if frequency > sampling / 2.0:
        raise CorruptionValidationError("frequency_hz must not exceed Nyquist")
    if not math.isfinite(phase_radians):
        raise CorruptionValidationError("phase_radians must be finite")
    time = (
        torch.arange(waveforms.shape[2], device=waveforms.device, dtype=waveforms.dtype) / sampling
    )
    interference = torch.sin(2.0 * math.pi * frequency * time + phase_radians).view(1, 1, -1)
    return waveforms + fraction * _example_rms(waveforms) * interference


def gaussian_noise_at_snr(
    waveforms: Tensor,
    *,
    snr_db: float,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Add zero-mean Gaussian noise at a per-record signal-to-noise ratio."""

    _validate_waveforms(waveforms)
    if not math.isfinite(snr_db):
        raise CorruptionValidationError("snr_db must be finite")
    noise = torch.randn(
        waveforms.shape,
        dtype=waveforms.dtype,
        device=waveforms.device,
        generator=generator,
    )
    noise_rms = noise.square().mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-12)
    target_noise_rms = _example_rms(waveforms) / (10.0 ** (snr_db / 20.0))
    return cast(Tensor, waveforms + noise * (target_noise_rms / noise_rms))


def amplitude_scale(waveforms: Tensor, *, factor: float) -> Tensor:
    """Scale every lead and time point without modifying the input tensor."""

    _validate_waveforms(waveforms)
    parsed = _positive_finite(factor, "factor")
    return waveforms * parsed


def zero_padded_time_shift(waveforms: Tensor, *, samples: int) -> Tensor:
    """Shift in time without circularly wrapping physiological content."""

    _validate_waveforms(waveforms)
    if isinstance(samples, bool) or not isinstance(samples, int):
        raise CorruptionValidationError("samples must be an integer")
    if abs(samples) >= waveforms.shape[2]:
        raise CorruptionValidationError("absolute shift must be shorter than the signal")
    shifted = torch.zeros_like(waveforms)
    if samples > 0:
        shifted[..., samples:] = waveforms[..., :-samples]
    elif samples < 0:
        shifted[..., :samples] = waveforms[..., -samples:]
    else:
        shifted.copy_(waveforms)
    return shifted


def drop_leads(waveforms: Tensor, *, lead_indices: Sequence[int]) -> Tensor:
    """Zero explicitly selected lead channels."""

    _validate_waveforms(waveforms)
    indices = tuple(lead_indices)
    if not indices:
        raise CorruptionValidationError("lead_indices must not be empty")
    if any(isinstance(index, bool) or not isinstance(index, int) for index in indices):
        raise CorruptionValidationError("lead_indices must contain integers")
    if len(set(indices)) != len(indices):
        raise CorruptionValidationError("lead_indices must be unique")
    if any(index < 0 or index >= len(LEADS) for index in indices):
        raise CorruptionValidationError("lead index is outside the canonical lead range")
    corrupted = waveforms.clone()
    corrupted[:, list(indices), :] = 0.0
    return corrupted


def permute_leads(waveforms: Tensor, *, permutation: Sequence[int]) -> Tensor:
    """Apply an explicit full lead permutation as a contract-violation audit."""

    _validate_waveforms(waveforms)
    order = tuple(permutation)
    if len(order) != len(LEADS) or set(order) != set(range(len(LEADS))):
        raise CorruptionValidationError(
            f"permutation must contain each index from 0 through {len(LEADS) - 1} once"
        )
    return waveforms[:, list(order), :].clone()
