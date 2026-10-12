"""Versioned quality-gate configuration independent of waveform computation.

Import this module to validate thresholds or inspect research presets without
loading NumPy or the assessment engine. Existing signal_quality imports remain
supported through re-exports; no preset is promoted by constructing it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def _is_finite_real(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


@dataclass(frozen=True, slots=True)
class SignalQualityConfig:
    """Validated and immutable thresholds for the canonical quality gate.

    The defaults form a versioned research configuration for a physical-mV,
    100 Hz, ten-second, 12-lead signal.  Alternative values require explicit
    construction and therefore cannot silently change the release behavior.
    """

    version: str = "canonical-12x1000-mv-v1"
    expected_leads: tuple[str, ...] = (
        "I",
        "II",
        "III",
        "aVR",
        "aVL",
        "aVF",
        "V1",
        "V2",
        "V3",
        "V4",
        "V5",
        "V6",
    )
    expected_sample_count: int = 1_000
    expected_sample_rate_hz: float = 100.0
    expected_duration_seconds: float = 10.0
    expected_unit: str = "mV"
    metadata_absolute_tolerance: float = 1e-9

    flat_step_tolerance_mv: float = 5e-4
    flat_warn_peak_to_peak_mv: float = 0.08
    flat_reacquire_peak_to_peak_mv: float = 0.03
    flat_warn_std_mv: float = 0.015
    flat_reacquire_std_mv: float = 0.006
    flat_warn_fraction: float = 0.985
    flat_reacquire_fraction: float = 0.997
    flat_run_warn_seconds: float | None = None
    flat_run_reacquire_seconds: float | None = None
    duplicate_lead_tolerance_mv: float | None = None

    clipping_equality_tolerance_mv: float = 1e-8
    clipping_warn_fraction: float = 0.05
    clipping_reacquire_fraction: float = 0.10
    clipping_warn_run_samples: int = 4
    clipping_reacquire_run_samples: int = 8

    amplitude_warn_mv: float = 5.0
    amplitude_reacquire_mv: float = 8.0
    spike_warn_step_mv: float = 2.0
    spike_reacquire_step_mv: float = 4.0

    spectral_floor_hz: float = 0.1
    baseline_wander_upper_hz: float = 0.7
    baseline_wander_warn_ratio: float = 0.45
    baseline_wander_reacquire_ratio: float = 0.65
    powerline_band_half_width_hz: float = 0.6
    powerline_warn_ratio: float = 0.20
    powerline_reacquire_ratio: float = 0.40
    high_frequency_lower_hz: float = 20.0
    high_frequency_upper_hz: float = 45.0
    high_frequency_warn_ratio: float = 0.30
    high_frequency_reacquire_ratio: float = 0.50

    limb_consistency_warn_ratio: float = 0.12
    limb_consistency_reacquire_ratio: float = 0.25
    dominant_polarity_quantile: float = 0.90
    reversal_min_polarity: float = 0.25
    reversal_max_inverse_correlation: float = -0.45
    reversal_min_evidence_fraction: float = 0.75

    def __post_init__(self) -> None:
        """Reject incoherent threshold sets at construction time."""

        canonical_leads = (
            "I",
            "II",
            "III",
            "aVR",
            "aVL",
            "aVF",
            "V1",
            "V2",
            "V3",
            "V4",
            "V5",
            "V6",
        )
        if not self.version.strip():
            raise ValueError("version must be non-empty")
        if self.expected_leads != canonical_leads:
            raise ValueError("expected_leads must use the canonical 12-lead order")
        for count_name, count in (
            ("expected_sample_count", self.expected_sample_count),
            ("clipping_warn_run_samples", self.clipping_warn_run_samples),
            ("clipping_reacquire_run_samples", self.clipping_reacquire_run_samples),
        ):
            if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                raise ValueError(f"{count_name} must be a positive integer")
        if not self.expected_unit.strip():
            raise ValueError("expected_unit must be non-empty")

        positive_values = {
            "expected_sample_rate_hz": self.expected_sample_rate_hz,
            "expected_duration_seconds": self.expected_duration_seconds,
            "metadata_absolute_tolerance": self.metadata_absolute_tolerance,
            "flat_step_tolerance_mv": self.flat_step_tolerance_mv,
            "flat_warn_peak_to_peak_mv": self.flat_warn_peak_to_peak_mv,
            "flat_reacquire_peak_to_peak_mv": self.flat_reacquire_peak_to_peak_mv,
            "flat_warn_std_mv": self.flat_warn_std_mv,
            "flat_reacquire_std_mv": self.flat_reacquire_std_mv,
            "clipping_equality_tolerance_mv": self.clipping_equality_tolerance_mv,
            "amplitude_warn_mv": self.amplitude_warn_mv,
            "amplitude_reacquire_mv": self.amplitude_reacquire_mv,
            "spike_warn_step_mv": self.spike_warn_step_mv,
            "spike_reacquire_step_mv": self.spike_reacquire_step_mv,
            "spectral_floor_hz": self.spectral_floor_hz,
            "baseline_wander_upper_hz": self.baseline_wander_upper_hz,
            "powerline_band_half_width_hz": self.powerline_band_half_width_hz,
            "high_frequency_lower_hz": self.high_frequency_lower_hz,
            "high_frequency_upper_hz": self.high_frequency_upper_hz,
            "limb_consistency_warn_ratio": self.limb_consistency_warn_ratio,
            "limb_consistency_reacquire_ratio": self.limb_consistency_reacquire_ratio,
            "reversal_min_polarity": self.reversal_min_polarity,
        }
        for name, value in positive_values.items():
            if not _is_finite_real(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")

        ratios = {
            "flat_warn_fraction": self.flat_warn_fraction,
            "flat_reacquire_fraction": self.flat_reacquire_fraction,
            "clipping_warn_fraction": self.clipping_warn_fraction,
            "clipping_reacquire_fraction": self.clipping_reacquire_fraction,
            "baseline_wander_warn_ratio": self.baseline_wander_warn_ratio,
            "baseline_wander_reacquire_ratio": self.baseline_wander_reacquire_ratio,
            "powerline_warn_ratio": self.powerline_warn_ratio,
            "powerline_reacquire_ratio": self.powerline_reacquire_ratio,
            "high_frequency_warn_ratio": self.high_frequency_warn_ratio,
            "high_frequency_reacquire_ratio": self.high_frequency_reacquire_ratio,
            "dominant_polarity_quantile": self.dominant_polarity_quantile,
            "reversal_min_evidence_fraction": self.reversal_min_evidence_fraction,
        }
        for name, value in ratios.items():
            if not _is_finite_real(value) or not 0.0 < value < 1.0:
                raise ValueError(f"{name} must be strictly between zero and one")

        if not _is_finite_real(self.reversal_max_inverse_correlation):
            raise ValueError("reversal_max_inverse_correlation must be finite")
        if not -1.0 < self.reversal_max_inverse_correlation < 0.0:
            raise ValueError("reversal_max_inverse_correlation must be between -1 and zero")
        expected_samples = self.expected_sample_rate_hz * self.expected_duration_seconds
        if not math.isclose(
            expected_samples,
            float(self.expected_sample_count),
            abs_tol=self.metadata_absolute_tolerance,
        ):
            raise ValueError("expected_sample_count must equal sample rate multiplied by duration")
        if not (
            self.expected_sample_count == 1_000
            and math.isclose(self.expected_sample_rate_hz, 100.0)
            and math.isclose(self.expected_duration_seconds, 10.0)
            and self.expected_unit == "mV"
        ):
            raise ValueError("the validated signal contract is exactly 12x1000 physical mV")
        if not (
            self.flat_reacquire_peak_to_peak_mv < self.flat_warn_peak_to_peak_mv
            and self.flat_reacquire_std_mv < self.flat_warn_std_mv
            and self.flat_warn_fraction < self.flat_reacquire_fraction
        ):
            raise ValueError("flatline warning and reacquisition thresholds are incoherent")
        if (self.flat_run_warn_seconds is None) != (self.flat_run_reacquire_seconds is None):
            raise ValueError("flat-run thresholds must be enabled or disabled together")
        if self.flat_run_warn_seconds is not None and self.flat_run_reacquire_seconds is not None:
            for name, value in (
                ("flat_run_warn_seconds", self.flat_run_warn_seconds),
                ("flat_run_reacquire_seconds", self.flat_run_reacquire_seconds),
            ):
                if (
                    not _is_finite_real(value)
                    or value <= 0.0
                ):
                    raise ValueError(f"{name} must be finite and positive")
            if not (
                self.flat_run_warn_seconds
                < self.flat_run_reacquire_seconds
                <= self.expected_duration_seconds
            ):
                raise ValueError(
                    "flat-run warning must precede reacquisition within the record duration"
                )
        tolerance = self.duplicate_lead_tolerance_mv
        if tolerance is not None and (
            not _is_finite_real(tolerance)
            or tolerance < 0.0
        ):
            raise ValueError("duplicate_lead_tolerance_mv must be finite and non-negative")
        if not (
            self.clipping_warn_fraction < self.clipping_reacquire_fraction
            and self.clipping_warn_run_samples < self.clipping_reacquire_run_samples
        ):
            raise ValueError("clipping warning thresholds must precede reacquisition")
        if not (
            self.amplitude_warn_mv < self.amplitude_reacquire_mv
            and self.spike_warn_step_mv < self.spike_reacquire_step_mv
        ):
            raise ValueError("amplitude and spike warning thresholds must be lower")
        if not (
            self.baseline_wander_warn_ratio < self.baseline_wander_reacquire_ratio
            and self.powerline_warn_ratio < self.powerline_reacquire_ratio
            and self.high_frequency_warn_ratio < self.high_frequency_reacquire_ratio
            and self.limb_consistency_warn_ratio < self.limb_consistency_reacquire_ratio
        ):
            raise ValueError("spectral and consistency warning thresholds must be lower")

        nyquist = self.expected_sample_rate_hz / 2.0
        if not (
            self.spectral_floor_hz
            < self.baseline_wander_upper_hz
            < self.high_frequency_lower_hz
            < self.high_frequency_upper_hz
            <= nyquist
        ):
            raise ValueError("spectral bands must be ordered and below Nyquist")


DEFAULT_SIGNAL_QUALITY_CONFIG = SignalQualityConfig()


# The v1 default stays frozen because historical protocols bind its version and
# behavior. v2 adds a duration rule for contiguous exactly-flat stretches, which
# whole-record flatline statistics cannot see (for example, a lead that drops
# out halfway through). Thresholds come from a development-fold check: on 400
# PTB-XL folds 1-8 records without an electrode-problem flag, the longest flat
# stretch was 0.28 s. v2 also rejects two canonical leads that carry the same
# waveform (an export or wiring fault); in the same sample the closest pair of
# distinct leads still differed by at least 0.035 mV somewhere in the record.
SENTINEL_V2_SIGNAL_QUALITY_CONFIG = SignalQualityConfig(
    version="canonical-12x1000-mv-v2",
    flat_run_warn_seconds=0.5,
    flat_run_reacquire_seconds=1.0,
    duplicate_lead_tolerance_mv=5e-4,
)
