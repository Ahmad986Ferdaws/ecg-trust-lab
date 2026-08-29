"""Deterministic, fail-closed assurance for canonical PTB-XL-style ECG inputs.

This module deliberately performs no diagnosis and never repairs, reorders, or
filters a waveform.  It validates a narrow physical-signal contract and emits
structured evidence that a downstream policy engine can use to stop inference
or request reacquisition.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


class QualityStatus(StrEnum):
    """Fail-closed disposition for an ECG quality assessment."""

    PASS = "pass"
    LIMITED = "limited"
    REACQUIRE = "reacquire"
    INVALID = "invalid"


class ReasonCode(StrEnum):
    """Stable machine-readable reasons produced by the assurance gate."""

    NON_NUMERIC_SIGNAL = "non_numeric_signal"
    NON_REAL_SIGNAL = "non_real_signal"
    WRONG_SIGNAL_SHAPE = "wrong_signal_shape"
    NONFINITE_SIGNAL = "nonfinite_signal"
    INVALID_METADATA = "invalid_metadata"
    LEAD_COUNT_MISMATCH = "lead_count_mismatch"
    MISSING_LEADS = "missing_leads"
    DUPLICATE_LEADS = "duplicate_leads"
    UNEXPECTED_LEADS = "unexpected_leads"
    LEAD_ORDER_MISMATCH = "lead_order_mismatch"
    SAMPLE_RATE_MISMATCH = "sample_rate_mismatch"
    DURATION_MISMATCH = "duration_mismatch"
    UNIT_COUNT_MISMATCH = "unit_count_mismatch"
    UNSUPPORTED_UNITS = "unsupported_units"
    FLATLINE = "flatline"
    CLIPPING_OR_SATURATION = "clipping_or_saturation"
    EXTREME_AMPLITUDE = "extreme_amplitude"
    EXTREME_SPIKES = "extreme_spikes"
    BASELINE_WANDER = "baseline_wander"
    POWERLINE_INTERFERENCE_50HZ = "powerline_interference_50hz"
    POWERLINE_INTERFERENCE_60HZ = "powerline_interference_60hz"
    HIGH_FREQUENCY_NOISE = "high_frequency_noise"
    LIMB_LEAD_INCONSISTENCY = "limb_lead_inconsistency"
    PROBABLE_LIMB_LEAD_REVERSAL = "probable_limb_lead_reversal"


class LimbLeadReversalKind(StrEnum):
    """Electrode reversals for which conservative morphology evidence exists."""

    RIGHT_ARM_LEFT_ARM = "right_arm_left_arm"
    RIGHT_ARM_LEFT_LEG = "right_arm_left_leg"


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
        if self.expected_sample_count <= 0:
            raise ValueError("expected_sample_count must be positive")
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
            if not math.isfinite(value) or value <= 0.0:
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
            if not math.isfinite(value) or not 0.0 < value < 1.0:
                raise ValueError(f"{name} must be strictly between zero and one")

        if not math.isfinite(self.reversal_max_inverse_correlation):
            raise ValueError("reversal_max_inverse_correlation must be finite")
        if not -1.0 < self.reversal_max_inverse_correlation < 0.0:
            raise ValueError("reversal_max_inverse_correlation must be between -1 and zero")
        if self.clipping_warn_run_samples <= 0 or self.clipping_reacquire_run_samples <= 0:
            raise ValueError("clipping run thresholds must be positive")

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


@dataclass(frozen=True, slots=True)
class SignalMetadata:
    """Metadata required to establish the physical waveform contract."""

    lead_names: tuple[str, ...]
    sample_rate_hz: float
    duration_seconds: float
    units: tuple[str, ...]

    @classmethod
    def canonical(
        cls,
        config: SignalQualityConfig = DEFAULT_SIGNAL_QUALITY_CONFIG,
    ) -> Self:
        """Construct metadata for the canonical signal contract."""

        return cls(
            lead_names=config.expected_leads,
            sample_rate_hz=config.expected_sample_rate_hz,
            duration_seconds=config.expected_duration_seconds,
            units=(config.expected_unit,) * len(config.expected_leads),
        )


@dataclass(frozen=True, slots=True)
class QualityIssue:
    """One deterministic validation or quality finding."""

    code: ReasonCode
    status: QualityStatus
    lead_name: str | None
    metric_name: str | None
    observed_value: float | None
    boundary_value: float | None


@dataclass(frozen=True, slots=True)
class LeadQualityMetrics:
    """Numerical evidence calculated for one lead in physical millivolts."""

    peak_to_peak_mv: float
    standard_deviation_mv: float
    maximum_absolute_amplitude_mv: float
    flat_step_fraction: float
    clipping_fraction: float
    longest_clipping_run_samples: int
    maximum_step_mv: float
    spike_step_fraction: float
    baseline_wander_power_ratio: float
    powerline_50hz_power_ratio: float
    powerline_60hz_power_ratio: float
    high_frequency_power_ratio: float


@dataclass(frozen=True, slots=True)
class LeadQualityFinding:
    """Quality disposition and evidence for one canonical lead."""

    lead_name: str
    lead_index: int
    status: QualityStatus
    reason_codes: tuple[ReasonCode, ...]
    issues: tuple[QualityIssue, ...]
    metrics: LeadQualityMetrics


@dataclass(frozen=True, slots=True)
class LimbLeadReversalEvidence:
    """Conservative morphology evidence, never an automatic correction."""

    probable_kind: LimbLeadReversalKind
    score: float
    evidence_codes: tuple[str, ...]
    dominant_polarities: tuple[tuple[str, float], ...]
    correlations: tuple[tuple[str, float], ...]


@dataclass(frozen=True, slots=True)
class SignalQualityReport:
    """Complete result of deterministic input assurance."""

    status: QualityStatus
    config_version: str
    global_issues: tuple[QualityIssue, ...]
    leads: tuple[LeadQualityFinding, ...]
    reversal_evidence: LimbLeadReversalEvidence | None

    @property
    def reason_codes(self) -> tuple[ReasonCode, ...]:
        """Return all reason codes once, preserving deterministic order."""

        codes = [issue.code for issue in self.global_issues]
        for lead in self.leads:
            codes.extend(lead.reason_codes)
        return tuple(dict.fromkeys(codes))

    @property
    def classification_allowed(self) -> bool:
        """Only an unqualified pass is eligible for downstream inference."""

        return self.status is QualityStatus.PASS


def assess_signal_quality(
    signal_mv: ArrayLike,
    metadata: SignalMetadata,
    *,
    config: SignalQualityConfig = DEFAULT_SIGNAL_QUALITY_CONFIG,
) -> SignalQualityReport:
    """Validate and assess a canonical ``(12, 1000)`` physical-mV ECG.

    The function is pure with respect to its arguments: it does not mutate,
    reorder, resample, filter, or otherwise repair the input.
    """

    try:
        raw_signal = np.asarray(signal_mv)
    except (TypeError, ValueError):
        issue = _global_issue(ReasonCode.NON_NUMERIC_SIGNAL, QualityStatus.INVALID)
        return _invalid_report(config, (issue,))
    if np.iscomplexobj(raw_signal) or np.issubdtype(raw_signal.dtype, np.bool_):
        issue = _global_issue(ReasonCode.NON_REAL_SIGNAL, QualityStatus.INVALID)
        return _invalid_report(config, (issue,))
    try:
        signal = np.asarray(raw_signal, dtype=np.float64)
    except (TypeError, ValueError):
        issue = _global_issue(ReasonCode.NON_NUMERIC_SIGNAL, QualityStatus.INVALID)
        return _invalid_report(config, (issue,))

    contract_issues = _validate_contract(signal, metadata, config)
    if contract_issues:
        return _invalid_report(config, contract_issues)

    lead_findings = tuple(
        _assess_lead(signal[index], lead_name, index, config)
        for index, lead_name in enumerate(config.expected_leads)
    )
    global_issues: list[QualityIssue] = []

    consistency_ratio = _limb_lead_consistency_ratio(signal)
    if consistency_ratio >= config.limb_consistency_reacquire_ratio:
        global_issues.append(
            QualityIssue(
                code=ReasonCode.LIMB_LEAD_INCONSISTENCY,
                status=QualityStatus.REACQUIRE,
                lead_name=None,
                metric_name="normalized_limb_identity_residual",
                observed_value=consistency_ratio,
                boundary_value=config.limb_consistency_reacquire_ratio,
            )
        )
    elif consistency_ratio >= config.limb_consistency_warn_ratio:
        global_issues.append(
            QualityIssue(
                code=ReasonCode.LIMB_LEAD_INCONSISTENCY,
                status=QualityStatus.LIMITED,
                lead_name=None,
                metric_name="normalized_limb_identity_residual",
                observed_value=consistency_ratio,
                boundary_value=config.limb_consistency_warn_ratio,
            )
        )

    reversal = _probable_limb_lead_reversal(signal, config)
    if reversal is not None:
        global_issues.append(
            QualityIssue(
                code=ReasonCode.PROBABLE_LIMB_LEAD_REVERSAL,
                status=QualityStatus.REACQUIRE,
                lead_name=None,
                metric_name="reversal_evidence_fraction",
                observed_value=reversal.score,
                boundary_value=config.reversal_min_evidence_fraction,
            )
        )

    statuses = [issue.status for issue in global_issues]
    statuses.extend(finding.status for finding in lead_findings)
    overall = _maximum_status(statuses)
    return SignalQualityReport(
        status=overall,
        config_version=config.version,
        global_issues=tuple(global_issues),
        leads=lead_findings,
        reversal_evidence=reversal,
    )


def _validate_contract(
    signal: FloatArray,
    metadata: SignalMetadata,
    config: SignalQualityConfig,
) -> tuple[QualityIssue, ...]:
    issues: list[QualityIssue] = []
    expected_shape = (len(config.expected_leads), config.expected_sample_count)
    if signal.shape != expected_shape:
        issues.append(_global_issue(ReasonCode.WRONG_SIGNAL_SHAPE, QualityStatus.INVALID))
    if not np.isfinite(signal).all():
        issues.append(_global_issue(ReasonCode.NONFINITE_SIGNAL, QualityStatus.INVALID))
    if not isinstance(metadata, SignalMetadata):
        issues.append(_global_issue(ReasonCode.INVALID_METADATA, QualityStatus.INVALID))
        return tuple(issues)

    lead_names = metadata.lead_names
    if len(lead_names) != len(config.expected_leads):
        issues.append(_global_issue(ReasonCode.LEAD_COUNT_MISMATCH, QualityStatus.INVALID))
    if len(set(lead_names)) != len(lead_names):
        issues.append(_global_issue(ReasonCode.DUPLICATE_LEADS, QualityStatus.INVALID))
    expected_set = set(config.expected_leads)
    observed_set = set(lead_names)
    if expected_set - observed_set:
        issues.append(_global_issue(ReasonCode.MISSING_LEADS, QualityStatus.INVALID))
    if observed_set - expected_set:
        issues.append(_global_issue(ReasonCode.UNEXPECTED_LEADS, QualityStatus.INVALID))
    if observed_set == expected_set and lead_names != config.expected_leads:
        issues.append(_global_issue(ReasonCode.LEAD_ORDER_MISMATCH, QualityStatus.INVALID))

    if not _matches_finite_number(
        metadata.sample_rate_hz,
        config.expected_sample_rate_hz,
        config.metadata_absolute_tolerance,
    ):
        issues.append(_global_issue(ReasonCode.SAMPLE_RATE_MISMATCH, QualityStatus.INVALID))
    if not _matches_finite_number(
        metadata.duration_seconds,
        config.expected_duration_seconds,
        config.metadata_absolute_tolerance,
    ):
        issues.append(_global_issue(ReasonCode.DURATION_MISMATCH, QualityStatus.INVALID))
    if len(metadata.units) != len(config.expected_leads):
        issues.append(_global_issue(ReasonCode.UNIT_COUNT_MISMATCH, QualityStatus.INVALID))
    if any(unit != config.expected_unit for unit in metadata.units):
        issues.append(_global_issue(ReasonCode.UNSUPPORTED_UNITS, QualityStatus.INVALID))
    return tuple(issues)


def _assess_lead(
    lead: FloatArray,
    lead_name: str,
    lead_index: int,
    config: SignalQualityConfig,
) -> LeadQualityFinding:
    metrics = _lead_metrics(lead, config)
    issues: list[QualityIssue] = []

    flat_status, flat_metric, flat_observed, flat_boundary = _flatline_status(metrics, config)
    if flat_status is not QualityStatus.PASS:
        issues.append(
            QualityIssue(
                code=ReasonCode.FLATLINE,
                status=flat_status,
                lead_name=lead_name,
                metric_name=flat_metric,
                observed_value=flat_observed,
                boundary_value=flat_boundary,
            )
        )

    clipping_status = _upper_threshold_status(
        metrics.clipping_fraction,
        config.clipping_warn_fraction,
        config.clipping_reacquire_fraction,
    )
    run_status = _upper_threshold_status(
        float(metrics.longest_clipping_run_samples),
        float(config.clipping_warn_run_samples),
        float(config.clipping_reacquire_run_samples),
    )
    clipping_status = _maximum_status((clipping_status, run_status))
    if clipping_status is not QualityStatus.PASS:
        use_run = _status_rank(run_status) >= _status_rank(
            _upper_threshold_status(
                metrics.clipping_fraction,
                config.clipping_warn_fraction,
                config.clipping_reacquire_fraction,
            )
        )
        issues.append(
            QualityIssue(
                code=ReasonCode.CLIPPING_OR_SATURATION,
                status=clipping_status,
                lead_name=lead_name,
                metric_name=("longest_clipping_run_samples" if use_run else "clipping_fraction"),
                observed_value=(
                    float(metrics.longest_clipping_run_samples)
                    if use_run
                    else metrics.clipping_fraction
                ),
                boundary_value=(
                    float(
                        config.clipping_reacquire_run_samples
                        if clipping_status is QualityStatus.REACQUIRE
                        else config.clipping_warn_run_samples
                    )
                    if use_run
                    else (
                        config.clipping_reacquire_fraction
                        if clipping_status is QualityStatus.REACQUIRE
                        else config.clipping_warn_fraction
                    )
                ),
            )
        )

    amplitude_status = _upper_threshold_status(
        metrics.maximum_absolute_amplitude_mv,
        config.amplitude_warn_mv,
        config.amplitude_reacquire_mv,
    )
    if amplitude_status is not QualityStatus.PASS:
        issues.append(
            _lead_metric_issue(
                ReasonCode.EXTREME_AMPLITUDE,
                amplitude_status,
                lead_name,
                "maximum_absolute_amplitude_mv",
                metrics.maximum_absolute_amplitude_mv,
                config.amplitude_warn_mv,
                config.amplitude_reacquire_mv,
            )
        )

    spike_status = _upper_threshold_status(
        metrics.maximum_step_mv,
        config.spike_warn_step_mv,
        config.spike_reacquire_step_mv,
    )
    if spike_status is not QualityStatus.PASS:
        issues.append(
            _lead_metric_issue(
                ReasonCode.EXTREME_SPIKES,
                spike_status,
                lead_name,
                "maximum_step_mv",
                metrics.maximum_step_mv,
                config.spike_warn_step_mv,
                config.spike_reacquire_step_mv,
            )
        )

    baseline_status = _upper_threshold_status(
        metrics.baseline_wander_power_ratio,
        config.baseline_wander_warn_ratio,
        config.baseline_wander_reacquire_ratio,
    )
    if baseline_status is not QualityStatus.PASS:
        issues.append(
            _lead_metric_issue(
                ReasonCode.BASELINE_WANDER,
                baseline_status,
                lead_name,
                "baseline_wander_power_ratio",
                metrics.baseline_wander_power_ratio,
                config.baseline_wander_warn_ratio,
                config.baseline_wander_reacquire_ratio,
            )
        )

    for reason, observed in (
        (ReasonCode.POWERLINE_INTERFERENCE_50HZ, metrics.powerline_50hz_power_ratio),
        (ReasonCode.POWERLINE_INTERFERENCE_60HZ, metrics.powerline_60hz_power_ratio),
    ):
        mains_status = _upper_threshold_status(
            observed,
            config.powerline_warn_ratio,
            config.powerline_reacquire_ratio,
        )
        if mains_status is not QualityStatus.PASS:
            issues.append(
                _lead_metric_issue(
                    reason,
                    mains_status,
                    lead_name,
                    (
                        "powerline_50hz_power_ratio"
                        if reason is ReasonCode.POWERLINE_INTERFERENCE_50HZ
                        else "powerline_60hz_power_ratio"
                    ),
                    observed,
                    config.powerline_warn_ratio,
                    config.powerline_reacquire_ratio,
                )
            )

    high_frequency_status = _upper_threshold_status(
        metrics.high_frequency_power_ratio,
        config.high_frequency_warn_ratio,
        config.high_frequency_reacquire_ratio,
    )
    if high_frequency_status is not QualityStatus.PASS:
        issues.append(
            _lead_metric_issue(
                ReasonCode.HIGH_FREQUENCY_NOISE,
                high_frequency_status,
                lead_name,
                "high_frequency_power_ratio",
                metrics.high_frequency_power_ratio,
                config.high_frequency_warn_ratio,
                config.high_frequency_reacquire_ratio,
            )
        )

    status = _maximum_status(issue.status for issue in issues)
    return LeadQualityFinding(
        lead_name=lead_name,
        lead_index=lead_index,
        status=status,
        reason_codes=tuple(issue.code for issue in issues),
        issues=tuple(issues),
        metrics=metrics,
    )


def _lead_metrics(lead: FloatArray, config: SignalQualityConfig) -> LeadQualityMetrics:
    differences = np.abs(np.diff(lead))
    peak_to_peak = float(np.ptp(lead))
    standard_deviation = float(np.std(lead))
    maximum_absolute = float(np.max(np.abs(lead)))
    flat_fraction = float(np.mean(differences <= config.flat_step_tolerance_mv))

    minimum = float(np.min(lead))
    maximum = float(np.max(lead))
    at_extreme = np.logical_or(
        np.isclose(lead, minimum, atol=config.clipping_equality_tolerance_mv, rtol=0.0),
        np.isclose(lead, maximum, atol=config.clipping_equality_tolerance_mv, rtol=0.0),
    )
    clipping_fraction = float(np.mean(at_extreme))
    longest_clipping_run = _longest_true_run(at_extreme)
    maximum_step = float(np.max(differences))
    spike_fraction = float(np.mean(differences >= config.spike_warn_step_mv))

    frequencies, power = _power_spectrum(lead, config.expected_sample_rate_hz)
    total_mask = frequencies >= config.spectral_floor_hz
    total_power = float(np.sum(power[total_mask]))
    if total_power <= np.finfo(np.float64).tiny:
        baseline_ratio = 0.0
        powerline_50_ratio = 0.0
        powerline_60_ratio = 0.0
        high_frequency_ratio = 0.0
    else:
        baseline_mask = np.logical_and(
            frequencies >= config.spectral_floor_hz,
            frequencies < config.baseline_wander_upper_hz,
        )
        powerline_50_mask = _frequency_band_mask(
            frequencies,
            _aliased_frequency(50.0, config.expected_sample_rate_hz),
            config.powerline_band_half_width_hz,
        )
        powerline_60_mask = _frequency_band_mask(
            frequencies,
            _aliased_frequency(60.0, config.expected_sample_rate_hz),
            config.powerline_band_half_width_hz,
        )
        high_frequency_mask = np.logical_and(
            frequencies >= config.high_frequency_lower_hz,
            frequencies <= config.high_frequency_upper_hz,
        )
        high_frequency_mask = np.logical_and(
            high_frequency_mask,
            np.logical_not(np.logical_or(powerline_50_mask, powerline_60_mask)),
        )
        baseline_ratio = float(np.sum(power[baseline_mask]) / total_power)
        powerline_50_ratio = float(np.sum(power[powerline_50_mask]) / total_power)
        powerline_60_ratio = float(np.sum(power[powerline_60_mask]) / total_power)
        high_frequency_ratio = float(np.sum(power[high_frequency_mask]) / total_power)

    return LeadQualityMetrics(
        peak_to_peak_mv=peak_to_peak,
        standard_deviation_mv=standard_deviation,
        maximum_absolute_amplitude_mv=maximum_absolute,
        flat_step_fraction=flat_fraction,
        clipping_fraction=clipping_fraction,
        longest_clipping_run_samples=longest_clipping_run,
        maximum_step_mv=maximum_step,
        spike_step_fraction=spike_fraction,
        baseline_wander_power_ratio=baseline_ratio,
        powerline_50hz_power_ratio=powerline_50_ratio,
        powerline_60hz_power_ratio=powerline_60_ratio,
        high_frequency_power_ratio=high_frequency_ratio,
    )


def _flatline_status(
    metrics: LeadQualityMetrics,
    config: SignalQualityConfig,
) -> tuple[QualityStatus, str | None, float | None, float | None]:
    critical_candidates = (
        (
            metrics.peak_to_peak_mv <= config.flat_reacquire_peak_to_peak_mv,
            "peak_to_peak_mv",
            metrics.peak_to_peak_mv,
            config.flat_reacquire_peak_to_peak_mv,
        ),
        (
            metrics.standard_deviation_mv <= config.flat_reacquire_std_mv,
            "standard_deviation_mv",
            metrics.standard_deviation_mv,
            config.flat_reacquire_std_mv,
        ),
        (
            metrics.flat_step_fraction >= config.flat_reacquire_fraction,
            "flat_step_fraction",
            metrics.flat_step_fraction,
            config.flat_reacquire_fraction,
        ),
    )
    for triggered, metric, observed, boundary in critical_candidates:
        if triggered:
            return QualityStatus.REACQUIRE, metric, observed, boundary

    warning_candidates = (
        (
            metrics.peak_to_peak_mv <= config.flat_warn_peak_to_peak_mv,
            "peak_to_peak_mv",
            metrics.peak_to_peak_mv,
            config.flat_warn_peak_to_peak_mv,
        ),
        (
            metrics.standard_deviation_mv <= config.flat_warn_std_mv,
            "standard_deviation_mv",
            metrics.standard_deviation_mv,
            config.flat_warn_std_mv,
        ),
        (
            metrics.flat_step_fraction >= config.flat_warn_fraction,
            "flat_step_fraction",
            metrics.flat_step_fraction,
            config.flat_warn_fraction,
        ),
    )
    for triggered, metric, observed, boundary in warning_candidates:
        if triggered:
            return QualityStatus.LIMITED, metric, observed, boundary
    return QualityStatus.PASS, None, None, None


def _limb_lead_consistency_ratio(signal: FloatArray) -> float:
    lead_i, lead_ii, lead_iii, avr, avl, avf = signal[:6]
    residuals = np.concatenate(
        (
            lead_i + lead_iii - lead_ii,
            avr + (lead_i + lead_ii) / 2.0,
            avl - (lead_i - lead_ii / 2.0),
            avf - (lead_ii - lead_i / 2.0),
        )
    )
    scale = float(np.sqrt(np.mean(np.square(signal[:6]))))
    residual_rms = float(np.sqrt(np.mean(np.square(residuals))))
    return residual_rms / max(scale, np.finfo(np.float64).eps)


def _probable_limb_lead_reversal(
    signal: FloatArray,
    config: SignalQualityConfig,
) -> LimbLeadReversalEvidence | None:
    lead_names = ("I", "II", "III", "aVR", "aVL", "aVF", "V5", "V6")
    indices = (0, 1, 2, 3, 4, 5, 10, 11)
    polarities = {
        name: _dominant_polarity(signal[index], config.dominant_polarity_quantile)
        for name, index in zip(lead_names, indices, strict=True)
    }
    correlations = {
        "I:V6": _safe_correlation(signal[0], signal[11]),
        "II:V5": _safe_correlation(signal[1], signal[10]),
    }

    minimum = config.reversal_min_polarity
    inverse = config.reversal_max_inverse_correlation
    ra_la_evidence = {
        "lead_i_dominantly_negative": polarities["I"] <= -minimum,
        "avr_dominantly_positive": polarities["aVR"] >= minimum,
        "v6_dominantly_positive": polarities["V6"] >= minimum,
        "lead_i_v6_inverse": correlations["I:V6"] <= inverse,
    }
    ra_ll_evidence = {
        "lead_ii_dominantly_negative": polarities["II"] <= -minimum,
        "avr_dominantly_positive": polarities["aVR"] >= minimum,
        "avf_dominantly_negative": polarities["aVF"] <= -minimum,
        "v5_dominantly_positive": polarities["V5"] >= minimum,
        "lead_ii_v5_inverse": correlations["II:V5"] <= inverse,
    }

    candidates: list[tuple[LimbLeadReversalKind, float, tuple[str, ...]]] = []
    ra_la_score = float(sum(ra_la_evidence.values()) / len(ra_la_evidence))
    if (
        ra_la_evidence["lead_i_dominantly_negative"]
        and ra_la_evidence["avr_dominantly_positive"]
        and ra_la_score >= config.reversal_min_evidence_fraction
    ):
        candidates.append(
            (
                LimbLeadReversalKind.RIGHT_ARM_LEFT_ARM,
                ra_la_score,
                tuple(name for name, present in ra_la_evidence.items() if present),
            )
        )

    ra_ll_score = float(sum(ra_ll_evidence.values()) / len(ra_ll_evidence))
    if (
        ra_ll_evidence["lead_ii_dominantly_negative"]
        and ra_ll_evidence["avr_dominantly_positive"]
        and ra_ll_evidence["avf_dominantly_negative"]
        and ra_ll_score >= config.reversal_min_evidence_fraction
    ):
        candidates.append(
            (
                LimbLeadReversalKind.RIGHT_ARM_LEFT_LEG,
                ra_ll_score,
                tuple(name for name, present in ra_ll_evidence.items() if present),
            )
        )

    if not candidates:
        return None
    kind, score, evidence = max(candidates, key=lambda candidate: candidate[1])
    return LimbLeadReversalEvidence(
        probable_kind=kind,
        score=score,
        evidence_codes=evidence,
        dominant_polarities=tuple((name, polarities[name]) for name in lead_names),
        correlations=tuple(correlations.items()),
    )


def _dominant_polarity(lead: FloatArray, quantile: float) -> float:
    centered = lead - np.median(lead)
    absolute = np.abs(centered)
    cutoff = float(np.quantile(absolute, quantile))
    selected = absolute >= cutoff
    denominator = float(np.sum(absolute[selected]))
    if denominator <= np.finfo(np.float64).eps:
        return 0.0
    return float(np.sum(centered[selected]) / denominator)


def _safe_correlation(left: FloatArray, right: FloatArray) -> float:
    left_centered = left - np.mean(left)
    right_centered = right - np.mean(right)
    denominator = float(
        np.sqrt(np.sum(np.square(left_centered)) * np.sum(np.square(right_centered)))
    )
    if denominator <= np.finfo(np.float64).eps:
        return 0.0
    return float(np.sum(left_centered * right_centered) / denominator)


def _power_spectrum(lead: FloatArray, sample_rate_hz: float) -> tuple[FloatArray, FloatArray]:
    centered = lead - np.mean(lead)
    windowed = centered * np.hanning(lead.size)
    spectrum = np.fft.rfft(windowed)
    frequencies = np.fft.rfftfreq(lead.size, d=1.0 / sample_rate_hz)
    power = np.square(np.abs(spectrum))
    return frequencies.astype(np.float64, copy=False), power.astype(np.float64, copy=False)


def _frequency_band_mask(
    frequencies: FloatArray,
    center_hz: float,
    half_width_hz: float,
) -> NDArray[np.bool_]:
    return np.abs(frequencies - center_hz) <= half_width_hz


def _aliased_frequency(frequency_hz: float, sample_rate_hz: float) -> float:
    wrapped = frequency_hz % sample_rate_hz
    nyquist = sample_rate_hz / 2.0
    return wrapped if wrapped <= nyquist else sample_rate_hz - wrapped


def _longest_true_run(mask: NDArray[np.bool_]) -> int:
    longest = 0
    current = 0
    for value in mask:
        if bool(value):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _upper_threshold_status(
    observed: float,
    warning: float,
    reacquire: float,
) -> QualityStatus:
    if observed >= reacquire:
        return QualityStatus.REACQUIRE
    if observed >= warning:
        return QualityStatus.LIMITED
    return QualityStatus.PASS


def _lead_metric_issue(
    code: ReasonCode,
    status: QualityStatus,
    lead_name: str,
    metric_name: str,
    observed: float,
    warning: float,
    reacquire: float,
) -> QualityIssue:
    return QualityIssue(
        code=code,
        status=status,
        lead_name=lead_name,
        metric_name=metric_name,
        observed_value=observed,
        boundary_value=reacquire if status is QualityStatus.REACQUIRE else warning,
    )


def _matches_finite_number(observed: object, expected: float, tolerance: float) -> bool:
    if isinstance(observed, bool) or not isinstance(observed, (int, float)):
        return False
    numeric = float(observed)
    return math.isfinite(numeric) and math.isclose(numeric, expected, abs_tol=tolerance)


def _global_issue(code: ReasonCode, status: QualityStatus) -> QualityIssue:
    return QualityIssue(
        code=code,
        status=status,
        lead_name=None,
        metric_name=None,
        observed_value=None,
        boundary_value=None,
    )


def _invalid_report(
    config: SignalQualityConfig,
    issues: tuple[QualityIssue, ...],
) -> SignalQualityReport:
    return SignalQualityReport(
        status=QualityStatus.INVALID,
        config_version=config.version,
        global_issues=issues,
        leads=(),
        reversal_evidence=None,
    )


def _status_rank(status: QualityStatus) -> int:
    return {
        QualityStatus.PASS: 0,
        QualityStatus.LIMITED: 1,
        QualityStatus.REACQUIRE: 2,
        QualityStatus.INVALID: 3,
    }[status]


def _maximum_status(statuses: Iterable[QualityStatus]) -> QualityStatus:
    materialized: tuple[QualityStatus, ...] = tuple(statuses)
    if not materialized:
        return QualityStatus.PASS
    return max(materialized, key=_status_rank)
