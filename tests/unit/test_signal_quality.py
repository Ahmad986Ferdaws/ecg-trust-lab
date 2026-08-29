from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest
from numpy.typing import NDArray

from ecg_trust.quality.signal_quality import (
    DEFAULT_SIGNAL_QUALITY_CONFIG,
    LeadQualityFinding,
    LimbLeadReversalKind,
    QualityStatus,
    ReasonCode,
    SignalMetadata,
    SignalQualityConfig,
    SignalQualityReport,
    assess_signal_quality,
)

FloatArray = NDArray[np.float64]


def _beat(t_seconds: FloatArray) -> FloatArray:
    phase = np.mod(t_seconds, 1.0)

    def pulse(center: float, width: float, amplitude: float) -> FloatArray:
        return amplitude * np.exp(-0.5 * np.square((phase - center) / width))

    return (
        pulse(0.18, 0.035, 0.10)
        + pulse(0.375, 0.014, -0.12)
        + pulse(0.400, 0.016, 1.10)
        + pulse(0.430, 0.018, -0.24)
        + pulse(0.660, 0.075, 0.28)
    )


def _clean_signal() -> FloatArray:
    config = DEFAULT_SIGNAL_QUALITY_CONFIG
    time = np.arange(config.expected_sample_count, dtype=np.float64) / (
        config.expected_sample_rate_hz
    )
    beat = _beat(time)
    lead_i = 0.75 * beat
    lead_ii = 1.00 * beat
    signal = np.stack(
        (
            lead_i,
            lead_ii,
            lead_ii - lead_i,
            -(lead_i + lead_ii) / 2.0,
            lead_i - lead_ii / 2.0,
            lead_ii - lead_i / 2.0,
            -0.40 * beat,
            -0.15 * beat,
            0.30 * beat,
            0.65 * beat,
            0.90 * beat,
            0.80 * beat,
        )
    )
    return signal.astype(np.float64, copy=False)


def _report_for(signal: FloatArray) -> SignalQualityReport:
    return assess_signal_quality(signal, SignalMetadata.canonical())


def _lead(report: SignalQualityReport, name: str) -> LeadQualityFinding:
    return next(finding for finding in report.leads if finding.lead_name == name)


def test_canonical_clean_signal_passes_with_structured_per_lead_metrics() -> None:
    signal = _clean_signal()
    before = signal.copy()

    first = _report_for(signal)
    second = _report_for(signal)

    assert first == second
    assert first.status is QualityStatus.PASS
    assert first.classification_allowed
    assert first.reason_codes == ()
    assert first.global_issues == ()
    assert first.reversal_evidence is None
    assert len(first.leads) == 12
    assert tuple(finding.lead_name for finding in first.leads) == (
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
    assert all(finding.status is QualityStatus.PASS for finding in first.leads)
    assert all(finding.metrics.peak_to_peak_mv > 0.0 for finding in first.leads)
    np.testing.assert_array_equal(signal, before)


@pytest.mark.parametrize(
    ("signal", "reason"),
    [
        (np.zeros((11, 1_000), dtype=np.float64), ReasonCode.WRONG_SIGNAL_SHAPE),
        (np.zeros((12, 999), dtype=np.float64), ReasonCode.WRONG_SIGNAL_SHAPE),
        (np.zeros((1_000, 12), dtype=np.float64), ReasonCode.WRONG_SIGNAL_SHAPE),
        (np.zeros(12_000, dtype=np.float64), ReasonCode.WRONG_SIGNAL_SHAPE),
    ],
)
def test_wrong_shapes_fail_closed_without_per_lead_analysis(
    signal: FloatArray,
    reason: ReasonCode,
) -> None:
    report = assess_signal_quality(signal, SignalMetadata.canonical())

    assert report.status is QualityStatus.INVALID
    assert not report.classification_allowed
    assert reason in report.reason_codes
    assert report.leads == ()


def test_non_numeric_and_nonfinite_inputs_are_invalid() -> None:
    non_numeric = assess_signal_quality([["not-a-number"]], SignalMetadata.canonical())
    complex_signal = assess_signal_quality(
        np.zeros((12, 1_000), dtype=np.complex128),
        SignalMetadata.canonical(),
    )
    boolean_signal = assess_signal_quality(
        np.zeros((12, 1_000), dtype=np.bool_),
        SignalMetadata.canonical(),
    )
    signal = _clean_signal()
    signal[0, 0] = np.nan
    nonfinite = _report_for(signal)

    assert non_numeric.status is QualityStatus.INVALID
    assert non_numeric.reason_codes == (ReasonCode.NON_NUMERIC_SIGNAL,)
    assert complex_signal.reason_codes == (ReasonCode.NON_REAL_SIGNAL,)
    assert boolean_signal.reason_codes == (ReasonCode.NON_REAL_SIGNAL,)
    assert nonfinite.status is QualityStatus.INVALID
    assert ReasonCode.NONFINITE_SIGNAL in nonfinite.reason_codes
    assert nonfinite.leads == ()


@pytest.mark.parametrize(
    ("metadata", "expected_reasons"),
    [
        (
            replace(
                SignalMetadata.canonical(),
                lead_names=SignalMetadata.canonical().lead_names[:-1],
            ),
            {ReasonCode.LEAD_COUNT_MISMATCH, ReasonCode.MISSING_LEADS},
        ),
        (
            replace(
                SignalMetadata.canonical(),
                lead_names=SignalMetadata.canonical().lead_names[:-1] + ("V5",),
            ),
            {ReasonCode.DUPLICATE_LEADS, ReasonCode.MISSING_LEADS},
        ),
        (
            replace(
                SignalMetadata.canonical(),
                lead_names=SignalMetadata.canonical().lead_names[:-1] + ("V7",),
            ),
            {ReasonCode.MISSING_LEADS, ReasonCode.UNEXPECTED_LEADS},
        ),
        (
            replace(
                SignalMetadata.canonical(),
                lead_names=("II", "I") + SignalMetadata.canonical().lead_names[2:],
            ),
            {ReasonCode.LEAD_ORDER_MISMATCH},
        ),
        (
            replace(SignalMetadata.canonical(), sample_rate_hz=500.0),
            {ReasonCode.SAMPLE_RATE_MISMATCH},
        ),
        (
            replace(SignalMetadata.canonical(), duration_seconds=9.99),
            {ReasonCode.DURATION_MISMATCH},
        ),
        (
            replace(SignalMetadata.canonical(), units=("mV",) * 11),
            {ReasonCode.UNIT_COUNT_MISMATCH},
        ),
        (
            replace(SignalMetadata.canonical(), units=("V",) * 12),
            {ReasonCode.UNSUPPORTED_UNITS},
        ),
    ],
)
def test_metadata_contract_violations_are_invalid_and_deterministic(
    metadata: SignalMetadata,
    expected_reasons: set[ReasonCode],
) -> None:
    report = assess_signal_quality(_clean_signal(), metadata)

    assert report.status is QualityStatus.INVALID
    assert not report.classification_allowed
    assert expected_reasons <= set(report.reason_codes)
    assert report.leads == ()


def test_flatline_is_hard_blocked_and_attributed_to_the_affected_lead() -> None:
    signal = _clean_signal()
    signal[6] = 0.0

    report = _report_for(signal)
    finding = _lead(report, "V1")

    assert report.status is QualityStatus.REACQUIRE
    assert not report.classification_allowed
    assert ReasonCode.FLATLINE in report.reason_codes
    assert finding.status is QualityStatus.REACQUIRE
    assert ReasonCode.FLATLINE in finding.reason_codes


def test_clipping_saturation_is_hard_blocked() -> None:
    signal = _clean_signal()
    signal[7] = np.clip(signal[7], -0.025, 0.025)

    report = _report_for(signal)
    finding = _lead(report, "V2")

    assert report.status is QualityStatus.REACQUIRE
    assert ReasonCode.CLIPPING_OR_SATURATION in finding.reason_codes
    assert finding.metrics.longest_clipping_run_samples >= 8


def test_extreme_single_sample_spike_is_hard_blocked() -> None:
    signal = _clean_signal()
    signal[8, 500] += 6.0

    report = _report_for(signal)
    finding = _lead(report, "V3")

    assert report.status is QualityStatus.REACQUIRE
    assert ReasonCode.EXTREME_SPIKES in finding.reason_codes
    assert finding.metrics.maximum_step_mv >= 4.0


def test_baseline_wander_has_limited_and_reacquire_operating_regions() -> None:
    signal = _clean_signal()
    config = DEFAULT_SIGNAL_QUALITY_CONFIG
    time = np.arange(config.expected_sample_count, dtype=np.float64) / (
        config.expected_sample_rate_hz
    )
    signal[9] += 1.6 * np.sin(2.0 * np.pi * 0.3 * time)
    hard_report = _report_for(signal)

    limited_config = replace(
        config,
        baseline_wander_warn_ratio=0.10,
        baseline_wander_reacquire_ratio=0.999,
    )
    limited_report = assess_signal_quality(
        signal,
        SignalMetadata.canonical(limited_config),
        config=limited_config,
    )

    assert hard_report.status is QualityStatus.REACQUIRE
    assert ReasonCode.BASELINE_WANDER in hard_report.reason_codes
    assert limited_report.status is QualityStatus.LIMITED
    assert not limited_report.classification_allowed
    assert ReasonCode.BASELINE_WANDER in limited_report.reason_codes


@pytest.mark.parametrize(
    ("frequency_hz", "reason"),
    [
        (50.0, ReasonCode.POWERLINE_INTERFERENCE_50HZ),
        (60.0, ReasonCode.POWERLINE_INTERFERENCE_60HZ),
    ],
)
def test_probable_powerline_interference_is_detected_at_100hz_aliases(
    frequency_hz: float,
    reason: ReasonCode,
) -> None:
    signal = _clean_signal()
    config = DEFAULT_SIGNAL_QUALITY_CONFIG
    time = np.arange(config.expected_sample_count, dtype=np.float64) / (
        config.expected_sample_rate_hz
    )
    signal[10] += 1.2 * np.cos(2.0 * np.pi * frequency_hz * time + 0.31)

    report = _report_for(signal)
    finding = _lead(report, "V5")

    assert report.status is QualityStatus.REACQUIRE
    assert reason in finding.reason_codes


def test_high_frequency_noise_is_detected_outside_powerline_bands() -> None:
    signal = _clean_signal()
    config = DEFAULT_SIGNAL_QUALITY_CONFIG
    time = np.arange(config.expected_sample_count, dtype=np.float64) / (
        config.expected_sample_rate_hz
    )
    signal[11] += 1.1 * np.sin(2.0 * np.pi * 30.0 * time + 0.17)

    report = _report_for(signal)
    finding = _lead(report, "V6")

    assert report.status is QualityStatus.REACQUIRE
    assert ReasonCode.HIGH_FREQUENCY_NOISE in finding.reason_codes
    assert finding.metrics.high_frequency_power_ratio >= 0.5


def test_limb_identity_inconsistency_is_reported_without_reordering() -> None:
    signal = _clean_signal()
    signal[[0, 6]] = signal[[6, 0]]
    before = signal.copy()

    report = _report_for(signal)

    assert report.status is QualityStatus.REACQUIRE
    assert ReasonCode.LIMB_LEAD_INCONSISTENCY in report.reason_codes
    np.testing.assert_array_equal(signal, before)


def test_probable_right_arm_left_arm_reversal_is_evidence_not_correction() -> None:
    signal = _clean_signal()
    original = signal.copy()
    signal[0] = -original[0]
    signal[1] = original[2]
    signal[2] = original[1]
    signal[3] = original[4]
    signal[4] = original[3]
    signal[5] = original[5]
    before = signal.copy()

    report = _report_for(signal)

    assert report.status is QualityStatus.REACQUIRE
    assert not report.classification_allowed
    assert ReasonCode.PROBABLE_LIMB_LEAD_REVERSAL in report.reason_codes
    assert report.reversal_evidence is not None
    assert report.reversal_evidence.probable_kind is LimbLeadReversalKind.RIGHT_ARM_LEFT_ARM
    assert report.reversal_evidence.score >= 0.75
    np.testing.assert_array_equal(signal, before)


def test_config_is_frozen_validated_and_versioned() -> None:
    config = SignalQualityConfig()

    with pytest.raises(FrozenInstanceError):
        config.expected_sample_rate_hz = 500.0  # type: ignore[misc]
    with pytest.raises(ValueError, match="warning thresholds must precede"):
        replace(config, clipping_warn_fraction=0.2)
    with pytest.raises(ValueError, match="must equal sample rate"):
        replace(config, expected_sample_count=5_000)

    report = assess_signal_quality(
        _clean_signal(),
        SignalMetadata.canonical(config),
        config=config,
    )
    assert report.config_version == config.version
