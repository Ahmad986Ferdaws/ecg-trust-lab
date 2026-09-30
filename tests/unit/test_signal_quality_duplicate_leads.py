from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

import numpy as np
import pytest
from numpy.typing import NDArray

from ecg_trust.contract_adapters import quality_report_to_contract
from ecg_trust.quality.signal_quality import (
    DEFAULT_SIGNAL_QUALITY_CONFIG,
    SENTINEL_V2_SIGNAL_QUALITY_CONFIG,
    QualityStatus,
    ReasonCode,
    SignalMetadata,
    SignalQualityConfig,
    SignalQualityReport,
    assess_signal_quality,
)

FloatArray = NDArray[np.float64]


def _beat(time: FloatArray) -> FloatArray:
    phase = np.mod(time, 1.0)

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
    time = np.arange(1_000, dtype=np.float64) / 100.0
    beat = _beat(time)
    lead_i = 0.75 * _beat(time + 0.02)
    lead_ii = beat
    return np.stack(
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


def _assess(signal: FloatArray, config: SignalQualityConfig) -> SignalQualityReport:
    return assess_signal_quality(signal, SignalMetadata.canonical(config), config=config)


def _copy_lead(source: int, target: int) -> FloatArray:
    signal = _clean_signal()
    signal[target] = signal[source]
    return signal


def test_clean_signal_passes_and_v1_keeps_the_check_disabled() -> None:
    assert DEFAULT_SIGNAL_QUALITY_CONFIG.duplicate_lead_tolerance_mv is None
    assert _assess(_clean_signal(), SENTINEL_V2_SIGNAL_QUALITY_CONFIG).status is QualityStatus.PASS


def test_copied_precordial_lead_is_sent_for_reacquisition_only_under_v2() -> None:
    signal = _copy_lead(source=8, target=10)

    legacy = _assess(signal, DEFAULT_SIGNAL_QUALITY_CONFIG)
    current = _assess(signal, SENTINEL_V2_SIGNAL_QUALITY_CONFIG)

    assert legacy.status is QualityStatus.PASS
    assert current.status is QualityStatus.REACQUIRE
    duplicated = [
        issue for issue in current.global_issues if issue.code is ReasonCode.DUPLICATE_LEADS
    ]
    assert len(duplicated) == 1
    assert duplicated[0].lead_name == "V5"
    assert duplicated[0].observed_value == pytest.approx(0.0)
    assert duplicated[0].metric_name == "max_abs_difference_to_earlier_lead_mv"
    contract = quality_report_to_contract(current, evaluated_at=datetime(2026, 1, 1, tzinfo=UTC))
    assert not contract.passed
    assert any(finding.affected_leads == ("V5",) for finding in contract.findings)


def test_near_copy_within_tolerance_is_reported_and_a_real_difference_is_not() -> None:
    near = _copy_lead(source=6, target=7)
    near[7, 100] += 4e-4
    distinct = _copy_lead(source=6, target=7)
    distinct[7, 100] += 2e-3

    assert _assess(near, SENTINEL_V2_SIGNAL_QUALITY_CONFIG).status is QualityStatus.REACQUIRE
    assert _assess(distinct, SENTINEL_V2_SIGNAL_QUALITY_CONFIG).status is QualityStatus.PASS


def test_flat_lead_suppresses_the_derived_limb_identity_duplicates() -> None:
    signal = _clean_signal()
    signal[0] = 0.0
    signal[2] = signal[1]
    signal[3] = -signal[1] / 2.0
    signal[4] = -signal[1] / 2.0
    signal[5] = signal[1]

    report = _assess(signal, SENTINEL_V2_SIGNAL_QUALITY_CONFIG)

    assert report.status is QualityStatus.REACQUIRE
    assert ReasonCode.FLATLINE in report.reason_codes
    assert ReasonCode.DUPLICATE_LEADS not in report.reason_codes


@pytest.mark.parametrize("tolerance", [-1e-6, float("nan"), float("inf"), True])
def test_invalid_duplicate_tolerance_is_rejected(tolerance: object) -> None:
    with pytest.raises(ValueError, match="duplicate_lead_tolerance_mv"):
        replace(DEFAULT_SIGNAL_QUALITY_CONFIG, duplicate_lead_tolerance_mv=tolerance)
