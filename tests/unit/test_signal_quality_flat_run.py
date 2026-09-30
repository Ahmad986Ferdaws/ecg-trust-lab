from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from numpy.typing import NDArray

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
V2_LEAD = 7


def _clean_signal() -> FloatArray:
    time = np.arange(1_000, dtype=np.float64) / 100.0
    phase = np.mod(time, 1.0)

    def pulse(center: float, width: float, amplitude: float) -> FloatArray:
        return amplitude * np.exp(-0.5 * np.square((phase - center) / width))

    beat = (
        pulse(0.18, 0.035, 0.10)
        + pulse(0.375, 0.014, -0.12)
        + pulse(0.400, 0.016, 1.10)
        + pulse(0.430, 0.018, -0.24)
        + pulse(0.660, 0.075, 0.28)
    )
    lead_i = 0.75 * beat
    lead_ii = 1.00 * beat
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


def _flat_tail(signal: FloatArray, seconds: float, leads: tuple[int, ...]) -> FloatArray:
    stressed = signal.copy()
    start = 1_000 - round(seconds * 100)
    for lead in leads:
        stressed[lead, start:] = 0.0
    return stressed


def test_v1_default_is_frozen_and_keeps_the_rule_disabled() -> None:
    config = DEFAULT_SIGNAL_QUALITY_CONFIG

    assert config.version == "canonical-12x1000-mv-v1"
    assert config.flat_run_warn_seconds is None
    assert config.flat_run_reacquire_seconds is None


@pytest.mark.parametrize(
    ("seconds", "leads"),
    [(7.0, (V2_LEAD,)), (5.0, tuple(range(12)))],
)
def test_v1_passes_partial_dropout_that_v2_sends_for_reacquisition(
    seconds: float,
    leads: tuple[int, ...],
) -> None:
    stressed = _flat_tail(_clean_signal(), seconds, leads)

    legacy = _assess(stressed, DEFAULT_SIGNAL_QUALITY_CONFIG)
    current = _assess(stressed, SENTINEL_V2_SIGNAL_QUALITY_CONFIG)

    assert legacy.status is QualityStatus.PASS
    assert current.status is QualityStatus.REACQUIRE
    assert current.config_version == "canonical-12x1000-mv-v2"
    assert ReasonCode.FLATLINE in current.reason_codes
    for lead in leads:
        issue = next(
            item for item in current.leads[lead].issues if item.code is ReasonCode.FLATLINE
        )
        assert issue.metric_name == "longest_flat_run_seconds"
        assert issue.observed_value is not None
        assert issue.observed_value >= seconds
        assert issue.boundary_value == pytest.approx(1.0)


def test_short_flat_stretch_is_limited_and_clean_signal_passes() -> None:
    clean = _clean_signal()
    short = _flat_tail(clean, 0.6, (V2_LEAD,))

    assert _assess(clean, SENTINEL_V2_SIGNAL_QUALITY_CONFIG).status is QualityStatus.PASS
    report = _assess(short, SENTINEL_V2_SIGNAL_QUALITY_CONFIG)

    assert report.status is QualityStatus.LIMITED
    assert not report.classification_allowed
    issue = report.leads[V2_LEAD].issues[0]
    assert issue.status is QualityStatus.LIMITED
    assert issue.boundary_value == pytest.approx(0.5)


def test_whole_record_flatline_evidence_keeps_precedence_when_more_severe() -> None:
    stressed = _flat_tail(_clean_signal(), 10.0, (V2_LEAD,))

    report = _assess(stressed, SENTINEL_V2_SIGNAL_QUALITY_CONFIG)
    flat = [issue for issue in report.leads[V2_LEAD].issues if issue.code is ReasonCode.FLATLINE]

    assert report.status is QualityStatus.REACQUIRE
    assert len(flat) == 1
    assert flat[0].metric_name == "peak_to_peak_mv"


@pytest.mark.parametrize(
    ("warning", "reacquire"),
    [
        (0.5, None),
        (None, 1.0),
        (1.0, 1.0),
        (2.0, 1.0),
        (0.0, 1.0),
        (0.5, 11.0),
        (float("nan"), 1.0),
        (True, 1.0),
    ],
)
def test_incoherent_flat_run_thresholds_are_rejected(
    warning: float | None,
    reacquire: float | None,
) -> None:
    with pytest.raises(ValueError, match=r"flat[-_]run"):
        replace(
            DEFAULT_SIGNAL_QUALITY_CONFIG,
            flat_run_warn_seconds=warning,
            flat_run_reacquire_seconds=reacquire,
        )
