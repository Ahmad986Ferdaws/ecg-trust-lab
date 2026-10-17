from __future__ import annotations

import warnings

import numpy as np
import pytest

from ecg_trust.quality.signal_quality import (
    QualityStatus,
    ReasonCode,
    SignalMetadata,
    assess_signal_quality,
)
from scripts.quality_gate_stress_report import synthetic_ecg


@pytest.mark.parametrize("value", [False, np.bool_(False), np.complex128(0.1 + 99j)])
@pytest.mark.parametrize("container", ["object", "list"])
def test_hidden_non_real_samples_are_refused_before_lossy_conversion(value, container: str) -> None:
    signal = synthetic_ecg().astype(object)
    signal[7, 0] = value
    if container == "list":
        signal = signal.tolist()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        report = assess_signal_quality(signal, SignalMetadata.canonical())
    assert report.status is QualityStatus.INVALID
    assert report.reason_codes == (ReasonCode.NON_REAL_SIGNAL,)
    assert not report.classification_allowed


@pytest.mark.parametrize("container", ["array", "object", "list", "tuple"])
def test_real_signal_containers_keep_the_same_quality_result(container: str) -> None:
    signal = synthetic_ecg()
    baseline = assess_signal_quality(signal, SignalMetadata.canonical())
    if container == "object":
        signal = signal.astype(object)
    elif container == "list":
        signal = signal.tolist()
    elif container == "tuple":
        signal = tuple(tuple(row) for row in signal.tolist())
    assert assess_signal_quality(signal, SignalMetadata.canonical()) == baseline
    assert baseline.classification_allowed


def test_ragged_input_still_returns_invalid_evidence() -> None:
    signal = synthetic_ecg().tolist()
    signal[0].pop()
    report = assess_signal_quality(signal, SignalMetadata.canonical())
    assert report.status is QualityStatus.INVALID
    assert not report.classification_allowed
