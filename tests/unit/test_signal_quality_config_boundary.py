from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from ecg_trust.quality.signal_quality import (
    DEFAULT_SIGNAL_QUALITY_CONFIG,
    SENTINEL_V2_SIGNAL_QUALITY_CONFIG,
    SignalMetadata,
    assess_signal_quality,
)
from scripts.quality_gate_stress_report import synthetic_ecg


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("expected_sample_count", 1000.0),
        ("clipping_warn_run_samples", True),
        ("clipping_warn_run_samples", 0.5),
        ("clipping_reacquire_run_samples", 8.5),
        ("clipping_reacquire_run_samples", float("inf")),
        ("metadata_absolute_tolerance", True),
        ("amplitude_warn_mv", True),
        ("spike_warn_step_mv", np.bool_(True)),
        ("spectral_floor_hz", "0.1"),
        ("flat_warn_fraction", "0.985"),
        ("reversal_max_inverse_correlation", "-0.45"),
        ("amplitude_reacquire_mv", 10**1000),
    ],
)
def test_invalid_numeric_configuration_is_rejected_at_construction(field, value) -> None:
    with pytest.raises(ValueError):
        replace(DEFAULT_SIGNAL_QUALITY_CONFIG, **{field: value})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0, 0.0])
@pytest.mark.parametrize("field", ["amplitude_warn_mv", "flat_warn_fraction"])
def test_invalid_ranges_remain_rejected(field: str, value: float) -> None:
    with pytest.raises(ValueError):
        replace(DEFAULT_SIGNAL_QUALITY_CONFIG, **{field: value})


@pytest.mark.parametrize(
    "config", [DEFAULT_SIGNAL_QUALITY_CONFIG, SENTINEL_V2_SIGNAL_QUALITY_CONFIG]
)
def test_valid_numeric_overrides_preserve_quality_reports(config) -> None:
    equivalent = replace(config, amplitude_warn_mv=5, expected_sample_rate_hz=100)
    signal = synthetic_ecg()
    assert assess_signal_quality(signal, SignalMetadata.canonical(config), config=config) == (
        assess_signal_quality(signal, SignalMetadata.canonical(equivalent), config=equivalent)
    )
