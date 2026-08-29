"""Deterministic, provenance-bound single-ECG stress scenarios."""

from ecg_trust.stress.scenarios import (
    CANONICAL_SAMPLING_FREQUENCY_HZ,
    CONTROLLED_SENSITIVITY_LABEL,
    PHYSICAL_MV_DOMAIN,
    AppliedStress,
    StressKind,
    StressProvenance,
    StressScenario,
    StressScenarioError,
    apply_stress_scenario,
    waveform_sha256,
)

__all__ = [
    "CANONICAL_SAMPLING_FREQUENCY_HZ",
    "CONTROLLED_SENSITIVITY_LABEL",
    "PHYSICAL_MV_DOMAIN",
    "AppliedStress",
    "StressKind",
    "StressProvenance",
    "StressScenario",
    "StressScenarioError",
    "apply_stress_scenario",
    "waveform_sha256",
]
