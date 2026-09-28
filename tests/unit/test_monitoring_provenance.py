from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from ecg_trust.contracts import TrustDecision as DecisionState
from ecg_trust.monitoring.provenance import (
    FrozenMonitoringReference,
    monitoring_config_from_dict,
    monitoring_config_sha256,
    monitoring_config_to_dict,
)
from ecg_trust.monitoring.trust_monitoring import (
    AggregateTelemetryWindow,
    RecommendedAction,
    TelemetryValidationError,
    TrustMonitoringConfig,
    compare_telemetry_windows,
)
from scripts.replay_trust_monitoring import main


def _window(index: int, abstentions: int = 0) -> AggregateTelemetryWindow:
    counts = {state: 0 for state in DecisionState}
    counts[DecisionState.PREDICTION_ALLOWED] = 500 - abstentions
    counts[DecisionState.ABSTAIN] = abstentions
    return AggregateTelemetryWindow.create(
        window_index=index, decision_counts=counts, quality_reason_counts={}, scores=[0.3] * 500
    )


def test_bound_replay_detects_same_version_threshold_drift_and_preserves_v1() -> None:
    config = TrustMonitoringConfig()
    changed = replace(config, rate_investigate_delta=0.045)
    reference, current = _window(0), _window(1, 20)
    old_reference = reference.to_dict()
    old_comparison = compare_telemetry_windows(reference, current, config=config)
    assert old_comparison.recommended_action is RecommendedAction.INVESTIGATE
    # Demonstrate the pre-existing version-only comparison accepts threshold drift.
    assert compare_telemetry_windows(reference, current, config=changed).recommended_action is (
        RecommendedAction.NONE
    )
    frozen = FrozenMonitoringReference(reference, config)
    with pytest.raises(TelemetryValidationError, match="configuration drift"):
        frozen.replay((current,), config=changed)
    assert frozen.reference.to_dict() == old_reference
    assert frozen.compare(current).to_json() == old_comparison.to_json()
    assert frozen.replay((current,))[0].to_json() == old_comparison.to_json()
    report = json.loads(frozen.replay_to_json((current,)))
    assert report["comparisons"][0]["comparison"] == old_comparison.to_dict()
    assert report["config_sha256"] == frozen.config_sha256
    assert report["frozen_reference_sha256"] == frozen.envelope_sha256


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("version", "trust-monitoring-v2"),
        ("score_bin_edges", (0.0, 0.5, 1.0)),
        ("minimum_reference_samples", 501),
        ("minimum_window_samples", 201),
        ("minimum_score_samples", 101),
        ("rate_investigate_delta", 0.021),
        ("rate_restrict_delta", 0.051),
        ("rate_pause_delta", 0.11),
        ("rate_rollback_delta", 0.21),
        ("psi_investigate", 0.11),
        ("psi_restrict", 0.21),
        ("psi_pause", 0.31),
        ("psi_rollback", 0.51),
        ("psi_smoothing_count", 1e-5),
    ],
)
def test_every_configuration_field_changes_identity(field: str, value: object) -> None:
    original = TrustMonitoringConfig()
    body = monitoring_config_to_dict(original)
    body[field] = value
    changed = monitoring_config_from_dict(body)
    assert monitoring_config_sha256(original) != monitoring_config_sha256(changed)


def test_round_trip_requires_canonical_hash_bound_content() -> None:
    frozen = FrozenMonitoringReference(_window(0), TrustMonitoringConfig())
    payload = frozen.to_json()
    loaded = FrozenMonitoringReference.from_json(payload, expected_sha256=frozen.envelope_sha256)
    assert loaded == frozen
    assert loaded.to_json() == payload
    for modified in (
        payload.replace('"rate_investigate_delta":0.02', '"rate_investigate_delta":0.04'),
        payload.replace('"count":500', '"count":499'),
        payload.replace('"config_sha256":"', '"config_sha256":"0'),
        payload[:-1] + ',"patient_id":"forbidden"}',
        payload[:-1] + ',"config_sha256":"duplicate"}',
        payload + "\n",
    ):
        with pytest.raises(TelemetryValidationError):
            FrozenMonitoringReference.from_json(modified)
    with pytest.raises(TelemetryValidationError, match="identity mismatch"):
        FrozenMonitoringReference.from_json(payload, expected_sha256="0" * 64)


def test_resealed_configuration_requires_new_external_identity() -> None:
    original = FrozenMonitoringReference(_window(0), TrustMonitoringConfig())
    replacement = FrozenMonitoringReference(
        _window(0), replace(original.config, rate_investigate_delta=0.045)
    )
    with pytest.raises(TelemetryValidationError, match="identity mismatch"):
        FrozenMonitoringReference.from_json(
            replacement.to_json(), expected_sha256=original.envelope_sha256
        )


def test_reference_and_replay_contract_checks_remain_active() -> None:
    config = TrustMonitoringConfig()
    with pytest.raises(TelemetryValidationError, match="config_version"):
        FrozenMonitoringReference(_window(0), replace(config, version="another-version"))
    with pytest.raises(TelemetryValidationError, match="bins"):
        FrozenMonitoringReference(_window(0), replace(config, score_bin_edges=(0.0, 0.5, 1.0)))
    frozen = FrozenMonitoringReference(_window(0), config)
    with pytest.raises(TelemetryValidationError, match="increasing"):
        frozen.replay((_window(2), _window(1)))
    body = monitoring_config_to_dict(config)
    del body["psi_smoothing_count"]
    with pytest.raises(TelemetryValidationError, match="fields"):
        monitoring_config_from_dict(body)


def test_cli_freeze_replay_and_configuration_drift(tmp_path: Path) -> None:
    reference = tmp_path / "reference.json"
    reference.write_text(json.dumps(_window(0).to_dict()))
    sealed = tmp_path / "sealed.json"
    assert main(["freeze", "--reference", str(reference), "--output", str(sealed)]) == 0
    frozen = FrozenMonitoringReference.from_json(sealed.read_text())
    windows = tmp_path / "windows.json"
    windows.write_text(json.dumps([_window(1, 20).to_dict(), _window(2).to_dict()]))
    result = tmp_path / "replay.json"
    arguments = [
        "replay",
        "--frozen-reference",
        str(sealed),
        "--expected-sha256",
        frozen.envelope_sha256,
        "--windows",
        str(windows),
        "--output",
        str(result),
    ]
    assert main(arguments) == 0
    report = json.loads(result.read_text())
    assert [row["comparison"]["recommended_action"] for row in report["comparisons"]] == [
        "investigate",
        "none",
    ]
    existing = result.read_bytes()
    with pytest.raises(SystemExit) as caught:
        main(arguments)
    assert caught.value.code == 2
    assert result.read_bytes() == existing
    changed = tmp_path / "changed.json"
    changed.write_text(
        json.dumps(monitoring_config_to_dict(replace(frozen.config, rate_investigate_delta=0.045)))
    )
    new_output = tmp_path / "changed-output.json"
    arguments[-1] = str(new_output)
    with pytest.raises(SystemExit) as caught:
        main([*arguments, "--config", str(changed)])
    assert caught.value.code == 2
    assert not new_output.exists()
