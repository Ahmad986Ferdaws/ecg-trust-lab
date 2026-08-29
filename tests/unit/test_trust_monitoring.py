from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest
from numpy.typing import ArrayLike, NDArray

from ecg_trust.monitoring.trust_monitoring import (
    TELEMETRY_SCHEMA_VERSION,
    AggregateTelemetryWindow,
    AlertSeverity,
    DecisionState,
    DecisionStateCount,
    EvidenceStatus,
    MetricFamily,
    MonitoringComparison,
    MonitoringStatus,
    QualityReasonCode,
    QualityReasonCount,
    RateComparison,
    RecommendedAction,
    ScoreHistogram,
    TelemetryValidationError,
    TrustMonitoringConfig,
    compare_telemetry_windows,
    population_stability_index,
    replay_against_frozen_reference,
)


@pytest.fixture
def config() -> TrustMonitoringConfig:
    return TrustMonitoringConfig(
        version="test-monitor-v1",
        score_bin_edges=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        minimum_reference_samples=100,
        minimum_window_samples=100,
        minimum_score_samples=50,
        rate_investigate_delta=0.02,
        rate_restrict_delta=0.05,
        rate_pause_delta=0.10,
        rate_rollback_delta=0.20,
        psi_investigate=0.10,
        psi_restrict=0.20,
        psi_pause=0.30,
        psi_rollback=0.50,
    )


def _stable_scores(count: int = 80) -> NDArray[np.float64]:
    pattern = np.asarray((0.10, 0.25, 0.45, 0.65, 0.85), dtype=np.float64)
    return np.resize(pattern, count)


def _window(
    index: int,
    config: TrustMonitoringConfig,
    *,
    sample_count: int = 100,
    abstain: int = 3,
    unsupported: int = 2,
    invalid: int = 2,
    reacquire: int = 3,
    flatline: int = 2,
    baseline_wander: int = 1,
    scores: ArrayLike | None = None,
) -> AggregateTelemetryWindow:
    allowed = sample_count - abstain - unsupported - invalid - reacquire
    if allowed < 0:
        raise AssertionError("test decision counts exceed sample_count")
    decision_counts = {
        DecisionState.INVALID_INPUT: invalid,
        DecisionState.REACQUIRE: reacquire,
        DecisionState.UNSUPPORTED_INPUT: unsupported,
        DecisionState.ABSTAIN: abstain,
        DecisionState.PREDICTION_ALLOWED: allowed,
    }
    quality_counts = {
        QualityReasonCode.FLATLINE: flatline,
        QualityReasonCode.BASELINE_WANDER: baseline_wander,
    }
    score_values = _stable_scores(min(80, sample_count)) if scores is None else scores
    return AggregateTelemetryWindow.create(
        window_index=index,
        decision_counts=decision_counts,
        quality_reason_counts=quality_counts,
        scores=score_values,
        config=config,
    )


def _rate(
    comparison: MonitoringComparison,
    family: MetricFamily,
    key: str,
) -> RateComparison:
    return next(
        item for item in comparison.rate_comparisons if item.family is family and item.key == key
    )


def test_stable_aggregate_window_is_ok_json_safe_and_round_trips(
    config: TrustMonitoringConfig,
) -> None:
    reference = _window(0, config)
    current = _window(1, config)

    comparison = compare_telemetry_windows(reference, current, config=config)
    payload = reference.to_dict()
    serialized = comparison.to_json()

    assert comparison.status is MonitoringStatus.OK
    assert comparison.severity is AlertSeverity.NONE
    assert comparison.recommended_action is RecommendedAction.NONE
    assert comparison.alerts == ()
    assert comparison.score_drift.psi == pytest.approx(0.0)
    assert comparison.score_drift.evidence_status is EvidenceStatus.SUFFICIENT
    assert AggregateTelemetryWindow.from_dict(payload) == reference
    assert json.loads(serialized) == comparison.to_dict()
    json.dumps(payload, allow_nan=False)
    assert set(payload) == {
        "schema_version",
        "config_version",
        "window_index",
        "sample_count",
        "decision_counts",
        "quality_reason_counts",
        "score_histogram",
    }
    assert payload["schema_version"] == TELEMETRY_SCHEMA_VERSION
    lowered = json.dumps(payload).lower()
    assert "waveform" not in lowered
    assert "patient" not in lowered
    assert "record_id" not in lowered
    assert "diagnosis" not in lowered

    unsupported_rate = _rate(
        comparison,
        MetricFamily.UNSUPPORTED_INPUT_RATE,
        DecisionState.UNSUPPORTED_INPUT.value,
    )
    abstention_rate = _rate(
        comparison,
        MetricFamily.ABSTENTION_RATE,
        DecisionState.ABSTAIN.value,
    )
    assert unsupported_rate.current_rate == pytest.approx(0.02)
    assert abstention_rate.current_rate == pytest.approx(0.03)


def test_minimum_sample_statuses_distinguish_insufficient_and_partial_evidence(
    config: TrustMonitoringConfig,
) -> None:
    small_reference = _window(0, config, sample_count=40, scores=np.full(20, 0.2))
    small_current = _window(1, config, sample_count=40, scores=np.full(20, 0.2))
    insufficient = compare_telemetry_windows(
        small_reference,
        small_current,
        config=config,
    )

    reference = _window(0, config)
    low_score_current = _window(1, config, scores=np.full(20, 0.2))
    partial = compare_telemetry_windows(reference, low_score_current, config=config)

    assert insufficient.status is MonitoringStatus.EVIDENCE_INSUFFICIENT
    assert insufficient.severity is AlertSeverity.NOTICE
    assert insufficient.recommended_action is RecommendedAction.INVESTIGATE
    assert insufficient.score_drift.psi is None
    assert set(insufficient.evidence_reasons) == {
        "reference_window_below_minimum",
        "current_window_below_minimum",
        "reference_score_histogram_below_minimum",
        "current_score_histogram_below_minimum",
    }
    assert partial.status is MonitoringStatus.PARTIAL_EVIDENCE
    assert partial.score_drift.evidence_status is EvidenceStatus.INSUFFICIENT
    assert partial.score_drift.psi is None
    assert all(
        item.evidence_status is EvidenceStatus.SUFFICIENT for item in partial.rate_comparisons
    )


@pytest.mark.parametrize(
    ("abstain", "severity", "action"),
    [
        (4, AlertSeverity.NONE, RecommendedAction.NONE),
        (6, AlertSeverity.WARNING, RecommendedAction.INVESTIGATE),
        (9, AlertSeverity.HIGH, RecommendedAction.RESTRICT),
        (14, AlertSeverity.CRITICAL, RecommendedAction.PAUSE),
        (24, AlertSeverity.EMERGENCY, RecommendedAction.ROLLBACK),
    ],
)
def test_abstention_rate_uses_explicit_action_ladder(
    config: TrustMonitoringConfig,
    abstain: int,
    severity: AlertSeverity,
    action: RecommendedAction,
) -> None:
    reference = _window(0, config)
    current = _window(1, config, abstain=abstain)

    comparison = compare_telemetry_windows(reference, current, config=config)
    abstention = _rate(
        comparison,
        MetricFamily.ABSTENTION_RATE,
        DecisionState.ABSTAIN.value,
    )

    assert abstention.severity is severity
    assert abstention.recommended_action is action
    assert comparison.recommended_action is action
    expected_status = (
        MonitoringStatus.OK if action is RecommendedAction.NONE else MonitoringStatus.ALERT
    )
    assert comparison.status is expected_status


def test_unsupported_input_and_quality_reason_rates_are_independently_visible(
    config: TrustMonitoringConfig,
) -> None:
    reference = _window(0, config)
    current = _window(1, config, unsupported=14, flatline=24)

    comparison = compare_telemetry_windows(reference, current, config=config)
    unsupported = _rate(
        comparison,
        MetricFamily.UNSUPPORTED_INPUT_RATE,
        DecisionState.UNSUPPORTED_INPUT.value,
    )
    flatline = _rate(
        comparison,
        MetricFamily.QUALITY_REASON,
        QualityReasonCode.FLATLINE.value,
    )

    assert unsupported.recommended_action is RecommendedAction.PAUSE
    assert flatline.recommended_action is RecommendedAction.ROLLBACK
    assert comparison.recommended_action is RecommendedAction.ROLLBACK
    assert comparison.severity is AlertSeverity.EMERGENCY


def test_score_distribution_psi_uses_frozen_bins_and_escalates_abrupt_shift(
    config: TrustMonitoringConfig,
) -> None:
    reference = _window(0, config, scores=np.full(80, 0.10))
    current = _window(1, config, scores=np.full(80, 0.90))

    comparison = compare_telemetry_windows(reference, current, config=config)

    assert comparison.status is MonitoringStatus.ALERT
    assert comparison.score_drift.psi is not None
    assert comparison.score_drift.psi >= config.psi_rollback
    assert comparison.score_drift.severity is AlertSeverity.EMERGENCY
    assert comparison.score_drift.recommended_action is RecommendedAction.ROLLBACK
    assert comparison.score_drift in comparison.alerts


def test_gradual_shift_replay_compares_every_window_to_frozen_reference(
    config: TrustMonitoringConfig,
) -> None:
    reference = _window(0, config)
    windows = tuple(
        _window(index, config, abstain=abstain)
        for index, abstain in enumerate((4, 6, 9, 14, 24), start=1)
    )

    comparisons = replay_against_frozen_reference(reference, windows, config=config)

    assert tuple(item.reference_window_index for item in comparisons) == (0, 0, 0, 0, 0)
    assert tuple(item.recommended_action for item in comparisons) == (
        RecommendedAction.NONE,
        RecommendedAction.INVESTIGATE,
        RecommendedAction.RESTRICT,
        RecommendedAction.PAUSE,
        RecommendedAction.ROLLBACK,
    )


def test_histogram_psi_is_deterministic_finite_and_symmetric() -> None:
    first = ScoreHistogram((0.0, 0.5, 1.0), (100, 0))
    second = ScoreHistogram((0.0, 0.5, 1.0), (0, 100))

    same = population_stability_index(first, first, smoothing_count=1e-6)
    forward = population_stability_index(first, second, smoothing_count=1e-6)
    reverse = population_stability_index(second, first, smoothing_count=1e-6)

    assert same == pytest.approx(0.0)
    assert math_is_finite(forward)
    assert forward == pytest.approx(reverse)
    assert forward > 0.0


@pytest.mark.parametrize(
    "scores",
    [
        [[0.1, 0.2]],
        [0.1, np.nan],
        [-0.1, 0.2],
        [0.1, 1.1],
        np.asarray([0.1 + 0.2j]),
        np.asarray([True, False]),
        ["not-a-score"],
    ],
)
def test_malformed_scores_are_rejected(
    config: TrustMonitoringConfig,
    scores: ArrayLike,
) -> None:
    with pytest.raises(TelemetryValidationError):
        ScoreHistogram.from_scores(scores, config)


def test_malformed_counts_histograms_and_configs_are_rejected(
    config: TrustMonitoringConfig,
) -> None:
    with pytest.raises(TelemetryValidationError, match="non-negative integer"):
        DecisionStateCount(DecisionState.PREDICTION_ALLOWED, -1)
    with pytest.raises(TelemetryValidationError, match="QualityReasonCode"):
        QualityReasonCount("patient_123", 1)  # type: ignore[arg-type]
    with pytest.raises(TelemetryValidationError, match="edge count minus one"):
        ScoreHistogram((0.0, 0.5, 1.0), (1,))
    with pytest.raises(TelemetryValidationError, match="strictly increasing"):
        ScoreHistogram((0.0, 0.5, 0.5), (1, 1))
    with pytest.raises(TelemetryValidationError, match="strictly increasing"):
        replace(config, rate_restrict_delta=0.01)
    with pytest.raises(TelemetryValidationError, match="closed interval"):
        replace(config, score_bin_edges=(0.1, 0.5, 1.0))
    with pytest.raises(TelemetryValidationError, match="smoothing_count"):
        replace(config, psi_smoothing_count=True)
    with pytest.raises(FrozenInstanceError):
        config.minimum_window_samples = 1  # type: ignore[misc]

    missing_state: dict[DecisionState, int] = {
        state: 0 for state in DecisionState if state is not DecisionState.INVALID_INPUT
    }
    with pytest.raises(TelemetryValidationError, match="every DecisionState"):
        AggregateTelemetryWindow.create(
            window_index=0,
            decision_counts=missing_state,
            quality_reason_counts={},
            scores=[],
            config=config,
        )
    with pytest.raises(TelemetryValidationError, match="QualityReasonCode"):
        AggregateTelemetryWindow.create(
            window_index=0,
            decision_counts={state: 0 for state in DecisionState},
            quality_reason_counts={"patient_123": 0},  # type: ignore[dict-item]
            scores=[],
            config=config,
        )


def test_window_constructor_rejects_inconsistent_aggregate_counts(
    config: TrustMonitoringConfig,
) -> None:
    decisions = tuple(DecisionStateCount(state, 0) for state in DecisionState)
    histogram = ScoreHistogram(config.score_bin_edges, (0,) * 5)

    with pytest.raises(TelemetryValidationError, match="sum exactly"):
        AggregateTelemetryWindow(
            config_version=config.version,
            window_index=0,
            sample_count=1,
            decision_counts=decisions,
            quality_reason_counts=(),
            score_histogram=histogram,
        )

    excessive_histogram = ScoreHistogram(config.score_bin_edges, (1, 1, 1, 1, 1))
    with pytest.raises(TelemetryValidationError, match="cannot exceed sample_count"):
        AggregateTelemetryWindow(
            config_version=config.version,
            window_index=0,
            sample_count=0,
            decision_counts=decisions,
            quality_reason_counts=(),
            score_histogram=excessive_histogram,
        )


def test_json_loader_rejects_unknown_schema_free_text_and_extra_identifier(
    config: TrustMonitoringConfig,
) -> None:
    payload = _window(0, config).to_dict()

    wrong_schema = dict(payload)
    wrong_schema["schema_version"] = "unknown"
    with pytest.raises(TelemetryValidationError, match="schema_version"):
        AggregateTelemetryWindow.from_dict(wrong_schema)

    extra_identifier = dict(payload)
    extra_identifier["patient_id"] = "forbidden"
    with pytest.raises(TelemetryValidationError, match="extra=.*patient_id"):
        AggregateTelemetryWindow.from_dict(extra_identifier)

    free_text_reason = json.loads(json.dumps(payload))
    free_text_reason["quality_reason_counts"][0]["reason"] = "patient_123"
    with pytest.raises(TelemetryValidationError, match="free-text"):
        AggregateTelemetryWindow.from_dict(free_text_reason)

    unknown_state = json.loads(json.dumps(payload))
    unknown_state["decision_counts"][0]["state"] = "diagnosis_positive"
    with pytest.raises(TelemetryValidationError, match="unknown decision state"):
        AggregateTelemetryWindow.from_dict(unknown_state)


def test_comparison_rejects_bin_config_and_window_order_mismatches(
    config: TrustMonitoringConfig,
) -> None:
    reference = _window(0, config)
    current = _window(1, config)
    other_config = replace(config, version="other-monitor-v1")
    other_window = _window(2, other_config)

    with pytest.raises(TelemetryValidationError, match="config_version"):
        compare_telemetry_windows(reference, other_window, config=config)
    with pytest.raises(TelemetryValidationError, match="must follow"):
        compare_telemetry_windows(current, reference, config=config)
    with pytest.raises(TelemetryValidationError, match="strictly increasing"):
        replay_against_frozen_reference(
            reference,
            (current, _window(1, config)),
            config=config,
        )

    mismatched = ScoreHistogram((0.0, 0.5, 1.0), (50, 50))
    with pytest.raises(TelemetryValidationError, match="identical frozen"):
        population_stability_index(
            reference.score_histogram,
            mismatched,
            smoothing_count=1e-6,
        )


def math_is_finite(value: float) -> bool:
    return bool(np.isfinite(value))
