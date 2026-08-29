"""Privacy-safe aggregate monitoring for a locked ECG trust pipeline.

Only counts and fixed-bin score histograms are retained.  The module has no
concept of a waveform, patient identifier, diagnosis, model update, or
retraining operation.  It compares every observation window with one frozen
reference window and returns recommendations for a separate governance layer.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Self, cast

import numpy as np
from numpy.typing import ArrayLike

from ecg_trust.contracts import TrustDecision as DecisionState

TELEMETRY_SCHEMA_VERSION = "ecg_trust.aggregate_telemetry.v1"
COMPARISON_SCHEMA_VERSION = "ecg_trust.monitoring_comparison.v1"


class TelemetryValidationError(ValueError):
    """Raised when aggregate telemetry is malformed or privacy-unsafe."""


class QualityReasonCode(StrEnum):
    """Closed aggregate vocabulary; free-text telemetry is not accepted."""

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


class EvidenceStatus(StrEnum):
    """Whether a comparison has the configured minimum aggregate evidence."""

    SUFFICIENT = "sufficient"
    INSUFFICIENT = "insufficient"


class MonitoringStatus(StrEnum):
    """Top-level state for one frozen-reference comparison."""

    OK = "ok"
    PARTIAL_EVIDENCE = "partial_evidence"
    EVIDENCE_INSUFFICIENT = "evidence_insufficient"
    ALERT = "alert"


class AlertSeverity(StrEnum):
    """Ordered alert severity; severity never executes an action."""

    NONE = "none"
    NOTICE = "notice"
    WARNING = "warning"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class RecommendedAction(StrEnum):
    """Governance action ladder without an automatic actuator."""

    NONE = "none"
    INVESTIGATE = "investigate"
    RESTRICT = "restrict"
    PAUSE = "pause"
    ROLLBACK = "rollback"


class MetricFamily(StrEnum):
    """Families emitted by deterministic window comparison."""

    DECISION_STATE = "decision_state"
    UNSUPPORTED_INPUT_RATE = "unsupported_input_rate"
    ABSTENTION_RATE = "abstention_rate"
    QUALITY_REASON = "quality_reason"
    SCORE_DISTRIBUTION = "score_distribution"


def _validate_increasing_thresholds(
    name: str,
    values: tuple[float, float, float, float],
    *,
    upper_bound: float | None,
) -> None:
    if any(isinstance(value, bool) or not math.isfinite(value) or value <= 0.0 for value in values):
        raise TelemetryValidationError(f"{name} must be finite and positive")
    if not all(left < right for left, right in zip(values, values[1:], strict=False)):
        raise TelemetryValidationError(f"{name} must be strictly increasing")
    if upper_bound is not None and values[-1] > upper_bound:
        raise TelemetryValidationError(f"{name} cannot exceed {upper_bound}")


@dataclass(frozen=True, slots=True)
class TrustMonitoringConfig:
    """Frozen thresholds, bins, and evidence requirements."""

    version: str = "trust-monitoring-v1"
    score_bin_edges: tuple[float, ...] = (
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
    )
    minimum_reference_samples: int = 500
    minimum_window_samples: int = 200
    minimum_score_samples: int = 100
    rate_investigate_delta: float = 0.02
    rate_restrict_delta: float = 0.05
    rate_pause_delta: float = 0.10
    rate_rollback_delta: float = 0.20
    psi_investigate: float = 0.10
    psi_restrict: float = 0.20
    psi_pause: float = 0.30
    psi_rollback: float = 0.50
    psi_smoothing_count: float = 1e-6

    def __post_init__(self) -> None:
        if not self.version or len(self.version) > 64 or not self.version.isascii():
            raise TelemetryValidationError("config version must be non-empty ASCII")
        if any(
            character not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for character in self.version
        ):
            raise TelemetryValidationError("config version contains unsupported characters")
        if len(self.score_bin_edges) < 3:
            raise TelemetryValidationError("score_bin_edges must define at least two bins")
        if any(isinstance(edge, bool) or not math.isfinite(edge) for edge in self.score_bin_edges):
            raise TelemetryValidationError("score_bin_edges must be finite")
        if any(
            right <= left
            for left, right in zip(
                self.score_bin_edges,
                self.score_bin_edges[1:],
                strict=False,
            )
        ):
            raise TelemetryValidationError("score_bin_edges must be strictly increasing")
        if self.score_bin_edges[0] != 0.0 or self.score_bin_edges[-1] != 1.0:
            raise TelemetryValidationError("score_bin_edges must span the closed interval [0, 1]")
        for name, value in (
            ("minimum_reference_samples", self.minimum_reference_samples),
            ("minimum_window_samples", self.minimum_window_samples),
            ("minimum_score_samples", self.minimum_score_samples),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise TelemetryValidationError(f"{name} must be a positive integer")
        _validate_increasing_thresholds(
            "rate deltas",
            (
                self.rate_investigate_delta,
                self.rate_restrict_delta,
                self.rate_pause_delta,
                self.rate_rollback_delta,
            ),
            upper_bound=1.0,
        )
        _validate_increasing_thresholds(
            "PSI thresholds",
            (self.psi_investigate, self.psi_restrict, self.psi_pause, self.psi_rollback),
            upper_bound=None,
        )
        if (
            isinstance(self.psi_smoothing_count, bool)
            or not math.isfinite(self.psi_smoothing_count)
            or self.psi_smoothing_count <= 0.0
        ):
            raise TelemetryValidationError("psi_smoothing_count must be finite and positive")


DEFAULT_TRUST_MONITORING_CONFIG = TrustMonitoringConfig()


@dataclass(frozen=True, slots=True)
class DecisionStateCount:
    """Aggregate count for one mutually exclusive decision state."""

    state: DecisionState
    count: int

    def __post_init__(self) -> None:
        if not isinstance(self.state, DecisionState):
            raise TelemetryValidationError("decision state must use DecisionState")
        _validate_count(self.count, "decision state count")

    def to_dict(self) -> dict[str, object]:
        return {"state": self.state.value, "count": self.count}


@dataclass(frozen=True, slots=True)
class QualityReasonCount:
    """Aggregate count for one closed-vocabulary quality reason."""

    reason: QualityReasonCode
    count: int

    def __post_init__(self) -> None:
        if not isinstance(self.reason, QualityReasonCode):
            raise TelemetryValidationError("quality reason must use QualityReasonCode")
        _validate_count(self.count, "quality reason count")

    def to_dict(self) -> dict[str, object]:
        return {"reason": self.reason.value, "count": self.count}


@dataclass(frozen=True, slots=True)
class ScoreHistogram:
    """Fixed-bin aggregate of model scores; individual scores are not retained."""

    bin_edges: tuple[float, ...]
    counts: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.bin_edges) < 3:
            raise TelemetryValidationError("histogram must contain at least two bins")
        if len(self.counts) != len(self.bin_edges) - 1:
            raise TelemetryValidationError("histogram count length must equal edge count minus one")
        if any(isinstance(edge, bool) or not math.isfinite(edge) for edge in self.bin_edges):
            raise TelemetryValidationError("histogram edges must be finite")
        if any(
            right <= left for left, right in zip(self.bin_edges, self.bin_edges[1:], strict=False)
        ):
            raise TelemetryValidationError("histogram edges must be strictly increasing")
        for count in self.counts:
            _validate_count(count, "histogram count")

    @property
    def sample_count(self) -> int:
        return sum(self.counts)

    def to_dict(self) -> dict[str, object]:
        return {"bin_edges": list(self.bin_edges), "counts": list(self.counts)}

    @classmethod
    def from_scores(
        cls,
        scores: ArrayLike,
        config: TrustMonitoringConfig = DEFAULT_TRUST_MONITORING_CONFIG,
    ) -> Self:
        """Aggregate ephemeral scores immediately into the frozen bins."""

        try:
            raw = np.asarray(scores)
        except (TypeError, ValueError) as error:
            raise TelemetryValidationError(
                "scores must be a numeric one-dimensional array"
            ) from error
        if raw.ndim != 1:
            raise TelemetryValidationError("scores must be one-dimensional")
        if np.iscomplexobj(raw) or np.issubdtype(raw.dtype, np.bool_):
            raise TelemetryValidationError("scores must be real numeric values")
        try:
            numeric = np.asarray(raw, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise TelemetryValidationError("scores must be numeric") from error
        if not np.isfinite(numeric).all():
            raise TelemetryValidationError("scores must be finite")
        if np.any(numeric < config.score_bin_edges[0]) or np.any(
            numeric > config.score_bin_edges[-1]
        ):
            raise TelemetryValidationError("scores must be inside the frozen score-bin range")
        counts, _ = np.histogram(numeric, bins=np.asarray(config.score_bin_edges))
        return cls(
            bin_edges=config.score_bin_edges,
            counts=tuple(int(value) for value in counts),
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> Self:
        _require_exact_keys(payload, {"bin_edges", "counts"}, "score histogram")
        edges = tuple(
            _require_float(value, f"bin_edges[{index}]")
            for index, value in enumerate(_require_list(payload["bin_edges"], "bin_edges"))
        )
        counts = tuple(
            _require_int(value, f"counts[{index}]", minimum=0)
            for index, value in enumerate(_require_list(payload["counts"], "counts"))
        )
        return cls(bin_edges=edges, counts=counts)


@dataclass(frozen=True, slots=True)
class AggregateTelemetryWindow:
    """One privacy-safe monitoring window containing aggregate values only."""

    config_version: str
    window_index: int
    sample_count: int
    decision_counts: tuple[DecisionStateCount, ...]
    quality_reason_counts: tuple[QualityReasonCount, ...]
    score_histogram: ScoreHistogram

    def __post_init__(self) -> None:
        if (
            not self.config_version
            or len(self.config_version) > 64
            or not self.config_version.isascii()
            or any(
                character not in "abcdefghijklmnopqrstuvwxyz0123456789-_"
                for character in self.config_version
            )
        ):
            raise TelemetryValidationError("config_version must be non-empty ASCII")
        _require_int(self.window_index, "window_index", minimum=0)
        _validate_count(self.sample_count, "sample_count")

        expected_states = tuple(DecisionState)
        observed_states = tuple(item.state for item in self.decision_counts)
        if observed_states != expected_states:
            raise TelemetryValidationError(
                "decision_counts must contain every DecisionState once in canonical order"
            )
        if sum(item.count for item in self.decision_counts) != self.sample_count:
            raise TelemetryValidationError("decision counts must sum exactly to sample_count")

        observed_reasons = tuple(item.reason for item in self.quality_reason_counts)
        if len(set(observed_reasons)) != len(observed_reasons):
            raise TelemetryValidationError("quality reason counts must be unique")
        if observed_reasons != tuple(sorted(observed_reasons, key=lambda reason: reason.value)):
            raise TelemetryValidationError("quality reason counts must use canonical sorted order")
        if any(item.count > self.sample_count for item in self.quality_reason_counts):
            raise TelemetryValidationError(
                "an individual quality reason count cannot exceed sample_count"
            )
        if self.score_histogram.sample_count > self.sample_count:
            raise TelemetryValidationError("score histogram cannot exceed sample_count")

    @classmethod
    def create(
        cls,
        *,
        window_index: int,
        decision_counts: Mapping[DecisionState, int],
        quality_reason_counts: Mapping[QualityReasonCode, int],
        scores: ArrayLike,
        config: TrustMonitoringConfig = DEFAULT_TRUST_MONITORING_CONFIG,
    ) -> Self:
        """Canonicalize aggregate counters and immediately bin transient scores."""

        if set(decision_counts) != set(DecisionState):
            raise TelemetryValidationError("decision_counts must contain every DecisionState")
        if any(not isinstance(reason, QualityReasonCode) for reason in quality_reason_counts):
            raise TelemetryValidationError("quality_reason_counts keys must use QualityReasonCode")
        decision_items = tuple(
            DecisionStateCount(state=state, count=decision_counts[state]) for state in DecisionState
        )
        quality_items = tuple(
            QualityReasonCount(reason=reason, count=count)
            for reason, count in sorted(
                quality_reason_counts.items(), key=lambda item: item[0].value
            )
        )
        sample_count = sum(item.count for item in decision_items)
        return cls(
            config_version=config.version,
            window_index=window_index,
            sample_count=sample_count,
            decision_counts=decision_items,
            quality_reason_counts=quality_items,
            score_histogram=ScoreHistogram.from_scores(scores, config),
        )

    def decision_count(self, state: DecisionState) -> int:
        return self.decision_counts[tuple(DecisionState).index(state)].count

    def quality_reason_count(self, reason: QualityReasonCode) -> int:
        for item in self.quality_reason_counts:
            if item.reason is reason:
                return item.count
        return 0

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": TELEMETRY_SCHEMA_VERSION,
            "config_version": self.config_version,
            "window_index": self.window_index,
            "sample_count": self.sample_count,
            "decision_counts": [item.to_dict() for item in self.decision_counts],
            "quality_reason_counts": [item.to_dict() for item in self.quality_reason_counts],
            "score_histogram": self.score_histogram.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> Self:
        required = {
            "schema_version",
            "config_version",
            "window_index",
            "sample_count",
            "decision_counts",
            "quality_reason_counts",
            "score_histogram",
        }
        _require_exact_keys(payload, required, "aggregate telemetry window")
        if payload["schema_version"] != TELEMETRY_SCHEMA_VERSION:
            raise TelemetryValidationError("unsupported aggregate telemetry schema_version")

        decision_items: list[DecisionStateCount] = []
        for index, value in enumerate(_require_list(payload["decision_counts"], "decision_counts")):
            item = _require_mapping(value, f"decision_counts[{index}]")
            _require_exact_keys(item, {"state", "count"}, f"decision_counts[{index}]")
            try:
                state = DecisionState(_require_string(item["state"], "state"))
            except ValueError as error:
                raise TelemetryValidationError("unknown decision state") from error
            decision_items.append(
                DecisionStateCount(
                    state=state,
                    count=_require_int(item["count"], "decision count", minimum=0),
                )
            )

        quality_items: list[QualityReasonCount] = []
        for index, value in enumerate(
            _require_list(payload["quality_reason_counts"], "quality_reason_counts")
        ):
            item = _require_mapping(value, f"quality_reason_counts[{index}]")
            _require_exact_keys(item, {"reason", "count"}, f"quality_reason_counts[{index}]")
            try:
                reason = QualityReasonCode(_require_string(item["reason"], "reason"))
            except ValueError as error:
                raise TelemetryValidationError("unknown or free-text quality reason") from error
            quality_items.append(
                QualityReasonCount(
                    reason=reason,
                    count=_require_int(item["count"], "quality count", minimum=0),
                )
            )

        histogram_payload = _require_mapping(payload["score_histogram"], "score_histogram")
        return cls(
            config_version=_require_string(payload["config_version"], "config_version"),
            window_index=_require_int(payload["window_index"], "window_index", minimum=0),
            sample_count=_require_int(payload["sample_count"], "sample_count", minimum=0),
            decision_counts=tuple(decision_items),
            quality_reason_counts=tuple(quality_items),
            score_histogram=ScoreHistogram.from_dict(histogram_payload),
        )


@dataclass(frozen=True, slots=True)
class RateComparison:
    """Reference/current comparison for one aggregate event rate."""

    family: MetricFamily
    key: str
    reference_count: int
    current_count: int
    reference_rate: float
    current_rate: float
    delta: float
    adverse_delta: float
    evidence_status: EvidenceStatus
    severity: AlertSeverity
    recommended_action: RecommendedAction

    def to_dict(self) -> dict[str, object]:
        return {
            "family": self.family.value,
            "key": self.key,
            "reference_count": self.reference_count,
            "current_count": self.current_count,
            "reference_rate": self.reference_rate,
            "current_rate": self.current_rate,
            "delta": self.delta,
            "adverse_delta": self.adverse_delta,
            "evidence_status": self.evidence_status.value,
            "severity": self.severity.value,
            "recommended_action": self.recommended_action.value,
        }


@dataclass(frozen=True, slots=True)
class ScoreDriftComparison:
    """Population Stability Index over frozen reference bins."""

    family: MetricFamily
    psi: float | None
    reference_score_samples: int
    current_score_samples: int
    evidence_status: EvidenceStatus
    severity: AlertSeverity
    recommended_action: RecommendedAction

    def to_dict(self) -> dict[str, object]:
        return {
            "family": self.family.value,
            "psi": self.psi,
            "reference_score_samples": self.reference_score_samples,
            "current_score_samples": self.current_score_samples,
            "evidence_status": self.evidence_status.value,
            "severity": self.severity.value,
            "recommended_action": self.recommended_action.value,
        }


@dataclass(frozen=True, slots=True)
class MonitoringComparison:
    """JSON-safe monitoring result and non-executing governance recommendation."""

    config_version: str
    reference_window_index: int
    current_window_index: int
    status: MonitoringStatus
    severity: AlertSeverity
    recommended_action: RecommendedAction
    evidence_reasons: tuple[str, ...]
    rate_comparisons: tuple[RateComparison, ...]
    score_drift: ScoreDriftComparison

    @property
    def alerts(self) -> tuple[RateComparison | ScoreDriftComparison, ...]:
        rate_alerts: tuple[RateComparison | ScoreDriftComparison, ...] = tuple(
            comparison
            for comparison in self.rate_comparisons
            if _severity_rank(comparison.severity) >= _severity_rank(AlertSeverity.WARNING)
        )
        if _severity_rank(self.score_drift.severity) >= _severity_rank(AlertSeverity.WARNING):
            return (*rate_alerts, self.score_drift)
        return rate_alerts

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": COMPARISON_SCHEMA_VERSION,
            "config_version": self.config_version,
            "reference_window_index": self.reference_window_index,
            "current_window_index": self.current_window_index,
            "status": self.status.value,
            "severity": self.severity.value,
            "recommended_action": self.recommended_action.value,
            "evidence_reasons": list(self.evidence_reasons),
            "rate_comparisons": [item.to_dict() for item in self.rate_comparisons],
            "score_drift": self.score_drift.to_dict(),
        }

    def to_json(self) -> str:
        """Serialize without NaN or Infinity, sorted for deterministic auditing."""

        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )


def compare_telemetry_windows(
    reference: AggregateTelemetryWindow,
    current: AggregateTelemetryWindow,
    *,
    config: TrustMonitoringConfig = DEFAULT_TRUST_MONITORING_CONFIG,
) -> MonitoringComparison:
    """Compare one aggregate window with a frozen earlier reference window."""

    _validate_comparison_inputs(reference, current, config)
    evidence_reasons = _evidence_reasons(reference, current, config)
    rates_have_evidence = (
        reference.sample_count >= config.minimum_reference_samples
        and current.sample_count >= config.minimum_window_samples
    )
    score_has_evidence = rates_have_evidence and (
        reference.score_histogram.sample_count >= config.minimum_score_samples
        and current.score_histogram.sample_count >= config.minimum_score_samples
    )

    rate_comparisons: list[RateComparison] = []
    for state in DecisionState:
        family = _decision_metric_family(state)
        reference_count = reference.decision_count(state)
        current_count = current.decision_count(state)
        rate_comparisons.append(
            _compare_rate(
                family=family,
                key=state.value,
                reference_count=reference_count,
                current_count=current_count,
                reference_samples=reference.sample_count,
                current_samples=current.sample_count,
                lower_is_adverse=state is DecisionState.PREDICTION_ALLOWED,
                evidence_sufficient=rates_have_evidence,
                config=config,
            )
        )

    reasons = sorted(
        set(item.reason for item in reference.quality_reason_counts)
        | set(item.reason for item in current.quality_reason_counts),
        key=lambda reason: reason.value,
    )
    for reason in reasons:
        rate_comparisons.append(
            _compare_rate(
                family=MetricFamily.QUALITY_REASON,
                key=reason.value,
                reference_count=reference.quality_reason_count(reason),
                current_count=current.quality_reason_count(reason),
                reference_samples=reference.sample_count,
                current_samples=current.sample_count,
                lower_is_adverse=False,
                evidence_sufficient=rates_have_evidence,
                config=config,
            )
        )

    score_drift = _compare_score_histograms(
        reference.score_histogram,
        current.score_histogram,
        evidence_sufficient=score_has_evidence,
        config=config,
    )
    severities = [item.severity for item in rate_comparisons]
    severities.append(score_drift.severity)
    severity = _maximum_severity(severities)
    actions = [item.recommended_action for item in rate_comparisons]
    actions.append(score_drift.recommended_action)
    action = _maximum_action(actions)

    has_alert = _severity_rank(severity) >= _severity_rank(AlertSeverity.WARNING)
    if has_alert:
        status = MonitoringStatus.ALERT
    elif not rates_have_evidence:
        status = MonitoringStatus.EVIDENCE_INSUFFICIENT
    elif not score_has_evidence:
        status = MonitoringStatus.PARTIAL_EVIDENCE
    else:
        status = MonitoringStatus.OK

    return MonitoringComparison(
        config_version=config.version,
        reference_window_index=reference.window_index,
        current_window_index=current.window_index,
        status=status,
        severity=severity,
        recommended_action=action,
        evidence_reasons=evidence_reasons,
        rate_comparisons=tuple(rate_comparisons),
        score_drift=score_drift,
    )


def replay_against_frozen_reference(
    reference: AggregateTelemetryWindow,
    windows: Sequence[AggregateTelemetryWindow],
    *,
    config: TrustMonitoringConfig = DEFAULT_TRUST_MONITORING_CONFIG,
) -> tuple[MonitoringComparison, ...]:
    """Replay ordered windows without updating or adapting the reference."""

    previous_index = reference.window_index
    comparisons: list[MonitoringComparison] = []
    for window in windows:
        if window.window_index <= previous_index:
            raise TelemetryValidationError("replay windows must have strictly increasing indices")
        comparisons.append(compare_telemetry_windows(reference, window, config=config))
        previous_index = window.window_index
    return tuple(comparisons)


def population_stability_index(
    reference: ScoreHistogram,
    current: ScoreHistogram,
    *,
    smoothing_count: float,
) -> float:
    """Return deterministic PSI using fixed shared bins and finite smoothing."""

    if reference.bin_edges != current.bin_edges:
        raise TelemetryValidationError("PSI requires identical frozen bin edges")
    if reference.sample_count <= 0 or current.sample_count <= 0:
        raise TelemetryValidationError("PSI requires non-empty histograms")
    if (
        isinstance(smoothing_count, bool)
        or not math.isfinite(smoothing_count)
        or smoothing_count <= 0.0
    ):
        raise TelemetryValidationError("PSI smoothing_count must be finite and positive")

    reference_counts = np.asarray(reference.counts, dtype=np.float64) + smoothing_count
    current_counts = np.asarray(current.counts, dtype=np.float64) + smoothing_count
    reference_proportions = reference_counts / np.sum(reference_counts)
    current_proportions = current_counts / np.sum(current_counts)
    psi = np.sum(
        (current_proportions - reference_proportions)
        * np.log(current_proportions / reference_proportions)
    )
    result = float(psi)
    if not math.isfinite(result) or result < 0.0:
        raise TelemetryValidationError("PSI computation produced an invalid result")
    return result


def _compare_rate(
    *,
    family: MetricFamily,
    key: str,
    reference_count: int,
    current_count: int,
    reference_samples: int,
    current_samples: int,
    lower_is_adverse: bool,
    evidence_sufficient: bool,
    config: TrustMonitoringConfig,
) -> RateComparison:
    reference_rate = _safe_rate(reference_count, reference_samples)
    current_rate = _safe_rate(current_count, current_samples)
    delta = current_rate - reference_rate
    adverse_delta = -delta if lower_is_adverse else delta
    adverse_delta = max(0.0, adverse_delta)
    if evidence_sufficient:
        severity, action = _threshold_alert(
            adverse_delta,
            investigate=config.rate_investigate_delta,
            restrict=config.rate_restrict_delta,
            pause=config.rate_pause_delta,
            rollback=config.rate_rollback_delta,
        )
        evidence = EvidenceStatus.SUFFICIENT
    else:
        severity = AlertSeverity.NOTICE
        action = RecommendedAction.INVESTIGATE
        evidence = EvidenceStatus.INSUFFICIENT
    return RateComparison(
        family=family,
        key=key,
        reference_count=reference_count,
        current_count=current_count,
        reference_rate=reference_rate,
        current_rate=current_rate,
        delta=delta,
        adverse_delta=adverse_delta,
        evidence_status=evidence,
        severity=severity,
        recommended_action=action,
    )


def _compare_score_histograms(
    reference: ScoreHistogram,
    current: ScoreHistogram,
    *,
    evidence_sufficient: bool,
    config: TrustMonitoringConfig,
) -> ScoreDriftComparison:
    if evidence_sufficient:
        psi = population_stability_index(
            reference,
            current,
            smoothing_count=config.psi_smoothing_count,
        )
        severity, action = _threshold_alert(
            psi,
            investigate=config.psi_investigate,
            restrict=config.psi_restrict,
            pause=config.psi_pause,
            rollback=config.psi_rollback,
        )
        evidence = EvidenceStatus.SUFFICIENT
    else:
        psi = None
        severity = AlertSeverity.NOTICE
        action = RecommendedAction.INVESTIGATE
        evidence = EvidenceStatus.INSUFFICIENT
    return ScoreDriftComparison(
        family=MetricFamily.SCORE_DISTRIBUTION,
        psi=psi,
        reference_score_samples=reference.sample_count,
        current_score_samples=current.sample_count,
        evidence_status=evidence,
        severity=severity,
        recommended_action=action,
    )


def _threshold_alert(
    observed: float,
    *,
    investigate: float,
    restrict: float,
    pause: float,
    rollback: float,
) -> tuple[AlertSeverity, RecommendedAction]:
    if observed >= rollback:
        return AlertSeverity.EMERGENCY, RecommendedAction.ROLLBACK
    if observed >= pause:
        return AlertSeverity.CRITICAL, RecommendedAction.PAUSE
    if observed >= restrict:
        return AlertSeverity.HIGH, RecommendedAction.RESTRICT
    if observed >= investigate:
        return AlertSeverity.WARNING, RecommendedAction.INVESTIGATE
    return AlertSeverity.NONE, RecommendedAction.NONE


def _validate_comparison_inputs(
    reference: AggregateTelemetryWindow,
    current: AggregateTelemetryWindow,
    config: TrustMonitoringConfig,
) -> None:
    if reference.config_version != config.version or current.config_version != config.version:
        raise TelemetryValidationError("telemetry config_version does not match monitor config")
    if current.window_index <= reference.window_index:
        raise TelemetryValidationError("current window must follow the frozen reference")
    if reference.score_histogram.bin_edges != config.score_bin_edges:
        raise TelemetryValidationError("reference histogram does not use frozen config bins")
    if current.score_histogram.bin_edges != config.score_bin_edges:
        raise TelemetryValidationError("current histogram does not use frozen config bins")


def _evidence_reasons(
    reference: AggregateTelemetryWindow,
    current: AggregateTelemetryWindow,
    config: TrustMonitoringConfig,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if reference.sample_count < config.minimum_reference_samples:
        reasons.append("reference_window_below_minimum")
    if current.sample_count < config.minimum_window_samples:
        reasons.append("current_window_below_minimum")
    if reference.score_histogram.sample_count < config.minimum_score_samples:
        reasons.append("reference_score_histogram_below_minimum")
    if current.score_histogram.sample_count < config.minimum_score_samples:
        reasons.append("current_score_histogram_below_minimum")
    return tuple(reasons)


def _decision_metric_family(state: DecisionState) -> MetricFamily:
    if state is DecisionState.UNSUPPORTED_INPUT:
        return MetricFamily.UNSUPPORTED_INPUT_RATE
    if state is DecisionState.ABSTAIN:
        return MetricFamily.ABSTENTION_RATE
    return MetricFamily.DECISION_STATE


def _safe_rate(count: int, sample_count: int) -> float:
    return float(count / sample_count) if sample_count > 0 else 0.0


def _maximum_severity(severities: Iterable[AlertSeverity]) -> AlertSeverity:
    materialized = tuple(severities)
    if not materialized:
        return AlertSeverity.NONE
    return max(materialized, key=_severity_rank)


def _maximum_action(actions: Iterable[RecommendedAction]) -> RecommendedAction:
    materialized = tuple(actions)
    if not materialized:
        return RecommendedAction.NONE
    return max(materialized, key=_action_rank)


def _severity_rank(severity: AlertSeverity) -> int:
    return {
        AlertSeverity.NONE: 0,
        AlertSeverity.NOTICE: 1,
        AlertSeverity.WARNING: 2,
        AlertSeverity.HIGH: 3,
        AlertSeverity.CRITICAL: 4,
        AlertSeverity.EMERGENCY: 5,
    }[severity]


def _action_rank(action: RecommendedAction) -> int:
    return {
        RecommendedAction.NONE: 0,
        RecommendedAction.INVESTIGATE: 1,
        RecommendedAction.RESTRICT: 2,
        RecommendedAction.PAUSE: 3,
        RecommendedAction.ROLLBACK: 4,
    }[action]


def _validate_count(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TelemetryValidationError(f"{name} must be a non-negative integer")


def _require_exact_keys(
    payload: Mapping[str, object],
    expected: set[str],
    context: str,
) -> None:
    observed = set(payload)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise TelemetryValidationError(f"{context} keys mismatch; missing={missing}, extra={extra}")


def _require_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TelemetryValidationError(f"{name} must be an object")
    if not all(isinstance(key, str) for key in value):
        raise TelemetryValidationError(f"{name} keys must be strings")
    return cast(Mapping[str, object], value)


def _require_list(value: object, name: str) -> list[object]:
    if not isinstance(value, list):
        raise TelemetryValidationError(f"{name} must be an array")
    return cast(list[object], value)


def _require_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise TelemetryValidationError(f"{name} must be a non-empty string")
    return value


def _require_int(value: object, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise TelemetryValidationError(f"{name} must be an integer >= {minimum}")
    return value


def _require_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TelemetryValidationError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise TelemetryValidationError(f"{name} must be finite")
    return result
