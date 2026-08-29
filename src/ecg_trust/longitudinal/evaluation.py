"""Aggregate-only evaluation for fixed-horizon binary ECG risk."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum

from ecg_trust.longitudinal.contracts import (
    AGGREGATE_ONLY_LIMIT,
    NON_CAUSAL_TARGET_LIMIT,
    RESEARCH_USE_LIMIT,
    BinaryFutureTarget,
    LongitudinalError,
    TargetStatus,
    finite_probability,
)


class MetricStatus(StrEnum):
    """Validity or evidence limitation for one aggregate metric."""

    OK = "ok"
    NO_OBSERVED_OUTCOMES = "no_observed_outcomes"
    INSUFFICIENT_SAMPLE_SIZE = "insufficient_sample_size"
    NO_POSITIVE_EVENTS = "no_positive_events"
    NO_NEGATIVE_EVENTS = "no_negative_events"


class EvaluationStatus(StrEnum):
    """Overall evidence state across the requested metrics."""

    OK = "ok"
    PARTIAL_EVIDENCE = "partial_evidence"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class CalibrationBinStatus(StrEnum):
    """Whether a calibration cell may be reported."""

    OK = "ok"
    EMPTY = "empty"
    SUPPRESSED_LOW_COUNT = "suppressed_low_count"


@dataclass(frozen=True, slots=True, init=False)
class RiskObservation:
    """Identifier-free predicted risk paired with target availability."""

    predicted_risk: float
    outcome: int | None
    target_status: TargetStatus

    @classmethod
    def create(
        cls,
        *,
        predicted_risk: float,
        outcome: int | None,
        target_status: TargetStatus | str,
    ) -> RiskObservation:
        risk = finite_probability(predicted_risk)
        try:
            status = TargetStatus(target_status)
        except (TypeError, ValueError) as exc:
            raise LongitudinalError("target_status is invalid") from exc
        if status is TargetStatus.OBSERVED:
            if outcome not in {0, 1} or isinstance(outcome, bool):
                raise LongitudinalError("observed risk rows require an integer outcome 0 or 1")
        elif outcome is not None:
            raise LongitudinalError("censored risk rows require outcome=null")
        instance = object.__new__(cls)
        object.__setattr__(instance, "predicted_risk", risk)
        object.__setattr__(instance, "outcome", outcome)
        object.__setattr__(instance, "target_status", status)
        return instance

    @classmethod
    def from_target(
        cls,
        target: BinaryFutureTarget,
        *,
        predicted_risk: float,
    ) -> RiskObservation:
        if not isinstance(target, BinaryFutureTarget):
            raise TypeError("target must be a BinaryFutureTarget")
        return cls.create(
            predicted_risk=predicted_risk,
            outcome=target.value,
            target_status=target.status,
        )


@dataclass(frozen=True, slots=True, init=False)
class RiskEvaluationConfig:
    """Evidence and privacy thresholds for one fixed prediction horizon."""

    horizon_days: int
    minimum_evaluable_count: int
    calibration_bin_count: int
    minimum_calibration_bin_count: int

    @classmethod
    def create(
        cls,
        *,
        horizon_days: int,
        minimum_evaluable_count: int = 20,
        calibration_bin_count: int = 10,
        minimum_calibration_bin_count: int = 5,
    ) -> RiskEvaluationConfig:
        horizon = _bounded_int(horizon_days, "horizon_days", minimum=1, maximum=36_525)
        evaluable = _bounded_int(
            minimum_evaluable_count,
            "minimum_evaluable_count",
            minimum=2,
            maximum=10_000_000,
        )
        bins = _bounded_int(
            calibration_bin_count,
            "calibration_bin_count",
            minimum=2,
            maximum=100,
        )
        minimum_bin = _bounded_int(
            minimum_calibration_bin_count,
            "minimum_calibration_bin_count",
            minimum=2,
            maximum=10_000_000,
        )
        instance = object.__new__(cls)
        object.__setattr__(instance, "horizon_days", horizon)
        object.__setattr__(instance, "minimum_evaluable_count", evaluable)
        object.__setattr__(instance, "calibration_bin_count", bins)
        object.__setattr__(instance, "minimum_calibration_bin_count", minimum_bin)
        return instance


@dataclass(frozen=True, slots=True)
class MetricEstimate:
    """One metric value or an explicit reason it was withheld."""

    value: float | None
    status: MetricStatus

    def to_dict(self) -> dict[str, object]:
        return {"value": self.value, "status": self.status.value}


@dataclass(frozen=True, slots=True)
class CalibrationBin:
    """One fixed-width aggregate bin with low-count suppression."""

    lower_bound: float
    upper_bound: float
    includes_upper_bound: bool
    count: int | None
    mean_predicted_risk: float | None
    observed_event_rate: float | None
    status: CalibrationBinStatus

    def to_dict(self) -> dict[str, object]:
        return {
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "includes_upper_bound": self.includes_upper_bound,
            "count": self.count,
            "mean_predicted_risk": self.mean_predicted_risk,
            "observed_event_rate": self.observed_event_rate,
            "status": self.status.value,
        }


@dataclass(frozen=True, slots=True)
class TimeDependentRiskEvaluation:
    """Privacy-safe aggregate fixed-horizon evaluation; no row identifiers."""

    target_name: str
    horizon_days: int
    status: EvaluationStatus
    total_prediction_count: int
    observed_outcome_count: int
    positive_count: int
    negative_count: int
    excluded_status_counts: tuple[tuple[str, int], ...]
    auroc: MetricEstimate
    average_precision: MetricEstimate
    brier_score: MetricEstimate
    calibration_bins: tuple[CalibrationBin, ...]

    def to_public_dict(self) -> dict[str, object]:
        return {
            "target_name": self.target_name,
            "horizon_days": self.horizon_days,
            "status": self.status.value,
            "total_prediction_count": self.total_prediction_count,
            "observed_outcome_count": self.observed_outcome_count,
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
            "excluded_status_counts": dict(self.excluded_status_counts),
            "auroc": self.auroc.to_dict(),
            "average_precision": self.average_precision.to_dict(),
            "brier_score": self.brier_score.to_dict(),
            "calibration_bins": [item.to_dict() for item in self.calibration_bins],
            "privacy_contract": AGGREGATE_ONLY_LIMIT,
            "research_use_limit": RESEARCH_USE_LIMIT,
            "target_interpretation": NON_CAUSAL_TARGET_LIMIT,
            "censoring_handling": "observed_targets_only_no_ipcw",
        }


def evaluate_time_dependent_binary_risk(
    target_name: str,
    observations: Iterable[RiskObservation],
    config: RiskEvaluationConfig,
) -> TimeDependentRiskEvaluation:
    """Evaluate discrimination, Brier loss, and fixed-width calibration.

    Right-censored and insufficient-follow-up rows are counted but never treated
    as negatives. AUROC and average precision require both outcome classes;
    Brier score requires only the configured minimum number of observed labels.
    """

    if not isinstance(target_name, str) or not target_name or target_name != target_name.strip():
        raise LongitudinalError("target_name must be non-empty canonical text")
    if not isinstance(config, RiskEvaluationConfig):
        raise TypeError("config must be a RiskEvaluationConfig")
    materialized = tuple(observations)
    if any(not isinstance(item, RiskObservation) for item in materialized):
        raise LongitudinalError("observations must contain RiskObservation values")
    observed = tuple(item for item in materialized if item.target_status is TargetStatus.OBSERVED)
    risks = [item.predicted_risk for item in observed]
    outcomes = [int(item.outcome) for item in observed if item.outcome is not None]
    if len(risks) != len(outcomes):
        raise LongitudinalError("observed risk rows are internally inconsistent")
    positive_count = sum(outcomes)
    negative_count = len(outcomes) - positive_count
    excluded = Counter(
        item.target_status.value
        for item in materialized
        if item.target_status is not TargetStatus.OBSERVED
    )

    shared_status = _sample_status(len(outcomes), config.minimum_evaluable_count)
    if shared_status is not MetricStatus.OK:
        auroc = MetricEstimate(value=None, status=shared_status)
        average_precision = MetricEstimate(value=None, status=shared_status)
        brier = MetricEstimate(value=None, status=shared_status)
        overall = EvaluationStatus.INSUFFICIENT_EVIDENCE
    else:
        brier = MetricEstimate(value=_brier(risks, outcomes), status=MetricStatus.OK)
        class_status = _class_status(positive_count, negative_count)
        if class_status is MetricStatus.OK:
            auroc = MetricEstimate(value=_auroc(risks, outcomes), status=MetricStatus.OK)
            average_precision = MetricEstimate(
                value=_average_precision(risks, outcomes),
                status=MetricStatus.OK,
            )
            overall = EvaluationStatus.OK
        else:
            auroc = MetricEstimate(value=None, status=class_status)
            average_precision = MetricEstimate(value=None, status=class_status)
            overall = EvaluationStatus.PARTIAL_EVIDENCE

    bins = _calibration_bins(
        risks,
        outcomes,
        bin_count=config.calibration_bin_count,
        minimum_bin_count=config.minimum_calibration_bin_count,
    )
    return TimeDependentRiskEvaluation(
        target_name=target_name,
        horizon_days=config.horizon_days,
        status=overall,
        total_prediction_count=len(materialized),
        observed_outcome_count=len(observed),
        positive_count=positive_count,
        negative_count=negative_count,
        excluded_status_counts=tuple(sorted(excluded.items())),
        auroc=auroc,
        average_precision=average_precision,
        brier_score=brier,
        calibration_bins=bins,
    )


def _sample_status(count: int, minimum: int) -> MetricStatus:
    if count == 0:
        return MetricStatus.NO_OBSERVED_OUTCOMES
    if count < minimum:
        return MetricStatus.INSUFFICIENT_SAMPLE_SIZE
    return MetricStatus.OK


def _class_status(positive_count: int, negative_count: int) -> MetricStatus:
    if positive_count == 0:
        return MetricStatus.NO_POSITIVE_EVENTS
    if negative_count == 0:
        return MetricStatus.NO_NEGATIVE_EVENTS
    return MetricStatus.OK


def _brier(risks: Sequence[float], outcomes: Sequence[int]) -> float:
    return math.fsum(
        (risk - outcome) ** 2 for risk, outcome in zip(risks, outcomes, strict=True)
    ) / len(outcomes)


def _auroc(risks: Sequence[float], outcomes: Sequence[int]) -> float:
    """Mann-Whitney AUROC with average ranks for tied predictions."""

    ordered = sorted(zip(risks, outcomes, strict=True), key=lambda item: item[0])
    positive_rank_sum = 0.0
    position = 0
    while position < len(ordered):
        end = position + 1
        while end < len(ordered) and ordered[end][0] == ordered[position][0]:
            end += 1
        average_rank = ((position + 1) + end) / 2.0
        positive_rank_sum += average_rank * sum(item[1] for item in ordered[position:end])
        position = end
    positives = sum(outcomes)
    negatives = len(outcomes) - positives
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def _average_precision(risks: Sequence[float], outcomes: Sequence[int]) -> float:
    """Non-interpolated AP, grouped at tied risk thresholds."""

    ordered = sorted(
        zip(risks, outcomes, strict=True),
        key=lambda item: item[0],
        reverse=True,
    )
    total_positives = sum(outcomes)
    true_positives = 0
    seen = 0
    previous_true_positives = 0
    value = 0.0
    position = 0
    while position < len(ordered):
        end = position + 1
        while end < len(ordered) and ordered[end][0] == ordered[position][0]:
            end += 1
        true_positives += sum(item[1] for item in ordered[position:end])
        seen += end - position
        value += ((true_positives - previous_true_positives) / total_positives) * (
            true_positives / seen
        )
        previous_true_positives = true_positives
        position = end
    return value


def _calibration_bins(
    risks: Sequence[float],
    outcomes: Sequence[int],
    *,
    bin_count: int,
    minimum_bin_count: int,
) -> tuple[CalibrationBin, ...]:
    grouped: dict[int, list[tuple[float, int]]] = defaultdict(list)
    for risk, outcome in zip(risks, outcomes, strict=True):
        index = min(int(risk * bin_count), bin_count - 1)
        grouped[index].append((risk, outcome))
    bins: list[CalibrationBin] = []
    for index in range(bin_count):
        lower = index / bin_count
        upper = (index + 1) / bin_count
        rows = grouped[index]
        if not rows:
            bins.append(
                CalibrationBin(
                    lower_bound=lower,
                    upper_bound=upper,
                    includes_upper_bound=index == bin_count - 1,
                    count=0,
                    mean_predicted_risk=None,
                    observed_event_rate=None,
                    status=CalibrationBinStatus.EMPTY,
                )
            )
        elif len(rows) < minimum_bin_count:
            bins.append(
                CalibrationBin(
                    lower_bound=lower,
                    upper_bound=upper,
                    includes_upper_bound=index == bin_count - 1,
                    count=None,
                    mean_predicted_risk=None,
                    observed_event_rate=None,
                    status=CalibrationBinStatus.SUPPRESSED_LOW_COUNT,
                )
            )
        else:
            bins.append(
                CalibrationBin(
                    lower_bound=lower,
                    upper_bound=upper,
                    includes_upper_bound=index == bin_count - 1,
                    count=len(rows),
                    mean_predicted_risk=math.fsum(item[0] for item in rows) / len(rows),
                    observed_event_rate=math.fsum(item[1] for item in rows) / len(rows),
                    status=CalibrationBinStatus.OK,
                )
            )
    return tuple(bins)


def _bounded_int(value: object, name: str, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise LongitudinalError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise LongitudinalError(f"{name} must be in [{minimum}, {maximum}]")
    return value
