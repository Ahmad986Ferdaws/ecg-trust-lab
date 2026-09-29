"""Aggregate target availability across risk bins beside observed-only metrics."""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

from ecg_trust.longitudinal.contracts import TargetStatus
from ecg_trust.longitudinal.evaluation import (
    RiskEvaluationConfig,
    RiskObservation,
    TimeDependentRiskEvaluation,
    evaluate_time_dependent_binary_risk,
)

FOLLOW_UP_REPORT_SCHEMA = "ecg_trust.longitudinal_risk_follow_up.v1"


class FollowUpBinStatus(StrEnum):
    OK = "ok"
    EMPTY = "empty"
    SUPPRESSED_LOW_COUNT = "suppressed_low_count"


@dataclass(frozen=True, slots=True)
class FollowUpCompletenessBin:
    """A fixed risk interval containing only disclosable aggregate status counts."""

    lower_bound: float
    upper_bound: float
    includes_upper_bound: bool
    status: FollowUpBinStatus
    counts: tuple[int, int, int] | None
    fractions: tuple[float, float, float] | None

    def to_public_dict(self) -> dict[str, object]:
        return {
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "includes_upper_bound": self.includes_upper_bound,
            "status": self.status.value,
            "total_prediction_count": None if self.counts is None else sum(self.counts),
            "target_status_counts": (
                None
                if self.counts is None
                else {
                    status.value: count
                    for status, count in zip(TargetStatus, self.counts, strict=True)
                }
            ),
            "target_status_fractions": (
                None
                if self.fractions is None
                else {
                    status.value: value
                    for status, value in zip(TargetStatus, self.fractions, strict=True)
                }
            ),
        }


@dataclass(frozen=True, slots=True)
class RiskEvaluationWithFollowUp:
    """An additive envelope; the embedded observed-only report is unchanged."""

    observed_only_evaluation: TimeDependentRiskEvaluation
    minimum_status_cell_count: int
    follow_up_bins: tuple[FollowUpCompletenessBin, ...]

    def to_public_dict(self) -> dict[str, object]:
        return {
            "schema_version": FOLLOW_UP_REPORT_SCHEMA,
            "observed_only_evaluation": self.observed_only_evaluation.to_public_dict(),
            "follow_up_completeness": {
                "bin_count": len(self.follow_up_bins),
                "minimum_status_cell_count": self.minimum_status_cell_count,
                "bins": [item.to_public_dict() for item in self.follow_up_bins],
                "interpretation": "target_availability_only_no_ipcw_or_selection_correction",
                "suppression": "whole_bin_if_any_nonzero_status_cell_below_minimum",
            },
        }


def evaluate_binary_risk_with_follow_up(
    target_name: str,
    observations: Iterable[RiskObservation],
    config: RiskEvaluationConfig,
) -> RiskEvaluationWithFollowUp:
    """Show where targets are unavailable without imputing excluded outcomes.

    Bin edges use the existing configured bin count, with left-closed intervals
    and an inclusive final endpoint at one. The existing calibration-bin minimum
    applies to each nonzero availability cell. If any cell is below this minimum,
    the whole bin's counts and fractions are suppressed.
    """

    materialized = tuple(observations)
    evaluation = evaluate_time_dependent_binary_risk(target_name, materialized, config)
    edges = tuple(
        index / config.calibration_bin_count for index in range(config.calibration_bin_count + 1)
    )
    counts = [[0, 0, 0] for _ in range(config.calibration_bin_count)]
    statuses = tuple(TargetStatus)
    for observation in materialized:
        index = min(bisect_right(edges, observation.predicted_risk) - 1, len(counts) - 1)
        counts[index][statuses.index(observation.target_status)] += 1
    bins: list[FollowUpCompletenessBin] = []
    for index, (observed, censored, insufficient) in enumerate(counts):
        cell_counts = (observed, censored, insufficient)
        total = sum(cell_counts)
        status = (
            FollowUpBinStatus.EMPTY
            if total == 0
            else FollowUpBinStatus.SUPPRESSED_LOW_COUNT
            if any(0 < count < config.minimum_calibration_bin_count for count in cell_counts)
            else FollowUpBinStatus.OK
        )
        suppressed = status is FollowUpBinStatus.SUPPRESSED_LOW_COUNT
        bins.append(
            FollowUpCompletenessBin(
                lower_bound=edges[index],
                upper_bound=edges[index + 1],
                includes_upper_bound=index == len(counts) - 1,
                status=status,
                counts=None if suppressed else cell_counts,
                fractions=(
                    None
                    if suppressed or total == 0
                    else (observed / total, censored / total, insufficient / total)
                ),
            )
        )
    return RiskEvaluationWithFollowUp(
        observed_only_evaluation=evaluation,
        minimum_status_cell_count=config.minimum_calibration_bin_count,
        follow_up_bins=tuple(bins),
    )
