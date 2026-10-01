from __future__ import annotations

import math

import numpy as np
import pytest

from ecg_trust.longitudinal import (
    CalibrationBinStatus,
    RiskEvaluationConfig,
    RiskObservation,
    TargetStatus,
    TimeDependentRiskEvaluation,
    evaluate_time_dependent_binary_risk,
)


def _evaluate(
    risks: list[float], outcomes: list[int], bin_count: int
) -> TimeDependentRiskEvaluation:
    rows = [
        RiskObservation.create(
            predicted_risk=risk, outcome=outcome, target_status=TargetStatus.OBSERVED
        )
        for risk, outcome in zip(risks, outcomes, strict=True)
    ]
    config = RiskEvaluationConfig.create(
        horizon_days=90,
        minimum_evaluable_count=2,
        calibration_bin_count=bin_count,
        minimum_calibration_bin_count=2,
    )
    return evaluate_time_dependent_binary_risk("future_event", rows, config)


def test_exact_point_fifty_eight_belongs_to_bin_starting_at_point_fifty_eight() -> None:
    assert 0.58 * 100 < 58  # The multiply-and-truncate implementation rounded left.
    report = _evaluate([0.58, 0.58], [0, 1], 100)
    assert report.calibration_bins[57].count == 0
    assert report.calibration_bins[58].count == 2
    assert report.calibration_bins[58].lower_bound == 0.58
    assert report.calibration_bins[58].mean_predicted_risk == 0.58


@pytest.mark.parametrize("bin_count", range(2, 101))
def test_every_configured_boundary_and_nextafter_neighbors_conserve_rows(bin_count: int) -> None:
    entries = [(0.0, 0), (float(np.nextafter(0.0, 1.0)), 0)]
    for index in range(1, bin_count):
        edge = index / bin_count
        entries.extend(
            [
                (float(np.nextafter(edge, 0.0)), index - 1),
                (edge, index),
                (float(np.nextafter(edge, 1.0)), index),
            ]
        )
    entries.extend([(float(np.nextafter(1.0, 0.0)), bin_count - 1), (1.0, bin_count - 1)])
    risks = [risk for risk, _ in entries for _ in range(2)]
    outcomes = [outcome for _ in entries for outcome in (0, 1)]
    report = _evaluate(risks, outcomes, bin_count)
    for index, cell in enumerate(report.calibration_bins):
        expected_risks = [
            risk for risk, expected_index in entries if expected_index == index for _ in range(2)
        ]
        assert cell.status is CalibrationBinStatus.OK
        assert cell.count == len(expected_risks)
        assert cell.mean_predicted_risk == math.fsum(expected_risks) / len(expected_risks)
        assert cell.observed_event_rate == 0.5
        assert cell.lower_bound == index / bin_count
        assert cell.upper_bound == (index + 1) / bin_count
        assert cell.includes_upper_bound is (index == bin_count - 1)
    assert sum(cell.count or 0 for cell in report.calibration_bins) == len(risks)
    assert math.fsum(
        (cell.count or 0) * (cell.mean_predicted_risk or 0.0) for cell in report.calibration_bins
    ) == pytest.approx(math.fsum(risks), rel=1e-14)
    assert sum(
        (cell.count or 0) * (cell.observed_event_rate or 0.0) for cell in report.calibration_bins
    ) == sum(outcomes)


@pytest.mark.parametrize("bin_count", [2, 3, 10, 50, 100])
def test_ordinary_interior_values_keep_previous_assignments_and_metrics(bin_count: int) -> None:
    risks = [
        (index + fraction) / bin_count for index in range(bin_count) for fraction in (0.25, 0.75)
    ]
    outcomes = [index % 2 for index in range(len(risks))]
    report = _evaluate(risks, outcomes, bin_count)
    for index, cell in enumerate(report.calibration_bins):
        previous_rows = [
            (risk, outcome)
            for risk, outcome in zip(risks, outcomes, strict=True)
            if min(int(risk * bin_count), bin_count - 1) == index
        ]
        assert cell.count == len(previous_rows) == 2
        assert cell.mean_predicted_risk == math.fsum(risk for risk, _ in previous_rows) / 2
        assert cell.observed_event_rate == sum(outcome for _, outcome in previous_rows) / 2
