from __future__ import annotations

import json
import math

import pytest

from ecg_trust.longitudinal import (
    EvaluationStatus,
    FollowUpBinStatus,
    LongitudinalError,
    RiskEvaluationConfig,
    RiskObservation,
    TargetStatus,
    evaluate_binary_risk_with_follow_up,
    evaluate_time_dependent_binary_risk,
)


def _config(bins: int = 2, minimum: int = 2) -> RiskEvaluationConfig:
    return RiskEvaluationConfig.create(
        horizon_days=90,
        minimum_evaluable_count=2,
        calibration_bin_count=bins,
        minimum_calibration_bin_count=minimum,
    )


def _row(
    risk: float, status: TargetStatus = TargetStatus.OBSERVED, outcome: int = 0
) -> RiskObservation:
    return RiskObservation.create(
        predicted_risk=risk,
        target_status=status,
        outcome=outcome if status is TargetStatus.OBSERVED else None,
    )


def _observed() -> list[RiskObservation]:
    return [_row(0.1), _row(0.2), _row(0.8, outcome=1), _row(0.9, outcome=1)]


def test_new_report_exposes_selection_pattern_hidden_by_identical_legacy_reports() -> None:
    observed = _observed()
    low_exclusions = [
        _row(0.2, TargetStatus.RIGHT_CENSORED),
        _row(0.3, TargetStatus.RIGHT_CENSORED),
        _row(0.1, TargetStatus.INSUFFICIENT_FOLLOW_UP),
        _row(0.4, TargetStatus.INSUFFICIENT_FOLLOW_UP),
    ]
    high_exclusions = [
        _row(0.7, TargetStatus.RIGHT_CENSORED),
        _row(0.8, TargetStatus.RIGHT_CENSORED),
        _row(0.6, TargetStatus.INSUFFICIENT_FOLLOW_UP),
        _row(0.9, TargetStatus.INSUFFICIENT_FOLLOW_UP),
    ]
    low = evaluate_binary_risk_with_follow_up("future_event", observed + low_exclusions, _config())
    high = evaluate_binary_risk_with_follow_up(
        "future_event", observed + high_exclusions, _config()
    )
    assert (
        low.observed_only_evaluation.to_public_dict()
        == high.observed_only_evaluation.to_public_dict()
    )
    assert low.follow_up_bins != high.follow_up_bins
    assert low.follow_up_bins[0].counts == (2, 2, 2)
    assert low.follow_up_bins[0].fractions == (1 / 3, 1 / 3, 1 / 3)
    assert high.follow_up_bins[0].counts == (2, 0, 0)
    assert high.follow_up_bins[0].fractions == (1.0, 0.0, 0.0)
    assert low.to_public_dict()["observed_only_evaluation"] == (
        evaluate_time_dependent_binary_risk(
            "future_event", observed + low_exclusions, _config()
        ).to_public_dict()
    )


def test_excluded_rows_do_not_change_observed_metrics_or_calibration() -> None:
    observed = _observed()
    baseline = evaluate_time_dependent_binary_risk("future_event", observed, _config())
    excluded = [
        _row(1.0, TargetStatus.RIGHT_CENSORED),
        _row(0.0, TargetStatus.RIGHT_CENSORED),
        _row(1.0, TargetStatus.INSUFFICIENT_FOLLOW_UP),
        _row(0.0, TargetStatus.INSUFFICIENT_FOLLOW_UP),
    ]
    report = evaluate_binary_risk_with_follow_up(
        "future_event", iter(observed + excluded), _config()
    )
    result = report.observed_only_evaluation
    assert (
        result.auroc,
        result.average_precision,
        result.brier_score,
        result.calibration_bins,
    ) == (
        baseline.auroc,
        baseline.average_precision,
        baseline.brier_score,
        baseline.calibration_bins,
    )
    reversed_outcomes = [
        _row(row.predicted_risk, outcome=1 - int(row.outcome or 0)) for row in observed
    ]
    changed = evaluate_binary_risk_with_follow_up(
        "future_event", reversed_outcomes + excluded, _config()
    )
    assert changed.follow_up_bins == report.follow_up_bins
    assert changed.observed_only_evaluation.auroc != result.auroc
    with pytest.raises(LongitudinalError, match="outcome=null"):
        RiskObservation.create(
            predicted_risk=0.5, outcome=0, target_status=TargetStatus.RIGHT_CENSORED
        )


@pytest.mark.parametrize(("bin_count", "boundary"), [(10, 0.1), (100, 0.58)])
def test_bin_edges_assign_zero_one_and_exact_boundaries(bin_count: int, boundary: float) -> None:
    risks = (0.0, math.nextafter(boundary, 0.0), boundary, math.nextafter(boundary, 1.0), 1.0)
    rows = [_row(risk, outcome=outcome) for risk in risks for outcome in (0, 1)]
    report = evaluate_binary_risk_with_follow_up("future_event", rows, _config(bin_count))
    for index, cell in enumerate(report.follow_up_bins):
        expected = sum(
            2
            for risk in risks
            if index / bin_count <= risk < (index + 1) / bin_count
            or (index == bin_count - 1 and risk == 1.0)
        )
        assert cell.counts == (expected, 0, 0)
        assert cell.includes_upper_bound is (index == bin_count - 1)
    assert sum(sum(cell.counts or ()) for cell in report.follow_up_bins) == len(rows)


def test_every_status_conserves_totals_and_proportions_in_disclosable_bins() -> None:
    rows = [
        _row((index + 0.5) / 7, status)
        for index in range(7)
        for status in TargetStatus
        for _ in range(2 + index)
    ]
    report = evaluate_binary_risk_with_follow_up("future_event", rows, _config(7))
    for cell in report.follow_up_bins:
        assert cell.status is FollowUpBinStatus.OK
        assert cell.fractions is not None
        assert sum(cell.fractions) == pytest.approx(1.0)
    actual = [
        sum((cell.counts or (0, 0, 0))[index] for cell in report.follow_up_bins)
        for index in range(3)
    ]
    assert actual == [sum(row.target_status is status for row in rows) for status in TargetStatus]
    assert sum(actual) == report.observed_only_evaluation.total_prediction_count == len(rows)


def test_empty_and_status_only_bins_are_explicit() -> None:
    empty = evaluate_binary_risk_with_follow_up("future_event", [], _config())
    assert all(cell.status is FollowUpBinStatus.EMPTY for cell in empty.follow_up_bins)
    assert all(cell.counts == (0, 0, 0) and cell.fractions is None for cell in empty.follow_up_bins)
    rows = [_row(0.1, TargetStatus.RIGHT_CENSORED)] * 2
    rows += [_row(0.9, TargetStatus.INSUFFICIENT_FOLLOW_UP)] * 2
    report = evaluate_binary_risk_with_follow_up("future_event", rows, _config(3))
    assert report.observed_only_evaluation.status is EvaluationStatus.INSUFFICIENT_EVIDENCE
    assert report.follow_up_bins[0].fractions == (0.0, 1.0, 0.0)
    assert report.follow_up_bins[1].status is FollowUpBinStatus.EMPTY
    assert report.follow_up_bins[2].fractions == (0.0, 0.0, 1.0)
    json.dumps(report.to_public_dict(), allow_nan=False)


@pytest.mark.parametrize("minimum", [2, 5])
def test_small_bins_and_rare_status_cells_are_suppressed(minimum: int) -> None:
    rows = [_row(0.1)] * minimum + [_row(0.2, TargetStatus.RIGHT_CENSORED)]
    rows += [_row(0.9, TargetStatus.INSUFFICIENT_FOLLOW_UP)]
    report = evaluate_binary_risk_with_follow_up("future_event", rows, _config(minimum=minimum))
    for cell in report.follow_up_bins:
        assert cell.status is FollowUpBinStatus.SUPPRESSED_LOW_COUNT
        assert cell.counts is None and cell.fractions is None
        public = cell.to_public_dict()
        assert public["total_prediction_count"] is None
        assert public["target_status_counts"] is None
        assert public["target_status_fractions"] is None


def test_report_stores_aggregates_and_does_not_retain_input_sequence() -> None:
    rows = _observed()
    report = evaluate_binary_risk_with_follow_up("future_event", rows, _config())
    before = json.dumps(report.to_public_dict(), sort_keys=True)
    rows.clear()
    assert json.dumps(report.to_public_dict(), sort_keys=True) == before
    for private_field in ("patient_id", "source_id", "predicted_risk", "waveform", "outcome"):
        assert f'"{private_field}":' not in before
    assert "observed_targets_only_no_ipcw" in before
    assert "target_availability_only_no_ipcw_or_selection_correction" in before
