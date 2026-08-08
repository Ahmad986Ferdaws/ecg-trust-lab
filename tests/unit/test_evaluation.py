from __future__ import annotations

import json

import numpy as np
import pytest

from ecg_trust.evaluation import (
    CalibrationLeakageError,
    EvaluationValidationError,
    compute_multilabel_metrics,
    compute_selective_predictions,
    fit_temperature_scaling,
    fixed_bin_ece,
    optimize_thresholds,
    stable_sigmoid,
    validate_multilabel_arrays,
)
from ecg_trust.protocol import LABEL_ORDER


def _nondegenerate_targets() -> np.ndarray:
    return np.asarray(
        [
            [0, 1, 0, 1, 0],
            [0, 0, 1, 1, 1],
            [1, 1, 0, 0, 1],
            [1, 0, 1, 0, 0],
        ],
        dtype=np.int64,
    )


def test_canonical_shape_order_binary_and_probability_contract() -> None:
    targets = _nondegenerate_targets()
    probabilities = np.where(targets == 1, 0.9, 0.1)

    validated_targets, validated_probabilities = validate_multilabel_arrays(
        targets, probabilities
    )
    assert validated_targets.shape == (4, 5)
    assert validated_probabilities.shape == (4, 5)

    with pytest.raises(EvaluationValidationError, match="label_order"):
        validate_multilabel_arrays(
            targets, probabilities, label_order=("MI", "NORM", "STTC", "CD", "HYP")
        )
    with pytest.raises(EvaluationValidationError, match="shape"):
        validate_multilabel_arrays(targets[:, :4], probabilities[:, :4])
    with pytest.raises(EvaluationValidationError, match="binary"):
        validate_multilabel_arrays(targets.astype(float) + 0.25, probabilities)
    with pytest.raises(EvaluationValidationError, match=r"\[0, 1\]"):
        validate_multilabel_arrays(targets, probabilities + 0.2)
    with pytest.raises(EvaluationValidationError, match="finite"):
        invalid = probabilities.copy()
        invalid[0, 0] = np.nan
        validate_multilabel_arrays(targets, invalid)


def test_per_label_and_macro_metrics_for_perfect_ranking() -> None:
    targets = _nondegenerate_targets()
    probabilities = np.where(targets == 1, 0.9, 0.1)

    report = compute_multilabel_metrics(targets, probabilities, ece_bins=10)

    assert report.n_samples == 4
    assert report.label_order == LABEL_ORDER
    assert report.macro.roc_auc == pytest.approx(1.0)
    assert report.macro.average_precision == pytest.approx(1.0)
    assert report.macro.brier_score == pytest.approx(0.01)
    assert report.macro.ece == pytest.approx(0.1)
    assert report.macro.roc_auc_labels == 5
    assert report.macro.average_precision_labels == 5
    assert all(metric.roc_auc == pytest.approx(1.0) for metric in report.per_label)
    assert all(metric.average_precision == pytest.approx(1.0) for metric in report.per_label)
    json.dumps(report.to_dict(), allow_nan=False)


def test_degenerate_labels_are_explicit_and_excluded_from_discrimination_macro() -> None:
    targets = _nondegenerate_targets()
    targets[:, 0] = 0
    targets[:, 4] = 1
    probabilities = np.where(targets == 1, 0.8, 0.2)

    report = compute_multilabel_metrics(targets, probabilities)

    assert report.per_label[0].degenerate_reason == "no_positive_examples"
    assert report.per_label[0].roc_auc is None
    assert report.per_label[0].average_precision is None
    assert report.per_label[4].degenerate_reason == "no_negative_examples"
    assert report.per_label[4].roc_auc is None
    assert report.per_label[4].average_precision is None
    assert report.macro.roc_auc_labels == 3
    assert report.macro.average_precision_labels == 3
    assert report.macro.roc_auc == pytest.approx(1.0)


def test_fixed_bin_ece_has_fixed_width_boundary_semantics() -> None:
    targets = np.asarray([0, 0, 1, 1])
    probabilities = np.asarray([0.0, 0.25, 0.75, 1.0])

    assert fixed_bin_ece(targets, probabilities, n_bins=4) == pytest.approx(0.125)
    with pytest.raises(EvaluationValidationError, match="at least 2"):
        fixed_bin_ece(targets, probabilities, n_bins=1)


def test_thresholds_are_fitted_on_fold_9_only_and_mark_degenerate_labels() -> None:
    targets = np.tile(np.asarray([[0], [0], [1], [1]]), (1, 5))
    targets[:, 4] = 0
    probabilities = np.tile(np.asarray([[0.1], [0.2], [0.3], [0.4]]), (1, 5))

    result = optimize_thresholds(
        y_true=targets,
        probabilities=probabilities,
        calibration_fold_ids=np.full(4, 9),
    )

    assert result.source_folds == (9,)
    assert result.thresholds[:4] == pytest.approx((0.3, 0.3, 0.3, 0.3))
    assert result.thresholds[4] == pytest.approx(0.5)
    assert result.per_label[4].status == "no_positive_examples"
    assert result.per_label[4].objective_value is None
    assert result.macro_objective == pytest.approx(1.0)
    assert result.apply(probabilities).dtype == np.bool_
    json.dumps(result.to_dict(), allow_nan=False)

    with pytest.raises(CalibrationLeakageError, match="fold 9 only"):
        optimize_thresholds(
            y_true=targets,
            probabilities=probabilities,
            calibration_fold_ids=np.asarray([9, 9, 9, 10]),
        )


def test_temperature_scaling_reduces_calibration_nll_without_leakage() -> None:
    targets_1d = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int64)
    logits_1d = np.asarray([-8.0, -4.0, 4.0, 8.0, -6.0, 6.0])
    targets = np.tile(targets_1d[:, None], (1, 5))
    logits = np.tile(logits_1d[:, None], (1, 5))

    result = fit_temperature_scaling(
        logits=logits,
        y_true=targets,
        calibration_fold_ids=np.full(targets.shape[0], 9),
    )

    assert result.source_folds == (9,)
    assert result.fitted_labels == LABEL_ORDER
    assert result.excluded_degenerate_labels == ()
    assert result.temperature > 1.0
    assert result.nll_before is not None
    assert result.nll_after is not None
    assert result.nll_after < result.nll_before
    assert result.predict_proba(logits).shape == logits.shape
    assert np.all((result.predict_proba(logits) >= 0) & (result.predict_proba(logits) <= 1))
    json.dumps(result.to_dict(), allow_nan=False)

    with pytest.raises(CalibrationLeakageError, match="fold 9 only"):
        fit_temperature_scaling(
            logits=logits,
            y_true=targets,
            calibration_fold_ids=np.full(targets.shape[0], 8),
        )


def test_temperature_scaling_reports_all_degenerate_calibration_labels() -> None:
    targets = np.zeros((4, 5), dtype=np.int64)
    logits = np.zeros((4, 5), dtype=np.float64)

    result = fit_temperature_scaling(
        logits=logits,
        y_true=targets,
        calibration_fold_ids=np.full(4, 9),
    )

    assert result.temperature == 1.0
    assert result.status == "no_non_degenerate_labels"
    assert result.converged is False
    assert result.fitted_labels == ()
    assert result.excluded_degenerate_labels == LABEL_ORDER
    assert result.nll_before is None
    assert result.nll_after is None


def test_selective_prediction_returns_risk_coverage_and_abstention_outputs() -> None:
    targets = np.asarray(
        [
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
        ]
    )
    probabilities = np.asarray(
        [
            [0.01, 0.01, 0.01, 0.01, 0.01],
            [0.99, 0.99, 0.99, 0.99, 0.99],
            [0.49, 0.49, 0.49, 0.49, 0.49],
            [0.49, 0.49, 0.49, 0.49, 0.49],
        ]
    )

    result = compute_selective_predictions(
        targets,
        probabilities,
        thresholds=(0.5,) * 5,
        coverage_targets=(0.0, 0.5, 1.0),
    )

    zero, half, full = result.coverage_points
    assert zero.selected_count == 0
    assert zero.hamming_risk is None
    assert zero.abstained_indices == (0, 1, 2, 3)
    assert half.selected_indices == (0, 1)
    assert half.abstained_count == 2
    assert half.hamming_risk == 0.0
    assert half.exact_match_accuracy == 1.0
    assert full.hamming_risk == pytest.approx(0.25)
    assert full.achieved_coverage == 1.0
    assert result.uncertainty_method == "mean_normalized_binary_entropy"
    json.dumps(result.to_dict(), allow_nan=False)


def test_selective_prediction_accepts_explicit_uncertainty_and_discrete_coverage() -> None:
    targets = _nondegenerate_targets()
    probabilities = np.where(targets == 1, 0.8, 0.2)
    uncertainty = np.asarray([0.4, 0.1, 0.3, 0.2])

    result = compute_selective_predictions(
        targets,
        probabilities,
        thresholds=(0.5,) * 5,
        coverage_targets=(0.51,),
        uncertainty=uncertainty,
    )

    point = result.coverage_points[0]
    assert point.selected_count == 3
    assert point.achieved_coverage == 0.75
    assert point.selected_indices == (1, 3, 2)
    assert result.uncertainty_method == "provided"


def test_numerically_stable_sigmoid_handles_extreme_finite_logits() -> None:
    probabilities = stable_sigmoid(np.asarray([-1_000.0, 0.0, 1_000.0]))

    assert probabilities[0] == pytest.approx(0.0)
    assert probabilities[1] == pytest.approx(0.5)
    assert probabilities[2] == pytest.approx(1.0)
    with pytest.raises(EvaluationValidationError, match="finite"):
        stable_sigmoid(np.asarray([np.inf]))
