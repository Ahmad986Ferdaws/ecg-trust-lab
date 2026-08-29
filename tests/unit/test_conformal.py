from __future__ import annotations

import json

import numpy as np
import pytest

from ecg_trust.conformal import (
    BinaryDecision,
    BinaryPredictionSets,
    ConformalMetrics,
    ConformalValidationError,
    LabelwiseBinaryConformal,
    UncertaintyKind,
    evaluate_prediction_sets,
)


def _calibrator() -> LabelwiseBinaryConformal:
    probabilities = np.asarray(
        [
            [0.9, 0.4],
            [0.8, 0.3],
            [0.7, 0.2],
            [0.6, 0.1],
        ],
        dtype=np.float64,
    )
    targets = np.ones_like(probabilities, dtype=np.int8)
    return LabelwiseBinaryConformal.fit(
        probabilities,
        targets,
        label_names=("low_error", "high_error"),
        alpha=0.5,
    )


def test_finite_sample_quantile_is_labelwise_and_uses_corrected_rank() -> None:
    calibrator = _calibrator()

    assert calibrator.n_calibration_samples == 4
    assert calibrator.quantile_rank == 3
    assert calibrator.quantile_level == 0.75
    assert calibrator.thresholds == pytest.approx((0.3, 0.8))
    assert calibrator.label_names == ("low_error", "high_error")


def test_prediction_sets_emit_singletons_empty_and_both_as_three_states() -> None:
    prediction_sets = _calibrator().predict(
        [
            [0.8, 0.5],
            [0.2, 0.5],
            [0.5, 0.5],
        ]
    )

    assert prediction_sets.decisions == (
        (BinaryDecision.SUPPORTED, BinaryDecision.UNCERTAIN),
        (BinaryDecision.NOT_SUPPORTED, BinaryDecision.UNCERTAIN),
        (BinaryDecision.UNCERTAIN, BinaryDecision.UNCERTAIN),
    )
    assert prediction_sets.uncertainty_kinds == (
        (None, UncertaintyKind.BOTH),
        (None, UncertaintyKind.BOTH),
        (UncertaintyKind.EMPTY, UncertaintyKind.BOTH),
    )
    assert prediction_sets.include_not_supported == (
        (False, True),
        (True, True),
        (False, True),
    )
    assert prediction_sets.include_supported == (
        (True, True),
        (False, True),
        (False, True),
    )


def test_empirical_coverage_and_set_size_metrics_are_exact_and_serializable() -> None:
    prediction_sets = _calibrator().predict([[0.8, 0.5], [0.2, 0.5], [0.5, 0.5]])
    targets = np.asarray([[1, 0], [0, 1], [1, 1]], dtype=np.int8)

    metrics = evaluate_prediction_sets(prediction_sets, targets)

    assert metrics.n_samples == 3
    assert metrics.n_labels == 2
    assert metrics.labelwise_coverage == pytest.approx((2 / 3, 1.0))
    assert metrics.marginal_coverage == pytest.approx(5 / 6)
    assert metrics.joint_sample_coverage == pytest.approx(2 / 3)
    assert metrics.mean_set_size == pytest.approx(8 / 6)
    assert metrics.labelwise_mean_set_size == pytest.approx((2 / 3, 2.0))
    assert metrics.singleton_fraction == pytest.approx(2 / 6)
    assert metrics.empty_fraction == pytest.approx(1 / 6)
    assert metrics.both_fraction == pytest.approx(3 / 6)
    metrics_payload = json.loads(json.dumps(metrics.to_dict(), allow_nan=False))
    assert ConformalMetrics.from_dict(metrics_payload) == metrics


def test_small_calibration_set_conservatively_returns_both_values() -> None:
    calibrator = LabelwiseBinaryConformal.fit(
        [[0.99], [0.01]],
        [[1], [0]],
        label_names=("rhythm",),
        alpha=0.1,
    )

    assert calibrator.quantile_rank == 3
    assert calibrator.quantile_level == 1.0
    assert calibrator.thresholds == (1.0,)
    prediction_sets = calibrator.predict([[0.0], [0.5], [1.0]])
    assert all(
        decision == BinaryDecision.UNCERTAIN
        for row in prediction_sets.decisions
        for decision in row
    )
    assert all(
        kind == UncertaintyKind.BOTH for row in prediction_sets.uncertainty_kinds for kind in row
    )


def test_split_conformal_empirical_labelwise_coverage_on_exchangeable_data() -> None:
    generator = np.random.default_rng(20260824)
    calibration_probabilities = generator.uniform(0.0, 1.0, size=(2_000, 4))
    calibration_targets = generator.binomial(1, calibration_probabilities)
    test_probabilities = generator.uniform(0.0, 1.0, size=(20_000, 4))
    test_targets = generator.binomial(1, test_probabilities)
    calibrator = LabelwiseBinaryConformal.fit(
        calibration_probabilities,
        calibration_targets,
        label_names=("NORM", "MI", "STTC", "CD"),
        alpha=0.1,
    )

    metrics = evaluate_prediction_sets(calibrator.predict(test_probabilities), test_targets)

    assert min(metrics.labelwise_coverage) >= 0.88
    assert max(metrics.labelwise_coverage) <= 0.93
    assert metrics.marginal_coverage >= 0.89


def test_calibrator_and_predictions_round_trip_with_semantic_validation() -> None:
    calibrator = _calibrator()
    artifact_payload = json.loads(json.dumps(calibrator.to_dict(), allow_nan=False))
    restored = LabelwiseBinaryConformal.from_dict(artifact_payload)
    assert restored == calibrator

    prediction_sets = calibrator.predict([[0.8, 0.5], [0.2, 0.5]])
    prediction_payload = json.loads(json.dumps(prediction_sets.to_dict(), allow_nan=False))
    assert BinaryPredictionSets.from_dict(prediction_payload) == prediction_sets

    tampered = dict(prediction_payload)
    decisions = [list(row) for row in prediction_payload["decisions"]]  # type: ignore[union-attr]
    decisions[0][0] = "not_supported"
    tampered["decisions"] = decisions
    with pytest.raises(ConformalValidationError, match="contradict"):
        BinaryPredictionSets.from_dict(tampered)


@pytest.mark.parametrize(
    ("probabilities", "targets", "labels", "alpha", "match"),
    [
        ([0.5, 0.5], [[0, 1]], ("a", "b"), 0.1, "two-dimensional"),
        ([[0.5, np.nan]], [[0, 1]], ("a", "b"), 0.1, "finite"),
        ([[0.5, 1.1]], [[0, 1]], ("a", "b"), 0.1, r"\[0, 1\]"),
        ([[0.5, 0.5]], [[0, 0.5]], ("a", "b"), 0.1, "binary"),
        ([[0.5, 0.5]], [[0, 1]], ("a",), 0.1, "expected 1"),
        ([[0.5, 0.5]], [[0, 1]], ("a", "a"), 0.1, "unique"),
        ([[0.5, 0.5]], [[0, 1]], ("a", "b"), 0.0, "strictly"),
        ([], [], ("a",), 0.1, "two-dimensional"),
    ],
)
def test_fit_rejects_malformed_inputs(
    probabilities: object,
    targets: object,
    labels: tuple[str, ...],
    alpha: float,
    match: str,
) -> None:
    with pytest.raises(ConformalValidationError, match=match):
        LabelwiseBinaryConformal.fit(
            probabilities,
            targets,
            label_names=labels,
            alpha=alpha,
        )


def test_prediction_and_metrics_reject_misalignment() -> None:
    calibrator = _calibrator()
    with pytest.raises(ConformalValidationError, match="expected 2"):
        calibrator.predict([[0.5]])

    prediction_sets = calibrator.predict([[0.5, 0.5]])
    with pytest.raises(ConformalValidationError, match="does not match"):
        evaluate_prediction_sets(prediction_sets, [[1]])

    with pytest.raises(ConformalValidationError, match="booleans"):
        BinaryPredictionSets.from_masks(
            label_names=("a",),
            include_not_supported=[[1]],
            include_supported=[[0]],
        )


def test_serialized_calibrator_rejects_invalid_quantile_and_thresholds() -> None:
    payload = _calibrator().to_dict()
    bad_rank = dict(payload)
    bad_rank["quantile_rank"] = 2
    with pytest.raises(ConformalValidationError, match="quantile_rank"):
        LabelwiseBinaryConformal.from_dict(bad_rank)

    bad_thresholds = dict(payload)
    bad_thresholds["thresholds"] = [0.3]
    with pytest.raises(ConformalValidationError, match="2 values"):
        LabelwiseBinaryConformal.from_dict(bad_thresholds)
