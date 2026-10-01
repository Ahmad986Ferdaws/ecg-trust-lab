from __future__ import annotations

import numpy as np
import pytest

from ecg_trust.calibration_hierarchy import CalibrationHierarchyError, weak_calibration
from ecg_trust.evaluation import EvaluationValidationError
from ecg_trust.protocol import LABEL_ORDER


def _simulate(
    count: int, *, scale: float, shift: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    true_logit = rng.normal(-0.5, 1.5, size=(count, 5))
    outcomes = (rng.uniform(size=(count, 5)) < 1 / (1 + np.exp(-true_logit))).astype(np.int64)
    reported = 1 / (1 + np.exp(-(scale * true_logit + shift)))
    return outcomes, reported


def test_well_calibrated_predictions_have_ideal_intercept_and_slope() -> None:
    outcomes, probabilities = _simulate(20_000, scale=1.0, shift=0.0, seed=1)

    results = weak_calibration(outcomes, probabilities)

    assert [item.label for item in results] == list(LABEL_ORDER)
    for item in results:
        assert item.converged
        assert item.calibration_in_the_large == pytest.approx(0.0, abs=0.06)
        assert item.calibration_slope == pytest.approx(1.0, abs=0.06)
        assert item.observed_expected_ratio == pytest.approx(1.0, abs=0.05)


def test_overconfident_and_shifted_predictions_are_detected() -> None:
    outcomes, overconfident = _simulate(20_000, scale=2.0, shift=0.0, seed=2)
    _, too_high = _simulate(20_000, scale=1.0, shift=1.0, seed=2)

    extreme = weak_calibration(outcomes, overconfident)[0]
    shifted = weak_calibration(outcomes, too_high)[0]

    assert extreme.calibration_slope == pytest.approx(0.5, abs=0.04)
    assert shifted.calibration_in_the_large == pytest.approx(-1.0, abs=0.08)
    assert shifted.observed_expected_ratio < 1.0
    assert shifted.to_dict()["converged"] is True


def test_separable_outcomes_report_no_finite_slope() -> None:
    probabilities = np.tile(np.linspace(0.05, 0.95, 40)[:, None], (1, 5))
    outcomes = (probabilities > 0.5).astype(np.int64)

    item = weak_calibration(outcomes, probabilities)[0]

    assert not item.converged
    assert item.calibration_slope is None
    assert item.slope_intercept is None
    assert np.isfinite(item.calibration_in_the_large)


def test_overlapping_classes_with_extreme_logits_still_converge() -> None:
    logits = np.array([5.0, 5.0, 27.63, 1.0])
    probabilities = np.tile((1.0 / (1.0 + np.exp(-logits)))[:, None], (1, 5))
    outcomes = np.tile(np.array([[1], [0], [0], [0]]), (1, 5))

    item = weak_calibration(outcomes, probabilities)[0]

    assert item.converged
    assert item.calibration_slope is not None
    assert np.isfinite(item.calibration_slope)


def test_reversed_association_with_saturated_logits_still_converges() -> None:
    logits = np.array([27.631, -5.0, -5.0, 27.631, 0.0])
    probabilities = np.tile((1.0 / (1.0 + np.exp(-logits)))[:, None], (1, 5))
    outcomes = np.tile(np.array([[0], [1], [0], [0], [1]]), (1, 5))

    item = weak_calibration(outcomes, probabilities)[0]

    assert item.converged
    assert item.calibration_slope is not None
    assert item.calibration_slope < 0.0


def test_clustered_logits_far_from_one_half_still_converge() -> None:
    rng = np.random.default_rng(11)
    logits = 10.0 + rng.normal(0.0, 1e-3, size=200)
    outcome_probability = 1.0 / (1.0 + np.exp(-(logits - 10.0) * 2_000.0))
    outcomes_column = (rng.uniform(size=200) < outcome_probability).astype(np.int64)
    probabilities = np.tile((1.0 / (1.0 + np.exp(-logits)))[:, None], (1, 5))
    outcomes = np.tile(outcomes_column[:, None], (1, 5))

    item = weak_calibration(outcomes, probabilities)[0]

    assert item.converged
    assert item.calibration_slope is not None
    assert item.calibration_slope > 0.0


def test_tiny_logit_spread_is_not_mistaken_for_a_zero_slope() -> None:
    from ecg_trust.calibration_hierarchy import _logistic_slope

    logits = np.array([-2e-10, -1e-10, 0.0, 0.0, 1e-10, 2e-10])
    outcomes = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])

    intercept, slope, converged = _logistic_slope(logits, outcomes, 100)
    standardized = (logits - logits.mean()) / logits.std()
    _, reference, reference_converged = _logistic_slope(standardized, outcomes, 100)

    assert converged and reference_converged
    assert slope == pytest.approx(reference / logits.std(), rel=1e-6)
    assert abs(slope) > 1e8
    assert np.isfinite(intercept)


def test_tiny_interior_probabilities_keep_their_spread() -> None:
    probabilities = np.tile(np.array([1e-15, 2e-15, 3e-15, 4e-15])[:, None], (1, 5))
    outcomes = np.tile(np.array([[0], [1], [0], [1]]), (1, 5))

    item = weak_calibration(outcomes, probabilities)[0]

    assert item.converged
    assert item.calibration_slope is not None


def test_large_but_finite_standardized_slope_is_reported() -> None:
    logits = np.concatenate(
        (np.full(10, -1.0), np.full(10, 1.0), np.full(4, -5e-7), np.full(4, 5e-7))
    )
    outcomes_column = np.concatenate(
        (np.zeros(10), np.ones(10), [1, 0, 0, 0], [1, 1, 1, 0])
    ).astype(np.int64)
    probabilities = np.tile((1.0 / (1.0 + np.exp(-logits)))[:, None], (1, 5))
    outcomes = np.tile(outcomes_column[:, None], (1, 5))

    item = weak_calibration(outcomes, probabilities)[0]

    assert item.converged
    assert item.calibration_slope is not None
    assert item.calibration_slope > 1e5


def test_reported_convergence_always_satisfies_the_score_equations() -> None:
    logits = np.array([7.590326, 7.590118, 7.590254, -700.0, 7.590194])
    outcomes_column = np.array([1, 0, 0, 0, 1])
    probabilities = np.tile((1.0 / (1.0 + np.exp(-logits)))[:, None], (1, 5))
    outcomes = np.tile(outcomes_column[:, None], (1, 5))

    item = weak_calibration(outcomes, probabilities)[0]

    if item.converged:
        assert item.calibration_slope is not None
        assert item.slope_intercept is not None
        with np.errstate(over="ignore"):
            fitted = 1.0 / (1.0 + np.exp(-(item.slope_intercept + item.calibration_slope * logits)))
        assert fitted.mean() == pytest.approx(outcomes_column.mean(), abs=1e-6)
    else:
        assert item.calibration_slope is None


def test_quasi_separation_with_a_shared_boundary_value_has_no_finite_slope() -> None:
    probabilities = np.tile(np.array([0.2, 0.4, 0.5, 0.5, 0.7])[:, None], (1, 5))
    outcomes = np.tile(np.array([[0], [0], [0], [1], [1]]), (1, 5))

    assert not weak_calibration(outcomes, probabilities)[0].converged


def test_polarized_predictions_get_the_exact_intercept_root() -> None:
    column = np.array([0.001, 0.001] + [0.999] * 8)
    outcomes_column = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    probabilities = np.tile(column[:, None], (1, 5))
    outcomes = np.tile(outcomes_column[:, None], (1, 5))

    item = weak_calibration(outcomes, probabilities)[0]
    logits = np.log(column) - np.log1p(-column)
    fitted = 1.0 / (1.0 + np.exp(-(item.calibration_in_the_large + logits)))

    assert item.calibration_in_the_large == pytest.approx(-8.005, abs=0.01)
    assert fitted.mean() == pytest.approx(0.2, abs=1e-9)


def test_extreme_probabilities_stay_finite() -> None:
    outcomes, probabilities = _simulate(2_000, scale=1.0, shift=0.0, seed=3)
    probabilities[:10] = 0.0
    probabilities[10:20] = 1.0

    for item in weak_calibration(outcomes, probabilities):
        assert np.isfinite(item.calibration_in_the_large)
        assert np.isfinite(item.observed_expected_ratio)


def test_all_zero_predictions_leave_the_ratio_undefined() -> None:
    outcomes = np.tile(np.array([[1], [0], [1], [0]]), (1, 5))

    item = weak_calibration(outcomes, np.zeros((4, 5)))[0]

    assert item.observed_expected_ratio is None
    assert item.to_dict()["observed_expected_ratio"] is None


def test_subnormal_predictions_keep_a_defined_ratio() -> None:
    probabilities = np.zeros((4, 5))
    probabilities[0] = np.nextafter(0.0, 1.0)
    outcomes = np.tile(np.array([[1], [0], [1], [0]]), (1, 5))

    item = weak_calibration(outcomes, probabilities)[0]

    assert item.observed_expected_ratio is not None
    assert item.observed_expected_ratio > 1e300


def test_iteration_budget_counts_the_final_update() -> None:
    from ecg_trust.calibration_hierarchy import _logistic_slope

    logits = np.array([-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0])
    outcomes = np.array([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0])

    one = _logistic_slope(logits, outcomes, 1)
    many = _logistic_slope(logits, outcomes, 50)

    assert many[2]
    if one[2]:
        assert one[1] == pytest.approx(many[1], rel=1e-6)


def test_weakly_identified_slope_is_not_reported_with_a_wrong_value() -> None:
    from ecg_trust.calibration_hierarchy import _logistic_slope

    logits = np.concatenate(
        (np.full(10, -1.0), np.full(10, 1.0), np.full(4, -1e-12), np.full(4, 1e-12))
    )
    outcomes = np.concatenate((np.zeros(10), np.ones(10), [1, 0, 0, 0], [1, 1, 1, 0]))

    _, slope, converged = _logistic_slope(logits, outcomes, 100)

    if converged:
        assert slope == pytest.approx(np.log(3.0) / 1e-12, rel=1e-3)
    else:
        assert slope > 1e3


def test_non_positive_definite_curvature_is_a_numerical_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ecg_trust import calibration_hierarchy as module

    def fail(_: object) -> object:
        raise np.linalg.LinAlgError("not positive definite")

    monkeypatch.setattr(module.np.linalg, "cholesky", fail)
    logits = np.array([-1.0, -1.0, 1.0, 1.0, 0.0])
    outcomes = np.array([0.0, 1.0, 1.0, 0.0, 1.0])

    assert module._logistic_slope(logits, outcomes, 10)[2] is False


def test_degenerate_labels_and_bad_controls_are_rejected() -> None:
    with pytest.raises(CalibrationHierarchyError, match="both outcomes"):
        weak_calibration(np.zeros((5, 5), dtype=np.int64), np.full((5, 5), 0.2))
    for bad in (0, True, 1.5):
        with pytest.raises(CalibrationHierarchyError, match="max_iterations"):
            weak_calibration(np.eye(5, dtype=np.int64), np.full((5, 5), 0.2), max_iterations=bad)  # type: ignore[arg-type]
    with pytest.raises(EvaluationValidationError):
        weak_calibration(np.eye(5, dtype=np.int64), np.full((5, 5), -0.1))
