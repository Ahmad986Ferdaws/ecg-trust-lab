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


def test_extreme_probabilities_stay_finite() -> None:
    outcomes, probabilities = _simulate(2_000, scale=1.0, shift=0.0, seed=3)
    probabilities[:10] = 0.0
    probabilities[10:20] = 1.0

    for item in weak_calibration(outcomes, probabilities):
        assert np.isfinite(item.calibration_in_the_large)
        assert np.isfinite(item.observed_expected_ratio)


def test_degenerate_labels_and_bad_controls_are_rejected() -> None:
    with pytest.raises(CalibrationHierarchyError, match="both outcomes"):
        weak_calibration(np.zeros((5, 5), dtype=np.int64), np.full((5, 5), 0.2))
    for bad in (0, True, 1.5):
        with pytest.raises(CalibrationHierarchyError, match="max_iterations"):
            weak_calibration(np.eye(5, dtype=np.int64), np.full((5, 5), 0.2), max_iterations=bad)  # type: ignore[arg-type]
    with pytest.raises(EvaluationValidationError):
        weak_calibration(np.eye(5, dtype=np.int64), np.full((5, 5), -0.1))
