"""Weak calibration checks from clinical prediction modelling.

Binned calibration errors summarize the whole reliability curve in one number.
Clinical reporting guidance (TRIPOD) and the calibration hierarchy of Van
Calster et al. ("A calibration hierarchy for risk models was defined: from
utopia to empirical data", J Clin Epidemiol 74, 2016) ask for two simpler,
interpretable quantities per outcome instead:

* calibration-in-the-large: the intercept ``a`` in ``logit P(y) = a + logit(p)``
  (0 means the average risk is right; negative means risks are too high), with
  the observed/expected ratio as a companion;
* calibration slope: ``b`` in ``logit P(y) = a + b logit(p)`` (1 is ideal; below
  1 means predictions are too extreme, above 1 too modest).

Both are fitted by Newton's method on ``logit(p)`` without regularization and
report whether they converged; perfectly separable data has no finite slope.
They are descriptive and never used to change predictions here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ecg_trust.evaluation import validate_multilabel_arrays
from ecg_trust.protocol import LABEL_ORDER

FloatArray = NDArray[np.float64]
_CLIP = 1e-12


class CalibrationHierarchyError(ValueError):
    """Raised when a label cannot support weak calibration estimates."""


@dataclass(frozen=True, slots=True)
class WeakCalibration:
    label: str
    observed_expected_ratio: float
    calibration_in_the_large: float
    calibration_slope: float | None
    slope_intercept: float | None
    converged: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "observed_expected_ratio": self.observed_expected_ratio,
            "calibration_in_the_large": self.calibration_in_the_large,
            "calibration_slope": self.calibration_slope,
            "slope_intercept": self.slope_intercept,
            "converged": self.converged,
        }


def _sigmoid(values: FloatArray) -> FloatArray:
    return np.asarray(0.5 * (1.0 + np.tanh(0.5 * values)), dtype=np.float64)


def _intercept_with_offset(logits: FloatArray, outcomes: FloatArray) -> float:
    intercept = 0.0
    for _ in range(100):
        fitted = _sigmoid(intercept + logits)
        gradient = float(np.sum(outcomes - fitted))
        curvature = float(np.sum(fitted * (1.0 - fitted)))
        if curvature <= 0.0:
            break
        step = gradient / curvature
        intercept += max(min(step, 5.0), -5.0)
        if abs(step) < 1e-10:
            break
    return intercept


def _logistic_slope(
    logits: FloatArray, outcomes: FloatArray, max_iterations: int
) -> tuple[float, float, bool]:
    design = np.column_stack((np.ones_like(logits), logits))
    coefficients = np.array([0.0, 1.0])

    def log_likelihood(beta: FloatArray) -> float:
        linear = design @ beta
        return float(np.sum(outcomes * linear - np.logaddexp(0.0, linear)))

    current = log_likelihood(coefficients)
    for _ in range(max_iterations):
        fitted = _sigmoid(design @ coefficients)
        weights = fitted * (1.0 - fitted)
        gradient = design.T @ (outcomes - fitted)
        hessian = design.T @ (design * weights[:, None])
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            return float(coefficients[0]), float(coefficients[1]), False
        scale = 1.0
        while scale > 1e-8:
            candidate = coefficients + scale * step
            value = log_likelihood(candidate)
            if value >= current - 1e-12:
                break
            scale *= 0.5
        coefficients = coefficients + scale * step
        improvement = log_likelihood(coefficients) - current
        current += improvement
        if not np.isfinite(coefficients).all() or abs(float(coefficients[1])) > 1e6:
            return float(coefficients[0]), float(coefficients[1]), False
        if float(np.max(np.abs(scale * step))) < 1e-9:
            return float(coefficients[0]), float(coefficients[1]), True
    return float(coefficients[0]), float(coefficients[1]), False


def weak_calibration(
    y_true: ArrayLike,
    probabilities: ArrayLike,
    *,
    max_iterations: int = 100,
    label_order: Sequence[str] = LABEL_ORDER,
) -> tuple[WeakCalibration, ...]:
    """Estimate calibration-in-the-large and slope for every canonical label."""

    if (
        isinstance(max_iterations, bool)
        or not isinstance(max_iterations, int)
        or max_iterations < 1
    ):
        raise CalibrationHierarchyError("max_iterations must be a positive integer")
    targets, scores = validate_multilabel_arrays(y_true, probabilities, label_order=label_order)
    results: list[WeakCalibration] = []
    for index, label in enumerate(label_order):
        outcomes = targets[:, index].astype(np.float64)
        positives = float(outcomes.sum())
        if positives == 0.0 or positives == outcomes.shape[0]:
            raise CalibrationHierarchyError(f"label {label!r} needs both outcomes")
        clipped = np.clip(scores[:, index], _CLIP, 1.0 - _CLIP)
        logits = np.log(clipped) - np.log1p(-clipped)
        expected = float(clipped.mean())
        intercept, slope, converged = _logistic_slope(logits, outcomes, max_iterations)
        results.append(
            WeakCalibration(
                label=label,
                observed_expected_ratio=float(outcomes.mean()) / expected,
                calibration_in_the_large=_intercept_with_offset(logits, outcomes),
                calibration_slope=slope if converged else None,
                slope_intercept=intercept if converged else None,
                converged=converged,
            )
        )
    return tuple(results)


__all__ = ["CalibrationHierarchyError", "WeakCalibration", "weak_calibration"]
