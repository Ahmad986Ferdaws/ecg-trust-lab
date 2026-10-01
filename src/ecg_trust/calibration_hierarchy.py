"""Weak calibration checks from clinical prediction modelling.

Binned calibration errors summarize the whole reliability curve in one number.
Clinical reporting guidance (TRIPOD) and the calibration hierarchy of Van
Calster et al. ("A calibration hierarchy for risk models was defined: from
utopia to empirical data", J Clin Epidemiol 74, 2016) ask for two simpler,
interpretable quantities per outcome instead:

* calibration-in-the-large: the intercept ``a`` in ``logit P(y) = a + logit(p)``
  (0 means the average risk is right; negative means risks are too high), with
  the observed/expected ratio as a companion (``None`` when every prediction
  is exactly zero, because the ratio is then undefined);
* calibration slope: ``b`` in ``logit P(y) = a + b logit(p)`` (1 is ideal; below
  1 means predictions are too extreme, above 1 too modest).

Both are fitted by Newton's method on ``logit(p)`` without regularization and
report whether they converged; perfectly separable data has no finite slope.
They are descriptive and never used to change predictions here.
"""

from __future__ import annotations

import math
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
    observed_expected_ratio: float | None
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
    # Sign-branched so neither tail cancels to exactly 0 or 1 prematurely.
    result = np.empty_like(values, dtype=np.float64)
    positive = values >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def _finite_logits(probabilities: FloatArray) -> FloatArray:
    """Exact logits for interior probabilities; finite stand-ins only for 0 and 1.

    Exact endpoints map just beyond the most extreme interior logit (and at
    least to ``logit(1e-12)``), so representable small probabilities such as
    1e-15 keep their real ordering and spacing.
    """

    interior = (probabilities > 0.0) & (probabilities < 1.0)
    logits = np.empty_like(probabilities, dtype=np.float64)
    logits[interior] = np.log(probabilities[interior]) - np.log1p(-probabilities[interior])
    floor = math.log(_CLIP) - math.log1p(-_CLIP)
    lowest = min(floor, float(logits[interior].min()) - 1.0) if interior.any() else floor
    highest = max(-floor, float(logits[interior].max()) + 1.0) if interior.any() else -floor
    logits[probabilities <= 0.0] = lowest
    logits[probabilities >= 1.0] = highest
    return logits


def _intercept_with_offset(logits: FloatArray, outcomes: FloatArray) -> float:
    """Root of ``sum(y - sigmoid(a + logit))`` in ``a`` by bracketed bisection.

    The score is strictly decreasing in ``a``, so a sign change brackets the
    unique root whenever both outcomes occur; bisection cannot cycle.
    """

    def score(intercept: float) -> float:
        return float(np.sum(outcomes - _sigmoid(intercept + logits)))

    low, high = -1.0, 1.0
    while score(low) <= 0.0:
        low *= 2.0
        if low < -1e6:
            raise CalibrationHierarchyError("calibration intercept could not be bracketed")
    while score(high) >= 0.0:
        high *= 2.0
        if high > 1e6:
            raise CalibrationHierarchyError("calibration intercept could not be bracketed")
    for _ in range(200):
        middle = 0.5 * (low + high)
        if middle in (low, high):
            break
        if score(middle) > 0.0:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def _logistic_slope(
    logits: FloatArray, outcomes: FloatArray, max_iterations: int
) -> tuple[float, float, bool]:
    # Fit on standardized logits so conditioning and convergence tolerances do
    # not depend on where or how tightly predictions cluster, then map the
    # coefficients back to the raw logit scale.
    centre = float(logits.mean())
    spread = float(logits.std())
    if spread == 0.0:
        return 0.0, 0.0, False
    intercept, slope, converged = _logistic_slope_centered(
        (logits - centre) / spread, outcomes, max_iterations
    )
    raw_slope = slope / spread
    return intercept - raw_slope * centre, raw_slope, converged


def _logistic_slope_centered(
    logits: FloatArray, outcomes: FloatArray, max_iterations: int
) -> tuple[float, float, bool]:
    positive_logits = logits[outcomes == 1.0]
    negative_logits = logits[outcomes == 0.0]
    # With one covariate, a finite maximum-likelihood slope exists only when the
    # two outcome classes overlap; (quasi-)complete separation drives it to
    # infinity, and saturated fits would otherwise look converged.
    if float(negative_logits.max()) <= float(positive_logits.min()) or float(
        positive_logits.max()
    ) <= float(negative_logits.min()):
        return 0.0, 0.0, False
    design = np.column_stack((np.ones_like(logits), logits))
    # Start from the intercept-only fit (slope 0): it never saturates, whereas
    # the identity start can saturate extreme logits and stall Newton's method.
    rate = float(outcomes.mean())
    coefficients = np.array([math.log(rate) - math.log1p(-rate), 0.0])

    def log_likelihood(beta: FloatArray) -> float:
        linear = design @ beta
        return float(np.sum(outcomes * linear - np.logaddexp(0.0, linear)))

    current = log_likelihood(coefficients)
    tolerance = 1e-9 * max(1.0, float(outcomes.shape[0]))
    for _ in range(max_iterations):
        fitted = _sigmoid(design @ coefficients)
        weights = fitted * (1.0 - fitted)
        gradient = design.T @ (outcomes - fitted)
        if float(np.max(np.abs(gradient))) < tolerance:
            return float(coefficients[0]), float(coefficients[1]), True
        hessian = design.T @ (design * weights[:, None])
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            return float(coefficients[0]), float(coefficients[1]), False
        if not np.isfinite(step).all():
            return float(coefficients[0]), float(coefficients[1]), False
        # Backtrack until the likelihood does not decrease; never apply a
        # rejected step. Newton directions are ascent directions here, so a
        # small enough scale is always accepted unless we are at the optimum.
        scale = 1.0
        accepted = False
        while scale >= 1e-16:
            candidate = coefficients + scale * step
            value = log_likelihood(candidate)
            if np.isfinite(value) and value >= current:
                accepted = True
                break
            scale *= 0.5
        if not accepted:
            return float(coefficients[0]), float(coefficients[1]), False
        if np.array_equal(candidate, coefficients):
            # A floating-point no-op step: no progress and the gradient test at
            # the top of the loop already failed, so this is not convergence.
            return float(coefficients[0]), float(coefficients[1]), False
        coefficients = candidate
        current = value
        if not np.isfinite(coefficients).all():
            return float(coefficients[0]), float(coefficients[1]), False
    # The last permitted update may itself satisfy the score equations.
    fitted = _sigmoid(design @ coefficients)
    final_gradient = design.T @ (outcomes - fitted)
    converged = float(np.max(np.abs(final_gradient))) < tolerance
    return float(coefficients[0]), float(coefficients[1]), converged


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
        logits = _finite_logits(scores[:, index])
        # Ratio of sums: a mean of subnormal probabilities can underflow to zero
        # even when some prediction is nonzero.
        expected_total = float(np.sum(scores[:, index]))
        all_zero = not bool(np.any(scores[:, index] > 0.0))
        intercept, slope, converged = _logistic_slope(logits, outcomes, max_iterations)
        results.append(
            WeakCalibration(
                label=label,
                observed_expected_ratio=(
                    None if all_zero else float(outcomes.sum()) / expected_total
                ),
                calibration_in_the_large=_intercept_with_offset(logits, outcomes),
                calibration_slope=slope if converged else None,
                slope_intercept=intercept if converged else None,
                converged=converged,
            )
        )
    return tuple(results)


__all__ = ["CalibrationHierarchyError", "WeakCalibration", "weak_calibration"]
