"""Estimate and correct per-label prevalence shift without target labels.

The SPH transport cohort differs sharply from PTB-XL in label prevalence
(its label-selected NORM share is far higher), so calibrated PTB-XL
probabilities are systematically miscalibrated there even when the class-
conditional ECG distributions are similar. Under the label-shift assumption
(``p(x | y)`` unchanged, only ``p(y)`` moves) Bayes' rule gives the adjusted
posterior for a new prior, and the EM procedure of Saerens, Latinne and
Decaestecker ("Adjusting the Outputs of a Classifier to New a Priori
Probabilities: A Simple Procedure", Neural Computation 14(1), 2002) estimates
that prior from unlabeled target predictions.

Each label is treated as an independent binary problem. Results are only as
good as the source calibration and the label-shift assumption; covariate or
concept shift violates it. This is a development tool: applying it to the
sealed SPH run would be a new analysis with its own protocol.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
_PRIOR_FLOOR = 1e-12


class PriorShiftError(ValueError):
    """Raised when prior-shift inputs or controls are malformed."""


@dataclass(frozen=True, slots=True)
class LabelPriorEstimate:
    source_prior: float
    target_prior: float
    iterations: int
    converged: bool


@dataclass(frozen=True, slots=True)
class PriorShiftResult:
    labels: tuple[LabelPriorEstimate, ...]
    adjusted_probabilities: FloatArray

    def to_dict(self) -> dict[str, object]:
        return {
            "labels": [
                {
                    "source_prior": item.source_prior,
                    "target_prior": item.target_prior,
                    "iterations": item.iterations,
                    "converged": item.converged,
                }
                for item in self.labels
            ]
        }


def _probabilities(values: ArrayLike) -> FloatArray:
    try:
        raw = np.asarray(values)
    except (TypeError, ValueError, OverflowError) as error:
        raise PriorShiftError("probabilities must be numeric") from error
    if raw.dtype == np.object_ or np.issubdtype(raw.dtype, np.bool_):
        raise PriorShiftError("probabilities must use a real numeric dtype")
    if np.iscomplexobj(raw) or not np.issubdtype(raw.dtype, np.number):
        raise PriorShiftError("probabilities must be real-valued")
    if raw.ndim != 2 or raw.shape[0] == 0 or raw.shape[1] == 0:
        raise PriorShiftError("probabilities must be a non-empty [records, labels] matrix")
    matrix = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(matrix).all() or np.any((matrix < 0.0) | (matrix > 1.0)):
        raise PriorShiftError("probabilities must be finite and lie in [0, 1]")
    return matrix


def _priors(values: Sequence[float], labels: int, name: str) -> tuple[float, ...]:
    priors = tuple(values)
    if len(priors) != labels:
        raise PriorShiftError(f"{name} must have one prior per label")
    for prior in priors:
        if (
            isinstance(prior, bool)
            or not isinstance(prior, (int, float))
            or not math.isfinite(prior)
            or not 0.0 < prior < 1.0
        ):
            raise PriorShiftError(f"{name} must lie strictly between zero and one")
    return tuple(float(prior) for prior in priors)


def _reweight(probabilities: FloatArray, source: float, target: float) -> FloatArray:
    positive = probabilities * (target / source)
    negative = (1.0 - probabilities) * ((1.0 - target) / (1.0 - source))
    denominator = positive + negative
    safe = np.where(denominator > 0.0, denominator, 1.0)
    return np.asarray(np.where(denominator > 0.0, positive / safe, probabilities), dtype=np.float64)


def adjust_to_priors(
    probabilities: ArrayLike,
    *,
    source_priors: Sequence[float],
    target_priors: Sequence[float],
) -> FloatArray:
    """Apply Bayes' prior correction for known source and target prevalences."""

    matrix = _probabilities(probabilities)
    sources = _priors(source_priors, matrix.shape[1], "source_priors")
    targets = _priors(target_priors, matrix.shape[1], "target_priors")
    return np.stack(
        [
            _reweight(matrix[:, index], sources[index], targets[index])
            for index in range(matrix.shape[1])
        ],
        axis=1,
    )


def estimate_label_shift(
    probabilities: ArrayLike,
    *,
    source_priors: Sequence[float],
    tolerance: float = 1e-8,
    max_iterations: int = 1_000,
) -> PriorShiftResult:
    """Estimate each label's target prevalence by EM and return adjusted probabilities."""

    matrix = _probabilities(probabilities)
    sources = _priors(source_priors, matrix.shape[1], "source_priors")
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or not math.isfinite(tolerance)
        or tolerance <= 0.0
    ):
        raise PriorShiftError("tolerance must be finite and positive")
    if (
        isinstance(max_iterations, bool)
        or not isinstance(max_iterations, int)
        or max_iterations < 1
    ):
        raise PriorShiftError("max_iterations must be a positive integer")

    estimates: list[LabelPriorEstimate] = []
    columns: list[FloatArray] = []
    for index, source in enumerate(sources):
        column = matrix[:, index]
        target = source
        converged = False
        iterations = 0
        while iterations < max_iterations:
            iterations += 1
            adjusted = _reweight(column, source, target)
            updated = min(max(float(adjusted.mean()), _PRIOR_FLOOR), 1.0 - _PRIOR_FLOOR)
            converged = abs(updated - target) <= tolerance
            target = updated
            if converged:
                break
        adjusted = _reweight(column, source, target)
        estimates.append(
            LabelPriorEstimate(
                source_prior=source,
                target_prior=target,
                iterations=iterations,
                converged=converged,
            )
        )
        columns.append(adjusted)
    return PriorShiftResult(
        labels=tuple(estimates), adjusted_probabilities=np.stack(columns, axis=1)
    )


__all__ = [
    "LabelPriorEstimate",
    "PriorShiftError",
    "PriorShiftResult",
    "adjust_to_priors",
    "estimate_label_shift",
]
