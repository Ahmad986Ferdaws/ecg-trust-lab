"""Equal-mass calibration error estimators for multilabel probabilities.

``evaluation.fixed_bin_ece`` uses equal-width bins. With rare labels most
equal-width bins are nearly empty, which makes the estimate noisy and biased
(the model card already carries a binning caveat for ECE). Equal-mass bins
put the same number of predictions in each bin:

* ``adaptive_calibration_error``: count-weighted ``|event rate - mean
  probability|`` over equal-mass bins (the adaptive scheme of Nixon et al.,
  "Measuring Calibration in Deep Learning", CVPR Workshops 2019).
* ``monotonic_sweep_calibration_error``: the equal-mass estimate at the largest
  candidate bin count whose bin event rates are monotone (ECE-sweep; Roelofs et al.,
  "Mitigating Bias in Calibration Error Estimation", AISTATS 2022).

Both are descriptive, read-only statistics that fit nothing.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ecg_trust.evaluation import validate_multilabel_arrays
from ecg_trust.protocol import LABEL_ORDER

FloatArray = NDArray[np.float64]


class CalibrationMetricError(ValueError):
    """Raised when a calibration-metric control is invalid."""


@dataclass(frozen=True, slots=True)
class LabelCalibrationError:
    """Equal-mass calibration error for one label."""

    label: str
    bins: int
    calibration_error: float


@dataclass(frozen=True, slots=True)
class MultilabelCalibrationError:
    """Per-label equal-mass calibration errors and their unweighted macro mean."""

    method: str
    labels: tuple[LabelCalibrationError, ...]

    @property
    def macro(self) -> float:
        return float(np.mean([item.calibration_error for item in self.labels]))

    def to_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "macro": self.macro,
            "labels": {
                item.label: {"bins": item.bins, "calibration_error": item.calibration_error}
                for item in self.labels
            },
        }


def _bin_count(n_bins: object, *, minimum: int = 1) -> int:
    if isinstance(n_bins, bool) or not isinstance(n_bins, int) or n_bins < minimum:
        raise CalibrationMetricError(f"bin count must be an integer of at least {minimum}")
    return n_bins


def _equal_mass_bins(
    scores: FloatArray, targets: FloatArray, n_bins: int
) -> tuple[FloatArray, FloatArray, FloatArray]:
    order = np.argsort(scores, kind="stable")
    partitions = [part for part in np.array_split(order, min(n_bins, scores.size)) if part.size]
    counts = np.array([part.size for part in partitions], dtype=np.float64)
    confidence = np.array([scores[part].mean() for part in partitions], dtype=np.float64)
    event_rate = np.array([targets[part].mean() for part in partitions], dtype=np.float64)
    return counts, confidence, event_rate


def _binary_ace(scores: FloatArray, targets: FloatArray, n_bins: int) -> float:
    counts, confidence, event_rate = _equal_mass_bins(scores, targets, n_bins)
    return float(np.sum(counts * np.abs(event_rate - confidence)) / counts.sum())


def _binary_sweep(scores: FloatArray, targets: FloatArray, max_bins: int) -> tuple[int, float]:
    # Equal-mass partitions for different bin counts are not nested, so a
    # non-monotone count does not rule out a larger monotone one: search them all.
    chosen = 1
    for bins in range(2, min(max_bins, scores.size) + 1):
        _, _, event_rate = _equal_mass_bins(scores, targets, bins)
        if not np.any(np.diff(event_rate) < 0.0):
            chosen = bins
    return chosen, _binary_ace(scores, targets, chosen)


def adaptive_calibration_error(
    y_true: ArrayLike,
    probabilities: ArrayLike,
    *,
    n_bins: int = 15,
    label_order: Sequence[str] = LABEL_ORDER,
) -> MultilabelCalibrationError:
    """Equal-mass calibration error per canonical label.

    Predictions are ordered by probability with a stable sort and split into
    ``min(n_bins, samples)`` nearly equal groups; tied probabilities that straddle
    a boundary are assigned by input order.
    """

    bins = _bin_count(n_bins)
    targets, scores = validate_multilabel_arrays(y_true, probabilities, label_order=label_order)
    target_float = targets.astype(np.float64)
    return MultilabelCalibrationError(
        method="equal_mass_ace",
        labels=tuple(
            LabelCalibrationError(
                label=label,
                bins=min(bins, scores.shape[0]),
                calibration_error=_binary_ace(scores[:, index], target_float[:, index], bins),
            )
            for index, label in enumerate(label_order)
        ),
    )


def monotonic_sweep_calibration_error(
    y_true: ArrayLike,
    probabilities: ArrayLike,
    *,
    max_bins: int = 100,
    label_order: Sequence[str] = LABEL_ORDER,
) -> MultilabelCalibrationError:
    """Equal-mass calibration error at the largest monotone bin count per label."""

    limit = _bin_count(max_bins, minimum=2)
    targets, scores = validate_multilabel_arrays(y_true, probabilities, label_order=label_order)
    target_float = targets.astype(np.float64)
    results: list[LabelCalibrationError] = []
    for index, label in enumerate(label_order):
        bins, error = _binary_sweep(scores[:, index], target_float[:, index], limit)
        results.append(LabelCalibrationError(label=label, bins=bins, calibration_error=error))
    return MultilabelCalibrationError(method="equal_mass_monotonic_sweep", labels=tuple(results))


__all__ = [
    "CalibrationMetricError",
    "LabelCalibrationError",
    "MultilabelCalibrationError",
    "adaptive_calibration_error",
    "monotonic_sweep_calibration_error",
]
