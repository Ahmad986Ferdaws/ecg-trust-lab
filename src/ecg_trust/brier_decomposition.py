"""Exact per-label Brier score decomposition for multilabel probabilities.

The frozen tables report the Brier score as a single number, which can't say
whether an improvement came from calibration (reliability) or from sharper
discrimination (resolution). Murphy's decomposition splits it, and the two
within-bin terms of Stephenson, Coelho and Jolliffe ("Two Extra Components in
the Brier Score Decomposition", Weather and Forecasting, 2008) make it exact
for binned continuous forecasts:

``Brier = reliability - resolution + uncertainty + within_bin_variance
- 2 * within_bin_covariance``

where the within-bin terms are pooled (count-weighted) population variance of
forecasts and covariance between forecasts and outcomes inside bins. Bins are
nearly equal-mass, but tied probabilities always share a bin, so results never
depend on row order. Everything is descriptive and read-only.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ecg_trust.evaluation import validate_multilabel_arrays
from ecg_trust.protocol import LABEL_ORDER

FloatArray = NDArray[np.float64]


class BrierDecompositionError(ValueError):
    """Raised when a decomposition control is invalid."""


@dataclass(frozen=True, slots=True)
class BrierComponents:
    """Exact five-term Brier decomposition for one label."""

    label: str
    bins: int
    brier: float
    reliability: float
    resolution: float
    uncertainty: float
    within_bin_variance: float
    within_bin_covariance: float

    @property
    def reconstructed(self) -> float:
        return (
            self.reliability
            - self.resolution
            + self.uncertainty
            + self.within_bin_variance
            - 2.0 * self.within_bin_covariance
        )

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "label": self.label,
            "bins": self.bins,
            "brier": self.brier,
            "reliability": self.reliability,
            "resolution": self.resolution,
            "uncertainty": self.uncertainty,
            "within_bin_variance": self.within_bin_variance,
            "within_bin_covariance": self.within_bin_covariance,
        }


def _tie_aware_partitions(scores: FloatArray, n_bins: int) -> list[NDArray[np.intp]]:
    """Near equal-mass bins whose boundaries never split tied scores.

    Each ideal equal-mass cut snaps to the nearest position where the sorted
    score changes (forward or backward), so distinct scores are kept in
    separate bins whenever the ties allow it.
    """

    order = np.argsort(scores, kind="stable")
    ordered = scores[order]
    count = ordered.shape[0]
    boundaries = np.flatnonzero(ordered[1:] != ordered[:-1]) + 1
    bins = min(n_bins, count)
    cuts: set[int] = set()
    if boundaries.size:
        for index in range(1, bins):
            ideal = index * count / bins
            nearest = int(boundaries[np.argmin(np.abs(boundaries - ideal))])
            cuts.add(nearest)
    return [part for part in np.split(order, sorted(cuts)) if part.size]


def _decompose(label: str, scores: FloatArray, targets: FloatArray, n_bins: int) -> BrierComponents:
    count = scores.shape[0]
    base_rate = float(targets.mean())
    partitions = _tie_aware_partitions(scores, n_bins)
    reliability = resolution = variance = covariance = 0.0
    for part in partitions:
        forecast = scores[part]
        outcome = targets[part]
        mean_forecast = float(forecast.mean())
        event_rate = float(outcome.mean())
        weight = part.size / count
        reliability += weight * (mean_forecast - event_rate) ** 2
        resolution += weight * (event_rate - base_rate) ** 2
        variance += float(np.sum(np.square(forecast - mean_forecast))) / count
        covariance += float(np.sum((outcome - event_rate) * (forecast - mean_forecast))) / count
    return BrierComponents(
        label=label,
        bins=len(partitions),
        brier=float(np.mean(np.square(scores - targets))),
        reliability=reliability,
        resolution=resolution,
        uncertainty=base_rate * (1.0 - base_rate),
        within_bin_variance=variance,
        within_bin_covariance=covariance,
    )


def brier_decomposition(
    y_true: ArrayLike,
    probabilities: ArrayLike,
    *,
    n_bins: int = 10,
    label_order: Sequence[str] = LABEL_ORDER,
) -> tuple[BrierComponents, ...]:
    """Decompose each canonical label's Brier score over equal-mass bins."""

    if isinstance(n_bins, bool) or not isinstance(n_bins, int) or n_bins < 1:
        raise BrierDecompositionError("n_bins must be a positive integer")
    targets, scores = validate_multilabel_arrays(y_true, probabilities, label_order=label_order)
    target_float = targets.astype(np.float64)
    return tuple(
        _decompose(label, scores[:, index], target_float[:, index], n_bins)
        for index, label in enumerate(label_order)
    )


__all__ = ["BrierComponents", "BrierDecompositionError", "brier_decomposition"]
