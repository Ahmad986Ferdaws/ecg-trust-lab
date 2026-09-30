"""Reference-normalized summaries of selective (abstaining) classification.

The existing risk-coverage analyses report the raw area under the risk-coverage
curve (AURC). Raw AURC mixes two things: how well an uncertainty score ranks
failures and how many failures the underlying classifier makes. The functions
here separate them without fitting anything:

* ``oracle_aurc`` is the AURC of a perfect ranking of the same per-sample losses.
* ``excess_aurc`` (E-AURC; Geifman, Uziel and El-Yaniv, ICLR 2019) subtracts
  that oracle, so it is zero for a perfect ranking regardless of accuracy.

Tied uncertainty values are handled by their expected value under a random
tie-break, so the result never depends on input order.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


class SelectiveMetricError(ValueError):
    """Raised when selective-classification inputs are malformed."""


@dataclass(frozen=True, slots=True)
class ExcessAurc:
    """Raw, oracle, and excess area under the risk-coverage curve."""

    count: int
    aurc: float
    oracle_aurc: float
    excess_aurc: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "count": self.count,
            "aurc": self.aurc,
            "oracle_aurc": self.oracle_aurc,
            "excess_aurc": self.excess_aurc,
        }


def _vector(values: ArrayLike, *, name: str, count: int | None = None) -> FloatArray:
    try:
        raw = np.asarray(values)
    except (TypeError, ValueError, OverflowError) as error:
        raise SelectiveMetricError(f"{name} must be a numeric vector") from error
    if raw.dtype == np.object_ or np.issubdtype(raw.dtype, np.bool_):
        raise SelectiveMetricError(f"{name} must use a real numeric dtype")
    if np.iscomplexobj(raw) or not np.issubdtype(raw.dtype, np.number):
        raise SelectiveMetricError(f"{name} must be real-valued")
    if raw.ndim != 1 or raw.size == 0:
        raise SelectiveMetricError(f"{name} must be a non-empty one-dimensional vector")
    try:
        vector = np.asarray(raw, dtype=np.float64)
    except OverflowError as error:
        raise SelectiveMetricError(f"{name} overflows float64") from error
    if not np.isfinite(vector).all():
        raise SelectiveMetricError(f"{name} must be finite")
    if count is not None and vector.shape[0] != count:
        raise SelectiveMetricError(f"{name} must have {count} entries")
    return vector


def _losses(values: ArrayLike) -> FloatArray:
    losses = _vector(values, name="losses")
    if np.any(losses < 0.0):
        raise SelectiveMetricError("losses must be non-negative")
    return losses


def selective_risk_curve(losses: ArrayLike, uncertainty: ArrayLike) -> FloatArray:
    """Return expected selective risk after accepting the ``k`` least-uncertain cases.

    Entry ``k - 1`` is the mean loss of the accepted prefix of size ``k``. Within a
    run of tied uncertainty values each accepted member contributes the tie
    group's mean loss, which is the expectation over random tie-breaking.
    """

    loss = _losses(losses)
    score = _vector(uncertainty, name="uncertainty", count=loss.shape[0])
    order = np.argsort(score, kind="stable")
    sorted_score = score[order]
    sorted_loss = loss[order]
    boundaries = np.flatnonzero(np.diff(sorted_score)) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [loss.shape[0]]))
    expected = np.empty_like(sorted_loss)
    for start, end in zip(starts, ends, strict=True):
        expected[start:end] = sorted_loss[start:end].mean()
    accepted = np.arange(1, loss.shape[0] + 1, dtype=np.float64)
    return np.asarray(np.cumsum(expected) / accepted, dtype=np.float64)


def aurc(losses: ArrayLike, uncertainty: ArrayLike) -> float:
    """Area under the risk-coverage curve, averaged over every coverage step."""

    return float(selective_risk_curve(losses, uncertainty).mean())


def oracle_aurc(losses: ArrayLike) -> float:
    """AURC of the ideal ranking that accepts the lowest-loss cases first."""

    loss = _losses(losses)
    ordered = np.sort(loss)
    accepted = np.arange(1, loss.shape[0] + 1, dtype=np.float64)
    return float((np.cumsum(ordered) / accepted).mean())


def excess_aurc(losses: ArrayLike, uncertainty: ArrayLike) -> ExcessAurc:
    """Return raw AURC, the oracle AURC for the same losses, and their difference."""

    loss = _losses(losses)
    raw = aurc(loss, uncertainty)
    oracle = oracle_aurc(loss)
    return ExcessAurc(
        count=int(loss.shape[0]),
        aurc=raw,
        oracle_aurc=oracle,
        excess_aurc=max(raw - oracle, 0.0),
    )


__all__ = [
    "ExcessAurc",
    "SelectiveMetricError",
    "aurc",
    "excess_aurc",
    "oracle_aurc",
    "selective_risk_curve",
]
