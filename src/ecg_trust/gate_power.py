"""Plan preregistered proportion gates by computing their pass probability.

A one-shot protocol often passes only if an upper confidence bound on an
error proportion is at most a frozen maximum. Before freezing such a rule,
ask how likely it is to pass if the system behaves exactly as designed.

Two rate models are supported:

* a known true error rate, where the count is binomial;
* a split-conformal threshold set at the ``rank``-th smallest of ``n``
  calibration scores. Every validation record shares that random threshold,
  so with exchangeable scores the rejection count is beta-binomial with
  parameters ``n + 1 - rank`` and ``rank`` (its mean rate is the familiar
  ``(n + 1 - rank) / (n + 1)``).

Example: the source-support completion protocol thresholded at the 794th of
834 calibration scores and required a one-sided 95% upper bound of at most 5%
on 465 validation records. Under the conformal model that rule passes about
10% of the time even when the detector behaves exactly as designed.

Upper bounds are exact Clopper-Pearson (conservative) or Wilson score
intervals for a binomial count. Clustered data such as several ECGs per
patient widens real intervals, so treat these numbers as optimistic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from scipy.stats import beta, betabinom, binom, norm  # type: ignore[import-untyped]


class GatePowerError(ValueError):
    """Raised when a gate-planning control is invalid."""


class BoundMethod(StrEnum):
    CLOPPER_PEARSON = "clopper_pearson"
    WILSON = "wilson"


@dataclass(frozen=True, slots=True)
class ConformalThreshold:
    """A threshold at the ``rank``-th smallest of ``calibration_size`` scores."""

    calibration_size: int
    rank: int

    def __post_init__(self) -> None:
        _count(self.calibration_size, "calibration_size", minimum=1)
        _count(self.rank, "rank", minimum=1)
        if self.rank > self.calibration_size:
            raise GatePowerError("rank cannot exceed calibration_size")

    @property
    def design_rate(self) -> float:
        return (self.calibration_size + 1 - self.rank) / (self.calibration_size + 1)


@dataclass(frozen=True, slots=True)
class GatePlan:
    """Pass characteristics of an upper-bound proportion gate."""

    sample_size: int
    maximum_rate: float
    confidence: float
    method: BoundMethod
    max_passing_count: int | None
    true_rate: float
    rate_model: str
    pass_probability: float

    def to_dict(self) -> dict[str, object]:
        return {
            "rate_model": self.rate_model,
            "sample_size": self.sample_size,
            "maximum_rate": self.maximum_rate,
            "confidence": self.confidence,
            "method": self.method.value,
            "max_passing_count": self.max_passing_count,
            "true_rate": self.true_rate,
            "pass_probability": self.pass_probability,
        }


def _count(value: object, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise GatePowerError(f"{name} must be an integer of at least {minimum}")
    return value


def _open_unit(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GatePowerError(f"{name} must be a real number")
    number = float(value)
    if not math.isfinite(number) or not 0.0 < number < 1.0:
        raise GatePowerError(f"{name} must lie strictly between zero and one")
    return number


def _closed_unit(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GatePowerError(f"{name} must be a real number")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise GatePowerError(f"{name} must lie in [0, 1]")
    return number


def split_conformal_rejection_rate(calibration_size: int, rank: int) -> float:
    """Design rejection rate when the threshold is the ``rank``-th smallest score.

    For exchangeable, tie-free scores a new inlier exceeds that order statistic
    with probability ``(n + 1 - rank) / (n + 1)``.
    """

    return ConformalThreshold(calibration_size, rank).design_rate


def upper_bound(
    count: int,
    sample_size: int,
    *,
    confidence: float = 0.95,
    method: BoundMethod = BoundMethod.CLOPPER_PEARSON,
) -> float:
    """One-sided upper confidence bound for a binomial proportion."""

    size = _count(sample_size, "sample_size", minimum=1)
    events = _count(count, "count", minimum=0)
    if events > size:
        raise GatePowerError("count cannot exceed sample_size")
    level = _open_unit(confidence, "confidence")
    bound_method = BoundMethod(method)
    if bound_method is BoundMethod.CLOPPER_PEARSON:
        return 1.0 if events == size else float(beta.ppf(level, events + 1, size - events))
    z = float(norm.ppf(level))
    proportion = events / size
    denominator = 1.0 + z * z / size
    centre = proportion + z * z / (2.0 * size)
    spread = z * math.sqrt(proportion * (1.0 - proportion) / size + z * z / (4.0 * size * size))
    return min(1.0, (centre + spread) / denominator)


def max_passing_count(
    sample_size: int,
    maximum_rate: float,
    *,
    confidence: float = 0.95,
    method: BoundMethod = BoundMethod.CLOPPER_PEARSON,
) -> int | None:
    """Largest event count whose upper bound is at most ``maximum_rate``."""

    size = _count(sample_size, "sample_size", minimum=1)
    ceiling = _open_unit(maximum_rate, "maximum_rate")
    level = _open_unit(confidence, "confidence")
    bound_method = BoundMethod(method)

    def passes(events: int) -> bool:
        return upper_bound(events, size, confidence=level, method=bound_method) <= ceiling

    # Upper bounds increase with the event count, so binary search the boundary.
    if not passes(0):
        return None
    low, high = 0, size
    while low < high:
        middle = (low + high + 1) // 2
        if passes(middle):
            low = middle
        else:
            high = middle - 1
    return low


def plan_gate(
    *,
    sample_size: int,
    maximum_rate: float,
    true_rate: float | None = None,
    conformal: ConformalThreshold | None = None,
    confidence: float = 0.95,
    method: BoundMethod = BoundMethod.CLOPPER_PEARSON,
) -> GatePlan:
    """Probability that the gate passes under exactly one rate model."""

    if (true_rate is None) == (conformal is None):
        raise GatePowerError("provide exactly one of true_rate or conformal")
    size = _count(sample_size, "sample_size", minimum=1)
    bound_method = BoundMethod(method)
    passing = max_passing_count(size, maximum_rate, confidence=confidence, method=bound_method)
    if conformal is not None:
        if not isinstance(conformal, ConformalThreshold):
            raise GatePowerError("conformal must be a ConformalThreshold")
        rate = conformal.design_rate
        model = "conformal_beta_binomial"
        probability = (
            0.0
            if passing is None
            else float(
                betabinom.cdf(
                    passing,
                    size,
                    conformal.calibration_size + 1 - conformal.rank,
                    conformal.rank,
                )
            )
        )
    else:
        rate = _closed_unit(true_rate, "true_rate")
        model = "binomial"
        probability = 0.0 if passing is None else float(binom.cdf(passing, size, rate))
    return GatePlan(
        sample_size=size,
        maximum_rate=float(maximum_rate),
        confidence=float(confidence),
        method=bound_method,
        max_passing_count=passing,
        true_rate=rate,
        rate_model=model,
        pass_probability=probability,
    )


def required_sample_size(
    *,
    maximum_rate: float,
    true_rate: float | None = None,
    conformal: ConformalThreshold | None = None,
    target_probability: float = 0.8,
    confidence: float = 0.95,
    method: BoundMethod = BoundMethod.CLOPPER_PEARSON,
    limit: int = 20_000,
) -> int | None:
    """Smallest sample size reaching ``target_probability``, or ``None`` within ``limit``.

    Pass probability is not monotone in sample size (binomial sawtooth), so every
    size is checked in order, even when the design rate is at or above the
    maximum (a low target can still be met by sampling luck). ``None`` means no
    size up to ``limit`` qualified.
    """

    ceiling = _open_unit(maximum_rate, "maximum_rate")
    target = _open_unit(target_probability, "target_probability")
    cap = _count(limit, "limit", minimum=1)
    if (true_rate is None) == (conformal is None):
        raise GatePowerError("provide exactly one of true_rate or conformal")
    if conformal is None:
        _closed_unit(true_rate, "true_rate")
    for size in range(1, cap + 1):
        plan = plan_gate(
            sample_size=size,
            maximum_rate=ceiling,
            true_rate=true_rate,
            conformal=conformal,
            confidence=confidence,
            method=method,
        )
        if plan.pass_probability >= target:
            return size
    return None


__all__ = [
    "BoundMethod",
    "ConformalThreshold",
    "GatePlan",
    "GatePowerError",
    "max_passing_count",
    "plan_gate",
    "required_sample_size",
    "split_conformal_rejection_rate",
    "upper_bound",
]
