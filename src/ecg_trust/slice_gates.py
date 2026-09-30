"""Block a candidate that regresses overall or in any evaluated group.

Averages hide weak groups: the r3 audit found age 80+ notably under-covered
by the global abstention gate even though aggregate coverage looked fine.
This gate compares a candidate with a reference, per group and overall, and
fails when either drop exceeds its limit. Groups too small to judge are
reported as ``insufficient`` and, by default, block the gate as well, so
thin evidence is never read as a pass.

Values are any "higher is better" or "lower is better" metric (coverage,
accuracy, selective risk). Both sides must report the same groups.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum


class SliceGateError(ValueError):
    """Raised when slice observations or gate controls are malformed."""


class SliceStatus(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    INSUFFICIENT = "insufficient"


class InsufficientPolicy(StrEnum):
    BLOCK = "block"
    REPORT = "report"


@dataclass(frozen=True, slots=True)
class SliceObservation:
    """One group's evaluated count and metric value."""

    group: str
    count: int
    value: float

    def __post_init__(self) -> None:
        if not isinstance(self.group, str) or not self.group.strip():
            raise SliceGateError("group must be a non-empty string")
        if isinstance(self.count, bool) or not isinstance(self.count, int) or self.count < 0:
            raise SliceGateError("count must be a non-negative integer")
        if (
            isinstance(self.value, bool)
            or not isinstance(self.value, (int, float))
            or not math.isfinite(self.value)
        ):
            raise SliceGateError("value must be a finite real number")


@dataclass(frozen=True, slots=True)
class SliceVerdict:
    group: str
    reference: float
    candidate: float
    degradation: float
    status: SliceStatus


@dataclass(frozen=True, slots=True)
class SliceGateResult:
    passed: bool
    overall_degradation: float | None
    overall_status: SliceStatus
    slices: tuple[SliceVerdict, ...]

    @property
    def failing_groups(self) -> tuple[str, ...]:
        return tuple(item.group for item in self.slices if item.status is SliceStatus.FAIL)

    @property
    def insufficient_groups(self) -> tuple[str, ...]:
        return tuple(item.group for item in self.slices if item.status is SliceStatus.INSUFFICIENT)

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "overall_degradation": self.overall_degradation,
            "overall_status": self.overall_status.value,
            "slices": [
                {
                    "group": item.group,
                    "reference": item.reference,
                    "candidate": item.candidate,
                    "degradation": item.degradation,
                    "status": item.status.value,
                }
                for item in self.slices
            ],
        }


def _limit(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SliceGateError(f"{name} must be a real number")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise SliceGateError(f"{name} must be finite and non-negative")
    return number


def _exceeds(degradation: float, limit: float) -> bool:
    # Treat float noise at the boundary (0.80 - 0.75 = 0.05000000000000004) as equal.
    return degradation > limit and not math.isclose(degradation, limit, rel_tol=1e-9, abs_tol=1e-12)


def _by_group(observations: Sequence[SliceObservation], name: str) -> dict[str, SliceObservation]:
    if not observations:
        raise SliceGateError(f"{name} must contain at least one group")
    result: dict[str, SliceObservation] = {}
    for item in observations:
        if not isinstance(item, SliceObservation):
            raise SliceGateError(f"{name} must contain SliceObservation items")
        if item.group in result:
            raise SliceGateError(f"{name} repeats group {item.group!r}")
        result[item.group] = item
    return result


def _weighted(
    observations: dict[str, SliceObservation],
    weights: dict[str, SliceObservation],
    groups: Sequence[str],
) -> float:
    total = sum(weights[group].count for group in groups)
    return math.fsum(observations[group].value * (weights[group].count / total) for group in groups)


def evaluate_slice_gate(
    reference: Sequence[SliceObservation],
    candidate: Sequence[SliceObservation],
    *,
    max_overall_drop: float = 0.02,
    max_slice_drop: float = 0.05,
    min_count: int = 30,
    higher_is_better: bool = True,
    insufficient_policy: InsufficientPolicy = InsufficientPolicy.BLOCK,
) -> SliceGateResult:
    """Compare candidate to reference overall and per group.

    The overall comparison covers only sufficiently large groups and weights
    both sides by their reference counts, so it reflects metric changes at a
    fixed group composition. With no sufficient group it is ``insufficient``
    and the gate fails regardless of ``insufficient_policy``.

    ``degradation`` is the move in the worse direction (positive = worse). A
    group is insufficient when either side has fewer than ``min_count`` records.
    """

    overall_limit = _limit(max_overall_drop, "max_overall_drop")
    slice_limit = _limit(max_slice_drop, "max_slice_drop")
    if isinstance(min_count, bool) or not isinstance(min_count, int) or min_count < 1:
        raise SliceGateError("min_count must be a positive integer")
    if not isinstance(higher_is_better, bool):
        raise SliceGateError("higher_is_better must be a boolean")
    policy = InsufficientPolicy(insufficient_policy)
    before = _by_group(reference, "reference")
    after = _by_group(candidate, "candidate")
    if set(before) != set(after):
        missing = sorted(set(before) ^ set(after))
        raise SliceGateError(f"reference and candidate must report the same groups: {missing}")
    sign = 1.0 if higher_is_better else -1.0

    verdicts: list[SliceVerdict] = []
    for group in sorted(before):
        old, new = before[group], after[group]
        degradation = sign * (old.value - new.value)
        if min(old.count, new.count) < min_count:
            status = SliceStatus.INSUFFICIENT
        elif _exceeds(degradation, slice_limit):
            status = SliceStatus.FAIL
        else:
            status = SliceStatus.PASS
        verdicts.append(
            SliceVerdict(
                group=group,
                reference=float(old.value),
                candidate=float(new.value),
                degradation=degradation,
                status=status,
            )
        )
    # The overall comparison uses only groups large enough to judge, and weights
    # both sides by the reference composition so a shift in group mix is not
    # mistaken for (or allowed to hide) a change in the metric.
    supported = [item.group for item in verdicts if item.status is not SliceStatus.INSUFFICIENT]
    overall: float | None = None
    if supported:
        overall = sign * (
            _weighted(before, before, supported) - _weighted(after, before, supported)
        )
        overall_status = SliceStatus.FAIL if _exceeds(overall, overall_limit) else SliceStatus.PASS
    else:
        overall_status = SliceStatus.INSUFFICIENT
    blocked = {SliceStatus.FAIL}
    if policy is InsufficientPolicy.BLOCK:
        blocked.add(SliceStatus.INSUFFICIENT)
    passed = overall_status is SliceStatus.PASS and not any(
        item.status in blocked for item in verdicts
    )
    return SliceGateResult(
        passed=passed,
        overall_degradation=overall,
        overall_status=overall_status,
        slices=tuple(verdicts),
    )


__all__ = [
    "InsufficientPolicy",
    "SliceGateError",
    "SliceGateResult",
    "SliceObservation",
    "SliceStatus",
    "SliceVerdict",
    "evaluate_slice_gate",
]
