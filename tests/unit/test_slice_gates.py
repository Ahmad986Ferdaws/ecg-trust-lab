from __future__ import annotations

import pytest

from ecg_trust.slice_gates import (
    InsufficientPolicy,
    SliceGateError,
    SliceObservation,
    SliceStatus,
    evaluate_slice_gate,
)


def _obs(*rows: tuple[str, int, float]) -> list[SliceObservation]:
    return [SliceObservation(group, count, value) for group, count, value in rows]


REFERENCE = _obs(("age_18_64", 600, 0.82), ("age_65_79", 300, 0.80), ("age_80_plus", 100, 0.78))


def test_unchanged_candidate_passes() -> None:
    result = evaluate_slice_gate(REFERENCE, REFERENCE)

    assert result.passed
    assert result.overall_degradation == pytest.approx(0.0)
    assert {item.status for item in result.slices} == {SliceStatus.PASS}


def test_one_group_regression_fails_even_when_the_average_barely_moves() -> None:
    candidate = _obs(("age_18_64", 600, 0.83), ("age_65_79", 300, 0.80), ("age_80_plus", 100, 0.70))

    result = evaluate_slice_gate(REFERENCE, candidate)

    assert result.overall_degradation == pytest.approx(0.002)
    assert result.overall_status is SliceStatus.PASS
    assert not result.passed
    assert result.failing_groups == ("age_80_plus",)


def test_overall_drop_fails_without_any_single_group_failing() -> None:
    candidate = _obs(("age_18_64", 600, 0.79), ("age_65_79", 300, 0.77), ("age_80_plus", 100, 0.75))

    result = evaluate_slice_gate(REFERENCE, candidate)

    assert result.failing_groups == ()
    assert result.overall_status is SliceStatus.FAIL
    assert not result.passed


def test_group_mix_shift_alone_is_not_a_regression() -> None:
    reference = _obs(("high", 900, 0.9), ("low", 100, 0.5))
    shifted = _obs(("high", 100, 0.9), ("low", 900, 0.5))
    hidden = _obs(("high", 100, 0.87), ("low", 900, 0.47))

    unchanged = evaluate_slice_gate(reference, shifted)
    degraded = evaluate_slice_gate(reference, hidden, max_slice_drop=0.05, max_overall_drop=0.02)

    assert unchanged.overall_degradation == pytest.approx(0.0)
    assert unchanged.passed
    assert degraded.overall_degradation == pytest.approx(0.03)
    assert degraded.overall_status is SliceStatus.FAIL


def test_lower_is_better_metrics_flip_the_direction() -> None:
    risk = _obs(("a", 100, 0.10), ("b", 100, 0.12))
    worse = _obs(("a", 100, 0.10), ("b", 100, 0.20))

    assert not evaluate_slice_gate(risk, worse, higher_is_better=False).passed
    assert evaluate_slice_gate(worse, risk, higher_is_better=False).passed


def test_small_groups_block_by_default_and_can_be_reported_only() -> None:
    candidate = _obs(("age_18_64", 600, 0.82), ("age_65_79", 300, 0.80), ("age_80_plus", 20, 0.78))

    blocked = evaluate_slice_gate(REFERENCE, candidate)
    reported = evaluate_slice_gate(
        REFERENCE, candidate, insufficient_policy=InsufficientPolicy.REPORT
    )

    assert blocked.insufficient_groups == ("age_80_plus",)
    assert not blocked.passed
    assert reported.passed
    assert reported.to_dict()["slices"][2]["status"] == "insufficient"  # type: ignore[index]


def test_drop_exactly_at_the_limit_passes() -> None:
    candidate = _obs(("a", 100, 0.75))

    assert evaluate_slice_gate(_obs(("a", 100, 0.80)), candidate, max_overall_drop=0.06).passed


@pytest.mark.parametrize(
    ("reference", "candidate", "kwargs", "message"),
    [
        (REFERENCE, REFERENCE[:2], {}, "same groups"),
        ([], [], {}, "at least one group"),
        (REFERENCE + REFERENCE[:1], REFERENCE, {}, "repeats"),
        (_obs(("a", 0, 0.5)), _obs(("a", 40, 0.5)), {}, "counted reference record"),
        (REFERENCE, REFERENCE, {"max_slice_drop": -0.1}, "non-negative"),
        (REFERENCE, REFERENCE, {"max_overall_drop": float("nan")}, "finite"),
        (REFERENCE, REFERENCE, {"min_count": 0}, "positive integer"),
        (REFERENCE, REFERENCE, {"min_count": True}, "positive integer"),
        (REFERENCE, REFERENCE, {"higher_is_better": 1}, "boolean"),
        (REFERENCE, REFERENCE, {"insufficient_policy": "ignore"}, "ignore"),
    ],
)
def test_malformed_inputs_are_rejected(
    reference: list[SliceObservation],
    candidate: list[SliceObservation],
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises((SliceGateError, ValueError), match=message):
        evaluate_slice_gate(reference, candidate, **kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("group", "count", "value"),
    [("", 1, 0.5), ("a", -1, 0.5), ("a", True, 0.5), ("a", 1, float("inf")), ("a", 1, True)],
)
def test_observation_validation(group: str, count: int, value: float) -> None:
    with pytest.raises(SliceGateError):
        SliceObservation(group, count, value)
