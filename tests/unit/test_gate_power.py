from __future__ import annotations

import json

import pytest
from scipy.stats import binom

from ecg_trust.gate_power import (
    BoundMethod,
    GatePowerError,
    max_passing_count,
    plan_gate,
    required_sample_size,
    split_conformal_rejection_rate,
    upper_bound,
)
from scripts import plan_gate as cli


def test_split_conformal_design_rate_matches_the_order_statistic_formula() -> None:
    assert split_conformal_rejection_rate(834, 794) == pytest.approx(41 / 835)
    assert split_conformal_rejection_rate(9, 9) == pytest.approx(0.1)


@pytest.mark.parametrize("method", list(BoundMethod))
def test_upper_bounds_are_monotone_and_bracket_the_estimate(method: BoundMethod) -> None:
    bounds = [upper_bound(count, 100, method=method) for count in range(0, 101, 10)]

    assert bounds == sorted(bounds)
    assert all(bound >= count / 100 for bound, count in zip(bounds, range(0, 101, 10), strict=True))
    assert upper_bound(100, 100, method=method) == pytest.approx(1.0)


def test_clopper_pearson_zero_count_has_the_closed_form_bound() -> None:
    assert upper_bound(0, 50) == pytest.approx(1.0 - 0.05 ** (1 / 50))


def test_source_support_rule_was_unlikely_to_pass_as_designed() -> None:
    design_rate = split_conformal_rejection_rate(834, 794)

    plan = plan_gate(sample_size=465, maximum_rate=0.05, true_rate=design_rate)

    assert plan.max_passing_count == 15
    assert upper_bound(15, 465) <= 0.05 < upper_bound(16, 465)
    assert plan.pass_probability == pytest.approx(binom.cdf(15, 465, design_rate))
    assert plan.pass_probability < 0.1
    assert upper_bound(25, 465) > 0.05


def test_a_stricter_threshold_makes_the_same_gate_usually_pass() -> None:
    design_rate = split_conformal_rejection_rate(834, 815)

    plan = plan_gate(sample_size=465, maximum_rate=0.05, true_rate=design_rate)

    assert plan.pass_probability > 0.8
    assert plan.to_dict()["method"] == "clopper_pearson"


def test_required_sample_size_reaches_the_target_or_reports_infeasible() -> None:
    size = required_sample_size(maximum_rate=0.05, true_rate=0.025, target_probability=0.8)

    assert size is not None
    assert plan_gate(sample_size=size, maximum_rate=0.05, true_rate=0.025).pass_probability >= 0.8
    assert required_sample_size(maximum_rate=0.05, true_rate=0.05) is None


def test_gate_that_no_count_can_pass_has_zero_probability() -> None:
    assert max_passing_count(5, 0.01) is None
    assert plan_gate(sample_size=5, maximum_rate=0.01, true_rate=0.0).pass_probability == 0.0


@pytest.mark.parametrize(
    "call",
    [
        lambda: split_conformal_rejection_rate(10, 11),
        lambda: split_conformal_rejection_rate(True, 1),
        lambda: upper_bound(5, 4),
        lambda: upper_bound(1, 10, confidence=1.0),
        lambda: plan_gate(sample_size=0, maximum_rate=0.05, true_rate=0.01),
        lambda: plan_gate(sample_size=10, maximum_rate=float("nan"), true_rate=0.01),
        lambda: plan_gate(sample_size=10, maximum_rate=0.05, true_rate=1.5),
    ],
)
def test_invalid_controls_are_rejected(call: object) -> None:
    with pytest.raises(GatePowerError):
        call()  # type: ignore[operator]


def test_cli_reports_json_and_rejects_bad_controls(capsys: pytest.CaptureFixture[str]) -> None:
    assert (
        cli.main(["--sample-size", "465", "--maximum-rate", "0.05", "--conformal", "834", "794"])
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["max_passing_count"] == 15
    assert report["pass_probability"] < 0.1

    assert cli.main(["--sample-size", "465", "--maximum-rate", "2", "--true-rate", "0.01"]) == 2
    assert "maximum_rate" in capsys.readouterr().err
