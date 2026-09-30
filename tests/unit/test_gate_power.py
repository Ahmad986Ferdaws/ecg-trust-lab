from __future__ import annotations

import json

import numpy as np
import pytest
from scipy.stats import betabinom, binom

from ecg_trust.gate_power import (
    BoundMethod,
    ConformalThreshold,
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
    threshold = ConformalThreshold(834, 794)

    plan = plan_gate(sample_size=465, maximum_rate=0.05, conformal=threshold)
    fixed_rate = plan_gate(sample_size=465, maximum_rate=0.05, true_rate=threshold.design_rate)

    assert plan.max_passing_count == 15
    assert upper_bound(15, 465) <= 0.05 < upper_bound(16, 465)
    assert plan.rate_model == "conformal_beta_binomial"
    assert plan.true_rate == pytest.approx(41 / 835)
    assert plan.pass_probability == pytest.approx(betabinom.cdf(15, 465, 41, 794))
    assert 0.09 < plan.pass_probability < 0.1
    assert fixed_rate.pass_probability == pytest.approx(binom.cdf(15, 465, 41 / 835))
    assert fixed_rate.pass_probability < plan.pass_probability
    assert upper_bound(25, 465) > 0.05


def test_conformal_count_distribution_matches_simulation() -> None:
    rng = np.random.default_rng(0)
    passes = 0
    trials = 4000
    for _ in range(trials):
        threshold = np.sort(rng.uniform(size=40))[35]
        passes += int((rng.uniform(size=60) > threshold).sum() <= 5)
    plan = plan_gate(sample_size=60, maximum_rate=0.2, conformal=ConformalThreshold(40, 36))
    expected = betabinom.cdf(5, 60, 5, 36)

    assert plan.pass_probability == pytest.approx(betabinom.cdf(plan.max_passing_count, 60, 5, 36))
    assert passes / trials == pytest.approx(expected, abs=0.03)


def test_a_stricter_threshold_makes_the_same_gate_usually_pass() -> None:
    plan = plan_gate(sample_size=465, maximum_rate=0.05, conformal=ConformalThreshold(834, 815))

    assert plan.pass_probability > 0.8
    assert plan.to_dict()["method"] == "clopper_pearson"


def test_required_sample_size_reaches_the_target_or_reports_infeasible() -> None:
    size = required_sample_size(maximum_rate=0.05, true_rate=0.025, target_probability=0.8)

    assert size is not None
    assert plan_gate(sample_size=size, maximum_rate=0.05, true_rate=0.025).pass_probability >= 0.8
    assert required_sample_size(maximum_rate=0.05, true_rate=0.05) is None


@pytest.mark.parametrize("method", list(BoundMethod))
def test_binary_search_matches_a_linear_scan(method: BoundMethod) -> None:
    for size in (1, 7, 50, 333):
        linear = None
        for events in range(size + 1):
            if upper_bound(events, size, method=method) > 0.2:
                break
            linear = events
        assert max_passing_count(size, 0.2, method=method) == linear


def test_required_sample_size_checks_every_candidate() -> None:
    size = required_sample_size(
        maximum_rate=0.01, true_rate=0.001, target_probability=0.1, limit=300
    )

    assert size == 299


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
        lambda: plan_gate(sample_size=10, maximum_rate=0.05),
        lambda: plan_gate(
            sample_size=10, maximum_rate=0.05, true_rate=0.01, conformal=ConformalThreshold(9, 9)
        ),
        lambda: ConformalThreshold(10, 0),
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
    assert report["rate_model"] == "conformal_beta_binomial"
    assert 0.09 < report["pass_probability"] < 0.1

    assert cli.main(["--sample-size", "465", "--maximum-rate", "2", "--true-rate", "0.01"]) == 2
    assert "maximum_rate" in capsys.readouterr().err
