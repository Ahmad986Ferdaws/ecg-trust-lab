from dataclasses import replace
from decimal import Decimal

import pytest

from scripts.estimate_research_pilot import PilotInputs, estimate, main


def scenario() -> PilotInputs:
    return PilotInputs(*(Decimal(value) for value in (5000, 24, 6, 100, 200, 100, 3000, 120)))


def test_hand_calculated_costs_capacity_and_break_even() -> None:
    result = estimate(scenario())
    assert result["labor_cost_usd"] == "3000.00"
    assert result["direct_cost_usd"] == "3300.00"
    assert result["contribution_usd"] == "1700.00"
    assert result["contribution_margin_percent"] == "34.00"
    assert result["monthly_capacity_pilots"] == 4
    assert result["break_even_pilots"] == 2
    assert result["break_even_within_capacity"] is True
    assert result["monthly_result_at_capacity_usd"] == "3800.00"


@pytest.mark.parametrize("fee", ["3300", "3000"])
def test_zero_or_negative_contribution_has_no_profitable_break_even(fee: str) -> None:
    result = estimate(replace(scenario(), fee_usd=Decimal(fee)))
    assert result["break_even_pilots"] is None
    assert result["break_even_within_capacity"] is False


def test_positive_unit_economics_can_still_exceed_delivery_capacity() -> None:
    result = estimate(replace(scenario(), monthly_available_hours=Decimal(59)))
    assert result["monthly_capacity_pilots"] == 1
    assert result["break_even_pilots"] == 2
    assert result["break_even_within_capacity"] is False
    assert result["monthly_result_at_capacity_usd"] == "-1300.00"


def test_fractional_dollar_inputs_remain_exact_until_display() -> None:
    result = estimate(replace(scenario(), loaded_hourly_cost_usd=Decimal("100.01")))
    assert result["direct_cost_usd"] == "3300.30"
    assert result["contribution_usd"] == "1699.70"


@pytest.mark.parametrize("value", [Decimal("NaN"), Decimal("Infinity"), Decimal(-1), True, 1.0])
def test_invalid_costs_do_not_produce_plausible_estimates(value: object) -> None:
    with pytest.raises(ValueError, match="finite Decimal"):
        replace(scenario(), compute_usd=value)


def test_free_offer_and_zero_effort_are_rejected() -> None:
    with pytest.raises(ValueError, match="fee_usd"):
        replace(scenario(), fee_usd=Decimal(0))
    with pytest.raises(ValueError, match="hours"):
        replace(scenario(), delivery_hours=Decimal(0), support_hours=Decimal(0))


@pytest.mark.parametrize("value", ["1e-100000", "0e+100000", "1000000001"])
def test_extreme_precision_and_magnitude_are_rejected(value: str) -> None:
    with pytest.raises(ValueError, match="finite Decimal"):
        replace(scenario(), fee_usd=Decimal(value))


def test_cli_requires_explicit_inputs_and_rejects_non_finite_values(capsys) -> None:
    args = []
    for name in scenario().__dataclass_fields__:
        args.extend(["--" + name.replace("_", "-"), str(getattr(scenario(), name))])
    assert main(args) == 0
    assert '"contribution_usd": "1700.00"' in capsys.readouterr().out
    args[1] = "NaN"
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 2
