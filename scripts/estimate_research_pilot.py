"""Estimate research-pilot delivery economics from explicit hypothetical inputs.

This calculator uses no patient data and makes no pricing or demand prediction.
Money is denominated in USD; calculations retain Decimal precision until display.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal, InvalidOperation


def _number(value: str) -> Decimal:
    if len(value) > 32:
        raise argparse.ArgumentTypeError("decimal input must be at most 32 characters")
    try:
        parsed = Decimal(value)
    except InvalidOperation as error:
        raise argparse.ArgumentTypeError("must be a finite non-negative decimal") from error
    if not _valid_decimal(parsed):
        raise argparse.ArgumentTypeError(
            "must be in [0, 1,000,000,000] with at most four decimal places"
        )
    return parsed


def _valid_decimal(value: object) -> bool:
    return (
        isinstance(value, Decimal)
        and value.is_finite()
        and Decimal(0) <= value <= Decimal("1000000000")
        and -4 <= int(value.as_tuple().exponent) <= 9
    )


@dataclass(frozen=True, slots=True)
class PilotInputs:
    """One pilot's delivery costs and the capacity of the same delivery team."""

    fee_usd: Decimal
    delivery_hours: Decimal
    support_hours: Decimal
    loaded_hourly_cost_usd: Decimal
    compute_usd: Decimal
    other_direct_usd: Decimal
    monthly_fixed_usd: Decimal
    monthly_available_hours: Decimal

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if not _valid_decimal(value):
                raise ValueError(
                    f"{name} must be a finite Decimal in [0, 1,000,000,000] "
                    "with at most four decimal places"
                )
        if self.fee_usd <= 0:
            raise ValueError("fee_usd must be positive")
        if self.delivery_hours + self.support_hours <= 0:
            raise ValueError("total delivery and support hours must be positive")


def estimate(inputs: PilotInputs) -> dict[str, object]:
    """Return unit contribution and a capacity-constrained monthly scenario.

    Fixed overhead must exclude labor already counted in the loaded hourly cost.
    Capacity assumes identical scopes, no other work, and enough demand; it is
    not a sales forecast. A non-positive unit contribution has no profitable
    break-even volume in this model.
    """

    hours = inputs.delivery_hours + inputs.support_hours
    labor = hours * inputs.loaded_hourly_cost_usd
    direct = labor + inputs.compute_usd + inputs.other_direct_usd
    contribution = inputs.fee_usd - direct
    capacity = int((inputs.monthly_available_hours / hours).to_integral_value(ROUND_FLOOR))
    break_even = (
        int((inputs.monthly_fixed_usd / contribution).to_integral_value(ROUND_CEILING))
        if contribution > 0
        else None
    )

    def money(value: Decimal) -> str:
        return format(value.quantize(Decimal("0.01")), "f")

    return {
        "schema_version": "ecg_trust.pilot_economics.v1",
        "basis": "user-supplied scenario; not observed revenue, demand or a price quote",
        "currency": "USD",
        "inputs": {
            name: format(getattr(inputs, name), "f") for name in inputs.__dataclass_fields__
        },
        "total_hours": format(hours, "f"),
        "labor_cost_usd": money(labor),
        "direct_cost_usd": money(direct),
        "contribution_usd": money(contribution),
        "contribution_margin_percent": money(contribution / inputs.fee_usd * 100),
        "monthly_capacity_pilots": capacity,
        "break_even_pilots": break_even,
        "break_even_within_capacity": break_even is not None and break_even <= capacity,
        "monthly_result_at_capacity_usd": money(capacity * contribution - inputs.monthly_fixed_usd),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in PilotInputs.__dataclass_fields__:
        parser.add_argument("--" + name.replace("_", "-"), type=_number, required=True)
    args = parser.parse_args(argv)
    try:
        result = estimate(PilotInputs(**vars(args)))
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
