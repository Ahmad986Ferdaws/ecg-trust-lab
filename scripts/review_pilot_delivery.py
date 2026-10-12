"""Compare a saved pilot estimate with a bounded, user-entered delivery record.

Run from the source checkout with ``python -m scripts.review_pilot_delivery``.
This is an arithmetic review, not verified revenue, customer acceptance or a forecast.
"""

from __future__ import annotations

import argparse
import json
from decimal import ROUND_HALF_EVEN, Decimal, InvalidOperation, localcontext
from pathlib import Path
from typing import Any

from scripts.estimate_research_pilot import PilotInputs, estimate

MAX_INPUT_BYTES = 65_536
MAX_WORK_ENTRIES = 250
CATEGORIES = ("setup", "engineering", "review", "handover", "support")
ACTUAL_KEYS = frozenset(
    {
        "schema_version",
        "basis",
        "support_window_closed",
        "fee_usd",
        "work",
        "compute_usd",
        "other_direct_usd",
    }
)
WORK_KEYS = frozenset({"category", "hours", "loaded_hourly_cost_usd"})


def _number(value: object) -> Decimal:
    if not isinstance(value, str) or len(value) > 32:
        raise ValueError("amounts and hours must be bounded decimal strings")
    try:
        parsed = Decimal(value)
    except InvalidOperation as error:
        raise ValueError("amounts and hours must be finite non-negative decimals") from error
    if (
        not parsed.is_finite()
        or not Decimal(0) <= parsed <= Decimal("1000000000")
        or not -4 <= int(parsed.as_tuple().exponent) <= 9
    ):
        raise ValueError("amounts and hours must be in [0, 1000000000] with at most four decimals")
    return parsed


def _money(value: Decimal) -> str:
    return format(value.quantize(Decimal("0.01"), rounding=ROUND_HALF_EVEN), "f")


def _percent(numerator: Decimal, denominator: Decimal) -> str | None:
    return _money(numerator / denominator * 100) if denominator else None


def _planned_inputs(document: object) -> PilotInputs:
    if not isinstance(document, dict) or not isinstance(document.get("inputs"), dict):
        raise ValueError("plan must be the complete JSON output of estimate_research_pilot")
    values = document["inputs"]
    if set(values) != set(PilotInputs.__dataclass_fields__):
        raise ValueError("plan input fields do not match the pilot estimate contract")
    inputs = PilotInputs(**{name: _number(values[name]) for name in values})
    # Recompute all derived fields; a stale or edited display must not be trusted.
    expected = estimate(inputs)
    if json.dumps(document, sort_keys=True, allow_nan=False) != json.dumps(
        expected, sort_keys=True
    ):
        raise ValueError("plan does not match its recomputed estimate")
    return inputs


def review(plan: object, actuals: object) -> dict[str, Any]:
    """Return reproducible cost/effort variances without assigning acceptance."""
    # The entry count and decimal bounds keep every intermediate below 50 digits.
    with localcontext() as context:
        context.prec = 50
        context.rounding = ROUND_HALF_EVEN
        inputs = _planned_inputs(plan)
        if not isinstance(actuals, dict) or set(actuals) != ACTUAL_KEYS:
            raise ValueError("delivery record has missing or unknown fields")
        if actuals["schema_version"] != "ecg_trust.pilot_actuals.v1":
            raise ValueError("unsupported delivery-record schema")
        if actuals["basis"] not in ("synthetic_example", "operator_record"):
            raise ValueError("basis must be synthetic_example or operator_record")
        if type(actuals["support_window_closed"]) is not bool:
            raise ValueError("support_window_closed must be a boolean")
        work = actuals["work"]
        if not isinstance(work, list) or not 1 <= len(work) <= MAX_WORK_ENTRIES:
            raise ValueError("work must contain between 1 and 250 entries")
        hours_by_category = dict.fromkeys(CATEGORIES, Decimal(0))
        labor_by_category = dict.fromkeys(CATEGORIES, Decimal(0))
        for entry in work:
            if not isinstance(entry, dict) or set(entry) != WORK_KEYS:
                raise ValueError("work entry has missing or unknown fields")
            category = entry["category"]
            if not isinstance(category, str) or category not in CATEGORIES:
                raise ValueError(
                    "work category must be setup, engineering, review, handover or support"
                )
            hours, rate = _number(entry["hours"]), _number(entry["loaded_hourly_cost_usd"])
            hours_by_category[category] += hours
            labor_by_category[category] += hours * rate
        fee = _number(actuals["fee_usd"])
        compute, other = _number(actuals["compute_usd"]), _number(actuals["other_direct_usd"])
        hours = sum(hours_by_category.values(), Decimal(0))
        labor = sum(labor_by_category.values(), Decimal(0))
        direct = labor + compute + other
        contribution = fee - direct
        planned_hours = inputs.delivery_hours + inputs.support_hours
        planned_direct = (
            planned_hours * inputs.loaded_hourly_cost_usd
            + inputs.compute_usd
            + inputs.other_direct_usd
        )
        planned_contribution = inputs.fee_usd - planned_direct
        return {
            "schema_version": "ecg_trust.pilot_delivery_review.v1",
            "currency": "USD",
            "basis": actuals["basis"],
            "status": "recorded_window_closed"
            if actuals["support_window_closed"]
            else "provisional",
            "notice": (
                "User-entered records; not independently verified payments, revenue or customer "
                "acceptance. An open support window leaves delivery costs incomplete. "
                "Contribution excludes fixed overhead, taxes and omitted costs."
            ),
            "inputs": {
                "planned": {
                    name: format(getattr(inputs, name), "f") for name in inputs.__dataclass_fields__
                },
                "recorded": json.loads(json.dumps(actuals)),
            },
            "planned": {
                "total_hours": format(planned_hours, "f"),
                "direct_cost_usd": _money(planned_direct),
                "contribution_usd": _money(planned_contribution),
            },
            "recorded": {
                "total_hours": format(hours, "f"),
                "labor_cost_usd": _money(labor),
                "direct_cost_usd": _money(direct),
                "contribution_usd": _money(contribution),
                "contribution_margin_percent": _percent(contribution, fee),
                "by_category": {
                    category: {
                        "hours": format(hours_by_category[category], "f"),
                        "labor_cost_usd": _money(labor_by_category[category]),
                    }
                    for category in CATEGORIES
                },
            },
            "variance": {
                "definition": "recorded minus planned; positive cost/effort variance is an overrun",
                "hours": format(hours - planned_hours, "f"),
                "hours_percent": _percent(hours - planned_hours, planned_hours),
                "direct_cost_usd": _money(direct - planned_direct),
                "direct_cost_percent": _percent(direct - planned_direct, planned_direct),
                "contribution_usd": _money(contribution - planned_contribution),
            },
        }


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON keys are not allowed")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    del value
    raise ValueError("non-finite JSON numbers are not allowed")


def _read_json(path: Path) -> object:
    try:
        if not path.is_file():
            raise ValueError("input must be a readable regular JSON file")
        with path.open("rb") as source:
            data = source.read(MAX_INPUT_BYTES + 1)
        if len(data) > MAX_INPUT_BYTES:
            raise ValueError("each input file must be at most 65536 bytes")
        return json.loads(data, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError("input must be a readable, bounded UTF JSON document") from error


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--actuals", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = review(_read_json(args.plan), _read_json(args.actuals))
    except (ValueError, TypeError, RecursionError):
        # Never echo private paths, input values or parser excerpts.
        parser.error(
            "invalid pilot inputs; check schemas, decimal bounds and the original plan export"
        )
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
