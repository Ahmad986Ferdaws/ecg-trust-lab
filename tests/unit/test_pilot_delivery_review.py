from __future__ import annotations

import copy
import json
import subprocess
import sys
from dataclasses import replace
from decimal import Decimal, localcontext
from pathlib import Path

import pytest

from scripts.estimate_research_pilot import estimate
from scripts.review_pilot_delivery import MAX_INPUT_BYTES, main, review
from tests.unit.test_pilot_economics import scenario


def actuals() -> dict:
    return {
        "schema_version": "ecg_trust.pilot_actuals.v1",
        "basis": "synthetic_example",
        "support_window_closed": False,
        "fee_usd": "5000",
        "work": [
            {"category": "engineering", "hours": "20", "loaded_hourly_cost_usd": "100"},
            {"category": "review", "hours": "10", "loaded_hourly_cost_usd": "120"},
            {"category": "support", "hours": "10", "loaded_hourly_cost_usd": "100"},
        ],
        "compute_usd": "300",
        "other_direct_usd": "100",
    }


def test_hand_calculated_overrun_keeps_costs_and_provisional_status_visible() -> None:
    result = review(estimate(scenario()), actuals())
    assert result["status"] == "provisional"
    assert result["basis"] == "synthetic_example"
    assert result["recorded"]["total_hours"] == "40"
    assert result["recorded"]["labor_cost_usd"] == "4200.00"
    assert result["recorded"]["direct_cost_usd"] == "4600.00"
    assert result["recorded"]["contribution_usd"] == "400.00"
    assert result["recorded"]["contribution_margin_percent"] == "8.00"
    assert result["variance"] == {
        "definition": "recorded minus planned; positive cost/effort variance is an overrun",
        "hours": "10",
        "hours_percent": "33.33",
        "direct_cost_usd": "1300.00",
        "direct_cost_percent": "39.39",
        "contribution_usd": "-1300.00",
    }
    assert result["recorded"]["by_category"]["review"] == {
        "hours": "10",
        "labor_cost_usd": "1200.00",
    }
    assert result["recorded"]["by_category"]["setup"]["hours"] == "0"


def test_closed_operator_record_is_not_a_customer_signoff() -> None:
    recorded = actuals()
    recorded.update(basis="operator_record", support_window_closed=True, fee_usd="3000")
    result = review(estimate(scenario()), recorded)
    assert result["status"] == "recorded_window_closed"
    assert result["basis"] == "operator_record"
    assert result["recorded"]["contribution_usd"] == "-1600.00"
    assert "not independently verified" in result["notice"]
    assert "acceptance" in result["notice"]


def test_free_pilot_and_zero_planned_cost_do_not_divide_by_zero() -> None:
    plan = estimate(
        replace(
            scenario(),
            loaded_hourly_cost_usd=Decimal(0),
            compute_usd=Decimal(0),
            other_direct_usd=Decimal(0),
        )
    )
    recorded = actuals()
    recorded["fee_usd"] = "0"
    result = review(plan, recorded)
    assert result["recorded"]["contribution_usd"] == "-4600.00"
    assert result["recorded"]["contribution_margin_percent"] is None
    assert result["variance"]["direct_cost_percent"] is None


def test_fractional_labor_is_summed_before_rounding_and_inputs_are_detached() -> None:
    recorded = actuals()
    recorded["work"] = [
        {"category": "review", "hours": "0.5", "loaded_hourly_cost_usd": "0.01"}
    ] * 2
    result = review(estimate(scenario()), recorded)
    assert result["recorded"]["labor_cost_usd"] == "0.01"
    recorded["work"][0]["hours"] = "100"
    assert result["inputs"]["recorded"]["work"][0]["hours"] == "0.5"
    replay = review(estimate(scenario()), result["inputs"]["recorded"])
    assert replay == result


def test_maximum_supported_workload_and_low_ambient_precision_remain_finite() -> None:
    plan = estimate(scenario())
    recorded = actuals()
    recorded["fee_usd"] = "0.0001"
    recorded["work"] = [
        {"category": "engineering", "hours": "1000000000", "loaded_hourly_cost_usd": "1000000000"}
    ] * 250
    with localcontext() as context:
        context.prec = 6
        result = review(plan, recorded)
    assert result["recorded"]["labor_cost_usd"] == "250000000000000000000.00"
    assert result["recorded"]["contribution_margin_percent"] is not None
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(
    "field,value",
    [
        ("contribution_usd", "9000.00"),
        ("schema_version", "unknown"),
        ("currency", "EUR"),
        ("monthly_capacity_pilots", True),
        ("extra", "private"),
    ],
)
def test_edited_or_stale_estimates_are_rejected(field: str, value: object) -> None:
    plan = estimate(scenario())
    plan[field] = value
    with pytest.raises(ValueError, match="recomputed"):
        review(plan, actuals())


@pytest.mark.parametrize("value", [True, 1, 1.5, "NaN", "Infinity", "-1", "1e-10000", "1000000001"])
def test_invalid_recorded_numbers_are_rejected(value: object) -> None:
    recorded = actuals()
    recorded["compute_usd"] = value
    with pytest.raises(ValueError):
        review(estimate(scenario()), recorded)


@pytest.mark.parametrize(
    "change",
    [
        {"support_window_closed": "yes"},
        {"basis": "audited_revenue"},
        {"currency": "EUR"},
        {"contact": "person@example.test"},
        {"work": []},
        {"work": actuals()["work"] * 84},
        {"work": [{"category": "unbounded", "hours": "1", "loaded_hourly_cost_usd": "1"}]},
        {
            "work": [
                {"category": "setup", "hours": "1", "loaded_hourly_cost_usd": "1", "name": "person"}
            ]
        },
    ],
)
def test_closed_record_schema_rejects_unknown_claims_and_fields(change: dict) -> None:
    recorded = actuals()
    recorded.update(copy.deepcopy(change))
    with pytest.raises(ValueError):
        review(estimate(scenario()), recorded)


def inputs_on_disk(tmp_path: Path) -> tuple[Path, Path]:
    plan, record = tmp_path / "plan.json", tmp_path / "actuals.json"
    plan.write_text(json.dumps(estimate(scenario())))
    record.write_text(json.dumps(actuals()))
    return plan, record


def test_module_cli_returns_the_same_review(tmp_path: Path) -> None:
    plan, record = inputs_on_disk(tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.review_pilot_delivery",
            "--plan",
            str(plan),
            "--actuals",
            str(record),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout) == review(estimate(scenario()), actuals())
    assert result.stderr == ""


@pytest.mark.parametrize(
    "content",
    [
        b'{"secret":"private-marker", "secret":"other"}',
        b'{"private-marker":NaN}',
        b"private-marker",
        b"\xffprivate-marker",
        b" " * (MAX_INPUT_BYTES + 1),
        b"[" * 5000 + b"private-marker" + b"]" * 5000,
    ],
)
def test_cli_refuses_malformed_oversized_and_duplicate_json_without_leaking(
    tmp_path: Path,
    capsys,
    content: bytes,
) -> None:
    plan, record = inputs_on_disk(tmp_path)
    record.write_bytes(content)
    with pytest.raises(SystemExit) as error:
        main(["--plan", str(plan), "--actuals", str(record)])
    assert error.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "private-marker" not in captured.err
    assert str(tmp_path) not in captured.err


def test_unreadable_input_is_safe_and_no_files_are_created(tmp_path: Path, capsys) -> None:
    with pytest.raises(SystemExit) as error:
        main(["--plan", str(tmp_path / "private-plan"), "--actuals", str(tmp_path)])
    assert error.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert str(tmp_path) not in captured.err
    assert list(tmp_path.iterdir()) == []
