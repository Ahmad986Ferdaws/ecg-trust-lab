from __future__ import annotations

import json

import numpy as np
import pytest

from scripts import quality_gate_stress_report as report


def test_v2_closes_every_v1_fail_open_case_and_passes_the_clean_signal() -> None:
    rows = {row["case_id"]: row for row in report.evaluate()}

    v1_fail_open = {case for case, row in rows.items() if row["v1"]["fail_open"]}  # type: ignore[index]
    v2_fail_open = {case for case, row in rows.items() if row["v2"]["fail_open"]}  # type: ignore[index]

    assert v1_fail_open == {
        "v2_flat_0_6s",
        "v2_flat_3s",
        "v2_flat_7s",
        "all_flat_5s",
        "v5_copies_v3",
    }
    assert v2_fail_open == set()
    for preset in ("v1", "v2"):
        assert rows["clean"][preset]["classification_allowed"] is True  # type: ignore[index]


def test_synthetic_source_is_deterministic_and_has_no_identical_leads() -> None:
    first = report.synthetic_ecg()
    second = report.synthetic_ecg()

    np.testing.assert_array_equal(first, second)
    differences = [
        float(np.max(np.abs(first[i] - first[j]))) for i in range(12) for j in range(i + 1, 12)
    ]
    assert min(differences) > 0.01


@pytest.mark.parametrize("output_format", ["markdown", "json"])
def test_cli_renders_both_formats(output_format: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert report.main(["--format", output_format]) == 0
    output = capsys.readouterr().out

    if output_format == "json":
        payload = json.loads(output)
        assert payload["presets"] == ["v1", "v2"]
        assert len(payload["cases"]) == len(report.CASES)
    else:
        assert output.startswith("| Case | Should refuse | v1 | v2 |")
        assert output.count("(fail-open)") == 5
