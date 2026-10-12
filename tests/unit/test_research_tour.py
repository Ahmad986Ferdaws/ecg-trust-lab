from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_research_tour as tour


@pytest.fixture(scope="module")
def report():
    return tour.build_report()


def test_tour_preserves_unfavorable_results_and_interpretation_limits(report) -> None:
    assert report["status"] == "examples_completed"
    assert "not clinical validation" in report["notice"]
    assert report["quality"]["source"] == "deterministic_synthetic_ecg"
    cases = report["quality"]["cases"]
    assert len(cases) == 9
    assert sum(row["v1"]["fail_open"] for row in cases) == 5
    assert sum(row["v2"]["fail_open"] for row in cases) == 0
    assert report["abstention"]["records"] == 2000
    assert report["abstention"]["seed"] == 0
    assert report["abstention"]["source"] == "synthetic demo (not model output)"
    assert set(report["abstention"]["losses"]) == {
        "mean_binary_log_loss",
        "thresholded_hamming_loss",
    }
    planning = report["gate_planning"]
    assert planning["sample_size"] == 200
    assert planning["true_rate"] == 0.02
    assert planning["rate_model"] == "binomial"
    assert 0.0 < planning["pass_probability"] < 1.0
    assert "clustering is not modeled" in planning["notice"]
    assert report["environment"]["python"]
    assert report["environment"]["packages"]["numpy"]
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("format_name", ["json", "markdown"])
def test_tour_cli_keeps_context_with_the_results(report, format_name, capsys, monkeypatch) -> None:
    monkeypatch.setattr(tour, "build_report", lambda: report)
    assert tour.main(["--format", format_name]) == 0
    output = capsys.readouterr().out
    if format_name == "json":
        assert json.loads(output) == report
    else:
        assert "(fail-open)" in output
        assert "canonical-12x1000-mv-v1" in output
        assert "canonical-12x1000-mv-v2" in output
        assert "2000 constructed records; seed 0" in output
        assert "patient clustering is not modeled" in output
        assert "not a software attestation" in output
        assert "one-sided confidence=0.95" in output


def test_component_failure_emits_no_partial_or_sensitive_report(monkeypatch, capsys) -> None:
    def fail():
        raise RuntimeError("/private/patient-folder secret=do-not-leak")

    monkeypatch.setattr(tour, "quality_report", fail)
    assert tour.main([]) == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == "error: synthetic tour failed; no report was produced\n"


def test_source_context_omits_paths_and_changed_filenames(monkeypatch) -> None:
    responses = iter(["a" * 40 + "\n", "?? private-customer-list.csv\n"])
    monkeypatch.setattr(
        tour.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout=next(responses))
    )
    assert tour._source_context(Path("/private/checkout")) == {
        "revision": "a" * 40,
        "has_uncommitted_changes": True,
    }


@pytest.mark.parametrize("error", [FileNotFoundError(), subprocess.TimeoutExpired("git", 5)])
def test_missing_checkout_context_remains_explicitly_unknown(monkeypatch, error) -> None:
    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(tour.subprocess, "run", fail)
    assert tour._source_context(Path(".")) == {
        "revision": None,
        "has_uncommitted_changes": None,
    }


def test_module_entrypoint_runs_without_private_inputs() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "scripts.run_research_tour", "--format", "json"],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    report = json.loads(result.stdout)
    assert report["report_type"] == "ecg_trust.synthetic_research_tour"
    assert report["quality"]["cases"][0]["case_id"] == "clean"
    assert result.stderr == ""


def test_tour_refuses_to_label_another_installation_as_this_checkout(monkeypatch, capsys) -> None:
    def should_not_run():
        raise AssertionError("examples must not run with the wrong package")

    monkeypatch.setattr(tour.ecg_trust, "__file__", "/private/other-package/ecg_trust/__init__.py")
    monkeypatch.setattr(tour, "quality_report", should_not_run)
    assert tour.main([]) == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert "install this checkout in editable mode" in output.err
    assert "/private" not in output.err
