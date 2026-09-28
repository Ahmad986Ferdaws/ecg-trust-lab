from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import yaml

from scripts.summarize_test_results import main, summarize

ROOT = Path(__file__).resolve().parents[2]


def test_real_pytest_failure_and_skips_produce_only_outcome_metadata(tmp_path: Path) -> None:
    tests = tmp_path / "test_sample.py"
    tests.write_text(
        """import pytest

def test_pass():
    print('PRIVATE_STDOUT_SENTINEL')

@pytest.mark.parametrize('record', ['PRIVATE_PARAMETER_SENTINEL'])
def test_failure(record):
    assert False, 'PRIVATE_ASSERTION_SENTINEL'

@pytest.mark.skip(reason='PRIVATE_SKIP_SENTINEL')
def test_skip():
    pass

@pytest.mark.xfail(reason='PRIVATE_XFAIL_SENTINEL')
def test_xfail():
    assert False

@pytest.fixture
def broken():
    raise RuntimeError('PRIVATE_FIXTURE_SENTINEL')

def test_error(broken):
    pass
""",
        encoding="utf-8",
    )
    report = tmp_path / "cpu-tests-junit.xml"
    workflow = yaml.safe_load((ROOT / ".github/workflows/cpu-quality.yml").read_text())
    test_step = next(
        step for step in workflow["jobs"]["quality"]["steps"] if step.get("id") == "tests"
    )
    # Execute the exact workflow test arguments against a small isolated suite.
    command = shlex.split(test_step["run"].replace("${{ runner.temp }}", str(tmp_path)))
    assert command[:4] == ["uv", "run", "--no-sync", "pytest"]
    assert not test_step.get("continue-on-error", False)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", *command[4:], "-c", os.devnull, str(tests)],
        cwd=tmp_path,
        env={**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "PRIVATE_ASSERTION_SENTINEL" in report.read_text()

    summary = tmp_path / "summary.md"
    summary.write_text("Existing summary\n", encoding="utf-8")
    out = tmp_path / "metadata"
    assert (
        main(
            [
                "--junit",
                str(report),
                "--output-dir",
                str(out),
                "--summary",
                str(summary),
                "--test-outcome",
                "failure",
            ]
        )
        == 0
    )
    expected = {"tests": 5, "passed": 1, "failures": 1, "errors": 1, "skipped": 2}
    assert json.loads((out / "test-metadata.json").read_text()) == {
        "report_available": True,
        "test_step_outcome": "failure",
        "counts": expected,
        "skip_categories": {"private artifacts": 0, "platform": 0, "GPU": 0, "other": 2},
    }
    assert summarize(out / "junit-metadata.xml")[0] == expected
    public = summary.read_text() + "".join(path.read_text() for path in out.iterdir())
    for private in ("PRIVATE_", "test_failure", "test_sample", str(tmp_path), "RuntimeError"):
        assert private not in public
    assert summary.read_text().startswith("Existing summary\n")
    assert "| Skipped | 2 |" in public
    assert "Test step: **failure**" in public


def test_report_drops_all_source_attributes_properties_and_text(tmp_path: Path) -> None:
    report = tmp_path / "source.xml"
    report.write_text(
        '<testsuites secret="PRIVATE"><testsuite name="PRIVATE" hostname="PRIVATE">'
        '<properties><property name="PRIVATE" value="PRIVATE"/></properties>'
        '<testcase name="PRIVATE" classname="PRIVATE" file="PRIVATE" time="PRIVATE">'
        '<properties><property name="PRIVATE" value="PRIVATE"/></properties>'
        '<failure type="PRIVATE" message="PRIVATE">PRIVATE</failure>'
        "<system-out>PRIVATE</system-out><system-err>PRIVATE</system-err>"
        "</testcase></testsuite></testsuites>",
        encoding="utf-8",
    )
    counts, clean = summarize(report)
    assert counts == {"tests": 1, "passed": 0, "failures": 1, "errors": 0, "skipped": 0}
    assert "PRIVATE" not in ET.tostring(clean, encoding="unicode")
    assert clean.find("testcase").attrib == {"name": "case-000001"}


@pytest.mark.parametrize(
    "content",
    [None, "<broken PRIVATE", "<PRIVATE />", '<testsuite tests="3"/>'],
)
def test_unavailable_report_is_explicit_and_never_reuses_old_results(
    tmp_path: Path, content: str | None, capsys: pytest.CaptureFixture[str]
) -> None:
    report = tmp_path / "PRIVATE.xml"
    if content is not None:
        report.write_text(content, encoding="utf-8")
    out = tmp_path / "metadata"
    out.mkdir()
    (out / "junit-metadata.xml").write_text("stale results", encoding="utf-8")
    summary = tmp_path / "summary.md"
    assert (
        main(
            [
                "--junit",
                str(report),
                "--output-dir",
                str(out),
                "--summary",
                str(summary),
                "--test-outcome",
                "failure",
            ]
        )
        == 1
    )
    assert json.loads((out / "test-metadata.json").read_text()) == {
        "report_available": False,
        "test_step_outcome": "failure",
        "counts": None,
        "skip_categories": None,
    }
    assert not (out / "junit-metadata.xml").exists()
    assert "counts are unknown" in summary.read_text()
    assert capsys.readouterr().err == "error: JUnit results are unavailable or invalid\n"


def test_empty_report_has_zero_outcomes(tmp_path: Path) -> None:
    report = tmp_path / "source.xml"
    report.write_text('<testsuites><testsuite tests="0"/></testsuites>', encoding="utf-8")
    counts, _ = summarize(report)
    assert counts == {"tests": 0, "passed": 0, "failures": 0, "errors": 0, "skipped": 0}


def test_only_tracked_test_identifiers_survive_without_parameter_values(tmp_path: Path) -> None:
    report = tmp_path / "source.xml"
    report.write_text(
        '<testsuite><testcase classname="tests.unit.test_benchmark" '
        'name="test_percentile_uses_linear_interpolation[PRIVATE_PARAMETER]">'
        '<failure message="PRIVATE_FAILURE"/></testcase>'
        '<testcase classname="tests.unit.test_benchmark" name="test_PRIVATE_FUNCTION"/>'
        '<testcase classname="tests.unit.test_PRIVATE_MODULE" name="test_fn"/>'
        '<testcase classname="tests.unit.test_benchmark" name="../PRIVATE"/>'
        "</testsuite>",
        encoding="utf-8",
    )
    _, clean = summarize(report)
    cases = list(clean)
    assert cases[0].attrib == {
        "classname": "tests.unit.test_benchmark",
        "name": "test_percentile_uses_linear_interpolation",
    }
    assert [case.attrib for case in cases[1:]] == [
        {"name": f"case-{index:06d}"} for index in range(2, 5)
    ]
    assert "PRIVATE" not in ET.tostring(clean, encoding="unicode")


def test_skip_summary_uses_only_finite_categories(tmp_path: Path) -> None:
    report = tmp_path / "source.xml"
    root = ET.Element("testsuite")
    for reason in (
        "private frozen normalization is not available in this checkout PRIVATE_PATH",
        "exact frozen Windows process contract PRIVATE_PATH",
        "frozen CUDA host unavailable PRIVATE_PATH",
        "PRIVATE_PATH",
    ):
        ET.SubElement(ET.SubElement(root, "testcase"), "skipped", message=reason)
    ET.ElementTree(root).write(report)
    out, summary = tmp_path / "metadata", tmp_path / "summary.md"
    assert (
        main(
            [
                "--junit",
                str(report),
                "--output-dir",
                str(out),
                "--summary",
                str(summary),
                "--test-outcome",
                "success",
            ]
        )
        == 0
    )
    metadata = json.loads((out / "test-metadata.json").read_text())
    assert metadata["skip_categories"] == {
        "private artifacts": 1,
        "platform": 1,
        "GPU": 1,
        "other": 1,
    }
    assert metadata["test_step_outcome"] == "success"
    public = summary.read_text() + (out / "junit-metadata.xml").read_text()
    assert "PRIVATE_PATH" not in public
    assert "Skip categories: private artifacts: 1, platform: 1, GPU: 1, other: 1." in public


def test_workflow_uploads_only_sanitized_files_after_failed_tests() -> None:
    workflow = yaml.safe_load((ROOT / ".github/workflows/cpu-quality.yml").read_text())
    steps = workflow["jobs"]["quality"]["steps"]
    summary = next(step for step in steps if step.get("id") == "test_report")
    assert "!cancelled()" in summary["if"]
    assert "steps.tests.outcome != 'skipped'" in summary["if"]
    upload = next(
        step for step in steps if step.get("uses", "").startswith("actions/upload-artifact@")
    )
    assert "!cancelled()" in upload["if"]
    assert "steps.test_report.outcome != 'skipped'" in upload["if"]
    assert upload["with"]["path"].splitlines() == [
        "${{ runner.temp }}/cpu-test-metadata/test-metadata.json",
        "${{ runner.temp }}/cpu-test-metadata/junit-metadata.xml",
    ]
    assert upload["with"]["retention-days"] == 7
    assert upload["with"].get("include-hidden-files", False) is False
