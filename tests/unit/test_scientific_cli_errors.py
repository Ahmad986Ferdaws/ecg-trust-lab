from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from ecg_trust.final_evaluation_spec import FinalEvaluationSpecIntegrityError
from ecg_trust.protocol import FINAL_TEST_CONFIRMATION
from scripts import build_subgroups, freeze_final_evaluation, release_pipeline

ROOT = Path(__file__).resolve().parents[2]
COMMANDS = [
    (build_subgroups, ["--refit-bundle", "missing.json", "--output", "output.json"]),
    (
        freeze_final_evaluation,
        [
            "--refit-bundle",
            "missing.json",
            "--subgroups",
            "missing.json",
            "--device",
            "cuda:0",
            "--output",
            "output.json",
        ],
    ),
    (release_pipeline, ["verify-refits", "--bundle", "missing.json"]),
]


@pytest.mark.parametrize("module,arguments", COMMANDS)
@pytest.mark.parametrize("invalid", ["missing", "malformed"])
def test_expected_protocol_errors_exit_cleanly_without_artifacts(
    tmp_path: Path, module: ModuleType, arguments: list[str], invalid: str
) -> None:
    protocol = tmp_path / "protocol.yaml"
    if invalid == "malformed":
        protocol.write_text("not: [valid yaml", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(module.__file__), *arguments, "--protocol", str(protocol)],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr.startswith("error:")
    assert "Traceback" not in result.stderr
    assert not (tmp_path / "output.json").exists()
    assert not list(tmp_path.rglob(".final-test-openings"))


@pytest.mark.parametrize("module,arguments", COMMANDS)
@pytest.mark.parametrize(
    "error_type", [AssertionError, RuntimeError, KeyboardInterrupt, SystemExit]
)
def test_programming_errors_and_control_flow_are_not_hidden(
    monkeypatch: pytest.MonkeyPatch,
    module: ModuleType,
    arguments: list[str],
    error_type: type[BaseException],
) -> None:
    def fail_protocol(*args: object, **kwargs: object) -> None:
        raise error_type("unexpected failure")

    monkeypatch.setattr(module, "load_protocol", fail_protocol)
    with pytest.raises(error_type):
        module.main(arguments)


def test_release_integrity_failure_cannot_open_final_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def invalid_spec(*args: object, **kwargs: object) -> None:
        raise FinalEvaluationSpecIntegrityError("specification hash differs")

    monkeypatch.setattr(release_pipeline, "_loaded_evaluation_spec", invalid_spec)
    monkeypatch.setattr(
        release_pipeline, "run_final_batch", lambda **kwargs: pytest.fail("final batch opened")
    )
    result = release_pipeline.main(
        [
            "run-final",
            "--protocol",
            str(ROOT / "configs" / "protocol.yaml"),
            "--refit-bundle",
            str(tmp_path / "refits.json"),
            "--calibration-bundle",
            str(tmp_path / "calibration.json"),
            "--evaluation-spec",
            str(tmp_path / "spec.json"),
            "--purpose",
            "synthetic test",
            "--operator",
            "fixture",
            "--confirmation",
            FINAL_TEST_CONFIRMATION,
        ]
    )
    assert result == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "error: specification hash differs\n"
    assert list(tmp_path.iterdir()) == []


def test_successful_readonly_verification_preserves_json_stdout(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    payload = {"schema_version": 1, "fixture": "verified"}
    monkeypatch.setattr(
        release_pipeline,
        "load_refit_bundle",
        lambda *args, **kwargs: SimpleNamespace(to_payload=lambda: payload),
    )
    result = release_pipeline.main(
        [
            "verify-refits",
            "--protocol",
            str(ROOT / "configs" / "protocol.yaml"),
            "--bundle",
            "synthetic.json",
        ]
    )
    assert result == 0
    captured = capsys.readouterr()
    assert captured.out == json.dumps(payload, sort_keys=True) + "\n"
    assert captured.err == ""
