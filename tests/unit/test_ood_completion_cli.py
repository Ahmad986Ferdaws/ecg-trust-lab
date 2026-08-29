from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.prepare_trust_sentinel_ood_completion as cli


def _result() -> SimpleNamespace:
    return SimpleNamespace(
        artifact_sha256="sha256:" + "a" * 64,
        distribution_policy=SimpleNamespace(artifact_sha256="sha256:" + "b" * 64),
        ood_positive_evaluation=SimpleNamespace(status="NOT_EVALUATED"),
        research_bundle_eligible=False,
        source_validation=SimpleNamespace(
            record_false_rejection_rate=0.06,
            source_record_support_coverage=0.94,
            cluster_bootstrap=SimpleNamespace(one_sided_upper=0.08),
        ),
        status="SOURCE_SUPPORT_GATE_TARGET_MISSED",
    )


def test_cli_has_no_scientific_overrides_and_prints_only_aggregate_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = tmp_path / "protocol.yaml"
    config.write_text("frozen: true\n", encoding="utf-8")
    captured: dict[str, object] = {}

    monkeypatch.setattr(cli, "verify_clean_git_revision", lambda root: "1" * 40)

    def fake_prepare(**kwargs: object) -> SimpleNamespace:
        captured.update(kwargs)
        return _result()

    monkeypatch.setattr(cli, "prepare_ood_completion", fake_prepare)

    status = cli.main(
        [
            "--project-root",
            str(tmp_path),
            "--config",
            "protocol.yaml",
        ]
    )
    output = json.loads(capsys.readouterr().out)

    assert status == 0
    assert captured["project_root"] == tmp_path.resolve()
    assert captured["config_path"] == config.resolve()
    assert captured["code_revision"] == "1" * 40
    assert output == {
        "artifact_sha256": "sha256:" + "a" * 64,
        "distribution_policy_sha256": "sha256:" + "b" * 64,
        "ood_positive_evaluation": "NOT_EVALUATED",
        "research_bundle_eligible": False,
        "source_false_rejection_rate": 0.06,
        "source_false_rejection_upper_95": 0.08,
        "source_record_support_coverage": 0.94,
        "status": "SOURCE_SUPPORT_GATE_TARGET_MISSED",
    }
    assert set(vars(cli._parser().parse_args([]))) == {"project_root", "config"}


def test_cli_preserves_lexical_config_symlink_for_pipeline_rejection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    outside_config = tmp_path / "outside.yaml"
    outside_config.write_text("not the frozen protocol\n", encoding="utf-8")
    symbolic_config = project / "protocol.yaml"
    try:
        symbolic_config.symlink_to(outside_config)
    except OSError:
        pytest.skip("creating a symbolic link is not permitted on this Windows host")

    captured: dict[str, object] = {}
    monkeypatch.setattr(cli, "verify_clean_git_revision", lambda root: "1" * 40)

    def reject_in_pipeline(**kwargs: object) -> SimpleNamespace:
        captured.update(kwargs)
        raise RuntimeError("pipeline rejected indirect config")

    monkeypatch.setattr(cli, "prepare_ood_completion", reject_in_pipeline)

    status = cli.main(
        ["--project-root", str(project), "--config", symbolic_config.name]
    )

    expected_project = Path(os.path.abspath(os.fspath(project)))
    expected_config = expected_project / symbolic_config.name
    assert status == 1
    assert captured["project_root"] == expected_project
    assert captured["config_path"] == expected_config
    assert Path(captured["config_path"]).is_symlink()
    assert Path(captured["config_path"]) != outside_config.resolve()
    assert capsys.readouterr().err == (
        "OOD_COMPLETION_FAILED: inspect the private local preflight "
        "and immutable output state.\n"
    )


@pytest.mark.parametrize("config_argument", ["../outside.yaml", "project-sibling/file.yaml"])
def test_cli_rejects_lexically_outside_config_before_git_or_pipeline(
    tmp_path: Path,
    config_argument: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    calls: list[str] = []

    def unexpected_git(_root: Path) -> str:
        calls.append("git")
        return "1" * 40

    def unexpected_pipeline(**_kwargs: object) -> SimpleNamespace:
        calls.append("pipeline")
        return _result()

    monkeypatch.setattr(cli, "verify_clean_git_revision", unexpected_git)
    monkeypatch.setattr(cli, "prepare_ood_completion", unexpected_pipeline)

    if config_argument.startswith("project-sibling"):
        argument = str(tmp_path / config_argument)
    else:
        argument = config_argument
    status = cli.main(["--project-root", str(project), "--config", argument])

    assert status == 1
    assert calls == []
    assert capsys.readouterr().err == (
        "OOD_COMPLETION_FAILED: inspect the private local preflight "
        "and immutable output state.\n"
    )


def test_cli_fails_with_a_sanitized_message(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = tmp_path / "protocol.yaml"
    config.write_text("frozen: true\n", encoding="utf-8")
    monkeypatch.setattr(
        cli,
        "verify_clean_git_revision",
        lambda root: (_ for _ in ()).throw(RuntimeError("private patient path")),
    )

    status = cli.main(
        ["--project-root", str(tmp_path), "--config", "protocol.yaml"]
    )
    captured = capsys.readouterr()

    assert status == 1
    assert captured.out == ""
    assert captured.err == (
        "OOD_COMPLETION_FAILED: inspect the private local preflight and immutable output state.\n"
    )
    assert "patient" not in captured.err
