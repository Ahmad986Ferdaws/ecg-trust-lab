from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from ecg_trust.sweep_config import SweepConfigError
from ecg_trust.sweep_runner import (
    ArchitectureSweepResult,
    BestCandidateResult,
    EqualBudgetSweepResult,
    SweepPreflightResult,
)


def _load_sweep_cli() -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "scripts" / "sweep.py"
    specification = importlib.util.spec_from_file_location("sweep_cli_under_test", path)
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


sweep_cli = _load_sweep_cli()


def _preflight(tmp_path: Path, *, persisted_plan: bool = False) -> SweepPreflightResult:
    plan_path = tmp_path / "candidate_plan.json"
    if persisted_plan:
        plan_path.write_text("{}\n", encoding="utf-8")
    return SweepPreflightResult(
        comparison_id="comparison-v2",
        candidate_plan_hash="sha256:" + "a" * 64,
        candidate_plan_path=plan_path,
        storage_path=tmp_path / "optuna.sqlite3",
        storage_exists=persisted_plan,
        existing_study_names=("resnet-study",) if persisted_plan else (),
        source_provenance={"source_tree_sha256": "sha256:" + "b" * 64},
    )


def _result(tmp_path: Path) -> EqualBudgetSweepResult:
    studies: list[ArchitectureSweepResult] = []
    best_by_architecture: dict[str, BestCandidateResult] = {}
    for architecture, score in (("resnet1d", 0.92), ("ecg_transformer", 0.91)):
        best = BestCandidateResult(
            architecture=architecture,
            candidate_index=3,
            trial_number=4,
            best_macro_auroc=score,
            best_epoch=6,
            completed_epochs=10,
            run_dir=tmp_path / architecture / "candidate0003-attempt01-seed2026",
            parameters={"batch_size": 128, "learning_rate": 0.001},
            experiment_config_hash="sha256:" + "c" * 64,
            resolved_config_hash="sha256:" + "d" * 64,
        )
        studies.append(
            ArchitectureSweepResult(
                architecture=architecture,
                study_name=f"{architecture}-study",
                completed_candidates=12,
                failed_attempts=1,
                total_attempts=13,
                budget_complete=True,
                study_summary_path=tmp_path / f"{architecture}-study.json",
                best=best,
            )
        )
        best_by_architecture[architecture] = best
    return EqualBudgetSweepResult(
        comparison_id="comparison-v2",
        candidate_plan_path=tmp_path / "candidate_plan.json",
        summary_path=tmp_path / "sweep_summary.json",
        studies=(studies[0], studies[1]),
        best_by_architecture=best_by_architecture,
    )


def _patch_loaders(monkeypatch: pytest.MonkeyPatch) -> tuple[object, object]:
    pair = object()
    protocol = object()
    monkeypatch.setattr(sweep_cli, "load_equal_budget_pair", lambda *args, **kwargs: pair)
    monkeypatch.setattr(sweep_cli, "load_protocol", lambda *args, **kwargs: protocol)
    return pair, protocol


def test_default_action_is_read_only_preflight() -> None:
    args = sweep_cli.parse_args([])

    assert args.action == "preflight"
    assert args.resume is False
    assert len(args.configs) == 2
    assert args.protocol.name == "protocol.yaml"


def test_resume_is_rejected_for_read_only_actions() -> None:
    with pytest.raises(SystemExit) as error:
        sweep_cli.parse_args(["status", "--resume"])

    assert error.value.code == 2


def test_preflight_loads_the_pair_once_and_prints_machine_readable_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pair = object()
    protocol = object()
    calls: dict[str, Any] = {}

    def load_pair(paths: tuple[Path, Path], *, base_dir: Path) -> object:
        calls["paths"] = tuple(paths)
        calls["base_dir"] = base_dir
        return pair

    monkeypatch.setattr(sweep_cli, "load_equal_budget_pair", load_pair)
    monkeypatch.setattr(sweep_cli, "load_protocol", lambda path: protocol)
    monkeypatch.setattr(
        sweep_cli,
        "preflight_equal_budget_sweeps",
        lambda observed_pair, *, protocol: _preflight(tmp_path),
    )

    exit_code = sweep_cli.main(
        ["preflight", "--configs", "resnet.yaml", "transformer.yaml"]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert calls["paths"] == (Path("resnet.yaml"), Path("transformer.yaml"))
    assert calls["base_dir"] == Path(sweep_cli.__file__).resolve().parents[1]
    assert payload["action"] == "preflight"
    assert payload["comparison_id"] == "comparison-v2"
    assert payload["existing_study_names"] == []


def test_status_is_read_only_and_forwards_the_validated_pair(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pair, protocol = _patch_loaders(monkeypatch)
    calls: list[tuple[object, object]] = []

    def read_status(observed_pair: object, *, protocol: object) -> dict[str, object]:
        calls.append((observed_pair, protocol))
        return {"comparison_id": "comparison-v2", "studies": {}}

    monkeypatch.setattr(sweep_cli, "read_sweep_status", read_status)

    assert sweep_cli.main(["status"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert calls == [(pair, protocol)]
    assert payload == {
        "action": "status",
        "comparison_id": "comparison-v2",
        "studies": {},
    }


@pytest.mark.parametrize("resume", [False, True])
def test_run_surfaces_preflight_counts_and_selected_candidates(
    resume: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pair, protocol = _patch_loaders(monkeypatch)
    preflight = _preflight(tmp_path, persisted_plan=resume)
    observed: dict[str, object] = {}
    monkeypatch.setattr(
        sweep_cli,
        "preflight_equal_budget_sweeps",
        lambda observed_pair, *, protocol: preflight,
    )

    def run(
        observed_pair: object,
        *,
        protocol: object,
        resume: bool,
    ) -> EqualBudgetSweepResult:
        observed.update(pair=observed_pair, protocol=protocol, resume=resume)
        return _result(tmp_path)

    monkeypatch.setattr(sweep_cli, "run_equal_budget_sweeps", run)
    arguments = ["run", "--resume"] if resume else ["run"]

    assert sweep_cli.main(arguments) == 0

    payload = json.loads(capsys.readouterr().out)
    assert observed == {"pair": pair, "protocol": protocol, "resume": resume}
    assert payload["mode"] == ("resume" if resume else "fresh")
    assert payload["preflight"]["candidate_plan_hash"] == preflight.candidate_plan_hash
    assert len(payload["studies"]) == 2
    assert payload["studies"][0]["completed_candidates"] == 12
    assert payload["studies"][0]["failed_attempts"] == 1
    assert payload["studies"][0]["best"]["candidate_index"] == 3
    assert payload["studies"][0]["best"]["best_fold8_macro_auroc"] == 0.92


def test_resume_requires_the_persisted_candidate_plan_before_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _patch_loaders(monkeypatch)
    monkeypatch.setattr(
        sweep_cli,
        "preflight_equal_budget_sweeps",
        lambda observed_pair, *, protocol: _preflight(tmp_path),
    )
    monkeypatch.setattr(
        sweep_cli,
        "run_equal_budget_sweeps",
        lambda *args, **kwargs: pytest.fail("runner must not start"),
    )

    assert sweep_cli.main(["run", "--resume"]) == 1

    assert "requires the existing immutable candidate-plan" in capsys.readouterr().err


def test_known_validation_error_returns_nonzero_without_training(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        sweep_cli,
        "load_equal_budget_pair",
        lambda *args, **kwargs: (_ for _ in ()).throw(SweepConfigError("unsafe pair")),
    )

    assert sweep_cli.main(["run"]) == 1

    assert capsys.readouterr().err.strip() == "error: unsafe pair"
