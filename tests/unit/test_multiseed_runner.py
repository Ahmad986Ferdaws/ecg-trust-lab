from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import cast

import numpy as np
import pytest

import ecg_trust.multiseed_runner as runner
from ecg_trust.experiment_config import DevelopmentExperimentConfig
from ecg_trust.experiment_runner import DevelopmentRunResult
from ecg_trust.predictions import (
    PredictionArtifact,
    create_prediction_artifact,
    save_prediction_artifact,
)
from ecg_trust.protocol import ExperimentProtocol, FoldRole


def _config(tmp_path: Path, architecture: str) -> DevelopmentExperimentConfig:
    (tmp_path / "manifest.parquet").write_bytes(b"synthetic manifest")
    (tmp_path / "normalization.json").write_bytes(b"synthetic normalization")
    return DevelopmentExperimentConfig.from_mapping(
        {
            "schema_version": 1,
            "run_name": f"{architecture}-winner-seed2026",
            "folds": {
                "train": [1, 2, 3, 4, 5, 6, 7],
                "model_selection": [8],
            },
            "data": {
                "manifest": str(tmp_path / "manifest.parquet"),
                "dataset_root": str(tmp_path / "ptbxl"),
                "normalization": str(tmp_path / "normalization.json"),
                "max_train_records": None,
                "max_validation_records": None,
            },
            "model": {"architecture": architecture, "preset": "matched_capacity"},
            "loader": {
                "batch_size": 64,
                "num_workers": 2,
                "pin_memory": True,
                "persistent_workers": True,
            },
            "optimization": {
                "epochs": 30,
                "learning_rate": 0.001,
                "weight_decay": 0.01,
                "warmup_epochs": 2,
                "minimum_lr_ratio": 0.01,
                "gradient_clip_norm": 1.0,
                "early_stopping_patience": 10,
                "early_stopping_min_delta": 0.0001,
                "scheduler": "warmup_cosine",
            },
            "runtime": {"seed": 2026, "device": "cpu", "bf16": False},
            "output": {"root_dir": str(tmp_path / "sweep-runs")},
        },
        base_dir=tmp_path,
    )


def _digest(character: str) -> str:
    return "sha256:" + character * 64


def _fake_source_provenance(tmp_path: Path, *, revision: str) -> dict[str, object]:
    runtime: dict[str, object] = {
        "python": "3.12.0",
        "implementation": "CPython",
        "platform": "synthetic-platform",
        "optuna": "4.0.0",
        "scipy": "1.0.0",
        "torch": "2.0.0",
    }
    return {
        "project_root": str(tmp_path.resolve()),
        "git_root": str(tmp_path.resolve()),
        "git_head": revision,
        "git_dirty": False,
        "git_status_sha256": "sha256:" + hashlib.sha256(b"").hexdigest(),
        "git_unavailable": False,
        "source_tree_sha256": _digest("3"),
        "dependency_lock_sha256": _digest("4"),
        "manifest_path": str((tmp_path / "manifest.parquet").resolve()),
        "manifest_sha256": runner._file_sha256(tmp_path / "manifest.parquet"),
        "normalization_path": str((tmp_path / "normalization.json").resolve()),
        "normalization_sha256": runner._file_sha256(tmp_path / "normalization.json"),
        "runtime_identity": runtime,
        "runtime_identity_sha256": runner._canonical_hash(runtime),
    }


def _fake_revision_provenance(tmp_path: Path) -> dict[str, object]:
    sweep_revision = "1" * 40
    execution_revision = "2" * 40
    sweep_source = _fake_source_provenance(tmp_path, revision=sweep_revision)
    execution_source = dict(sweep_source)
    execution_source["git_head"] = execution_revision
    kernel_paths = list(runner._SCIENTIFIC_KERNEL_PATHS)
    return runner._hashed_payload(
        {
            "schema_version": 1,
            "artifact_type": runner._REVISION_PROVENANCE_TYPE,
            "sweep_snapshot": {
                "git_tree": "3" * 40,
                "source_provenance": sweep_source,
            },
            "execution_snapshot": {
                "git_tree": "4" * 40,
                "source_provenance": execution_source,
            },
            "scientific_kernel": {
                "policy": "ptbxl_development_training_kernel_v1",
                "sweep_revision": sweep_revision,
                "execution_revision": execution_revision,
                "paths": kernel_paths,
                "paths_sha256": runner._canonical_hash({"paths": kernel_paths}),
                "git_diff_sha256": runner._EMPTY_SHA256,
                "unchanged": True,
                "allowed_changed_paths": [],
                "allowed_changed_paths_sha256": runner._canonical_hash({"paths": []}),
            },
        }
    )


def _winner(tmp_path: Path, architecture: str) -> runner.WinnerSource:
    config = _config(tmp_path, architecture)
    return runner.WinnerSource(
        architecture=architecture,
        candidate_index=4 if architecture == "resnet1d" else 7,
        trial_number=5 if architecture == "resnet1d" else 8,
        run_dir=(tmp_path / "sweep-runs" / config.run_name).resolve(),
        experiment_config=config,
        scientific_config_sha256=runner._canonical_hash(
            runner._scientific_payload(config)
        ),
        best_epoch=8,
        best_macro_auroc=0.92,
        completed_epochs=19,
        artifact_sha256={name: _digest("a") for name in runner._SOURCE_FILENAMES},
        manifest_sha256=runner._file_sha256(tmp_path / "manifest.parquet"),
        normalization_sha256=runner._file_sha256(tmp_path / "normalization.json"),
    )


def _patched_summary_loader(
    tmp_path: Path,
) -> tuple[
    Path,
    Callable[..., tuple[Mapping[str, object], dict[str, runner.WinnerSource], str, str]],
]:
    summary_path = tmp_path / "sweep_summary.json"
    summary_path.write_text('{"synthetic":true}\n', encoding="utf-8")
    candidate_path = tmp_path / "candidate_plan.json"
    candidate_path.write_text('{"synthetic":true}\n', encoding="utf-8")
    summary_hash = runner._file_sha256(summary_path)
    candidate_file_hash = runner._file_sha256(candidate_path)
    winners = {
        architecture: _winner(tmp_path, architecture)
        for architecture in runner.ARCHITECTURES
    }
    summary: dict[str, object] = {
        "comparison_id": "paired-v1",
        "candidate_plan_path": str(candidate_path.resolve()),
        "candidate_plan_hash": _digest("d"),
        "source_provenance": _fake_source_provenance(
            tmp_path, revision="1" * 40
        ),
    }

    def load(
        path: Path,
        *,
        protocol: ExperimentProtocol,
    ) -> tuple[Mapping[str, object], dict[str, runner.WinnerSource], str, str]:
        assert path.resolve() == summary_path.resolve()
        assert protocol == ExperimentProtocol.canonical()
        return summary, winners, candidate_file_hash, summary_hash

    return summary_path, load


def _create_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[runner.MultiSeedPlanResult, ExperimentProtocol]:
    protocol = ExperimentProtocol.canonical()
    summary_path, loader = _patched_summary_loader(tmp_path)
    monkeypatch.setattr(runner, "_load_and_verify_sweep_summary", loader)
    revision_provenance = _fake_revision_provenance(tmp_path)
    monkeypatch.setattr(
        runner,
        "_build_revision_provenance",
        lambda summary: revision_provenance,
    )
    result = runner.create_multiseed_plan(
        summary_path,
        output_root=tmp_path / "confirmation",
        protocol=protocol,
    )
    return result, protocol


def _member_payload(path: Path) -> Mapping[str, object]:
    return cast(
        Mapping[str, object],
        json.loads(path.read_text(encoding="utf-8")),
    )


def _fake_verified_run(
    member: Mapping[str, object],
    run_dir: Path,
) -> runner.VerifiedRun:
    paths = {name: run_dir / name for name in runner._SOURCE_FILENAMES}
    hashes = {name: _digest("e") for name in runner._SOURCE_FILENAMES}
    return runner.VerifiedRun(
        run_dir=run_dir,
        run_name=run_dir.name,
        seed=cast(int, member["seed"]),
        resolved_config_hash=_digest("f"),
        manifest_sha256=_digest("b"),
        normalization_sha256=_digest("c"),
        best_epoch=8,
        best_macro_auroc=0.92,
        completed_epochs=19,
        paths=paths,
        hashes=hashes,
    )


def _prediction(
    protocol: ExperimentProtocol,
    *,
    architecture: str,
    seed: int,
) -> PredictionArtifact:
    targets = np.asarray(
        [[(row + label) % 2 for label in range(5)] for row in range(10)],
        dtype=np.int8,
    )
    return create_prediction_artifact(
        ecg_id=np.arange(100, 110),
        patient_id=np.arange(200, 210),
        strat_fold=np.full(10, 8, dtype=np.int8),
        targets=targets,
        raw_logits=np.zeros((10, 5), dtype=np.float64),
        model_name=f"{architecture}-seed{seed}",
        model_seed=seed,
        protocol=protocol,
        config_hash=_digest("f"),
        manifest_hash=_digest("b"),
        fold_role=FoldRole.MODEL_SELECTION,
        created_at_utc="2026-08-08T00:00:00Z",
    )


def test_plan_contains_exact_paired_seed_set_and_is_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, protocol = _create_plan(tmp_path, monkeypatch)
    loaded = runner.load_multiseed_plan(result.plan_path, protocol=protocol)
    members = [
        cast(Mapping[str, object], item) for item in cast(list[object], loaded["members"])
    ]
    identities = [(member["architecture"], member["seed"]) for member in members]

    assert identities == list(runner._execution_order())
    assert sum(member["source_kind"] == "reused_sweep_winner" for member in members) == 2
    assert sum(member["source_kind"] == "confirmation_training" for member in members) == 4
    assert loaded["train_folds"] == [1, 2, 3, 4, 5, 6, 7]
    assert loaded["model_selection_folds"] == [8]
    repeated = runner.create_multiseed_plan(
        tmp_path / "sweep_summary.json",
        output_root=tmp_path / "confirmation",
        protocol=protocol,
    )
    assert repeated.plan_sha256 == result.plan_sha256


def test_member_plan_create_load_round_trip_binds_inputs_and_revisions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, protocol = _create_plan(tmp_path, monkeypatch)
    root = runner.load_multiseed_plan(result.plan_path, protocol=protocol)
    root_revision_hash = cast(str, root["revision_provenance_sha256"])

    for path in result.member_plan_paths:
        raw = _member_payload(path)
        loaded = runner._load_member_plan(
            path,
            expected_hash=cast(str, raw["artifact_sha256"]),
            protocol=protocol,
        )
        template = DevelopmentExperimentConfig.from_mapping(
            cast(Mapping[str, object], loaded["experiment_template"]),
            base_dir=path.parent,
        )
        assert loaded["manifest_sha256"] == runner._file_sha256(
            template.data.manifest_path
        )
        assert loaded["normalization_sha256"] == runner._file_sha256(
            template.data.normalization_path
        )
        assert loaded["sweep_revision"] == "1" * 40
        assert loaded["execution_revision"] == "2" * 40
        assert loaded["revision_provenance_sha256"] == root_revision_hash


def test_member_plan_tampering_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, protocol = _create_plan(tmp_path, monkeypatch)
    path = result.member_plan_paths[0]
    payload = dict(_member_payload(path))
    payload["seed"] = 2027
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(runner.MultiSeedRunnerError, match="SHA-256 mismatch"):
        runner.load_multiseed_plan(result.plan_path, protocol=protocol)


def test_attempt_config_changes_only_seed_run_name_and_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, _ = _create_plan(tmp_path, monkeypatch)
    member = _member_payload(result.member_plan_paths[2])
    template = DevelopmentExperimentConfig.from_mapping(
        cast(Mapping[str, object], member["experiment_template"]),
        base_dir=tmp_path,
    )
    attempt = runner._attempt_config(member, attempt=1)

    assert attempt.runtime.seed == 2027
    assert attempt.run_name == "resnet1d-confirmation-seed2027-attempt01"
    assert attempt.output.root_dir == Path(cast(str, member["attempt_root"]))
    assert runner._scientific_payload(attempt) == runner._scientific_payload(template)
    assert attempt.train_folds == (1, 2, 3, 4, 5, 6, 7)
    assert attempt.validation_folds == (8,)


def test_failed_attempt_retries_the_identical_scientific_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, protocol = _create_plan(tmp_path, monkeypatch)
    member = _member_payload(result.member_plan_paths[2])
    configs: list[DevelopmentExperimentConfig] = []

    def execute(
        config: DevelopmentExperimentConfig,
        *,
        protocol: ExperimentProtocol,
    ) -> DevelopmentRunResult:
        assert protocol == ExperimentProtocol.canonical()
        configs.append(config)
        if len(configs) == 1:
            raise RuntimeError("synthetic interruption")
        run_dir = config.output.root_dir / config.run_name
        run_dir.mkdir(parents=True)
        return DevelopmentRunResult(
            run_dir=run_dir,
            history_path=run_dir / "history.jsonl",
            best_checkpoint_path=run_dir / "best.ckpt",
            last_checkpoint_path=run_dir / "last.ckpt",
            best_epoch=8,
            best_macro_auroc=0.92,
            completed_epochs=19,
            stopped_early=True,
            resolved_config_hash=_digest("f"),
            protocol_hash=protocol.protocol_hash,
            manifest_hash=_digest("b"),
        )

    monkeypatch.setattr(
        runner,
        "_verify_complete_run",
        lambda member, run_dir, **kwargs: _fake_verified_run(member, run_dir),
    )
    verified = runner._train_member(
        member,
        protocol=protocol,
        resume=False,
        experiment_executor=execute,
    )

    assert verified.seed == 2027
    assert [config.run_name for config in configs] == [
        "resnet1d-confirmation-seed2027-attempt00",
        "resnet1d-confirmation-seed2027-attempt01",
    ]
    assert runner._scientific_payload(configs[0]) == runner._scientific_payload(configs[1])
    status0 = runner._load_attempt_status(
        Path(cast(str, member["attempt_root"])) / "attempt00_status.json",
        member=member,
        attempt=0,
    )
    status1 = runner._load_attempt_status(
        Path(cast(str, member["attempt_root"])) / "attempt01_status.json",
        member=member,
        attempt=1,
    )
    assert status0["status"] == "failed"
    assert status1["status"] == "complete"
    assert status0["execution_revision"] == "2" * 40
    assert status1["execution_revision"] == "2" * 40
    attempt_plan = _member_payload(
        Path(cast(str, member["attempt_root"])) / "attempt01_plan.json"
    )
    assert attempt_plan["sweep_revision"] == "1" * 40
    assert attempt_plan["execution_revision"] == "2" * 40


def test_run_dispatches_four_trainings_and_commits_six_prediction_completions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, protocol = _create_plan(tmp_path, monkeypatch)
    trained: list[tuple[str, int]] = []

    def verify(
        member: Mapping[str, object],
        run_dir: Path,
        **kwargs: object,
    ) -> runner.VerifiedRun:
        return _fake_verified_run(member, run_dir)

    def train(
        member: Mapping[str, object],
        **kwargs: object,
    ) -> runner.VerifiedRun:
        identity = (cast(str, member["architecture"]), cast(int, member["seed"]))
        trained.append(identity)
        run_dir = Path(cast(str, member["attempt_root"])) / (
            f"{identity[0]}-confirmation-seed{identity[1]}-attempt00"
        )
        return _fake_verified_run(member, run_dir)

    def export(
        member: Mapping[str, object],
        run: runner.VerifiedRun,
        **kwargs: object,
    ) -> tuple[PredictionArtifact, str, str]:
        return (
            _prediction(
                protocol,
                architecture=cast(str, member["architecture"]),
                seed=cast(int, member["seed"]),
            ),
            _digest("1"),
            _digest("2"),
        )

    monkeypatch.setattr(runner, "_verify_complete_run", verify)
    monkeypatch.setattr(runner, "_train_member", train)
    monkeypatch.setattr(runner, "_export_or_verify_prediction", export)
    completed = runner.run_multiseed_confirmation(
        result.plan_path,
        protocol=protocol,
    )

    assert trained == [
        ("resnet1d", 2027),
        ("ecg_transformer", 2027),
        ("ecg_transformer", 2028),
        ("resnet1d", 2028),
    ]
    assert len(completed.completion_paths) == 6
    assert len(completed.prediction_paths) == 6
    for path in completed.completion_paths:
        payload = cast(Mapping[str, object], json.loads(path.read_text(encoding="utf-8")))
        assert set(payload) == runner._MEMBER_COMPLETION_KEYS
        assert payload["status"] == "complete"
        runner._verify_self_hash(payload, "member completion")


def test_prediction_verifier_accepts_exporters_unprefixed_checkpoint_hash(
    tmp_path: Path,
) -> None:
    protocol = ExperimentProtocol.canonical()
    member: dict[str, object] = {
        "seed": 2027,
        "prediction_path": str((tmp_path / "fold8.npz").resolve()),
        "prediction_json_path": str((tmp_path / "fold8.json").resolve()),
    }
    run = _fake_verified_run(member, tmp_path / "resnet1d-seed2027")
    artifact = _prediction(protocol, architecture="resnet1d", seed=2027)
    artifact = create_prediction_artifact(
        ecg_id=artifact.ecg_id,
        patient_id=artifact.patient_id,
        strat_fold=artifact.strat_fold,
        targets=artifact.targets,
        raw_logits=artifact.raw_logits,
        model_name=run.run_name,
        model_seed=run.seed,
        protocol=protocol,
        config_hash=run.resolved_config_hash,
        manifest_hash=run.manifest_sha256,
        fold_role=FoldRole.MODEL_SELECTION,
        created_at_utc="2026-08-08T00:00:00Z",
        extra_metadata={
            "lineage": "development",
            "checkpoint_sha256": run.hashes["best.ckpt"].removeprefix("sha256:"),
            "checkpoint_epoch": run.best_epoch,
        },
    )
    save_prediction_artifact(artifact, tmp_path / "fold8.npz", protocol=protocol)

    verified, _, _ = runner._verify_prediction(member, run, protocol=protocol)

    assert verified.extra_metadata["checkpoint_sha256"] == "e" * 64


def test_study_winner_is_recomputed_instead_of_trusted() -> None:
    attempts: list[dict[str, object]] = []
    for candidate in range(12):
        attempts.append(
            {
                "state": "COMPLETE",
                "candidate_index": candidate,
                "trial_number": candidate,
                "completed_epochs": 20,
                "best_fold8_uncalibrated_macro_roc_auc": 0.90 + candidate / 1_000,
            }
        )
    study: dict[str, object] = {
        "schema_version": 2,
        "comparison_id": "paired-v1",
        "architecture": "resnet1d",
        "study_name": "resnet-study",
        "candidate_plan_hash": _digest("d"),
        "sweep_config_hash": _digest("e"),
        "base_experiment_config_hash": _digest("f"),
        "objective": {},
        "seed_policy": {},
        "tie_break": {},
        "failure_policy": {},
        "required_complete_candidates": 12,
        "completed_candidates": 12,
        "budget_complete": True,
        "attempt_counts": {"COMPLETE": 12},
        "selection_released": False,
        "best_candidate": None,
        "study_user_attrs": {},
        "attempts": attempts,
    }

    with pytest.raises(runner.MultiSeedRunnerError, match="deterministic best trial"):
        runner._verify_study_winner(
            study,
            attempts[0],
            architecture="resnet1d",
            comparison_id="paired-v1",
            candidate_plan_hash=_digest("d"),
        )


def test_revision_proof_allows_orchestration_change_but_rejects_kernel_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(repo), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    git("init")
    git("config", "user.email", "tests@example.invalid")
    git("config", "user.name", "ECG Trust Tests")
    files = {
        "configs/protocol.yaml": "schema_version: 1\n",
        "src/ecg_trust/training.py": "KERNEL = 1\n",
        "src/ecg_trust/multiseed_runner.py": "ORCHESTRATION = 1\n",
        "pyproject.toml": "[project]\nname = 'synthetic'\nversion = '0'\n",
        "uv.lock": "version = 1\n",
        "manifest.parquet": "manifest\n",
        "normalization.json": "{}\n",
    }
    for relative, content in files.items():
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    git("add", ".")
    git("commit", "-m", "sweep A")
    sweep_revision = git("rev-parse", "HEAD")
    runtime = {
        "python": "3.12.0",
        "implementation": "CPython",
        "platform": "synthetic-platform",
        "optuna": "4.0.0",
        "scipy": "1.0.0",
        "torch": "2.0.0",
    }
    source: dict[str, object] = {
        "project_root": str(repo.resolve()),
        "git_root": str(repo.resolve()),
        "git_head": sweep_revision,
        "git_dirty": False,
        "git_status_sha256": runner._EMPTY_SHA256,
        "git_unavailable": False,
        "source_tree_sha256": runner._source_tree_hash(repo),
        "dependency_lock_sha256": runner._file_sha256(repo / "uv.lock"),
        "manifest_path": str((repo / "manifest.parquet").resolve()),
        "manifest_sha256": runner._file_sha256(repo / "manifest.parquet"),
        "normalization_path": str((repo / "normalization.json").resolve()),
        "normalization_sha256": runner._file_sha256(repo / "normalization.json"),
        "runtime_identity": runtime,
        "runtime_identity_sha256": runner._canonical_hash(runtime),
    }
    (repo / "src/ecg_trust/multiseed_runner.py").write_text(
        "ORCHESTRATION = 2\n", encoding="utf-8"
    )
    git("add", "src/ecg_trust/multiseed_runner.py")
    git("commit", "-m", "confirmation orchestration B")
    execution_revision = git("rev-parse", "HEAD")
    monkeypatch.setattr(runner, "_current_runtime_identity", lambda: runtime)

    proof = runner._build_revision_provenance({"source_provenance": source})
    kernel = cast(Mapping[str, object], proof["scientific_kernel"])
    assert kernel["sweep_revision"] == sweep_revision
    assert kernel["execution_revision"] == execution_revision
    assert kernel["git_diff_sha256"] == runner._EMPTY_SHA256
    assert kernel["allowed_changed_paths"] == [
        "src/ecg_trust/multiseed_runner.py"
    ]
    runner._verify_revision_provenance(proof, expected_sweep_source=source)

    (repo / "src/ecg_trust/training.py").write_text("KERNEL = 2\n", encoding="utf-8")
    git("add", "src/ecg_trust/training.py")
    git("commit", "-m", "scientific kernel changed")
    with pytest.raises(runner.MultiSeedRunnerError, match="scientific kernel changed"):
        runner._build_revision_provenance({"source_provenance": source})
