from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
import torch

from ecg_trust.experiment_config import DevelopmentExperimentConfig
from ecg_trust.protocol import LABEL_ORDER, ExperimentProtocol
from ecg_trust.sweep_config import EqualBudgetSweepPair, load_equal_budget_pair
from ecg_trust.sweep_runner import (
    SweepRunnerError,
    TrialOutcome,
    _comparison_writer_lock,
    build_candidate_plan,
    preflight_equal_budget_sweeps,
    read_sweep_status,
    run_equal_budget_sweeps,
)

_RUN_NAME = re.compile(
    r"(?P<architecture>resnet1d|ecg_transformer)-candidate(?P<candidate>\d+)-"
    r"attempt(?P<attempt>\d+)-seed(?P<seed>\d+)"
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _protocol() -> ExperimentProtocol:
    return ExperimentProtocol.canonical()


def _pair(tmp_path: Path) -> EqualBudgetSweepPair:
    root = _project_root()
    pair = load_equal_budget_pair(
        (
            root / "configs" / "sweep_resnet_equal_budget.yaml",
            root / "configs" / "sweep_transformer_equal_budget.yaml",
        ),
        base_dir=root,
    )
    storage = replace(
        pair.resnet.storage,
        sqlite_path=tmp_path / "optuna.sqlite3",
        output_root=tmp_path / "sweep",
    )
    return EqualBudgetSweepPair.create(
        [replace(pair.resnet, storage=storage), replace(pair.transformer, storage=storage)]
    )


def _canonical_hash(value: Mapping[str, object]) -> str:
    serialized = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return "sha256:" + hashlib.sha256(serialized.encode()).hexdigest()


def _bare_file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_identity(config: DevelopmentExperimentConfig) -> tuple[str, int, int, int]:
    match = _RUN_NAME.fullmatch(config.run_name)
    assert match is not None
    return (
        match.group("architecture"),
        int(match.group("candidate")),
        int(match.group("attempt")),
        int(match.group("seed")),
    )


def _metrics(score: float) -> dict[str, object]:
    return {
        "n_samples": 20,
        "label_order": list(LABEL_ORDER),
        "ece_bins": 15,
        "per_label": [
            {
                "label": label,
                "positives": 10,
                "negatives": 10,
                "prevalence": 0.5,
                "roc_auc": score,
                "average_precision": score,
                "brier_score": 0.2,
                "ece": 0.1,
                "degenerate_reason": None,
            }
            for label in LABEL_ORDER
        ],
        "macro": {
            "roc_auc": score,
            "average_precision": score,
            "brier_score": 0.2,
            "ece": 0.1,
            "roc_auc_labels": 5,
            "average_precision_labels": 5,
        },
    }


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _write_valid_artifacts(
    config: DevelopmentExperimentConfig,
    protocol: ExperimentProtocol,
    *,
    score: float,
    completed_epochs: int,
) -> TrialOutcome:
    run_dir = config.output.root_dir / config.run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    resolved = config.to_resolved_dict()
    model = cast(dict[str, object], resolved["model"])
    model["trainable_parameters"] = 1
    resolved["optimizer"] = {"name": "AdamW"}
    resolved["effective_data"] = {"train_records": 10, "validation_records": 10}
    resolved_hash = _canonical_hash(resolved)
    _write_json(
        run_dir / "resolved_config.json",
        {"config_hash": resolved_hash, "config": resolved},
    )
    _write_json(run_dir / "protocol.json", protocol.to_resolved_dict())

    records: list[dict[str, object]] = []
    for epoch in range(completed_epochs):
        observed = score if epoch == 0 else score + (0.00005 if epoch == 1 else -0.01)
        records.append(
            {
                "epoch": epoch,
                "validation_macro_auroc": observed,
                "validation_metrics": _metrics(observed),
                "improved": epoch == 0,
            }
        )
    (run_dir / "history.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    manifest_hash = _bare_file_hash(config.data.manifest_path)
    normalization_hash = _bare_file_hash(config.data.normalization_path)
    _write_json(
        run_dir / "run_metadata.json",
        {
            "status": "complete",
            "seed": config.runtime.seed,
            "source_config_hash": config.config_hash,
            "resolved_config_hash": resolved_hash,
            "protocol_hash": protocol.protocol_hash,
            "manifest_hash": manifest_hash,
            "normalization_file_hash": normalization_hash,
            "completed_epochs": completed_epochs,
            "best_epoch": 0,
            "best_validation_macro_auroc": score,
        },
    )
    stopper = {
        "patience": 10,
        "mode": "max",
        "min_delta": 0.0001,
        "best_score": score,
        "best_epoch": 0,
        "bad_epochs": completed_epochs - 1,
        "stopped": False,
    }

    def checkpoint(epoch: int) -> dict[str, object]:
        return {
            "schema_version": 1,
            "epoch": epoch,
            "protocol_hash": protocol.protocol_hash,
            "manifest_hash": manifest_hash,
            "config": resolved,
            "config_hash": resolved_hash,
            "model_state_dict": {},
            "optimizer_state_dict": {},
            "scaler_state_dict": None,
            "early_stopping_state_dict": stopper,
        }

    torch.save(checkpoint(0), run_dir / "best.ckpt")
    torch.save(checkpoint(completed_epochs - 1), run_dir / "last.ckpt")
    return TrialOutcome(
        best_macro_auroc=score,
        best_epoch=0,
        completed_epochs=completed_epochs,
        runtime_seconds=0.01,
        peak_allocated_bytes=0,
        peak_reserved_bytes=0,
        run_dir=run_dir,
        defined_label_count=5,
        probabilities_calibrated=False,
        resolved_config_hash=resolved_hash,
    )


def _executor(
    calls: list[tuple[str, int, int, int]],
    *,
    fail_once: set[tuple[str, int]] | None = None,
) -> Callable[[DevelopmentExperimentConfig, ExperimentProtocol], TrialOutcome]:
    failed: set[tuple[str, int]] = set()

    def execute(
        config: DevelopmentExperimentConfig,
        protocol: ExperimentProtocol,
    ) -> TrialOutcome:
        architecture, candidate, attempt, seed = _run_identity(config)
        calls.append((architecture, candidate, attempt, seed))
        assert seed == config.runtime.seed == 2026
        assert config.optimization.epochs == 30
        assert config.train_folds == tuple(range(1, 8))
        assert config.validation_folds == (8,)
        identity = (architecture, candidate)
        if fail_once and identity in fail_once and identity not in failed:
            failed.add(identity)
            partial = config.output.root_dir / config.run_name
            partial.mkdir(parents=True, exist_ok=False)
            (partial / "partial.txt").write_text("preserved", encoding="utf-8")
            raise RuntimeError("synthetic transient failure")
        completed_epochs = 3 if candidate == 0 else 2
        return _write_valid_artifacts(
            config,
            protocol,
            score=0.8,
            completed_epochs=completed_epochs,
        )

    return execute


def test_candidate_plan_is_deterministic_balanced_and_persistable(tmp_path: Path) -> None:
    pair = _pair(tmp_path)
    first = build_candidate_plan(pair)
    second = build_candidate_plan(pair)

    assert first == second
    assert len(first.candidates) == 12
    assert first.plan_hash.startswith("sha256:")
    for field, choices in (
        ("batch_size", pair.resnet.search_space.batch_size),
        ("gradient_clip_norm", pair.resnet.search_space.gradient_clip_norm),
        ("warmup_epochs", pair.resnet.search_space.warmup_epochs),
        ("minimum_lr_ratio", pair.resnet.search_space.minimum_lr_ratio),
    ):
        counts = Counter(candidate.parameters[field] for candidate in first.candidates)
        assert counts == Counter({choice: 4 for choice in choices})


def test_fresh_preflight_is_read_only_and_hashes_scientific_inputs(tmp_path: Path) -> None:
    pair = _pair(tmp_path)
    result = preflight_equal_budget_sweeps(pair, protocol=_protocol())

    assert result.storage_exists is False
    assert result.existing_study_names == ()
    assert not result.candidate_plan_path.exists()
    assert not pair.resnet.storage.output_root.exists()
    assert str(result.source_provenance["manifest_sha256"]).startswith("sha256:")
    assert str(result.source_provenance["normalization_sha256"]).startswith("sha256:")


def test_paired_run_alternates_order_uses_fixed_budget_and_hides_study_winners(
    tmp_path: Path,
) -> None:
    pair = _pair(tmp_path)
    calls: list[tuple[str, int, int, int]] = []
    result = run_equal_budget_sweeps(
        pair, protocol=_protocol(), executor=_executor(calls)
    )

    assert len(calls) == 24
    for candidate in range(12):
        pair_calls = calls[candidate * 2 : candidate * 2 + 2]
        expected = (
            ["resnet1d", "ecg_transformer"]
            if candidate % 2 == 0
            else ["ecg_transformer", "resnet1d"]
        )
        assert [call[0] for call in pair_calls] == expected
        assert {call[1] for call in pair_calls} == {candidate}
        assert {call[3] for call in pair_calls} == {2026}
    assert set(result.best_by_architecture) == {"resnet1d", "ecg_transformer"}
    assert all(best.candidate_index == 1 for best in result.best_by_architecture.values())
    for architecture in ("resnet1d", "ecg_transformer"):
        study_summary = json.loads(
            (pair.resnet.storage.output_root / f"{architecture}_study.json").read_text()
        )
        assert study_summary["budget_complete"] is True
        assert study_summary["selection_released"] is False
        assert study_summary["best_candidate"] is None
        assert all(
            attempt["literal_max_macro_auroc"] >= attempt["selected_checkpoint_score"]
            for attempt in study_summary["attempts"]
            if attempt["state"] == "COMPLETE"
        )
    final = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert final["all_candidate_budgets_complete"] is True
    assert len(final["paired_execution_order"]) == 12
    assert set(final["best_by_architecture"]) == {"resnet1d", "ecg_transformer"}


def test_completed_resume_does_not_execute_new_attempts(tmp_path: Path) -> None:
    pair = _pair(tmp_path)
    run_equal_budget_sweeps(pair, protocol=_protocol(), executor=_executor([]))

    def forbidden(
        config: DevelopmentExperimentConfig,
        protocol: ExperimentProtocol,
    ) -> TrialOutcome:
        del config, protocol
        raise AssertionError("completed resume must not execute")

    resumed = run_equal_budget_sweeps(
        pair,
        protocol=_protocol(),
        resume=True,
        executor=forbidden,
    )
    assert all(study.completed_candidates == 12 for study in resumed.studies)


def test_failure_retries_identical_candidate_in_new_immutable_attempt(tmp_path: Path) -> None:
    pair = _pair(tmp_path)
    calls: list[tuple[str, int, int, int]] = []
    failures = {("resnet1d", 0), ("ecg_transformer", 0)}
    result = run_equal_budget_sweeps(
        pair,
        protocol=_protocol(),
        executor=_executor(calls, fail_once=failures),
    )

    assert len(calls) == 26
    for architecture in ("resnet1d", "ecg_transformer"):
        candidate_zero = [call for call in calls if call[:2] == (architecture, 0)]
        assert [call[2] for call in candidate_zero] == [0, 1]
        assert {call[3] for call in candidate_zero} == {2026}
        partial = (
            pair.resnet.storage.output_root
            / "trials"
            / architecture
            / f"{architecture}-candidate00-attempt00-seed2026"
            / "partial.txt"
        )
        assert partial.read_text(encoding="utf-8") == "preserved"
    assert all(study.completed_candidates == 12 for study in result.studies)
    assert all(study.failed_attempts == 1 for study in result.studies)


def test_exclusive_writer_lock_rejects_concurrent_run(tmp_path: Path) -> None:
    pair = _pair(tmp_path)
    with (
        _comparison_writer_lock(pair.resnet.storage.output_root),
        pytest.raises(SweepRunnerError, match="already locked"),
    ):
        run_equal_budget_sweeps(
            pair,
            protocol=_protocol(),
            executor=_executor([]),
        )


def test_resume_rejects_sweep_config_provenance_drift(tmp_path: Path) -> None:
    pair = _pair(tmp_path)
    run_equal_budget_sweeps(pair, protocol=_protocol(), executor=_executor([]))
    changed_resnet = replace(
        pair.resnet,
        seed_policy=replace(pair.resnet.seed_policy, experiment_seed=2027),
    )
    changed_transformer = replace(
        pair.transformer,
        seed_policy=replace(pair.transformer.seed_policy, experiment_seed=2027),
    )
    changed = EqualBudgetSweepPair.create([changed_resnet, changed_transformer])

    with pytest.raises(SweepRunnerError, match="provenance drifted"):
        run_equal_budget_sweeps(
            changed,
            protocol=_protocol(),
            resume=True,
            executor=_executor([]),
        )


def test_status_and_resume_reject_tampered_complete_artifact(tmp_path: Path) -> None:
    pair = _pair(tmp_path)
    result = run_equal_budget_sweeps(
        pair, protocol=_protocol(), executor=_executor([])
    )
    target = result.best_by_architecture["resnet1d"].run_dir / "history.jsonl"
    target.write_text(target.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")

    with pytest.raises(SweepRunnerError, match="history length"):
        read_sweep_status(pair, protocol=_protocol())
    with pytest.raises(SweepRunnerError, match="history length"):
        run_equal_budget_sweeps(
            pair,
            protocol=_protocol(),
            resume=True,
            executor=_executor([]),
        )


def test_status_never_releases_per_study_best(tmp_path: Path) -> None:
    pair = _pair(tmp_path)
    run_equal_budget_sweeps(pair, protocol=_protocol(), executor=_executor([]))
    status = read_sweep_status(pair, protocol=_protocol())
    studies = cast(dict[str, object], status["studies"])
    for raw in studies.values():
        entry = cast(dict[str, object], raw)
        payload = cast(dict[str, object], entry["payload"])
        assert payload["selection_released"] is False
        assert payload["best_candidate"] is None
