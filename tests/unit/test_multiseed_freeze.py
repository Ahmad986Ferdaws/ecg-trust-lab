from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import Dataset, TensorDataset

from ecg_trust.constants import LEADS, TARGET_COLUMNS
from ecg_trust.data.dataset import NormalizationProvenance, NormalizationStats
from ecg_trust.data.manifest import sha256_file
from ecg_trust.experiment_config import ModelConfig
from ecg_trust.experiment_runner import training_manifest_sha256
from ecg_trust.multiseed_freeze import (
    ARCHITECTURES,
    CONFIRMATION_SEEDS,
    FreezeCreation,
    MultiSeedFreezeError,
    canonical_sha256,
    create_multiseed_freeze_payload,
    file_sha256,
    load_confirmation_member,
    load_multiseed_freeze,
    materialize_refit_recipes,
    publish_multiseed_freeze_bundle,
    write_multiseed_freeze,
)
from ecg_trust.predictions import create_prediction_artifact, save_prediction_artifact
from ecg_trust.protocol import LABEL_ORDER, TRAIN_FOLDS, ExperimentProtocol, FoldRole
from ecg_trust.refit_config import PostSweepRefitConfig, load_refit_config
from ecg_trust.refit_runner import (
    FrozenRefitError,
    load_refit_completion,
    run_frozen_refit,
)
from ecg_trust.training import EarlyStopping, save_checkpoint


def _json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _manifest(tmp_path: Path) -> tuple[pd.DataFrame, Path, Path]:
    rows: list[dict[str, object]] = []
    for index in range(10):
        row: dict[str, object] = {
            "ecg_id": 1000 + index,
            "patient_id": 2000 + index,
            "record_path": f"records/fold8-{index}",
            "strat_fold": 8,
        }
        row.update(
            {
                target: int((index + label_index) % 2 == 0)
                for label_index, target in enumerate(TARGET_COLUMNS)
            }
        )
        rows.append(row)
    train: dict[str, object] = {
        "ecg_id": 999,
        "patient_id": 1999,
        "record_path": "records/train",
        "strat_fold": 1,
    }
    train.update({target: index % 2 for index, target in enumerate(TARGET_COLUMNS)})
    rows.append(train)
    frame = pd.DataFrame(rows)
    manifest_path = tmp_path / "manifest.csv"
    frame.to_csv(manifest_path, index=False)
    provenance = NormalizationProvenance(
        dataset_version="1.0.3",
        manifest_sha256=training_manifest_sha256(frame, TRAIN_FOLDS),
        training_folds=TRAIN_FOLDS,
        record_count=1,
        sample_count=1000,
        sampling_frequency_hz=100.0,
        samples_per_record=1000,
        path_column="record_path",
        fold_column="strat_fold",
        target_columns=TARGET_COLUMNS,
    )
    normalization = NormalizationStats(
        mean=tuple(0.0 for _ in LEADS),
        std=tuple(1.0 for _ in LEADS),
        leads=LEADS,
        provenance=provenance,
    )
    normalization_path = tmp_path / "normalization.json"
    normalization.save(normalization_path)
    return frame, manifest_path, normalization_path


def _development_config(
    *,
    architecture: str,
    seed: int,
    run_name: str,
    run_dir: Path,
    manifest_path: Path,
    normalization_path: Path,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "run_name": run_name,
        "folds": {"train": list(TRAIN_FOLDS), "model_selection": [8]},
        "data": {
            "manifest": str(manifest_path.resolve()),
            "dataset_root": str((manifest_path.parent / "records").resolve()),
            "normalization": str(normalization_path.resolve()),
            "max_train_records": None,
            "max_validation_records": None,
        },
        "model": {
            "architecture": architecture,
            "preset": "smoke",
            "class": "torch.nn.modules.linear.Linear",
            "trainable_parameters": 25,
            "resolved_architecture_config": {},
        },
        "loader": {
            "batch_size": 4,
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
        },
        "optimization": {
            "epochs": 30,
            "learning_rate": 0.01,
            "weight_decay": 0.001,
            "warmup_epochs": 1,
            "minimum_lr_ratio": 0.1,
            "gradient_clip_norm": 1.0,
            "early_stopping_patience": 10,
            "early_stopping_min_delta": 0.0001,
            "scheduler": "warmup_cosine",
        },
        "runtime": {"seed": seed, "device": "cpu", "bf16": True},
        "output": {"root_dir": str(run_dir.parent.resolve())},
        "effective_data": {"train_records": 1, "validation_records": 10},
        "optimizer": {"name": "AdamW", "betas": [0.8, 0.95], "eps": 1e-7},
    }


def _member(
    *,
    tmp_path: Path,
    frame: pd.DataFrame,
    manifest_path: Path,
    normalization_path: Path,
    architecture: str,
    seed: int,
    best_epoch: int,
    prediction_fold: int = 8,
) -> tuple[Path, dict[str, object], Path]:
    protocol = ExperimentProtocol.canonical()
    run_name = f"{architecture}-confirmation-seed{seed}-attempt00"
    run_dir = tmp_path / "members" / architecture / f"seed{seed}" / run_name
    run_dir.mkdir(parents=True)
    config = _development_config(
        architecture=architecture,
        seed=seed,
        run_name=run_name,
        run_dir=run_dir,
        manifest_path=manifest_path,
        normalization_path=normalization_path,
    )
    config_hash = canonical_sha256(config)
    resolved_path = run_dir / "resolved_config.json"
    _json(resolved_path, {"config_hash": config_hash, "config": config})
    manifest_hash = sha256_file(manifest_path)
    normalization_hash = sha256_file(normalization_path)
    model = nn.Linear(4, 5)
    optimizer = AdamW(
        model.parameters(),
        lr=0.01,
        betas=(0.8, 0.95),
        eps=1e-7,
        weight_decay=0.001,
    )
    stopper = EarlyStopping(patience=10, mode="max", min_delta=0.0001)
    for epoch in range(best_epoch + 1):
        stopper.update(0.5 + 0.5 * (epoch + 1) / (best_epoch + 1), epoch)
    checkpoint_path = run_dir / "best.ckpt"
    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        scaler=None,
        epoch=best_epoch,
        protocol_hash=protocol.protocol_hash,
        config=config,
        manifest_hash=manifest_hash,
        early_stopping=stopper,
    )
    completed_epochs = best_epoch + 1
    history_path = run_dir / "history.jsonl"
    history_rows = []
    for epoch in range(completed_epochs):
        score = 0.5 + 0.5 * (epoch + 1) / completed_epochs
        row: dict[str, object] = {
            "epoch": epoch,
            "validation_macro_auroc": score,
            "improved": True,
        }
        if epoch == best_epoch:
            row["validation_metrics"] = {
                "label_order": list(LABEL_ORDER),
                "macro": {"roc_auc": 1.0, "roc_auc_labels": 5},
                "per_label": [
                    {"label": label, "roc_auc": 1.0} for label in LABEL_ORDER
                ],
            }
        history_rows.append(json.dumps(row, sort_keys=True))
    history_path.write_text("\n".join(history_rows) + "\n", encoding="utf-8")
    metadata_path = run_dir / "run_metadata.json"
    _json(
        metadata_path,
        {
            "status": "complete",
            "seed": seed,
            "resolved_config_hash": config_hash,
            "protocol_hash": protocol.protocol_hash,
            "manifest_hash": manifest_hash,
            "normalization_file_hash": normalization_hash,
            "best_epoch": best_epoch,
            "completed_epochs": completed_epochs,
            "best_validation_macro_auroc": 1.0,
        },
    )
    selected = frame.loc[frame["strat_fold"] == 8].reset_index(drop=True)
    targets = selected.loc[:, list(TARGET_COLUMNS)].to_numpy(dtype=np.int8)
    logits = np.where(targets == 1, 4.0, -4.0)
    folds = np.full(len(selected), prediction_fold, dtype=np.int8)
    prediction = create_prediction_artifact(
        ecg_id=selected["ecg_id"].to_numpy(),
        patient_id=selected["patient_id"].to_numpy(),
        strat_fold=folds,
        targets=targets,
        raw_logits=logits,
        model_name=run_name,
        model_seed=seed,
        protocol=protocol,
        config_hash=config_hash,
        manifest_hash=manifest_hash,
        fold_role=(
            FoldRole.MODEL_SELECTION if prediction_fold == 8 else FoldRole.CALIBRATION
        ),
        created_at_utc="2026-08-08T00:00:00Z",
        extra_metadata={
            "lineage": "development",
            "checkpoint_path": str(checkpoint_path.resolve()),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "checkpoint_epoch": best_epoch,
            "resolved_config_path": str(resolved_path.resolve()),
            "normalization_sha256": normalization_hash,
        },
    )
    prediction_path = tmp_path / "predictions" / f"{architecture}-seed{seed}-fold8.npz"
    files = save_prediction_artifact(prediction, prediction_path, protocol=protocol)
    member_plan = run_dir.parent / "member_plan.json"
    _json(member_plan, {"architecture": architecture, "seed": seed})
    completion_body: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "ecg_trust.multiseed_member_completion",
        "comparison_id": "paired-test-v1",
        "architecture": architecture,
        "seed": seed,
        "status": "complete",
        "member_plan_path": str(member_plan.resolve()),
        "member_plan_sha256": file_sha256(member_plan),
        "run_dir": str(run_dir.resolve()),
        "run_metadata_path": str(metadata_path.resolve()),
        "run_metadata_sha256": file_sha256(metadata_path),
        "resolved_config_path": str(resolved_path.resolve()),
        "resolved_config_sha256": file_sha256(resolved_path),
        "history_path": str(history_path.resolve()),
        "history_sha256": file_sha256(history_path),
        "best_checkpoint_path": str(checkpoint_path.resolve()),
        "best_checkpoint_sha256": file_sha256(checkpoint_path),
        "config_hash": config_hash,
        "protocol_hash": protocol.protocol_hash,
        "manifest_hash": "sha256:" + manifest_hash,
        "normalization_sha256": "sha256:" + normalization_hash,
        "best_epoch": best_epoch,
        "best_validation_macro_auroc": 1.0,
        "completed_epochs": completed_epochs,
        "prediction_path": str(files.npz_path.resolve()),
        "prediction_npz_sha256": file_sha256(files.npz_path),
        "prediction_json_path": str(files.json_path.resolve()),
        "prediction_artifact_sha256": files.artifact_sha256,
    }
    completion = dict(completion_body)
    completion["artifact_sha256"] = canonical_sha256(completion_body)
    completion_path = run_dir.parent / "member_completion.json"
    _json(completion_path, completion)
    return completion_path, config, resolved_path


@pytest.fixture
def freeze_inputs(tmp_path: Path) -> dict[str, object]:
    lock_path = tmp_path / "uv.lock"
    lock_path.write_text("version = 1\n", encoding="utf-8")
    frame, manifest_path, normalization_path = _manifest(tmp_path)
    completions: list[Path] = []
    winners: dict[str, object] = {}
    epochs = {2026: 1, 2027: 3, 2028: 5}
    for architecture in ARCHITECTURES:
        for seed in CONFIRMATION_SEEDS:
            completion, config, resolved = _member(
                tmp_path=tmp_path,
                frame=frame,
                manifest_path=manifest_path,
                normalization_path=normalization_path,
                architecture=architecture,
                seed=seed,
                best_epoch=epochs[seed],
            )
            completions.append(completion)
            if seed == 2026:
                optimization = config["optimization"]
                loader = config["loader"]
                assert isinstance(optimization, dict)
                assert isinstance(loader, dict)
                winners[architecture] = {
                    "state": "COMPLETE",
                    "probabilities_calibrated": False,
                    "trial_number": 3,
                    "candidate_index": 2,
                    "attempt_index": 0,
                    "resolved_config_hash": canonical_sha256(config),
                    "run_dir": str(resolved.parent.resolve()),
                    "artifact_sha256": {
                        "resolved_config.json": file_sha256(resolved),
                    },
                    "parameters": {
                        "batch_size": loader["batch_size"],
                        "gradient_clip_norm": optimization["gradient_clip_norm"],
                        "learning_rate": optimization["learning_rate"],
                        "minimum_lr_ratio": optimization["minimum_lr_ratio"],
                        "warmup_epochs": optimization["warmup_epochs"],
                        "weight_decay": optimization["weight_decay"],
                    },
                }
    sweep_summary = tmp_path / "sweep_summary.json"
    _json(
        sweep_summary,
        {
            "schema_version": 2,
            "comparison_id": "paired-test-v1",
            "protocol_hash": ExperimentProtocol.canonical().protocol_hash,
            "all_candidate_budgets_complete": True,
            "equal_candidate_plan_verified": True,
            "required_complete_candidates_per_architecture": 12,
            "candidate_plan_hash": "sha256:" + "a" * 64,
            "objective": {
                "name": "fold8_uncalibrated_macro_roc_auc",
                "direction": "maximize",
                "pruning": "none",
                "required_label_count": 5,
                "require_all_labels_defined": True,
            },
            "source_provenance": {
                "project_root": str(tmp_path.resolve()),
                "manifest_sha256": file_sha256(manifest_path),
                "normalization_sha256": file_sha256(normalization_path),
            },
            "best_by_architecture": winners,
        },
    )
    return {
        "tmp_path": tmp_path,
        "frame": frame,
        "manifest": manifest_path,
        "normalization": normalization_path,
        "completions": completions,
        "sweep": sweep_summary,
        "creation": FreezeCreation(
            timestamp_utc="2026-08-08T12:00:00Z",
            code_revision="unavailable",
            dependency_lock_sha256=file_sha256(lock_path),
            software_versions={"python": "3.12", "numpy": "2", "torch": "2"},
        ),
    }


def _freeze_payload(inputs: dict[str, object]) -> dict[str, object]:
    return create_multiseed_freeze_payload(
        sweep_summary_path=Path(inputs["sweep"]),
        member_completion_paths=list(inputs["completions"]),
        protocol=ExperimentProtocol.canonical(),
        refit_output_root=Path(inputs["tmp_path"]) / "refits",
        creation=inputs["creation"],  # type: ignore[arg-type]
    )


def _post_sweep_config(inputs: dict[str, object]) -> PostSweepRefitConfig:
    output = Path(inputs["tmp_path"]) / "freeze.json"
    freeze = write_multiseed_freeze(
        output,
        sweep_summary_path=Path(inputs["sweep"]),
        member_completion_paths=list(inputs["completions"]),
        protocol=ExperimentProtocol.canonical(),
        refit_output_root=Path(inputs["tmp_path"]) / "refits",
        creation=inputs["creation"],  # type: ignore[arg-type]
    )
    recipes = materialize_refit_recipes(freeze, output.parent / "recipes")
    recipe_path = next(
        path for path in recipes if path.name == "resnet1d-seed2028-refit.json"
    )
    config = load_refit_config(recipe_path, base_dir=Path(inputs["tmp_path"]))
    assert isinstance(config, PostSweepRefitConfig)
    return config


def _tiny_dataset(
    manifest: pd.DataFrame,
    root_dir: Path,
    *,
    folds: tuple[int, ...],
    normalization: NormalizationStats,
    protocol: ExperimentProtocol,
) -> Dataset[tuple[Tensor, Tensor]]:
    del manifest, root_dir, folds, normalization, protocol
    generator = torch.Generator().manual_seed(8)
    inputs = torch.randn(12, 4, generator=generator)
    indices = torch.arange(12)
    targets = torch.stack(
        [((indices + label) % 2).float() for label in range(5)], dim=1
    )
    return TensorDataset(inputs, targets)


def _linear_model(model_config: ModelConfig) -> nn.Module:
    assert model_config.architecture == "resnet1d"
    return nn.Linear(4, 5)


def test_freeze_recomputes_tie_and_median_epoch_budgets(
    freeze_inputs: dict[str, object],
) -> None:
    first = _freeze_payload(freeze_inputs)
    second = _freeze_payload(freeze_inputs)
    assert first == second
    assert first["decision"]["status"] == "practical_tie"  # type: ignore[index]
    assert first["decision"]["primary_architecture"] == "resnet1d"  # type: ignore[index]
    architectures = first["architectures"]
    assert isinstance(architectures, dict)
    assert architectures["resnet1d"]["frozen_refit_epochs"] == 4
    assert architectures["ecg_transformer"]["frozen_refit_epochs"] == 4
    assert len(first["refit_recipes"]) == 6  # type: ignore[arg-type]


def test_freeze_rejects_missing_seed_and_fold9_prediction(
    freeze_inputs: dict[str, object],
) -> None:
    completions = list(freeze_inputs["completions"])
    with pytest.raises(MultiSeedFreezeError, match="exactly both architectures"):
        create_multiseed_freeze_payload(
            sweep_summary_path=Path(freeze_inputs["sweep"]),
            member_completion_paths=completions[:-1],
            protocol=ExperimentProtocol.canonical(),
            refit_output_root=Path(freeze_inputs["tmp_path"]) / "refits",
            creation=freeze_inputs["creation"],  # type: ignore[arg-type]
        )

    frame = freeze_inputs["frame"]
    assert isinstance(frame, pd.DataFrame)
    fold9, _, _ = _member(
        tmp_path=Path(freeze_inputs["tmp_path"]) / "fold9-case",
        frame=frame,
        manifest_path=Path(freeze_inputs["manifest"]),
        normalization_path=Path(freeze_inputs["normalization"]),
        architecture="resnet1d",
        seed=2026,
        best_epoch=1,
        prediction_fold=9,
    )
    with pytest.raises(MultiSeedFreezeError, match="fold 8 only"):
        load_confirmation_member(fold9, protocol=ExperimentProtocol.canonical())


def test_freeze_is_non_overwriting_and_rejects_derived_tampering(
    freeze_inputs: dict[str, object],
) -> None:
    output = Path(freeze_inputs["tmp_path"]) / "freeze.json"
    freeze = write_multiseed_freeze(
        output,
        sweep_summary_path=Path(freeze_inputs["sweep"]),
        member_completion_paths=list(freeze_inputs["completions"]),
        protocol=ExperimentProtocol.canonical(),
        refit_output_root=Path(freeze_inputs["tmp_path"]) / "refits",
        creation=freeze_inputs["creation"],  # type: ignore[arg-type]
    )
    assert freeze.path == output.resolve()
    with pytest.raises(MultiSeedFreezeError, match="already exists"):
        write_multiseed_freeze(
            output,
            sweep_summary_path=Path(freeze_inputs["sweep"]),
            member_completion_paths=list(freeze_inputs["completions"]),
            protocol=ExperimentProtocol.canonical(),
            refit_output_root=Path(freeze_inputs["tmp_path"]) / "refits",
            creation=freeze_inputs["creation"],  # type: ignore[arg-type]
        )
    tampered = freeze.payload
    tampered["decision"]["primary_architecture"] = "ecg_transformer"  # type: ignore[index]
    body = dict(tampered)
    body.pop("artifact_sha256")
    tampered["artifact_sha256"] = canonical_sha256(body)
    tampered_path = output.with_name("tampered.json")
    _json(tampered_path, tampered)
    with pytest.raises(MultiSeedFreezeError, match="decision"):
        load_multiseed_freeze(
            tampered_path,
            protocol=ExperimentProtocol.canonical(),
            verify_sources=False,
        )


def test_materialized_recipe_runs_fresh_median_budget_refit(
    freeze_inputs: dict[str, object],
) -> None:
    output = Path(freeze_inputs["tmp_path"]) / "freeze.json"
    freeze = write_multiseed_freeze(
        output,
        sweep_summary_path=Path(freeze_inputs["sweep"]),
        member_completion_paths=list(freeze_inputs["completions"]),
        protocol=ExperimentProtocol.canonical(),
        refit_output_root=Path(freeze_inputs["tmp_path"]) / "refits",
        creation=freeze_inputs["creation"],  # type: ignore[arg-type]
    )
    recipes = materialize_refit_recipes(freeze, output.parent / "recipes")
    recipe_path = next(path for path in recipes if path.name == "resnet1d-seed2028-refit.json")
    config = load_refit_config(recipe_path, base_dir=Path(freeze_inputs["tmp_path"]))
    assert isinstance(config, PostSweepRefitConfig)
    assert config.selection.frozen_epochs == 4
    assert config.source.best_epoch == 5

    calls: list[tuple[int, ...]] = []

    def dataset_factory(
        manifest: pd.DataFrame,
        root_dir: Path,
        *,
        folds: tuple[int, ...],
        normalization: NormalizationStats,
        protocol: ExperimentProtocol,
    ) -> Dataset[tuple[Tensor, Tensor]]:
        del manifest, root_dir, normalization, protocol
        calls.append(folds)
        generator = torch.Generator().manual_seed(8)
        inputs = torch.randn(12, 4, generator=generator)
        indices = torch.arange(12)
        targets = torch.stack(
            [((indices + label) % 2).float() for label in range(5)], dim=1
        )
        return TensorDataset(inputs, targets)

    def model_factory(model_config: ModelConfig) -> nn.Module:
        assert model_config.architecture == "resnet1d"
        return nn.Linear(4, 5)

    result = run_frozen_refit(
        config,
        protocol=ExperimentProtocol.canonical(),
        dataset_factory=dataset_factory,
        model_factory=model_factory,
    )
    assert calls == [tuple(range(1, 9))]
    assert result.frozen_epochs == 4
    assert result.completion_path is not None
    completion = load_refit_completion(
        result.completion_path,
        protocol=ExperimentProtocol.canonical(),
        verify_sources=True,
    )
    assert completion["freeze_artifact_sha256"] == freeze.artifact_sha256
    assert completion["frozen_epochs"] == 4
    assert completion["downstream_provenance"] == config.downstream_provenance.to_dict()
    final_checkpoint = torch.load(
        result.final_checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    parameter_group = final_checkpoint["optimizer_state_dict"]["param_groups"][0]
    assert parameter_group["betas"] == (0.8, 0.95)
    assert parameter_group["eps"] == pytest.approx(1e-7)


def test_refit_rejects_fold8_manifest_drift_before_factories(
    freeze_inputs: dict[str, object],
) -> None:
    config = _post_sweep_config(freeze_inputs)
    manifest_path = Path(freeze_inputs["manifest"])
    frame = pd.read_csv(manifest_path)
    fold8_index = frame.index[frame["strat_fold"] == 8][0]
    target = TARGET_COLUMNS[0]
    frame.loc[fold8_index, target] = 1 - int(frame.loc[fold8_index, target])
    frame.to_csv(manifest_path, index=False)
    calls: list[str] = []

    def forbidden_model(model_config: ModelConfig) -> nn.Module:
        del model_config
        calls.append("model")
        return nn.Linear(4, 5)

    with pytest.raises(FrozenRefitError, match="current manifest SHA-256"):
        run_frozen_refit(
            config,
            protocol=ExperimentProtocol.canonical(),
            dataset_factory=_tiny_dataset,
            model_factory=forbidden_model,
        )
    assert calls == []


def test_refit_rejects_regenerated_normalization_before_factories(
    freeze_inputs: dict[str, object],
) -> None:
    config = _post_sweep_config(freeze_inputs)
    normalization_path = Path(freeze_inputs["normalization"])
    original = NormalizationStats.load(normalization_path)
    changed_mean = (0.25, *original.mean[1:])
    NormalizationStats(
        mean=changed_mean,
        std=original.std,
        leads=original.leads,
        provenance=original.provenance,
    ).save(normalization_path)
    calls: list[str] = []

    def forbidden_model(model_config: ModelConfig) -> nn.Module:
        del model_config
        calls.append("model")
        return nn.Linear(4, 5)

    with pytest.raises(FrozenRefitError, match="current normalization SHA-256"):
        run_frozen_refit(
            config,
            protocol=ExperimentProtocol.canonical(),
            dataset_factory=_tiny_dataset,
            model_factory=forbidden_model,
        )
    assert calls == []


def test_refit_rejects_dependency_lock_drift_before_model_creation(
    freeze_inputs: dict[str, object],
) -> None:
    config = _post_sweep_config(freeze_inputs)
    (Path(freeze_inputs["tmp_path"]) / "uv.lock").write_text(
        "version = 2\n", encoding="utf-8"
    )
    calls: list[str] = []

    def forbidden_model(model_config: ModelConfig) -> nn.Module:
        del model_config
        calls.append("model")
        return nn.Linear(4, 5)

    with pytest.raises(FrozenRefitError, match="dependency lock differs"):
        run_frozen_refit(
            config,
            protocol=ExperimentProtocol.canonical(),
            dataset_factory=_tiny_dataset,
            model_factory=forbidden_model,
        )
    assert calls == []


def test_refit_rejects_fresh_model_identity_before_dataset_creation(
    freeze_inputs: dict[str, object],
) -> None:
    config = _post_sweep_config(freeze_inputs)
    dataset_calls: list[str] = []

    def forbidden_dataset(
        manifest: pd.DataFrame,
        root_dir: Path,
        *,
        folds: tuple[int, ...],
        normalization: NormalizationStats,
        protocol: ExperimentProtocol,
    ) -> Dataset[tuple[Tensor, Tensor]]:
        del manifest, root_dir, folds, normalization, protocol
        dataset_calls.append("dataset")
        raise AssertionError("dataset factory must not run after model identity drift")

    def wrong_model(model_config: ModelConfig) -> nn.Module:
        del model_config
        return nn.Sequential(nn.Linear(4, 5))

    with pytest.raises(FrozenRefitError, match="fresh model metadata"):
        run_frozen_refit(
            config,
            protocol=ExperimentProtocol.canonical(),
            dataset_factory=forbidden_dataset,
            model_factory=wrong_model,
        )
    assert dataset_calls == []


def test_interrupted_refit_retries_in_new_immutable_attempt(
    freeze_inputs: dict[str, object],
) -> None:
    config = _post_sweep_config(freeze_inputs)

    def incompatible_dataset(
        manifest: pd.DataFrame,
        root_dir: Path,
        *,
        folds: tuple[int, ...],
        normalization: NormalizationStats,
        protocol: ExperimentProtocol,
    ) -> Dataset[tuple[Tensor, Tensor]]:
        del manifest, root_dir, folds, normalization, protocol
        return TensorDataset(torch.randn(4, 3), torch.zeros(4, 5))

    with pytest.raises(RuntimeError):
        run_frozen_refit(
            config,
            protocol=ExperimentProtocol.canonical(),
            dataset_factory=incompatible_dataset,
            model_factory=_linear_model,
        )
    attempt_root = config.output.root_dir / config.run_name
    first = attempt_root / "attempt00"
    assert (first / "attempt_identity.json").is_file()
    assert not (first / "refit_completion.json").exists()

    result = run_frozen_refit(
        config,
        protocol=ExperimentProtocol.canonical(),
        dataset_factory=_tiny_dataset,
        model_factory=_linear_model,
    )
    assert result.run_dir == attempt_root / "attempt01"
    assert (first / "refit_metadata.json").is_file()
    assert result.completion_path is not None


def test_completion_cannot_rebind_drifted_manifest_away_from_freeze(
    freeze_inputs: dict[str, object],
) -> None:
    config = _post_sweep_config(freeze_inputs)
    result = run_frozen_refit(
        config,
        protocol=ExperimentProtocol.canonical(),
        dataset_factory=_tiny_dataset,
        model_factory=_linear_model,
    )
    assert result.completion_path is not None
    completion = json.loads(result.completion_path.read_text(encoding="utf-8"))
    manifest_path = Path(freeze_inputs["manifest"])
    frame = pd.read_csv(manifest_path)
    fold8_index = frame.index[frame["strat_fold"] == 8][0]
    frame.loc[fold8_index, TARGET_COLUMNS[0]] = 1 - int(
        frame.loc[fold8_index, TARGET_COLUMNS[0]]
    )
    frame.to_csv(manifest_path, index=False)
    drifted_hash = file_sha256(manifest_path)
    metadata_path = Path(completion["files"]["metadata"]["path"])
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["manifest_hash"] = drifted_hash.removeprefix("sha256:")
    _json(metadata_path, metadata)
    completion["manifest_hash"] = drifted_hash
    completion["files"]["manifest"]["sha256"] = drifted_hash
    completion["files"]["metadata"]["sha256"] = file_sha256(metadata_path)
    body = dict(completion)
    body.pop("artifact_sha256")
    completion["artifact_sha256"] = canonical_sha256(body)
    _json(result.completion_path, completion)
    with pytest.raises(FrozenRefitError, match="manifest hash differs from frozen"):
        load_refit_completion(
            result.completion_path,
            protocol=ExperimentProtocol.canonical(),
            verify_sources=True,
        )


def test_bundle_publication_preflights_all_destinations_and_commits_freeze_last(
    freeze_inputs: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ecg_trust.multiseed_freeze as freeze_module

    blocked_output = Path(freeze_inputs["tmp_path"]) / "blocked-freeze.json"
    blocked_recipes = Path(freeze_inputs["tmp_path"]) / "blocked-recipes"
    conflicting = blocked_recipes / "resnet1d-seed2026-refit.json"
    _json(conflicting, {"not": "the frozen recipe"})
    with pytest.raises(MultiSeedFreezeError, match="different content"):
        publish_multiseed_freeze_bundle(
            blocked_output,
            recipes_dir=blocked_recipes,
            sweep_summary_path=Path(freeze_inputs["sweep"]),
            member_completion_paths=list(freeze_inputs["completions"]),
            protocol=ExperimentProtocol.canonical(),
            refit_output_root=Path(freeze_inputs["tmp_path"]) / "refits",
            creation=freeze_inputs["creation"],  # type: ignore[arg-type]
        )
    assert not blocked_output.exists()
    assert list(blocked_recipes.iterdir()) == [conflicting]

    output = Path(freeze_inputs["tmp_path"]) / "published-freeze.json"
    recipes_dir = Path(freeze_inputs["tmp_path"]) / "published-recipes"
    writes: list[Path] = []
    original_write = freeze_module.write_new_json

    def tracking_write(path: Path, payload: Mapping[str, object]) -> None:
        writes.append(Path(path).resolve())
        original_write(path, payload)

    monkeypatch.setattr(freeze_module, "write_new_json", tracking_write)
    freeze, recipes = publish_multiseed_freeze_bundle(
        output,
        recipes_dir=recipes_dir,
        sweep_summary_path=Path(freeze_inputs["sweep"]),
        member_completion_paths=list(freeze_inputs["completions"]),
        protocol=ExperimentProtocol.canonical(),
        refit_output_root=Path(freeze_inputs["tmp_path"]) / "refits",
        creation=freeze_inputs["creation"],  # type: ignore[arg-type]
    )
    assert freeze.path == output.resolve()
    assert len(recipes) == 6
    assert writes[-1] == output.resolve()
