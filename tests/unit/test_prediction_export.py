from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import pandas as pd
import pytest
import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import Dataset

import ecg_trust.refit_runner as refit_runner_module
from ecg_trust.constants import LEADS, TARGET_COLUMNS
from ecg_trust.data.dataset import NormalizationProvenance, NormalizationStats
from ecg_trust.data.manifest import sha256_file
from ecg_trust.experiment_config import ModelConfig
from ecg_trust.experiment_runner import build_experiment_model, training_manifest_sha256
from ecg_trust.models import count_parameters
from ecg_trust.prediction_export import (
    PredictionExportError,
    PredictionExportRequest,
    export_checkpoint_predictions,
)
from ecg_trust.predictions import load_prediction_artifact
from ecg_trust.protocol import (
    FINAL_TEST_CONFIRMATION,
    TRAIN_FOLDS,
    ExperimentProtocol,
    FinalTestAccessError,
    FoldRole,
    authorize_final_test_access,
)
from ecg_trust.training import EarlyStopping, save_checkpoint


@dataclass(frozen=True)
class _Bundle:
    checkpoint: Path
    resolved_config: Path
    manifest: Path
    normalization: Path
    dataset_root: Path


class _SyntheticECGs(Dataset[tuple[Tensor, Tensor]]):
    def __init__(self, manifest: pd.DataFrame) -> None:
        self.manifest = manifest.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        signal = torch.full((12, 1_000), float(index) / 100.0, dtype=torch.float32)
        target = torch.tensor(
            self.manifest.loc[index, list(TARGET_COLUMNS)].to_numpy(dtype="float32"),
            dtype=torch.float32,
        )
        return signal, target


def _canonical(value: dict[str, object]) -> str:
    serialized = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return "sha256:" + hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _manifest(*, mixed_patient: bool = False) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for fold in range(1, 11):
        row: dict[str, object] = {
            "ecg_id": 10_000 + fold,
            "patient_id": 500 if mixed_patient else 500 + fold,
            "strat_fold": fold,
            "record_path": f"records/fold_{fold}",
        }
        row.update(
            {
                target: int((fold + index) % 2 == 0)
                for index, target in enumerate(TARGET_COLUMNS)
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _stats(manifest: pd.DataFrame) -> NormalizationStats:
    count = int(manifest["strat_fold"].isin(TRAIN_FOLDS).sum())
    return NormalizationStats(
        mean=tuple(0.0 for _ in LEADS),
        std=tuple(1.0 for _ in LEADS),
        leads=LEADS,
        provenance=NormalizationProvenance(
            dataset_version="1.0.3",
            manifest_sha256=training_manifest_sha256(manifest, TRAIN_FOLDS),
            training_folds=TRAIN_FOLDS,
            record_count=count,
            sample_count=count * 1_000,
            sampling_frequency_hz=100.0,
            samples_per_record=1_000,
            path_column="record_path",
            fold_column="strat_fold",
            target_columns=TARGET_COLUMNS,
        ),
    )


def _model_metadata(model: nn.Module, selection: ModelConfig) -> dict[str, object]:
    return {
        "architecture": selection.architecture,
        "preset": selection.preset,
        "class": f"{type(model).__module__}.{type(model).__qualname__}",
        "trainable_parameters": count_parameters(model),
        "resolved_architecture_config": asdict(model.config),  # type: ignore[attr-defined]
    }


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _bundle(
    tmp_path: Path,
    *,
    lineage: str,
    architecture: str,
    mixed_patient: bool = False,
    manifest_format: str = "parquet",
) -> _Bundle:
    run_dir = tmp_path / lineage
    run_dir.mkdir(parents=True)
    manifest = _manifest(mixed_patient=mixed_patient)
    manifest_path = run_dir / f"manifest.{manifest_format}"
    if manifest_format == "parquet":
        manifest.to_parquet(manifest_path, index=False)
    elif manifest_format == "csv":
        manifest.to_csv(manifest_path, index=False)
    else:
        raise AssertionError(f"unsupported test manifest format: {manifest_format}")
    normalization_path = run_dir / "normalization.json"
    stats = _stats(manifest)
    stats.save(normalization_path)
    dataset_root = run_dir / "records"
    dataset_root.mkdir()
    selection = ModelConfig.from_mapping(
        {"architecture": architecture, "preset": "smoke"}
    )
    model = build_experiment_model(selection)
    model_metadata = _model_metadata(model, selection)
    loader = {
        "batch_size": 2,
        "num_workers": 0,
        "pin_memory": False,
        "persistent_workers": False,
    }
    runtime = {"seed": 71, "device": "cpu", "bf16": False}
    output = {"root_dir": str(run_dir / "outputs")}
    if lineage == "development":
        config: dict[str, object] = {
            "schema_version": 1,
            "run_name": "synthetic_dev_resnet",
            "folds": {
                "train": list(TRAIN_FOLDS),
                "model_selection": [8],
            },
            "data": {
                "manifest": str(manifest_path),
                "dataset_root": str(dataset_root),
                "normalization": str(normalization_path),
                "max_train_records": None,
                "max_validation_records": None,
            },
            "model": model_metadata,
            "loader": loader,
            "optimization": {
                "epochs": 2,
                "learning_rate": 0.001,
                "weight_decay": 0.01,
                "warmup_epochs": 0,
                "minimum_lr_ratio": 0.1,
                "gradient_clip_norm": 1.0,
                "early_stopping_patience": 2,
                "early_stopping_min_delta": 0.0,
                "scheduler": "warmup_cosine",
            },
            "runtime": runtime,
            "output": output,
            "effective_data": {"train_records": 7, "validation_records": 1},
            "optimizer": {"name": "AdamW", "betas": [0.9, 0.999], "eps": 1e-8},
        }
        checkpoint_path = run_dir / "best.ckpt"
        resolved_path = run_dir / "resolved_config.json"
        metadata_path = run_dir / "run_metadata.json"
        epoch = 0
        stopper: EarlyStopping | None = EarlyStopping(patience=2, mode="max")
        stopper.update(0.75, epoch)
    else:
        selection_provenance: dict[str, object] = {
            "checkpoint": "synthetic_development.ckpt",
            "checkpoint_sha256": "a" * 64,
            "checkpoint_config_hash": "sha256:" + "b" * 64,
            "selected_epoch": 1,
            "selected_epoch_count": 2,
            "selected_macro_auroc": 0.75,
        }
        config = {
            "schema_version": 1,
            "run_kind": "frozen_refit",
            "run_name": "synthetic_refit_transformer",
            "folds": {
                "refit": list(range(1, 9)),
                "normalization": list(TRAIN_FOLDS),
            },
            "data": {
                "manifest": str(manifest_path),
                "dataset_root": str(dataset_root),
                "normalization": str(normalization_path),
            },
            "selection": {
                "development_checkpoint": str(run_dir / "synthetic_development.ckpt"),
                "selection_metric": "fold8_macro_auroc",
                "frozen_epochs": 2,
            },
            "model": model_metadata,
            "loader": loader,
            "optimization": {
                "learning_rate": 0.001,
                "weight_decay": 0.01,
                "warmup_epochs": 0,
                "minimum_lr_ratio": 0.1,
                "gradient_clip_norm": 1.0,
                "scheduler": "warmup_cosine",
            },
            "runtime": runtime,
            "output": output,
            "selection_provenance": selection_provenance,
            "effective_data": {"refit_records": 8},
            "checkpoint_roles": {
                "best_training_loss.ckpt": "diagnostic minimum training loss only",
                "last.ckpt": "crash-recovery state from the latest completed epoch",
                "final.ckpt": "authoritative frozen-epoch refit artifact",
            },
        }
        checkpoint_path = run_dir / "final.ckpt"
        resolved_path = run_dir / "resolved_refit_config.json"
        metadata_path = run_dir / "refit_metadata.json"
        epoch = 1
        stopper = None

    config_hash = _canonical(config)
    _write_json(resolved_path, {"config_hash": config_hash, "config": config})
    optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    checkpoint = save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        scaler=None,
        epoch=epoch,
        protocol_hash=ExperimentProtocol.canonical().protocol_hash,
        config=config,
        manifest_hash=sha256_file(manifest_path),
        early_stopping=stopper,
    )
    assert checkpoint.config_hash == config_hash
    common_metadata: dict[str, object] = {
        "status": "complete",
        "seed": 71,
        "resolved_config_hash": config_hash,
        "protocol_hash": ExperimentProtocol.canonical().protocol_hash,
        "manifest_hash": sha256_file(manifest_path),
        "normalization_file_hash": sha256_file(normalization_path),
        "normalization_provenance": stats.provenance.to_dict(),
    }
    if lineage == "development":
        common_metadata.update({"best_epoch": 0, "completed_epochs": 1})
    else:
        common_metadata.update(
            {
                "run_kind": "frozen_refit",
                "refit_folds": list(range(1, 9)),
                "normalization_folds": list(TRAIN_FOLDS),
                "frozen_epochs": 2,
                "completed_epochs": 2,
                "early_stopping_enabled": False,
                "model_selection_enabled": False,
                "authoritative_checkpoint": "final.ckpt",
                "final_epoch": 1,
                "selection_provenance": cast(
                    dict[str, object], config["selection_provenance"]
                ),
            }
        )
    _write_json(metadata_path, common_metadata)
    return _Bundle(
        checkpoint=checkpoint_path,
        resolved_config=resolved_path,
        manifest=manifest_path,
        normalization=normalization_path,
        dataset_root=dataset_root,
    )


def _post_sweep_bundle(tmp_path: Path) -> tuple[_Bundle, dict[str, object]]:
    run_dir = tmp_path / "post-sweep" / "attempt00"
    run_dir.mkdir(parents=True)
    manifest = _manifest()
    manifest_path = run_dir / "manifest.parquet"
    manifest.to_parquet(manifest_path, index=False)
    normalization_path = run_dir / "normalization.json"
    stats = _stats(manifest)
    stats.save(normalization_path)
    dataset_root = run_dir / "records"
    dataset_root.mkdir()

    selection = ModelConfig.from_mapping(
        {"architecture": "resnet1d", "preset": "smoke"}
    )
    model = build_experiment_model(selection)
    model_metadata = _model_metadata(model, selection)
    freeze_path = tmp_path / "multi-seed-freeze.json"
    freeze_path.write_text("{}\n", encoding="utf-8")
    fake_source = tmp_path / "source"
    fake_hash = "sha256:" + "a" * 64
    source = {
        "member_completion": str(fake_source / "member_completion.json"),
        "member_completion_sha256": fake_hash,
        "manifest_sha256": "sha256:" + sha256_file(manifest_path),
        "normalization_sha256": "sha256:" + sha256_file(normalization_path),
        "run_metadata": str(fake_source / "run_metadata.json"),
        "run_metadata_sha256": fake_hash,
        "resolved_config": str(fake_source / "resolved_config.json"),
        "resolved_config_file_sha256": fake_hash,
        "resolved_config_hash": fake_hash,
        "history": str(fake_source / "history.jsonl"),
        "history_sha256": fake_hash,
        "best_checkpoint": str(fake_source / "best.ckpt"),
        "best_checkpoint_sha256": fake_hash,
        "prediction": str(fake_source / "fold8.npz"),
        "prediction_npz_sha256": fake_hash,
        "prediction_json": str(fake_source / "fold8.json"),
        "prediction_artifact_sha256": fake_hash,
        "best_epoch": 1,
        "best_validation_macro_auroc": 0.91,
    }
    freeze_hash = "sha256:" + "b" * 64
    recipe_hash = "sha256:" + "c" * 64
    downstream = {
        "project_root": str(tmp_path.resolve()),
        "code_revision": "d" * 40,
        "dependency_lock_sha256": "sha256:" + "e" * 64,
    }
    selection_provenance = {
        "checkpoint": str(fake_source / "best.ckpt"),
        "checkpoint_sha256": "a" * 64,
        "checkpoint_config_hash": fake_hash,
        "selected_epoch": 1,
        "selected_epoch_count": 2,
        "selected_macro_auroc": 0.91,
        "source_seed": 71,
        "member_completion_sha256": fake_hash,
        "freeze_artifact_sha256": freeze_hash,
        "recipe_sha256": recipe_hash,
    }
    checkpoint_roles = {
        "best_training_loss.ckpt": "diagnostic minimum training loss only",
        "last.ckpt": "crash-recovery state from the latest completed epoch",
        "final.ckpt": "authoritative frozen-epoch refit artifact",
    }
    config: dict[str, object] = {
        "schema_version": 2,
        "run_kind": "post_sweep_frozen_refit",
        "freeze_artifact": str(freeze_path.resolve()),
        "freeze_artifact_sha256": freeze_hash,
        "recipe_sha256": recipe_hash,
        "comparison_id": "synthetic-comparison-v1",
        "architecture": "resnet1d",
        "confirmation_seed": 71,
        "run_name": "synthetic_post_sweep_resnet",
        "initialization": "fresh",
        "folds": {
            "refit": list(range(1, 9)),
            "normalization": list(TRAIN_FOLDS),
        },
        "data": {
            "manifest": str(manifest_path),
            "dataset_root": str(dataset_root),
            "normalization": str(normalization_path),
        },
        "source": source,
        "selection": {
            "objective": "fold8_uncalibrated_macro_roc_auc",
            "architecture_mean_macro_auroc": 0.91,
            "frozen_epochs": 2,
            "epoch_budget_rule": (
                "max(warmup_epochs+1,median("
                "selected_zero_based_best_epoch+1_across_seeds))"
            ),
        },
        "model": model_metadata,
        "model_identity": model_metadata,
        "loader": {
            "batch_size": 2,
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
        },
        "optimization": {
            "learning_rate": 0.001,
            "weight_decay": 0.01,
            "warmup_epochs": 0,
            "minimum_lr_ratio": 0.1,
            "gradient_clip_norm": 1.0,
            "scheduler": "warmup_cosine",
        },
        "optimizer": {"name": "AdamW", "betas": [0.8, 0.95], "eps": 1e-7},
        "runtime": {"seed": 71, "device": "cpu", "bf16": False},
        "output": {"root_dir": str(tmp_path / "refits")},
        "downstream_provenance": downstream,
        "selection_provenance": selection_provenance,
        "freeze_binding": {
            "path": str(freeze_path.resolve()),
            "artifact_sha256": freeze_hash,
            "comparison_id": "synthetic-comparison-v1",
            "recipe_sha256": recipe_hash,
        },
        "attempt_index": 0,
        "effective_data": {"refit_records": 8},
        "checkpoint_roles": checkpoint_roles,
    }
    config_hash = _canonical(config)
    resolved_path = run_dir / "resolved_refit_config.json"
    _write_json(resolved_path, {"config_hash": config_hash, "config": config})
    checkpoint_path = run_dir / "final.ckpt"
    optimizer = AdamW(
        model.parameters(),
        lr=0.001,
        betas=(0.8, 0.95),
        eps=1e-7,
        weight_decay=0.01,
    )
    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        scaler=None,
        epoch=1,
        protocol_hash=ExperimentProtocol.canonical().protocol_hash,
        config=config,
        manifest_hash=sha256_file(manifest_path),
        early_stopping=None,
    )
    metadata_path = run_dir / "refit_metadata.json"
    metadata = {
        "status": "complete",
        "run_kind": "post_sweep_frozen_refit",
        "seed": 71,
        "refit_folds": list(range(1, 9)),
        "normalization_folds": list(TRAIN_FOLDS),
        "frozen_epochs": 2,
        "completed_epochs": 2,
        "early_stopping_enabled": False,
        "model_selection_enabled": False,
        "authoritative_checkpoint": "final.ckpt",
        "final_epoch": 1,
        "resolved_config_hash": config_hash,
        "protocol_hash": ExperimentProtocol.canonical().protocol_hash,
        "manifest_hash": sha256_file(manifest_path),
        "normalization_file_hash": sha256_file(normalization_path),
        "normalization_provenance": stats.provenance.to_dict(),
        "selection_provenance": selection_provenance,
        "comparison_id": "synthetic-comparison-v1",
        "architecture": "resnet1d",
        "confirmation_seed": 71,
        "freeze_artifact_path": str(freeze_path.resolve()),
        "freeze_artifact_sha256": freeze_hash,
        "recipe_sha256": recipe_hash,
        "initialization": "fresh",
        "attempt_index": 0,
        "downstream_provenance": downstream,
        "final_checkpoint_sha256": "sha256:" + sha256_file(checkpoint_path),
    }
    _write_json(metadata_path, metadata)

    def entry(path: Path) -> dict[str, object]:
        return {"path": str(path.resolve()), "sha256": "sha256:" + sha256_file(path)}

    completion: dict[str, object] = {
        "comparison_id": "synthetic-comparison-v1",
        "architecture": "resnet1d",
        "seed": 71,
        "status": "complete",
        "run_name": "synthetic_post_sweep_resnet",
        "run_dir": str(run_dir.resolve()),
        "freeze_artifact_path": str(freeze_path.resolve()),
        "freeze_artifact_sha256": freeze_hash,
        "recipe_sha256": recipe_hash,
        "refit_folds": list(range(1, 9)),
        "normalization_folds": list(TRAIN_FOLDS),
        "frozen_epochs": 2,
        "protocol_hash": ExperimentProtocol.canonical().protocol_hash,
        "manifest_hash": "sha256:" + sha256_file(manifest_path),
        "normalization_hash": "sha256:" + sha256_file(normalization_path),
        "downstream_provenance": downstream,
        "selection_provenance": selection_provenance,
        "files": {
            "final_checkpoint": entry(checkpoint_path),
            "resolved_config": {
                **entry(resolved_path),
                "config_hash": config_hash,
            },
            "metadata": entry(metadata_path),
            "manifest": entry(manifest_path),
            "normalization": entry(normalization_path),
        },
        "artifact_sha256": "sha256:" + "f" * 64,
    }
    _write_json(run_dir / "refit_completion.json", completion)
    return (
        _Bundle(
            checkpoint=checkpoint_path,
            resolved_config=resolved_path,
            manifest=manifest_path,
            normalization=normalization_path,
            dataset_root=dataset_root,
        ),
        completion,
    )


def _factory(calls: list[tuple[int, ...]]):
    def factory(
        manifest: pd.DataFrame,
        root_dir: Path,
        *,
        folds: tuple[int, ...],
        normalization: NormalizationStats,
        protocol: ExperimentProtocol,
        test_access: object,
    ) -> Dataset[tuple[Tensor, Tensor]]:
        del root_dir, normalization, protocol, test_access
        calls.append(folds)
        return _SyntheticECGs(manifest)

    return factory


def _request(bundle: _Bundle, role: FoldRole, output: Path) -> PredictionExportRequest:
    return PredictionExportRequest(
        checkpoint_path=bundle.checkpoint,
        resolved_config_path=bundle.resolved_config,
        manifest_path=bundle.manifest,
        dataset_root=bundle.dataset_root,
        normalization_path=bundle.normalization,
        output_path=output,
        fold_role=role,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        device="cpu",
        bf16=False,
    )


def test_development_best_checkpoint_exports_only_fold_8(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path, lineage="development", architecture="resnet1d")
    calls: list[tuple[int, ...]] = []
    protocol = ExperimentProtocol.canonical()
    result = export_checkpoint_predictions(
        _request(bundle, FoldRole.MODEL_SELECTION, tmp_path / "fold8.npz"),
        protocol=protocol,
        dataset_factory=_factory(calls),
    )

    artifact = load_prediction_artifact(result.files.npz_path, protocol=protocol)
    assert calls == [(8,)]
    assert result.lineage == "development"
    assert artifact.fold_role is FoldRole.MODEL_SELECTION
    assert artifact.folds == (8,)
    assert artifact.n_samples == 1
    assert artifact.model_seed == 71
    assert artifact.extra_metadata["checkpoint_epoch"] == 0
    assert artifact.raw_logits.shape == (1, 5)


def test_frozen_refit_final_checkpoint_exports_fold_9(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path, lineage="frozen_refit", architecture="ecg_transformer")
    calls: list[tuple[int, ...]] = []
    protocol = ExperimentProtocol.canonical()
    result = export_checkpoint_predictions(
        _request(bundle, FoldRole.CALIBRATION, tmp_path / "fold9.npz"),
        protocol=protocol,
        dataset_factory=_factory(calls),
    )

    artifact = load_prediction_artifact(result.files.json_path, protocol=protocol)
    assert calls == [(9,)]
    assert result.lineage == "frozen_refit"
    assert artifact.fold_role is FoldRole.CALIBRATION
    assert artifact.folds == (9,)
    assert artifact.extra_metadata["checkpoint_epoch"] == 1
    assert artifact.extra_metadata["inference_device"] == "cpu"


def test_final_export_requires_protocol_token(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path, lineage="frozen_refit", architecture="resnet1d")
    protocol = ExperimentProtocol.canonical()
    request = _request(bundle, FoldRole.FINAL_TEST, tmp_path / "fold10.npz")
    with pytest.raises(FinalTestAccessError, match="sealed"):
        export_checkpoint_predictions(
            request,
            protocol=protocol,
            dataset_factory=_factory([]),
        )

    token = authorize_final_test_access(
        protocol,
        purpose="synthetic offline exporter test",
        confirmation=FINAL_TEST_CONFIRMATION,
    )
    calls: list[tuple[int, ...]] = []
    result = export_checkpoint_predictions(
        request,
        protocol=protocol,
        test_access=token,
        dataset_factory=_factory(calls),
    )
    artifact = load_prediction_artifact(
        result.files.npz_path,
        protocol=protocol,
        test_access=token,
    )
    assert calls == [(10,)]
    assert artifact.fold_role is FoldRole.FINAL_TEST
    assert artifact.extra_metadata["final_test_purpose"] == "synthetic offline exporter test"


def test_export_rejects_wrong_lineage_before_dataset_creation(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path, lineage="development", architecture="resnet1d")
    calls: list[tuple[int, ...]] = []
    with pytest.raises(PredictionExportError, match="may export only"):
        export_checkpoint_predictions(
            _request(bundle, FoldRole.CALIBRATION, tmp_path / "wrong.npz"),
            protocol=ExperimentProtocol.canonical(),
            dataset_factory=_factory(calls),
        )
    assert calls == []


def test_export_rejects_normalization_tampering_and_mixed_patients(
    tmp_path: Path,
) -> None:
    tampered = _bundle(tmp_path / "tampered", lineage="development", architecture="resnet1d")
    payload = json.loads(tampered.normalization.read_text(encoding="utf-8"))
    payload["mean"][0] = 0.5
    _write_json(tampered.normalization, payload)
    with pytest.raises(PredictionExportError, match="normalization_file_hash"):
        export_checkpoint_predictions(
            _request(tampered, FoldRole.MODEL_SELECTION, tmp_path / "tampered.npz"),
            protocol=ExperimentProtocol.canonical(),
            dataset_factory=_factory([]),
        )

    mixed = _bundle(
        tmp_path / "mixed",
        lineage="development",
        architecture="resnet1d",
        mixed_patient=True,
    )
    with pytest.raises(PredictionExportError, match="multiple folds"):
        export_checkpoint_predictions(
            _request(mixed, FoldRole.MODEL_SELECTION, tmp_path / "mixed.npz"),
            protocol=ExperimentProtocol.canonical(),
            dataset_factory=_factory([]),
        )


def test_export_rejects_model_metadata_drift_and_overwrite(tmp_path: Path) -> None:
    drifted = _bundle(tmp_path / "drift", lineage="development", architecture="resnet1d")
    wrapper = json.loads(drifted.resolved_config.read_text(encoding="utf-8"))
    wrapper["config"]["model"]["trainable_parameters"] += 1
    wrapper["config_hash"] = _canonical(wrapper["config"])
    _write_json(drifted.resolved_config, wrapper)
    metadata_path = drifted.resolved_config.parent / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["resolved_config_hash"] = wrapper["config_hash"]
    _write_json(metadata_path, metadata)
    with pytest.raises(PredictionExportError, match="trainable_parameters"):
        export_checkpoint_predictions(
            _request(drifted, FoldRole.MODEL_SELECTION, tmp_path / "drift.npz"),
            protocol=ExperimentProtocol.canonical(),
            dataset_factory=_factory([]),
        )

    valid = _bundle(tmp_path / "valid", lineage="development", architecture="resnet1d")
    request = _request(valid, FoldRole.MODEL_SELECTION, tmp_path / "immutable.npz")
    export_checkpoint_predictions(
        request,
        protocol=ExperimentProtocol.canonical(),
        dataset_factory=_factory([]),
    )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        export_checkpoint_predictions(
            request,
            protocol=ExperimentProtocol.canonical(),
            dataset_factory=_factory([]),
        )


def test_post_sweep_refit_binds_completion_and_keeps_public_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, completion = _post_sweep_bundle(tmp_path)
    completion_calls: list[tuple[Path, bool]] = []

    def load_completion(
        path: str | Path,
        *,
        protocol: ExperimentProtocol,
        verify_sources: bool = True,
    ) -> dict[str, object]:
        assert protocol == ExperimentProtocol.canonical()
        completion_calls.append((Path(path).resolve(), verify_sources))
        return completion

    monkeypatch.setattr(refit_runner_module, "load_refit_completion", load_completion)
    calls: list[tuple[int, ...]] = []
    protocol = ExperimentProtocol.canonical()
    result = export_checkpoint_predictions(
        _request(bundle, FoldRole.CALIBRATION, tmp_path / "post-sweep-fold9.npz"),
        protocol=protocol,
        dataset_factory=_factory(calls),
    )

    artifact = load_prediction_artifact(result.files.npz_path, protocol=protocol)
    assert completion_calls == [
        (bundle.resolved_config.parent / "refit_completion.json", True)
    ]
    assert calls == [(9,)]
    assert result.lineage == "frozen_refit"
    assert artifact.extra_metadata["lineage"] == "frozen_refit"
    assert (
        artifact.extra_metadata["refit_run_kind"]
        == "post_sweep_frozen_refit"
    )
    assert (
        artifact.extra_metadata["refit_completion_sha256"]
        == completion["artifact_sha256"]
    )
    assert artifact.extra_metadata["freeze_artifact_sha256"] == (
        completion["freeze_artifact_sha256"]
    )


def test_post_sweep_refit_rejects_completion_drift_before_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, completion = _post_sweep_bundle(tmp_path)
    drifted = json.loads(json.dumps(completion))
    drifted["downstream_provenance"]["code_revision"] = "tampered"
    monkeypatch.setattr(
        refit_runner_module,
        "load_refit_completion",
        lambda *args, **kwargs: drifted,
    )
    calls: list[tuple[int, ...]] = []
    with pytest.raises(PredictionExportError, match="downstream_provenance"):
        export_checkpoint_predictions(
            _request(bundle, FoldRole.CALIBRATION, tmp_path / "drifted-fold9.npz"),
            protocol=ExperimentProtocol.canonical(),
            dataset_factory=_factory(calls),
        )
    assert calls == []


def test_fold9_parquet_reads_no_fold10_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _bundle(
        tmp_path,
        lineage="frozen_refit",
        architecture="resnet1d",
    )
    original = pd.read_parquet
    reads: list[dict[str, object]] = []

    def guarded_read_parquet(*args: object, **kwargs: object) -> pd.DataFrame:
        reads.append(dict(kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", guarded_read_parquet)
    export_checkpoint_predictions(
        _request(bundle, FoldRole.CALIBRATION, tmp_path / "guarded-parquet.npz"),
        protocol=ExperimentProtocol.canonical(),
        dataset_factory=_factory([]),
    )

    target_reads = [
        call
        for call in reads
        if set(TARGET_COLUMNS).intersection(cast(list[str], call.get("columns", [])))
    ]
    assert len(target_reads) == 1
    assert target_reads[0]["filters"] == [
        ("strat_fold", "in", [1, 2, 3, 4, 5, 6, 7, 9])
    ]
    identity_reads = [call for call in reads if call not in target_reads]
    assert identity_reads
    assert all(
        not set(TARGET_COLUMNS).intersection(
            cast(list[str], call.get("columns", []))
        )
        for call in identity_reads
    )


def test_fold9_csv_skips_fold10_before_loading_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _bundle(
        tmp_path,
        lineage="frozen_refit",
        architecture="resnet1d",
        manifest_format="csv",
    )
    original = pd.read_csv
    reads: list[dict[str, object]] = []

    def guarded_read_csv(*args: object, **kwargs: object) -> pd.DataFrame:
        reads.append(dict(kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", guarded_read_csv)
    export_checkpoint_predictions(
        _request(bundle, FoldRole.CALIBRATION, tmp_path / "guarded-csv.npz"),
        protocol=ExperimentProtocol.canonical(),
        dataset_factory=_factory([]),
    )

    target_reads = [
        call
        for call in reads
        if set(TARGET_COLUMNS).intersection(cast(list[str], call.get("usecols", [])))
    ]
    assert len(target_reads) == 1
    skiprows = target_reads[0]["skiprows"]
    assert callable(skiprows)
    assert skiprows(0) is False
    assert skiprows(9) is False
    assert skiprows(10) is True
