from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pandas as pd
import pytest
import torch
from torch import Tensor, nn
from torch.utils.data import Dataset, TensorDataset

from ecg_trust.constants import LEADS, TARGET_COLUMNS
from ecg_trust.data.dataset import NormalizationProvenance, NormalizationStats
from ecg_trust.experiment_config import (
    DevelopmentExperimentConfig,
    ExperimentConfigError,
    ModelConfig,
    load_experiment_config,
)
from ecg_trust.experiment_runner import (
    DevelopmentRunnerError,
    run_development_experiment,
    training_manifest_sha256,
)
from ecg_trust.protocol import TRAIN_FOLDS, ExperimentProtocol


def _config_payload(tmp_path: Path) -> dict[str, object]:
    return {
        "schema_version": 1,
        "run_name": "offline_wiring",
        "folds": {
            "train": [1, 2, 3, 4, 5, 6, 7],
            "model_selection": [8],
        },
        "data": {
            "manifest": "manifest.csv",
            "dataset_root": "records",
            "normalization": "normalization.json",
            "max_train_records": None,
            "max_validation_records": None,
        },
        "model": {"architecture": "resnet1d", "preset": "smoke"},
        "loader": {
            "batch_size": 4,
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
        },
        "optimization": {
            "epochs": 2,
            "learning_rate": 0.01,
            "weight_decay": 0.001,
            "warmup_epochs": 0,
            "minimum_lr_ratio": 0.1,
            "gradient_clip_norm": 1.0,
            "early_stopping_patience": 5,
            "early_stopping_min_delta": 0.0,
            "scheduler": "warmup_cosine",
        },
        "runtime": {"seed": 123, "device": "cpu", "bf16": True},
        "output": {"root_dir": str(tmp_path / "runs")},
    }


def _manifest_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for patient_id, record_path, fold in (
        (101, "records/train", 1),
        (202, "records/validation", 8),
    ):
        row: dict[str, object] = {
            "patient_id": patient_id,
            "record_path": record_path,
            "strat_fold": fold,
        }
        row.update(
            {
                target: int((fold + target_index) % 2 == 0)
                for target_index, target in enumerate(TARGET_COLUMNS)
            }
        )
        if not any(row[target] for target in TARGET_COLUMNS):
            row[TARGET_COLUMNS[0]] = 1
        rows.append(row)
    return pd.DataFrame(rows)


def _normalization(
    *,
    folds: tuple[int, ...] = TRAIN_FOLDS,
    manifest_sha256: str = "a" * 64,
) -> NormalizationStats:
    provenance = NormalizationProvenance(
        dataset_version="1.0.3",
        manifest_sha256=manifest_sha256,
        training_folds=folds,
        record_count=1,
        sample_count=1_000,
        sampling_frequency_hz=100.0,
        samples_per_record=1_000,
        path_column="record_path",
        fold_column="strat_fold",
        target_columns=TARGET_COLUMNS,
    )
    return NormalizationStats(
        mean=tuple(0.0 for _ in LEADS),
        std=tuple(1.0 for _ in LEADS),
        leads=LEADS,
        provenance=provenance,
    )


def _synthetic_dataset(sample_count: int) -> TensorDataset:
    generator = torch.Generator().manual_seed(91 + sample_count)
    inputs = torch.randn(sample_count, 4, generator=generator)
    indices = torch.arange(sample_count)
    targets = torch.stack(
        [((indices + label_index) % 2).float() for label_index in range(5)], dim=1
    )
    return TensorDataset(inputs, targets)


def test_config_rejects_any_fold_9_or_10_access(tmp_path: Path) -> None:
    payload = _config_payload(tmp_path)
    folds = cast(dict[str, object], payload["folds"])
    folds["model_selection"] = [8, 9]

    with pytest.raises(ExperimentConfigError, match="immutable"):
        DevelopmentExperimentConfig.from_mapping(payload, base_dir=tmp_path)


def test_bundled_configs_parse_and_matched_runs_share_training_budget() -> None:
    project_root = Path(__file__).resolve().parents[2]
    smoke = load_experiment_config(
        project_root / "configs" / "train_smoke.yaml", base_dir=project_root
    )
    resnet = load_experiment_config(
        project_root / "configs" / "train_resnet_matched.yaml", base_dir=project_root
    )
    transformer = load_experiment_config(
        project_root / "configs" / "train_transformer_matched.yaml",
        base_dir=project_root,
    )

    assert smoke.train_folds == TRAIN_FOLDS
    assert smoke.validation_folds == (8,)
    assert resnet.model.architecture == "resnet1d"
    assert transformer.model.architecture == "ecg_transformer"
    assert resnet.model.preset == transformer.model.preset == "matched_capacity"
    assert resnet.loader == transformer.loader
    assert resnet.optimization == transformer.optimization
    assert resnet.runtime == transformer.runtime
    assert resnet.config_hash.startswith("sha256:")


def test_runner_wires_only_development_folds_and_writes_auditable_artifacts(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.csv"
    manifest = _manifest_frame()
    manifest.to_csv(manifest_path, index=False)
    normalization_path = tmp_path / "normalization.json"
    _normalization(manifest_sha256=training_manifest_sha256(manifest, TRAIN_FOLDS)).save(
        normalization_path
    )
    config = DevelopmentExperimentConfig.from_mapping(
        _config_payload(tmp_path), base_dir=tmp_path
    )
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
        return _synthetic_dataset(14 if folds == TRAIN_FOLDS else 10)

    def model_factory(model_config: ModelConfig) -> nn.Module:
        assert model_config.architecture == "resnet1d"
        return nn.Linear(4, 5)

    protocol = ExperimentProtocol.canonical()
    result = run_development_experiment(
        config,
        protocol=protocol,
        dataset_factory=dataset_factory,
        model_factory=model_factory,
    )

    assert calls == [TRAIN_FOLDS, (8,)]
    assert all(fold < 9 for call in calls for fold in call)
    assert result.best_checkpoint_path.is_file()
    assert result.last_checkpoint_path.is_file()
    assert result.history_path.is_file()
    assert result.completed_epochs == 2
    assert 0.0 <= result.best_macro_auroc <= 1.0

    history = [
        json.loads(line)
        for line in result.history_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(history) == 2
    assert all(record["validation_metrics"]["n_samples"] == 10 for record in history)
    assert all(record["train_samples"] == 14 for record in history)
    assert all(record["train_samples_per_second"] > 0 for record in history)
    assert all(record["vram"]["peak_allocated_bytes"] == 0 for record in history)

    checkpoint = torch.load(result.last_checkpoint_path, weights_only=True)
    assert checkpoint["protocol_hash"] == protocol.protocol_hash
    assert checkpoint["manifest_hash"] == result.manifest_hash
    assert checkpoint["config_hash"] == result.resolved_config_hash
    assert checkpoint["early_stopping_state_dict"] is not None
    assert checkpoint["scaler_state_dict"] is None

    resolved = json.loads(
        (result.run_dir / "resolved_config.json").read_text(encoding="utf-8")
    )
    metadata = json.loads((result.run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    recorded_protocol = json.loads(
        (result.run_dir / "protocol.json").read_text(encoding="utf-8")
    )
    assert resolved["config_hash"] == result.resolved_config_hash
    assert resolved["config"]["model"]["trainable_parameters"] == 25
    assert metadata["status"] == "complete"
    assert metadata["seed"] == 123
    assert metadata["runtime"]["device"] == "cpu"
    assert metadata["runtime"]["bf16_autocast"] is False
    assert metadata["manifest_hash"] == result.manifest_hash
    assert metadata["overall_samples_per_second"] > 0
    assert metadata["normalization_provenance"]["training_folds"] == list(TRAIN_FOLDS)
    assert recorded_protocol["protocol_hash"] == protocol.protocol_hash


def test_runner_rejects_normalization_contaminated_by_fold_9(tmp_path: Path) -> None:
    _manifest_frame().to_csv(tmp_path / "manifest.csv", index=False)
    _normalization(folds=(*TRAIN_FOLDS, 9)).save(tmp_path / "normalization.json")
    config = DevelopmentExperimentConfig.from_mapping(
        _config_payload(tmp_path), base_dir=tmp_path
    )
    dataset_called = False

    def forbidden_dataset_factory(
        manifest: pd.DataFrame,
        root_dir: Path,
        *,
        folds: tuple[int, ...],
        normalization: NormalizationStats,
        protocol: ExperimentProtocol,
    ) -> Dataset[tuple[Tensor, Tensor]]:
        del manifest, root_dir, folds, normalization, protocol
        nonlocal dataset_called
        dataset_called = True
        return _synthetic_dataset(10)

    with pytest.raises(DevelopmentRunnerError, match="exclusively"):
        run_development_experiment(
            config,
            protocol=ExperimentProtocol.canonical(),
            dataset_factory=forbidden_dataset_factory,
            model_factory=lambda _: nn.Linear(4, 5),
        )
    assert not dataset_called


def test_runner_rejects_patient_leakage_before_dataset_construction(tmp_path: Path) -> None:
    manifest = _manifest_frame()
    manifest.loc[:, "patient_id"] = 101
    manifest.to_csv(tmp_path / "manifest.csv", index=False)
    _normalization().save(tmp_path / "normalization.json")
    config = DevelopmentExperimentConfig.from_mapping(
        _config_payload(tmp_path), base_dir=tmp_path
    )
    dataset_called = False

    def forbidden_dataset_factory(
        manifest: pd.DataFrame,
        root_dir: Path,
        *,
        folds: tuple[int, ...],
        normalization: NormalizationStats,
        protocol: ExperimentProtocol,
    ) -> Dataset[tuple[Tensor, Tensor]]:
        del manifest, root_dir, folds, normalization, protocol
        nonlocal dataset_called
        dataset_called = True
        return _synthetic_dataset(10)

    with pytest.raises(DevelopmentRunnerError, match="multiple folds"):
        run_development_experiment(
            config,
            protocol=ExperimentProtocol.canonical(),
            dataset_factory=forbidden_dataset_factory,
            model_factory=lambda _: nn.Linear(4, 5),
        )
    assert not dataset_called


def test_runner_rejects_normalization_from_a_different_manifest(tmp_path: Path) -> None:
    manifest = _manifest_frame()
    manifest.to_csv(tmp_path / "manifest.csv", index=False)
    _normalization(manifest_sha256="f" * 64).save(tmp_path / "normalization.json")
    config = DevelopmentExperimentConfig.from_mapping(
        _config_payload(tmp_path), base_dir=tmp_path
    )

    with pytest.raises(DevelopmentRunnerError, match="current training manifest"):
        run_development_experiment(
            config,
            protocol=ExperimentProtocol.canonical(),
            dataset_factory=lambda *args, **kwargs: _synthetic_dataset(10),
            model_factory=lambda _: nn.Linear(4, 5),
        )
