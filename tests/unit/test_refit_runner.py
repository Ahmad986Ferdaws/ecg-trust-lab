from __future__ import annotations

import copy
import json
from pathlib import Path

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
from ecg_trust.protocol import TRAIN_FOLDS, ExperimentProtocol
from ecg_trust.refit_config import (
    REFIT_FOLDS,
    FrozenRefitConfig,
    RefitConfigError,
    load_refit_config,
)
from ecg_trust.refit_runner import FrozenRefitError, run_frozen_refit
from ecg_trust.training import EarlyStopping, save_checkpoint


def _payload(tmp_path: Path, *, frozen_epochs: int = 2) -> dict[str, object]:
    return {
        "schema_version": 1,
        "run_kind": "frozen_refit",
        "run_name": "offline_frozen_refit",
        "folds": {
            "refit": [1, 2, 3, 4, 5, 6, 7, 8],
            "normalization": [1, 2, 3, 4, 5, 6, 7],
        },
        "data": {
            "manifest": "manifest.csv",
            "dataset_root": "records",
            "normalization": "normalization.json",
        },
        "selection": {
            "development_checkpoint": "selected_development.ckpt",
            "selection_metric": "fold8_macro_auroc",
            "frozen_epochs": frozen_epochs,
        },
        "model": {"architecture": "resnet1d", "preset": "smoke"},
        "loader": {
            "batch_size": 4,
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
        },
        "optimization": {
            "learning_rate": 0.01,
            "weight_decay": 0.001,
            "warmup_epochs": 0,
            "minimum_lr_ratio": 0.1,
            "gradient_clip_norm": 1.0,
            "scheduler": "warmup_cosine",
        },
        "runtime": {"seed": 314, "device": "cpu", "bf16": True},
        "output": {"root_dir": str(tmp_path / "runs" / "refit")},
    }


def _manifest() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for patient_id, record_path, fold in (
        (101, "records/train", 1),
        (202, "records/model_selection", 8),
        (303, "records/calibration", 9),
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
        rows.append(row)
    return pd.DataFrame(rows)


def _stats(
    manifest: pd.DataFrame,
    *,
    training_folds: tuple[int, ...] = TRAIN_FOLDS,
) -> NormalizationStats:
    provenance = NormalizationProvenance(
        dataset_version="1.0.3",
        manifest_sha256=training_manifest_sha256(manifest, training_folds),
        training_folds=training_folds,
        record_count=sum(int(fold in training_folds) for fold in manifest["strat_fold"]),
        sample_count=(
            1_000 * sum(int(fold in training_folds) for fold in manifest["strat_fold"])
        ),
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


def _development_config() -> dict[str, object]:
    return {
        "model": {"architecture": "resnet1d", "preset": "smoke"},
        "loader": {"batch_size": 4},
        "optimization": {
            "epochs": 20,
            "learning_rate": 0.01,
            "weight_decay": 0.001,
            "warmup_epochs": 0,
            "minimum_lr_ratio": 0.1,
            "gradient_clip_norm": 1.0,
            "scheduler": "warmup_cosine",
            "early_stopping_patience": 5,
            "early_stopping_min_delta": 0.0,
        },
        "runtime": {"seed": 777, "device": "cpu", "bf16": False},
    }


def _write_inputs(
    tmp_path: Path,
    *,
    normalization_folds: tuple[int, ...] = TRAIN_FOLDS,
    selected_epoch: int = 1,
    development_config: dict[str, object] | None = None,
) -> None:
    manifest = _manifest()
    manifest_path = tmp_path / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    _stats(manifest, training_folds=normalization_folds).save(
        tmp_path / "normalization.json"
    )
    model = nn.Linear(4, 5)
    optimizer = AdamW(model.parameters(), lr=0.01)
    stopper = EarlyStopping(patience=5, mode="max")
    for epoch in range(selected_epoch + 1):
        stopper.update(0.60 + epoch * 0.05, epoch)
    save_checkpoint(
        tmp_path / "selected_development.ckpt",
        model=model,
        optimizer=optimizer,
        scaler=None,
        epoch=selected_epoch,
        protocol_hash=ExperimentProtocol.canonical().protocol_hash,
        config=development_config or _development_config(),
        manifest_hash=sha256_file(manifest_path),
        early_stopping=stopper,
    )


def _synthetic_dataset(sample_count: int = 12) -> TensorDataset:
    generator = torch.Generator().manual_seed(52)
    inputs = torch.randn(sample_count, 4, generator=generator)
    indices = torch.arange(sample_count)
    targets = torch.stack(
        [((indices + label_index) % 2).float() for label_index in range(5)], dim=1
    )
    return TensorDataset(inputs, targets)


def test_refit_config_forbids_fold_9_and_early_stopping(tmp_path: Path) -> None:
    fold_payload = copy.deepcopy(_payload(tmp_path))
    fold_payload["folds"]["refit"].append(9)  # type: ignore[index, union-attr]
    with pytest.raises(RefitConfigError, match="exactly folds 1-8"):
        FrozenRefitConfig.from_mapping(fold_payload, base_dir=tmp_path)

    early_stop_payload = copy.deepcopy(_payload(tmp_path))
    early_stop_payload["optimization"]["early_stopping_patience"] = 2  # type: ignore[index]
    with pytest.raises(RefitConfigError, match="unexpected"):
        FrozenRefitConfig.from_mapping(early_stop_payload, base_dir=tmp_path)


def test_bundled_refit_configs_parse_and_share_frozen_budget() -> None:
    project_root = Path(__file__).resolve().parents[2]
    resnet = load_refit_config(
        project_root / "configs" / "refit_resnet_frozen.yaml",
        base_dir=project_root,
    )
    transformer = load_refit_config(
        project_root / "configs" / "refit_transformer_frozen.yaml",
        base_dir=project_root,
    )

    assert resnet.refit_folds == transformer.refit_folds == REFIT_FOLDS
    assert resnet.normalization_folds == transformer.normalization_folds == TRAIN_FOLDS
    assert resnet.selection.frozen_epochs == transformer.selection.frozen_epochs
    assert resnet.loader == transformer.loader
    assert resnet.optimization == transformer.optimization
    assert resnet.model.architecture == "resnet1d"
    assert transformer.model.architecture == "ecg_transformer"
    assert resnet.run_kind == transformer.run_kind == "frozen_refit"


def test_frozen_refit_uses_only_folds_1_to_8_and_writes_distinct_artifacts(
    tmp_path: Path,
) -> None:
    _write_inputs(tmp_path)
    config = FrozenRefitConfig.from_mapping(_payload(tmp_path), base_dir=tmp_path)
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
        return _synthetic_dataset()

    def model_factory(model_config: ModelConfig) -> nn.Module:
        assert model_config.architecture == "resnet1d"
        return nn.Linear(4, 5)

    protocol = ExperimentProtocol.canonical()
    result = run_frozen_refit(
        config,
        protocol=protocol,
        dataset_factory=dataset_factory,
        model_factory=model_factory,
    )

    assert calls == [REFIT_FOLDS]
    assert all(fold < 9 for fold in calls[0])
    assert result.frozen_epochs == 2
    assert result.best_training_loss_checkpoint_path.name == "best_training_loss.ckpt"
    assert result.final_checkpoint_path.name == "final.ckpt"
    assert result.best_training_loss_checkpoint_path.is_file()
    assert result.last_checkpoint_path.is_file()
    assert result.final_checkpoint_path.is_file()
    history = [
        json.loads(line)
        for line in result.history_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(history) == 2
    assert all(record["model_selection_metric"] is None for record in history)
    assert all(record["early_stopping"] is False for record in history)
    assert all("validation_loss" not in record for record in history)

    final_checkpoint = torch.load(result.final_checkpoint_path, weights_only=True)
    assert final_checkpoint["epoch"] == 1
    assert final_checkpoint["early_stopping_state_dict"] is None
    assert final_checkpoint["protocol_hash"] == protocol.protocol_hash
    assert final_checkpoint["manifest_hash"] == result.manifest_hash
    assert final_checkpoint["config_hash"] == result.resolved_config_hash
    assert final_checkpoint["config"]["run_kind"] == "frozen_refit"
    metadata = json.loads(
        (result.run_dir / "refit_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["status"] == "complete"
    assert metadata["run_kind"] == "frozen_refit"
    assert metadata["early_stopping_enabled"] is False
    assert metadata["model_selection_enabled"] is False
    assert metadata["authoritative_checkpoint"] == "final.ckpt"
    assert metadata["refit_folds"] == list(REFIT_FOLDS)
    assert metadata["normalization_folds"] == list(TRAIN_FOLDS)
    assert metadata["best_training_loss_is_model_selection"] is False
    assert metadata["selection_provenance"]["selected_epoch"] == 1


def test_refit_rejects_fold_8_normalization_before_dataset_creation(tmp_path: Path) -> None:
    _write_inputs(tmp_path, normalization_folds=REFIT_FOLDS)
    config = FrozenRefitConfig.from_mapping(_payload(tmp_path), base_dir=tmp_path)
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
        return _synthetic_dataset()

    with pytest.raises(FrozenRefitError, match="only on folds 1-7"):
        run_frozen_refit(
            config,
            protocol=ExperimentProtocol.canonical(),
            dataset_factory=forbidden_dataset_factory,
            model_factory=lambda _: nn.Linear(4, 5),
        )
    assert not dataset_called


def test_refit_rejects_unfrozen_epoch_or_optimizer_changes(tmp_path: Path) -> None:
    _write_inputs(tmp_path)
    wrong_epoch = FrozenRefitConfig.from_mapping(
        _payload(tmp_path, frozen_epochs=3), base_dir=tmp_path
    )
    with pytest.raises(FrozenRefitError, match=r"best epoch \+ 1"):
        run_frozen_refit(
            wrong_epoch,
            protocol=ExperimentProtocol.canonical(),
            dataset_factory=lambda *args, **kwargs: _synthetic_dataset(),
            model_factory=lambda _: nn.Linear(4, 5),
        )

    changed_payload = _payload(tmp_path)
    changed_payload["optimization"]["learning_rate"] = 0.02  # type: ignore[index]
    changed_optimization = FrozenRefitConfig.from_mapping(
        changed_payload, base_dir=tmp_path
    )
    with pytest.raises(FrozenRefitError, match="learning_rate differs"):
        run_frozen_refit(
            changed_optimization,
            protocol=ExperimentProtocol.canonical(),
            dataset_factory=lambda *args, **kwargs: _synthetic_dataset(),
            model_factory=lambda _: nn.Linear(4, 5),
        )
