"""Leakage-safe development runner for folds 1-7 versus fold 8."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
import tempfile
import time
from collections.abc import Mapping, Sized
from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Subset

from ecg_trust.constants import TARGET_COLUMNS
from ecg_trust.data.dataset import NormalizationStats, PTBXLDataset
from ecg_trust.data.manifest import ManifestError, sha256_file, validate_relative_path
from ecg_trust.evaluation import compute_multilabel_metrics
from ecg_trust.experiment_config import DevelopmentExperimentConfig, ModelConfig
from ecg_trust.models import (
    MATCHED_CAPACITY_PRESET,
    ECGTransformer,
    ECGTransformerConfig,
    ResNet1D,
    ResNet1DConfig,
    count_parameters,
)
from ecg_trust.protocol import ExperimentProtocol, FoldRole
from ecg_trust.training import (
    EarlyStopping,
    TrainingRuntime,
    evaluate,
    save_checkpoint,
    seed_dataloader_worker,
    seed_everything,
    select_device,
    train_one_epoch,
)


class DevelopmentRunnerError(RuntimeError):
    """Raised when an experiment cannot preserve its development protocol."""


class DatasetFactory(Protocol):
    def __call__(
        self,
        manifest: pd.DataFrame,
        root_dir: Path,
        *,
        folds: tuple[int, ...],
        normalization: NormalizationStats,
        protocol: ExperimentProtocol,
    ) -> Dataset[tuple[Tensor, Tensor]]: ...


class ModelFactory(Protocol):
    def __call__(self, config: ModelConfig) -> nn.Module: ...


@dataclass(frozen=True, slots=True)
class DevelopmentRunResult:
    """Primary artifacts and selection result from one completed run."""

    run_dir: Path
    history_path: Path
    best_checkpoint_path: Path
    last_checkpoint_path: Path
    best_epoch: int
    best_macro_auroc: float
    completed_epochs: int
    stopped_early: bool
    resolved_config_hash: str
    protocol_hash: str
    manifest_hash: str


def _default_dataset_factory(
    manifest: pd.DataFrame,
    root_dir: Path,
    *,
    folds: tuple[int, ...],
    normalization: NormalizationStats,
    protocol: ExperimentProtocol,
) -> Dataset[tuple[Tensor, Tensor]]:
    return PTBXLDataset(
        manifest,
        root_dir,
        folds=folds,
        normalization=normalization,
        protocol=protocol,
    )


def build_experiment_model(config: ModelConfig) -> nn.Module:
    """Instantiate a smoke model or one side of the matched-capacity pair."""

    if config.preset == "matched_capacity":
        if config.architecture == "resnet1d":
            model: nn.Module = ResNet1D(MATCHED_CAPACITY_PRESET.resnet_config)
            expected_parameters = MATCHED_CAPACITY_PRESET.expected_resnet_parameters
        else:
            model = ECGTransformer(MATCHED_CAPACITY_PRESET.transformer_config)
            expected_parameters = MATCHED_CAPACITY_PRESET.expected_transformer_parameters
        observed_parameters = count_parameters(model)
        if observed_parameters != expected_parameters:
            raise DevelopmentRunnerError(
                f"matched-capacity model drift: expected {expected_parameters}, "
                f"observed {observed_parameters}"
            )
        return model

    if config.architecture == "resnet1d":
        return ResNet1D(
            ResNet1DConfig(
                stage_channels=(16, 32),
                blocks_per_stage=(1, 1),
                block_dropout=0.0,
                classifier_dropout=0.0,
            )
        )
    return ECGTransformer(
        ECGTransformerConfig(
            patch_size=50,
            patch_stride=50,
            embedding_dim=64,
            depth=2,
            num_heads=4,
            mlp_ratio=2.0,
            dropout=0.0,
            attention_dropout=0.0,
        )
    )


def _read_manifest(path: Path) -> pd.DataFrame:
    try:
        if path.suffix.casefold() == ".parquet":
            frame = pd.read_parquet(path)
        elif path.suffix.casefold() == ".csv":
            frame = pd.read_csv(path)
        else:
            raise DevelopmentRunnerError("manifest must be a .parquet or .csv file")
    except (OSError, ValueError) as error:
        raise DevelopmentRunnerError(f"could not load manifest {path}: {error}") from error
    if frame.empty:
        raise DevelopmentRunnerError("manifest must not be empty")
    return cast(pd.DataFrame, frame)


def training_manifest_sha256(
    manifest: pd.DataFrame,
    training_folds: tuple[int, ...],
) -> str:
    """Reproduce the normalization provenance hash from training manifest rows."""

    required = {"record_path", "strat_fold", *TARGET_COLUMNS}
    missing = sorted(required.difference(manifest.columns))
    if missing:
        raise DevelopmentRunnerError(f"manifest is missing required columns: {missing}")
    try:
        fold_values = pd.to_numeric(manifest["strat_fold"], errors="raise").to_numpy(
            dtype=np.float64
        )
        target_values = (
            manifest.loc[:, list(TARGET_COLUMNS)]
            .apply(pd.to_numeric, errors="raise")
            .to_numpy(dtype=np.float64)
        )
    except (TypeError, ValueError) as error:
        raise DevelopmentRunnerError("manifest folds and targets must be numeric") from error
    if not np.isfinite(fold_values).all() or not np.equal(
        fold_values, np.floor(fold_values)
    ).all():
        raise DevelopmentRunnerError("manifest folds must be finite integers")
    if not np.isfinite(target_values).all() or not np.isin(target_values, (0.0, 1.0)).all():
        raise DevelopmentRunnerError("manifest targets must be finite binary values")

    selected_positions = np.flatnonzero(
        np.isin(fold_values.astype(np.int64), training_folds)
    )
    if selected_positions.size == 0:
        raise DevelopmentRunnerError("manifest contains no protocol training rows")
    serialized_rows: list[str] = []
    for position in selected_positions:
        try:
            reference = validate_relative_path(manifest.iloc[int(position)]["record_path"])
        except ManifestError as error:
            raise DevelopmentRunnerError(f"invalid manifest record path: {error}") from error
        row = {
            "fold": int(fold_values[position]),
            "path": reference,
            "targets": [int(value) for value in target_values[position]],
        }
        serialized_rows.append(
            json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)
        )
    content = "\n".join(sorted(serialized_rows)).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def _assert_patient_fold_disjoint(manifest: pd.DataFrame) -> None:
    missing = sorted({"patient_id", "strat_fold"}.difference(manifest.columns))
    if missing:
        raise DevelopmentRunnerError(f"manifest is missing required columns: {missing}")
    if manifest["patient_id"].isna().any():
        raise DevelopmentRunnerError("manifest patient_id values must not be missing")
    fold_counts = manifest.groupby("patient_id", dropna=False)["strat_fold"].nunique()
    leaked = fold_counts[fold_counts != 1]
    if not leaked.empty:
        preview = ", ".join(str(patient_id) for patient_id in leaked.index[:10])
        raise DevelopmentRunnerError(
            f"patients occur in multiple folds; first offending IDs: {preview}"
        )


def _validate_development_folds(
    config: DevelopmentExperimentConfig,
    protocol: ExperimentProtocol,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    train_folds = protocol.folds_for(FoldRole.TRAIN)
    validation_folds = protocol.folds_for(FoldRole.MODEL_SELECTION)
    if config.train_folds != train_folds or config.validation_folds != validation_folds:
        raise DevelopmentRunnerError(
            "runner accepts only protocol train folds 1-7 and model-selection fold 8"
        )
    permitted = set(range(1, 9))
    requested = set(train_folds) | set(validation_folds)
    if requested != permitted:
        raise DevelopmentRunnerError(
            "development runner must cover exactly folds 1-8 and can never access folds 9-10"
        )
    return train_folds, validation_folds


def _validate_normalization(
    stats: NormalizationStats,
    *,
    protocol: ExperimentProtocol,
    train_folds: tuple[int, ...],
) -> None:
    provenance = stats.provenance
    if provenance.training_folds != train_folds:
        raise DevelopmentRunnerError(
            "normalization must be fitted exclusively on protocol training folds 1-7"
        )
    if provenance.dataset_version != protocol.dataset_version:
        raise DevelopmentRunnerError("normalization dataset version does not match protocol")
    if provenance.target_columns != TARGET_COLUMNS:
        raise DevelopmentRunnerError("normalization target order does not match the manifest")
    if provenance.path_column != "record_path" or provenance.fold_column != "strat_fold":
        raise DevelopmentRunnerError("normalization manifest columns do not match the runner")
    if provenance.samples_per_record != 1_000 or not math.isclose(
        provenance.sampling_frequency_hz, 100.0, rel_tol=0.0, abs_tol=1e-9
    ):
        raise DevelopmentRunnerError("normalization is not for 100 Hz, 1000-sample signals")


def _limited_dataset(
    dataset: Dataset[tuple[Tensor, Tensor]], limit: int | None
) -> Dataset[tuple[Tensor, Tensor]]:
    dataset_size = len(cast(Sized, dataset))
    if dataset_size < 1:
        raise DevelopmentRunnerError("selected dataset is empty")
    if limit is None or limit >= dataset_size:
        return dataset
    return Subset(dataset, range(limit))


def _data_loader(
    dataset: Dataset[tuple[Tensor, Tensor]],
    *,
    config: DevelopmentExperimentConfig,
    runtime: TrainingRuntime,
    shuffle: bool,
    generator: torch.Generator | None,
) -> DataLoader[tuple[Tensor, Tensor]]:
    workers = config.loader.num_workers
    return DataLoader(
        dataset,
        batch_size=config.loader.batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=config.loader.pin_memory and runtime.device.type == "cuda",
        persistent_workers=config.loader.persistent_workers and workers > 0,
        worker_init_fn=seed_dataloader_worker if workers > 0 else None,
        generator=generator,
        drop_last=False,
    )


def _learning_rate(config: DevelopmentExperimentConfig, epoch: int) -> float:
    optimization = config.optimization
    if optimization.warmup_epochs and epoch < optimization.warmup_epochs:
        return optimization.learning_rate * (epoch + 1) / optimization.warmup_epochs
    cosine_epochs = optimization.epochs - optimization.warmup_epochs
    if cosine_epochs <= 1:
        progress = 1.0
    else:
        progress = (epoch - optimization.warmup_epochs) / (cosine_epochs - 1)
    multiplier = optimization.minimum_lr_ratio + (
        1.0 - optimization.minimum_lr_ratio
    ) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return optimization.learning_rate * multiplier


def _set_learning_rate(optimizer: AdamW, learning_rate: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = learning_rate


def _mapping_hash(value: Mapping[str, object]) -> str:
    serialized = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return "sha256:" + hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    serialized = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _append_jsonl(path: Path, value: Mapping[str, object]) -> None:
    serialized = (
        json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        os.write(descriptor, serialized)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _model_metadata(model: nn.Module, selection: ModelConfig) -> dict[str, object]:
    metadata: dict[str, object] = {
        "architecture": selection.architecture,
        "preset": selection.preset,
        "class": f"{type(model).__module__}.{type(model).__qualname__}",
        "trainable_parameters": count_parameters(model),
    }
    raw_config = getattr(model, "config", None)
    if raw_config is not None and is_dataclass(raw_config) and not isinstance(raw_config, type):
        metadata["resolved_architecture_config"] = asdict(raw_config)
    if selection.preset == "matched_capacity":
        metadata["capacity_match"] = MATCHED_CAPACITY_PRESET.metadata()
    return metadata


def _runtime_metadata(runtime: TrainingRuntime) -> dict[str, object]:
    metadata: dict[str, object] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "device": str(runtime.device),
        "bf16_autocast": runtime.bf16_enabled,
    }
    if runtime.device.type == "cuda":
        properties = torch.cuda.get_device_properties(runtime.device)
        metadata.update(
            {
                "cuda_device_name": torch.cuda.get_device_name(runtime.device),
                "cuda_device_capability": list(
                    torch.cuda.get_device_capability(runtime.device)
                ),
                "cuda_total_memory_bytes": properties.total_memory,
            }
        )
    return metadata


def _vram_metadata(runtime: TrainingRuntime) -> dict[str, object]:
    if runtime.device.type != "cuda":
        return {
            "peak_allocated_bytes": 0,
            "peak_reserved_bytes": 0,
        }
    return {
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(runtime.device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(runtime.device),
    }


def _synchronize(runtime: TrainingRuntime) -> None:
    if runtime.device.type == "cuda":
        torch.cuda.synchronize(runtime.device)


def run_development_experiment(
    config: DevelopmentExperimentConfig,
    *,
    protocol: ExperimentProtocol,
    dataset_factory: DatasetFactory = _default_dataset_factory,
    model_factory: ModelFactory = build_experiment_model,
) -> DevelopmentRunResult:
    """Train on folds 1-7 and select on fold 8 without touching folds 9-10."""

    train_folds, validation_folds = _validate_development_folds(config, protocol)
    try:
        normalization = NormalizationStats.load(config.data.normalization_path)
    except ValueError as error:
        raise DevelopmentRunnerError(f"invalid normalization: {error}") from error
    _validate_normalization(normalization, protocol=protocol, train_folds=train_folds)
    manifest = _read_manifest(config.data.manifest_path)
    _assert_patient_fold_disjoint(manifest)
    normalization_manifest_hash = training_manifest_sha256(manifest, train_folds)
    if normalization_manifest_hash != normalization.provenance.manifest_sha256:
        raise DevelopmentRunnerError(
            "normalization provenance does not match the current training manifest rows"
        )
    try:
        manifest_hash = sha256_file(config.data.manifest_path)
        normalization_file_hash = sha256_file(config.data.normalization_path)
    except OSError as error:
        raise DevelopmentRunnerError(f"could not hash experiment inputs: {error}") from error

    runtime = select_device(config.runtime.device, enable_bf16=config.runtime.bf16)
    train_generator = seed_everything(config.runtime.seed)
    model = model_factory(config.model).to(runtime.device)
    train_dataset = _limited_dataset(
        dataset_factory(
            manifest,
            config.data.dataset_root,
            folds=train_folds,
            normalization=normalization,
            protocol=protocol,
        ),
        config.data.max_train_records,
    )
    validation_dataset = _limited_dataset(
        dataset_factory(
            manifest,
            config.data.dataset_root,
            folds=validation_folds,
            normalization=normalization,
            protocol=protocol,
        ),
        config.data.max_validation_records,
    )
    train_loader = _data_loader(
        train_dataset,
        config=config,
        runtime=runtime,
        shuffle=True,
        generator=train_generator,
    )
    validation_loader = _data_loader(
        validation_dataset,
        config=config,
        runtime=runtime,
        shuffle=False,
        generator=None,
    )
    optimizer = AdamW(
        model.parameters(),
        lr=config.optimization.learning_rate,
        weight_decay=config.optimization.weight_decay,
    )
    early_stopping = EarlyStopping(
        patience=config.optimization.early_stopping_patience,
        mode="max",
        min_delta=config.optimization.early_stopping_min_delta,
    )

    run_dir = config.output.root_dir / config.run_name
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise DevelopmentRunnerError(
            f"run directory already exists; refusing to mix artifacts: {run_dir}"
        ) from error
    history_path = run_dir / "history.jsonl"
    best_checkpoint_path = run_dir / "best.ckpt"
    last_checkpoint_path = run_dir / "last.ckpt"

    resolved_config = config.to_resolved_dict()
    resolved_config["model"] = _model_metadata(model, config.model)
    resolved_config["optimizer"] = {
        "name": "AdamW",
        "betas": [0.9, 0.999],
        "eps": 1e-8,
    }
    resolved_config["effective_data"] = {
        "train_records": len(cast(Sized, train_dataset)),
        "validation_records": len(cast(Sized, validation_dataset)),
    }
    resolved_config_hash = _mapping_hash(resolved_config)
    _atomic_json(
        run_dir / "resolved_config.json",
        {"config_hash": resolved_config_hash, "config": resolved_config},
    )
    _atomic_json(run_dir / "protocol.json", protocol.to_resolved_dict())

    started_at = datetime.now(UTC)
    metadata: dict[str, object] = {
        "status": "running",
        "started_at_utc": started_at.isoformat(),
        "seed": config.runtime.seed,
        "source_config_hash": config.config_hash,
        "resolved_config_hash": resolved_config_hash,
        "protocol_hash": protocol.protocol_hash,
        "manifest_hash": manifest_hash,
        "normalization_file_hash": normalization_file_hash,
        "normalization_provenance": normalization.provenance.to_dict(),
        "runtime": _runtime_metadata(runtime),
    }
    _atomic_json(run_dir / "run_metadata.json", metadata)

    overall_start = time.perf_counter()
    completed_epochs = 0
    processed_samples = 0
    if runtime.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(runtime.device)
    for epoch in range(config.optimization.epochs):
        learning_rate = _learning_rate(config, epoch)
        _set_learning_rate(optimizer, learning_rate)

        _synchronize(runtime)
        train_start = time.perf_counter()
        train_result = train_one_epoch(
            model,
            train_loader,
            optimizer,
            runtime,
            scaler=None,
            max_grad_norm=config.optimization.gradient_clip_norm,
        )
        _synchronize(runtime)
        train_seconds = time.perf_counter() - train_start

        _synchronize(runtime)
        validation_start = time.perf_counter()
        validation_result = evaluate(model, validation_loader, runtime)
        _synchronize(runtime)
        validation_seconds = time.perf_counter() - validation_start
        metrics = compute_multilabel_metrics(
            validation_result.targets.numpy(), validation_result.probabilities.numpy()
        )
        macro_auroc = metrics.macro.roc_auc
        if macro_auroc is None:
            raise DevelopmentRunnerError(
                "validation macro AUROC is undefined; fold 8 must contain both classes"
            )
        improved = early_stopping.update(macro_auroc, epoch)

        if improved:
            save_checkpoint(
                best_checkpoint_path,
                model=model,
                optimizer=optimizer,
                scaler=None,
                epoch=epoch,
                protocol_hash=protocol.protocol_hash,
                config=resolved_config,
                manifest_hash=manifest_hash,
                early_stopping=early_stopping,
            )
        save_checkpoint(
            last_checkpoint_path,
            model=model,
            optimizer=optimizer,
            scaler=None,
            epoch=epoch,
            protocol_hash=protocol.protocol_hash,
            config=resolved_config,
            manifest_hash=manifest_hash,
            early_stopping=early_stopping,
        )

        epoch_record: dict[str, object] = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "train_loss": train_result.loss,
            "validation_loss": validation_result.loss,
            "validation_macro_auroc": macro_auroc,
            "validation_metrics": metrics.to_dict(),
            "improved": improved,
            "bad_epochs": early_stopping.bad_epochs,
            "stopped": early_stopping.stopped,
            "train_samples": train_result.sample_count,
            "validation_samples": validation_result.sample_count,
            "train_seconds": train_seconds,
            "validation_seconds": validation_seconds,
            "train_samples_per_second": train_result.sample_count / train_seconds,
            "validation_samples_per_second": (
                validation_result.sample_count / validation_seconds
            ),
            "max_gradient_norm": train_result.max_gradient_norm,
            "vram": _vram_metadata(runtime),
        }
        _append_jsonl(history_path, epoch_record)
        completed_epochs = epoch + 1
        processed_samples += train_result.sample_count + validation_result.sample_count
        if early_stopping.stopped:
            break

    if early_stopping.best_epoch is None or early_stopping.best_score is None:
        raise DevelopmentRunnerError("experiment completed without a selectable checkpoint")
    finished_at = datetime.now(UTC)
    elapsed_seconds = time.perf_counter() - overall_start
    metadata.update(
        {
            "status": "complete",
            "finished_at_utc": finished_at.isoformat(),
            "elapsed_seconds": elapsed_seconds,
            "completed_epochs": completed_epochs,
            "stopped_early": early_stopping.stopped,
            "best_epoch": early_stopping.best_epoch,
            "best_validation_macro_auroc": early_stopping.best_score,
            "processed_samples": processed_samples,
            "overall_samples_per_second": processed_samples / elapsed_seconds,
            "vram": _vram_metadata(runtime),
        }
    )
    _atomic_json(run_dir / "run_metadata.json", metadata)
    return DevelopmentRunResult(
        run_dir=run_dir,
        history_path=history_path,
        best_checkpoint_path=best_checkpoint_path,
        last_checkpoint_path=last_checkpoint_path,
        best_epoch=early_stopping.best_epoch,
        best_macro_auroc=early_stopping.best_score,
        completed_epochs=completed_epochs,
        stopped_early=early_stopping.stopped,
        resolved_config_hash=resolved_config_hash,
        protocol_hash=protocol.protocol_hash,
        manifest_hash=manifest_hash,
    )
