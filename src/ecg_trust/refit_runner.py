"""Frozen trustworthy-track refit on folds 1-8 without model selection."""

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
from typing import Any, Protocol, cast

import pandas as pd  # type: ignore[import-untyped]
import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from ecg_trust.constants import TARGET_COLUMNS
from ecg_trust.data.dataset import NormalizationStats, PTBXLDataset
from ecg_trust.data.manifest import sha256_file
from ecg_trust.experiment_config import ModelConfig
from ecg_trust.experiment_runner import build_experiment_model, training_manifest_sha256
from ecg_trust.models import count_parameters
from ecg_trust.protocol import ExperimentProtocol
from ecg_trust.refit_config import FrozenRefitConfig, RefitOptimizationConfig
from ecg_trust.training import (
    TrainingRuntime,
    save_checkpoint,
    seed_dataloader_worker,
    seed_everything,
    select_device,
    train_one_epoch,
)


class FrozenRefitError(RuntimeError):
    """Raised when a refit violates its frozen selection contract."""


class RefitDatasetFactory(Protocol):
    def __call__(
        self,
        manifest: pd.DataFrame,
        root_dir: Path,
        *,
        folds: tuple[int, ...],
        normalization: NormalizationStats,
        protocol: ExperimentProtocol,
    ) -> Dataset[tuple[Tensor, Tensor]]: ...


class RefitModelFactory(Protocol):
    def __call__(self, config: ModelConfig) -> nn.Module: ...


@dataclass(frozen=True, slots=True)
class SelectedDevelopmentProvenance:
    checkpoint_path: Path
    checkpoint_sha256: str
    checkpoint_config_hash: str
    selected_epoch: int
    selected_macro_auroc: float

    def to_dict(self) -> dict[str, object]:
        return {
            "checkpoint": str(self.checkpoint_path),
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_config_hash": self.checkpoint_config_hash,
            "selected_epoch": self.selected_epoch,
            "selected_epoch_count": self.selected_epoch + 1,
            "selected_macro_auroc": self.selected_macro_auroc,
        }


@dataclass(frozen=True, slots=True)
class FrozenRefitResult:
    run_dir: Path
    history_path: Path
    best_training_loss_checkpoint_path: Path
    last_checkpoint_path: Path
    final_checkpoint_path: Path
    frozen_epochs: int
    best_training_loss_epoch: int
    best_training_loss: float
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


def _mapping(value: object, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise FrozenRefitError(f"{context} must be a string-keyed mapping")
    return cast(Mapping[str, Any], value)


def _canonical_hash(value: Mapping[str, object]) -> str:
    try:
        serialized = json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise FrozenRefitError("resolved config is not finite JSON") from error
    return "sha256:" + hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _read_manifest(path: Path) -> pd.DataFrame:
    try:
        if path.suffix.casefold() == ".parquet":
            frame = pd.read_parquet(path)
        elif path.suffix.casefold() == ".csv":
            frame = pd.read_csv(path)
        else:
            raise FrozenRefitError("manifest must be a .parquet or .csv file")
    except (OSError, ValueError) as error:
        raise FrozenRefitError(f"could not load manifest {path}: {error}") from error
    if frame.empty:
        raise FrozenRefitError("manifest must not be empty")
    return cast(pd.DataFrame, frame)


def _assert_patient_fold_disjoint(manifest: pd.DataFrame) -> None:
    missing = sorted({"patient_id", "strat_fold"}.difference(manifest.columns))
    if missing:
        raise FrozenRefitError(f"manifest is missing required columns: {missing}")
    if manifest["patient_id"].isna().any():
        raise FrozenRefitError("manifest patient_id values must not be missing")
    counts = manifest.groupby("patient_id", dropna=False)["strat_fold"].nunique()
    leaked = counts[counts != 1]
    if not leaked.empty:
        raise FrozenRefitError("manifest contains patients assigned to multiple folds")


def _validate_normalization(
    stats: NormalizationStats,
    *,
    config: FrozenRefitConfig,
    protocol: ExperimentProtocol,
    manifest: pd.DataFrame,
) -> None:
    provenance = stats.provenance
    if provenance.training_folds != config.normalization_folds:
        raise FrozenRefitError("normalization must remain fitted only on folds 1-7")
    if provenance.dataset_version != protocol.dataset_version:
        raise FrozenRefitError("normalization dataset version does not match protocol")
    if provenance.target_columns != TARGET_COLUMNS:
        raise FrozenRefitError("normalization target order does not match the manifest")
    if provenance.path_column != "record_path" or provenance.fold_column != "strat_fold":
        raise FrozenRefitError("normalization manifest columns do not match the refit runner")
    observed_hash = training_manifest_sha256(manifest, config.normalization_folds)
    if observed_hash != provenance.manifest_sha256:
        raise FrozenRefitError(
            "normalization provenance does not match current folds-1-7 manifest rows"
        )


def _equal_number(left: object, right: float) -> bool:
    return (
        not isinstance(left, bool)
        and isinstance(left, (int, float))
        and math.isclose(float(left), right, rel_tol=0.0, abs_tol=1e-12)
    )


def _verify_frozen_optimization(
    development: Mapping[str, Any],
    config: FrozenRefitConfig,
) -> None:
    optimization = _mapping(development.get("optimization"), "development optimization")
    comparisons = {
        "learning_rate": config.optimization.learning_rate,
        "weight_decay": config.optimization.weight_decay,
        "minimum_lr_ratio": config.optimization.minimum_lr_ratio,
        "gradient_clip_norm": config.optimization.gradient_clip_norm,
    }
    for key, expected in comparisons.items():
        if not _equal_number(optimization.get(key), expected):
            raise FrozenRefitError(f"refit optimization.{key} differs from selected run")
    warmup = optimization.get("warmup_epochs")
    if isinstance(warmup, bool) or warmup != config.optimization.warmup_epochs:
        raise FrozenRefitError("refit optimization.warmup_epochs differs from selected run")
    if optimization.get("scheduler") != config.optimization.scheduler:
        raise FrozenRefitError("refit scheduler differs from selected run")
    loader = _mapping(development.get("loader"), "development loader")
    if loader.get("batch_size") != config.loader.batch_size:
        raise FrozenRefitError("refit batch_size differs from selected run")


def _load_selected_development(
    config: FrozenRefitConfig,
    *,
    protocol_hash: str,
    manifest_hash: str,
) -> SelectedDevelopmentProvenance:
    path = config.selection.development_checkpoint_path
    try:
        checkpoint_sha256 = sha256_file(path)
        decoded: object = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError) as error:
        raise FrozenRefitError(
            f"could not load selected development checkpoint: {error}"
        ) from error
    checkpoint = _mapping(decoded, "development checkpoint")
    if checkpoint.get("protocol_hash") != protocol_hash:
        raise FrozenRefitError("selected checkpoint protocol hash does not match")
    if checkpoint.get("manifest_hash") != manifest_hash:
        raise FrozenRefitError("selected checkpoint manifest hash does not match")
    stored_config_hash = checkpoint.get("config_hash")
    if not isinstance(stored_config_hash, str):
        raise FrozenRefitError("selected checkpoint config hash is missing")
    development_config = _mapping(checkpoint.get("config"), "development config")
    observed_config_hash = _canonical_hash(
        cast(Mapping[str, object], development_config)
    )
    if observed_config_hash != stored_config_hash:
        raise FrozenRefitError("selected checkpoint config hash failed verification")

    model = _mapping(development_config.get("model"), "development model")
    if model.get("architecture") != config.model.architecture:
        raise FrozenRefitError("refit architecture differs from selected run")
    if model.get("preset") != config.model.preset:
        raise FrozenRefitError("refit model preset differs from selected run")
    _verify_frozen_optimization(development_config, config)

    selected_epoch = checkpoint.get("epoch")
    if isinstance(selected_epoch, bool) or not isinstance(selected_epoch, int):
        raise FrozenRefitError("selected checkpoint epoch is invalid")
    if selected_epoch < 0 or selected_epoch + 1 != config.selection.frozen_epochs:
        raise FrozenRefitError(
            "selection.frozen_epochs must equal selected best epoch + 1"
        )
    stopper = _mapping(
        checkpoint.get("early_stopping_state_dict"),
        "development early-stopping state",
    )
    if stopper.get("mode") != "max" or stopper.get("best_epoch") != selected_epoch:
        raise FrozenRefitError("development checkpoint is not its selected best epoch")
    selected_score = stopper.get("best_score")
    if isinstance(selected_score, bool) or not isinstance(selected_score, (int, float)):
        raise FrozenRefitError("selected checkpoint has no finite macro AUROC")
    selected_macro_auroc = float(selected_score)
    if not math.isfinite(selected_macro_auroc) or not 0.0 <= selected_macro_auroc <= 1.0:
        raise FrozenRefitError("selected checkpoint macro AUROC must be finite and in [0, 1]")
    return SelectedDevelopmentProvenance(
        checkpoint_path=path,
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_config_hash=stored_config_hash,
        selected_epoch=selected_epoch,
        selected_macro_auroc=selected_macro_auroc,
    )


def _learning_rate(config: RefitOptimizationConfig, *, epoch: int, epochs: int) -> float:
    if config.warmup_epochs and epoch < config.warmup_epochs:
        return config.learning_rate * (epoch + 1) / config.warmup_epochs
    cosine_epochs = epochs - config.warmup_epochs
    progress = (
        1.0
        if cosine_epochs <= 1
        else (epoch - config.warmup_epochs) / (cosine_epochs - 1)
    )
    multiplier = config.minimum_lr_ratio + (1.0 - config.minimum_lr_ratio) * 0.5 * (
        1.0 + math.cos(math.pi * progress)
    )
    return config.learning_rate * multiplier


def _data_loader(
    dataset: Dataset[tuple[Tensor, Tensor]],
    *,
    config: FrozenRefitConfig,
    runtime: TrainingRuntime,
    generator: torch.Generator,
) -> DataLoader[tuple[Tensor, Tensor]]:
    workers = config.loader.num_workers
    return DataLoader(
        dataset,
        batch_size=config.loader.batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=config.loader.pin_memory and runtime.device.type == "cuda",
        persistent_workers=config.loader.persistent_workers and workers > 0,
        worker_init_fn=seed_dataloader_worker if workers > 0 else None,
        generator=generator,
        drop_last=False,
    )


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    serialized = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
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


def _vram(runtime: TrainingRuntime) -> dict[str, int]:
    if runtime.device.type != "cuda":
        return {"peak_allocated_bytes": 0, "peak_reserved_bytes": 0}
    return {
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(runtime.device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(runtime.device),
    }


def _synchronize(runtime: TrainingRuntime) -> None:
    if runtime.device.type == "cuda":
        torch.cuda.synchronize(runtime.device)


def run_frozen_refit(
    config: FrozenRefitConfig,
    *,
    protocol: ExperimentProtocol,
    dataset_factory: RefitDatasetFactory = _default_dataset_factory,
    model_factory: RefitModelFactory = build_experiment_model,
) -> FrozenRefitResult:
    """Refit a selected model for a fixed epoch count on folds 1-8 only."""

    guarded_folds = protocol.guard_fold_access(config.refit_folds)
    if guarded_folds != tuple(range(1, 9)):
        raise FrozenRefitError("frozen refit must use exactly folds 1-8")
    manifest = _read_manifest(config.data.manifest_path)
    _assert_patient_fold_disjoint(manifest)
    try:
        normalization = NormalizationStats.load(config.data.normalization_path)
        manifest_hash = sha256_file(config.data.manifest_path)
        normalization_file_hash = sha256_file(config.data.normalization_path)
    except (OSError, ValueError) as error:
        raise FrozenRefitError(f"could not validate refit inputs: {error}") from error
    _validate_normalization(
        normalization,
        config=config,
        protocol=protocol,
        manifest=manifest,
    )
    selection = _load_selected_development(
        config,
        protocol_hash=protocol.protocol_hash,
        manifest_hash=manifest_hash,
    )

    runtime = select_device(config.runtime.device, enable_bf16=config.runtime.bf16)
    generator = seed_everything(config.runtime.seed)
    model = model_factory(config.model).to(runtime.device)
    dataset = dataset_factory(
        manifest,
        config.data.dataset_root,
        folds=guarded_folds,
        normalization=normalization,
        protocol=protocol,
    )
    dataset_size = len(cast(Sized, dataset))
    if dataset_size < 1:
        raise FrozenRefitError("refit dataset is empty")
    loader = _data_loader(
        dataset,
        config=config,
        runtime=runtime,
        generator=generator,
    )
    optimizer = AdamW(
        model.parameters(),
        lr=config.optimization.learning_rate,
        weight_decay=config.optimization.weight_decay,
    )

    run_dir = config.output.root_dir / config.run_name
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise FrozenRefitError(
            f"refit run directory already exists; refusing to mix artifacts: {run_dir}"
        ) from error
    history_path = run_dir / "refit_history.jsonl"
    best_path = run_dir / "best_training_loss.ckpt"
    last_path = run_dir / "last.ckpt"
    final_path = run_dir / "final.ckpt"

    resolved_config = config.to_resolved_dict()
    resolved_config["model"] = _model_metadata(model, config.model)
    resolved_config["selection_provenance"] = selection.to_dict()
    resolved_config["effective_data"] = {"refit_records": dataset_size}
    resolved_config["checkpoint_roles"] = {
        "best_training_loss.ckpt": "diagnostic minimum training loss only",
        "last.ckpt": "crash-recovery state from the latest completed epoch",
        "final.ckpt": "authoritative frozen-epoch refit artifact",
    }
    resolved_config_hash = _canonical_hash(resolved_config)
    _atomic_json(
        run_dir / "resolved_refit_config.json",
        {"config_hash": resolved_config_hash, "config": resolved_config},
    )
    _atomic_json(run_dir / "protocol.json", protocol.to_resolved_dict())

    started_at = datetime.now(UTC)
    metadata: dict[str, object] = {
        "status": "running",
        "run_kind": "frozen_refit",
        "started_at_utc": started_at.isoformat(),
        "seed": config.runtime.seed,
        "refit_folds": list(guarded_folds),
        "normalization_folds": list(config.normalization_folds),
        "frozen_epochs": config.selection.frozen_epochs,
        "early_stopping_enabled": False,
        "model_selection_enabled": False,
        "authoritative_checkpoint": "final.ckpt",
        "source_config_hash": config.config_hash,
        "resolved_config_hash": resolved_config_hash,
        "protocol_hash": protocol.protocol_hash,
        "manifest_hash": manifest_hash,
        "normalization_file_hash": normalization_file_hash,
        "normalization_provenance": normalization.provenance.to_dict(),
        "selection_provenance": selection.to_dict(),
        "runtime": _runtime_metadata(runtime),
    }
    _atomic_json(run_dir / "refit_metadata.json", metadata)

    best_training_loss = math.inf
    best_training_loss_epoch = -1
    processed_samples = 0
    overall_start = time.perf_counter()
    if runtime.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(runtime.device)
    for epoch in range(config.selection.frozen_epochs):
        learning_rate = _learning_rate(
            config.optimization,
            epoch=epoch,
            epochs=config.selection.frozen_epochs,
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        _synchronize(runtime)
        epoch_start = time.perf_counter()
        result = train_one_epoch(
            model,
            loader,
            optimizer,
            runtime,
            scaler=None,
            max_grad_norm=config.optimization.gradient_clip_norm,
        )
        _synchronize(runtime)
        elapsed = time.perf_counter() - epoch_start
        improved_training_loss = result.loss < best_training_loss
        if improved_training_loss:
            best_training_loss = result.loss
            best_training_loss_epoch = epoch
            save_checkpoint(
                best_path,
                model=model,
                optimizer=optimizer,
                scaler=None,
                epoch=epoch,
                protocol_hash=protocol.protocol_hash,
                config=resolved_config,
                manifest_hash=manifest_hash,
                early_stopping=None,
            )
        save_checkpoint(
            last_path,
            model=model,
            optimizer=optimizer,
            scaler=None,
            epoch=epoch,
            protocol_hash=protocol.protocol_hash,
            config=resolved_config,
            manifest_hash=manifest_hash,
            early_stopping=None,
        )
        _append_jsonl(
            history_path,
            {
                "epoch": epoch,
                "learning_rate": learning_rate,
                "training_loss": result.loss,
                "improved_training_loss": improved_training_loss,
                "samples": result.sample_count,
                "seconds": elapsed,
                "samples_per_second": result.sample_count / elapsed,
                "max_gradient_norm": result.max_gradient_norm,
                "vram": _vram(runtime),
                "model_selection_metric": None,
                "early_stopping": False,
            },
        )
        processed_samples += result.sample_count

    final_epoch = config.selection.frozen_epochs - 1
    save_checkpoint(
        final_path,
        model=model,
        optimizer=optimizer,
        scaler=None,
        epoch=final_epoch,
        protocol_hash=protocol.protocol_hash,
        config=resolved_config,
        manifest_hash=manifest_hash,
        early_stopping=None,
    )
    elapsed_seconds = time.perf_counter() - overall_start
    metadata.update(
        {
            "status": "complete",
            "finished_at_utc": datetime.now(UTC).isoformat(),
            "elapsed_seconds": elapsed_seconds,
            "completed_epochs": config.selection.frozen_epochs,
            "processed_samples": processed_samples,
            "overall_samples_per_second": processed_samples / elapsed_seconds,
            "best_training_loss": best_training_loss,
            "best_training_loss_epoch": best_training_loss_epoch,
            "best_training_loss_is_model_selection": False,
            "final_epoch": final_epoch,
            "vram": _vram(runtime),
        }
    )
    _atomic_json(run_dir / "refit_metadata.json", metadata)
    return FrozenRefitResult(
        run_dir=run_dir,
        history_path=history_path,
        best_training_loss_checkpoint_path=best_path,
        last_checkpoint_path=last_path,
        final_checkpoint_path=final_path,
        frozen_epochs=config.selection.frozen_epochs,
        best_training_loss_epoch=best_training_loss_epoch,
        best_training_loss=best_training_loss,
        resolved_config_hash=resolved_config_hash,
        protocol_hash=protocol.protocol_hash,
        manifest_hash=manifest_hash,
    )
