"""Frozen trustworthy-track refit on folds 1-8 without model selection."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
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
from ecg_trust.models import MATCHED_CAPACITY_PRESET, count_parameters
from ecg_trust.multiseed_freeze import (
    MultiSeedFreezeArtifact,
    MultiSeedFreezeError,
    canonical_sha256,
    capture_downstream_provenance,
    file_sha256,
    load_confirmation_member,
    load_multiseed_freeze,
    normalized_recipe_template,
    verify_self_hash,
    write_new_json,
)
from ecg_trust.protocol import ExperimentProtocol
from ecg_trust.refit_config import (
    FrozenRefitConfig,
    PostSweepRefitConfig,
    RefitConfig,
    RefitOptimizationConfig,
)
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
    source_seed: int | None = None
    member_completion_sha256: str | None = None
    freeze_artifact_sha256: str | None = None
    recipe_sha256: str | None = None

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "checkpoint": str(self.checkpoint_path),
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_config_hash": self.checkpoint_config_hash,
            "selected_epoch": self.selected_epoch,
            "selected_epoch_count": self.selected_epoch + 1,
            "selected_macro_auroc": self.selected_macro_auroc,
        }
        if self.source_seed is not None:
            result.update(
                {
                    "source_seed": self.source_seed,
                    "member_completion_sha256": self.member_completion_sha256,
                    "freeze_artifact_sha256": self.freeze_artifact_sha256,
                    "recipe_sha256": self.recipe_sha256,
                }
            )
        return result


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
    completion_path: Path | None = None
    completion_sha256: str | None = None
    freeze_artifact_sha256: str | None = None


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


def _string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FrozenRefitError(f"{context} must be a non-empty string")
    return value


def _read_json_mapping(path: Path, context: str) -> Mapping[str, Any]:
    try:
        decoded: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FrozenRefitError(f"could not read {context}: {error}") from error
    return _mapping(decoded, context)


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
    config: RefitConfig,
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
    config: RefitConfig,
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


def _post_sweep_source_expectations(config: PostSweepRefitConfig) -> dict[str, object]:
    source = config.source
    return {
        "completion_path": source.member_completion_path,
        "completion_sha256": source.member_completion_sha256,
        "manifest_sha256": source.manifest_sha256,
        "normalization_sha256": source.normalization_sha256,
        "run_metadata_path": source.run_metadata_path,
        "run_metadata_sha256": source.run_metadata_sha256,
        "resolved_config_path": source.resolved_config_path,
        "resolved_config_sha256": source.resolved_config_file_sha256,
        "config_hash": source.resolved_config_hash,
        "history_path": source.history_path,
        "history_sha256": source.history_sha256,
        "best_checkpoint_path": source.best_checkpoint_path,
        "best_checkpoint_sha256": source.best_checkpoint_sha256,
        "prediction_path": source.prediction_path,
        "prediction_npz_sha256": source.prediction_npz_sha256,
        "prediction_json_path": source.prediction_json_path,
        "prediction_artifact_sha256": source.prediction_artifact_sha256,
        "best_epoch": source.best_epoch,
        "best_validation_macro_auroc": source.best_validation_macro_auroc,
    }


def _verify_post_sweep_policy(
    development: Mapping[str, object],
    config: PostSweepRefitConfig,
) -> None:
    model = _mapping(development.get("model"), "development model")
    if model.get("architecture") != config.architecture:
        raise FrozenRefitError("refit architecture differs from frozen member")
    if model.get("preset") != config.model.preset:
        raise FrozenRefitError("refit model preset differs from frozen member")
    if dict(model) != dict(config.model_identity):
        raise FrozenRefitError("frozen model identity differs from development member")
    loader = _mapping(development.get("loader"), "development loader")
    if dict(loader) != config.loader.to_dict():
        raise FrozenRefitError("full refit loader policy differs from frozen member")
    runtime = _mapping(development.get("runtime"), "development runtime")
    if dict(runtime) != config.runtime.to_dict():
        raise FrozenRefitError("refit seed, precision, or device policy differs from frozen member")
    data = _mapping(development.get("data"), "development data")
    expected_data = {
        "manifest": str(config.data.manifest_path),
        "dataset_root": str(config.data.dataset_root),
        "normalization": str(config.data.normalization_path),
    }
    if any(data.get(key) != value for key, value in expected_data.items()):
        raise FrozenRefitError("refit data paths differ from frozen member")
    optimizer = _mapping(development.get("optimizer"), "development optimizer")
    if dict(optimizer) != config.optimizer.to_dict():
        raise FrozenRefitError("refit AdamW policy differs from frozen member")
    _verify_frozen_optimization(cast(Mapping[str, Any], development), config)


def _load_post_sweep_development(
    config: PostSweepRefitConfig,
    *,
    protocol: ExperimentProtocol,
    current_manifest_sha256: str,
    current_normalization_sha256: str,
) -> tuple[SelectedDevelopmentProvenance, MultiSeedFreezeArtifact]:
    try:
        freeze = load_multiseed_freeze(
            config.freeze_artifact_path,
            protocol=protocol,
            verify_sources=True,
        )
    except MultiSeedFreezeError as error:
        raise FrozenRefitError(f"multi-seed freeze verification failed: {error}") from error
    if freeze.artifact_sha256 != config.freeze_artifact_sha256:
        raise FrozenRefitError("recipe freeze_artifact_sha256 does not match its file")
    if freeze.comparison_id != config.comparison_id:
        raise FrozenRefitError("recipe comparison_id does not match the freeze")
    try:
        observed_downstream = capture_downstream_provenance(
            config.downstream_provenance.project_root
        )
    except MultiSeedFreezeError as error:
        raise FrozenRefitError(f"could not verify downstream provenance: {error}") from error
    if observed_downstream != config.downstream_provenance.to_dict():
        raise FrozenRefitError(
            "current downstream code revision or dependency lock differs from freeze"
        )
    expected_recipe = freeze.recipe_template(config.architecture, config.confirmation_seed)
    observed_recipe = normalized_recipe_template(config.to_resolved_dict())
    if observed_recipe != expected_recipe:
        raise FrozenRefitError("post-sweep refit recipe is not an exact member of the freeze")
    recipe_body = dict(observed_recipe)
    recipe_hash = recipe_body.pop("recipe_sha256", None)
    if canonical_sha256(recipe_body) != recipe_hash or recipe_hash != config.recipe_sha256:
        raise FrozenRefitError("post-sweep refit recipe hash is invalid")
    try:
        member = load_confirmation_member(
            config.source.member_completion_path,
            protocol=protocol,
        )
    except MultiSeedFreezeError as error:
        raise FrozenRefitError(
            f"frozen confirmation member failed verification: {error}"
        ) from error
    if member.architecture != config.architecture or member.seed != config.confirmation_seed:
        raise FrozenRefitError("frozen confirmation member identity differs from recipe")
    expected_source = _post_sweep_source_expectations(config)
    observed_source: dict[str, object] = {
        "completion_path": member.completion_path,
        "completion_sha256": member.completion_sha256,
        "manifest_sha256": member.manifest_hash,
        "normalization_sha256": member.normalization_sha256,
        "run_metadata_path": member.run_metadata_path,
        "run_metadata_sha256": member.run_metadata_sha256,
        "resolved_config_path": member.resolved_config_path,
        "resolved_config_sha256": member.resolved_config_sha256,
        "config_hash": member.config_hash,
        "history_path": member.history_path,
        "history_sha256": member.history_sha256,
        "best_checkpoint_path": member.best_checkpoint_path,
        "best_checkpoint_sha256": member.best_checkpoint_sha256,
        "prediction_path": member.prediction_path,
        "prediction_npz_sha256": member.prediction_npz_sha256,
        "prediction_json_path": member.prediction_json_path,
        "prediction_artifact_sha256": member.prediction_artifact_sha256,
        "best_epoch": member.best_epoch,
        "best_validation_macro_auroc": member.recomputed_macro_auroc,
    }
    if observed_source != expected_source:
        raise FrozenRefitError("recipe source evidence differs from member completion")
    manifest_expectations = {
        freeze.manifest_sha256,
        member.manifest_hash,
        config.source.manifest_sha256,
    }
    if manifest_expectations != {current_manifest_sha256}:
        raise FrozenRefitError(
            "current manifest SHA-256 differs from frozen recipe/member/freeze lineage"
        )
    normalization_expectations = {
        freeze.normalization_sha256,
        member.normalization_sha256,
        config.source.normalization_sha256,
    }
    if normalization_expectations != {current_normalization_sha256}:
        raise FrozenRefitError(
            "current normalization SHA-256 differs from frozen recipe/member/freeze lineage"
        )
    _verify_post_sweep_policy(member.resolved_config, config)
    return (
        SelectedDevelopmentProvenance(
            checkpoint_path=member.best_checkpoint_path,
            checkpoint_sha256=member.best_checkpoint_sha256.removeprefix("sha256:"),
            checkpoint_config_hash=member.config_hash,
            selected_epoch=member.best_epoch,
            selected_macro_auroc=member.recomputed_macro_auroc,
            source_seed=member.seed,
            member_completion_sha256=member.completion_sha256,
            freeze_artifact_sha256=freeze.artifact_sha256,
            recipe_sha256=config.recipe_sha256,
        ),
        freeze,
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
    config: RefitConfig,
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
        "resolved_architecture_config": {},
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


REFIT_COMPLETION_SCHEMA_VERSION = 1
REFIT_COMPLETION_ARTIFACT_TYPE = "ecg_trust.refit_completion"
_REFIT_COMPLETION_KEYS = {
    "schema_version",
    "artifact_type",
    "comparison_id",
    "architecture",
    "seed",
    "status",
    "run_name",
    "run_dir",
    "freeze_artifact_path",
    "freeze_artifact_sha256",
    "recipe_sha256",
    "source_member_completion_path",
    "source_member_completion_sha256",
    "refit_folds",
    "normalization_folds",
    "frozen_epochs",
    "protocol_hash",
    "manifest_hash",
    "normalization_hash",
    "downstream_provenance",
    "selection_provenance",
    "selection_lineage_sha256",
    "files",
    "artifact_sha256",
}


def _completion_file(path: Path) -> dict[str, object]:
    return {"path": str(path.resolve()), "sha256": file_sha256(path)}


def _write_refit_completion(
    *,
    config: PostSweepRefitConfig,
    run_dir: Path,
    resolved_path: Path,
    metadata_path: Path,
    protocol_path: Path,
    history_path: Path,
    final_path: Path,
    selection: SelectedDevelopmentProvenance,
    resolved_config_hash: str,
    protocol_hash: str,
    manifest_hash: str,
    normalization_hash: str,
) -> tuple[Path, str]:
    selection_payload = selection.to_dict()
    body: dict[str, object] = {
        "schema_version": REFIT_COMPLETION_SCHEMA_VERSION,
        "artifact_type": REFIT_COMPLETION_ARTIFACT_TYPE,
        "comparison_id": config.comparison_id,
        "architecture": config.architecture,
        "seed": config.confirmation_seed,
        "status": "complete",
        "run_name": config.run_name,
        "run_dir": str(run_dir.resolve()),
        "freeze_artifact_path": str(config.freeze_artifact_path),
        "freeze_artifact_sha256": config.freeze_artifact_sha256,
        "recipe_sha256": config.recipe_sha256,
        "source_member_completion_path": str(config.source.member_completion_path),
        "source_member_completion_sha256": config.source.member_completion_sha256,
        "refit_folds": list(config.refit_folds),
        "normalization_folds": list(config.normalization_folds),
        "frozen_epochs": config.selection.frozen_epochs,
        "protocol_hash": protocol_hash,
        "manifest_hash": "sha256:" + manifest_hash.removeprefix("sha256:"),
        "normalization_hash": "sha256:" + normalization_hash.removeprefix("sha256:"),
        "downstream_provenance": config.downstream_provenance.to_dict(),
        "selection_provenance": selection_payload,
        "selection_lineage_sha256": canonical_sha256(selection_payload),
        "files": {
            "final_checkpoint": _completion_file(final_path),
            "resolved_config": {
                **_completion_file(resolved_path),
                "config_hash": resolved_config_hash,
            },
            "metadata": _completion_file(metadata_path),
            "protocol": _completion_file(protocol_path),
            "history": _completion_file(history_path),
            "manifest": _completion_file(config.data.manifest_path),
            "normalization": _completion_file(config.data.normalization_path),
            "source_checkpoint": _completion_file(selection.checkpoint_path),
            "attempt_identity": _completion_file(run_dir / "attempt_identity.json"),
        },
    }
    payload = dict(body)
    digest = canonical_sha256(body)
    payload["artifact_sha256"] = digest
    path = run_dir / "refit_completion.json"
    write_new_json(path, payload)
    return path, digest


def load_refit_completion(
    path: str | Path,
    *,
    protocol: ExperimentProtocol,
    verify_sources: bool = True,
) -> Mapping[str, object]:
    """Load a self-hashed per-run receipt and optionally re-hash every source."""

    completion_path = Path(path).resolve()
    try:
        decoded: object = json.loads(completion_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FrozenRefitError(f"could not read refit completion: {error}") from error
    root = _mapping(decoded, "refit completion")
    if set(root) != _REFIT_COMPLETION_KEYS:
        raise FrozenRefitError("refit completion keys are not canonical")
    if root.get("schema_version") != REFIT_COMPLETION_SCHEMA_VERSION:
        raise FrozenRefitError("refit completion schema_version is unsupported")
    if root.get("artifact_type") != REFIT_COMPLETION_ARTIFACT_TYPE:
        raise FrozenRefitError("refit completion artifact_type is invalid")
    if root.get("status") != "complete":
        raise FrozenRefitError("refit completion status must be complete")
    try:
        verify_self_hash(cast(Mapping[str, object], root), "refit completion")
    except MultiSeedFreezeError as error:
        raise FrozenRefitError(str(error)) from error
    if root.get("protocol_hash") != protocol.protocol_hash:
        raise FrozenRefitError("refit completion protocol hash does not match")
    selection_lineage = _mapping(
        root.get("selection_provenance"), "refit completion selection provenance"
    )
    if canonical_sha256(selection_lineage) != root.get("selection_lineage_sha256"):
        raise FrozenRefitError("refit completion selection lineage hash is invalid")
    if verify_sources:
        freeze_path = Path(
            _string(root.get("freeze_artifact_path"), "completion freeze artifact path")
        ).resolve()
        try:
            freeze = load_multiseed_freeze(
                freeze_path,
                protocol=protocol,
                verify_sources=True,
            )
        except MultiSeedFreezeError as error:
            raise FrozenRefitError(f"completion freeze verification failed: {error}") from error
        if freeze.artifact_sha256 != root.get("freeze_artifact_sha256"):
            raise FrozenRefitError("refit completion freeze hash is invalid")
        architecture = _string(root.get("architecture"), "completion architecture")
        seed = root.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise FrozenRefitError("completion seed must be an integer")
        try:
            recipe = freeze.recipe_template(architecture, seed)
        except MultiSeedFreezeError as error:
            raise FrozenRefitError(str(error)) from error
        if recipe.get("recipe_sha256") != root.get("recipe_sha256"):
            raise FrozenRefitError("completion recipe hash differs from freeze")
        if recipe.get("comparison_id") != root.get("comparison_id"):
            raise FrozenRefitError("completion comparison_id differs from freeze recipe")
        recipe_source = _mapping(recipe.get("source"), "freeze recipe source")
        completion_source_path = Path(
            _string(
                root.get("source_member_completion_path"),
                "completion source member path",
            )
        ).resolve()
        expected_source_path = Path(
            _string(recipe_source.get("member_completion"), "recipe member completion")
        ).resolve()
        if completion_source_path != expected_source_path:
            raise FrozenRefitError("completion source member path differs from freeze recipe")
        if root.get("source_member_completion_sha256") != recipe_source.get(
            "member_completion_sha256"
        ):
            raise FrozenRefitError("completion source member hash differs from freeze recipe")
        try:
            observed_source_hash = file_sha256(completion_source_path)
        except MultiSeedFreezeError as error:
            raise FrozenRefitError(str(error)) from error
        if observed_source_hash != root.get("source_member_completion_sha256"):
            raise FrozenRefitError("completion source member file hash mismatch")
        if root.get("manifest_hash") != freeze.manifest_sha256 or root.get(
            "manifest_hash"
        ) != recipe_source.get("manifest_sha256"):
            raise FrozenRefitError("completion manifest hash differs from frozen lineage")
        if root.get("normalization_hash") != freeze.normalization_sha256 or root.get(
            "normalization_hash"
        ) != recipe_source.get("normalization_sha256"):
            raise FrozenRefitError(
                "completion normalization hash differs from frozen lineage"
            )
        if root.get("downstream_provenance") != recipe.get("downstream_provenance"):
            raise FrozenRefitError("completion downstream provenance differs from freeze")
        files = _mapping(root.get("files"), "refit completion files")
        expected_names = {
            "final_checkpoint",
            "resolved_config",
            "metadata",
            "protocol",
            "history",
            "manifest",
            "normalization",
            "source_checkpoint",
            "attempt_identity",
        }
        if set(files) != expected_names:
            raise FrozenRefitError("refit completion file inventory is not canonical")
        for name in expected_names:
            entry = _mapping(files[name], f"refit completion {name}")
            allowed = {"path", "sha256", "config_hash"} if name == "resolved_config" else {
                "path",
                "sha256",
            }
            if set(entry) != allowed:
                raise FrozenRefitError(f"refit completion {name} keys are invalid")
            source = Path(_string(entry.get("path"), f"refit completion {name} path"))
            try:
                observed = file_sha256(source)
            except MultiSeedFreezeError as error:
                raise FrozenRefitError(str(error)) from error
            if observed != entry.get("sha256"):
                raise FrozenRefitError(f"refit completion {name} hash mismatch")
        manifest_entry = _mapping(files["manifest"], "refit manifest entry")
        normalization_entry = _mapping(files["normalization"], "refit normalization entry")
        if manifest_entry.get("sha256") != root.get("manifest_hash"):
            raise FrozenRefitError("completion manifest file is not the frozen manifest")
        if normalization_entry.get("sha256") != root.get("normalization_hash"):
            raise FrozenRefitError(
                "completion normalization file is not the frozen normalization"
            )
        resolved_entry = _mapping(files["resolved_config"], "refit resolved config entry")
        resolved_wrapper = _read_json_mapping(
            Path(_string(resolved_entry["path"], "refit resolved config path")),
            "refit resolved config",
        )
        if set(resolved_wrapper) != {"config_hash", "config"}:
            raise FrozenRefitError("resolved refit config wrapper keys are invalid")
        resolved_body = _mapping(
            resolved_wrapper.get("config"), "resolved refit config body"
        )
        if canonical_sha256(resolved_body) != resolved_wrapper.get(
            "config_hash"
        ) or resolved_wrapper.get("config_hash") != resolved_entry.get("config_hash"):
            raise FrozenRefitError("resolved refit config content hash is invalid")
        resolved_expectations: dict[str, object] = {
            "run_kind": "post_sweep_frozen_refit",
            "comparison_id": root.get("comparison_id"),
            "architecture": architecture,
            "confirmation_seed": seed,
            "run_name": root.get("run_name"),
            "freeze_artifact_sha256": freeze.artifact_sha256,
            "recipe_sha256": root.get("recipe_sha256"),
            "downstream_provenance": root.get("downstream_provenance"),
        }
        drift = [
            key
            for key, expected in resolved_expectations.items()
            if resolved_body.get(key) != expected
        ]
        if drift:
            raise FrozenRefitError(
                "resolved refit config differs from completion: " + ", ".join(drift)
            )
        metadata_entry = _mapping(files["metadata"], "refit metadata entry")
        metadata = _read_json_mapping(
            Path(_string(metadata_entry["path"], "refit metadata path")),
            "refit metadata",
        )
        if metadata.get("status") != "complete":
            raise FrozenRefitError("refit completion points to incomplete metadata")
        if metadata.get("freeze_artifact_sha256") != root.get("freeze_artifact_sha256"):
            raise FrozenRefitError("refit metadata freeze hash disagrees with completion")
        metadata_expectations = {
            "protocol_hash": root.get("protocol_hash"),
            "manifest_hash": str(root.get("manifest_hash")).removeprefix("sha256:"),
            "normalization_file_hash": str(root.get("normalization_hash")).removeprefix(
                "sha256:"
            ),
            "downstream_provenance": root.get("downstream_provenance"),
            "recipe_sha256": root.get("recipe_sha256"),
        }
        metadata_drift = [
            key
            for key, expected in metadata_expectations.items()
            if metadata.get(key) != expected
        ]
        if metadata_drift:
            raise FrozenRefitError(
                "refit metadata differs from completion: "
                + ", ".join(metadata_drift)
            )
        final_entry = _mapping(files["final_checkpoint"], "final checkpoint entry")
        if metadata.get("final_checkpoint_sha256") != final_entry.get("sha256"):
            raise FrozenRefitError("refit metadata final checkpoint hash disagrees")
    return cast(Mapping[str, object], root)


_ATTEMPT_NAME = re.compile(r"attempt(\d{2})\Z")
_ATTEMPT_IDENTITY_KEYS = {
    "schema_version",
    "artifact_type",
    "comparison_id",
    "architecture",
    "seed",
    "run_name",
    "attempt_index",
    "freeze_artifact_sha256",
    "recipe_sha256",
    "downstream_provenance",
    "artifact_sha256",
}


def _attempt_identity(
    config: PostSweepRefitConfig, *, attempt_index: int
) -> dict[str, object]:
    body: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "ecg_trust.refit_attempt",
        "comparison_id": config.comparison_id,
        "architecture": config.architecture,
        "seed": config.confirmation_seed,
        "run_name": config.run_name,
        "attempt_index": attempt_index,
        "freeze_artifact_sha256": config.freeze_artifact_sha256,
        "recipe_sha256": config.recipe_sha256,
        "downstream_provenance": config.downstream_provenance.to_dict(),
    }
    body["artifact_sha256"] = canonical_sha256(body)
    return body


def _verify_attempt_identity(
    path: Path,
    config: PostSweepRefitConfig,
    *,
    attempt_index: int,
) -> None:
    observed = _read_json_mapping(path, "refit attempt identity")
    if set(observed) != _ATTEMPT_IDENTITY_KEYS:
        raise FrozenRefitError("existing refit attempt identity keys are invalid")
    try:
        verify_self_hash(cast(Mapping[str, object], observed), "refit attempt identity")
    except MultiSeedFreezeError as error:
        raise FrozenRefitError(str(error)) from error
    if dict(observed) != _attempt_identity(config, attempt_index=attempt_index):
        raise FrozenRefitError(
            "existing refit attempt belongs to a different recipe, seed, or provenance"
        )


def _allocate_post_sweep_attempt(
    config: PostSweepRefitConfig,
    *,
    protocol: ExperimentProtocol,
) -> tuple[Path, int]:
    base = (config.output.root_dir / config.run_name).resolve()
    if base.exists() and not base.is_dir():
        raise FrozenRefitError(f"refit attempt root is not a directory: {base}")
    base.mkdir(parents=True, exist_ok=True)
    observed_indices: list[int] = []
    for entry in base.iterdir():
        match = _ATTEMPT_NAME.fullmatch(entry.name)
        if match is None or not entry.is_dir():
            raise FrozenRefitError(f"unexpected entry in immutable refit attempt root: {entry}")
        attempt_index = int(match.group(1))
        observed_indices.append(attempt_index)
        identity_path = entry / "attempt_identity.json"
        if not identity_path.exists() and not any(entry.iterdir()):
            continue
        if not identity_path.is_file():
            raise FrozenRefitError(
                f"existing refit attempt has no immutable identity: {entry}"
            )
        _verify_attempt_identity(
            identity_path,
            config,
            attempt_index=attempt_index,
        )
        completion_path = entry / "refit_completion.json"
        if completion_path.exists():
            load_refit_completion(completion_path, protocol=protocol, verify_sources=True)
            raise FrozenRefitError(
                f"a completed refit already exists for this recipe: {completion_path}"
            )
    attempt_index = max(observed_indices, default=-1) + 1
    if attempt_index > 99:
        raise FrozenRefitError("refit retry attempt limit of 100 has been exhausted")
    run_dir = base / f"attempt{attempt_index:02d}"
    try:
        run_dir.mkdir(exist_ok=False)
    except FileExistsError as error:
        raise FrozenRefitError(
            "concurrent refit attempt reservation detected; retry the command"
        ) from error
    write_new_json(
        run_dir / "attempt_identity.json",
        _attempt_identity(config, attempt_index=attempt_index),
    )
    return run_dir, attempt_index


def run_frozen_refit(
    config: RefitConfig,
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
    freeze: MultiSeedFreezeArtifact | None = None
    if isinstance(config, PostSweepRefitConfig):
        selection, freeze = _load_post_sweep_development(
            config,
            protocol=protocol,
            current_manifest_sha256="sha256:" + manifest_hash,
            current_normalization_sha256="sha256:" + normalization_file_hash,
        )
    else:
        selection = _load_selected_development(
            config,
            protocol_hash=protocol.protocol_hash,
            manifest_hash=manifest_hash,
        )

    runtime = select_device(config.runtime.device, enable_bf16=config.runtime.bf16)
    generator = seed_everything(config.runtime.seed)
    model = model_factory(config.model).to(runtime.device)
    model_metadata = _model_metadata(model, config.model)
    if isinstance(config, PostSweepRefitConfig) and model_metadata != dict(
        config.model_identity
    ):
        raise FrozenRefitError(
            "fresh model metadata or trainable parameter count differs from frozen identity"
        )
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
    if isinstance(config, PostSweepRefitConfig):
        optimizer = AdamW(
            model.parameters(),
            lr=config.optimization.learning_rate,
            betas=config.optimizer.betas,
            eps=config.optimizer.eps,
            weight_decay=config.optimization.weight_decay,
        )
        run_dir, attempt_index = _allocate_post_sweep_attempt(
            config,
            protocol=protocol,
        )
    else:
        optimizer = AdamW(
            model.parameters(),
            lr=config.optimization.learning_rate,
            weight_decay=config.optimization.weight_decay,
        )
        run_dir = config.output.root_dir / config.run_name
        attempt_index = None
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
    resolved_config["model"] = model_metadata
    resolved_config["selection_provenance"] = selection.to_dict()
    if freeze is not None:
        resolved_config["freeze_binding"] = {
            "path": str(freeze.path),
            "artifact_sha256": freeze.artifact_sha256,
            "comparison_id": freeze.comparison_id,
            "recipe_sha256": cast(PostSweepRefitConfig, config).recipe_sha256,
        }
        resolved_config["attempt_index"] = attempt_index
    resolved_config["effective_data"] = {"refit_records": dataset_size}
    resolved_config["checkpoint_roles"] = {
        "best_training_loss.ckpt": "diagnostic minimum training loss only",
        "last.ckpt": "crash-recovery state from the latest completed epoch",
        "final.ckpt": "authoritative frozen-epoch refit artifact",
    }
    resolved_config_hash = _canonical_hash(resolved_config)
    resolved_path = run_dir / "resolved_refit_config.json"
    protocol_path = run_dir / "protocol.json"
    metadata_path = run_dir / "refit_metadata.json"
    _atomic_json(
        resolved_path,
        {"config_hash": resolved_config_hash, "config": resolved_config},
    )
    _atomic_json(protocol_path, protocol.to_resolved_dict())

    started_at = datetime.now(UTC)
    metadata: dict[str, object] = {
        "status": "running",
        "run_kind": config.run_kind,
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
    if isinstance(config, PostSweepRefitConfig):
        metadata.update(
            {
                "comparison_id": config.comparison_id,
                "architecture": config.architecture,
                "confirmation_seed": config.confirmation_seed,
                "freeze_artifact_path": str(config.freeze_artifact_path),
                "freeze_artifact_sha256": config.freeze_artifact_sha256,
                "recipe_sha256": config.recipe_sha256,
                "initialization": "fresh",
                "attempt_index": attempt_index,
                "downstream_provenance": config.downstream_provenance.to_dict(),
            }
        )
    _atomic_json(metadata_path, metadata)

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
    completion_path: Path | None = None
    completion_sha256: str | None = None
    final_checkpoint_sha256: str | None = None
    if isinstance(config, PostSweepRefitConfig):
        try:
            final_checkpoint_sha256 = file_sha256(final_path)
        except MultiSeedFreezeError as error:
            raise FrozenRefitError(f"could not hash final refit checkpoint: {error}") from error
        metadata["final_checkpoint_sha256"] = final_checkpoint_sha256
        metadata["freeze_artifact_sha256"] = config.freeze_artifact_sha256
    _atomic_json(metadata_path, metadata)
    if isinstance(config, PostSweepRefitConfig):
        completion_path, completion_sha256 = _write_refit_completion(
            config=config,
            run_dir=run_dir,
            resolved_path=resolved_path,
            metadata_path=metadata_path,
            protocol_path=protocol_path,
            history_path=history_path,
            final_path=final_path,
            selection=selection,
            resolved_config_hash=resolved_config_hash,
            protocol_hash=protocol.protocol_hash,
            manifest_hash=manifest_hash,
            normalization_hash=normalization_file_hash,
        )
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
        completion_path=completion_path,
        completion_sha256=completion_sha256,
        freeze_artifact_sha256=(
            config.freeze_artifact_sha256
            if isinstance(config, PostSweepRefitConfig)
            else None
        ),
    )
