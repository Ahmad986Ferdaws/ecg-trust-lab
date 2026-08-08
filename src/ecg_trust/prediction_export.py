"""Provenance-bound checkpoint inference and immutable prediction export.

The exporter is intentionally stricter than a generic inference helper.  It
accepts only a selected development checkpoint for fold 8 or the authoritative
frozen-refit checkpoint for folds 9/10, validates the complete lineage before
constructing a dataset, and writes the existing immutable prediction format.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from collections.abc import Mapping, Sized
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Literal, Protocol, cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from ecg_trust.constants import TARGET_COLUMNS
from ecg_trust.data.dataset import NormalizationStats, PTBXLDataset
from ecg_trust.data.manifest import sha256_file
from ecg_trust.experiment_config import (
    DevelopmentExperimentConfig,
    ExperimentConfigError,
    ModelConfig,
)
from ecg_trust.experiment_runner import build_experiment_model, training_manifest_sha256
from ecg_trust.models import MATCHED_CAPACITY_PRESET, count_parameters
from ecg_trust.predictions import (
    PredictionArtifactFiles,
    create_prediction_artifact,
    save_prediction_artifact,
)
from ecg_trust.protocol import (
    TRAIN_FOLDS,
    ExperimentProtocol,
    FinalTestAccessToken,
    FoldRole,
)
from ecg_trust.refit_config import FrozenRefitConfig, RefitConfigError
from ecg_trust.training import (
    CheckpointValidationError,
    EarlyStopping,
    TrainingRuntime,
    load_checkpoint,
    seed_dataloader_worker,
    select_device,
)

ExportLineage = Literal["development", "frozen_refit"]
_SUPPORTED_EXPORT_ROLES = frozenset(
    {FoldRole.MODEL_SELECTION, FoldRole.CALIBRATION, FoldRole.FINAL_TEST}
)


class PredictionExportError(RuntimeError):
    """Raised when checkpoint inference cannot preserve scientific lineage."""


class ExportDatasetFactory(Protocol):
    """Construct a role-restricted ECG dataset for prediction."""

    def __call__(
        self,
        manifest: pd.DataFrame,
        root_dir: Path,
        *,
        folds: tuple[int, ...],
        normalization: NormalizationStats,
        protocol: ExperimentProtocol,
        test_access: FinalTestAccessToken | None,
    ) -> Dataset[tuple[Tensor, Tensor]]: ...


@dataclass(frozen=True, slots=True)
class PredictionExportRequest:
    """Files and runtime settings for one immutable fold prediction export."""

    checkpoint_path: Path
    resolved_config_path: Path
    output_path: Path
    fold_role: FoldRole
    run_metadata_path: Path | None = None
    manifest_path: Path | None = None
    dataset_root: Path | None = None
    normalization_path: Path | None = None
    batch_size: int | None = None
    num_workers: int | None = None
    pin_memory: bool = True
    persistent_workers: bool = True
    device: str = "auto"
    bf16: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.fold_role, FoldRole):
            raise TypeError("fold_role must be a FoldRole")
        if self.fold_role not in _SUPPORTED_EXPORT_ROLES:
            raise PredictionExportError(
                "prediction export supports only model_selection, calibration, or final_test"
            )
        if self.output_path.suffix.casefold() != ".npz":
            raise PredictionExportError("prediction output path must end in .npz")
        if self.batch_size is not None and (
            isinstance(self.batch_size, bool) or self.batch_size < 1
        ):
            raise PredictionExportError("batch_size must be a positive integer")
        if self.num_workers is not None and (
            isinstance(self.num_workers, bool) or self.num_workers < 0
        ):
            raise PredictionExportError("num_workers must be a non-negative integer")
        if not isinstance(self.pin_memory, bool) or not isinstance(
            self.persistent_workers, bool
        ):
            raise TypeError("pin_memory and persistent_workers must be booleans")
        if not isinstance(self.device, str) or not self.device.strip():
            raise PredictionExportError("device must be a non-empty device string")
        if not isinstance(self.bf16, bool):
            raise TypeError("bf16 must be boolean")


@dataclass(frozen=True, slots=True)
class PredictionExportResult:
    """Saved artifact identities and the runtime that produced them."""

    files: PredictionArtifactFiles
    lineage: ExportLineage
    fold_role: FoldRole
    folds: tuple[int, ...]
    record_count: int
    model_name: str
    model_seed: int
    checkpoint_sha256: str
    config_hash: str
    manifest_hash: str
    normalization_sha256: str
    device: str
    bf16_enabled: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "files": self.files.to_dict(),
            "lineage": self.lineage,
            "fold_role": self.fold_role.value,
            "folds": list(self.folds),
            "record_count": self.record_count,
            "model_name": self.model_name,
            "model_seed": self.model_seed,
            "checkpoint_sha256": self.checkpoint_sha256,
            "config_hash": self.config_hash,
            "manifest_hash": self.manifest_hash,
            "normalization_sha256": self.normalization_sha256,
            "device": self.device,
            "bf16_enabled": self.bf16_enabled,
        }


@dataclass(frozen=True, slots=True)
class _ResolvedRun:
    lineage: ExportLineage
    config: dict[str, object]
    config_hash: str
    typed_config: DevelopmentExperimentConfig | FrozenRefitConfig
    model_config: ModelConfig
    model_metadata: Mapping[str, object]
    run_name: str
    seed: int
    manifest_path: Path
    dataset_root: Path
    normalization_path: Path
    batch_size: int
    num_workers: int
    learning_rate: float
    weight_decay: float


@dataclass(frozen=True, slots=True)
class _ValidatedInputs:
    manifest: pd.DataFrame
    selected_manifest: pd.DataFrame
    normalization: NormalizationStats
    manifest_hash: str
    normalization_hash: str
    expected_checkpoint_epoch: int


class _IndexedDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    """Attach stable row positions to an otherwise ordinary ECG dataset."""

    def __init__(self, dataset: Dataset[tuple[Tensor, Tensor]]) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(cast(Sized, self.dataset))

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        signal, target = self.dataset[index]
        return torch.tensor(index, dtype=torch.int64), signal, target


def _default_dataset_factory(
    manifest: pd.DataFrame,
    root_dir: Path,
    *,
    folds: tuple[int, ...],
    normalization: NormalizationStats,
    protocol: ExperimentProtocol,
    test_access: FinalTestAccessToken | None,
) -> Dataset[tuple[Tensor, Tensor]]:
    return PTBXLDataset(
        manifest,
        root_dir,
        folds=folds,
        normalization=normalization,
        protocol=protocol,
        test_access=test_access,
    )


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise PredictionExportError(f"{context} must be a string-keyed mapping")
    return cast(Mapping[str, object], value)


def _keys(
    value: Mapping[str, object],
    *,
    required: set[str],
    context: str,
    optional: set[str] | None = None,
) -> None:
    allowed = required | (optional or set())
    missing = sorted(required.difference(value))
    unexpected = sorted(set(value).difference(allowed))
    if missing or unexpected:
        raise PredictionExportError(
            f"{context} keys are invalid; missing={missing}, unexpected={unexpected}"
        )


def _string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PredictionExportError(f"{context} must be a non-empty string")
    return value


def _integer(value: object, context: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise PredictionExportError(f"{context} must be an integer >= {minimum}")
    return value


def _boolean(value: object, context: str) -> bool:
    if not isinstance(value, bool):
        raise PredictionExportError(f"{context} must be boolean")
    return value


def _read_json(path: Path, context: str) -> Mapping[str, object]:
    if not path.is_file():
        raise PredictionExportError(f"{context} is missing: {path}")
    if path.stat().st_size > 100_000_000:
        raise PredictionExportError(f"{context} is unreasonably large")
    try:
        decoded: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PredictionExportError(f"could not decode {context}: {error}") from error
    return _mapping(decoded, context)


def _canonical_mapping(value: Mapping[str, object]) -> tuple[dict[str, object], str]:
    try:
        serialized = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        decoded: object = json.loads(serialized)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise PredictionExportError("resolved config must be finite JSON") from error
    if not isinstance(decoded, dict):
        raise PredictionExportError("resolved config must be a JSON object")
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return cast(dict[str, object], decoded), f"sha256:{digest}"


def _model_selection_mapping(raw: Mapping[str, object]) -> dict[str, object]:
    model = _mapping(raw.get("model"), "resolved config model")
    return {
        "architecture": model.get("architecture"),
        "preset": model.get("preset"),
    }


def _development_base(raw: Mapping[str, object]) -> dict[str, object]:
    required = {
        "schema_version",
        "run_name",
        "folds",
        "data",
        "model",
        "loader",
        "optimization",
        "runtime",
        "output",
        "effective_data",
        "optimizer",
    }
    _keys(raw, required=required, context="resolved development config")
    return {
        key: (_model_selection_mapping(raw) if key == "model" else raw[key])
        for key in (
            "schema_version",
            "run_name",
            "folds",
            "data",
            "model",
            "loader",
            "optimization",
            "runtime",
            "output",
        )
    }


def _refit_base(raw: Mapping[str, object]) -> dict[str, object]:
    required = {
        "schema_version",
        "run_kind",
        "run_name",
        "folds",
        "data",
        "selection",
        "model",
        "loader",
        "optimization",
        "runtime",
        "output",
        "selection_provenance",
        "effective_data",
        "checkpoint_roles",
    }
    _keys(raw, required=required, context="resolved frozen-refit config")
    return {
        key: (_model_selection_mapping(raw) if key == "model" else raw[key])
        for key in (
            "schema_version",
            "run_kind",
            "run_name",
            "folds",
            "data",
            "selection",
            "model",
            "loader",
            "optimization",
            "runtime",
            "output",
        )
    }


def _load_resolved_run(path: Path) -> _ResolvedRun:
    wrapper = _read_json(path, "resolved config wrapper")
    _keys(wrapper, required={"config_hash", "config"}, context="resolved config wrapper")
    raw_config = _mapping(wrapper["config"], "resolved config")
    config, computed_hash = _canonical_mapping(raw_config)
    stored_hash = _string(wrapper["config_hash"], "resolved config hash")
    if stored_hash != computed_hash:
        raise PredictionExportError("resolved config wrapper hash does not match its content")

    try:
        if config.get("run_kind") == "frozen_refit":
            typed: DevelopmentExperimentConfig | FrozenRefitConfig = (
                FrozenRefitConfig.from_mapping(_refit_base(config), base_dir=path.parent)
            )
            lineage: ExportLineage = "frozen_refit"
            model_config = typed.model
            manifest_path = typed.data.manifest_path
            dataset_root = typed.data.dataset_root
            normalization_path = typed.data.normalization_path
            learning_rate = typed.optimization.learning_rate
            weight_decay = typed.optimization.weight_decay
        elif "run_kind" not in config:
            typed = DevelopmentExperimentConfig.from_mapping(
                _development_base(config), base_dir=path.parent
            )
            lineage = "development"
            model_config = typed.model
            manifest_path = typed.data.manifest_path
            dataset_root = typed.data.dataset_root
            normalization_path = typed.data.normalization_path
            learning_rate = typed.optimization.learning_rate
            weight_decay = typed.optimization.weight_decay
        else:
            raise PredictionExportError("resolved config has an unsupported run_kind")
    except (ExperimentConfigError, RefitConfigError) as error:
        raise PredictionExportError(f"invalid resolved run config: {error}") from error

    model_metadata = _mapping(config["model"], "resolved model metadata")
    return _ResolvedRun(
        lineage=lineage,
        config=config,
        config_hash=computed_hash,
        typed_config=typed,
        model_config=model_config,
        model_metadata=model_metadata,
        run_name=typed.run_name,
        seed=typed.runtime.seed,
        manifest_path=manifest_path,
        dataset_root=dataset_root,
        normalization_path=normalization_path,
        batch_size=typed.loader.batch_size,
        num_workers=typed.loader.num_workers,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )


def _json_normalize(value: object) -> object:
    try:
        return json.loads(json.dumps(value, allow_nan=False, sort_keys=True))
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise PredictionExportError("model metadata is not finite JSON") from error


def _validate_model_metadata(model: nn.Module, run: _ResolvedRun) -> None:
    required = {
        "architecture",
        "preset",
        "class",
        "trainable_parameters",
        "resolved_architecture_config",
    }
    optional = (
        {"capacity_match"}
        if run.lineage == "development" and run.model_config.preset == "matched_capacity"
        else set()
    )
    _keys(
        run.model_metadata,
        required=required,
        optional=optional,
        context="resolved model metadata",
    )
    expected_class = f"{type(model).__module__}.{type(model).__qualname__}"
    comparisons: dict[str, object] = {
        "architecture": run.model_config.architecture,
        "preset": run.model_config.preset,
        "class": expected_class,
        "trainable_parameters": count_parameters(model),
    }
    for field, expected in comparisons.items():
        if run.model_metadata.get(field) != expected:
            raise PredictionExportError(
                f"resolved model metadata {field} does not match the instantiated model"
            )
    raw_config = getattr(model, "config", None)
    if raw_config is None or not is_dataclass(raw_config) or isinstance(raw_config, type):
        raise PredictionExportError("instantiated model has no dataclass architecture config")
    if _json_normalize(run.model_metadata["resolved_architecture_config"]) != _json_normalize(
        asdict(raw_config)
    ):
        raise PredictionExportError(
            "resolved architecture config does not match the instantiated model"
        )
    if optional and _json_normalize(run.model_metadata["capacity_match"]) != _json_normalize(
        MATCHED_CAPACITY_PRESET.metadata()
    ):
        raise PredictionExportError("matched-capacity metadata does not match the preset")


def _read_manifest(path: Path) -> pd.DataFrame:
    try:
        if path.suffix.casefold() == ".parquet":
            frame = pd.read_parquet(path)
        elif path.suffix.casefold() == ".csv":
            frame = pd.read_csv(path)
        else:
            raise PredictionExportError("manifest must be a .parquet or .csv file")
    except (OSError, ValueError) as error:
        raise PredictionExportError(f"could not load manifest {path}: {error}") from error
    if frame.empty:
        raise PredictionExportError("manifest must not be empty")
    return cast(pd.DataFrame, frame)


def _validated_manifest(
    frame: pd.DataFrame,
    *,
    expected_folds: tuple[int, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"ecg_id", "patient_id", "strat_fold", "record_path", *TARGET_COLUMNS}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise PredictionExportError(f"manifest is missing required columns: {missing}")
    if frame["ecg_id"].isna().any() or frame["patient_id"].isna().any():
        raise PredictionExportError("manifest ECG and patient identifiers must not be missing")
    if frame["ecg_id"].duplicated().any():
        raise PredictionExportError("manifest ecg_id values must be unique")
    try:
        folds = pd.to_numeric(frame["strat_fold"], errors="raise").to_numpy(
            dtype=np.float64
        )
        targets = (
            frame.loc[:, list(TARGET_COLUMNS)]
            .apply(pd.to_numeric, errors="raise")
            .to_numpy(dtype=np.float64)
        )
    except (TypeError, ValueError) as error:
        raise PredictionExportError("manifest folds and targets must be numeric") from error
    if not np.isfinite(folds).all() or not np.equal(folds, np.floor(folds)).all():
        raise PredictionExportError("manifest folds must be finite integers")
    if not np.isin(folds, np.arange(1, 11)).all():
        raise PredictionExportError("manifest folds must lie in the canonical range 1-10")
    if not np.isfinite(targets).all() or not np.isin(targets, (0.0, 1.0)).all():
        raise PredictionExportError("manifest targets must be finite binary values")
    fold_frame = frame.copy()
    fold_frame["strat_fold"] = folds.astype(np.int8)
    counts = fold_frame.groupby("patient_id", dropna=False)["strat_fold"].nunique()
    leaked = counts[counts != 1]
    if not leaked.empty:
        preview = ", ".join(str(value) for value in leaked.index[:10])
        raise PredictionExportError(
            f"patients occur in multiple folds; first offending IDs: {preview}"
        )
    selected = fold_frame.loc[
        fold_frame["strat_fold"].isin(expected_folds)
    ].reset_index(drop=True)
    if selected.empty:
        raise PredictionExportError(
            f"manifest has no rows for requested role folds {expected_folds}"
        )
    observed = tuple(sorted(int(value) for value in selected["strat_fold"].unique()))
    if observed != expected_folds:
        raise PredictionExportError(
            f"requested role requires folds {expected_folds}, but manifest provides {observed}"
        )
    return fold_frame, selected


def _validate_normalization(
    stats: NormalizationStats,
    *,
    manifest: pd.DataFrame,
    protocol: ExperimentProtocol,
) -> None:
    provenance = stats.provenance
    if provenance.dataset_version != protocol.dataset_version:
        raise PredictionExportError("normalization dataset version does not match protocol")
    if provenance.training_folds != TRAIN_FOLDS:
        raise PredictionExportError("normalization must be fitted only on folds 1-7")
    if provenance.path_column != "record_path" or provenance.fold_column != "strat_fold":
        raise PredictionExportError("normalization manifest columns are not canonical")
    if provenance.target_columns != TARGET_COLUMNS:
        raise PredictionExportError("normalization target order is not canonical")
    if provenance.samples_per_record != 1_000 or not math.isclose(
        provenance.sampling_frequency_hz, 100.0, rel_tol=0.0, abs_tol=1e-9
    ):
        raise PredictionExportError("normalization is not for 100 Hz, 1000-sample ECGs")
    training_rows = int(manifest["strat_fold"].isin(TRAIN_FOLDS).sum())
    if provenance.record_count != training_rows:
        raise PredictionExportError("normalization training record count does not match manifest")
    try:
        observed_hash = training_manifest_sha256(manifest, TRAIN_FOLDS)
    except (OSError, ValueError, RuntimeError) as error:
        raise PredictionExportError(
            f"could not verify normalization provenance: {error}"
        ) from error
    if observed_hash != provenance.manifest_sha256:
        raise PredictionExportError(
            "normalization provenance does not match current folds-1-7 manifest rows"
        )


def _metadata_path(request: PredictionExportRequest, run: _ResolvedRun) -> Path:
    if request.run_metadata_path is not None:
        return request.run_metadata_path
    filename = "run_metadata.json" if run.lineage == "development" else "refit_metadata.json"
    return request.resolved_config_path.parent / filename


def _validate_common_metadata(
    metadata: Mapping[str, object],
    *,
    run: _ResolvedRun,
    protocol: ExperimentProtocol,
    manifest_hash: str,
    normalization_hash: str,
    normalization: NormalizationStats,
) -> None:
    if metadata.get("status") != "complete":
        raise PredictionExportError("run metadata status must be complete")
    comparisons: dict[str, object] = {
        "seed": run.seed,
        "resolved_config_hash": run.config_hash,
        "protocol_hash": protocol.protocol_hash,
        "manifest_hash": manifest_hash,
        "normalization_file_hash": normalization_hash,
    }
    for field, expected in comparisons.items():
        if metadata.get(field) != expected:
            raise PredictionExportError(f"run metadata {field} does not match export inputs")
    stored_provenance = _mapping(
        metadata.get("normalization_provenance"), "run normalization provenance"
    )
    if _json_normalize(stored_provenance) != _json_normalize(
        normalization.provenance.to_dict()
    ):
        raise PredictionExportError("run normalization provenance does not match its file")


def _validate_lineage_metadata(
    metadata: Mapping[str, object],
    *,
    request: PredictionExportRequest,
    run: _ResolvedRun,
) -> int:
    if run.lineage == "development":
        if request.checkpoint_path.name != "best.ckpt":
            raise PredictionExportError(
                "development fold-8 export requires the selected best.ckpt"
            )
        if request.fold_role is not FoldRole.MODEL_SELECTION:
            raise PredictionExportError(
                "development checkpoints may export only model_selection fold 8"
            )
        return _integer(metadata.get("best_epoch"), "run metadata best_epoch")

    if request.checkpoint_path.name != "final.ckpt":
        raise PredictionExportError(
            "calibration/final export requires the authoritative refit final.ckpt"
        )
    if request.fold_role not in {FoldRole.CALIBRATION, FoldRole.FINAL_TEST}:
        raise PredictionExportError(
            "frozen-refit checkpoints may export only calibration or final_test"
        )
    if not isinstance(run.typed_config, FrozenRefitConfig):
        raise PredictionExportError("internal frozen-refit lineage type mismatch")
    refit = run.typed_config
    requirements: dict[str, object] = {
        "run_kind": "frozen_refit",
        "refit_folds": list(range(1, 9)),
        "normalization_folds": list(TRAIN_FOLDS),
        "frozen_epochs": refit.selection.frozen_epochs,
        "completed_epochs": refit.selection.frozen_epochs,
        "early_stopping_enabled": False,
        "model_selection_enabled": False,
        "authoritative_checkpoint": "final.ckpt",
        "final_epoch": refit.selection.frozen_epochs - 1,
    }
    for field, expected in requirements.items():
        if metadata.get(field) != expected:
            raise PredictionExportError(f"refit metadata {field} is not authoritative")
    metadata_selection = _mapping(
        metadata.get("selection_provenance"), "refit selection provenance"
    )
    config_selection = _mapping(
        run.config.get("selection_provenance"), "resolved refit selection provenance"
    )
    if _json_normalize(metadata_selection) != _json_normalize(config_selection):
        raise PredictionExportError("refit selection provenance does not match resolved config")
    checkpoint_roles = _mapping(
        run.config.get("checkpoint_roles"), "resolved refit checkpoint roles"
    )
    if checkpoint_roles.get("final.ckpt") != "authoritative frozen-epoch refit artifact":
        raise PredictionExportError(
            "resolved refit config does not declare final.ckpt authoritative"
        )
    return refit.selection.frozen_epochs - 1


def _validate_inputs(
    request: PredictionExportRequest,
    *,
    run: _ResolvedRun,
    protocol: ExperimentProtocol,
    expected_folds: tuple[int, ...],
) -> _ValidatedInputs:
    manifest_path = request.manifest_path or run.manifest_path
    normalization_path = request.normalization_path or run.normalization_path
    try:
        manifest_hash = sha256_file(manifest_path)
        normalization_hash = sha256_file(normalization_path)
        normalization = NormalizationStats.load(normalization_path)
    except (OSError, ValueError) as error:
        raise PredictionExportError(f"could not validate data inputs: {error}") from error
    manifest = _read_manifest(manifest_path)
    manifest, selected = _validated_manifest(manifest, expected_folds=expected_folds)
    _validate_normalization(normalization, manifest=manifest, protocol=protocol)
    metadata = _read_json(_metadata_path(request, run), "run metadata")
    _validate_common_metadata(
        metadata,
        run=run,
        protocol=protocol,
        manifest_hash=manifest_hash,
        normalization_hash=normalization_hash,
        normalization=normalization,
    )
    expected_epoch = _validate_lineage_metadata(metadata, request=request, run=run)
    return _ValidatedInputs(
        manifest=manifest,
        selected_manifest=selected,
        normalization=normalization,
        manifest_hash=manifest_hash,
        normalization_hash=normalization_hash,
        expected_checkpoint_epoch=expected_epoch,
    )


def _load_inference_model(
    request: PredictionExportRequest,
    *,
    run: _ResolvedRun,
    protocol: ExperimentProtocol,
    inputs: _ValidatedInputs,
) -> tuple[nn.Module, str, int]:
    try:
        model = build_experiment_model(run.model_config)
    except (RuntimeError, ValueError) as error:
        raise PredictionExportError(f"could not instantiate resolved model: {error}") from error
    _validate_model_metadata(model, run)
    optimizer = AdamW(
        model.parameters(),
        lr=run.learning_rate,
        weight_decay=run.weight_decay,
    )
    stopper: EarlyStopping | None = None
    if isinstance(run.typed_config, DevelopmentExperimentConfig):
        stopper = EarlyStopping(
            patience=run.typed_config.optimization.early_stopping_patience,
            mode="max",
            min_delta=run.typed_config.optimization.early_stopping_min_delta,
        )
    try:
        checkpoint = load_checkpoint(
            request.checkpoint_path,
            model=model,
            optimizer=optimizer,
            scaler=None,
            expected_protocol_hash=protocol.protocol_hash,
            expected_config=run.config,
            expected_manifest_hash=inputs.manifest_hash,
            early_stopping=stopper,
            map_location="cpu",
            strict_model=True,
        )
        checkpoint_hash = sha256_file(request.checkpoint_path)
    except (OSError, RuntimeError, CheckpointValidationError, ValueError) as error:
        raise PredictionExportError(f"checkpoint validation failed: {error}") from error
    if checkpoint.config_hash != run.config_hash:
        raise PredictionExportError("checkpoint config hash does not match resolved wrapper")
    if checkpoint.epoch != inputs.expected_checkpoint_epoch:
        raise PredictionExportError(
            "checkpoint epoch does not match the completed run's authoritative epoch"
        )
    if run.lineage == "development" and (
        stopper is None or stopper.best_epoch != checkpoint.epoch
    ):
        raise PredictionExportError("development checkpoint is not its selected best epoch")
    del optimizer
    return model, checkpoint_hash, checkpoint.epoch


def _seed_inference(seed: int, runtime: TrainingRuntime) -> torch.Generator:
    """Seed deterministic inference without initializing CUDA for CPU exports."""

    if not 0 <= seed < 2**32:
        raise PredictionExportError("model seed is outside the uint32 range")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if runtime.device.type == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def _loader(
    dataset: Dataset[tuple[Tensor, Tensor]],
    *,
    request: PredictionExportRequest,
    run: _ResolvedRun,
    runtime: TrainingRuntime,
    generator: torch.Generator,
) -> DataLoader[tuple[Tensor, Tensor, Tensor]]:
    batch_size = request.batch_size or run.batch_size
    workers = run.num_workers if request.num_workers is None else request.num_workers
    return DataLoader(
        _IndexedDataset(dataset),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=request.pin_memory and runtime.device.type == "cuda",
        persistent_workers=request.persistent_workers and workers > 0,
        worker_init_fn=seed_dataloader_worker if workers > 0 else None,
        generator=generator,
        drop_last=False,
    )


def _run_inference(
    model: nn.Module,
    loader: DataLoader[tuple[Tensor, Tensor, Tensor]],
    *,
    runtime: TrainingRuntime,
    selected_manifest: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    positions: list[Tensor] = []
    logits: list[Tensor] = []
    targets: list[Tensor] = []
    with torch.inference_mode():
        for raw_positions, signals, raw_targets in loader:
            if raw_positions.ndim != 1 or signals.ndim != 3 or raw_targets.ndim != 2:
                raise PredictionExportError("prediction batch has invalid tensor ranks")
            if signals.shape[0] != raw_targets.shape[0] or raw_targets.shape[1] != len(
                TARGET_COLUMNS
            ):
                raise PredictionExportError("prediction batch signals and targets do not align")
            if not torch.isfinite(signals).all().item():
                raise PredictionExportError("prediction batch contains non-finite ECG values")
            if not torch.isfinite(raw_targets).all().item() or not torch.all(
                (raw_targets == 0.0) | (raw_targets == 1.0)
            ).item():
                raise PredictionExportError("prediction batch targets must be finite and binary")
            device_signals = signals.to(
                device=runtime.device,
                dtype=torch.float32,
                non_blocking=runtime.device.type == "cuda",
            )
            with torch.autocast(
                device_type=runtime.device.type,
                dtype=torch.bfloat16,
                enabled=runtime.bf16_enabled,
            ):
                raw_logits = model(device_signals)
            if not isinstance(raw_logits, Tensor) or raw_logits.shape != raw_targets.shape:
                raise PredictionExportError(
                    "model logits must align with five-label prediction targets"
                )
            batch_logits = raw_logits.detach().to(device="cpu", dtype=torch.float32)
            if not torch.isfinite(batch_logits).all().item():
                raise PredictionExportError("model produced non-finite logits")
            positions.append(raw_positions.to(device="cpu", dtype=torch.int64))
            logits.append(batch_logits)
            targets.append(raw_targets.to(device="cpu", dtype=torch.float32))
    if not positions:
        raise PredictionExportError("prediction loader produced no samples")
    all_positions = torch.cat(positions).numpy()
    expected_positions = np.arange(len(selected_manifest), dtype=np.int64)
    if not np.array_equal(all_positions, expected_positions):
        raise PredictionExportError("prediction loader row order is not deterministic and complete")
    all_logits = torch.cat(logits).numpy().astype(np.float64, copy=False)
    all_targets = torch.cat(targets).numpy().astype(np.int8, copy=False)
    expected_targets = selected_manifest.loc[:, list(TARGET_COLUMNS)].to_numpy(
        dtype=np.int8
    )
    if not np.array_equal(all_targets, expected_targets):
        raise PredictionExportError("dataset targets do not align with selected manifest rows")
    return all_logits, all_targets


def export_checkpoint_predictions(
    request: PredictionExportRequest,
    *,
    protocol: ExperimentProtocol,
    test_access: FinalTestAccessToken | None = None,
    dataset_factory: ExportDatasetFactory = _default_dataset_factory,
) -> PredictionExportResult:
    """Validate one checkpoint lineage, infer one role, and save predictions."""

    if not isinstance(request, PredictionExportRequest):
        raise TypeError("request must be a PredictionExportRequest")
    if not isinstance(protocol, ExperimentProtocol):
        raise TypeError("protocol must be an ExperimentProtocol")
    expected_folds = protocol.folds_for(request.fold_role, test_access=test_access)
    run = _load_resolved_run(request.resolved_config_path)
    inputs = _validate_inputs(
        request,
        run=run,
        protocol=protocol,
        expected_folds=expected_folds,
    )
    model, checkpoint_hash, checkpoint_epoch = _load_inference_model(
        request,
        run=run,
        protocol=protocol,
        inputs=inputs,
    )
    try:
        runtime = select_device(request.device, enable_bf16=request.bf16)
    except (RuntimeError, ValueError) as error:
        raise PredictionExportError(f"could not select inference runtime: {error}") from error
    generator = _seed_inference(run.seed, runtime)
    model = model.to(runtime.device)
    selected = inputs.selected_manifest
    dataset_root = request.dataset_root or run.dataset_root
    try:
        dataset = dataset_factory(
            selected,
            dataset_root,
            folds=expected_folds,
            normalization=inputs.normalization,
            protocol=protocol,
            test_access=test_access,
        )
    except (OSError, PermissionError, RuntimeError, ValueError) as error:
        raise PredictionExportError(f"could not construct prediction dataset: {error}") from error
    if len(cast(Sized, dataset)) != len(selected):
        raise PredictionExportError("prediction dataset size does not match selected manifest")
    data_loader = _loader(
        dataset,
        request=request,
        run=run,
        runtime=runtime,
        generator=generator,
    )
    raw_logits, targets = _run_inference(
        model,
        data_loader,
        runtime=runtime,
        selected_manifest=selected,
    )
    final_purpose = (
        test_access.purpose
        if request.fold_role is FoldRole.FINAL_TEST and test_access is not None
        else None
    )
    artifact = create_prediction_artifact(
        ecg_id=selected["ecg_id"].to_numpy(),
        patient_id=selected["patient_id"].to_numpy(),
        strat_fold=selected["strat_fold"].to_numpy(dtype=np.int8),
        targets=targets,
        raw_logits=raw_logits,
        model_name=run.run_name,
        model_seed=run.seed,
        protocol=protocol,
        config_hash=run.config_hash,
        manifest_hash=inputs.manifest_hash,
        fold_role=request.fold_role,
        producer="ecg_trust.prediction_export",
        extra_metadata={
            "lineage": run.lineage,
            "checkpoint_path": str(request.checkpoint_path.resolve()),
            "checkpoint_sha256": checkpoint_hash,
            "checkpoint_epoch": checkpoint_epoch,
            "resolved_config_path": str(request.resolved_config_path.resolve()),
            "normalization_sha256": inputs.normalization_hash,
            "inference_device": str(runtime.device),
            "inference_bf16": runtime.bf16_enabled,
            "inference_batch_size": request.batch_size or run.batch_size,
            "inference_num_workers": (
                run.num_workers if request.num_workers is None else request.num_workers
            ),
            "final_test_purpose": final_purpose,
        },
        test_access=test_access,
    )
    files = save_prediction_artifact(
        artifact,
        request.output_path,
        protocol=protocol,
        test_access=test_access,
    )
    return PredictionExportResult(
        files=files,
        lineage=run.lineage,
        fold_role=request.fold_role,
        folds=artifact.folds,
        record_count=artifact.n_samples,
        model_name=artifact.model_name,
        model_seed=artifact.model_seed,
        checkpoint_sha256=checkpoint_hash,
        config_hash=artifact.config_hash,
        manifest_hash=artifact.manifest_hash,
        normalization_sha256=inputs.normalization_hash,
        device=str(runtime.device),
        bf16_enabled=runtime.bf16_enabled,
    )


__all__ = [
    "ExportDatasetFactory",
    "PredictionExportError",
    "PredictionExportRequest",
    "PredictionExportResult",
    "export_checkpoint_predictions",
]
