"""Strict declarative configuration for development-fold experiments."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self, cast

import yaml  # type: ignore[import-untyped]

from ecg_trust.protocol import MODEL_SELECTION_FOLDS, TRAIN_FOLDS

EXPERIMENT_CONFIG_SCHEMA_VERSION = 1
_RUN_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}\Z")


class ExperimentConfigError(ValueError):
    """Raised when an experiment configuration is incomplete or unsafe."""


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ExperimentConfigError(f"{context} must be a string-keyed mapping")
    return cast(Mapping[str, object], value)


def _keys(
    value: Mapping[str, object],
    *,
    required: set[str],
    context: str,
) -> None:
    missing = sorted(required.difference(value))
    unexpected = sorted(set(value).difference(required))
    if missing or unexpected:
        raise ExperimentConfigError(
            f"{context} has invalid keys; missing={missing}, unexpected={unexpected}"
        )


def _string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExperimentConfigError(f"{context} must be a non-empty string")
    return value


def _integer(value: object, context: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ExperimentConfigError(f"{context} must be an integer >= {minimum}")
    return value


def _number(value: object, context: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ExperimentConfigError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ExperimentConfigError(f"{context} must be finite and >= {minimum}")
    return result


def _boolean(value: object, context: str) -> bool:
    if not isinstance(value, bool):
        raise ExperimentConfigError(f"{context} must be boolean")
    return value


def _optional_positive_integer(value: object, context: str) -> int | None:
    return None if value is None else _integer(value, context, minimum=1)


def _fold_tuple(value: object, context: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise ExperimentConfigError(f"{context} must be a list of integers")
    return tuple(_integer(item, f"{context} item", minimum=1) for item in value)


def _resolve_path(value: object, context: str, base_dir: Path) -> Path:
    path = Path(_string(value, context))
    return path.resolve() if path.is_absolute() else (base_dir / path).resolve()


@dataclass(frozen=True, slots=True)
class DataConfig:
    manifest_path: Path
    dataset_root: Path
    normalization_path: Path
    max_train_records: int | None
    max_validation_records: int | None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object], *, base_dir: Path) -> Self:
        required = {
            "manifest",
            "dataset_root",
            "normalization",
            "max_train_records",
            "max_validation_records",
        }
        _keys(raw, required=required, context="data")
        return cls(
            manifest_path=_resolve_path(raw["manifest"], "data.manifest", base_dir),
            dataset_root=_resolve_path(raw["dataset_root"], "data.dataset_root", base_dir),
            normalization_path=_resolve_path(
                raw["normalization"], "data.normalization", base_dir
            ),
            max_train_records=_optional_positive_integer(
                raw["max_train_records"], "data.max_train_records"
            ),
            max_validation_records=_optional_positive_integer(
                raw["max_validation_records"], "data.max_validation_records"
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "manifest": str(self.manifest_path),
            "dataset_root": str(self.dataset_root),
            "normalization": str(self.normalization_path),
            "max_train_records": self.max_train_records,
            "max_validation_records": self.max_validation_records,
        }


@dataclass(frozen=True, slots=True)
class ModelConfig:
    architecture: Literal["resnet1d", "ecg_transformer"]
    preset: Literal["smoke", "matched_capacity"]

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> Self:
        _keys(raw, required={"architecture", "preset"}, context="model")
        architecture = _string(raw["architecture"], "model.architecture")
        preset = _string(raw["preset"], "model.preset")
        if architecture not in {"resnet1d", "ecg_transformer"}:
            raise ExperimentConfigError(
                "model.architecture must be 'resnet1d' or 'ecg_transformer'"
            )
        if preset not in {"smoke", "matched_capacity"}:
            raise ExperimentConfigError("model.preset must be 'smoke' or 'matched_capacity'")
        return cls(
            architecture=cast(Literal["resnet1d", "ecg_transformer"], architecture),
            preset=cast(Literal["smoke", "matched_capacity"], preset),
        )

    def to_dict(self) -> dict[str, object]:
        return {"architecture": self.architecture, "preset": self.preset}


@dataclass(frozen=True, slots=True)
class LoaderConfig:
    batch_size: int
    num_workers: int
    pin_memory: bool
    persistent_workers: bool

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> Self:
        required = {"batch_size", "num_workers", "pin_memory", "persistent_workers"}
        _keys(raw, required=required, context="loader")
        config = cls(
            batch_size=_integer(raw["batch_size"], "loader.batch_size", minimum=1),
            num_workers=_integer(raw["num_workers"], "loader.num_workers"),
            pin_memory=_boolean(raw["pin_memory"], "loader.pin_memory"),
            persistent_workers=_boolean(
                raw["persistent_workers"], "loader.persistent_workers"
            ),
        )
        if config.persistent_workers and config.num_workers == 0:
            raise ExperimentConfigError(
                "loader.persistent_workers requires loader.num_workers > 0"
            )
        return config

    def to_dict(self) -> dict[str, object]:
        return {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.persistent_workers,
        }


@dataclass(frozen=True, slots=True)
class OptimizationConfig:
    epochs: int
    learning_rate: float
    weight_decay: float
    warmup_epochs: int
    minimum_lr_ratio: float
    gradient_clip_norm: float
    early_stopping_patience: int
    early_stopping_min_delta: float
    scheduler: Literal["warmup_cosine"]

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> Self:
        required = {
            "epochs",
            "learning_rate",
            "weight_decay",
            "warmup_epochs",
            "minimum_lr_ratio",
            "gradient_clip_norm",
            "early_stopping_patience",
            "early_stopping_min_delta",
            "scheduler",
        }
        _keys(raw, required=required, context="optimization")
        scheduler = _string(raw["scheduler"], "optimization.scheduler")
        if scheduler != "warmup_cosine":
            raise ExperimentConfigError("optimization.scheduler must be 'warmup_cosine'")
        config = cls(
            epochs=_integer(raw["epochs"], "optimization.epochs", minimum=1),
            learning_rate=_number(
                raw["learning_rate"], "optimization.learning_rate", minimum=0.0
            ),
            weight_decay=_number(
                raw["weight_decay"], "optimization.weight_decay", minimum=0.0
            ),
            warmup_epochs=_integer(
                raw["warmup_epochs"], "optimization.warmup_epochs"
            ),
            minimum_lr_ratio=_number(
                raw["minimum_lr_ratio"], "optimization.minimum_lr_ratio", minimum=0.0
            ),
            gradient_clip_norm=_number(
                raw["gradient_clip_norm"], "optimization.gradient_clip_norm", minimum=0.0
            ),
            early_stopping_patience=_integer(
                raw["early_stopping_patience"],
                "optimization.early_stopping_patience",
                minimum=1,
            ),
            early_stopping_min_delta=_number(
                raw["early_stopping_min_delta"],
                "optimization.early_stopping_min_delta",
                minimum=0.0,
            ),
            scheduler="warmup_cosine",
        )
        if config.learning_rate <= 0.0:
            raise ExperimentConfigError("optimization.learning_rate must be positive")
        if config.gradient_clip_norm <= 0.0:
            raise ExperimentConfigError("optimization.gradient_clip_norm must be positive")
        if config.warmup_epochs >= config.epochs:
            raise ExperimentConfigError("optimization.warmup_epochs must be less than epochs")
        if config.minimum_lr_ratio > 1.0:
            raise ExperimentConfigError("optimization.minimum_lr_ratio must be in [0, 1]")
        return config

    def to_dict(self) -> dict[str, object]:
        return {
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "warmup_epochs": self.warmup_epochs,
            "minimum_lr_ratio": self.minimum_lr_ratio,
            "gradient_clip_norm": self.gradient_clip_norm,
            "early_stopping_patience": self.early_stopping_patience,
            "early_stopping_min_delta": self.early_stopping_min_delta,
            "scheduler": self.scheduler,
        }


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    seed: int
    device: str
    bf16: bool

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> Self:
        _keys(raw, required={"seed", "device", "bf16"}, context="runtime")
        seed = _integer(raw["seed"], "runtime.seed")
        if seed >= 2**32:
            raise ExperimentConfigError("runtime.seed must be smaller than 2**32")
        device = _string(raw["device"], "runtime.device").casefold()
        valid_device = device in {"auto", "cpu", "cuda"} or (
            device.startswith("cuda:") and device[5:].isdigit()
        )
        if not valid_device:
            raise ExperimentConfigError("runtime.device must be auto, cpu, cuda, or cuda:<index>")
        return cls(seed=seed, device=device, bf16=_boolean(raw["bf16"], "runtime.bf16"))

    def to_dict(self) -> dict[str, object]:
        return {"seed": self.seed, "device": self.device, "bf16": self.bf16}


@dataclass(frozen=True, slots=True)
class OutputConfig:
    root_dir: Path

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object], *, base_dir: Path) -> Self:
        _keys(raw, required={"root_dir"}, context="output")
        return cls(root_dir=_resolve_path(raw["root_dir"], "output.root_dir", base_dir))

    def to_dict(self) -> dict[str, object]:
        return {"root_dir": str(self.root_dir)}


@dataclass(frozen=True, slots=True)
class DevelopmentExperimentConfig:
    """Fully resolved config whose fold fields are fixed to development roles."""

    schema_version: int
    run_name: str
    train_folds: tuple[int, ...]
    validation_folds: tuple[int, ...]
    data: DataConfig
    model: ModelConfig
    loader: LoaderConfig
    optimization: OptimizationConfig
    runtime: RuntimeConfig
    output: OutputConfig

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object], *, base_dir: str | Path) -> Self:
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
        }
        _keys(raw, required=required, context="experiment")
        schema_version = _integer(raw["schema_version"], "schema_version", minimum=1)
        if schema_version != EXPERIMENT_CONFIG_SCHEMA_VERSION:
            raise ExperimentConfigError(
                f"schema_version must be {EXPERIMENT_CONFIG_SCHEMA_VERSION}"
            )
        run_name = _string(raw["run_name"], "run_name")
        if not _RUN_NAME.fullmatch(run_name) or run_name in {".", ".."}:
            raise ExperimentConfigError(
                "run_name must be a safe 1-80 character identifier"
            )
        folds = _mapping(raw["folds"], "folds")
        _keys(folds, required={"train", "model_selection"}, context="folds")
        train_folds = _fold_tuple(folds["train"], "folds.train")
        validation_folds = _fold_tuple(
            folds["model_selection"], "folds.model_selection"
        )
        if train_folds != TRAIN_FOLDS or validation_folds != MODEL_SELECTION_FOLDS:
            raise ExperimentConfigError(
                "development runner folds are immutable: train must be 1-7 and "
                "model_selection must be fold 8"
            )
        resolved_base = Path(base_dir).resolve()
        return cls(
            schema_version=schema_version,
            run_name=run_name,
            train_folds=train_folds,
            validation_folds=validation_folds,
            data=DataConfig.from_mapping(
                _mapping(raw["data"], "data"), base_dir=resolved_base
            ),
            model=ModelConfig.from_mapping(_mapping(raw["model"], "model")),
            loader=LoaderConfig.from_mapping(_mapping(raw["loader"], "loader")),
            optimization=OptimizationConfig.from_mapping(
                _mapping(raw["optimization"], "optimization")
            ),
            runtime=RuntimeConfig.from_mapping(_mapping(raw["runtime"], "runtime")),
            output=OutputConfig.from_mapping(
                _mapping(raw["output"], "output"), base_dir=resolved_base
            ),
        )

    def to_resolved_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "run_name": self.run_name,
            "folds": {
                "train": list(self.train_folds),
                "model_selection": list(self.validation_folds),
            },
            "data": self.data.to_dict(),
            "model": self.model.to_dict(),
            "loader": self.loader.to_dict(),
            "optimization": self.optimization.to_dict(),
            "runtime": self.runtime.to_dict(),
            "output": self.output.to_dict(),
        }

    @property
    def config_hash(self) -> str:
        serialized = json.dumps(
            self.to_resolved_dict(),
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return "sha256:" + hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def load_experiment_config(
    path: str | Path,
    *,
    base_dir: str | Path | None = None,
) -> DevelopmentExperimentConfig:
    """Load strict YAML and resolve relative paths against ``base_dir``."""

    config_path = Path(path)
    try:
        decoded: object = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ExperimentConfigError(f"could not read config {config_path}: {error}") from error
    except yaml.YAMLError as error:
        raise ExperimentConfigError(f"invalid YAML in {config_path}: {error}") from error
    resolved_base = Path(base_dir) if base_dir is not None else config_path.parent
    return DevelopmentExperimentConfig.from_mapping(
        _mapping(decoded, str(config_path)), base_dir=resolved_base
    )
