"""Typed configuration for frozen folds-1-8 model refits."""

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

from ecg_trust.experiment_config import LoaderConfig, ModelConfig, OutputConfig, RuntimeConfig
from ecg_trust.protocol import MODEL_SELECTION_FOLDS, TRAIN_FOLDS

REFIT_CONFIG_SCHEMA_VERSION = 1
REFIT_FOLDS: tuple[int, ...] = TRAIN_FOLDS + MODEL_SELECTION_FOLDS
_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}\Z")


class RefitConfigError(ValueError):
    """Raised when a frozen-refit configuration is incomplete or unsafe."""


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise RefitConfigError(f"{context} must be a string-keyed mapping")
    return cast(Mapping[str, object], value)


def _keys(value: Mapping[str, object], *, required: set[str], context: str) -> None:
    missing = sorted(required.difference(value))
    unexpected = sorted(set(value).difference(required))
    if missing or unexpected:
        raise RefitConfigError(
            f"{context} has invalid keys; missing={missing}, unexpected={unexpected}"
        )


def _string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RefitConfigError(f"{context} must be a non-empty string")
    return value


def _integer(value: object, context: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RefitConfigError(f"{context} must be an integer >= {minimum}")
    return value


def _number(value: object, context: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RefitConfigError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise RefitConfigError(f"{context} must be finite and >= {minimum}")
    return result


def _path(value: object, context: str, base_dir: Path) -> Path:
    candidate = Path(_string(value, context))
    return candidate.resolve() if candidate.is_absolute() else (base_dir / candidate).resolve()


def _folds(value: object, context: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise RefitConfigError(f"{context} must be a list of integers")
    return tuple(_integer(item, f"{context} item", minimum=1) for item in value)


@dataclass(frozen=True, slots=True)
class RefitDataConfig:
    manifest_path: Path
    dataset_root: Path
    normalization_path: Path

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object], *, base_dir: Path) -> Self:
        _keys(
            raw,
            required={"manifest", "dataset_root", "normalization"},
            context="data",
        )
        return cls(
            manifest_path=_path(raw["manifest"], "data.manifest", base_dir),
            dataset_root=_path(raw["dataset_root"], "data.dataset_root", base_dir),
            normalization_path=_path(
                raw["normalization"], "data.normalization", base_dir
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "manifest": str(self.manifest_path),
            "dataset_root": str(self.dataset_root),
            "normalization": str(self.normalization_path),
        }


@dataclass(frozen=True, slots=True)
class FrozenSelectionConfig:
    development_checkpoint_path: Path
    selection_metric: Literal["fold8_macro_auroc"]
    frozen_epochs: int

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object], *, base_dir: Path) -> Self:
        _keys(
            raw,
            required={"development_checkpoint", "selection_metric", "frozen_epochs"},
            context="selection",
        )
        selection_metric = _string(raw["selection_metric"], "selection.selection_metric")
        if selection_metric != "fold8_macro_auroc":
            raise RefitConfigError(
                "selection.selection_metric must be 'fold8_macro_auroc'"
            )
        return cls(
            development_checkpoint_path=_path(
                raw["development_checkpoint"],
                "selection.development_checkpoint",
                base_dir,
            ),
            selection_metric="fold8_macro_auroc",
            frozen_epochs=_integer(
                raw["frozen_epochs"], "selection.frozen_epochs", minimum=1
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "development_checkpoint": str(self.development_checkpoint_path),
            "selection_metric": self.selection_metric,
            "frozen_epochs": self.frozen_epochs,
        }


@dataclass(frozen=True, slots=True)
class RefitOptimizationConfig:
    learning_rate: float
    weight_decay: float
    warmup_epochs: int
    minimum_lr_ratio: float
    gradient_clip_norm: float
    scheduler: Literal["warmup_cosine"]

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, object],
        *,
        frozen_epochs: int,
    ) -> Self:
        required = {
            "learning_rate",
            "weight_decay",
            "warmup_epochs",
            "minimum_lr_ratio",
            "gradient_clip_norm",
            "scheduler",
        }
        _keys(raw, required=required, context="optimization")
        scheduler = _string(raw["scheduler"], "optimization.scheduler")
        if scheduler != "warmup_cosine":
            raise RefitConfigError("optimization.scheduler must be 'warmup_cosine'")
        config = cls(
            learning_rate=_number(
                raw["learning_rate"], "optimization.learning_rate"
            ),
            weight_decay=_number(
                raw["weight_decay"], "optimization.weight_decay"
            ),
            warmup_epochs=_integer(
                raw["warmup_epochs"], "optimization.warmup_epochs"
            ),
            minimum_lr_ratio=_number(
                raw["minimum_lr_ratio"], "optimization.minimum_lr_ratio"
            ),
            gradient_clip_norm=_number(
                raw["gradient_clip_norm"], "optimization.gradient_clip_norm"
            ),
            scheduler="warmup_cosine",
        )
        if config.learning_rate <= 0.0:
            raise RefitConfigError("optimization.learning_rate must be positive")
        if config.gradient_clip_norm <= 0.0:
            raise RefitConfigError("optimization.gradient_clip_norm must be positive")
        if config.minimum_lr_ratio > 1.0:
            raise RefitConfigError("optimization.minimum_lr_ratio must be in [0, 1]")
        if config.warmup_epochs >= frozen_epochs:
            raise RefitConfigError(
                "optimization.warmup_epochs must be less than selection.frozen_epochs"
            )
        return config

    def to_dict(self) -> dict[str, object]:
        return {
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "warmup_epochs": self.warmup_epochs,
            "minimum_lr_ratio": self.minimum_lr_ratio,
            "gradient_clip_norm": self.gradient_clip_norm,
            "scheduler": self.scheduler,
        }


@dataclass(frozen=True, slots=True)
class FrozenRefitConfig:
    """Immutable refit recipe bound to a selected development checkpoint."""

    schema_version: int
    run_kind: Literal["frozen_refit"]
    run_name: str
    refit_folds: tuple[int, ...]
    normalization_folds: tuple[int, ...]
    data: RefitDataConfig
    selection: FrozenSelectionConfig
    model: ModelConfig
    loader: LoaderConfig
    optimization: RefitOptimizationConfig
    runtime: RuntimeConfig
    output: OutputConfig

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object], *, base_dir: str | Path) -> Self:
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
        }
        _keys(raw, required=required, context="refit")
        schema_version = _integer(raw["schema_version"], "schema_version", minimum=1)
        if schema_version != REFIT_CONFIG_SCHEMA_VERSION:
            raise RefitConfigError(
                f"schema_version must be {REFIT_CONFIG_SCHEMA_VERSION}"
            )
        run_kind = _string(raw["run_kind"], "run_kind")
        if run_kind != "frozen_refit":
            raise RefitConfigError("run_kind must be 'frozen_refit'")
        run_name = _string(raw["run_name"], "run_name")
        if not _SAFE_NAME.fullmatch(run_name) or run_name in {".", ".."}:
            raise RefitConfigError("run_name must be a safe 1-80 character identifier")
        folds = _mapping(raw["folds"], "folds")
        _keys(folds, required={"refit", "normalization"}, context="folds")
        refit_folds = _folds(folds["refit"], "folds.refit")
        normalization_folds = _folds(folds["normalization"], "folds.normalization")
        if refit_folds != REFIT_FOLDS:
            raise RefitConfigError("folds.refit must be exactly folds 1-8")
        if normalization_folds != TRAIN_FOLDS:
            raise RefitConfigError("folds.normalization must remain exactly folds 1-7")
        resolved_base = Path(base_dir).resolve()
        selection = FrozenSelectionConfig.from_mapping(
            _mapping(raw["selection"], "selection"), base_dir=resolved_base
        )
        return cls(
            schema_version=schema_version,
            run_kind="frozen_refit",
            run_name=run_name,
            refit_folds=refit_folds,
            normalization_folds=normalization_folds,
            data=RefitDataConfig.from_mapping(
                _mapping(raw["data"], "data"), base_dir=resolved_base
            ),
            selection=selection,
            model=ModelConfig.from_mapping(_mapping(raw["model"], "model")),
            loader=LoaderConfig.from_mapping(_mapping(raw["loader"], "loader")),
            optimization=RefitOptimizationConfig.from_mapping(
                _mapping(raw["optimization"], "optimization"),
                frozen_epochs=selection.frozen_epochs,
            ),
            runtime=RuntimeConfig.from_mapping(_mapping(raw["runtime"], "runtime")),
            output=OutputConfig.from_mapping(
                _mapping(raw["output"], "output"), base_dir=resolved_base
            ),
        )

    def to_resolved_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "run_kind": self.run_kind,
            "run_name": self.run_name,
            "folds": {
                "refit": list(self.refit_folds),
                "normalization": list(self.normalization_folds),
            },
            "data": self.data.to_dict(),
            "selection": self.selection.to_dict(),
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


def load_refit_config(
    path: str | Path,
    *,
    base_dir: str | Path | None = None,
) -> FrozenRefitConfig:
    """Load a strict frozen-refit YAML configuration."""

    config_path = Path(path)
    try:
        decoded: object = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise RefitConfigError(f"could not read refit config {config_path}: {error}") from error
    except yaml.YAMLError as error:
        raise RefitConfigError(f"invalid YAML in {config_path}: {error}") from error
    resolved_base = Path(base_dir) if base_dir is not None else config_path.parent
    return FrozenRefitConfig.from_mapping(
        _mapping(decoded, str(config_path)), base_dir=resolved_base
    )
