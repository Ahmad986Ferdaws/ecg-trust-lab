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
POST_SWEEP_REFIT_CONFIG_SCHEMA_VERSION = 2
REFIT_FOLDS: tuple[int, ...] = TRAIN_FOLDS + MODEL_SELECTION_FOLDS
_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}\Z")
_SHA256_PATTERN = re.compile(r"(?:sha256:)?([0-9a-fA-F]{64})\Z")
_POST_SWEEP_RUN_KIND = "post_sweep_frozen_refit"
_POST_SWEEP_OBJECTIVE = "fold8_uncalibrated_macro_roc_auc"
_EPOCH_BUDGET_RULE = (
    "max(warmup_epochs+1,median(selected_zero_based_best_epoch+1_across_seeds))"
)


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


def _sha256(value: object, context: str) -> str:
    text = _string(value, context)
    match = _SHA256_PATTERN.fullmatch(text)
    if match is None:
        raise RefitConfigError(f"{context} must be a SHA-256 digest")
    return "sha256:" + match.group(1).lower()


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


@dataclass(frozen=True, slots=True)
class FrozenConfirmationSource:
    """Exact completed development evidence named by a post-sweep recipe."""

    member_completion_path: Path
    member_completion_sha256: str
    manifest_sha256: str
    normalization_sha256: str
    run_metadata_path: Path
    run_metadata_sha256: str
    resolved_config_path: Path
    resolved_config_file_sha256: str
    resolved_config_hash: str
    history_path: Path
    history_sha256: str
    best_checkpoint_path: Path
    best_checkpoint_sha256: str
    prediction_path: Path
    prediction_npz_sha256: str
    prediction_json_path: Path
    prediction_artifact_sha256: str
    best_epoch: int
    best_validation_macro_auroc: float

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object], *, base_dir: Path) -> Self:
        required = {
            "member_completion",
            "member_completion_sha256",
            "manifest_sha256",
            "normalization_sha256",
            "run_metadata",
            "run_metadata_sha256",
            "resolved_config",
            "resolved_config_file_sha256",
            "resolved_config_hash",
            "history",
            "history_sha256",
            "best_checkpoint",
            "best_checkpoint_sha256",
            "prediction",
            "prediction_npz_sha256",
            "prediction_json",
            "prediction_artifact_sha256",
            "best_epoch",
            "best_validation_macro_auroc",
        }
        _keys(raw, required=required, context="source")
        best_epoch = _integer(raw["best_epoch"], "source.best_epoch")
        if best_epoch >= 30:
            raise RefitConfigError("source.best_epoch must be below the 30-epoch ceiling")
        score = _number(
            raw["best_validation_macro_auroc"],
            "source.best_validation_macro_auroc",
        )
        if score > 1.0:
            raise RefitConfigError("source.best_validation_macro_auroc must be in [0, 1]")
        return cls(
            member_completion_path=_path(
                raw["member_completion"], "source.member_completion", base_dir
            ),
            member_completion_sha256=_sha256(
                raw["member_completion_sha256"], "source.member_completion_sha256"
            ),
            manifest_sha256=_sha256(
                raw["manifest_sha256"], "source.manifest_sha256"
            ),
            normalization_sha256=_sha256(
                raw["normalization_sha256"], "source.normalization_sha256"
            ),
            run_metadata_path=_path(raw["run_metadata"], "source.run_metadata", base_dir),
            run_metadata_sha256=_sha256(
                raw["run_metadata_sha256"], "source.run_metadata_sha256"
            ),
            resolved_config_path=_path(
                raw["resolved_config"], "source.resolved_config", base_dir
            ),
            resolved_config_file_sha256=_sha256(
                raw["resolved_config_file_sha256"],
                "source.resolved_config_file_sha256",
            ),
            resolved_config_hash=_sha256(
                raw["resolved_config_hash"], "source.resolved_config_hash"
            ),
            history_path=_path(raw["history"], "source.history", base_dir),
            history_sha256=_sha256(raw["history_sha256"], "source.history_sha256"),
            best_checkpoint_path=_path(
                raw["best_checkpoint"], "source.best_checkpoint", base_dir
            ),
            best_checkpoint_sha256=_sha256(
                raw["best_checkpoint_sha256"], "source.best_checkpoint_sha256"
            ),
            prediction_path=_path(raw["prediction"], "source.prediction", base_dir),
            prediction_npz_sha256=_sha256(
                raw["prediction_npz_sha256"], "source.prediction_npz_sha256"
            ),
            prediction_json_path=_path(
                raw["prediction_json"], "source.prediction_json", base_dir
            ),
            prediction_artifact_sha256=_sha256(
                raw["prediction_artifact_sha256"],
                "source.prediction_artifact_sha256",
            ),
            best_epoch=best_epoch,
            best_validation_macro_auroc=score,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "member_completion": str(self.member_completion_path),
            "member_completion_sha256": self.member_completion_sha256,
            "manifest_sha256": self.manifest_sha256,
            "normalization_sha256": self.normalization_sha256,
            "run_metadata": str(self.run_metadata_path),
            "run_metadata_sha256": self.run_metadata_sha256,
            "resolved_config": str(self.resolved_config_path),
            "resolved_config_file_sha256": self.resolved_config_file_sha256,
            "resolved_config_hash": self.resolved_config_hash,
            "history": str(self.history_path),
            "history_sha256": self.history_sha256,
            "best_checkpoint": str(self.best_checkpoint_path),
            "best_checkpoint_sha256": self.best_checkpoint_sha256,
            "prediction": str(self.prediction_path),
            "prediction_npz_sha256": self.prediction_npz_sha256,
            "prediction_json": str(self.prediction_json_path),
            "prediction_artifact_sha256": self.prediction_artifact_sha256,
            "best_epoch": self.best_epoch,
            "best_validation_macro_auroc": self.best_validation_macro_auroc,
        }


@dataclass(frozen=True, slots=True)
class FrozenOptimizerPolicy:
    """Complete AdamW identity inherited from the selected development run."""

    name: Literal["AdamW"]
    betas: tuple[float, float]
    eps: float

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> Self:
        _keys(raw, required={"name", "betas", "eps"}, context="optimizer")
        if raw["name"] != "AdamW":
            raise RefitConfigError("optimizer.name must remain AdamW")
        raw_betas = raw["betas"]
        if not isinstance(raw_betas, list) or len(raw_betas) != 2:
            raise RefitConfigError("optimizer.betas must contain exactly two values")
        beta1 = _number(raw_betas[0], "optimizer.betas[0]")
        beta2 = _number(raw_betas[1], "optimizer.betas[1]")
        if beta1 >= 1.0 or beta2 >= 1.0:
            raise RefitConfigError("optimizer betas must be in [0, 1)")
        eps = _number(raw["eps"], "optimizer.eps")
        if eps <= 0.0:
            raise RefitConfigError("optimizer.eps must be positive")
        return cls(name="AdamW", betas=(beta1, beta2), eps=eps)

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "betas": list(self.betas), "eps": self.eps}


@dataclass(frozen=True, slots=True)
class FrozenDownstreamProvenance:
    """Code and dependency state authorized for downstream refitting."""

    project_root: Path
    code_revision: str
    dependency_lock_sha256: str

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object], *, base_dir: Path) -> Self:
        _keys(
            raw,
            required={"project_root", "code_revision", "dependency_lock_sha256"},
            context="downstream_provenance",
        )
        return cls(
            project_root=_path(
                raw["project_root"], "downstream_provenance.project_root", base_dir
            ),
            code_revision=_string(
                raw["code_revision"], "downstream_provenance.code_revision"
            ),
            dependency_lock_sha256=_sha256(
                raw["dependency_lock_sha256"],
                "downstream_provenance.dependency_lock_sha256",
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "project_root": str(self.project_root),
            "code_revision": self.code_revision,
            "dependency_lock_sha256": self.dependency_lock_sha256,
        }


@dataclass(frozen=True, slots=True)
class FrozenArchitectureSelection:
    objective: Literal["fold8_uncalibrated_macro_roc_auc"]
    architecture_mean_macro_auroc: float
    frozen_epochs: int
    epoch_budget_rule: str

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> Self:
        _keys(
            raw,
            required={
                "objective",
                "architecture_mean_macro_auroc",
                "frozen_epochs",
                "epoch_budget_rule",
            },
            context="selection",
        )
        if raw["objective"] != _POST_SWEEP_OBJECTIVE:
            raise RefitConfigError("selection.objective is not the frozen fold-8 objective")
        if raw["epoch_budget_rule"] != _EPOCH_BUDGET_RULE:
            raise RefitConfigError("selection.epoch_budget_rule is not the frozen median rule")
        score = _number(
            raw["architecture_mean_macro_auroc"],
            "selection.architecture_mean_macro_auroc",
        )
        if score > 1.0:
            raise RefitConfigError("selection architecture mean must be in [0, 1]")
        epochs = _integer(raw["frozen_epochs"], "selection.frozen_epochs", minimum=1)
        if epochs > 30:
            raise RefitConfigError("selection.frozen_epochs must not exceed 30")
        return cls(
            objective="fold8_uncalibrated_macro_roc_auc",
            architecture_mean_macro_auroc=score,
            frozen_epochs=epochs,
            epoch_budget_rule=_EPOCH_BUDGET_RULE,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "objective": self.objective,
            "architecture_mean_macro_auroc": self.architecture_mean_macro_auroc,
            "frozen_epochs": self.frozen_epochs,
            "epoch_budget_rule": self.epoch_budget_rule,
        }


@dataclass(frozen=True, slots=True)
class PostSweepRefitConfig:
    """Version-2 recipe bound byte-for-byte to a multi-seed freeze."""

    schema_version: int
    run_kind: Literal["post_sweep_frozen_refit"]
    freeze_artifact_path: Path
    freeze_artifact_sha256: str
    recipe_sha256: str
    comparison_id: str
    architecture: Literal["resnet1d", "ecg_transformer"]
    confirmation_seed: int
    run_name: str
    initialization: Literal["fresh"]
    refit_folds: tuple[int, ...]
    normalization_folds: tuple[int, ...]
    data: RefitDataConfig
    source: FrozenConfirmationSource
    selection: FrozenArchitectureSelection
    model: ModelConfig
    model_identity: Mapping[str, object]
    loader: LoaderConfig
    optimization: RefitOptimizationConfig
    optimizer: FrozenOptimizerPolicy
    runtime: RuntimeConfig
    output: OutputConfig
    downstream_provenance: FrozenDownstreamProvenance

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object], *, base_dir: str | Path) -> Self:
        required = {
            "schema_version",
            "run_kind",
            "freeze_artifact",
            "freeze_artifact_sha256",
            "recipe_sha256",
            "comparison_id",
            "architecture",
            "confirmation_seed",
            "run_name",
            "initialization",
            "folds",
            "data",
            "source",
            "selection",
            "model",
            "model_identity",
            "loader",
            "optimization",
            "optimizer",
            "runtime",
            "output",
            "downstream_provenance",
        }
        _keys(raw, required=required, context="post-sweep refit")
        if raw["schema_version"] != POST_SWEEP_REFIT_CONFIG_SCHEMA_VERSION:
            raise RefitConfigError("post-sweep refit schema_version must be 2")
        if raw["run_kind"] != _POST_SWEEP_RUN_KIND:
            raise RefitConfigError("post-sweep refit run_kind is invalid")
        if raw["initialization"] != "fresh":
            raise RefitConfigError("post-sweep refit initialization must be fresh")
        resolved_base = Path(base_dir).resolve()
        run_name = _string(raw["run_name"], "run_name")
        if not _SAFE_NAME.fullmatch(run_name) or run_name in {".", ".."}:
            raise RefitConfigError("run_name must be a safe 1-80 character identifier")
        architecture = _string(raw["architecture"], "architecture")
        if architecture not in {"resnet1d", "ecg_transformer"}:
            raise RefitConfigError("architecture must be resnet1d or ecg_transformer")
        folds = _mapping(raw["folds"], "folds")
        _keys(folds, required={"refit", "normalization"}, context="folds")
        refit_folds = _folds(folds["refit"], "folds.refit")
        normalization_folds = _folds(folds["normalization"], "folds.normalization")
        if refit_folds != REFIT_FOLDS:
            raise RefitConfigError("folds.refit must be exactly folds 1-8")
        if normalization_folds != TRAIN_FOLDS:
            raise RefitConfigError("folds.normalization must remain exactly folds 1-7")
        seed = _integer(raw["confirmation_seed"], "confirmation_seed")
        selection = FrozenArchitectureSelection.from_mapping(
            _mapping(raw["selection"], "selection")
        )
        model = ModelConfig.from_mapping(_mapping(raw["model"], "model"))
        model_identity = _mapping(raw["model_identity"], "model_identity")
        required_identity = {
            "architecture",
            "preset",
            "class",
            "trainable_parameters",
            "resolved_architecture_config",
        }
        allowed_identity = required_identity | {"capacity_match"}
        missing_identity = sorted(required_identity.difference(model_identity))
        unexpected_identity = sorted(set(model_identity).difference(allowed_identity))
        if missing_identity or unexpected_identity:
            raise RefitConfigError(
                "model_identity has invalid keys; "
                f"missing={missing_identity}, unexpected={unexpected_identity}"
            )
        if model_identity.get("architecture") != architecture:
            raise RefitConfigError("model_identity architecture differs from recipe")
        if model_identity.get("preset") != model.preset:
            raise RefitConfigError("model_identity preset differs from recipe")
        _string(model_identity.get("class"), "model_identity.class")
        _integer(
            model_identity.get("trainable_parameters"),
            "model_identity.trainable_parameters",
            minimum=1,
        )
        _mapping(
            model_identity.get("resolved_architecture_config"),
            "model_identity.resolved_architecture_config",
        )
        if model.preset == "matched_capacity":
            _mapping(model_identity.get("capacity_match"), "model_identity.capacity_match")
        elif "capacity_match" in model_identity:
            raise RefitConfigError(
                "smoke model_identity must not contain matched-capacity metadata"
            )
        runtime = RuntimeConfig.from_mapping(_mapping(raw["runtime"], "runtime"))
        if model.architecture != architecture:
            raise RefitConfigError("model architecture differs from recipe architecture")
        if runtime.seed != seed:
            raise RefitConfigError("runtime seed differs from confirmation_seed")
        return cls(
            schema_version=2,
            run_kind="post_sweep_frozen_refit",
            freeze_artifact_path=_path(
                raw["freeze_artifact"], "freeze_artifact", resolved_base
            ),
            freeze_artifact_sha256=_sha256(
                raw["freeze_artifact_sha256"], "freeze_artifact_sha256"
            ),
            recipe_sha256=_sha256(raw["recipe_sha256"], "recipe_sha256"),
            comparison_id=_string(raw["comparison_id"], "comparison_id"),
            architecture=architecture,
            confirmation_seed=seed,
            run_name=run_name,
            initialization="fresh",
            refit_folds=refit_folds,
            normalization_folds=normalization_folds,
            data=RefitDataConfig.from_mapping(
                _mapping(raw["data"], "data"), base_dir=resolved_base
            ),
            source=FrozenConfirmationSource.from_mapping(
                _mapping(raw["source"], "source"), base_dir=resolved_base
            ),
            selection=selection,
            model=model,
            model_identity=dict(model_identity),
            loader=LoaderConfig.from_mapping(_mapping(raw["loader"], "loader")),
            optimization=RefitOptimizationConfig.from_mapping(
                _mapping(raw["optimization"], "optimization"),
                frozen_epochs=selection.frozen_epochs,
            ),
            optimizer=FrozenOptimizerPolicy.from_mapping(
                _mapping(raw["optimizer"], "optimizer")
            ),
            runtime=runtime,
            output=OutputConfig.from_mapping(
                _mapping(raw["output"], "output"), base_dir=resolved_base
            ),
            downstream_provenance=FrozenDownstreamProvenance.from_mapping(
                _mapping(raw["downstream_provenance"], "downstream_provenance"),
                base_dir=resolved_base,
            ),
        )

    def to_resolved_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "run_kind": self.run_kind,
            "freeze_artifact": str(self.freeze_artifact_path),
            "freeze_artifact_sha256": self.freeze_artifact_sha256,
            "recipe_sha256": self.recipe_sha256,
            "comparison_id": self.comparison_id,
            "architecture": self.architecture,
            "confirmation_seed": self.confirmation_seed,
            "run_name": self.run_name,
            "initialization": self.initialization,
            "folds": {
                "refit": list(self.refit_folds),
                "normalization": list(self.normalization_folds),
            },
            "data": self.data.to_dict(),
            "source": self.source.to_dict(),
            "selection": self.selection.to_dict(),
            "model": self.model.to_dict(),
            "model_identity": dict(self.model_identity),
            "loader": self.loader.to_dict(),
            "optimization": self.optimization.to_dict(),
            "optimizer": self.optimizer.to_dict(),
            "runtime": self.runtime.to_dict(),
            "output": self.output.to_dict(),
            "downstream_provenance": self.downstream_provenance.to_dict(),
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


RefitConfig = FrozenRefitConfig | PostSweepRefitConfig


def load_refit_config(
    path: str | Path,
    *,
    base_dir: str | Path | None = None,
) -> RefitConfig:
    """Load a strict frozen-refit YAML configuration."""

    config_path = Path(path)
    try:
        serialized = config_path.read_text(encoding="utf-8")
        decoded: object = (
            json.loads(serialized)
            if config_path.suffix.casefold() == ".json"
            else yaml.safe_load(serialized)
        )
    except OSError as error:
        raise RefitConfigError(f"could not read refit config {config_path}: {error}") from error
    except json.JSONDecodeError as error:
        raise RefitConfigError(f"invalid JSON in {config_path}: {error}") from error
    except yaml.YAMLError as error:
        raise RefitConfigError(f"invalid YAML in {config_path}: {error}") from error
    resolved_base = Path(base_dir) if base_dir is not None else config_path.parent
    mapping = _mapping(decoded, str(config_path))
    schema_version = mapping.get("schema_version")
    if schema_version == REFIT_CONFIG_SCHEMA_VERSION:
        return FrozenRefitConfig.from_mapping(mapping, base_dir=resolved_base)
    if schema_version == POST_SWEEP_REFIT_CONFIG_SCHEMA_VERSION:
        return PostSweepRefitConfig.from_mapping(mapping, base_dir=resolved_base)
    raise RefitConfigError(
        "refit schema_version must be 1 (legacy) or 2 (post-sweep freeze-bound)"
    )
