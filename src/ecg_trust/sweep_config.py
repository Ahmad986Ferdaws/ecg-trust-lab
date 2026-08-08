"""Strict configuration for paired, candidate-matched development sweeps."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self, cast

import yaml  # type: ignore[import-untyped]

from ecg_trust.experiment_config import DevelopmentExperimentConfig, load_experiment_config
from ecg_trust.protocol import MODEL_SELECTION_FOLDS, TRAIN_FOLDS

SWEEP_CONFIG_SCHEMA_VERSION = 2
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}\Z")
_TIE_BREAK_ORDER = (
    "objective_desc",
    "completed_epochs_asc",
    "candidate_index_asc",
    "trial_number_asc",
)


class SweepConfigError(ValueError):
    """Raised when a sweep can violate the paired-selection protocol."""


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise SweepConfigError(f"{context} must be a string-keyed mapping")
    return cast(Mapping[str, object], value)


def _keys(value: Mapping[str, object], *, required: set[str], context: str) -> None:
    missing = sorted(required.difference(value))
    unexpected = sorted(set(value).difference(required))
    if missing or unexpected:
        raise SweepConfigError(
            f"{context} has invalid keys; missing={missing}, unexpected={unexpected}"
        )


def _string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SweepConfigError(f"{context} must be a non-empty string")
    return value


def _safe_id(value: object, context: str) -> str:
    parsed = _string(value, context)
    if not _SAFE_ID.fullmatch(parsed) or parsed in {".", ".."}:
        raise SweepConfigError(f"{context} must be a safe 1-80 character identifier")
    return parsed


def _integer(value: object, context: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise SweepConfigError(f"{context} must be an integer >= {minimum}")
    return value


def _number(value: object, context: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SweepConfigError(f"{context} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < minimum:
        raise SweepConfigError(f"{context} must be finite and >= {minimum}")
    return parsed


def _boolean(value: object, context: str) -> bool:
    if not isinstance(value, bool):
        raise SweepConfigError(f"{context} must be boolean")
    return value


def _path(value: object, context: str, base_dir: Path) -> Path:
    candidate = Path(_string(value, context))
    return candidate.resolve() if candidate.is_absolute() else (base_dir / candidate).resolve()


def _sequence(value: object, context: str) -> Sequence[object]:
    if not isinstance(value, list) or not value:
        raise SweepConfigError(f"{context} must be a non-empty list")
    return cast(Sequence[object], value)


def _literal_tuple(value: object, context: str) -> tuple[str, ...]:
    sequence = _sequence(value, context)
    if not all(isinstance(item, str) and item for item in sequence):
        raise SweepConfigError(f"{context} must contain non-empty strings")
    return tuple(cast(Sequence[str], sequence))


@dataclass(frozen=True, slots=True)
class FloatRange:
    low: float
    high: float
    log: bool

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, object],
        *,
        context: str,
        positive: bool,
    ) -> Self:
        _keys(raw, required={"low", "high", "log"}, context=context)
        low = _number(raw["low"], f"{context}.low")
        high = _number(raw["high"], f"{context}.high")
        log = _boolean(raw["log"], f"{context}.log")
        if high <= low:
            raise SweepConfigError(f"{context}.high must be greater than low")
        if positive and low <= 0.0:
            raise SweepConfigError(f"{context}.low must be positive")
        if log and low <= 0.0:
            raise SweepConfigError(f"{context}.low must be positive for log sampling")
        return cls(low=low, high=high, log=log)

    def to_dict(self) -> dict[str, object]:
        return {"low": self.low, "high": self.high, "log": self.log}


def _int_choices(value: object, context: str, *, minimum: int) -> tuple[int, ...]:
    choices = tuple(
        _integer(item, f"{context} item", minimum=minimum)
        for item in _sequence(value, context)
    )
    if len(set(choices)) != len(choices):
        raise SweepConfigError(f"{context} must not contain duplicates")
    return choices


def _float_choices(
    value: object,
    context: str,
    *,
    positive: bool,
    maximum: float | None = None,
) -> tuple[float, ...]:
    choices = tuple(_number(item, f"{context} item") for item in _sequence(value, context))
    if positive and any(item <= 0.0 for item in choices):
        raise SweepConfigError(f"{context} values must be positive")
    if maximum is not None and any(item > maximum for item in choices):
        raise SweepConfigError(f"{context} values must be <= {maximum}")
    if len(set(choices)) != len(choices):
        raise SweepConfigError(f"{context} must not contain duplicates")
    return choices


@dataclass(frozen=True, slots=True)
class SweepBudget:
    complete_candidates: int
    max_epochs: int

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> Self:
        _keys(raw, required={"complete_candidates", "max_epochs"}, context="budget")
        config = cls(
            complete_candidates=_integer(
                raw["complete_candidates"], "budget.complete_candidates", minimum=1
            ),
            max_epochs=_integer(raw["max_epochs"], "budget.max_epochs", minimum=2),
        )
        if config.complete_candidates != 12:
            raise SweepConfigError("budget.complete_candidates must be exactly 12")
        if config.max_epochs != 30:
            raise SweepConfigError("budget.max_epochs must be exactly 30")
        return config

    def to_dict(self) -> dict[str, int]:
        return {
            "complete_candidates": self.complete_candidates,
            "max_epochs": self.max_epochs,
        }


@dataclass(frozen=True, slots=True)
class CandidateDesign:
    algorithm: Literal["scipy_qmc_latin_hypercube"]
    version: int
    seed: int

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> Self:
        _keys(raw, required={"algorithm", "version", "seed"}, context="candidate_design")
        algorithm = _string(raw["algorithm"], "candidate_design.algorithm")
        if algorithm != "scipy_qmc_latin_hypercube":
            raise SweepConfigError(
                "candidate_design.algorithm must be 'scipy_qmc_latin_hypercube'"
            )
        version = _integer(raw["version"], "candidate_design.version", minimum=1)
        if version != 1:
            raise SweepConfigError("candidate_design.version must be 1")
        seed = _integer(raw["seed"], "candidate_design.seed")
        if seed >= 2**32:
            raise SweepConfigError("candidate_design.seed must be smaller than 2**32")
        return cls(algorithm="scipy_qmc_latin_hypercube", version=version, seed=seed)

    def to_dict(self) -> dict[str, object]:
        return {"algorithm": self.algorithm, "version": self.version, "seed": self.seed}


@dataclass(frozen=True, slots=True)
class ObjectiveConfig:
    name: Literal["fold8_uncalibrated_macro_roc_auc"]
    direction: Literal["maximize"]
    required_label_count: int
    require_all_labels_defined: bool
    pruning: Literal["none"]

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> Self:
        required = {
            "name",
            "direction",
            "required_label_count",
            "require_all_labels_defined",
            "pruning",
        }
        _keys(raw, required=required, context="objective")
        name = _string(raw["name"], "objective.name")
        direction = _string(raw["direction"], "objective.direction")
        pruning = _string(raw["pruning"], "objective.pruning")
        if name != "fold8_uncalibrated_macro_roc_auc":
            raise SweepConfigError(
                "objective.name must be 'fold8_uncalibrated_macro_roc_auc'"
            )
        if direction != "maximize":
            raise SweepConfigError("objective.direction must be 'maximize'")
        if pruning != "none":
            raise SweepConfigError("objective.pruning must be 'none'")
        label_count = _integer(
            raw["required_label_count"], "objective.required_label_count", minimum=1
        )
        if label_count != 5:
            raise SweepConfigError("objective.required_label_count must be 5")
        require_defined = _boolean(
            raw["require_all_labels_defined"], "objective.require_all_labels_defined"
        )
        if not require_defined:
            raise SweepConfigError("objective must require all five labels to be defined")
        return cls(
            name="fold8_uncalibrated_macro_roc_auc",
            direction="maximize",
            required_label_count=label_count,
            require_all_labels_defined=True,
            pruning="none",
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "direction": self.direction,
            "required_label_count": self.required_label_count,
            "require_all_labels_defined": self.require_all_labels_defined,
            "pruning": self.pruning,
        }


@dataclass(frozen=True, slots=True)
class SeedPolicy:
    kind: Literal["fixed_across_candidates"]
    experiment_seed: int

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> Self:
        _keys(raw, required={"kind", "experiment_seed"}, context="seed_policy")
        kind = _string(raw["kind"], "seed_policy.kind")
        if kind != "fixed_across_candidates":
            raise SweepConfigError("seed_policy.kind must be 'fixed_across_candidates'")
        seed = _integer(raw["experiment_seed"], "seed_policy.experiment_seed")
        if seed >= 2**32:
            raise SweepConfigError("seed_policy.experiment_seed must be smaller than 2**32")
        return cls(kind="fixed_across_candidates", experiment_seed=seed)

    def to_dict(self) -> dict[str, object]:
        return {"kind": self.kind, "experiment_seed": self.experiment_seed}


@dataclass(frozen=True, slots=True)
class TieBreakPolicy:
    order: tuple[str, ...]

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> Self:
        _keys(raw, required={"order"}, context="tie_break")
        order = _literal_tuple(raw["order"], "tie_break.order")
        if order != _TIE_BREAK_ORDER:
            raise SweepConfigError(f"tie_break.order must be exactly {list(_TIE_BREAK_ORDER)!r}")
        return cls(order=order)

    def to_dict(self) -> dict[str, object]:
        return {"order": list(self.order)}


@dataclass(frozen=True, slots=True)
class FailurePolicy:
    retry_same_candidate: bool
    failed_attempts_consume_budget: bool
    interrupted_attempts_mark_failed: bool
    max_attempts_per_candidate: int

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> Self:
        required = {
            "retry_same_candidate",
            "failed_attempts_consume_budget",
            "interrupted_attempts_mark_failed",
            "max_attempts_per_candidate",
        }
        _keys(raw, required=required, context="failure_policy")
        config = cls(
            retry_same_candidate=_boolean(
                raw["retry_same_candidate"], "failure_policy.retry_same_candidate"
            ),
            failed_attempts_consume_budget=_boolean(
                raw["failed_attempts_consume_budget"],
                "failure_policy.failed_attempts_consume_budget",
            ),
            interrupted_attempts_mark_failed=_boolean(
                raw["interrupted_attempts_mark_failed"],
                "failure_policy.interrupted_attempts_mark_failed",
            ),
            max_attempts_per_candidate=_integer(
                raw["max_attempts_per_candidate"],
                "failure_policy.max_attempts_per_candidate",
                minimum=1,
            ),
        )
        if not config.retry_same_candidate:
            raise SweepConfigError("failure_policy must retry the same candidate")
        if config.failed_attempts_consume_budget:
            raise SweepConfigError("failed attempts must not consume candidate budget")
        if not config.interrupted_attempts_mark_failed:
            raise SweepConfigError("interrupted attempts must be marked failed on resume")
        return config

    def to_dict(self) -> dict[str, object]:
        return {
            "retry_same_candidate": self.retry_same_candidate,
            "failed_attempts_consume_budget": self.failed_attempts_consume_budget,
            "interrupted_attempts_mark_failed": self.interrupted_attempts_mark_failed,
            "max_attempts_per_candidate": self.max_attempts_per_candidate,
        }


@dataclass(frozen=True, slots=True)
class SweepStorage:
    sqlite_path: Path
    output_root: Path

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object], *, base_dir: Path) -> Self:
        _keys(raw, required={"sqlite", "output_root"}, context="storage")
        sqlite_path = _path(raw["sqlite"], "storage.sqlite", base_dir)
        if sqlite_path.suffix.casefold() not in {".sqlite", ".sqlite3", ".db"}:
            raise SweepConfigError("storage.sqlite must be a SQLite file path")
        return cls(
            sqlite_path=sqlite_path,
            output_root=_path(raw["output_root"], "storage.output_root", base_dir),
        )

    def to_dict(self) -> dict[str, str]:
        return {"sqlite": str(self.sqlite_path), "output_root": str(self.output_root)}


@dataclass(frozen=True, slots=True)
class SweepSearchSpace:
    learning_rate: FloatRange
    weight_decay: FloatRange
    batch_size: tuple[int, ...]
    gradient_clip_norm: tuple[float, ...]
    warmup_epochs: tuple[int, ...]
    minimum_lr_ratio: tuple[float, ...]

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object], *, max_epochs: int) -> Self:
        required = {
            "learning_rate",
            "weight_decay",
            "batch_size",
            "gradient_clip_norm",
            "warmup_epochs",
            "minimum_lr_ratio",
        }
        _keys(raw, required=required, context="search_space")
        config = cls(
            learning_rate=FloatRange.from_mapping(
                _mapping(raw["learning_rate"], "search_space.learning_rate"),
                context="search_space.learning_rate",
                positive=True,
            ),
            weight_decay=FloatRange.from_mapping(
                _mapping(raw["weight_decay"], "search_space.weight_decay"),
                context="search_space.weight_decay",
                positive=False,
            ),
            batch_size=_int_choices(raw["batch_size"], "search_space.batch_size", minimum=1),
            gradient_clip_norm=_float_choices(
                raw["gradient_clip_norm"],
                "search_space.gradient_clip_norm",
                positive=True,
            ),
            warmup_epochs=_int_choices(
                raw["warmup_epochs"], "search_space.warmup_epochs", minimum=0
            ),
            minimum_lr_ratio=_float_choices(
                raw["minimum_lr_ratio"],
                "search_space.minimum_lr_ratio",
                positive=False,
                maximum=1.0,
            ),
        )
        if any(value >= max_epochs for value in config.warmup_epochs):
            raise SweepConfigError("warmup choices must be below budget.max_epochs")
        categorical = (
            config.batch_size,
            config.gradient_clip_norm,
            config.warmup_epochs,
            config.minimum_lr_ratio,
        )
        if any(len(choices) != 3 for choices in categorical):
            raise SweepConfigError("each categorical search dimension must have 3 choices")
        return config

    def to_dict(self) -> dict[str, object]:
        return {
            "learning_rate": self.learning_rate.to_dict(),
            "weight_decay": self.weight_decay.to_dict(),
            "batch_size": list(self.batch_size),
            "gradient_clip_norm": list(self.gradient_clip_norm),
            "warmup_epochs": list(self.warmup_epochs),
            "minimum_lr_ratio": list(self.minimum_lr_ratio),
        }


@dataclass(frozen=True, slots=True)
class SweepConfig:
    schema_version: int
    comparison_id: str
    study_name: str
    architecture: Literal["resnet1d", "ecg_transformer"]
    base_experiment_path: Path
    base_experiment: DevelopmentExperimentConfig
    budget: SweepBudget
    candidate_design: CandidateDesign
    objective: ObjectiveConfig
    seed_policy: SeedPolicy
    tie_break: TieBreakPolicy
    failure_policy: FailurePolicy
    storage: SweepStorage
    search_space: SweepSearchSpace

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object], *, base_dir: str | Path) -> Self:
        required = {
            "schema_version",
            "comparison_id",
            "study_name",
            "architecture",
            "base_experiment",
            "budget",
            "candidate_design",
            "objective",
            "seed_policy",
            "tie_break",
            "failure_policy",
            "storage",
            "search_space",
        }
        _keys(raw, required=required, context="sweep")
        schema_version = _integer(raw["schema_version"], "schema_version", minimum=1)
        if schema_version != SWEEP_CONFIG_SCHEMA_VERSION:
            raise SweepConfigError(f"schema_version must be {SWEEP_CONFIG_SCHEMA_VERSION}")
        architecture = _string(raw["architecture"], "architecture")
        if architecture not in {"resnet1d", "ecg_transformer"}:
            raise SweepConfigError("architecture must be resnet1d or ecg_transformer")
        resolved_base = Path(base_dir).resolve()
        base_path = _path(raw["base_experiment"], "base_experiment", resolved_base)
        base_experiment = load_experiment_config(base_path, base_dir=resolved_base)
        if base_experiment.model.architecture != architecture:
            raise SweepConfigError("base experiment architecture does not match sweep")
        if base_experiment.model.preset != "matched_capacity":
            raise SweepConfigError("sweeps require the fixed matched_capacity preset")
        if (
            base_experiment.train_folds != TRAIN_FOLDS
            or base_experiment.validation_folds != MODEL_SELECTION_FOLDS
        ):
            raise SweepConfigError("sweeps may use only folds 1-7 and fold 8")
        budget = SweepBudget.from_mapping(_mapping(raw["budget"], "budget"))
        return cls(
            schema_version=schema_version,
            comparison_id=_safe_id(raw["comparison_id"], "comparison_id"),
            study_name=_safe_id(raw["study_name"], "study_name"),
            architecture=architecture,
            base_experiment_path=base_path,
            base_experiment=base_experiment,
            budget=budget,
            candidate_design=CandidateDesign.from_mapping(
                _mapping(raw["candidate_design"], "candidate_design")
            ),
            objective=ObjectiveConfig.from_mapping(_mapping(raw["objective"], "objective")),
            seed_policy=SeedPolicy.from_mapping(
                _mapping(raw["seed_policy"], "seed_policy")
            ),
            tie_break=TieBreakPolicy.from_mapping(_mapping(raw["tie_break"], "tie_break")),
            failure_policy=FailurePolicy.from_mapping(
                _mapping(raw["failure_policy"], "failure_policy")
            ),
            storage=SweepStorage.from_mapping(
                _mapping(raw["storage"], "storage"), base_dir=resolved_base
            ),
            search_space=SweepSearchSpace.from_mapping(
                _mapping(raw["search_space"], "search_space"),
                max_epochs=budget.max_epochs,
            ),
        )

    def to_resolved_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "comparison_id": self.comparison_id,
            "study_name": self.study_name,
            "architecture": self.architecture,
            "base_experiment": str(self.base_experiment_path),
            "base_experiment_hash": self.base_experiment.config_hash,
            "budget": self.budget.to_dict(),
            "candidate_design": self.candidate_design.to_dict(),
            "objective": self.objective.to_dict(),
            "seed_policy": self.seed_policy.to_dict(),
            "tie_break": self.tie_break.to_dict(),
            "failure_policy": self.failure_policy.to_dict(),
            "storage": self.storage.to_dict(),
            "search_space": self.search_space.to_dict(),
        }

    @property
    def config_hash(self) -> str:
        serialized = json.dumps(
            self.to_resolved_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        return "sha256:" + hashlib.sha256(serialized.encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class EqualBudgetSweepPair:
    resnet: SweepConfig
    transformer: SweepConfig

    @classmethod
    def create(cls, configs: Sequence[SweepConfig]) -> Self:
        if len(configs) != 2:
            raise SweepConfigError("comparison requires exactly two sweep configs")
        by_architecture = {config.architecture: config for config in configs}
        if set(by_architecture) != {"resnet1d", "ecg_transformer"}:
            raise SweepConfigError("comparison requires one config per architecture")
        pair = cls(
            resnet=by_architecture["resnet1d"],
            transformer=by_architecture["ecg_transformer"],
        )
        pair._validate_equal_contract()
        return pair

    @property
    def configs(self) -> tuple[SweepConfig, SweepConfig]:
        return (self.resnet, self.transformer)

    def _validate_equal_contract(self) -> None:
        left, right = self.configs
        if left.comparison_id != right.comparison_id:
            raise SweepConfigError("paired sweeps must share comparison_id")
        if left.study_name == right.study_name:
            raise SweepConfigError("paired sweeps must have distinct study names")
        equal_fields = {
            "budget": (left.budget, right.budget),
            "candidate design": (left.candidate_design, right.candidate_design),
            "objective": (left.objective, right.objective),
            "seed policy": (left.seed_policy, right.seed_policy),
            "tie break": (left.tie_break, right.tie_break),
            "failure policy": (left.failure_policy, right.failure_policy),
            "search space": (left.search_space, right.search_space),
            "storage": (left.storage, right.storage),
            "data inputs": (left.base_experiment.data, right.base_experiment.data),
        }
        changed = [name for name, values in equal_fields.items() if values[0] != values[1]]
        if changed:
            raise SweepConfigError(f"paired sweep contract differs: {changed}")
        left_loader = left.base_experiment.loader
        right_loader = right.base_experiment.loader
        fixed_loader_left = (
            left_loader.num_workers,
            left_loader.pin_memory,
            left_loader.persistent_workers,
        )
        fixed_loader_right = (
            right_loader.num_workers,
            right_loader.pin_memory,
            right_loader.persistent_workers,
        )
        if fixed_loader_left != fixed_loader_right:
            raise SweepConfigError("paired sweeps must share fixed loader policy")
        left_optimization = left.base_experiment.optimization
        right_optimization = right.base_experiment.optimization
        fixed_optimization_left = (
            left_optimization.early_stopping_patience,
            left_optimization.early_stopping_min_delta,
            left_optimization.scheduler,
        )
        fixed_optimization_right = (
            right_optimization.early_stopping_patience,
            right_optimization.early_stopping_min_delta,
            right_optimization.scheduler,
        )
        if fixed_optimization_left != fixed_optimization_right:
            raise SweepConfigError("paired sweeps must share early-stopping policy")
        left_runtime = left.base_experiment.runtime
        right_runtime = right.base_experiment.runtime
        if (left_runtime.device, left_runtime.bf16) != (
            right_runtime.device,
            right_runtime.bf16,
        ):
            raise SweepConfigError("paired sweeps must share device and precision policy")


def load_sweep_config(
    path: str | Path,
    *,
    base_dir: str | Path | None = None,
) -> SweepConfig:
    """Load one strict sweep YAML."""

    config_path = Path(path)
    try:
        decoded: object = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise SweepConfigError(f"could not read sweep config {config_path}: {error}") from error
    except yaml.YAMLError as error:
        raise SweepConfigError(f"invalid sweep YAML in {config_path}: {error}") from error
    resolved_base = Path(base_dir) if base_dir is not None else config_path.parent
    return SweepConfig.from_mapping(_mapping(decoded, str(config_path)), base_dir=resolved_base)


def load_equal_budget_pair(
    paths: Sequence[str | Path],
    *,
    base_dir: str | Path | None = None,
) -> EqualBudgetSweepPair:
    """Load and jointly validate the paired ResNet/transformer sweep."""

    configs = [load_sweep_config(path, base_dir=base_dir) for path in paths]
    return EqualBudgetSweepPair.create(configs)
