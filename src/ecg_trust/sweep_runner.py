"""Deterministic, resumable execution for paired PTB-XL development sweeps."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import platform
import secrets
import subprocess
import tempfile
import time
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import optuna
import psutil  # type: ignore[import-untyped]
import scipy  # type: ignore[import-untyped]
import torch
from optuna.distributions import BaseDistribution
from optuna.samplers import BaseSampler
from optuna.study import Study
from optuna.trial import FrozenTrial, Trial, TrialState
from scipy.stats import qmc  # type: ignore[import-untyped]

from ecg_trust.experiment_config import DevelopmentExperimentConfig
from ecg_trust.experiment_runner import DevelopmentRunResult, run_development_experiment
from ecg_trust.protocol import LABEL_ORDER, ExperimentProtocol, FoldRole
from ecg_trust.sweep_config import EqualBudgetSweepPair, FloatRange, SweepConfig

SWEEP_ARTIFACT_SCHEMA_VERSION = 2
_DIMENSIONS = (
    "learning_rate",
    "weight_decay",
    "batch_size",
    "gradient_clip_norm",
    "warmup_epochs",
    "minimum_lr_ratio",
)
_TERMINAL_STATES = frozenset({TrialState.COMPLETE, TrialState.FAIL})
_PAIRED_EXECUTION_POLICY = "candidate_index_pairs_alternating_first_architecture_v1"


class SweepRunnerError(RuntimeError):
    """Raised when execution cannot preserve the paired sweep contract."""


@dataclass(frozen=True, slots=True)
class Candidate:
    index: int
    unit_coordinates: tuple[float, ...]
    parameters: Mapping[str, float | int]

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_index": self.index,
            "unit_coordinates": {
                name: value for name, value in zip(_DIMENSIONS, self.unit_coordinates, strict=True)
            },
            "parameters": dict(self.parameters),
        }


@dataclass(frozen=True, slots=True)
class CandidatePlan:
    comparison_id: str
    algorithm: str
    algorithm_version: int
    scipy_version: str
    design_seed: int
    dimensions: tuple[str, ...]
    candidates: tuple[Candidate, ...]
    plan_hash: str

    def _hash_payload(self) -> dict[str, object]:
        return {
            "schema_version": SWEEP_ARTIFACT_SCHEMA_VERSION,
            "comparison_id": self.comparison_id,
            "algorithm": self.algorithm,
            "algorithm_version": self.algorithm_version,
            "scipy_version": self.scipy_version,
            "design_seed": self.design_seed,
            "dimensions": list(self.dimensions),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._hash_payload(), "plan_hash": self.plan_hash}


@dataclass(frozen=True, slots=True)
class TrialOutcome:
    """Validated uncalibrated fold-8 objective and resource record."""

    best_macro_auroc: float
    best_epoch: int
    completed_epochs: int
    runtime_seconds: float
    peak_allocated_bytes: int
    peak_reserved_bytes: int
    run_dir: Path
    defined_label_count: int
    probabilities_calibrated: bool
    resolved_config_hash: str | None = None
    stopped_early: bool = False

    def __post_init__(self) -> None:
        if not math.isfinite(self.best_macro_auroc) or not 0.0 <= self.best_macro_auroc <= 1.0:
            raise SweepRunnerError("best_macro_auroc must be finite and in [0, 1]")
        if self.best_epoch < 0 or self.completed_epochs < 1:
            raise SweepRunnerError("epoch counts must be positive and zero-based as applicable")
        if self.best_epoch >= self.completed_epochs:
            raise SweepRunnerError("best_epoch must be below completed_epochs")
        if not math.isfinite(self.runtime_seconds) or self.runtime_seconds < 0.0:
            raise SweepRunnerError("runtime_seconds must be finite and non-negative")
        if self.peak_allocated_bytes < 0 or self.peak_reserved_bytes < 0:
            raise SweepRunnerError("VRAM byte counts must be non-negative")


@dataclass(frozen=True, slots=True)
class VerifiedTrialArtifacts:
    """Integrity and objective facts independently recovered from a run directory."""

    selected_checkpoint_score: float
    maximum_observed_score: float
    maximum_observed_epoch: int
    manifest_hash: str
    normalization_file_hash: str
    artifact_sha256: Mapping[str, str]


class TrialExecutor(Protocol):
    def __call__(
        self,
        config: DevelopmentExperimentConfig,
        protocol: ExperimentProtocol,
    ) -> TrialOutcome: ...


@dataclass(frozen=True, slots=True)
class BestCandidateResult:
    architecture: str
    candidate_index: int
    trial_number: int
    best_macro_auroc: float
    best_epoch: int
    completed_epochs: int
    run_dir: Path
    parameters: Mapping[str, object]
    experiment_config_hash: str
    resolved_config_hash: str | None


@dataclass(frozen=True, slots=True)
class ArchitectureSweepResult:
    architecture: str
    study_name: str
    completed_candidates: int
    failed_attempts: int
    total_attempts: int
    budget_complete: bool
    study_summary_path: Path
    best: BestCandidateResult | None


@dataclass(frozen=True, slots=True)
class EqualBudgetSweepResult:
    comparison_id: str
    candidate_plan_path: Path
    summary_path: Path
    studies: tuple[ArchitectureSweepResult, ArchitectureSweepResult]
    best_by_architecture: Mapping[str, BestCandidateResult]


@dataclass(frozen=True, slots=True)
class SweepPreflightResult:
    comparison_id: str
    candidate_plan_hash: str
    candidate_plan_path: Path
    storage_path: Path
    storage_exists: bool
    existing_study_names: tuple[str, ...]
    source_provenance: Mapping[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "comparison_id": self.comparison_id,
            "candidate_plan_hash": self.candidate_plan_hash,
            "candidate_plan_path": str(self.candidate_plan_path),
            "storage_path": str(self.storage_path),
            "storage_exists": self.storage_exists,
            "existing_study_names": list(self.existing_study_names),
            "source_provenance": dict(self.source_provenance),
        }


class EnqueuedPlanOnlySampler(BaseSampler):
    """Fail closed if a trial is not backed by an enqueued plan row."""

    def infer_relative_search_space(
        self,
        study: Study,
        trial: FrozenTrial,
    ) -> dict[str, BaseDistribution]:
        del study, trial
        return {}

    def sample_relative(
        self,
        study: Study,
        trial: FrozenTrial,
        search_space: dict[str, BaseDistribution],
    ) -> dict[str, Any]:
        del study, trial, search_space
        return {}

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        del study, trial, param_distribution
        raise SweepRunnerError(
            f"parameter {param_name!r} was not supplied by the persisted candidate plan"
        )


def _canonical_hash(value: Mapping[str, object]) -> str:
    serialized = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return "sha256:" + hashlib.sha256(serialized.encode()).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise SweepRunnerError(f"could not hash required artifact {path}: {error}") from error
    return "sha256:" + digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    serialized = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _live_local_lock_owner(payload: Mapping[str, object]) -> bool | None:
    if payload.get("hostname") != platform.node():
        return None
    pid = payload.get("pid")
    created = payload.get("process_create_time")
    if isinstance(pid, bool) or not isinstance(pid, int):
        return None
    if isinstance(created, bool) or not isinstance(created, (int, float)):
        return None
    try:
        process = psutil.Process(pid)
        return math.isclose(process.create_time(), float(created), abs_tol=0.01)
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        return False
    except (psutil.AccessDenied, OSError):
        return None


@contextmanager
def _comparison_writer_lock(output_root: Path) -> Iterator[Path]:
    """Hold a process-identity lock for every comparison mutation and training run."""

    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / ".sweep-writer.lock"
    process = psutil.Process(os.getpid())
    payload: dict[str, object] = {
        "schema_version": 1,
        "token": secrets.token_hex(16),
        "hostname": platform.node(),
        "pid": os.getpid(),
        "process_create_time": process.create_time(),
        "created_at_unix": time.time(),
    }
    serialized = json.dumps(payload, sort_keys=True, allow_nan=False) + "\n"
    descriptor: int | None = None
    for _ in range(2):
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            break
        except FileExistsError as error:
            existing = _read_json_mapping(path, "comparison writer lock")
            owner_live = _live_local_lock_owner(existing)
            if owner_live is False:
                try:
                    path.unlink()
                except OSError as unlink_error:
                    raise SweepRunnerError(
                        f"could not recover stale sweep writer lock {path}: {unlink_error}"
                    ) from unlink_error
                continue
            owner = {
                key: existing.get(key)
                for key in ("hostname", "pid", "process_create_time", "created_at_unix")
            }
            raise SweepRunnerError(
                f"comparison is already locked by another writer: {owner}"
            ) from error
    if descriptor is None:
        raise SweepRunnerError(f"could not acquire comparison writer lock {path}")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        yield path
    finally:
        try:
            current = _read_json_mapping(path, "comparison writer lock")
            if current.get("token") == payload["token"]:
                path.unlink(missing_ok=True)
        except SweepRunnerError:
            # Never delete a lock that cannot be proven to belong to this process.
            pass


def _map_float(coordinate: float, value: FloatRange) -> float:
    if value.log:
        low = math.log(value.low)
        high = math.log(value.high)
        return math.exp(low + coordinate * (high - low))
    return value.low + coordinate * (value.high - value.low)


def _map_choice(coordinate: float, choices: Sequence[float | int]) -> float | int:
    index = min(int(coordinate * len(choices)), len(choices) - 1)
    return choices[index]


def build_candidate_plan(pair: EqualBudgetSweepPair) -> CandidatePlan:
    """Build the versioned 12-row Latin-hypercube plan shared by both models."""

    config = pair.resnet
    design = qmc.LatinHypercube(
        d=len(_DIMENSIONS),
        scramble=True,
        seed=config.candidate_design.seed,
    )
    matrix = design.random(n=config.budget.complete_candidates)
    space = config.search_space
    candidates: list[Candidate] = []
    for index, raw_row in enumerate(matrix):
        coordinates = tuple(float(value) for value in raw_row)
        parameters: dict[str, float | int] = {
            "learning_rate": _map_float(coordinates[0], space.learning_rate),
            "weight_decay": _map_float(coordinates[1], space.weight_decay),
            "batch_size": _map_choice(coordinates[2], space.batch_size),
            "gradient_clip_norm": _map_choice(
                coordinates[3], space.gradient_clip_norm
            ),
            "warmup_epochs": _map_choice(coordinates[4], space.warmup_epochs),
            "minimum_lr_ratio": _map_choice(
                coordinates[5], space.minimum_lr_ratio
            ),
        }
        candidates.append(
            Candidate(
                index=index,
                unit_coordinates=coordinates,
                parameters=parameters,
            )
        )
    provisional = CandidatePlan(
        comparison_id=config.comparison_id,
        algorithm=config.candidate_design.algorithm,
        algorithm_version=config.candidate_design.version,
        scipy_version=scipy.__version__,
        design_seed=config.candidate_design.seed,
        dimensions=_DIMENSIONS,
        candidates=tuple(candidates),
        plan_hash="",
    )
    return CandidatePlan(
        comparison_id=provisional.comparison_id,
        algorithm=provisional.algorithm,
        algorithm_version=provisional.algorithm_version,
        scipy_version=provisional.scipy_version,
        design_seed=provisional.design_seed,
        dimensions=provisional.dimensions,
        candidates=provisional.candidates,
        plan_hash=_canonical_hash(provisional._hash_payload()),
    )


def _candidate_plan_path(pair: EqualBudgetSweepPair) -> Path:
    return pair.resnet.storage.output_root / "candidate_plan.json"


def _validate_existing_plan(path: Path, expected: CandidatePlan) -> None:
    try:
        decoded: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SweepRunnerError(f"could not read candidate plan {path}: {error}") from error
    if decoded != expected.to_dict():
        raise SweepRunnerError("persisted candidate plan differs from the resolved plan")


def _persist_candidate_plan(
    pair: EqualBudgetSweepPair,
    plan: CandidatePlan,
    *,
    resume: bool,
) -> Path:
    path = _candidate_plan_path(pair)
    if path.exists():
        if not resume:
            raise SweepRunnerError("candidate plan already exists; explicit resume is required")
        _validate_existing_plan(path, plan)
        return path
    if resume and pair.resnet.storage.sqlite_path.exists():
        raise SweepRunnerError("resume refused because SQLite exists without candidate plan")
    _atomic_json(path, plan.to_dict())
    return path


def _mutable_section(payload: dict[str, object], name: str) -> dict[str, object]:
    value = payload.get(name)
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise SweepRunnerError(f"resolved base experiment has invalid {name!r} section")
    return cast(dict[str, object], value)


def _claim_candidate_parameters(
    trial: Trial,
    candidate: Candidate,
    sweep: SweepConfig,
) -> dict[str, object]:
    expected = candidate.parameters
    space = sweep.search_space
    claimed: dict[str, object] = {
        "learning_rate": trial.suggest_float(
            "learning_rate",
            space.learning_rate.low,
            space.learning_rate.high,
            log=space.learning_rate.log,
        ),
        "weight_decay": trial.suggest_float(
            "weight_decay",
            space.weight_decay.low,
            space.weight_decay.high,
            log=space.weight_decay.log,
        ),
        "batch_size": trial.suggest_categorical(
            "batch_size", list(space.batch_size)
        ),
        "gradient_clip_norm": trial.suggest_categorical(
            "gradient_clip_norm", list(space.gradient_clip_norm)
        ),
        "warmup_epochs": trial.suggest_categorical(
            "warmup_epochs", list(space.warmup_epochs)
        ),
        "minimum_lr_ratio": trial.suggest_categorical(
            "minimum_lr_ratio", list(space.minimum_lr_ratio)
        ),
    }
    if claimed != dict(expected):
        raise SweepRunnerError("Optuna trial parameters differ from candidate plan")
    return claimed


def _resolved_trial_experiment(
    sweep: SweepConfig,
    candidate: Candidate,
    attempt_index: int,
) -> DevelopmentExperimentConfig:
    if attempt_index < 0:
        raise SweepRunnerError("attempt_index must be non-negative")
    sampled = candidate.parameters
    experiment_seed = sweep.seed_policy.experiment_seed
    run_name = (
        f"{sweep.architecture}-candidate{candidate.index:02d}-"
        f"attempt{attempt_index:02d}-seed{experiment_seed}"
    )
    payload = copy.deepcopy(sweep.base_experiment.to_resolved_dict())
    payload["run_name"] = run_name
    _mutable_section(payload, "runtime")["seed"] = experiment_seed
    _mutable_section(payload, "loader")["batch_size"] = sampled["batch_size"]
    optimization = _mutable_section(payload, "optimization")
    optimization.update(
        {
            "epochs": sweep.budget.max_epochs,
            "learning_rate": sampled["learning_rate"],
            "weight_decay": sampled["weight_decay"],
            "gradient_clip_norm": sampled["gradient_clip_norm"],
            "warmup_epochs": sampled["warmup_epochs"],
            "minimum_lr_ratio": sampled["minimum_lr_ratio"],
        }
    )
    trial_root = sweep.storage.output_root / "trials" / sweep.architecture
    _mutable_section(payload, "output")["root_dir"] = str(trial_root)
    experiment = DevelopmentExperimentConfig.from_mapping(payload, base_dir=Path.cwd())
    if experiment.train_folds != tuple(range(1, 8)):
        raise SweepRunnerError("generated attempt escaped training folds 1-7")
    if experiment.validation_folds != (8,):
        raise SweepRunnerError("generated attempt escaped model-selection fold 8")
    if experiment.optimization.epochs != 30:
        raise SweepRunnerError("generated attempt must use a 30-epoch scheduler horizon")
    if experiment.runtime.seed != experiment_seed:
        raise SweepRunnerError("generated attempt changed the fixed HPO seed")
    if experiment.model.architecture != sweep.architecture:
        raise SweepRunnerError("generated attempt changed architecture")
    if experiment.model.preset != "matched_capacity":
        raise SweepRunnerError("generated attempt changed matched-capacity preset")
    return experiment


def build_trial_experiment(
    sweep: SweepConfig,
    candidate: Candidate,
    attempt_index: int,
    trial: Trial,
) -> DevelopmentExperimentConfig:
    """Resolve one immutable attempt without mutating the checked-in base config."""

    _claim_candidate_parameters(trial, candidate, sweep)
    return _resolved_trial_experiment(sweep, candidate, attempt_index)


def _object_mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise SweepRunnerError(f"{context} must be a string-keyed mapping")
    return cast(Mapping[str, object], value)


def _finite_float(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SweepRunnerError(f"{context} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise SweepRunnerError(f"{context} must be finite and non-negative")
    return parsed


def _nonnegative_int(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SweepRunnerError(f"{context} must be a non-negative integer")
    return value


def _best_epoch_defined_labels(result: DevelopmentRunResult) -> int:
    matching: Mapping[str, object] | None = None
    try:
        lines = result.history_path.read_text(encoding="utf-8").splitlines()
        for line in lines:
            decoded: object = json.loads(line)
            record = _object_mapping(decoded, "history record")
            if record.get("epoch") == result.best_epoch:
                matching = record
                break
    except (OSError, json.JSONDecodeError) as error:
        raise SweepRunnerError(f"could not inspect best-epoch history: {error}") from error
    if matching is None:
        raise SweepRunnerError("history has no record for the selected best epoch")
    metrics = _object_mapping(matching.get("validation_metrics"), "validation metrics")
    label_order = metrics.get("label_order")
    if label_order != list(LABEL_ORDER):
        raise SweepRunnerError("validation metrics changed the canonical label order")
    macro = _object_mapping(metrics.get("macro"), "validation macro metrics")
    label_count = _nonnegative_int(macro.get("roc_auc_labels"), "roc_auc_labels")
    per_label = metrics.get("per_label")
    if not isinstance(per_label, list) or len(per_label) != len(LABEL_ORDER):
        raise SweepRunnerError("validation metrics must contain all five labels")
    for raw in per_label:
        label = _object_mapping(raw, "per-label metric")
        if label.get("roc_auc") is None:
            raise SweepRunnerError("fold-8 ROC-AUC is undefined for at least one label")
    recorded_score = _finite_float(
        matching.get("validation_macro_auroc"), "validation_macro_auroc"
    )
    if not math.isclose(recorded_score, result.best_macro_auroc, rel_tol=0.0, abs_tol=1e-12):
        raise SweepRunnerError("best result differs from its uncalibrated history record")
    return label_count


def _outcome_from_development_result(result: DevelopmentRunResult) -> TrialOutcome:
    metadata_path = result.run_dir / "run_metadata.json"
    try:
        decoded: object = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SweepRunnerError(f"could not read {metadata_path}: {error}") from error
    metadata = _object_mapping(decoded, "development run metadata")
    if metadata.get("status") != "complete":
        raise SweepRunnerError("development run metadata is not complete")
    vram = _object_mapping(metadata.get("vram"), "development run VRAM")
    return TrialOutcome(
        best_macro_auroc=result.best_macro_auroc,
        best_epoch=result.best_epoch,
        completed_epochs=result.completed_epochs,
        runtime_seconds=_finite_float(metadata.get("elapsed_seconds"), "elapsed_seconds"),
        peak_allocated_bytes=_nonnegative_int(
            vram.get("peak_allocated_bytes"), "peak_allocated_bytes"
        ),
        peak_reserved_bytes=_nonnegative_int(
            vram.get("peak_reserved_bytes"), "peak_reserved_bytes"
        ),
        run_dir=result.run_dir,
        defined_label_count=_best_epoch_defined_labels(result),
        probabilities_calibrated=False,
        resolved_config_hash=result.resolved_config_hash,
        stopped_early=result.stopped_early,
    )


def execute_development_trial(
    config: DevelopmentExperimentConfig,
    protocol: ExperimentProtocol,
) -> TrialOutcome:
    """Execute one candidate through the existing development runner."""

    return _outcome_from_development_result(
        run_development_experiment(config, protocol=protocol)
    )


def _read_json_mapping(path: Path, context: str) -> Mapping[str, object]:
    try:
        decoded: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SweepRunnerError(f"could not read {context} at {path}: {error}") from error
    return _object_mapping(decoded, context)


def _same_float(left: object, right: float, context: str) -> None:
    parsed = _finite_float(left, context)
    if not math.isclose(parsed, right, rel_tol=0.0, abs_tol=1e-12):
        raise SweepRunnerError(f"{context} does not match the selected objective")


def _history_artifact_facts(
    history_path: Path,
    *,
    selected_epoch: int,
    selected_score: float,
    completed_epochs: int,
) -> tuple[float, int]:
    try:
        lines = history_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise SweepRunnerError(
            f"could not read objective history {history_path}: {error}"
        ) from error
    if len(lines) != completed_epochs:
        raise SweepRunnerError("objective history length differs from completed_epochs")
    scores: list[tuple[float, int]] = []
    selected: Mapping[str, object] | None = None
    for expected_epoch, line in enumerate(lines):
        try:
            decoded: object = json.loads(line)
        except json.JSONDecodeError as error:
            raise SweepRunnerError(f"objective history line is invalid JSON: {error}") from error
        record = _object_mapping(decoded, "objective history record")
        if record.get("epoch") != expected_epoch:
            raise SweepRunnerError("objective history epochs are not contiguous and zero-based")
        score = _finite_float(
            record.get("validation_macro_auroc"), "validation_macro_auroc"
        )
        if score > 1.0:
            raise SweepRunnerError("validation_macro_auroc must be in [0, 1]")
        scores.append((score, expected_epoch))
        if expected_epoch == selected_epoch:
            selected = record
    if selected is None:
        raise SweepRunnerError("selected checkpoint epoch is absent from objective history")
    _same_float(
        selected.get("validation_macro_auroc"),
        selected_score,
        "selected history score",
    )
    if selected.get("improved") is not True:
        raise SweepRunnerError("selected checkpoint history row was not marked improved")
    metrics = _object_mapping(selected.get("validation_metrics"), "validation metrics")
    if metrics.get("label_order") != list(LABEL_ORDER):
        raise SweepRunnerError("selected objective changed the canonical label order")
    macro = _object_mapping(metrics.get("macro"), "validation macro metrics")
    if macro.get("roc_auc_labels") != len(LABEL_ORDER):
        raise SweepRunnerError("selected objective does not define all five labels")
    _same_float(macro.get("roc_auc"), selected_score, "selected macro ROC-AUC")
    per_label = metrics.get("per_label")
    if not isinstance(per_label, list) or len(per_label) != len(LABEL_ORDER):
        raise SweepRunnerError("selected objective must contain all five per-label metrics")
    for expected_label, raw_label in zip(LABEL_ORDER, per_label, strict=True):
        label = _object_mapping(raw_label, "per-label metric")
        if label.get("label") != expected_label or label.get("roc_auc") is None:
            raise SweepRunnerError("selected objective has undefined or reordered label AUROC")
        auc = _finite_float(label.get("roc_auc"), f"{expected_label} ROC-AUC")
        if auc > 1.0:
            raise SweepRunnerError(f"{expected_label} ROC-AUC must be in [0, 1]")
    maximum_score, maximum_epoch = max(scores, key=lambda item: (item[0], -item[1]))
    return maximum_score, maximum_epoch


def _checkpoint_mapping(path: Path, context: str) -> Mapping[str, object]:
    try:
        decoded: object = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise SweepRunnerError(f"could not load {context} {path}: {error}") from error
    return _object_mapping(decoded, context)


def _verify_checkpoint_identity(
    checkpoint: Mapping[str, object],
    *,
    epoch: int,
    selected_epoch: int,
    protocol_hash: str,
    manifest_hash: str,
    resolved_config_hash: str,
    resolved_config: Mapping[str, object],
    selected_score: float,
) -> None:
    required = {
        "schema_version",
        "epoch",
        "protocol_hash",
        "manifest_hash",
        "config",
        "config_hash",
        "model_state_dict",
        "optimizer_state_dict",
        "scaler_state_dict",
        "early_stopping_state_dict",
    }
    if set(checkpoint) != required or checkpoint.get("schema_version") != 1:
        raise SweepRunnerError("checkpoint keys or schema version are unsupported")
    if checkpoint.get("epoch") != epoch:
        raise SweepRunnerError("checkpoint epoch disagrees with run metadata")
    if checkpoint.get("protocol_hash") != protocol_hash:
        raise SweepRunnerError("checkpoint protocol hash disagrees with sweep protocol")
    if checkpoint.get("manifest_hash") != manifest_hash:
        raise SweepRunnerError("checkpoint manifest hash disagrees with run metadata")
    if checkpoint.get("config_hash") != resolved_config_hash:
        raise SweepRunnerError("checkpoint resolved-config hash disagrees with metadata")
    if checkpoint.get("config") != resolved_config:
        raise SweepRunnerError("checkpoint embedded config disagrees with resolved config")
    stopper = _object_mapping(
        checkpoint.get("early_stopping_state_dict"), "checkpoint early-stopping state"
    )
    if stopper.get("best_epoch") != selected_epoch:
        raise SweepRunnerError("checkpoint selected epoch disagrees with run metadata")
    _same_float(stopper.get("best_score"), selected_score, "checkpoint selected score")


def _verify_trial_artifacts(
    config: SweepConfig,
    protocol: ExperimentProtocol,
    candidate: Candidate,
    attempt_index: int,
    *,
    run_dir: Path,
    selected_score: float,
    selected_epoch: int,
    completed_epochs: int,
    expected_experiment_config_hash: str,
    expected_resolved_config_hash: str | None,
) -> VerifiedTrialArtifacts:
    expected_experiment = _resolved_trial_experiment(config, candidate, attempt_index)
    if expected_experiment.config_hash != expected_experiment_config_hash:
        raise SweepRunnerError("trial experiment-config hash drifted from the candidate plan")
    expected_run_dir = expected_experiment.output.root_dir / expected_experiment.run_name
    if run_dir.resolve() != expected_run_dir.resolve():
        raise SweepRunnerError("trial run directory drifted from its immutable attempt identity")
    paths = {
        "resolved_config.json": run_dir / "resolved_config.json",
        "protocol.json": run_dir / "protocol.json",
        "run_metadata.json": run_dir / "run_metadata.json",
        "history.jsonl": run_dir / "history.jsonl",
        "best.ckpt": run_dir / "best.ckpt",
        "last.ckpt": run_dir / "last.ckpt",
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise SweepRunnerError(f"COMPLETE attempt is missing required artifacts: {missing}")

    resolved_wrapper = _read_json_mapping(paths["resolved_config.json"], "resolved config")
    resolved_config_hash = resolved_wrapper.get("config_hash")
    if not isinstance(resolved_config_hash, str) or not resolved_config_hash:
        raise SweepRunnerError("resolved config has no config_hash")
    resolved_config = _object_mapping(resolved_wrapper.get("config"), "resolved config body")
    if _canonical_hash(resolved_config) != resolved_config_hash:
        raise SweepRunnerError("resolved config content does not match its hash")
    if (
        expected_resolved_config_hash is not None
        and resolved_config_hash != expected_resolved_config_hash
    ):
        raise SweepRunnerError("executor resolved-config hash disagrees with its artifact")
    expected_sections = expected_experiment.to_resolved_dict()
    for key in (
        "schema_version",
        "run_name",
        "folds",
        "data",
        "loader",
        "optimization",
        "runtime",
        "output",
    ):
        if resolved_config.get(key) != expected_sections[key]:
            raise SweepRunnerError(f"resolved config section {key!r} drifted from candidate")
    expected_model = _object_mapping(expected_sections["model"], "expected model")
    actual_model = _object_mapping(resolved_config.get("model"), "resolved model")
    if any(actual_model.get(key) != value for key, value in expected_model.items()):
        raise SweepRunnerError("resolved model architecture or preset drifted")

    recorded_protocol = _read_json_mapping(paths["protocol.json"], "protocol artifact")
    if recorded_protocol != protocol.to_resolved_dict():
        raise SweepRunnerError("protocol artifact does not match the sweep protocol")
    metadata = _read_json_mapping(paths["run_metadata.json"], "run metadata")
    if metadata.get("status") != "complete":
        raise SweepRunnerError("run metadata is not complete")
    expected_metadata: dict[str, object] = {
        "seed": config.seed_policy.experiment_seed,
        "source_config_hash": expected_experiment_config_hash,
        "resolved_config_hash": resolved_config_hash,
        "protocol_hash": protocol.protocol_hash,
        "completed_epochs": completed_epochs,
        "best_epoch": selected_epoch,
    }
    for key, value in expected_metadata.items():
        if metadata.get(key) != value:
            raise SweepRunnerError(f"run metadata field {key!r} disagrees with trial")
    _same_float(
        metadata.get("best_validation_macro_auroc"),
        selected_score,
        "metadata selected score",
    )
    manifest_hash = metadata.get("manifest_hash")
    normalization_hash = metadata.get("normalization_file_hash")
    if not isinstance(manifest_hash, str) or not manifest_hash:
        raise SweepRunnerError("run metadata has no manifest hash")
    if not isinstance(normalization_hash, str) or not normalization_hash:
        raise SweepRunnerError("run metadata has no normalization hash")
    expected_manifest_hash = _file_sha256(
        config.base_experiment.data.manifest_path
    ).removeprefix("sha256:")
    expected_normalization_hash = _file_sha256(
        config.base_experiment.data.normalization_path
    ).removeprefix("sha256:")
    if manifest_hash.removeprefix("sha256:") != expected_manifest_hash:
        raise SweepRunnerError("run metadata manifest hash disagrees with current input")
    if normalization_hash.removeprefix("sha256:") != expected_normalization_hash:
        raise SweepRunnerError("run metadata normalization hash disagrees with current input")

    maximum_score, maximum_epoch = _history_artifact_facts(
        paths["history.jsonl"],
        selected_epoch=selected_epoch,
        selected_score=selected_score,
        completed_epochs=completed_epochs,
    )
    best_checkpoint = _checkpoint_mapping(paths["best.ckpt"], "best checkpoint")
    _verify_checkpoint_identity(
        best_checkpoint,
        epoch=selected_epoch,
        selected_epoch=selected_epoch,
        protocol_hash=protocol.protocol_hash,
        manifest_hash=manifest_hash,
        resolved_config_hash=resolved_config_hash,
        resolved_config=resolved_config,
        selected_score=selected_score,
    )
    last_checkpoint = _checkpoint_mapping(paths["last.ckpt"], "last checkpoint")
    _verify_checkpoint_identity(
        last_checkpoint,
        epoch=completed_epochs - 1,
        selected_epoch=selected_epoch,
        protocol_hash=protocol.protocol_hash,
        manifest_hash=manifest_hash,
        resolved_config_hash=resolved_config_hash,
        resolved_config=resolved_config,
        selected_score=selected_score,
    )
    return VerifiedTrialArtifacts(
        selected_checkpoint_score=selected_score,
        maximum_observed_score=maximum_score,
        maximum_observed_epoch=maximum_epoch,
        manifest_hash=manifest_hash,
        normalization_file_hash=normalization_hash,
        artifact_sha256={name: _file_sha256(path) for name, path in paths.items()},
    )


def _sqlite_url(path: Path) -> str:
    return "sqlite:///" + path.resolve().as_posix()


def _run_git(root: Path, arguments: list[str]) -> tuple[bool, str]:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return False, f"{type(error).__name__}: {error}"
    output = completed.stdout.strip()
    if completed.returncode != 0:
        detail = completed.stderr.strip() or output or f"exit {completed.returncode}"
        return False, detail
    return True, output


def _source_tree_hash(root: Path) -> str:
    candidates: set[Path] = set()
    for directory, pattern in (("src", "*.py"), ("scripts", "*.py"), ("configs", "*.yaml")):
        base = root / directory
        if base.is_dir():
            candidates.update(path for path in base.rglob(pattern) if path.is_file())
    for name in ("pyproject.toml", "uv.lock"):
        path = root / name
        if path.is_file():
            candidates.add(path)
    digest = hashlib.sha256()
    for path in sorted(candidates, key=lambda item: item.as_posix()):
        relative = path.relative_to(root).as_posix().encode()
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return "sha256:" + digest.hexdigest()


def source_provenance(config: SweepConfig) -> dict[str, object]:
    """Capture source identity used by resume drift gates."""

    project_root = config.base_experiment_path.parent.parent.resolve()
    root_ok, root_output = _run_git(project_root, ["rev-parse", "--show-toplevel"])
    git_root = Path(root_output).resolve() if root_ok else project_root
    head_ok, head_output = _run_git(git_root, ["rev-parse", "HEAD"])
    status_ok, status_output = _run_git(
        git_root, ["status", "--porcelain=v1", "--untracked-files=all"]
    )
    lockfile = project_root / "uv.lock"
    manifest = config.base_experiment.data.manifest_path
    normalization = config.base_experiment.data.normalization_path
    runtime_identity: dict[str, object] = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "optuna": str(optuna.__version__),
        "scipy": str(scipy.__version__),
        "torch": str(torch.__version__),
    }
    return {
        "project_root": str(project_root),
        "git_root": str(git_root) if root_ok else None,
        "git_head": head_output if head_ok else None,
        "git_dirty": bool(status_output) if status_ok else None,
        "git_status_sha256": (
            "sha256:" + hashlib.sha256(status_output.encode()).hexdigest()
            if status_ok
            else None
        ),
        "git_unavailable": not (root_ok and head_ok and status_ok),
        "source_tree_sha256": _source_tree_hash(project_root),
        "dependency_lock_sha256": _file_sha256(lockfile),
        "manifest_path": str(manifest),
        "manifest_sha256": _file_sha256(manifest),
        "normalization_path": str(normalization),
        "normalization_sha256": _file_sha256(normalization),
        "runtime_identity": runtime_identity,
        "runtime_identity_sha256": _canonical_hash(runtime_identity),
    }


def _study_attrs(
    config: SweepConfig,
    protocol: ExperimentProtocol,
    plan: CandidatePlan,
    provenance: Mapping[str, object],
) -> dict[str, object]:
    return {
        "artifact_schema_version": SWEEP_ARTIFACT_SCHEMA_VERSION,
        "comparison_id": config.comparison_id,
        "architecture": config.architecture,
        "sweep_config_hash": config.config_hash,
        "base_experiment_config_hash": config.base_experiment.config_hash,
        "protocol_hash": protocol.protocol_hash,
        "candidate_plan_hash": plan.plan_hash,
        "objective": config.objective.to_dict(),
        "seed_policy": config.seed_policy.to_dict(),
        "tie_break": config.tie_break.to_dict(),
        "failure_policy": config.failure_policy.to_dict(),
        "source_provenance": dict(provenance),
    }


def _existing_study_names(path: Path) -> tuple[str, ...]:
    if not path.exists():
        return ()
    try:
        return tuple(optuna.study.get_all_study_names(storage=_sqlite_url(path)))
    except Exception as error:
        raise SweepRunnerError(f"could not inspect Optuna storage: {error}") from error


def _load_or_create_study(
    config: SweepConfig,
    *,
    protocol: ExperimentProtocol,
    plan: CandidatePlan,
    provenance: Mapping[str, object],
    resume: bool,
) -> Study:
    config.storage.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    storage = _sqlite_url(config.storage.sqlite_path)
    names = _existing_study_names(config.storage.sqlite_path)
    exists = config.study_name in names
    if exists and not resume:
        raise SweepRunnerError(
            f"study {config.study_name!r} exists; explicit resume is required"
        )
    sampler = EnqueuedPlanOnlySampler()
    try:
        study = (
            optuna.load_study(
                study_name=config.study_name, storage=storage, sampler=sampler
            )
            if exists
            else optuna.create_study(
                study_name=config.study_name,
                storage=storage,
                sampler=sampler,
                direction="maximize",
                load_if_exists=False,
            )
        )
    except Exception as error:
        raise SweepRunnerError(f"could not open Optuna study: {error}") from error
    expected = _study_attrs(config, protocol, plan, provenance)
    if exists:
        missing = sorted(set(expected).difference(study.user_attrs))
        changed = sorted(
            key
            for key, value in expected.items()
            if key in study.user_attrs and study.user_attrs[key] != value
        )
        if missing or changed:
            raise SweepRunnerError(
                "resume refused because study provenance drifted; "
                f"missing={missing}, changed={changed}"
            )
    else:
        for key, value in expected.items():
            study.set_user_attr(key, value)
    return study


def _candidate_index(trial: FrozenTrial) -> int:
    value = trial.user_attrs.get("candidate_index")
    if isinstance(value, bool) or not isinstance(value, int):
        raise SweepRunnerError(f"trial {trial.number} has no candidate_index")
    return value


def _attempt_index(trial: FrozenTrial) -> int:
    value = trial.user_attrs.get("attempt_index")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SweepRunnerError(f"trial {trial.number} has no valid attempt_index")
    return value


def _verified_complete_trial(
    config: SweepConfig,
    protocol: ExperimentProtocol,
    plan: CandidatePlan,
    trial: FrozenTrial,
) -> VerifiedTrialArtifacts:
    candidate_index = _candidate_index(trial)
    attempt_index = _attempt_index(trial)
    value = trial.value
    run_dir = trial.user_attrs.get("run_dir")
    best_epoch = trial.user_attrs.get("best_epoch")
    completed_epochs = trial.user_attrs.get("completed_epochs")
    experiment_hash = trial.user_attrs.get("experiment_config_hash")
    resolved_hash = trial.user_attrs.get("resolved_config_hash")
    if not isinstance(value, float):
        raise SweepRunnerError(f"COMPLETE trial {trial.number} has no scalar objective")
    if not isinstance(run_dir, str) or not run_dir:
        raise SweepRunnerError(f"COMPLETE trial {trial.number} has no run directory")
    if isinstance(best_epoch, bool) or not isinstance(best_epoch, int):
        raise SweepRunnerError(f"COMPLETE trial {trial.number} has no best epoch")
    if isinstance(completed_epochs, bool) or not isinstance(completed_epochs, int):
        raise SweepRunnerError(f"COMPLETE trial {trial.number} has no completed epochs")
    if not isinstance(experiment_hash, str) or not experiment_hash:
        raise SweepRunnerError(f"COMPLETE trial {trial.number} has no experiment hash")
    if not isinstance(resolved_hash, str) or not resolved_hash:
        raise SweepRunnerError(f"COMPLETE trial {trial.number} has no resolved-config hash")
    verified = _verify_trial_artifacts(
        config,
        protocol,
        plan.candidates[candidate_index],
        attempt_index,
        run_dir=Path(run_dir),
        selected_score=value,
        selected_epoch=best_epoch,
        completed_epochs=completed_epochs,
        expected_experiment_config_hash=experiment_hash,
        expected_resolved_config_hash=resolved_hash,
    )
    expected_attrs: dict[str, object] = {
        "selected_checkpoint_score": verified.selected_checkpoint_score,
        "literal_max_macro_auroc": verified.maximum_observed_score,
        "literal_max_epoch": verified.maximum_observed_epoch,
        "manifest_hash": verified.manifest_hash,
        "normalization_file_hash": verified.normalization_file_hash,
        "artifact_sha256": dict(verified.artifact_sha256),
        "defined_label_count": len(LABEL_ORDER),
        "probabilities_calibrated": False,
    }
    changed = [
        key for key, expected in expected_attrs.items() if trial.user_attrs.get(key) != expected
    ]
    if changed:
        raise SweepRunnerError(
            f"COMPLETE trial {trial.number} artifact attributes drifted: {changed}"
        )
    return verified


def _reconcile_trials(
    config: SweepConfig,
    study: Study,
    plan: CandidatePlan,
    protocol: ExperimentProtocol,
    *,
    resume: bool,
) -> None:
    seen_attempts: set[tuple[int, int]] = set()
    for trial in study.get_trials(deepcopy=False):
        candidate_index = _candidate_index(trial)
        attempt_index = _attempt_index(trial)
        if candidate_index < 0 or candidate_index >= len(plan.candidates):
            raise SweepRunnerError(f"trial {trial.number} has invalid candidate_index")
        identity = (candidate_index, attempt_index)
        if identity in seen_attempts:
            raise SweepRunnerError(f"duplicate candidate attempt identity {identity}")
        seen_attempts.add(identity)
        fixed = trial.system_attrs.get("fixed_params")
        if fixed != dict(plan.candidates[candidate_index].parameters):
            raise SweepRunnerError(f"trial {trial.number} fixed params drifted from plan")
        if trial.state == TrialState.PRUNED:
            raise SweepRunnerError("PRUNED trial found although pruning is disabled")
        if trial.state == TrialState.RUNNING:
            if not resume:
                raise SweepRunnerError("RUNNING trial requires explicit resume")
            study.tell(trial.number, state=TrialState.FAIL, skip_if_finished=True)
        elif trial.state == TrialState.COMPLETE:
            _verified_complete_trial(config, protocol, plan, trial)
    for candidate in plan.candidates:
        attempts = _trials_for_candidate(study, candidate.index)
        indices = sorted(_attempt_index(trial) for trial in attempts)
        if indices != list(range(len(indices))):
            raise SweepRunnerError(
                f"candidate {candidate.index} attempt indices are not contiguous"
            )
        if len(attempts) > config.failure_policy.max_attempts_per_candidate:
            raise SweepRunnerError(f"candidate {candidate.index} exceeded its attempt cap")
        completed = [trial for trial in attempts if trial.state == TrialState.COMPLETE]
        if len(completed) > 1:
            raise SweepRunnerError(
                f"candidate {candidate.index} has multiple COMPLETE attempts"
            )
        if completed and any(trial.state not in _TERMINAL_STATES for trial in attempts):
            raise SweepRunnerError(
                f"candidate {candidate.index} has pending work after completion"
            )


def _trials_for_candidate(study: Study, candidate_index: int) -> list[FrozenTrial]:
    return [
        trial
        for trial in study.get_trials(deepcopy=False)
        if _candidate_index(trial) == candidate_index
    ]


def _completed_for_candidate(study: Study, candidate_index: int) -> FrozenTrial | None:
    completed = [
        trial
        for trial in _trials_for_candidate(study, candidate_index)
        if trial.state == TrialState.COMPLETE
    ]
    if len(completed) > 1:
        raise SweepRunnerError(f"candidate {candidate_index} has multiple COMPLETE trials")
    return completed[0] if completed else None


def _trial_record(trial: FrozenTrial) -> dict[str, object]:
    failure = trial.user_attrs.get("failure")
    if trial.state == TrialState.FAIL and failure is None:
        failure = "interrupted RUNNING attempt marked FAIL during resume"
    return {
        "trial_number": trial.number,
        "candidate_index": trial.user_attrs.get("candidate_index"),
        "attempt_index": trial.user_attrs.get("attempt_index"),
        "state": trial.state.name,
        "parameters": dict(trial.params),
        "best_fold8_uncalibrated_macro_roc_auc": trial.value,
        "selected_checkpoint_score": trial.user_attrs.get("selected_checkpoint_score"),
        "literal_max_macro_auroc": trial.user_attrs.get("literal_max_macro_auroc"),
        "literal_max_epoch": trial.user_attrs.get("literal_max_epoch"),
        "best_epoch": trial.user_attrs.get("best_epoch"),
        "completed_epochs": trial.user_attrs.get("completed_epochs"),
        "defined_label_count": trial.user_attrs.get("defined_label_count"),
        "probabilities_calibrated": trial.user_attrs.get("probabilities_calibrated"),
        "runtime_seconds": trial.user_attrs.get("runtime_seconds"),
        "wall_seconds": trial.user_attrs.get("wall_seconds"),
        "vram": {
            "peak_allocated_bytes": trial.user_attrs.get("peak_allocated_bytes"),
            "peak_reserved_bytes": trial.user_attrs.get("peak_reserved_bytes"),
        },
        "experiment_config_hash": trial.user_attrs.get("experiment_config_hash"),
        "resolved_config_hash": trial.user_attrs.get("resolved_config_hash"),
        "manifest_hash": trial.user_attrs.get("manifest_hash"),
        "normalization_file_hash": trial.user_attrs.get("normalization_file_hash"),
        "artifact_sha256": trial.user_attrs.get("artifact_sha256"),
        "run_dir": trial.user_attrs.get("run_dir"),
        "failure": failure,
    }


def _completed_candidates(study: Study, plan: CandidatePlan) -> list[FrozenTrial]:
    completed: list[FrozenTrial] = []
    for candidate in plan.candidates:
        trial = _completed_for_candidate(study, candidate.index)
        if trial is not None:
            completed.append(trial)
    return completed


def _best_trial(completed: Sequence[FrozenTrial]) -> FrozenTrial:
    if not completed:
        raise SweepRunnerError("no completed candidate is selectable")

    def key(trial: FrozenTrial) -> tuple[float, int, int, int]:
        value = trial.value
        epochs = trial.user_attrs.get("completed_epochs")
        if not isinstance(value, float):
            raise SweepRunnerError("completed candidate has no scalar objective")
        if isinstance(epochs, bool) or not isinstance(epochs, int):
            raise SweepRunnerError("completed candidate has no completed_epochs")
        return (value, -epochs, -_candidate_index(trial), -trial.number)

    return max(completed, key=key)


def _best_result(architecture: str, trial: FrozenTrial) -> BestCandidateResult:
    value = trial.value
    best_epoch = trial.user_attrs.get("best_epoch")
    completed_epochs = trial.user_attrs.get("completed_epochs")
    run_dir = trial.user_attrs.get("run_dir")
    config_hash = trial.user_attrs.get("experiment_config_hash")
    resolved_hash = trial.user_attrs.get("resolved_config_hash")
    if not isinstance(value, float):
        raise SweepRunnerError("best candidate has no objective")
    if isinstance(best_epoch, bool) or not isinstance(best_epoch, int):
        raise SweepRunnerError("best candidate has no best_epoch")
    if isinstance(completed_epochs, bool) or not isinstance(completed_epochs, int):
        raise SweepRunnerError("best candidate has no completed_epochs")
    if not isinstance(run_dir, str) or not run_dir:
        raise SweepRunnerError("best candidate has no run_dir")
    if not isinstance(config_hash, str) or not config_hash:
        raise SweepRunnerError("best candidate has no experiment config hash")
    if resolved_hash is not None and not isinstance(resolved_hash, str):
        raise SweepRunnerError("best candidate has invalid resolved config hash")
    return BestCandidateResult(
        architecture=architecture,
        candidate_index=_candidate_index(trial),
        trial_number=trial.number,
        best_macro_auroc=value,
        best_epoch=best_epoch,
        completed_epochs=completed_epochs,
        run_dir=Path(run_dir),
        parameters=dict(trial.params),
        experiment_config_hash=config_hash,
        resolved_config_hash=resolved_hash,
    )


def _study_payload(
    config: SweepConfig,
    study: Study,
    plan: CandidatePlan,
) -> dict[str, object]:
    trials = study.get_trials(deepcopy=False)
    completed = _completed_candidates(study, plan)
    budget_complete = len(completed) == config.budget.complete_candidates
    state_counts = Counter(trial.state.name for trial in trials)
    return {
        "schema_version": SWEEP_ARTIFACT_SCHEMA_VERSION,
        "comparison_id": config.comparison_id,
        "architecture": config.architecture,
        "study_name": config.study_name,
        "candidate_plan_hash": plan.plan_hash,
        "sweep_config_hash": config.config_hash,
        "base_experiment_config_hash": config.base_experiment.config_hash,
        "objective": config.objective.to_dict(),
        "seed_policy": config.seed_policy.to_dict(),
        "tie_break": config.tie_break.to_dict(),
        "failure_policy": config.failure_policy.to_dict(),
        "required_complete_candidates": config.budget.complete_candidates,
        "completed_candidates": len(completed),
        "budget_complete": budget_complete,
        "attempt_counts": dict(sorted(state_counts.items())),
        "selection_released": False,
        "best_candidate": None,
        "study_user_attrs": dict(study.user_attrs),
        "attempts": [_trial_record(trial) for trial in trials],
    }


def _write_study_summary(config: SweepConfig, study: Study, plan: CandidatePlan) -> Path:
    path = config.storage.output_root / f"{config.architecture}_study.json"
    _atomic_json(path, _study_payload(config, study, plan))
    return path


def _architecture_result(
    config: SweepConfig,
    study: Study,
    plan: CandidatePlan,
    *,
    release_selection: bool = False,
) -> ArchitectureSweepResult:
    trials = study.get_trials(deepcopy=False)
    completed = _completed_candidates(study, plan)
    budget_complete = len(completed) == config.budget.complete_candidates
    best = _best_trial(completed) if budget_complete and release_selection else None
    return ArchitectureSweepResult(
        architecture=config.architecture,
        study_name=config.study_name,
        completed_candidates=len(completed),
        failed_attempts=sum(trial.state == TrialState.FAIL for trial in trials),
        total_attempts=len(trials),
        budget_complete=budget_complete,
        study_summary_path=_write_study_summary(config, study, plan),
        best=_best_result(config.architecture, best) if best is not None else None,
    )


def _validate_outcome(config: SweepConfig, outcome: TrialOutcome) -> None:
    if outcome.defined_label_count != config.objective.required_label_count:
        raise SweepRunnerError("objective did not define ROC-AUC for all five labels")
    if outcome.probabilities_calibrated:
        raise SweepRunnerError("HPO objective must use uncalibrated fold-8 probabilities")
    if outcome.completed_epochs > config.budget.max_epochs:
        raise SweepRunnerError("attempt exceeded the 30-epoch scheduler horizon")
    if not isinstance(outcome.resolved_config_hash, str) or not outcome.resolved_config_hash:
        raise SweepRunnerError("attempt did not return its resolved-config hash")


def _execute_candidate(
    config: SweepConfig,
    study: Study,
    plan: CandidatePlan,
    candidate: Candidate,
    protocol: ExperimentProtocol,
    executor: TrialExecutor,
) -> None:
    if _completed_for_candidate(study, candidate.index) is not None:
        return
    attempts = _trials_for_candidate(study, candidate.index)
    waiting = [trial for trial in attempts if trial.state == TrialState.WAITING]
    if len(waiting) > 1:
        raise SweepRunnerError(f"candidate {candidate.index} has multiple WAITING attempts")
    max_attempts = config.failure_policy.max_attempts_per_candidate
    while _completed_for_candidate(study, candidate.index) is None:
        attempts = _trials_for_candidate(study, candidate.index)
        if len(attempts) >= max_attempts and not any(
            trial.state == TrialState.WAITING for trial in attempts
        ):
            raise SweepRunnerError(
                f"candidate {candidate.index} exhausted {max_attempts} attempts"
            )
        waiting = [trial for trial in attempts if trial.state == TrialState.WAITING]
        if not waiting:
            attempt_index = max((_attempt_index(trial) for trial in attempts), default=-1) + 1
            study.enqueue_trial(
                dict(candidate.parameters),
                user_attrs={
                    "candidate_index": candidate.index,
                    "attempt_index": attempt_index,
                },
                skip_if_exists=False,
            )
        trial = study.ask()
        dequeued_candidate = trial.user_attrs.get("candidate_index")
        if dequeued_candidate != candidate.index:
            raise SweepRunnerError("Optuna dequeued a candidate out of plan order")
        attempt_index = cast(int, trial.user_attrs["attempt_index"])
        started = time.perf_counter()
        try:
            experiment = build_trial_experiment(config, candidate, attempt_index, trial)
            run_dir = experiment.output.root_dir / experiment.run_name
            if run_dir.exists():
                raise SweepRunnerError(f"immutable attempt directory exists: {run_dir}")
            trial.set_user_attr("experiment_config_hash", experiment.config_hash)
            trial.set_user_attr("run_dir", str(run_dir))
            outcome = executor(experiment, protocol)
            _validate_outcome(config, outcome)
            if outcome.run_dir.resolve() != run_dir.resolve():
                raise SweepRunnerError("executor returned an unexpected attempt directory")
            verified = _verify_trial_artifacts(
                config,
                protocol,
                candidate,
                attempt_index,
                run_dir=outcome.run_dir,
                selected_score=outcome.best_macro_auroc,
                selected_epoch=outcome.best_epoch,
                completed_epochs=outcome.completed_epochs,
                expected_experiment_config_hash=experiment.config_hash,
                expected_resolved_config_hash=outcome.resolved_config_hash,
            )
            trial.set_user_attr("best_epoch", outcome.best_epoch)
            trial.set_user_attr("completed_epochs", outcome.completed_epochs)
            trial.set_user_attr("defined_label_count", outcome.defined_label_count)
            trial.set_user_attr(
                "probabilities_calibrated", outcome.probabilities_calibrated
            )
            trial.set_user_attr("runtime_seconds", outcome.runtime_seconds)
            trial.set_user_attr("wall_seconds", time.perf_counter() - started)
            trial.set_user_attr("peak_allocated_bytes", outcome.peak_allocated_bytes)
            trial.set_user_attr("peak_reserved_bytes", outcome.peak_reserved_bytes)
            trial.set_user_attr("resolved_config_hash", outcome.resolved_config_hash)
            trial.set_user_attr(
                "selected_checkpoint_score", verified.selected_checkpoint_score
            )
            trial.set_user_attr(
                "literal_max_macro_auroc", verified.maximum_observed_score
            )
            trial.set_user_attr("literal_max_epoch", verified.maximum_observed_epoch)
            trial.set_user_attr("manifest_hash", verified.manifest_hash)
            trial.set_user_attr(
                "normalization_file_hash", verified.normalization_file_hash
            )
            trial.set_user_attr("artifact_sha256", dict(verified.artifact_sha256))
            study.tell(trial, outcome.best_macro_auroc)
            _write_study_summary(config, study, plan)
        except Exception as error:
            trial.set_user_attr("wall_seconds", time.perf_counter() - started)
            trial.set_user_attr("failure", f"{type(error).__name__}: {error}"[:1000])
            study.tell(trial, state=TrialState.FAIL, skip_if_finished=True)
            _write_study_summary(config, study, plan)


def run_sweep_study(
    config: SweepConfig,
    *,
    protocol: ExperimentProtocol,
    plan: CandidatePlan,
    resume: bool,
    executor: TrialExecutor = execute_development_trial,
    provenance: Mapping[str, object] | None = None,
) -> tuple[ArchitectureSweepResult, Study]:
    """Run one architecture until all 12 planned candidates are COMPLETE."""

    if protocol.folds_for(FoldRole.TRAIN) != tuple(range(1, 8)):
        raise SweepRunnerError("sweeps require protocol training folds 1-7")
    if protocol.folds_for(FoldRole.MODEL_SELECTION) != (8,):
        raise SweepRunnerError("sweeps require model-selection fold 8")
    captured = dict(provenance or source_provenance(config))
    with _comparison_writer_lock(config.storage.output_root):
        study = _prepare_study(
            config,
            protocol=protocol,
            plan=plan,
            provenance=captured,
            resume=resume,
        )
        for candidate in plan.candidates:
            _execute_candidate(config, study, plan, candidate, protocol, executor)
        _reconcile_trials(config, study, plan, protocol, resume=True)
        return _architecture_result(config, study, plan), study


def _prepare_study(
    config: SweepConfig,
    *,
    protocol: ExperimentProtocol,
    plan: CandidatePlan,
    provenance: Mapping[str, object],
    resume: bool,
) -> Study:
    study = _load_or_create_study(
        config,
        protocol=protocol,
        plan=plan,
        provenance=provenance,
        resume=resume,
    )
    _reconcile_trials(config, study, plan, protocol, resume=resume)
    _write_study_summary(config, study, plan)
    return study


def preflight_equal_budget_sweeps(
    pair: EqualBudgetSweepPair,
    *,
    protocol: ExperimentProtocol,
) -> SweepPreflightResult:
    """Resolve plan/provenance/status without writing files or studies."""

    if protocol.folds_for(FoldRole.TRAIN) != tuple(range(1, 8)):
        raise SweepRunnerError("preflight requires folds 1-7")
    if protocol.folds_for(FoldRole.MODEL_SELECTION) != (8,):
        raise SweepRunnerError("preflight requires fold 8 selection")
    plan = build_candidate_plan(pair)
    plan_path = _candidate_plan_path(pair)
    if plan_path.exists():
        _validate_existing_plan(plan_path, plan)
    elif pair.resnet.storage.sqlite_path.exists():
        raise SweepRunnerError("SQLite storage exists without its candidate plan")
    provenance = source_provenance(pair.resnet)
    return SweepPreflightResult(
        comparison_id=pair.resnet.comparison_id,
        candidate_plan_hash=plan.plan_hash,
        candidate_plan_path=plan_path,
        storage_path=pair.resnet.storage.sqlite_path,
        storage_exists=pair.resnet.storage.sqlite_path.exists(),
        existing_study_names=_existing_study_names(pair.resnet.storage.sqlite_path),
        source_provenance=provenance,
    )


def read_sweep_status(
    pair: EqualBudgetSweepPair,
    *,
    protocol: ExperimentProtocol,
) -> dict[str, object]:
    """Read and reconcile persisted study status without mutating it."""

    preflight = preflight_equal_budget_sweeps(pair, protocol=protocol)
    plan = build_candidate_plan(pair)
    provenance = preflight.source_provenance
    studies: dict[str, object] = {}
    storage = _sqlite_url(pair.resnet.storage.sqlite_path)
    for config in pair.configs:
        if config.study_name not in preflight.existing_study_names:
            studies[config.architecture] = {
                "study_name": config.study_name,
                "exists": False,
                "completed_candidates": 0,
            }
            continue
        study = optuna.load_study(
            study_name=config.study_name,
            storage=storage,
            sampler=EnqueuedPlanOnlySampler(),
        )
        expected = _study_attrs(config, protocol, plan, provenance)
        if any(study.user_attrs.get(key) != value for key, value in expected.items()):
            raise SweepRunnerError("status refused because study provenance drifted")
        # Read-only status rejects unresolved RUNNING/PRUNED state rather than changing it.
        for trial in study.get_trials(deepcopy=False):
            if trial.state == TrialState.PRUNED:
                raise SweepRunnerError("status found PRUNED trial while pruning is disabled")
            candidate_index = _candidate_index(trial)
            _attempt_index(trial)
            fixed = trial.system_attrs.get("fixed_params")
            if fixed != dict(plan.candidates[candidate_index].parameters):
                raise SweepRunnerError("status found trial parameters drifted from plan")
            if trial.state == TrialState.COMPLETE:
                _verified_complete_trial(config, protocol, plan, trial)
        studies[config.architecture] = {
            "study_name": config.study_name,
            "exists": True,
            "payload": _study_payload(config, study, plan),
        }
    return {
        "schema_version": SWEEP_ARTIFACT_SCHEMA_VERSION,
        "comparison_id": pair.resnet.comparison_id,
        "preflight": preflight.to_dict(),
        "studies": studies,
    }


def _final_summary(
    pair: EqualBudgetSweepPair,
    protocol: ExperimentProtocol,
    plan: CandidatePlan,
    plan_path: Path,
    results: tuple[ArchitectureSweepResult, ArchitectureSweepResult],
    studies: tuple[Study, Study],
    provenance: Mapping[str, object],
) -> dict[str, object]:
    best_by_architecture: dict[str, object] = {}
    study_payloads: dict[str, object] = {}
    for config, result, study in zip(pair.configs, results, studies, strict=True):
        study_payloads[config.architecture] = _study_payload(config, study, plan)
        if result.best is not None:
            best = _best_trial(_completed_candidates(study, plan))
            best_by_architecture[config.architecture] = _trial_record(best)
    return {
        "schema_version": SWEEP_ARTIFACT_SCHEMA_VERSION,
        "comparison_id": pair.resnet.comparison_id,
        "protocol_hash": protocol.protocol_hash,
        "candidate_plan_path": str(plan_path),
        "candidate_plan_hash": plan.plan_hash,
        "equal_candidate_plan_verified": True,
        "paired_execution_policy": _PAIRED_EXECUTION_POLICY,
        "paired_execution_order": _paired_execution_order(plan),
        "required_complete_candidates_per_architecture": 12,
        "all_candidate_budgets_complete": True,
        "objective": pair.resnet.objective.to_dict(),
        "seed_policy": pair.resnet.seed_policy.to_dict(),
        "tie_break": pair.resnet.tie_break.to_dict(),
        "failure_policy": pair.resnet.failure_policy.to_dict(),
        "source_provenance": dict(provenance),
        "best_by_architecture": best_by_architecture,
        "studies": study_payloads,
        "warning": "Fold-8 development selection only; not a final-test result.",
    }


def _paired_execution_order(plan: CandidatePlan) -> list[dict[str, object]]:
    return [
        {
            "candidate_index": candidate.index,
            "architecture_order": (
                ["resnet1d", "ecg_transformer"]
                if candidate.index % 2 == 0
                else ["ecg_transformer", "resnet1d"]
            ),
        }
        for candidate in plan.candidates
    ]


def run_equal_budget_sweeps(
    pair: EqualBudgetSweepPair,
    *,
    protocol: ExperimentProtocol,
    resume: bool = False,
    executor: TrialExecutor = execute_development_trial,
) -> EqualBudgetSweepResult:
    """Run the persisted paired plan and select only after 12+12 completions."""

    if protocol.folds_for(FoldRole.TRAIN) != tuple(range(1, 8)):
        raise SweepRunnerError("sweeps require protocol training folds 1-7")
    if protocol.folds_for(FoldRole.MODEL_SELECTION) != (8,):
        raise SweepRunnerError("sweeps require model-selection fold 8")
    plan = build_candidate_plan(pair)
    provenance = source_provenance(pair.resnet)
    with _comparison_writer_lock(pair.resnet.storage.output_root):
        plan_path = _persist_candidate_plan(pair, plan, resume=resume)
        studies_by_architecture: dict[str, Study] = {
            config.architecture: _prepare_study(
                config,
                protocol=protocol,
                plan=plan,
                provenance=provenance,
                resume=resume,
            )
            for config in pair.configs
        }
        configs_by_architecture: dict[str, SweepConfig] = {
            config.architecture: config for config in pair.configs
        }
        for candidate in plan.candidates:
            architecture_order = (
                ("resnet1d", "ecg_transformer")
                if candidate.index % 2 == 0
                else ("ecg_transformer", "resnet1d")
            )
            for architecture in architecture_order:
                config = configs_by_architecture[architecture]
                study = studies_by_architecture[architecture]
                _execute_candidate(
                    config,
                    study,
                    plan,
                    candidate,
                    protocol,
                    executor,
                )
        for config in pair.configs:
            _reconcile_trials(
                config,
                studies_by_architecture[config.architecture],
                plan,
                protocol,
                resume=True,
            )
        results = cast(
            tuple[ArchitectureSweepResult, ArchitectureSweepResult],
            tuple(
                _architecture_result(
                    config,
                    studies_by_architecture[config.architecture],
                    plan,
                    release_selection=True,
                )
                for config in pair.configs
            ),
        )
        studies = cast(
            tuple[Study, Study],
            tuple(studies_by_architecture[config.architecture] for config in pair.configs),
        )
        if not all(result.budget_complete for result in results):
            raise SweepRunnerError(
                "selection is forbidden before both 12-candidate budgets complete"
            )
        if any(result.best is None for result in results):
            raise SweepRunnerError(
                "every architecture requires a selectable completed candidate"
            )
        best_by_architecture = {
            result.architecture: cast(BestCandidateResult, result.best)
            for result in results
        }
        summary_path = pair.resnet.storage.output_root / "sweep_summary.json"
        _atomic_json(
            summary_path,
            _final_summary(
                pair,
                protocol,
                plan,
                plan_path,
                results,
                studies,
                provenance,
            ),
        )
        return EqualBudgetSweepResult(
            comparison_id=pair.resnet.comparison_id,
            candidate_plan_path=plan_path,
            summary_path=summary_path,
            studies=results,
            best_by_architecture=best_by_architecture,
        )
