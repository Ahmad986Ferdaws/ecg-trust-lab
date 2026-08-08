"""Immutable paired multi-seed confirmation on PTB-XL development folds.

This module starts from the completed schema-v2 paired sweep summary.  It
reuses each architecture's seed-2026 winning run, trains only the fixed 2027
and 2028 confirmation seeds, and exports an integrity-checked fold-8
prediction artifact for every member.  Fold 9 and fold 10 are deliberately not
representable by this API.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import socket
import subprocess
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, cast

import optuna
import psutil  # type: ignore[import-untyped]
import scipy  # type: ignore[import-untyped]
import torch

from ecg_trust.experiment_config import DevelopmentExperimentConfig
from ecg_trust.experiment_runner import DevelopmentRunResult, run_development_experiment
from ecg_trust.prediction_export import (
    PredictionExportRequest,
    PredictionExportResult,
    export_checkpoint_predictions,
)
from ecg_trust.predictions import (
    PredictionArtifact,
    assert_prediction_artifacts_aligned,
    load_prediction_artifact,
)
from ecg_trust.protocol import (
    LABEL_ORDER,
    MODEL_SELECTION_FOLDS,
    TRAIN_FOLDS,
    ExperimentProtocol,
    FoldRole,
)

MULTISEED_PLAN_SCHEMA_VERSION = 1
MULTISEED_PLAN_TYPE = "ecg_trust.multiseed_confirmation_plan"
MEMBER_PLAN_TYPE = "ecg_trust.multiseed_member_plan"
ATTEMPT_PLAN_TYPE = "ecg_trust.multiseed_attempt_plan"
ATTEMPT_STATUS_TYPE = "ecg_trust.multiseed_attempt_status"
MEMBER_COMPLETION_TYPE = "ecg_trust.multiseed_member_completion"
CONFIRMATION_SEEDS: tuple[int, ...] = (2026, 2027, 2028)
ARCHITECTURES: tuple[str, ...] = ("resnet1d", "ecg_transformer")
MAX_ATTEMPTS = 3
_SWEEP_SCHEMA_VERSION = 2
_HASH_PREFIX = "sha256:"
_SOURCE_FILENAMES = (
    "run_metadata.json",
    "resolved_config.json",
    "history.jsonl",
    "best.ckpt",
    "last.ckpt",
    "protocol.json",
)
_SCIENTIFIC_KERNEL_PATHS = (
    "configs/protocol.yaml",
    "src/ecg_trust/constants.py",
    "src/ecg_trust/protocol.py",
    "src/ecg_trust/data",
    "src/ecg_trust/experiment_config.py",
    "src/ecg_trust/experiment_runner.py",
    "src/ecg_trust/training.py",
    "src/ecg_trust/models",
    "src/ecg_trust/evaluation.py",
    "src/ecg_trust/predictions.py",
    "src/ecg_trust/prediction_export.py",
)
_RUNTIME_IDENTITY_KEYS = {
    "python",
    "implementation",
    "platform",
    "optuna",
    "scipy",
    "torch",
}
_REVISION_PROVENANCE_TYPE = "ecg_trust.multiseed_revision_provenance"
_EMPTY_SHA256 = _HASH_PREFIX + hashlib.sha256(b"").hexdigest()
_ROOT_PLAN_KEYS = {
    "schema_version",
    "artifact_type",
    "comparison_id",
    "sweep_summary_path",
    "sweep_summary_sha256",
    "candidate_plan_path",
    "candidate_plan_file_sha256",
    "candidate_plan_hash",
    "protocol_hash",
    "train_folds",
    "model_selection_folds",
    "architectures",
    "seeds",
    "max_attempts_per_trained_member",
    "execution_order",
    "revision_provenance",
    "revision_provenance_sha256",
    "members",
    "artifact_sha256",
}
_MEMBER_COMPLETION_KEYS = {
    "schema_version",
    "artifact_type",
    "comparison_id",
    "architecture",
    "seed",
    "status",
    "member_plan_path",
    "member_plan_sha256",
    "run_dir",
    "run_metadata_path",
    "run_metadata_sha256",
    "resolved_config_path",
    "resolved_config_sha256",
    "history_path",
    "history_sha256",
    "best_checkpoint_path",
    "best_checkpoint_sha256",
    "config_hash",
    "protocol_hash",
    "manifest_hash",
    "normalization_sha256",
    "best_epoch",
    "best_validation_macro_auroc",
    "completed_epochs",
    "prediction_path",
    "prediction_npz_sha256",
    "prediction_json_path",
    "prediction_artifact_sha256",
    "artifact_sha256",
}


class MultiSeedRunnerError(RuntimeError):
    """Raised when confirmation cannot preserve its immutable protocol."""


class DevelopmentExecutor(Protocol):
    """Callable used to execute one folds-1-7/fold-8 experiment."""

    def __call__(
        self,
        config: DevelopmentExperimentConfig,
        *,
        protocol: ExperimentProtocol,
    ) -> DevelopmentRunResult: ...


class PredictionExporter(Protocol):
    """Callable used to export one model-selection prediction artifact."""

    def __call__(
        self,
        request: PredictionExportRequest,
        *,
        protocol: ExperimentProtocol,
    ) -> PredictionExportResult: ...


@dataclass(frozen=True, slots=True)
class WinnerSource:
    """Integrity-verified seed-2026 architecture winner."""

    architecture: str
    candidate_index: int
    trial_number: int
    run_dir: Path
    experiment_config: DevelopmentExperimentConfig
    scientific_config_sha256: str
    best_epoch: int
    best_macro_auroc: float
    completed_epochs: int
    artifact_sha256: Mapping[str, str]
    manifest_sha256: str
    normalization_sha256: str


@dataclass(frozen=True, slots=True)
class VerifiedRun:
    """A complete development run ready for fold-8 prediction export."""

    run_dir: Path
    run_name: str
    seed: int
    resolved_config_hash: str
    manifest_sha256: str
    normalization_sha256: str
    best_epoch: int
    best_macro_auroc: float
    completed_epochs: int
    paths: Mapping[str, Path]
    hashes: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class MultiSeedPlanResult:
    """Paths and identity of one persisted confirmation plan."""

    comparison_id: str
    confirmation_dir: Path
    plan_path: Path
    plan_sha256: str
    member_plan_paths: tuple[Path, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "comparison_id": self.comparison_id,
            "confirmation_dir": str(self.confirmation_dir),
            "plan_path": str(self.plan_path),
            "plan_sha256": self.plan_sha256,
            "member_plan_paths": [str(path) for path in self.member_plan_paths],
        }


@dataclass(frozen=True, slots=True)
class MultiSeedRunResult:
    """Completed six-member confirmation output."""

    comparison_id: str
    plan_path: Path
    completion_paths: tuple[Path, ...]
    prediction_paths: tuple[Path, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "comparison_id": self.comparison_id,
            "plan_path": str(self.plan_path),
            "completion_paths": [str(path) for path in self.completion_paths],
            "prediction_paths": [str(path) for path in self.prediction_paths],
            "complete_members": len(self.completion_paths),
        }


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise MultiSeedRunnerError(f"{context} must be a string-keyed mapping")
    return cast(Mapping[str, object], value)


def _sequence(value: object, context: str) -> Sequence[object]:
    if not isinstance(value, list):
        raise MultiSeedRunnerError(f"{context} must be a list")
    return cast(Sequence[object], value)


def _keys(value: Mapping[str, object], *, required: set[str], context: str) -> None:
    missing = sorted(required.difference(value))
    unexpected = sorted(set(value).difference(required))
    if missing or unexpected:
        raise MultiSeedRunnerError(
            f"{context} has invalid keys; missing={missing}, unexpected={unexpected}"
        )


def _string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MultiSeedRunnerError(f"{context} must be a non-empty string")
    return value


def _integer(value: object, context: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise MultiSeedRunnerError(f"{context} must be an integer >= {minimum}")
    return value


def _number(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MultiSeedRunnerError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise MultiSeedRunnerError(f"{context} must be finite")
    return result


def _boolean(value: object, context: str) -> bool:
    if not isinstance(value, bool):
        raise MultiSeedRunnerError(f"{context} must be boolean")
    return value


def _hash(value: object, context: str) -> str:
    text = _string(value, context)
    digest = text[len(_HASH_PREFIX) :] if text.startswith(_HASH_PREFIX) else text
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise MultiSeedRunnerError(f"{context} must be a lower-case SHA-256 value")
    return _HASH_PREFIX + digest


def _path(value: object, context: str) -> Path:
    path = Path(_string(value, context))
    if not path.is_absolute():
        raise MultiSeedRunnerError(f"{context} must be an absolute path")
    return path.resolve()


def _canonical_bytes(value: Mapping[str, object]) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise MultiSeedRunnerError("artifact must contain finite JSON values") from error


def _canonical_hash(value: Mapping[str, object]) -> str:
    return _HASH_PREFIX + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise MultiSeedRunnerError(f"could not hash {path}: {error}") from error
    return _HASH_PREFIX + digest.hexdigest()


def _read_json(path: Path, context: str) -> Mapping[str, object]:
    if not path.is_file():
        raise MultiSeedRunnerError(f"{context} is missing: {path}")
    if path.stat().st_size > 100_000_000:
        raise MultiSeedRunnerError(f"{context} is unreasonably large")
    try:
        decoded: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise MultiSeedRunnerError(f"could not decode {context}: {error}") from error
    return _mapping(decoded, context)


def _hashed_payload(payload: Mapping[str, object]) -> dict[str, object]:
    if "artifact_sha256" in payload:
        raise MultiSeedRunnerError("unhashed payload unexpectedly contains artifact_sha256")
    result = dict(payload)
    result["artifact_sha256"] = _canonical_hash(payload)
    return result


def _verify_self_hash(payload: Mapping[str, object], context: str) -> str:
    stored = _hash(payload.get("artifact_sha256"), f"{context}.artifact_sha256")
    unhashed = dict(payload)
    del unhashed["artifact_sha256"]
    observed = _canonical_hash(unhashed)
    if stored != observed:
        raise MultiSeedRunnerError(f"{context} artifact SHA-256 mismatch")
    return stored


def _git_bytes(root: Path, arguments: Sequence[str], context: str) -> bytes:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=False,
            capture_output=True,
        )
    except OSError as error:
        raise MultiSeedRunnerError(f"could not run git for {context}: {error}") from error
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise MultiSeedRunnerError(
            f"git {context} failed: {detail or f'exit {completed.returncode}'}"
        )
    return completed.stdout


def _git_text(root: Path, arguments: Sequence[str], context: str) -> str:
    try:
        return _git_bytes(root, arguments, context).decode("utf-8").strip()
    except UnicodeDecodeError as error:
        raise MultiSeedRunnerError(f"git {context} returned non-UTF-8 text") from error


def _git_object(value: object, context: str) -> str:
    text = _string(value, context)
    if len(text) not in {40, 64} or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise MultiSeedRunnerError(f"{context} must be a full lower-case Git object ID")
    return text


def _source_tree_hash(root: Path) -> str:
    """Match the sweep's complete source/config/lock snapshot algorithm."""

    candidates: set[Path] = set()
    for directory, pattern in (
        ("src", "*.py"),
        ("scripts", "*.py"),
        ("configs", "*.yaml"),
    ):
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
    return _HASH_PREFIX + digest.hexdigest()


def _current_runtime_identity() -> dict[str, object]:
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "optuna": str(optuna.__version__),
        "scipy": str(scipy.__version__),
        "torch": str(torch.__version__),
    }


def _validated_sweep_source(value: object) -> Mapping[str, object]:
    source = _mapping(value, "sweep source provenance")
    required = {
        "project_root",
        "git_root",
        "git_head",
        "git_dirty",
        "git_status_sha256",
        "git_unavailable",
        "source_tree_sha256",
        "dependency_lock_sha256",
        "manifest_path",
        "manifest_sha256",
        "normalization_path",
        "normalization_sha256",
        "runtime_identity",
        "runtime_identity_sha256",
    }
    _keys(source, required=required, context="sweep source provenance")
    if source.get("git_unavailable") is not False or source.get("git_dirty") is not False:
        raise MultiSeedRunnerError(
            "sweep source must be a clean, Git-addressable commit before confirmation"
        )
    project_root = _path(source["project_root"], "sweep project root")
    git_root = _path(source["git_root"], "sweep git root")
    if project_root != git_root:
        raise MultiSeedRunnerError("sweep project root and Git root must be identical")
    _git_object(source["git_head"], "sweep Git head")
    if _hash(source["git_status_sha256"], "sweep Git status hash") != _EMPTY_SHA256:
        raise MultiSeedRunnerError("clean sweep must bind the empty Git status hash")
    for field in (
        "source_tree_sha256",
        "dependency_lock_sha256",
        "manifest_sha256",
        "normalization_sha256",
    ):
        _hash(source[field], f"sweep {field}")
    runtime = _mapping(source["runtime_identity"], "sweep runtime identity")
    _keys(runtime, required=_RUNTIME_IDENTITY_KEYS, context="sweep runtime identity")
    if _canonical_hash(runtime) != _hash(
        source["runtime_identity_sha256"], "sweep runtime identity hash"
    ):
        raise MultiSeedRunnerError("sweep runtime identity hash mismatch")
    _path(source["manifest_path"], "sweep manifest path")
    _path(source["normalization_path"], "sweep normalization path")
    return source


def _git_tree(root: Path, revision: str, context: str) -> str:
    resolved = _git_text(
        root,
        ["rev-parse", "--verify", f"{revision}^{{tree}}"],
        f"resolve {context} tree",
    )
    return _git_object(resolved, f"{context} Git tree")


def _require_ancestor(root: Path, ancestor: str, descendant: str) -> None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "merge-base", "--is-ancestor", ancestor, descendant],
            check=False,
            capture_output=True,
        )
    except OSError as error:
        raise MultiSeedRunnerError(f"could not verify Git ancestry: {error}") from error
    if completed.returncode == 1:
        raise MultiSeedRunnerError("confirmation commit B is not downstream of sweep commit A")
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise MultiSeedRunnerError(f"could not verify Git ancestry: {detail}")


def _current_source_snapshot(source: Mapping[str, object]) -> dict[str, object]:
    project_root = _path(source["project_root"], "sweep project root")
    observed_root = Path(
        _git_text(project_root, ["rev-parse", "--show-toplevel"], "find repository root")
    ).resolve()
    if observed_root != project_root:
        raise MultiSeedRunnerError("current Git root differs from the sweep repository")
    status = _git_bytes(
        project_root,
        ["status", "--porcelain=v1", "--untracked-files=all"],
        "read working-tree status",
    ).strip()
    if status:
        raise MultiSeedRunnerError(
            "confirmation plan requires a fully clean commit-B working tree"
        )
    head = _git_object(
        _git_text(project_root, ["rev-parse", "HEAD"], "resolve confirmation HEAD"),
        "confirmation Git head",
    )
    runtime = _current_runtime_identity()
    sweep_runtime = _mapping(source["runtime_identity"], "sweep runtime identity")
    if runtime != dict(sweep_runtime):
        mismatches = sorted(
            key for key in _RUNTIME_IDENTITY_KEYS if runtime.get(key) != sweep_runtime.get(key)
        )
        raise MultiSeedRunnerError(
            "confirmation runtime differs from sweep runtime keys: " + ", ".join(mismatches)
        )
    manifest = _path(source["manifest_path"], "sweep manifest path")
    normalization = _path(source["normalization_path"], "sweep normalization path")
    manifest_hash = _file_sha256(manifest)
    normalization_hash = _file_sha256(normalization)
    if manifest_hash != source.get("manifest_sha256"):
        raise MultiSeedRunnerError("confirmation manifest differs from sweep A")
    if normalization_hash != source.get("normalization_sha256"):
        raise MultiSeedRunnerError("confirmation normalization differs from sweep A")
    lockfile = project_root / "uv.lock"
    if not lockfile.is_file():
        raise MultiSeedRunnerError("confirmation dependency lock is missing")
    return {
        "project_root": str(project_root),
        "git_root": str(project_root),
        "git_head": head,
        "git_dirty": False,
        "git_status_sha256": _EMPTY_SHA256,
        "git_unavailable": False,
        "source_tree_sha256": _source_tree_hash(project_root),
        "dependency_lock_sha256": _file_sha256(lockfile),
        "manifest_path": str(manifest),
        "manifest_sha256": manifest_hash,
        "normalization_path": str(normalization),
        "normalization_sha256": normalization_hash,
        "runtime_identity": runtime,
        "runtime_identity_sha256": _canonical_hash(runtime),
    }


def _changed_paths(root: Path, sweep_revision: str, execution_revision: str) -> list[str]:
    raw = _git_bytes(
        root,
        ["diff", "--name-only", "-z", sweep_revision, execution_revision, "--"],
        "enumerate A-to-B changed paths",
    )
    try:
        paths = [part.decode("utf-8") for part in raw.split(b"\0") if part]
    except UnicodeDecodeError as error:
        raise MultiSeedRunnerError("Git changed paths are not UTF-8") from error
    if paths != sorted(set(paths)):
        raise MultiSeedRunnerError("Git changed-path enumeration is not canonical")
    return paths


def _build_revision_provenance(summary: Mapping[str, object]) -> dict[str, object]:
    source = _validated_sweep_source(summary.get("source_provenance"))
    root = _path(source["git_root"], "sweep Git root")
    sweep_revision = _git_object(source["git_head"], "sweep Git head")
    sweep_tree = _git_tree(root, sweep_revision, "sweep")
    execution_source = _current_source_snapshot(source)
    execution_revision = _git_object(
        execution_source["git_head"], "confirmation Git head"
    )
    _require_ancestor(root, sweep_revision, execution_revision)
    execution_tree = _git_tree(root, execution_revision, "confirmation")
    kernel_diff = _git_bytes(
        root,
        [
            "diff",
            "--no-ext-diff",
            "--binary",
            sweep_revision,
            execution_revision,
            "--",
            *_SCIENTIFIC_KERNEL_PATHS,
        ],
        "compare the scientific kernel from A to B",
    )
    if kernel_diff:
        changed = _git_text(
            root,
            [
                "diff",
                "--name-only",
                sweep_revision,
                execution_revision,
                "--",
                *_SCIENTIFIC_KERNEL_PATHS,
            ],
            "list changed scientific-kernel paths",
        )
        raise MultiSeedRunnerError(
            "scientific kernel changed between sweep A and confirmation B: "
            + changed.replace("\n", ", ")
        )
    allowed = _changed_paths(root, sweep_revision, execution_revision)
    kernel = {
        "policy": "ptbxl_development_training_kernel_v1",
        "sweep_revision": sweep_revision,
        "execution_revision": execution_revision,
        "paths": list(_SCIENTIFIC_KERNEL_PATHS),
        "paths_sha256": _canonical_hash({"paths": list(_SCIENTIFIC_KERNEL_PATHS)}),
        "git_diff_sha256": _HASH_PREFIX + hashlib.sha256(kernel_diff).hexdigest(),
        "unchanged": True,
        "allowed_changed_paths": allowed,
        "allowed_changed_paths_sha256": _canonical_hash({"paths": allowed}),
    }
    return _hashed_payload(
        {
            "schema_version": 1,
            "artifact_type": _REVISION_PROVENANCE_TYPE,
            "sweep_snapshot": {
                "git_tree": sweep_tree,
                "source_provenance": dict(source),
            },
            "execution_snapshot": {
                "git_tree": execution_tree,
                "source_provenance": execution_source,
            },
            "scientific_kernel": kernel,
        }
    )


def _validate_revision_provenance(
    value: object,
    *,
    expected_sweep_source: Mapping[str, object] | None = None,
) -> Mapping[str, object]:
    provenance = _mapping(value, "revision provenance")
    _keys(
        provenance,
        required={
            "schema_version",
            "artifact_type",
            "sweep_snapshot",
            "execution_snapshot",
            "scientific_kernel",
            "artifact_sha256",
        },
        context="revision provenance",
    )
    _verify_self_hash(provenance, "revision provenance")
    if provenance.get("schema_version") != 1:
        raise MultiSeedRunnerError("unsupported revision-provenance schema")
    if provenance.get("artifact_type") != _REVISION_PROVENANCE_TYPE:
        raise MultiSeedRunnerError("unexpected revision-provenance artifact type")
    sweep = _mapping(provenance["sweep_snapshot"], "sweep revision snapshot")
    execution = _mapping(
        provenance["execution_snapshot"], "execution revision snapshot"
    )
    for snapshot, context in (
        (sweep, "sweep revision snapshot"),
        (execution, "execution revision snapshot"),
    ):
        _keys(snapshot, required={"git_tree", "source_provenance"}, context=context)
        _git_object(snapshot["git_tree"], f"{context} Git tree")
        _validated_sweep_source(snapshot["source_provenance"])
    sweep_source = _validated_sweep_source(sweep["source_provenance"])
    execution_source = _validated_sweep_source(execution["source_provenance"])
    if expected_sweep_source is not None and dict(sweep_source) != dict(
        expected_sweep_source
    ):
        raise MultiSeedRunnerError("revision proof sweep snapshot differs from sweep summary")
    kernel = _mapping(provenance["scientific_kernel"], "scientific kernel proof")
    _keys(
        kernel,
        required={
            "policy",
            "sweep_revision",
            "execution_revision",
            "paths",
            "paths_sha256",
            "git_diff_sha256",
            "unchanged",
            "allowed_changed_paths",
            "allowed_changed_paths_sha256",
        },
        context="scientific kernel proof",
    )
    if kernel.get("policy") != "ptbxl_development_training_kernel_v1":
        raise MultiSeedRunnerError("scientific-kernel path policy changed")
    paths = tuple(
        _string(item, "scientific-kernel path")
        for item in _sequence(kernel["paths"], "scientific-kernel paths")
    )
    if paths != _SCIENTIFIC_KERNEL_PATHS:
        raise MultiSeedRunnerError("scientific-kernel path set changed")
    if kernel.get("paths_sha256") != _canonical_hash({"paths": list(paths)}):
        raise MultiSeedRunnerError("scientific-kernel path-set hash mismatch")
    sweep_revision = _git_object(kernel["sweep_revision"], "kernel sweep revision")
    execution_revision = _git_object(
        kernel["execution_revision"], "kernel execution revision"
    )
    if sweep_revision != sweep_source.get("git_head"):
        raise MultiSeedRunnerError("kernel sweep revision differs from snapshot A")
    if execution_revision != execution_source.get("git_head"):
        raise MultiSeedRunnerError("kernel execution revision differs from snapshot B")
    if kernel.get("unchanged") is not True or kernel.get("git_diff_sha256") != _EMPTY_SHA256:
        raise MultiSeedRunnerError("scientific-kernel proof is not an empty A-to-B diff")
    allowed = [
        _string(item, "allowed changed path")
        for item in _sequence(kernel["allowed_changed_paths"], "allowed changed paths")
    ]
    if allowed != sorted(set(allowed)) or any(
        Path(item).is_absolute() or "\\" in item or ".." in Path(item).parts
        for item in allowed
    ):
        raise MultiSeedRunnerError("allowed A-to-B changed paths are not canonical")
    if kernel.get("allowed_changed_paths_sha256") != _canonical_hash(
        {"paths": allowed}
    ):
        raise MultiSeedRunnerError("allowed changed-path hash mismatch")
    return provenance


def _verify_revision_provenance(
    value: object,
    *,
    expected_sweep_source: Mapping[str, object] | None = None,
) -> Mapping[str, object]:
    provenance = _validate_revision_provenance(
        value, expected_sweep_source=expected_sweep_source
    )
    sweep = _mapping(provenance["sweep_snapshot"], "sweep revision snapshot")
    sweep_source = _mapping(sweep["source_provenance"], "sweep source provenance")
    rebuilt = _build_revision_provenance({"source_provenance": sweep_source})
    if provenance != rebuilt:
        raise MultiSeedRunnerError(
            "current commit/runtime no longer matches the sealed A-to-B revision proof"
        )
    return provenance


def _atomic_write_new(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise MultiSeedRunnerError(f"immutable artifact already exists: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        serialized = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(serialized)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        with suppress(OSError):
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
        raise


def _write_or_verify(path: Path, payload: Mapping[str, object], context: str) -> None:
    if path.exists():
        observed = _read_json(path, context)
        if observed != payload:
            raise MultiSeedRunnerError(f"persisted {context} differs from requested plan")
        return
    _atomic_write_new(path, payload)


def _base_experiment_mapping(raw: Mapping[str, object]) -> dict[str, object]:
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
    model = _mapping(raw["model"], "resolved model")
    return {
        "schema_version": raw["schema_version"],
        "run_name": raw["run_name"],
        "folds": raw["folds"],
        "data": raw["data"],
        "model": {
            "architecture": model.get("architecture"),
            "preset": model.get("preset"),
        },
        "loader": raw["loader"],
        "optimization": raw["optimization"],
        "runtime": raw["runtime"],
        "output": raw["output"],
    }


def _scientific_payload(config: DevelopmentExperimentConfig) -> dict[str, object]:
    resolved = config.to_resolved_dict()
    runtime = _mapping(resolved["runtime"], "experiment runtime")
    return {
        "schema_version": resolved["schema_version"],
        "folds": resolved["folds"],
        "data": resolved["data"],
        "model": resolved["model"],
        "loader": resolved["loader"],
        "optimization": resolved["optimization"],
        "runtime": {"device": runtime["device"], "bf16": runtime["bf16"]},
    }


def _load_resolved_experiment(path: Path) -> tuple[DevelopmentExperimentConfig, str]:
    wrapper = _read_json(path, "resolved config")
    _keys(wrapper, required={"config", "config_hash"}, context="resolved config")
    raw = _mapping(wrapper["config"], "resolved config payload")
    computed = _canonical_hash(raw)
    stored = _hash(wrapper["config_hash"], "resolved config hash")
    if computed != stored:
        raise MultiSeedRunnerError("resolved config content hash mismatch")
    try:
        config = DevelopmentExperimentConfig.from_mapping(
            _base_experiment_mapping(raw), base_dir=path.parent
        )
    except (TypeError, ValueError) as error:
        raise MultiSeedRunnerError(f"invalid resolved experiment config: {error}") from error
    if config.train_folds != TRAIN_FOLDS or config.validation_folds != MODEL_SELECTION_FOLDS:
        raise MultiSeedRunnerError("confirmation experiment may use only folds 1-7 and fold 8")
    return config, stored


def _verify_candidate_plan(
    path: Path,
    *,
    expected_comparison_id: str,
    expected_plan_hash: str,
) -> str:
    payload = _read_json(path, "candidate plan")
    required = {
        "schema_version",
        "comparison_id",
        "algorithm",
        "algorithm_version",
        "scipy_version",
        "design_seed",
        "dimensions",
        "candidates",
        "plan_hash",
    }
    _keys(payload, required=required, context="candidate plan")
    if _integer(payload["schema_version"], "candidate plan schema", minimum=1) != 2:
        raise MultiSeedRunnerError("candidate plan must use schema version 2")
    if _string(payload["comparison_id"], "candidate plan comparison") != expected_comparison_id:
        raise MultiSeedRunnerError("candidate plan comparison_id mismatch")
    stored = _hash(payload["plan_hash"], "candidate plan hash")
    unhashed = dict(payload)
    del unhashed["plan_hash"]
    if _canonical_hash(unhashed) != stored or stored != expected_plan_hash:
        raise MultiSeedRunnerError("candidate plan content hash mismatch")
    candidates = _sequence(payload["candidates"], "candidate plan candidates")
    if len(candidates) != 12:
        raise MultiSeedRunnerError("candidate plan must contain exactly 12 candidates")
    return _file_sha256(path)


def _trial_sort_key(trial: Mapping[str, object]) -> tuple[float, int, int, int]:
    return (
        _number(
            trial.get("best_fold8_uncalibrated_macro_roc_auc"),
            "trial objective",
        ),
        -_integer(trial.get("completed_epochs"), "trial completed_epochs", minimum=1),
        -_integer(trial.get("candidate_index"), "trial candidate_index"),
        -_integer(trial.get("trial_number"), "trial number"),
    )


def _verify_study_winner(
    study: Mapping[str, object],
    winner: Mapping[str, object],
    *,
    architecture: str,
    comparison_id: str,
    candidate_plan_hash: str,
) -> None:
    required = {
        "schema_version",
        "comparison_id",
        "architecture",
        "study_name",
        "candidate_plan_hash",
        "sweep_config_hash",
        "base_experiment_config_hash",
        "objective",
        "seed_policy",
        "tie_break",
        "failure_policy",
        "required_complete_candidates",
        "completed_candidates",
        "budget_complete",
        "attempt_counts",
        "selection_released",
        "best_candidate",
        "study_user_attrs",
        "attempts",
    }
    _keys(study, required=required, context=f"{architecture} study")
    expected_scalars: dict[str, object] = {
        "schema_version": 2,
        "comparison_id": comparison_id,
        "architecture": architecture,
        "candidate_plan_hash": candidate_plan_hash,
        "required_complete_candidates": 12,
        "completed_candidates": 12,
        "budget_complete": True,
        "selection_released": False,
        "best_candidate": None,
    }
    for field, expected in expected_scalars.items():
        if study.get(field) != expected:
            raise MultiSeedRunnerError(f"{architecture} study {field} is invalid")
    attempts = [_mapping(item, f"{architecture} attempt") for item in _sequence(
        study["attempts"], f"{architecture} attempts"
    )]
    complete = [attempt for attempt in attempts if attempt.get("state") == "COMPLETE"]
    if len(complete) != 12:
        raise MultiSeedRunnerError(f"{architecture} study must have 12 COMPLETE attempts")
    candidate_indices = {
        _integer(item.get("candidate_index"), "candidate index") for item in complete
    }
    if candidate_indices != set(range(12)):
        raise MultiSeedRunnerError(f"{architecture} COMPLETE candidates are not exactly 0-11")
    selected = max(complete, key=_trial_sort_key)
    if dict(selected) != dict(winner):
        raise MultiSeedRunnerError(f"{architecture} winner is not the deterministic best trial")


def _checkpoint_mapping(path: Path) -> Mapping[str, object]:
    try:
        decoded: object = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise MultiSeedRunnerError(f"could not inspect checkpoint {path}: {error}") from error
    return _mapping(decoded, f"checkpoint {path}")


def _verify_checkpoint(
    path: Path,
    *,
    epoch: int,
    score: float,
    config_hash: str,
    protocol_hash: str,
    manifest_sha256: str,
) -> None:
    checkpoint = _checkpoint_mapping(path)
    if checkpoint.get("epoch") != epoch:
        raise MultiSeedRunnerError("best checkpoint epoch mismatch")
    comparisons: dict[str, object] = {
        "config_hash": config_hash,
        "protocol_hash": protocol_hash,
    }
    for field, expected in comparisons.items():
        if checkpoint.get(field) != expected:
            raise MultiSeedRunnerError(f"best checkpoint {field} mismatch")
    if _hash(checkpoint.get("manifest_hash"), "checkpoint manifest hash") != manifest_sha256:
        raise MultiSeedRunnerError("best checkpoint manifest hash mismatch")
    stored_config = _mapping(checkpoint.get("config"), "checkpoint config")
    if _canonical_hash(stored_config) != config_hash:
        raise MultiSeedRunnerError("best checkpoint embedded config hash mismatch")
    stopper = _mapping(
        checkpoint.get("early_stopping_state_dict"), "checkpoint early-stopping state"
    )
    if stopper.get("mode") != "max" or stopper.get("best_epoch") != epoch:
        raise MultiSeedRunnerError("best checkpoint early-stopping selection mismatch")
    stored_score = _number(stopper.get("best_score"), "checkpoint best score")
    if stored_score != score:
        raise MultiSeedRunnerError("best checkpoint score mismatch")


def _read_history(
    path: Path,
    *,
    completed_epochs: int,
    best_epoch: int,
    best_score: float,
) -> None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise MultiSeedRunnerError(f"could not read history {path}: {error}") from error
    if len(lines) != completed_epochs:
        raise MultiSeedRunnerError("history length does not match completed_epochs")
    selected: Mapping[str, object] | None = None
    for expected_epoch, line in enumerate(lines):
        try:
            decoded: object = json.loads(line)
        except json.JSONDecodeError as error:
            raise MultiSeedRunnerError(f"history line is invalid JSON: {error}") from error
        record = _mapping(decoded, "history record")
        if record.get("epoch") != expected_epoch:
            raise MultiSeedRunnerError("history epochs must be consecutive from zero")
        if expected_epoch == best_epoch:
            selected = record
    if selected is None or selected.get("improved") is not True:
        raise MultiSeedRunnerError("selected history row is absent or not improved")
    if _number(selected.get("validation_macro_auroc"), "history selected score") != best_score:
        raise MultiSeedRunnerError("history selected score mismatch")
    metrics = _mapping(selected.get("validation_metrics"), "history validation metrics")
    if tuple(_sequence(metrics.get("label_order"), "history label order")) != LABEL_ORDER:
        raise MultiSeedRunnerError("history label order is not canonical")
    macro = _mapping(metrics.get("macro"), "history macro metrics")
    if macro.get("roc_auc_labels") != len(LABEL_ORDER):
        raise MultiSeedRunnerError("history objective does not define all five labels")
    if _number(macro.get("roc_auc"), "history macro ROC-AUC") != best_score:
        raise MultiSeedRunnerError("history macro ROC-AUC mismatch")
    per_label = _sequence(metrics.get("per_label"), "history per-label metrics")
    if len(per_label) != len(LABEL_ORDER):
        raise MultiSeedRunnerError("history must contain five per-label metrics")
    for index, item in enumerate(per_label):
        metric = _mapping(item, "history per-label metric")
        if metric.get("label") != LABEL_ORDER[index] or metric.get("roc_auc") is None:
            raise MultiSeedRunnerError("history contains undefined or reordered label AUROC")


def _source_artifact_paths(run_dir: Path) -> dict[str, Path]:
    return {name: run_dir / name for name in _SOURCE_FILENAMES}


def _verify_winner_source(
    winner: Mapping[str, object],
    *,
    architecture: str,
    protocol: ExperimentProtocol,
    manifest_sha256: str,
    normalization_sha256: str,
) -> WinnerSource:
    if winner.get("state") != "COMPLETE" or winner.get("failure") is not None:
        raise MultiSeedRunnerError(f"{architecture} winner is not COMPLETE")
    required_values: dict[str, object] = {
        "defined_label_count": 5,
        "probabilities_calibrated": False,
    }
    for field, expected in required_values.items():
        if winner.get(field) != expected:
            raise MultiSeedRunnerError(f"{architecture} winner {field} is invalid")
    run_dir = _path(winner.get("run_dir"), f"{architecture} winner run_dir")
    artifact_hashes_raw = _mapping(
        winner.get("artifact_sha256"), f"{architecture} winner artifact hashes"
    )
    if set(artifact_hashes_raw) != set(_SOURCE_FILENAMES):
        raise MultiSeedRunnerError(f"{architecture} winner artifact hash set is incomplete")
    artifact_hashes = {
        name: _hash(value, f"{architecture} {name} hash")
        for name, value in artifact_hashes_raw.items()
    }
    paths = _source_artifact_paths(run_dir)
    for name, path in paths.items():
        if _file_sha256(path) != artifact_hashes[name]:
            raise MultiSeedRunnerError(f"{architecture} winner {name} hash mismatch")

    config, resolved_hash = _load_resolved_experiment(paths["resolved_config.json"])
    if config.model.architecture != architecture or config.model.preset != "matched_capacity":
        raise MultiSeedRunnerError(f"{architecture} winner model identity mismatch")
    if config.runtime.seed != CONFIRMATION_SEEDS[0]:
        raise MultiSeedRunnerError(f"{architecture} winner seed must be 2026")
    if resolved_hash != _hash(winner.get("resolved_config_hash"), "winner config hash"):
        raise MultiSeedRunnerError(f"{architecture} winner resolved config hash mismatch")

    metadata = _read_json(paths["run_metadata.json"], "winner run metadata")
    best_epoch = _integer(winner.get("best_epoch"), "winner best_epoch")
    completed_epochs = _integer(
        winner.get("completed_epochs"), "winner completed_epochs", minimum=1
    )
    score = _number(
        winner.get("best_fold8_uncalibrated_macro_roc_auc"), "winner objective"
    )
    expected_metadata: dict[str, object] = {
        "status": "complete",
        "seed": 2026,
        "source_config_hash": winner.get("experiment_config_hash"),
        "resolved_config_hash": resolved_hash,
        "protocol_hash": protocol.protocol_hash,
        "completed_epochs": completed_epochs,
        "best_epoch": best_epoch,
        "best_validation_macro_auroc": score,
    }
    for field, expected in expected_metadata.items():
        if metadata.get(field) != expected:
            raise MultiSeedRunnerError(f"{architecture} winner metadata {field} mismatch")
    if _hash(metadata.get("manifest_hash"), "winner manifest hash") != manifest_sha256:
        raise MultiSeedRunnerError(f"{architecture} winner manifest hash mismatch")
    if _hash(
        metadata.get("normalization_file_hash"), "winner normalization hash"
    ) != normalization_sha256:
        raise MultiSeedRunnerError(f"{architecture} winner normalization hash mismatch")
    _read_history(
        paths["history.jsonl"],
        completed_epochs=completed_epochs,
        best_epoch=best_epoch,
        best_score=score,
    )
    _verify_checkpoint(
        paths["best.ckpt"],
        epoch=best_epoch,
        score=score,
        config_hash=resolved_hash,
        protocol_hash=protocol.protocol_hash,
        manifest_sha256=manifest_sha256,
    )
    if winner.get("selected_checkpoint_score") != score:
        raise MultiSeedRunnerError(f"{architecture} selected checkpoint score mismatch")
    if _hash(winner.get("manifest_hash"), "winner manifest hash") != manifest_sha256:
        raise MultiSeedRunnerError(f"{architecture} winner summary manifest hash mismatch")
    if _hash(
        winner.get("normalization_file_hash"), "winner normalization hash"
    ) != normalization_sha256:
        raise MultiSeedRunnerError(
            f"{architecture} winner summary normalization hash mismatch"
        )
    return WinnerSource(
        architecture=architecture,
        candidate_index=_integer(winner.get("candidate_index"), "winner candidate index"),
        trial_number=_integer(winner.get("trial_number"), "winner trial number"),
        run_dir=run_dir,
        experiment_config=config,
        scientific_config_sha256=_canonical_hash(_scientific_payload(config)),
        best_epoch=best_epoch,
        best_macro_auroc=score,
        completed_epochs=completed_epochs,
        artifact_sha256=artifact_hashes,
        manifest_sha256=manifest_sha256,
        normalization_sha256=normalization_sha256,
    )


def _load_and_verify_sweep_summary(
    path: Path,
    *,
    protocol: ExperimentProtocol,
) -> tuple[Mapping[str, object], dict[str, WinnerSource], str, str]:
    summary = _read_json(path, "sweep summary")
    required = {
        "schema_version",
        "comparison_id",
        "protocol_hash",
        "candidate_plan_path",
        "candidate_plan_hash",
        "equal_candidate_plan_verified",
        "paired_execution_policy",
        "paired_execution_order",
        "required_complete_candidates_per_architecture",
        "all_candidate_budgets_complete",
        "objective",
        "seed_policy",
        "tie_break",
        "failure_policy",
        "source_provenance",
        "best_by_architecture",
        "studies",
        "warning",
    }
    _keys(summary, required=required, context="sweep summary")
    if _integer(summary["schema_version"], "sweep schema", minimum=1) != _SWEEP_SCHEMA_VERSION:
        raise MultiSeedRunnerError("sweep summary must use schema version 2")
    if summary.get("protocol_hash") != protocol.protocol_hash:
        raise MultiSeedRunnerError("sweep summary protocol hash mismatch")
    fixed: dict[str, object] = {
        "equal_candidate_plan_verified": True,
        "required_complete_candidates_per_architecture": 12,
        "all_candidate_budgets_complete": True,
    }
    for field, expected in fixed.items():
        if summary.get(field) != expected:
            raise MultiSeedRunnerError(f"sweep summary {field} is invalid")
    objective = _mapping(summary["objective"], "sweep objective")
    if objective != {
        "direction": "maximize",
        "name": "fold8_uncalibrated_macro_roc_auc",
        "pruning": "none",
        "require_all_labels_defined": True,
        "required_label_count": 5,
    }:
        raise MultiSeedRunnerError("sweep objective is not the fixed fold-8 objective")
    seed_policy = _mapping(summary["seed_policy"], "sweep seed policy")
    if seed_policy != {"experiment_seed": 2026, "kind": "fixed_across_candidates"}:
        raise MultiSeedRunnerError("sweep seed policy is not fixed seed 2026")

    comparison_id = _string(summary["comparison_id"], "comparison_id")
    candidate_plan_hash = _hash(summary["candidate_plan_hash"], "candidate plan hash")
    candidate_plan_path = _path(summary["candidate_plan_path"], "candidate plan path")
    candidate_plan_file_sha256 = _verify_candidate_plan(
        candidate_plan_path,
        expected_comparison_id=comparison_id,
        expected_plan_hash=candidate_plan_hash,
    )
    source = _mapping(summary["source_provenance"], "source provenance")
    manifest_sha256 = _hash(source.get("manifest_sha256"), "source manifest hash")
    normalization_sha256 = _hash(
        source.get("normalization_sha256"), "source normalization hash"
    )
    manifest_path = _path(source.get("manifest_path"), "source manifest path")
    normalization_path = _path(
        source.get("normalization_path"), "source normalization path"
    )
    if _file_sha256(manifest_path) != manifest_sha256:
        raise MultiSeedRunnerError("sweep manifest file drifted after training")
    if _file_sha256(normalization_path) != normalization_sha256:
        raise MultiSeedRunnerError("sweep normalization file drifted after training")
    best = _mapping(summary["best_by_architecture"], "best_by_architecture")
    studies = _mapping(summary["studies"], "studies")
    if set(best) != set(ARCHITECTURES) or set(studies) != set(ARCHITECTURES):
        raise MultiSeedRunnerError("sweep summary must contain exactly both architectures")

    winners: dict[str, WinnerSource] = {}
    for architecture in ARCHITECTURES:
        winner = _mapping(best[architecture], f"{architecture} winner")
        _verify_study_winner(
            _mapping(studies[architecture], f"{architecture} study"),
            winner,
            architecture=architecture,
            comparison_id=comparison_id,
            candidate_plan_hash=candidate_plan_hash,
        )
        winners[architecture] = _verify_winner_source(
            winner,
            architecture=architecture,
            protocol=protocol,
            manifest_sha256=manifest_sha256,
            normalization_sha256=normalization_sha256,
        )
    if winners["resnet1d"].manifest_sha256 != winners["ecg_transformer"].manifest_sha256:
        raise MultiSeedRunnerError("winner manifest hashes differ")
    if (
        winners["resnet1d"].normalization_sha256
        != winners["ecg_transformer"].normalization_sha256
    ):
        raise MultiSeedRunnerError("winner normalization hashes differ")
    return summary, winners, candidate_plan_file_sha256, _file_sha256(path)


def _execution_order() -> tuple[tuple[str, int], ...]:
    return (
        ("resnet1d", 2026),
        ("ecg_transformer", 2026),
        ("resnet1d", 2027),
        ("ecg_transformer", 2027),
        ("ecg_transformer", 2028),
        ("resnet1d", 2028),
    )


def _member_plan_payload(
    *,
    comparison_id: str,
    confirmation_dir: Path,
    architecture: str,
    seed: int,
    winner: WinnerSource,
    sweep_summary_path: Path,
    sweep_summary_sha256: str,
    candidate_plan_hash: str,
    protocol_hash: str,
    revision_provenance: Mapping[str, object],
    revision_provenance_sha256: str,
) -> dict[str, object]:
    member_dir = confirmation_dir / "members" / architecture / f"seed{seed}"
    prediction_path = confirmation_dir / "predictions" / f"{architecture}-seed{seed}-fold8.npz"
    source_kind = "reused_sweep_winner" if seed == 2026 else "confirmation_training"
    return {
        "schema_version": MULTISEED_PLAN_SCHEMA_VERSION,
        "artifact_type": MEMBER_PLAN_TYPE,
        "comparison_id": comparison_id,
        "architecture": architecture,
        "seed": seed,
        "source_kind": source_kind,
        "train_folds": list(TRAIN_FOLDS),
        "model_selection_folds": list(MODEL_SELECTION_FOLDS),
        "protocol_hash": protocol_hash,
        "sweep_summary_path": str(sweep_summary_path),
        "sweep_summary_sha256": sweep_summary_sha256,
        "candidate_plan_hash": candidate_plan_hash,
        "winning_sweep_candidate": winner.candidate_index,
        "winning_sweep_trial": winner.trial_number,
        "winning_run_dir": str(winner.run_dir),
        "winning_artifact_sha256": dict(winner.artifact_sha256),
        "winning_best_epoch": winner.best_epoch,
        "winning_best_macro_auroc": winner.best_macro_auroc,
        "manifest_sha256": winner.manifest_sha256,
        "normalization_sha256": winner.normalization_sha256,
        "scientific_config_sha256": winner.scientific_config_sha256,
        "sweep_revision": _mapping(
            _mapping(
                revision_provenance["sweep_snapshot"], "sweep revision snapshot"
            )["source_provenance"],
            "sweep source provenance",
        )["git_head"],
        "execution_revision": _mapping(
            _mapping(
                revision_provenance["execution_snapshot"],
                "execution revision snapshot",
            )["source_provenance"],
            "execution source provenance",
        )["git_head"],
        "revision_provenance": dict(revision_provenance),
        "revision_provenance_sha256": revision_provenance_sha256,
        "experiment_template": winner.experiment_config.to_resolved_dict(),
        "allowed_config_differences": ["run_name", "runtime.seed", "output.root_dir"],
        "member_dir": str(member_dir),
        "attempt_root": str(member_dir / "attempts"),
        "max_attempts": 0 if seed == 2026 else MAX_ATTEMPTS,
        "prediction_path": str(prediction_path),
        "prediction_json_path": str(prediction_path.with_suffix(".json")),
        "completion_path": str(member_dir / "member_completion.json"),
    }


def create_multiseed_plan(
    sweep_summary_path: str | Path,
    *,
    output_root: str | Path,
    protocol: ExperimentProtocol,
) -> MultiSeedPlanResult:
    """Verify a completed paired sweep and persist its deterministic six-member plan."""

    summary_path = Path(sweep_summary_path).resolve()
    summary, winners, candidate_file_hash, summary_file_hash = (
        _load_and_verify_sweep_summary(summary_path, protocol=protocol)
    )
    comparison_id = _string(summary["comparison_id"], "comparison_id")
    candidate_plan_path = _path(summary["candidate_plan_path"], "candidate plan path")
    candidate_plan_hash = _hash(summary["candidate_plan_hash"], "candidate plan hash")
    confirmation_dir = Path(output_root).resolve() / comparison_id
    revision_provenance = _build_revision_provenance(summary)
    revision_provenance_sha256 = _hash(
        revision_provenance["artifact_sha256"], "revision provenance hash"
    )

    member_entries: list[dict[str, object]] = []
    member_paths: list[Path] = []
    for architecture, seed in _execution_order():
        payload = _hashed_payload(
            _member_plan_payload(
                comparison_id=comparison_id,
                confirmation_dir=confirmation_dir,
                architecture=architecture,
                seed=seed,
                winner=winners[architecture],
                sweep_summary_path=summary_path,
                sweep_summary_sha256=summary_file_hash,
                candidate_plan_hash=candidate_plan_hash,
                protocol_hash=protocol.protocol_hash,
                revision_provenance=revision_provenance,
                revision_provenance_sha256=revision_provenance_sha256,
            )
        )
        member_dir = _path(payload["member_dir"], "member directory")
        member_path = member_dir / "member_plan.json"
        _write_or_verify(member_path, payload, "member plan")
        member_hash = _hash(payload["artifact_sha256"], "member plan hash")
        member_entries.append(
            {
                "architecture": architecture,
                "seed": seed,
                "source_kind": payload["source_kind"],
                "member_plan_path": str(member_path),
                "member_plan_sha256": member_hash,
                "completion_path": payload["completion_path"],
                "prediction_path": payload["prediction_path"],
                "prediction_json_path": payload["prediction_json_path"],
            }
        )
        member_paths.append(member_path)

    root_payload = _hashed_payload(
        {
            "schema_version": MULTISEED_PLAN_SCHEMA_VERSION,
            "artifact_type": MULTISEED_PLAN_TYPE,
            "comparison_id": comparison_id,
            "sweep_summary_path": str(summary_path),
            "sweep_summary_sha256": summary_file_hash,
            "candidate_plan_path": str(candidate_plan_path),
            "candidate_plan_file_sha256": candidate_file_hash,
            "candidate_plan_hash": candidate_plan_hash,
            "protocol_hash": protocol.protocol_hash,
            "train_folds": list(TRAIN_FOLDS),
            "model_selection_folds": list(MODEL_SELECTION_FOLDS),
            "architectures": list(ARCHITECTURES),
            "seeds": list(CONFIRMATION_SEEDS),
            "max_attempts_per_trained_member": MAX_ATTEMPTS,
            "execution_order": [
                {"architecture": architecture, "seed": seed}
                for architecture, seed in _execution_order()
            ],
            "revision_provenance": revision_provenance,
            "revision_provenance_sha256": revision_provenance_sha256,
            "members": member_entries,
        }
    )
    plan_path = confirmation_dir / "multiseed_plan.json"
    _write_or_verify(plan_path, root_payload, "multi-seed plan")
    return MultiSeedPlanResult(
        comparison_id=comparison_id,
        confirmation_dir=confirmation_dir,
        plan_path=plan_path,
        plan_sha256=_hash(root_payload["artifact_sha256"], "plan hash"),
        member_plan_paths=tuple(member_paths),
    )


def _load_member_plan(
    path: Path,
    *,
    expected_hash: str,
    protocol: ExperimentProtocol,
) -> Mapping[str, object]:
    payload = _read_json(path, "member plan")
    required = {
        "schema_version",
        "artifact_type",
        "comparison_id",
        "architecture",
        "seed",
        "source_kind",
        "train_folds",
        "model_selection_folds",
        "protocol_hash",
        "sweep_summary_path",
        "sweep_summary_sha256",
        "candidate_plan_hash",
        "winning_sweep_candidate",
        "winning_sweep_trial",
        "winning_run_dir",
        "winning_artifact_sha256",
        "winning_best_epoch",
        "winning_best_macro_auroc",
        "manifest_sha256",
        "normalization_sha256",
        "scientific_config_sha256",
        "sweep_revision",
        "execution_revision",
        "revision_provenance",
        "revision_provenance_sha256",
        "experiment_template",
        "allowed_config_differences",
        "member_dir",
        "attempt_root",
        "max_attempts",
        "prediction_path",
        "prediction_json_path",
        "completion_path",
        "artifact_sha256",
    }
    _keys(payload, required=required, context="member plan")
    observed_hash = _verify_self_hash(payload, "member plan")
    if observed_hash != expected_hash:
        raise MultiSeedRunnerError("member plan hash differs from root plan")
    if payload.get("schema_version") != MULTISEED_PLAN_SCHEMA_VERSION:
        raise MultiSeedRunnerError("unsupported member-plan schema version")
    if payload.get("artifact_type") != MEMBER_PLAN_TYPE:
        raise MultiSeedRunnerError("unexpected member-plan artifact type")
    if payload.get("protocol_hash") != protocol.protocol_hash:
        raise MultiSeedRunnerError("member plan protocol hash mismatch")
    if tuple(_sequence(payload["train_folds"], "member train folds")) != TRAIN_FOLDS:
        raise MultiSeedRunnerError("member plan train folds must be 1-7")
    if (
        tuple(_sequence(payload["model_selection_folds"], "member selection folds"))
        != MODEL_SELECTION_FOLDS
    ):
        raise MultiSeedRunnerError("member plan model-selection folds must be fold 8")
    seed = _integer(payload["seed"], "member seed")
    if seed not in CONFIRMATION_SEEDS:
        raise MultiSeedRunnerError("member seed is not preregistered")
    architecture = _string(payload["architecture"], "member architecture")
    if architecture not in ARCHITECTURES:
        raise MultiSeedRunnerError("member architecture is unsupported")
    source_kind = _string(payload["source_kind"], "member source_kind")
    expected_kind = "reused_sweep_winner" if seed == 2026 else "confirmation_training"
    if source_kind != expected_kind:
        raise MultiSeedRunnerError("member source kind disagrees with its seed")
    expected_attempts = 0 if seed == 2026 else MAX_ATTEMPTS
    if payload.get("max_attempts") != expected_attempts:
        raise MultiSeedRunnerError("member max_attempts is invalid")
    if payload.get("allowed_config_differences") != [
        "run_name",
        "runtime.seed",
        "output.root_dir",
    ]:
        raise MultiSeedRunnerError("member config mutation policy changed")
    revision_provenance = _validate_revision_provenance(
        payload["revision_provenance"]
    )
    revision_hash = _hash(
        revision_provenance["artifact_sha256"], "member revision provenance hash"
    )
    if payload.get("revision_provenance_sha256") != revision_hash:
        raise MultiSeedRunnerError("member revision-provenance hash mismatch")
    kernel = _mapping(
        revision_provenance["scientific_kernel"], "member scientific kernel proof"
    )
    if payload.get("sweep_revision") != kernel.get("sweep_revision"):
        raise MultiSeedRunnerError("member sweep revision differs from proof")
    if payload.get("execution_revision") != kernel.get("execution_revision"):
        raise MultiSeedRunnerError("member execution revision differs from proof")
    template = DevelopmentExperimentConfig.from_mapping(
        _mapping(payload["experiment_template"], "experiment template"),
        base_dir=path.parent,
    )
    if template.model.architecture != architecture or template.runtime.seed != 2026:
        raise MultiSeedRunnerError("member experiment template is not its sweep winner")
    if _canonical_hash(_scientific_payload(template)) != payload.get(
        "scientific_config_sha256"
    ):
        raise MultiSeedRunnerError("member scientific config hash mismatch")
    if _file_sha256(template.data.manifest_path) != payload.get("manifest_sha256"):
        raise MultiSeedRunnerError("member manifest file drifted after planning")
    if _file_sha256(template.data.normalization_path) != payload.get(
        "normalization_sha256"
    ):
        raise MultiSeedRunnerError("member normalization file drifted after planning")
    return payload


def load_multiseed_plan(
    path: str | Path,
    *,
    protocol: ExperimentProtocol,
) -> Mapping[str, object]:
    """Load a plan and revalidate every immutable upstream binding."""

    plan_path = Path(path).resolve()
    payload = _read_json(plan_path, "multi-seed plan")
    _keys(payload, required=_ROOT_PLAN_KEYS, context="multi-seed plan")
    _verify_self_hash(payload, "multi-seed plan")
    if payload.get("schema_version") != MULTISEED_PLAN_SCHEMA_VERSION:
        raise MultiSeedRunnerError("unsupported multi-seed plan schema version")
    if payload.get("artifact_type") != MULTISEED_PLAN_TYPE:
        raise MultiSeedRunnerError("unexpected multi-seed plan artifact type")
    if payload.get("protocol_hash") != protocol.protocol_hash:
        raise MultiSeedRunnerError("multi-seed plan protocol hash mismatch")
    if tuple(_sequence(payload["train_folds"], "plan train folds")) != TRAIN_FOLDS:
        raise MultiSeedRunnerError("multi-seed plan train folds must be 1-7")
    if (
        tuple(_sequence(payload["model_selection_folds"], "plan selection folds"))
        != MODEL_SELECTION_FOLDS
    ):
        raise MultiSeedRunnerError("multi-seed plan model-selection folds must be fold 8")
    if tuple(_sequence(payload["architectures"], "plan architectures")) != ARCHITECTURES:
        raise MultiSeedRunnerError("multi-seed plan architecture order changed")
    if tuple(_sequence(payload["seeds"], "plan seeds")) != CONFIRMATION_SEEDS:
        raise MultiSeedRunnerError("multi-seed plan seed set changed")
    if payload.get("max_attempts_per_trained_member") != MAX_ATTEMPTS:
        raise MultiSeedRunnerError("multi-seed max-attempt policy changed")

    summary_path = _path(payload["sweep_summary_path"], "plan sweep summary path")
    if _file_sha256(summary_path) != payload.get("sweep_summary_sha256"):
        raise MultiSeedRunnerError("completed sweep summary drifted after planning")
    summary, _, candidate_file_hash, summary_hash = _load_and_verify_sweep_summary(
        summary_path, protocol=protocol
    )
    if summary_hash != payload.get("sweep_summary_sha256"):
        raise MultiSeedRunnerError("sweep summary hash does not match plan")
    if summary.get("comparison_id") != payload.get("comparison_id"):
        raise MultiSeedRunnerError("sweep/confirmation comparison_id mismatch")
    if summary.get("candidate_plan_hash") != payload.get("candidate_plan_hash"):
        raise MultiSeedRunnerError("sweep/confirmation candidate-plan hash mismatch")
    if candidate_file_hash != payload.get("candidate_plan_file_sha256"):
        raise MultiSeedRunnerError("candidate-plan file drifted after planning")
    sweep_source = _validated_sweep_source(summary.get("source_provenance"))
    revision_provenance = _verify_revision_provenance(
        payload["revision_provenance"], expected_sweep_source=sweep_source
    )
    revision_hash = _hash(
        revision_provenance["artifact_sha256"], "plan revision provenance hash"
    )
    if payload.get("revision_provenance_sha256") != revision_hash:
        raise MultiSeedRunnerError("plan revision-provenance hash mismatch")

    execution = [
        (
            _string(_mapping(item, "execution member")["architecture"], "architecture"),
            _integer(_mapping(item, "execution member")["seed"], "seed"),
        )
        for item in _sequence(payload["execution_order"], "execution_order")
    ]
    if tuple(execution) != _execution_order():
        raise MultiSeedRunnerError("multi-seed execution order changed")
    members = _sequence(payload["members"], "members")
    if len(members) != 6:
        raise MultiSeedRunnerError("multi-seed plan must contain exactly six members")
    observed: list[tuple[str, int]] = []
    for item in members:
        entry = _mapping(item, "root member entry")
        _keys(
            entry,
            required={
                "architecture",
                "seed",
                "source_kind",
                "member_plan_path",
                "member_plan_sha256",
                "completion_path",
                "prediction_path",
                "prediction_json_path",
            },
            context="root member entry",
        )
        architecture = _string(entry["architecture"], "member architecture")
        seed = _integer(entry["seed"], "member seed")
        member_plan = _load_member_plan(
            _path(entry["member_plan_path"], "member plan path"),
            expected_hash=_hash(entry["member_plan_sha256"], "member plan hash"),
            protocol=protocol,
        )
        comparisons = {
            "architecture": architecture,
            "seed": seed,
            "source_kind": entry["source_kind"],
            "completion_path": entry["completion_path"],
            "prediction_path": entry["prediction_path"],
            "prediction_json_path": entry["prediction_json_path"],
            "revision_provenance": revision_provenance,
            "revision_provenance_sha256": revision_hash,
        }
        for field, expected in comparisons.items():
            if member_plan.get(field) != expected:
                raise MultiSeedRunnerError(f"root/member plan {field} mismatch")
        observed.append((architecture, seed))
    if tuple(observed) != _execution_order():
        raise MultiSeedRunnerError("root plan member order changed")
    return payload


def _attempt_name(architecture: str, seed: int, attempt: int) -> str:
    return f"{architecture}-confirmation-seed{seed}-attempt{attempt:02d}"


def _attempt_paths(member: Mapping[str, object], attempt: int) -> tuple[Path, Path, Path]:
    root = _path(member["attempt_root"], "attempt root")
    name = _attempt_name(
        _string(member["architecture"], "member architecture"),
        _integer(member["seed"], "member seed"),
        attempt,
    )
    return (
        root / f"attempt{attempt:02d}_plan.json",
        root / f"attempt{attempt:02d}_status.json",
        root / name,
    )


def _attempt_config(
    member: Mapping[str, object],
    *,
    attempt: int,
) -> DevelopmentExperimentConfig:
    template = dict(_mapping(member["experiment_template"], "experiment template"))
    architecture = _string(member["architecture"], "member architecture")
    seed = _integer(member["seed"], "member seed")
    attempt_root = _path(member["attempt_root"], "attempt root")
    template["run_name"] = _attempt_name(architecture, seed, attempt)
    runtime = dict(_mapping(template["runtime"], "experiment runtime"))
    runtime["seed"] = seed
    template["runtime"] = runtime
    template["output"] = {"root_dir": str(attempt_root)}
    try:
        config = DevelopmentExperimentConfig.from_mapping(template, base_dir=attempt_root)
    except (TypeError, ValueError) as error:
        raise MultiSeedRunnerError(f"could not build confirmation config: {error}") from error
    if _canonical_hash(_scientific_payload(config)) != member.get(
        "scientific_config_sha256"
    ):
        raise MultiSeedRunnerError("confirmation config changed a scientific field")
    if config.runtime.seed != seed:
        raise MultiSeedRunnerError("confirmation config seed mismatch")
    return config


def _attempt_plan_payload(
    member: Mapping[str, object],
    *,
    member_plan_path: Path,
    member_plan_hash: str,
    attempt: int,
    config: DevelopmentExperimentConfig,
) -> dict[str, object]:
    _, _, run_dir = _attempt_paths(member, attempt)
    return _hashed_payload(
        {
            "schema_version": MULTISEED_PLAN_SCHEMA_VERSION,
            "artifact_type": ATTEMPT_PLAN_TYPE,
            "comparison_id": member["comparison_id"],
            "architecture": member["architecture"],
            "seed": member["seed"],
            "attempt": attempt,
            "member_plan_path": str(member_plan_path),
            "member_plan_sha256": member_plan_hash,
            "sweep_revision": member["sweep_revision"],
            "execution_revision": member["execution_revision"],
            "revision_provenance_sha256": member["revision_provenance_sha256"],
            "scientific_config_sha256": member["scientific_config_sha256"],
            "experiment_config_hash": config.config_hash,
            "experiment_config": config.to_resolved_dict(),
            "run_dir": str(run_dir),
        }
    )


def _verify_attempt_plan(
    path: Path,
    *,
    expected: Mapping[str, object],
) -> None:
    observed = _read_json(path, "attempt plan")
    _verify_self_hash(observed, "attempt plan")
    if observed != expected:
        raise MultiSeedRunnerError("persisted attempt plan differs from fixed config")


def _attempt_status_payload(
    member: Mapping[str, object],
    *,
    attempt: int,
    status: Literal["complete", "failed"],
    run_dir: Path,
    reason: str | None,
    verified: VerifiedRun | None,
) -> dict[str, object]:
    return _hashed_payload(
        {
            "schema_version": MULTISEED_PLAN_SCHEMA_VERSION,
            "artifact_type": ATTEMPT_STATUS_TYPE,
            "comparison_id": member["comparison_id"],
            "architecture": member["architecture"],
            "seed": member["seed"],
            "attempt": attempt,
            "execution_revision": member["execution_revision"],
            "revision_provenance_sha256": member["revision_provenance_sha256"],
            "status": status,
            "run_dir": str(run_dir),
            "reason": reason,
            "resolved_config_hash": (
                verified.resolved_config_hash if verified is not None else None
            ),
            "best_epoch": verified.best_epoch if verified is not None else None,
            "best_validation_macro_auroc": (
                verified.best_macro_auroc if verified is not None else None
            ),
            "completed_epochs": verified.completed_epochs if verified is not None else None,
            "run_artifact_sha256": dict(verified.hashes) if verified is not None else None,
        }
    )


def _load_attempt_status(
    path: Path,
    *,
    member: Mapping[str, object],
    attempt: int,
) -> Mapping[str, object]:
    payload = _read_json(path, "attempt status")
    required = {
        "schema_version",
        "artifact_type",
        "comparison_id",
        "architecture",
        "seed",
        "attempt",
        "execution_revision",
        "revision_provenance_sha256",
        "status",
        "run_dir",
        "reason",
        "resolved_config_hash",
        "best_epoch",
        "best_validation_macro_auroc",
        "completed_epochs",
        "run_artifact_sha256",
        "artifact_sha256",
    }
    _keys(payload, required=required, context="attempt status")
    _verify_self_hash(payload, "attempt status")
    expected_values = {
        "schema_version": MULTISEED_PLAN_SCHEMA_VERSION,
        "artifact_type": ATTEMPT_STATUS_TYPE,
        "comparison_id": member["comparison_id"],
        "architecture": member["architecture"],
        "seed": member["seed"],
        "attempt": attempt,
        "execution_revision": member["execution_revision"],
        "revision_provenance_sha256": member["revision_provenance_sha256"],
    }
    for field, expected in expected_values.items():
        if payload.get(field) != expected:
            raise MultiSeedRunnerError(f"attempt status {field} mismatch")
    if payload.get("status") not in {"complete", "failed"}:
        raise MultiSeedRunnerError("attempt status is invalid")
    return payload


def _write_attempt_status(
    path: Path,
    member: Mapping[str, object],
    *,
    attempt: int,
    status: Literal["complete", "failed"],
    run_dir: Path,
    reason: str | None,
    verified: VerifiedRun | None,
) -> None:
    payload = _attempt_status_payload(
        member,
        attempt=attempt,
        status=status,
        run_dir=run_dir,
        reason=reason,
        verified=verified,
    )
    _atomic_write_new(path, payload)


def _verify_complete_run(
    member: Mapping[str, object],
    run_dir: Path,
    *,
    protocol: ExperimentProtocol,
    expected_attempt: int | None,
) -> VerifiedRun:
    revision_provenance = _verify_revision_provenance(
        member["revision_provenance"]
    )
    paths = _source_artifact_paths(run_dir)
    for path in paths.values():
        if not path.is_file():
            raise MultiSeedRunnerError(f"completed run artifact is missing: {path}")
    config, resolved_hash = _load_resolved_experiment(paths["resolved_config.json"])
    architecture = _string(member["architecture"], "member architecture")
    seed = _integer(member["seed"], "member seed")
    if config.model.architecture != architecture or config.model.preset != "matched_capacity":
        raise MultiSeedRunnerError("completed run architecture/preset mismatch")
    if config.runtime.seed != seed:
        raise MultiSeedRunnerError("completed run seed mismatch")
    if _canonical_hash(_scientific_payload(config)) != member.get(
        "scientific_config_sha256"
    ):
        raise MultiSeedRunnerError("completed run scientific config drifted")
    if expected_attempt is None:
        if seed != 2026 or run_dir != _path(member["winning_run_dir"], "winner run dir"):
            raise MultiSeedRunnerError("reused member is not the selected sweep winner")
    else:
        expected_name = _attempt_name(architecture, seed, expected_attempt)
        if config.run_name != expected_name or run_dir.name != expected_name:
            raise MultiSeedRunnerError("completed attempt run name mismatch")
        if config.output.root_dir != _path(member["attempt_root"], "attempt root"):
            raise MultiSeedRunnerError("completed attempt output root mismatch")

    metadata = _read_json(paths["run_metadata.json"], "run metadata")
    if metadata.get("status") != "complete":
        raise MultiSeedRunnerError("run metadata status must be complete")
    if metadata.get("seed") != seed:
        raise MultiSeedRunnerError("run metadata seed mismatch")
    if metadata.get("source_config_hash") != config.config_hash:
        raise MultiSeedRunnerError("run source config hash mismatch")
    if metadata.get("resolved_config_hash") != resolved_hash:
        raise MultiSeedRunnerError("run resolved config hash mismatch")
    if metadata.get("protocol_hash") != protocol.protocol_hash:
        raise MultiSeedRunnerError("run protocol hash mismatch")
    execution_snapshot = _mapping(
        revision_provenance["execution_snapshot"], "execution revision snapshot"
    )
    execution_source = _mapping(
        execution_snapshot["source_provenance"], "execution source provenance"
    )
    expected_runtime = _mapping(
        execution_source["runtime_identity"], "execution runtime identity"
    )
    run_runtime = _mapping(metadata.get("runtime"), "run runtime metadata")
    for field in ("python", "platform", "torch"):
        if run_runtime.get(field) != expected_runtime.get(field):
            raise MultiSeedRunnerError(f"run runtime {field} differs from commit-B proof")
    manifest_hash = _hash(metadata.get("manifest_hash"), "run manifest hash")
    normalization_hash = _hash(
        metadata.get("normalization_file_hash"), "run normalization hash"
    )
    template = DevelopmentExperimentConfig.from_mapping(
        _mapping(member["experiment_template"], "experiment template"),
        base_dir=run_dir,
    )
    expected_manifest = _file_sha256(template.data.manifest_path)
    expected_normalization = _file_sha256(template.data.normalization_path)
    if expected_manifest != member.get("manifest_sha256"):
        raise MultiSeedRunnerError("member manifest hash differs from its sweep winner")
    if expected_normalization != member.get("normalization_sha256"):
        raise MultiSeedRunnerError("member normalization hash differs from its sweep winner")
    if manifest_hash != expected_manifest:
        raise MultiSeedRunnerError("run manifest file hash mismatch")
    if normalization_hash != expected_normalization:
        raise MultiSeedRunnerError("run normalization file hash mismatch")
    best_epoch = _integer(metadata.get("best_epoch"), "run best_epoch")
    completed_epochs = _integer(
        metadata.get("completed_epochs"), "run completed_epochs", minimum=1
    )
    if completed_epochs > config.optimization.epochs or best_epoch >= completed_epochs:
        raise MultiSeedRunnerError("run epoch metadata is invalid")
    best_score = _number(
        metadata.get("best_validation_macro_auroc"), "run best validation score"
    )
    _read_history(
        paths["history.jsonl"],
        completed_epochs=completed_epochs,
        best_epoch=best_epoch,
        best_score=best_score,
    )
    _verify_checkpoint(
        paths["best.ckpt"],
        epoch=best_epoch,
        score=best_score,
        config_hash=resolved_hash,
        protocol_hash=protocol.protocol_hash,
        manifest_sha256=manifest_hash,
    )
    hashes = {name: _file_sha256(path) for name, path in paths.items()}
    if seed == 2026:
        expected_hashes = {
            name: _hash(value, f"winner {name} hash")
            for name, value in _mapping(
                member["winning_artifact_sha256"], "winner artifact hashes"
            ).items()
        }
        if hashes != expected_hashes:
            raise MultiSeedRunnerError("reused winner artifacts drifted after planning")
    return VerifiedRun(
        run_dir=run_dir,
        run_name=config.run_name,
        seed=seed,
        resolved_config_hash=resolved_hash,
        manifest_sha256=manifest_hash,
        normalization_sha256=normalization_hash,
        best_epoch=best_epoch,
        best_macro_auroc=best_score,
        completed_epochs=completed_epochs,
        paths=paths,
        hashes=hashes,
    )


def _verify_attempt_status_run(
    status: Mapping[str, object],
    run: VerifiedRun,
) -> None:
    expected: dict[str, object] = {
        "resolved_config_hash": run.resolved_config_hash,
        "best_epoch": run.best_epoch,
        "best_validation_macro_auroc": run.best_macro_auroc,
        "completed_epochs": run.completed_epochs,
        "run_artifact_sha256": dict(run.hashes),
    }
    for field, value in expected.items():
        if status.get(field) != value:
            raise MultiSeedRunnerError(f"attempt status {field} drifted from its run")


def _prediction_paths(member: Mapping[str, object]) -> tuple[Path, Path]:
    return (
        _path(member["prediction_path"], "prediction path"),
        _path(member["prediction_json_path"], "prediction JSON path"),
    )


def _verify_prediction(
    member: Mapping[str, object],
    run: VerifiedRun,
    *,
    protocol: ExperimentProtocol,
) -> tuple[PredictionArtifact, str, str]:
    prediction_path, prediction_json = _prediction_paths(member)
    if not prediction_path.is_file() or not prediction_json.is_file():
        raise MultiSeedRunnerError("fold-8 prediction artifact pair is incomplete")
    try:
        artifact = load_prediction_artifact(
            prediction_path,
            protocol=protocol,
            expected_config_hash=run.resolved_config_hash,
            expected_manifest_hash=run.manifest_sha256,
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise MultiSeedRunnerError(f"invalid fold-8 prediction artifact: {error}") from error
    if artifact.fold_role is not FoldRole.MODEL_SELECTION or artifact.folds != (8,):
        raise MultiSeedRunnerError("confirmation prediction must contain fold 8 only")
    if artifact.model_seed != run.seed or artifact.model_name != run.run_name:
        raise MultiSeedRunnerError("prediction model identity does not match completed run")
    if artifact.calibrated_probabilities is not None:
        raise MultiSeedRunnerError("fold-8 confirmation predictions must be uncalibrated")
    extra = artifact.extra_metadata
    expected_extra: dict[str, object] = {
        "lineage": "development",
        "checkpoint_sha256": run.hashes["best.ckpt"],
        "checkpoint_epoch": run.best_epoch,
    }
    for field, expected in expected_extra.items():
        if extra.get(field) != expected:
            raise MultiSeedRunnerError(f"prediction {field} lineage mismatch")
    integrity = _hash(artifact.integrity_sha256, "prediction artifact integrity")
    return artifact, _file_sha256(prediction_path), integrity


def _export_or_verify_prediction(
    member: Mapping[str, object],
    run: VerifiedRun,
    *,
    protocol: ExperimentProtocol,
    prediction_exporter: PredictionExporter,
) -> tuple[PredictionArtifact, str, str]:
    _verify_revision_provenance(member["revision_provenance"])
    prediction_path, prediction_json = _prediction_paths(member)
    if prediction_path.exists() or prediction_json.exists():
        if not prediction_path.is_file() or not prediction_json.is_file():
            raise MultiSeedRunnerError("partial immutable prediction artifact exists")
        return _verify_prediction(member, run, protocol=protocol)
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    config, _ = _load_resolved_experiment(run.paths["resolved_config.json"])
    request = PredictionExportRequest(
        checkpoint_path=run.paths["best.ckpt"],
        resolved_config_path=run.paths["resolved_config.json"],
        run_metadata_path=run.paths["run_metadata.json"],
        output_path=prediction_path,
        fold_role=FoldRole.MODEL_SELECTION,
        batch_size=config.loader.batch_size,
        num_workers=config.loader.num_workers,
        pin_memory=config.loader.pin_memory,
        persistent_workers=config.loader.persistent_workers,
        device=config.runtime.device,
        bf16=config.runtime.bf16,
    )
    try:
        result = prediction_exporter(request, protocol=protocol)
    except (OSError, RuntimeError, ValueError) as error:
        raise MultiSeedRunnerError(f"fold-8 prediction export failed: {error}") from error
    if result.fold_role is not FoldRole.MODEL_SELECTION or result.folds != (8,):
        raise MultiSeedRunnerError("prediction exporter returned a non-fold-8 result")
    return _verify_prediction(member, run, protocol=protocol)


def _completion_payload(
    member: Mapping[str, object],
    *,
    member_plan_path: Path,
    member_plan_sha256: str,
    run: VerifiedRun,
    prediction_npz_sha256: str,
    prediction_artifact_sha256: str,
) -> dict[str, object]:
    prediction_path, prediction_json = _prediction_paths(member)
    return _hashed_payload(
        {
            "schema_version": MULTISEED_PLAN_SCHEMA_VERSION,
            "artifact_type": MEMBER_COMPLETION_TYPE,
            "comparison_id": member["comparison_id"],
            "architecture": member["architecture"],
            "seed": member["seed"],
            "status": "complete",
            "member_plan_path": str(member_plan_path),
            "member_plan_sha256": member_plan_sha256,
            "run_dir": str(run.run_dir),
            "run_metadata_path": str(run.paths["run_metadata.json"]),
            "run_metadata_sha256": run.hashes["run_metadata.json"],
            "resolved_config_path": str(run.paths["resolved_config.json"]),
            "resolved_config_sha256": run.hashes["resolved_config.json"],
            "history_path": str(run.paths["history.jsonl"]),
            "history_sha256": run.hashes["history.jsonl"],
            "best_checkpoint_path": str(run.paths["best.ckpt"]),
            "best_checkpoint_sha256": run.hashes["best.ckpt"],
            "config_hash": run.resolved_config_hash,
            "protocol_hash": member["protocol_hash"],
            "manifest_hash": run.manifest_sha256,
            "normalization_sha256": run.normalization_sha256,
            "best_epoch": run.best_epoch,
            "best_validation_macro_auroc": run.best_macro_auroc,
            "completed_epochs": run.completed_epochs,
            "prediction_path": str(prediction_path),
            "prediction_npz_sha256": prediction_npz_sha256,
            "prediction_json_path": str(prediction_json),
            "prediction_artifact_sha256": prediction_artifact_sha256,
        }
    )


def _verify_completion(
    path: Path,
    member: Mapping[str, object],
    *,
    member_plan_path: Path,
    member_plan_sha256: str,
    protocol: ExperimentProtocol,
) -> tuple[Mapping[str, object], PredictionArtifact]:
    payload = _read_json(path, "member completion")
    _keys(payload, required=_MEMBER_COMPLETION_KEYS, context="member completion")
    _verify_self_hash(payload, "member completion")
    fixed: dict[str, object] = {
        "schema_version": MULTISEED_PLAN_SCHEMA_VERSION,
        "artifact_type": MEMBER_COMPLETION_TYPE,
        "comparison_id": member["comparison_id"],
        "architecture": member["architecture"],
        "seed": member["seed"],
        "status": "complete",
        "member_plan_path": str(member_plan_path),
        "member_plan_sha256": member_plan_sha256,
        "prediction_path": member["prediction_path"],
        "prediction_json_path": member["prediction_json_path"],
        "protocol_hash": protocol.protocol_hash,
    }
    for field, expected in fixed.items():
        if payload.get(field) != expected:
            raise MultiSeedRunnerError(f"member completion {field} mismatch")
    run_dir = _path(payload["run_dir"], "completion run_dir")
    expected_attempt: int | None = None
    if member["source_kind"] == "confirmation_training":
        prefix = _attempt_name(
            _string(member["architecture"], "architecture"),
            _integer(member["seed"], "seed"),
            0,
        )[:-2]
        if not run_dir.name.startswith(prefix):
            raise MultiSeedRunnerError("completion run name is not a confirmation attempt")
        suffix = run_dir.name[len(prefix) :]
        if len(suffix) != 2 or not suffix.isdigit():
            raise MultiSeedRunnerError("completion attempt suffix is invalid")
        expected_attempt = int(suffix)
        if expected_attempt >= MAX_ATTEMPTS:
            raise MultiSeedRunnerError("completion attempt exceeds retry budget")
    run = _verify_complete_run(
        member,
        run_dir,
        protocol=protocol,
        expected_attempt=expected_attempt,
    )
    prediction, npz_hash, integrity = _verify_prediction(member, run, protocol=protocol)
    expected_fields: dict[str, object] = {
        "run_metadata_path": str(run.paths["run_metadata.json"]),
        "run_metadata_sha256": run.hashes["run_metadata.json"],
        "resolved_config_path": str(run.paths["resolved_config.json"]),
        "resolved_config_sha256": run.hashes["resolved_config.json"],
        "history_path": str(run.paths["history.jsonl"]),
        "history_sha256": run.hashes["history.jsonl"],
        "best_checkpoint_path": str(run.paths["best.ckpt"]),
        "best_checkpoint_sha256": run.hashes["best.ckpt"],
        "config_hash": run.resolved_config_hash,
        "manifest_hash": run.manifest_sha256,
        "normalization_sha256": run.normalization_sha256,
        "best_epoch": run.best_epoch,
        "best_validation_macro_auroc": run.best_macro_auroc,
        "completed_epochs": run.completed_epochs,
        "prediction_npz_sha256": npz_hash,
        "prediction_artifact_sha256": integrity,
    }
    for field, expected in expected_fields.items():
        if payload.get(field) != expected:
            raise MultiSeedRunnerError(f"member completion {field} drifted")
    return payload, prediction


def _find_complete_attempt(
    member: Mapping[str, object],
    *,
    protocol: ExperimentProtocol,
    resume: bool,
) -> VerifiedRun | None:
    for attempt in range(MAX_ATTEMPTS):
        plan_path, status_path, run_dir = _attempt_paths(member, attempt)
        config = _attempt_config(member, attempt=attempt)
        expected_plan = _attempt_plan_payload(
            member,
            member_plan_path=_path(member["member_dir"], "member directory")
            / "member_plan.json",
            member_plan_hash=_hash(member["artifact_sha256"], "member plan hash"),
            attempt=attempt,
            config=config,
        )
        if plan_path.exists():
            _verify_attempt_plan(plan_path, expected=expected_plan)
        if status_path.exists():
            status = _load_attempt_status(
                status_path, member=member, attempt=attempt
            )
            if status["status"] == "failed":
                continue
            verified = _verify_complete_run(
                member,
                run_dir,
                protocol=protocol,
                expected_attempt=attempt,
            )
            _verify_attempt_status_run(status, verified)
            return verified
        if not run_dir.exists():
            continue
        metadata_path = run_dir / "run_metadata.json"
        status_value: object = None
        if metadata_path.is_file():
            status_value = _read_json(metadata_path, "attempt metadata").get("status")
        if status_value == "complete":
            verified = _verify_complete_run(
                member,
                run_dir,
                protocol=protocol,
                expected_attempt=attempt,
            )
            if not resume:
                raise MultiSeedRunnerError("existing completed attempt requires --resume")
            _write_attempt_status(
                status_path,
                member,
                attempt=attempt,
                status="complete",
                run_dir=run_dir,
                reason=None,
                verified=verified,
            )
            return verified
        if not resume:
            raise MultiSeedRunnerError("existing interrupted attempt requires --resume")
        _write_attempt_status(
            status_path,
            member,
            attempt=attempt,
            status="failed",
            run_dir=run_dir,
            reason="interrupted attempt preserved and marked failed during resume",
            verified=None,
        )
    return None


def _next_attempt(member: Mapping[str, object]) -> int:
    for attempt in range(MAX_ATTEMPTS):
        plan_path, status_path, run_dir = _attempt_paths(member, attempt)
        if not plan_path.exists() and not status_path.exists() and not run_dir.exists():
            return attempt
        if plan_path.exists() and not status_path.exists() and not run_dir.exists():
            return attempt
    raise MultiSeedRunnerError("confirmation member exhausted its three-attempt retry budget")


def _train_member(
    member: Mapping[str, object],
    *,
    protocol: ExperimentProtocol,
    resume: bool,
    experiment_executor: DevelopmentExecutor,
) -> VerifiedRun:
    existing = _find_complete_attempt(member, protocol=protocol, resume=resume)
    if existing is not None:
        return existing
    while True:
        attempt = _next_attempt(member)
        plan_path, status_path, run_dir = _attempt_paths(member, attempt)
        config = _attempt_config(member, attempt=attempt)
        member_plan_path = _path(member["member_dir"], "member directory") / "member_plan.json"
        attempt_payload = _attempt_plan_payload(
            member,
            member_plan_path=member_plan_path,
            member_plan_hash=_hash(member["artifact_sha256"], "member plan hash"),
            attempt=attempt,
            config=config,
        )
        _write_or_verify(plan_path, attempt_payload, "attempt plan")
        try:
            result = experiment_executor(config, protocol=protocol)
            if result.run_dir.resolve() != run_dir.resolve():
                raise MultiSeedRunnerError("development executor returned an unexpected run dir")
            verified = _verify_complete_run(
                member,
                run_dir,
                protocol=protocol,
                expected_attempt=attempt,
            )
        except Exception as error:
            _write_attempt_status(
                status_path,
                member,
                attempt=attempt,
                status="failed",
                run_dir=run_dir,
                reason=f"{type(error).__name__}: {error}",
                verified=None,
            )
            if attempt + 1 >= MAX_ATTEMPTS:
                raise MultiSeedRunnerError(
                    "confirmation member failed its three fixed-config attempts"
                ) from error
            continue
        _write_attempt_status(
            status_path,
            member,
            attempt=attempt,
            status="complete",
            run_dir=run_dir,
            reason=None,
            verified=verified,
        )
        return verified


def _generated_state_exists(plan: Mapping[str, object]) -> bool:
    for raw in _sequence(plan["members"], "members"):
        entry = _mapping(raw, "member entry")
        for field in ("completion_path", "prediction_path", "prediction_json_path"):
            if _path(entry[field], field).exists():
                return True
        member = _read_json(_path(entry["member_plan_path"], "member plan"), "member plan")
        attempt_root = _path(member["attempt_root"], "attempt root")
        if attempt_root.exists() and any(attempt_root.iterdir()):
            return True
    return False


def _lock_payload(plan_path: Path, plan_hash: str, token: str) -> dict[str, object]:
    return {
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "plan_path": str(plan_path),
        "plan_sha256": plan_hash,
        "token": token,
    }


def _lock_is_live(payload: Mapping[str, object]) -> bool | None:
    if payload.get("hostname") != socket.gethostname():
        return None
    pid = payload.get("pid")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid < 1:
        return None
    try:
        return bool(psutil.pid_exists(pid))
    except (OSError, psutil.Error):
        return None


@contextmanager
def _writer_lock(plan_path: Path, plan_hash: str) -> Iterator[None]:
    lock_path = plan_path.parent / ".confirmation-writer.lock"
    token = hashlib.sha256(os.urandom(32)).hexdigest()
    payload = _lock_payload(plan_path, plan_hash, token)
    serialized = json.dumps(payload, sort_keys=True, allow_nan=False).encode("utf-8")
    while True:
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            try:
                current = _read_json(lock_path, "confirmation writer lock")
            except MultiSeedRunnerError as error:
                raise MultiSeedRunnerError(
                    "confirmation writer lock exists but cannot be validated"
                ) from error
            live = _lock_is_live(current)
            if live is not False:
                raise MultiSeedRunnerError(
                    "confirmation is already locked by another writer"
                ) from None
            stale_path = lock_path.with_name(
                f".confirmation-writer.stale-{current.get('pid', 'unknown')}-{token[:8]}.json"
            )
            try:
                os.replace(lock_path, stale_path)
            except OSError as error:
                raise MultiSeedRunnerError(
                    f"could not preserve stale writer lock: {error}"
                ) from error
            continue
        try:
            os.write(descriptor, serialized)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        break
    try:
        yield
    finally:
        try:
            current = _read_json(lock_path, "confirmation writer lock")
            if current.get("token") != token:
                raise MultiSeedRunnerError("confirmation writer lock ownership changed")
            lock_path.unlink()
        except FileNotFoundError as error:
            raise MultiSeedRunnerError("confirmation writer lock disappeared") from error


def run_multiseed_confirmation(
    plan_path: str | Path,
    *,
    protocol: ExperimentProtocol,
    resume: bool = False,
    experiment_executor: DevelopmentExecutor = run_development_experiment,
    prediction_exporter: PredictionExporter = export_checkpoint_predictions,
) -> MultiSeedRunResult:
    """Complete all six members and their immutable fold-8 prediction exports."""

    resolved_plan_path = Path(plan_path).resolve()
    plan = load_multiseed_plan(resolved_plan_path, protocol=protocol)
    plan_hash = _hash(plan["artifact_sha256"], "multi-seed plan hash")
    with _writer_lock(resolved_plan_path, plan_hash):
        if not resume and _generated_state_exists(plan):
            raise MultiSeedRunnerError(
                "confirmation outputs already exist; use --resume for the exact plan"
            )
        completions: list[Path] = []
        predictions: list[Path] = []
        artifacts: list[PredictionArtifact] = []
        for raw_entry in _sequence(plan["members"], "members"):
            entry = _mapping(raw_entry, "member entry")
            member_plan_path = _path(entry["member_plan_path"], "member plan path")
            member_plan_hash = _hash(entry["member_plan_sha256"], "member plan hash")
            member = _load_member_plan(
                member_plan_path,
                expected_hash=member_plan_hash,
                protocol=protocol,
            )
            completion_path = _path(member["completion_path"], "completion path")
            if completion_path.exists():
                _, artifact = _verify_completion(
                    completion_path,
                    member,
                    member_plan_path=member_plan_path,
                    member_plan_sha256=member_plan_hash,
                    protocol=protocol,
                )
            else:
                if member["source_kind"] == "reused_sweep_winner":
                    run = _verify_complete_run(
                        member,
                        _path(member["winning_run_dir"], "winning run dir"),
                        protocol=protocol,
                        expected_attempt=None,
                    )
                else:
                    run = _train_member(
                        member,
                        protocol=protocol,
                        resume=resume,
                        experiment_executor=experiment_executor,
                    )
                artifact, npz_hash, artifact_hash = _export_or_verify_prediction(
                    member,
                    run,
                    protocol=protocol,
                    prediction_exporter=prediction_exporter,
                )
                completion = _completion_payload(
                    member,
                    member_plan_path=member_plan_path,
                    member_plan_sha256=member_plan_hash,
                    run=run,
                    prediction_npz_sha256=npz_hash,
                    prediction_artifact_sha256=artifact_hash,
                )
                _atomic_write_new(completion_path, completion)
            artifacts.append(artifact)
            completions.append(completion_path)
            predictions.append(_path(member["prediction_path"], "prediction path"))
        first = artifacts[0]
        for artifact in artifacts[1:]:
            try:
                assert_prediction_artifacts_aligned(first, artifact)
            except ValueError as error:
                raise MultiSeedRunnerError(
                    f"six-member fold-8 predictions are not aligned: {error}"
                ) from error
        return MultiSeedRunResult(
            comparison_id=_string(plan["comparison_id"], "comparison_id"),
            plan_path=resolved_plan_path,
            completion_paths=tuple(completions),
            prediction_paths=tuple(predictions),
        )


def _member_read_status(
    entry: Mapping[str, object],
    *,
    protocol: ExperimentProtocol,
) -> tuple[dict[str, object], PredictionArtifact | None]:
    member_plan_path = _path(entry["member_plan_path"], "member plan path")
    member_plan_hash = _hash(entry["member_plan_sha256"], "member plan hash")
    member = _load_member_plan(
        member_plan_path,
        expected_hash=member_plan_hash,
        protocol=protocol,
    )
    completion_path = _path(member["completion_path"], "completion path")
    base: dict[str, object] = {
        "architecture": member["architecture"],
        "seed": member["seed"],
        "source_kind": member["source_kind"],
        "member_plan_path": str(member_plan_path),
        "member_plan_sha256": member_plan_hash,
        "completion_path": str(completion_path),
        "prediction_path": member["prediction_path"],
        "prediction_json_path": member["prediction_json_path"],
    }
    if completion_path.exists():
        completion, artifact = _verify_completion(
            completion_path,
            member,
            member_plan_path=member_plan_path,
            member_plan_sha256=member_plan_hash,
            protocol=protocol,
        )
        return {
            **base,
            "state": "complete",
            "run_dir": completion["run_dir"],
            "best_epoch": completion["best_epoch"],
            "best_validation_macro_auroc": completion[
                "best_validation_macro_auroc"
            ],
            "completed_epochs": completion["completed_epochs"],
            "prediction_artifact_sha256": completion[
                "prediction_artifact_sha256"
            ],
        }, artifact

    prediction_path, prediction_json = _prediction_paths(member)
    if prediction_path.exists() != prediction_json.exists():
        raise MultiSeedRunnerError("partial prediction artifact exists without completion")
    if member["source_kind"] == "reused_sweep_winner":
        run = _verify_complete_run(
            member,
            _path(member["winning_run_dir"], "winning run dir"),
            protocol=protocol,
            expected_attempt=None,
        )
        if prediction_path.exists():
            artifact, _, integrity = _verify_prediction(member, run, protocol=protocol)
            return {
                **base,
                "state": "prediction_ready_completion_pending",
                "run_dir": str(run.run_dir),
                "best_epoch": run.best_epoch,
                "best_validation_macro_auroc": run.best_macro_auroc,
                "completed_epochs": run.completed_epochs,
                "prediction_artifact_sha256": integrity,
            }, artifact
        return {
            **base,
            "state": "prediction_pending",
            "run_dir": str(run.run_dir),
            "best_epoch": run.best_epoch,
            "best_validation_macro_auroc": run.best_macro_auroc,
            "completed_epochs": run.completed_epochs,
            "prediction_artifact_sha256": None,
        }, None

    latest_state = "planned"
    latest_run: str | None = None
    for attempt in range(MAX_ATTEMPTS):
        plan_path, status_path, run_dir = _attempt_paths(member, attempt)
        if status_path.exists():
            status = _load_attempt_status(status_path, member=member, attempt=attempt)
            latest_state = f"attempt_{status['status']}"
            latest_run = str(run_dir)
            if status["status"] == "complete":
                run = _verify_complete_run(
                    member,
                    run_dir,
                    protocol=protocol,
                    expected_attempt=attempt,
                )
                _verify_attempt_status_run(status, run)
                if prediction_path.exists():
                    artifact, _, integrity = _verify_prediction(
                        member, run, protocol=protocol
                    )
                    return {
                        **base,
                        "state": "prediction_ready_completion_pending",
                        "run_dir": str(run.run_dir),
                        "best_epoch": run.best_epoch,
                        "best_validation_macro_auroc": run.best_macro_auroc,
                        "completed_epochs": run.completed_epochs,
                        "prediction_artifact_sha256": integrity,
                    }, artifact
                return {
                    **base,
                    "state": "prediction_pending",
                    "run_dir": str(run.run_dir),
                    "best_epoch": run.best_epoch,
                    "best_validation_macro_auroc": run.best_macro_auroc,
                    "completed_epochs": run.completed_epochs,
                    "prediction_artifact_sha256": None,
                }, None
        elif run_dir.exists():
            latest_state = "attempt_interrupted_or_running"
            latest_run = str(run_dir)
        elif plan_path.exists():
            latest_state = "attempt_planned"
            latest_run = str(run_dir)
    return {
        **base,
        "state": latest_state,
        "run_dir": latest_run,
        "best_epoch": None,
        "best_validation_macro_auroc": None,
        "completed_epochs": None,
        "prediction_artifact_sha256": None,
    }, None


def read_multiseed_status(
    plan_path: str | Path,
    *,
    protocol: ExperimentProtocol,
) -> dict[str, object]:
    """Read and integrity-check confirmation progress without changing state."""

    resolved_plan_path = Path(plan_path).resolve()
    plan = load_multiseed_plan(resolved_plan_path, protocol=protocol)
    statuses: list[dict[str, object]] = []
    artifacts: list[PredictionArtifact] = []
    for raw_entry in _sequence(plan["members"], "members"):
        status, artifact = _member_read_status(
            _mapping(raw_entry, "member entry"), protocol=protocol
        )
        statuses.append(status)
        if artifact is not None:
            artifacts.append(artifact)
    if len(artifacts) > 1:
        first = artifacts[0]
        for artifact in artifacts[1:]:
            try:
                assert_prediction_artifacts_aligned(first, artifact)
            except ValueError as error:
                raise MultiSeedRunnerError(
                    f"available fold-8 predictions are not aligned: {error}"
                ) from error
    complete = sum(status["state"] == "complete" for status in statuses)
    return {
        "comparison_id": plan["comparison_id"],
        "plan_path": str(resolved_plan_path),
        "plan_sha256": plan["artifact_sha256"],
        "complete_members": complete,
        "required_members": 6,
        "all_complete": complete == 6,
        "aligned_prediction_members": len(artifacts),
        "members": statuses,
    }


__all__ = [
    "ARCHITECTURES",
    "CONFIRMATION_SEEDS",
    "MAX_ATTEMPTS",
    "MEMBER_COMPLETION_TYPE",
    "MULTISEED_PLAN_SCHEMA_VERSION",
    "MultiSeedPlanResult",
    "MultiSeedRunResult",
    "MultiSeedRunnerError",
    "create_multiseed_plan",
    "load_multiseed_plan",
    "read_multiseed_status",
    "run_multiseed_confirmation",
]
