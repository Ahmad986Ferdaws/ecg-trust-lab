"""Immutable multi-seed confirmation freeze for the PTB-XL trustworthy track.

The builder accepts six explicit member-completion receipts.  It never searches
prediction directories and it opens only fold-8 prediction artifacts.  The
resulting JSON is self-hashed, non-overwriting, and contains the six exact
post-sweep refit recipe templates.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Literal, cast

import numpy as np
import torch

from ecg_trust.constants import LEADS
from ecg_trust.data.manifest import sha256_file
from ecg_trust.evaluation import compute_multilabel_metrics
from ecg_trust.multiseed_runner import (
    MultiSeedRunnerError,
    load_multiseed_member_plan,
)
from ecg_trust.predictions import (
    PredictionArtifact,
    PredictionArtifactError,
    assert_prediction_artifacts_aligned,
    load_prediction_artifact,
)
from ecg_trust.protocol import LABEL_ORDER, TRAIN_FOLDS, ExperimentProtocol, FoldRole

FREEZE_SCHEMA_VERSION = 1
FREEZE_ARTIFACT_TYPE = "ecg_trust.multiseed_freeze"
MEMBER_COMPLETION_ARTIFACT_TYPE = "ecg_trust.multiseed_member_completion"
REFIT_RECIPE_SCHEMA_VERSION = 2
REFIT_RECIPE_RUN_KIND = "post_sweep_frozen_refit"
CONFIRMATION_SEEDS: tuple[int, ...] = (2026, 2027, 2028)
ARCHITECTURES: tuple[str, ...] = ("resnet1d", "ecg_transformer")
PRACTICAL_MARGIN = 0.005
FREEZE_PATH_PLACEHOLDER = "${FREEZE_ARTIFACT_PATH}"
FREEZE_HASH_PLACEHOLDER = "${FREEZE_ARTIFACT_SHA256}"
OBJECTIVE = "fold8_uncalibrated_macro_roc_auc"
EPOCH_BUDGET_RULE = (
    "max(warmup_epochs+1,median(selected_zero_based_best_epoch+1_across_seeds))"
)
TIE_POLICY = "practical_tie_designates_resnet1d_by_parsimony"
CARRY_FORWARD_POLICY = "both_architectures_all_three_seeds"

_SHA256_RE = re.compile(r"(?:sha256:)?([0-9a-fA-F]{64})\Z")
_TIMESTAMP_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z\Z")

Architecture = Literal["resnet1d", "ecg_transformer"]


class MultiSeedFreezeError(RuntimeError):
    """Raised when confirmation evidence cannot be frozen safely."""


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise MultiSeedFreezeError(f"{context} must be a string-keyed mapping")
    return cast(Mapping[str, object], value)


def _dict(value: object, context: str) -> dict[str, object]:
    return dict(_mapping(value, context))


def _sequence(value: object, context: str) -> list[object]:
    if not isinstance(value, list):
        raise MultiSeedFreezeError(f"{context} must be a JSON list")
    return value


def _keys(value: Mapping[str, object], *, required: set[str], context: str) -> None:
    missing = sorted(required.difference(value))
    unexpected = sorted(set(value).difference(required))
    if missing or unexpected:
        raise MultiSeedFreezeError(
            f"{context} keys are invalid; missing={missing}, unexpected={unexpected}"
        )


def _string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MultiSeedFreezeError(f"{context} must be a non-empty string")
    return value


def _integer(value: object, context: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise MultiSeedFreezeError(f"{context} must be an integer >= {minimum}")
    return value


def _finite_float(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MultiSeedFreezeError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise MultiSeedFreezeError(f"{context} must be finite")
    return result


def _normalized_sha256(value: object, context: str) -> str:
    text = _string(value, context)
    match = _SHA256_RE.fullmatch(text)
    if match is None:
        raise MultiSeedFreezeError(f"{context} must be a SHA-256 digest")
    return "sha256:" + match.group(1).lower()


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise MultiSeedFreezeError("artifact content must be finite canonical JSON") from error


def canonical_sha256(value: object) -> str:
    """Return the project's canonical, prefixed JSON SHA-256."""

    return "sha256:" + hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    """Return a prefixed SHA-256 for a required file."""

    try:
        return "sha256:" + sha256_file(path)
    except OSError as error:
        raise MultiSeedFreezeError(f"could not hash required file {path}: {error}") from error


def _read_json(path: Path, context: str) -> dict[str, object]:
    if not path.is_file():
        raise MultiSeedFreezeError(f"{context} is missing: {path}")
    try:
        decoded: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise MultiSeedFreezeError(f"could not read {context} {path}: {error}") from error
    return _dict(decoded, context)


def _path(value: object, context: str, *, base_dir: Path) -> Path:
    candidate = Path(_string(value, context))
    return candidate.resolve() if candidate.is_absolute() else (base_dir / candidate).resolve()


def _same_float(left: object, right: float, context: str, *, tolerance: float = 1e-12) -> None:
    observed = _finite_float(left, context)
    if not math.isclose(observed, right, rel_tol=0.0, abs_tol=tolerance):
        raise MultiSeedFreezeError(f"{context} does not match its source evidence")


def _verify_file(path: Path, expected: object, context: str) -> str:
    expected_hash = _normalized_sha256(expected, f"{context} expected hash")
    observed = file_sha256(path)
    if observed != expected_hash:
        raise MultiSeedFreezeError(f"{context} SHA-256 mismatch")
    return observed


def _self_hash_payload(payload: Mapping[str, object]) -> dict[str, object]:
    result = dict(payload)
    if "artifact_sha256" in result:
        raise MultiSeedFreezeError("self-hashed payload already contains artifact_sha256")
    result["artifact_sha256"] = canonical_sha256(result)
    return result


def verify_self_hash(payload: Mapping[str, object], context: str) -> str:
    stored = _normalized_sha256(payload.get("artifact_sha256"), f"{context} hash")
    body = dict(payload)
    body.pop("artifact_sha256", None)
    observed = canonical_sha256(body)
    if observed != stored:
        raise MultiSeedFreezeError(f"{context} self-hash is invalid")
    return stored


def write_new_json(path: Path, payload: Mapping[str, object]) -> None:
    """Commit complete JSON atomically without ever replacing an existing path."""

    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise MultiSeedFreezeError(f"immutable artifact already exists: {path}")
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise MultiSeedFreezeError(f"immutable artifact already exists: {path}") from error
        except OSError as error:
            # A same-directory hard link is the atomic, non-overwriting commit.  Do
            # not fall back to replace/rename semantics that can overwrite data.
            raise MultiSeedFreezeError(f"could not atomically commit {path}: {error}") from error
    finally:
        temporary.unlink(missing_ok=True)


_MEMBER_KEYS = {
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


@dataclass(frozen=True, slots=True)
class VerifiedConfirmationMember:
    """Fully verified fold-8 evidence for one architecture/seed member."""

    completion_path: Path
    completion_sha256: str
    comparison_id: str
    architecture: Architecture
    seed: int
    member_plan_path: Path
    member_plan_sha256: str
    run_dir: Path
    run_metadata_path: Path
    run_metadata_sha256: str
    resolved_config_path: Path
    resolved_config_sha256: str
    history_path: Path
    history_sha256: str
    best_checkpoint_path: Path
    best_checkpoint_sha256: str
    config_hash: str
    protocol_hash: str
    manifest_hash: str
    normalization_sha256: str
    best_epoch: int
    best_validation_macro_auroc: float
    recomputed_macro_auroc: float
    completed_epochs: int
    prediction_path: Path
    prediction_npz_sha256: str
    prediction_json_path: Path
    prediction_artifact_sha256: str
    alignment_sha256: str
    resolved_config: Mapping[str, object]
    prediction: PredictionArtifact

    @property
    def selected_epoch_count(self) -> int:
        return self.best_epoch + 1

    def to_freeze_dict(self) -> dict[str, object]:
        return {
            "architecture": self.architecture,
            "seed": self.seed,
            "status": "complete",
            "member_completion_path": str(self.completion_path),
            "member_completion_sha256": self.completion_sha256,
            "manifest_sha256": self.manifest_hash,
            "normalization_sha256": self.normalization_sha256,
            "member_plan_path": str(self.member_plan_path),
            "member_plan_sha256": self.member_plan_sha256,
            "run_dir": str(self.run_dir),
            "run_metadata_path": str(self.run_metadata_path),
            "run_metadata_sha256": self.run_metadata_sha256,
            "resolved_config_path": str(self.resolved_config_path),
            "resolved_config_file_sha256": self.resolved_config_sha256,
            "resolved_config_hash": self.config_hash,
            "history_path": str(self.history_path),
            "history_sha256": self.history_sha256,
            "best_checkpoint_path": str(self.best_checkpoint_path),
            "best_checkpoint_sha256": self.best_checkpoint_sha256,
            "prediction_path": str(self.prediction_path),
            "prediction_npz_sha256": self.prediction_npz_sha256,
            "prediction_json_path": str(self.prediction_json_path),
            "prediction_artifact_sha256": self.prediction_artifact_sha256,
            "alignment_sha256": self.alignment_sha256,
            "best_epoch": self.best_epoch,
            "selected_epoch_count": self.selected_epoch_count,
            "macro_auroc": self.recomputed_macro_auroc,
        }


def _resolved_config(path: Path, expected_hash: str) -> dict[str, object]:
    wrapper = _read_json(path, "resolved development config")
    _keys(wrapper, required={"config_hash", "config"}, context="resolved config wrapper")
    stored = _normalized_sha256(wrapper["config_hash"], "resolved config hash")
    if stored != expected_hash:
        raise MultiSeedFreezeError("completion config_hash disagrees with resolved config")
    config = _dict(wrapper["config"], "resolved development config body")
    if canonical_sha256(config) != stored:
        raise MultiSeedFreezeError("resolved development config content hash is invalid")
    return config


def _validate_development_config(
    config: Mapping[str, object],
    *,
    architecture: Architecture,
    seed: int,
) -> None:
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
    _keys(config, required=required, context="resolved development config")
    folds = _mapping(config["folds"], "development folds")
    _keys(folds, required={"train", "model_selection"}, context="development folds")
    if folds["train"] != list(TRAIN_FOLDS) or folds["model_selection"] != [8]:
        raise MultiSeedFreezeError("confirmation run must use folds 1-7 and fold 8 only")
    model = _mapping(config["model"], "development model")
    if model.get("architecture") != architecture:
        raise MultiSeedFreezeError("completion architecture disagrees with resolved config")
    runtime = _mapping(config["runtime"], "development runtime")
    if runtime.get("seed") != seed:
        raise MultiSeedFreezeError("completion seed disagrees with resolved config")
    data = _mapping(config["data"], "development data")
    if data.get("max_train_records") is not None or data.get("max_validation_records") is not None:
        raise MultiSeedFreezeError("confirmation may not use record-count subsampling")
    optimization = _mapping(config["optimization"], "development optimization")
    if optimization.get("epochs") != 30:
        raise MultiSeedFreezeError("confirmation scheduler horizon must remain 30 epochs")
    if optimization.get("scheduler") != "warmup_cosine":
        raise MultiSeedFreezeError("confirmation scheduler must remain warmup_cosine")
    optimizer = _mapping(config["optimizer"], "development optimizer")
    _keys(
        optimizer,
        required={"name", "betas", "eps"},
        context="development optimizer",
    )
    if optimizer.get("name") != "AdamW":
        raise MultiSeedFreezeError("confirmation optimizer must remain AdamW")
    betas = optimizer.get("betas")
    if (
        not isinstance(betas, list)
        or len(betas) != 2
        or any(
            isinstance(beta, bool)
            or not isinstance(beta, (int, float))
            or not 0.0 <= float(beta) < 1.0
            for beta in betas
        )
    ):
        raise MultiSeedFreezeError("confirmation AdamW betas are invalid")
    if _finite_float(optimizer.get("eps"), "confirmation AdamW eps") <= 0.0:
        raise MultiSeedFreezeError("confirmation AdamW eps must be positive")


def _history_facts(
    path: Path,
    *,
    completed_epochs: int,
    best_epoch: int,
    best_score: float,
) -> None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise MultiSeedFreezeError(f"could not read development history: {error}") from error
    if len(lines) != completed_epochs:
        raise MultiSeedFreezeError("history length disagrees with completed_epochs")
    selected: Mapping[str, object] | None = None
    for expected_epoch, line in enumerate(lines):
        try:
            raw: object = json.loads(line)
        except json.JSONDecodeError as error:
            raise MultiSeedFreezeError(f"history contains invalid JSON: {error}") from error
        record = _mapping(raw, "history row")
        if record.get("epoch") != expected_epoch:
            raise MultiSeedFreezeError("history epochs must be contiguous and zero-based")
        score = _finite_float(record.get("validation_macro_auroc"), "history macro AUROC")
        if not 0.0 <= score <= 1.0:
            raise MultiSeedFreezeError("history macro AUROC must be in [0, 1]")
        if expected_epoch == best_epoch:
            selected = record
    if selected is None:
        raise MultiSeedFreezeError("history does not contain the selected best epoch")
    _same_float(selected.get("validation_macro_auroc"), best_score, "history best score")
    if selected.get("improved") is not True:
        raise MultiSeedFreezeError("selected history row is not marked improved")
    metrics = _mapping(selected.get("validation_metrics"), "selected validation metrics")
    if metrics.get("label_order") != list(LABEL_ORDER):
        raise MultiSeedFreezeError("selected history label order is not canonical")
    macro = _mapping(metrics.get("macro"), "selected macro metrics")
    if macro.get("roc_auc_labels") != len(LABEL_ORDER):
        raise MultiSeedFreezeError("selected history does not define all five label AUROCs")


def _checkpoint_facts(
    path: Path,
    *,
    config: Mapping[str, object],
    config_hash: str,
    protocol_hash: str,
    manifest_hash: str,
    best_epoch: int,
    best_score: float,
) -> None:
    try:
        decoded: object = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise MultiSeedFreezeError(
            f"could not load best development checkpoint: {error}"
        ) from error
    checkpoint = _mapping(decoded, "best development checkpoint")
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
    _keys(checkpoint, required=required, context="best development checkpoint")
    comparisons: dict[str, object] = {
        "schema_version": 1,
        "epoch": best_epoch,
        "protocol_hash": protocol_hash,
        "manifest_hash": manifest_hash.removeprefix("sha256:"),
        "config_hash": config_hash,
        "config": config,
    }
    for field, expected in comparisons.items():
        observed = checkpoint.get(field)
        if field == "manifest_hash" and isinstance(observed, str):
            observed = observed.removeprefix("sha256:")
        if observed != expected:
            raise MultiSeedFreezeError(f"best checkpoint {field} disagrees with completion")
    stopper = _mapping(checkpoint["early_stopping_state_dict"], "early-stopping state")
    if stopper.get("mode") != "max" or stopper.get("best_epoch") != best_epoch:
        raise MultiSeedFreezeError("best checkpoint is not the selected maximum-mode checkpoint")
    _same_float(stopper.get("best_score"), best_score, "checkpoint best score")
    optimizer_policy = _mapping(config["optimizer"], "resolved optimizer policy")
    optimizer_state = _mapping(
        checkpoint["optimizer_state_dict"], "checkpoint optimizer state"
    )
    parameter_groups = _sequence(
        optimizer_state.get("param_groups"), "checkpoint optimizer parameter groups"
    )
    if not parameter_groups:
        raise MultiSeedFreezeError("checkpoint optimizer has no parameter groups")
    expected_betas = tuple(
        _finite_float(value, "resolved AdamW beta")
        for value in _sequence(optimizer_policy.get("betas"), "resolved AdamW betas")
    )
    expected_eps = _finite_float(optimizer_policy.get("eps"), "resolved AdamW eps")
    expected_weight_decay = _finite_float(
        _mapping(config["optimization"], "resolved optimization").get("weight_decay"),
        "resolved weight decay",
    )
    for raw_group in parameter_groups:
        group = _mapping(raw_group, "checkpoint optimizer parameter group")
        observed_betas = group.get("betas")
        if not isinstance(observed_betas, (list, tuple)) or tuple(
            float(value) for value in observed_betas
        ) != expected_betas:
            raise MultiSeedFreezeError("checkpoint AdamW betas disagree with config")
        _same_float(group.get("eps"), expected_eps, "checkpoint AdamW eps")
        _same_float(
            group.get("weight_decay"),
            expected_weight_decay,
            "checkpoint AdamW weight decay",
        )


def load_confirmation_member(
    completion_path: str | Path,
    *,
    protocol: ExperimentProtocol,
) -> VerifiedConfirmationMember:
    """Verify one explicit completion receipt and all of its fold-8 evidence."""

    path = Path(completion_path).resolve()
    root = _read_json(path, "multi-seed member completion")
    _keys(root, required=_MEMBER_KEYS, context="member completion")
    if root["schema_version"] != 1:
        raise MultiSeedFreezeError("member completion schema_version must be 1")
    if root["artifact_type"] != MEMBER_COMPLETION_ARTIFACT_TYPE:
        raise MultiSeedFreezeError("member completion artifact_type is invalid")
    verify_self_hash(root, "member completion")
    completion_file_hash = file_sha256(path)
    if root["status"] != "complete":
        raise MultiSeedFreezeError("member completion status must be complete")
    architecture_raw = _string(root["architecture"], "member architecture")
    if architecture_raw not in ARCHITECTURES:
        raise MultiSeedFreezeError("member architecture is unsupported")
    architecture = cast(Architecture, architecture_raw)
    seed = _integer(root["seed"], "member seed")
    comparison_id = _string(root["comparison_id"], "member comparison_id")
    base = path.parent
    run_dir = _path(root["run_dir"], "member run_dir", base_dir=base)
    member_plan = _path(root["member_plan_path"], "member plan", base_dir=base)
    metadata_path = _path(root["run_metadata_path"], "run metadata", base_dir=base)
    resolved_path = _path(root["resolved_config_path"], "resolved config", base_dir=base)
    history_path = _path(root["history_path"], "history", base_dir=base)
    checkpoint_path = _path(root["best_checkpoint_path"], "best checkpoint", base_dir=base)
    prediction_path = _path(root["prediction_path"], "fold-8 prediction", base_dir=base)
    prediction_json = _path(
        root["prediction_json_path"], "fold-8 prediction sidecar", base_dir=base
    )
    for artifact_path in (metadata_path, resolved_path, history_path, checkpoint_path):
        if artifact_path.parent != run_dir:
            raise MultiSeedFreezeError("development artifacts must be siblings in run_dir")

    member_plan_hash = _normalized_sha256(
        root["member_plan_sha256"], "member plan artifact hash"
    )
    try:
        member_plan_payload = load_multiseed_member_plan(
            member_plan,
            expected_hash=member_plan_hash,
            protocol=protocol,
        )
    except (OSError, MultiSeedRunnerError, ValueError) as error:
        raise MultiSeedFreezeError(f"invalid member plan: {error}") from error
    metadata_hash = _verify_file(metadata_path, root["run_metadata_sha256"], "run metadata")
    resolved_file_hash = _verify_file(
        resolved_path, root["resolved_config_sha256"], "resolved config"
    )
    history_hash = _verify_file(history_path, root["history_sha256"], "history")
    checkpoint_hash = _verify_file(
        checkpoint_path, root["best_checkpoint_sha256"], "best checkpoint"
    )
    prediction_npz_hash = _verify_file(
        prediction_path, root["prediction_npz_sha256"], "prediction archive"
    )
    if prediction_json != prediction_path.with_suffix(".json"):
        raise MultiSeedFreezeError("prediction JSON must be the NPZ same-stem sidecar")
    if not prediction_json.is_file():
        raise MultiSeedFreezeError("prediction JSON sidecar is missing")

    config_hash = _normalized_sha256(root["config_hash"], "member config_hash")
    protocol_hash = _normalized_sha256(root["protocol_hash"], "member protocol_hash")
    if protocol_hash != protocol.protocol_hash:
        raise MultiSeedFreezeError("member protocol hash disagrees with freeze protocol")
    manifest_hash = _normalized_sha256(root["manifest_hash"], "member manifest_hash")
    normalization_hash = _normalized_sha256(
        root["normalization_sha256"], "member normalization hash"
    )
    expected_member_plan_values: dict[str, object] = {
        "comparison_id": comparison_id,
        "architecture": architecture,
        "seed": seed,
        "source_kind": "reused_sweep_winner" if seed == 2026 else "confirmation_training",
        "protocol_hash": protocol_hash,
        "manifest_sha256": manifest_hash,
        "normalization_sha256": normalization_hash,
    }
    for field, expected in expected_member_plan_values.items():
        if member_plan_payload.get(field) != expected:
            raise MultiSeedFreezeError(f"member plan {field} disagrees with completion")
    expected_member_plan_paths = {
        "completion_path": path,
        "prediction_path": prediction_path,
        "prediction_json_path": prediction_json,
    }
    for field, expected in expected_member_plan_paths.items():
        observed = _path(member_plan_payload.get(field), f"member plan {field}", base_dir=base)
        if observed != expected:
            raise MultiSeedFreezeError(f"member plan {field} disagrees with completion")
    best_epoch = _integer(root["best_epoch"], "member best_epoch")
    completed_epochs = _integer(root["completed_epochs"], "member completed_epochs", minimum=1)
    if best_epoch >= completed_epochs or completed_epochs > 30:
        raise MultiSeedFreezeError("member epoch accounting is invalid")
    best_score = _finite_float(
        root["best_validation_macro_auroc"], "member best macro AUROC"
    )
    if not 0.0 <= best_score <= 1.0:
        raise MultiSeedFreezeError("member best macro AUROC must be in [0, 1]")
    resolved_config = _resolved_config(resolved_path, config_hash)
    _validate_development_config(resolved_config, architecture=architecture, seed=seed)

    metadata = _read_json(metadata_path, "development run metadata")
    expected_metadata: dict[str, object] = {
        "status": "complete",
        "seed": seed,
        "resolved_config_hash": config_hash,
        "protocol_hash": protocol_hash,
        "best_epoch": best_epoch,
        "completed_epochs": completed_epochs,
    }
    for field, expected in expected_metadata.items():
        if metadata.get(field) != expected:
            raise MultiSeedFreezeError(f"run metadata {field} disagrees with completion")
    if _normalized_sha256(metadata.get("manifest_hash"), "metadata manifest hash") != manifest_hash:
        raise MultiSeedFreezeError("run metadata manifest hash disagrees with completion")
    if (
        _normalized_sha256(
            metadata.get("normalization_file_hash"), "metadata normalization hash"
        )
        != normalization_hash
    ):
        raise MultiSeedFreezeError("run metadata normalization hash disagrees with completion")
    _same_float(
        metadata.get("best_validation_macro_auroc"),
        best_score,
        "run metadata best macro AUROC",
    )
    _history_facts(
        history_path,
        completed_epochs=completed_epochs,
        best_epoch=best_epoch,
        best_score=best_score,
    )
    _checkpoint_facts(
        checkpoint_path,
        config=resolved_config,
        config_hash=config_hash,
        protocol_hash=protocol_hash,
        manifest_hash=manifest_hash,
        best_epoch=best_epoch,
        best_score=best_score,
    )

    try:
        prediction = load_prediction_artifact(
            prediction_path,
            protocol=protocol,
            expected_config_hash=config_hash,
            expected_manifest_hash=manifest_hash,
        )
    except (OSError, PredictionArtifactError, ValueError) as error:
        raise MultiSeedFreezeError(f"could not verify fold-8 prediction: {error}") from error
    if prediction.fold_role is not FoldRole.MODEL_SELECTION or prediction.folds != (8,):
        raise MultiSeedFreezeError(
            "confirmation prediction must contain model-selection fold 8 only"
        )
    if prediction.calibrated_probabilities is not None:
        raise MultiSeedFreezeError("confirmation prediction must contain uncalibrated logits only")
    if prediction.model_seed != seed:
        raise MultiSeedFreezeError("prediction seed disagrees with completion")
    prediction_artifact_hash = _normalized_sha256(
        root["prediction_artifact_sha256"], "prediction artifact hash"
    )
    if prediction.integrity_sha256 != prediction_artifact_hash:
        raise MultiSeedFreezeError("prediction artifact hash disagrees with completion")
    extra = prediction.extra_metadata
    if extra.get("lineage") != "development":
        raise MultiSeedFreezeError("confirmation prediction must have development lineage")
    if (
        _normalized_sha256(extra.get("checkpoint_sha256"), "prediction checkpoint hash")
        != checkpoint_hash
    ):
        raise MultiSeedFreezeError("prediction checkpoint hash disagrees with completion")
    if extra.get("checkpoint_epoch") != best_epoch:
        raise MultiSeedFreezeError("prediction checkpoint epoch disagrees with completion")
    resolved_extra = _path(
        extra.get("resolved_config_path"),
        "prediction resolved config path",
        base_dir=prediction_json.parent,
    )
    if resolved_extra != resolved_path:
        raise MultiSeedFreezeError("prediction resolved config path disagrees with completion")
    if (
        _normalized_sha256(extra.get("normalization_sha256"), "prediction normalization hash")
        != normalization_hash
    ):
        raise MultiSeedFreezeError("prediction normalization hash disagrees with completion")

    metrics = compute_multilabel_metrics(
        prediction.targets,
        prediction.probabilities(require_calibrated=False),
        label_order=LABEL_ORDER,
    )
    score = metrics.macro.roc_auc
    if score is None or metrics.macro.roc_auc_labels != len(LABEL_ORDER):
        raise MultiSeedFreezeError("fold-8 prediction does not define all five label AUROCs")
    _same_float(score, best_score, "recomputed fold-8 macro AUROC")
    return VerifiedConfirmationMember(
        completion_path=path,
        completion_sha256=completion_file_hash,
        comparison_id=comparison_id,
        architecture=architecture,
        seed=seed,
        member_plan_path=member_plan,
        member_plan_sha256=member_plan_hash,
        run_dir=run_dir,
        run_metadata_path=metadata_path,
        run_metadata_sha256=metadata_hash,
        resolved_config_path=resolved_path,
        resolved_config_sha256=resolved_file_hash,
        history_path=history_path,
        history_sha256=history_hash,
        best_checkpoint_path=checkpoint_path,
        best_checkpoint_sha256=checkpoint_hash,
        config_hash=config_hash,
        protocol_hash=protocol_hash,
        manifest_hash=manifest_hash,
        normalization_sha256=normalization_hash,
        best_epoch=best_epoch,
        best_validation_macro_auroc=best_score,
        recomputed_macro_auroc=score,
        completed_epochs=completed_epochs,
        prediction_path=prediction_path,
        prediction_npz_sha256=prediction_npz_hash,
        prediction_json_path=prediction_json,
        prediction_artifact_sha256=prediction_artifact_hash,
        alignment_sha256=prediction.alignment_sha256,
        resolved_config=resolved_config,
        prediction=prediction,
    )


@dataclass(frozen=True, slots=True)
class _SweepArchitecture:
    architecture: Architecture
    selection_record: Mapping[str, object]
    selection_record_sha256: str
    trial_number: int
    candidate_index: int
    attempt_index: int
    resolved_config_hash: str
    resolved_config_path: Path
    resolved_config_file_sha256: str
    resolved_config: Mapping[str, object]
    parameters: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class _SweepEvidence:
    path: Path
    file_sha256: str
    comparison_id: str
    protocol_hash: str
    candidate_plan_hash: str
    manifest_hash: str
    normalization_hash: str
    source_provenance: Mapping[str, object]
    architectures: Mapping[str, _SweepArchitecture]


def _load_sweep_summary(path: Path, protocol: ExperimentProtocol) -> _SweepEvidence:
    root = _read_json(path, "completed paired sweep summary")
    if root.get("schema_version") != 2:
        raise MultiSeedFreezeError("sweep summary schema_version must be 2")
    if root.get("all_candidate_budgets_complete") is not True:
        raise MultiSeedFreezeError("both sweep candidate budgets must be complete")
    if root.get("equal_candidate_plan_verified") is not True:
        raise MultiSeedFreezeError("sweep summary did not verify the equal candidate plan")
    if root.get("required_complete_candidates_per_architecture") != 12:
        raise MultiSeedFreezeError("sweep summary does not contain the fixed 12+12 budget")
    protocol_hash = _normalized_sha256(root.get("protocol_hash"), "sweep protocol hash")
    if protocol_hash != protocol.protocol_hash:
        raise MultiSeedFreezeError("sweep protocol hash disagrees with freeze protocol")
    objective = _mapping(root.get("objective"), "sweep objective")
    if (
        objective.get("name") != OBJECTIVE
        or objective.get("direction") != "maximize"
        or objective.get("pruning") != "none"
        or objective.get("required_label_count") != len(LABEL_ORDER)
        or objective.get("require_all_labels_defined") is not True
    ):
        raise MultiSeedFreezeError("sweep objective is not the frozen five-label contract")
    source = _mapping(root.get("source_provenance"), "sweep source provenance")
    manifest_hash = _normalized_sha256(source.get("manifest_sha256"), "sweep manifest hash")
    normalization_hash = _normalized_sha256(
        source.get("normalization_sha256"), "sweep normalization hash"
    )
    best = _mapping(root.get("best_by_architecture"), "sweep winners")
    if set(best) != set(ARCHITECTURES):
        raise MultiSeedFreezeError("sweep summary must expose exactly two architecture winners")
    architectures: dict[str, _SweepArchitecture] = {}
    for architecture_raw in ARCHITECTURES:
        architecture = cast(Architecture, architecture_raw)
        record = _mapping(best[architecture], f"{architecture} sweep winner")
        if record.get("state") != "COMPLETE":
            raise MultiSeedFreezeError("sweep winner must be a COMPLETE trial")
        if record.get("probabilities_calibrated") is not False:
            raise MultiSeedFreezeError("sweep winner must use uncalibrated fold-8 probabilities")
        trial_number = _integer(record.get("trial_number"), "winner trial number")
        candidate_index = _integer(record.get("candidate_index"), "winner candidate index")
        attempt_index = _integer(record.get("attempt_index"), "winner attempt index")
        resolved_hash = _normalized_sha256(
            record.get("resolved_config_hash"), "winner resolved config hash"
        )
        run_dir = _path(record.get("run_dir"), "winner run directory", base_dir=path.parent)
        resolved_path = run_dir / "resolved_config.json"
        artifact_hashes = _mapping(record.get("artifact_sha256"), "winner artifact hashes")
        resolved_file_hash = _verify_file(
            resolved_path,
            artifact_hashes.get("resolved_config.json"),
            "winner resolved config",
        )
        config = _resolved_config(resolved_path, resolved_hash)
        _validate_development_config(config, architecture=architecture, seed=2026)
        parameters = _mapping(record.get("parameters"), "winner parameters")
        optimization = _mapping(config["optimization"], "winner optimization")
        loader = _mapping(config["loader"], "winner loader")
        expected_parameters: dict[str, object] = {
            "batch_size": loader.get("batch_size"),
            "gradient_clip_norm": optimization.get("gradient_clip_norm"),
            "learning_rate": optimization.get("learning_rate"),
            "minimum_lr_ratio": optimization.get("minimum_lr_ratio"),
            "warmup_epochs": optimization.get("warmup_epochs"),
            "weight_decay": optimization.get("weight_decay"),
        }
        if dict(parameters) != expected_parameters:
            raise MultiSeedFreezeError("winner parameters disagree with its resolved config")
        architectures[architecture] = _SweepArchitecture(
            architecture=architecture,
            selection_record=record,
            selection_record_sha256=canonical_sha256(record),
            trial_number=trial_number,
            candidate_index=candidate_index,
            attempt_index=attempt_index,
            resolved_config_hash=resolved_hash,
            resolved_config_path=resolved_path,
            resolved_config_file_sha256=resolved_file_hash,
            resolved_config=config,
            parameters=parameters,
        )
    return _SweepEvidence(
        path=path,
        file_sha256=file_sha256(path),
        comparison_id=_string(root.get("comparison_id"), "sweep comparison_id"),
        protocol_hash=protocol_hash,
        candidate_plan_hash=_normalized_sha256(
            root.get("candidate_plan_hash"), "candidate plan hash"
        ),
        manifest_hash=manifest_hash,
        normalization_hash=normalization_hash,
        source_provenance=source,
        architectures=architectures,
    )


def _scientific_development_config(config: Mapping[str, object]) -> dict[str, object]:
    runtime = _dict(config["runtime"], "development runtime")
    runtime.pop("seed", None)
    return {
        "schema_version": config["schema_version"],
        "folds": config["folds"],
        "data": config["data"],
        "model": config["model"],
        "loader": config["loader"],
        "optimization": config["optimization"],
        "runtime_without_seed": runtime,
        "effective_data": config["effective_data"],
        "optimizer": config["optimizer"],
    }


def _validate_members_against_sweep(
    members: Sequence[VerifiedConfirmationMember],
    sweep: _SweepEvidence,
) -> None:
    expected = {
        (architecture, seed)
        for architecture in ARCHITECTURES
        for seed in CONFIRMATION_SEEDS
    }
    observed = {(member.architecture, member.seed) for member in members}
    if len(members) != len(expected) or observed != expected:
        raise MultiSeedFreezeError(
            "confirmation requires exactly both architectures at seeds 2026, 2027, 2028"
        )
    if len(observed) != len(members):
        raise MultiSeedFreezeError("confirmation contains a duplicate architecture/seed member")
    first = members[0]
    for member in members:
        if member.comparison_id != sweep.comparison_id:
            raise MultiSeedFreezeError("member comparison_id disagrees with sweep")
        if member.protocol_hash != sweep.protocol_hash:
            raise MultiSeedFreezeError("member protocol hash disagrees with sweep")
        if member.manifest_hash != sweep.manifest_hash:
            raise MultiSeedFreezeError("member manifest hash disagrees with sweep")
        if member.normalization_sha256 != sweep.normalization_hash:
            raise MultiSeedFreezeError("member normalization hash disagrees with sweep")
        try:
            assert_prediction_artifacts_aligned(first.prediction, member.prediction)
        except PredictionArtifactError as error:
            raise MultiSeedFreezeError(
                f"confirmation predictions are not aligned: {error}"
            ) from error
        winner = sweep.architectures[member.architecture]
        if _scientific_development_config(member.resolved_config) != _scientific_development_config(
            winner.resolved_config
        ):
            raise MultiSeedFreezeError(
                f"{member.architecture} seed {member.seed} scientific config drifted from winner"
            )


def _architecture_status(delta: float) -> tuple[str, Architecture]:
    if delta >= PRACTICAL_MARGIN:
        return "transformer_selected", "ecg_transformer"
    if delta <= -PRACTICAL_MARGIN:
        return "resnet1d_selected", "resnet1d"
    return "practical_tie", "resnet1d"


def _model_selection(model: Mapping[str, object]) -> dict[str, object]:
    return {
        "architecture": model.get("architecture"),
        "preset": model.get("preset"),
    }


def _refit_recipe_template(
    *,
    member: VerifiedConfirmationMember,
    architecture_mean: float,
    frozen_epochs: int,
    refit_output_root: Path,
    downstream_provenance: Mapping[str, object],
) -> dict[str, object]:
    development = member.resolved_config
    data = _mapping(development["data"], "development data")
    loader = _mapping(development["loader"], "development loader")
    optimization = _mapping(development["optimization"], "development optimization")
    runtime = _mapping(development["runtime"], "development runtime")
    template: dict[str, object] = {
        "schema_version": REFIT_RECIPE_SCHEMA_VERSION,
        "run_kind": REFIT_RECIPE_RUN_KIND,
        "freeze_artifact": FREEZE_PATH_PLACEHOLDER,
        "freeze_artifact_sha256": FREEZE_HASH_PLACEHOLDER,
        "comparison_id": member.comparison_id,
        "architecture": member.architecture,
        "confirmation_seed": member.seed,
        "run_name": f"{member.architecture}_refit_folds1-8_seed{member.seed}",
        "initialization": "fresh",
        "folds": {
            "refit": list(range(1, 9)),
            "normalization": list(TRAIN_FOLDS),
        },
        "data": {
            "manifest": data["manifest"],
            "dataset_root": data["dataset_root"],
            "normalization": data["normalization"],
        },
        "source": {
            "member_completion": str(member.completion_path),
            "member_completion_sha256": member.completion_sha256,
            "manifest_sha256": member.manifest_hash,
            "normalization_sha256": member.normalization_sha256,
            "run_metadata": str(member.run_metadata_path),
            "run_metadata_sha256": member.run_metadata_sha256,
            "resolved_config": str(member.resolved_config_path),
            "resolved_config_file_sha256": member.resolved_config_sha256,
            "resolved_config_hash": member.config_hash,
            "history": str(member.history_path),
            "history_sha256": member.history_sha256,
            "best_checkpoint": str(member.best_checkpoint_path),
            "best_checkpoint_sha256": member.best_checkpoint_sha256,
            "prediction": str(member.prediction_path),
            "prediction_npz_sha256": member.prediction_npz_sha256,
            "prediction_json": str(member.prediction_json_path),
            "prediction_artifact_sha256": member.prediction_artifact_sha256,
            "best_epoch": member.best_epoch,
            "best_validation_macro_auroc": member.recomputed_macro_auroc,
        },
        "selection": {
            "objective": OBJECTIVE,
            "architecture_mean_macro_auroc": architecture_mean,
            "frozen_epochs": frozen_epochs,
            "epoch_budget_rule": EPOCH_BUDGET_RULE,
        },
        "model": _model_selection(_mapping(development["model"], "development model")),
        "model_identity": dict(
            _mapping(development["model"], "development model identity")
        ),
        "loader": dict(loader),
        "optimization": {
            key: optimization[key]
            for key in (
                "learning_rate",
                "weight_decay",
                "warmup_epochs",
                "minimum_lr_ratio",
                "gradient_clip_norm",
                "scheduler",
            )
        },
        "optimizer": dict(
            _mapping(development["optimizer"], "development optimizer policy")
        ),
        "runtime": {
            "seed": member.seed,
            "device": runtime["device"],
            "bf16": runtime["bf16"],
        },
        "output": {"root_dir": str(refit_output_root)},
        "downstream_provenance": dict(downstream_provenance),
    }
    template["recipe_sha256"] = canonical_sha256(template)
    return template


@dataclass(frozen=True, slots=True)
class FreezeCreation:
    """Reproducible creation metadata; injectable for deterministic tests."""

    timestamp_utc: str
    code_revision: str
    dependency_lock_sha256: str
    software_versions: Mapping[str, str]

    def to_dict(self) -> dict[str, object]:
        if _TIMESTAMP_RE.fullmatch(self.timestamp_utc) is None:
            raise MultiSeedFreezeError("freeze timestamp must use YYYY-MM-DDTHH:MM:SSZ")
        return {
            "timestamp_utc": self.timestamp_utc,
            "code_revision": _string(self.code_revision, "freeze code revision"),
            "dependency_lock_sha256": _normalized_sha256(
                self.dependency_lock_sha256, "dependency lock hash"
            ),
            "software_versions": dict(self.software_versions),
        }


def _git_revision(project_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    return result.stdout.strip() or "unavailable"


def default_freeze_creation(project_root: Path) -> FreezeCreation:
    downstream = capture_downstream_provenance(project_root)
    return FreezeCreation(
        timestamp_utc=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        code_revision=cast(str, downstream["code_revision"]),
        dependency_lock_sha256=cast(str, downstream["dependency_lock_sha256"]),
        software_versions={
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": str(torch.__version__),
        },
    )


def capture_downstream_provenance(project_root: str | Path) -> dict[str, object]:
    """Capture the code and dependency state authorized for downstream work."""

    resolved_root = Path(project_root).resolve()
    return {
        "project_root": str(resolved_root),
        "code_revision": _git_revision(resolved_root),
        "dependency_lock_sha256": file_sha256(resolved_root / "uv.lock"),
    }


def create_multiseed_freeze_payload(
    *,
    sweep_summary_path: str | Path,
    member_completion_paths: Sequence[str | Path],
    protocol: ExperimentProtocol,
    refit_output_root: str | Path,
    creation: FreezeCreation | None = None,
) -> dict[str, object]:
    """Build and verify the canonical freeze payload without writing it."""

    sweep_path = Path(sweep_summary_path).resolve()
    sweep = _load_sweep_summary(sweep_path, protocol)
    source_project_root = _path(
        sweep.source_provenance.get("project_root"),
        "sweep project root",
        base_dir=sweep_path.parent,
    )
    created = creation or default_freeze_creation(source_project_root)
    created_payload = created.to_dict()
    downstream_provenance = {
        "project_root": str(source_project_root),
        "code_revision": created_payload["code_revision"],
        "dependency_lock_sha256": created_payload["dependency_lock_sha256"],
    }
    members = [
        load_confirmation_member(completion, protocol=protocol)
        for completion in member_completion_paths
    ]
    _validate_members_against_sweep(members, sweep)
    members.sort(key=lambda item: (ARCHITECTURES.index(item.architecture), item.seed))
    by_architecture = {
        architecture: [member for member in members if member.architecture == architecture]
        for architecture in ARCHITECTURES
    }
    architecture_payloads: dict[str, object] = {}
    means: dict[str, float] = {}
    frozen_epochs: dict[str, int] = {}
    for architecture in ARCHITECTURES:
        architecture_members = by_architecture[architecture]
        scores = [member.recomputed_macro_auroc for member in architecture_members]
        mean_score = math.fsum(scores) / len(scores)
        winner = sweep.architectures[architecture]
        optimization = _mapping(winner.resolved_config["optimization"], "winner optimization")
        warmup = _integer(optimization.get("warmup_epochs"), "winner warmup epochs")
        selected_counts = [member.selected_epoch_count for member in architecture_members]
        budget = max(warmup + 1, int(median(selected_counts)))
        if budget > 30:
            raise MultiSeedFreezeError("frozen refit epoch budget exceeds the 30-epoch ceiling")
        means[architecture] = mean_score
        frozen_epochs[architecture] = budget
        model = _mapping(winner.resolved_config["model"], "winner model metadata")
        _integer(
            model.get("trainable_parameters"), "model parameter count", minimum=1
        )
        optimizer = _mapping(winner.resolved_config["optimizer"], "winner optimizer")
        architecture_payloads[architecture] = {
            "winning_sweep_trial": {
                "trial_number": winner.trial_number,
                "candidate_index": winner.candidate_index,
                "attempt_index": winner.attempt_index,
                "selection_record_sha256": winner.selection_record_sha256,
                "resolved_config_path": str(winner.resolved_config_path),
                "resolved_config_file_sha256": winner.resolved_config_file_sha256,
                "resolved_config_hash": winner.resolved_config_hash,
                "hyperparameters": dict(winner.parameters),
                "hyperparameters_sha256": canonical_sha256(winner.parameters),
            },
            "model": dict(model),
            "optimizer": dict(optimizer),
            "confirmation_members": [member.to_freeze_dict() for member in architecture_members],
            "per_seed_scores": {
                str(member.seed): member.recomputed_macro_auroc
                for member in architecture_members
            },
            "mean_macro_auroc": mean_score,
            "selected_epoch_counts": selected_counts,
            "warmup_epochs": warmup,
            "frozen_refit_epochs": budget,
        }

    delta = means["ecg_transformer"] - means["resnet1d"]
    status, primary = _architecture_status(delta)
    paired_differences = {
        str(seed): next(
            member.recomputed_macro_auroc
            for member in by_architecture["ecg_transformer"]
            if member.seed == seed
        )
        - next(
            member.recomputed_macro_auroc
            for member in by_architecture["resnet1d"]
            if member.seed == seed
        )
        for seed in CONFIRMATION_SEEDS
    }
    resolved_refit_root = Path(refit_output_root).resolve()
    recipes = [
        _refit_recipe_template(
            member=member,
            architecture_mean=means[member.architecture],
            frozen_epochs=frozen_epochs[member.architecture],
            refit_output_root=resolved_refit_root,
            downstream_provenance=downstream_provenance,
        )
        for member in members
    ]
    payload: dict[str, object] = {
        "schema_version": FREEZE_SCHEMA_VERSION,
        "artifact_type": FREEZE_ARTIFACT_TYPE,
        "comparison_id": sweep.comparison_id,
        "protocol_hash": sweep.protocol_hash,
        "manifest_hash": sweep.manifest_hash,
        "normalization_hash": sweep.normalization_hash,
        "label_order": list(LABEL_ORDER),
        "input_resolution": {
            "dataset_version": protocol.dataset_version,
            "sampling_frequency_hz": 100.0,
            "samples_per_record": 1000,
        },
        "input_shape": [len(LEADS), 1000],
        "lead_order": list(LEADS),
        "fold_roles": protocol.to_resolved_dict()["folds"]["roles"],  # type: ignore[index]
        "sweep_sources": {
            "sweep_summary_path": str(sweep.path),
            "sweep_summary_sha256": sweep.file_sha256,
            "candidate_manifest_hash": sweep.candidate_plan_hash,
            "resnet_selection_artifact_hash": sweep.architectures[
                "resnet1d"
            ].selection_record_sha256,
            "transformer_selection_artifact_hash": sweep.architectures[
                "ecg_transformer"
            ].selection_record_sha256,
            "source_provenance": dict(sweep.source_provenance),
            "refit_output_root": str(resolved_refit_root),
        },
        "confirmation_plan": {
            "seeds": list(CONFIRMATION_SEEDS),
            "objective": OBJECTIVE,
            "direction": "maximize",
            "practical_margin": PRACTICAL_MARGIN,
            "tie_policy": TIE_POLICY,
            "epoch_budget_rule": EPOCH_BUDGET_RULE,
            "carry_forward_policy": CARRY_FORWARD_POLICY,
        },
        "architectures": architecture_payloads,
        "decision": {
            "delta_transformer_minus_resnet1d": delta,
            "status": status,
            "primary_architecture": primary,
            "frozen_comparators": list(ARCHITECTURES),
            "paired_seed_differences": paired_differences,
            "practical_margin": PRACTICAL_MARGIN,
        },
        "refit_recipes": recipes,
        "created": created_payload,
    }
    return _self_hash_payload(payload)


_FREEZE_KEYS = {
    "schema_version",
    "artifact_type",
    "comparison_id",
    "protocol_hash",
    "manifest_hash",
    "normalization_hash",
    "label_order",
    "input_resolution",
    "input_shape",
    "lead_order",
    "fold_roles",
    "sweep_sources",
    "confirmation_plan",
    "architectures",
    "decision",
    "refit_recipes",
    "created",
    "artifact_sha256",
}


def _validate_recipe_templates(root: Mapping[str, object]) -> None:
    architectures = _mapping(root["architectures"], "freeze architectures")
    recipes = _sequence(root["refit_recipes"], "freeze refit_recipes")
    if len(recipes) != 6:
        raise MultiSeedFreezeError("freeze must contain exactly six refit recipes")
    seen: set[tuple[str, int]] = set()
    for raw in recipes:
        recipe = _mapping(raw, "refit recipe")
        stored_recipe_hash = _normalized_sha256(recipe.get("recipe_sha256"), "recipe hash")
        body = dict(recipe)
        body.pop("recipe_sha256", None)
        if canonical_sha256(body) != stored_recipe_hash:
            raise MultiSeedFreezeError("refit recipe self-hash is invalid")
        if recipe.get("schema_version") != REFIT_RECIPE_SCHEMA_VERSION:
            raise MultiSeedFreezeError("refit recipe schema_version is invalid")
        if recipe.get("run_kind") != REFIT_RECIPE_RUN_KIND:
            raise MultiSeedFreezeError("refit recipe run_kind is invalid")
        if recipe.get("freeze_artifact") != FREEZE_PATH_PLACEHOLDER or recipe.get(
            "freeze_artifact_sha256"
        ) != FREEZE_HASH_PLACEHOLDER:
            raise MultiSeedFreezeError("embedded refit recipe freeze placeholders are invalid")
        architecture = _string(recipe.get("architecture"), "recipe architecture")
        seed = _integer(recipe.get("confirmation_seed"), "recipe seed")
        identity = (architecture, seed)
        if identity in seen:
            raise MultiSeedFreezeError("freeze contains a duplicate refit recipe")
        seen.add(identity)
        if architecture not in ARCHITECTURES or seed not in CONFIRMATION_SEEDS:
            raise MultiSeedFreezeError("refit recipe architecture/seed is not frozen")
        if recipe.get("confirmation_seed") != _mapping(
            recipe.get("runtime"), "recipe runtime"
        ).get("seed"):
            raise MultiSeedFreezeError("refit recipe runtime seed drifted")
        folds = _mapping(recipe.get("folds"), "recipe folds")
        if folds.get("refit") != list(range(1, 9)) or folds.get("normalization") != list(
            TRAIN_FOLDS
        ):
            raise MultiSeedFreezeError("refit recipe may use only folds 1-8 and normalization 1-7")
        architecture_payload = _mapping(architectures[architecture], "architecture freeze")
        selection = _mapping(recipe.get("selection"), "recipe selection")
        if selection.get("frozen_epochs") != architecture_payload.get("frozen_refit_epochs"):
            raise MultiSeedFreezeError("refit recipe epoch budget drifted from architecture freeze")
        source = _mapping(recipe.get("source"), "recipe source")
        members = _sequence(
            architecture_payload.get("confirmation_members"), "architecture members"
        )
        member = next(
            (
                _mapping(candidate, "architecture member")
                for candidate in members
                if _mapping(candidate, "architecture member").get("seed") == seed
            ),
            None,
        )
        if member is None:
            raise MultiSeedFreezeError("refit recipe has no corresponding frozen member")
        source_checks = {
            "member_completion_sha256": "member_completion_sha256",
            "manifest_sha256": "manifest_sha256",
            "normalization_sha256": "normalization_sha256",
            "run_metadata_sha256": "run_metadata_sha256",
            "resolved_config_hash": "resolved_config_hash",
            "history_sha256": "history_sha256",
            "best_checkpoint_sha256": "best_checkpoint_sha256",
            "prediction_artifact_sha256": "prediction_artifact_sha256",
            "best_epoch": "best_epoch",
            "best_validation_macro_auroc": "macro_auroc",
        }
        for recipe_key, member_key in source_checks.items():
            if source.get(recipe_key) != member.get(member_key):
                raise MultiSeedFreezeError(f"refit recipe source {recipe_key} drifted")
        if source.get("manifest_sha256") != root.get("manifest_hash"):
            raise MultiSeedFreezeError("refit recipe manifest hash drifted from freeze")
        if source.get("normalization_sha256") != root.get("normalization_hash"):
            raise MultiSeedFreezeError("refit recipe normalization hash drifted from freeze")
        if recipe.get("model_identity") != architecture_payload.get("model"):
            raise MultiSeedFreezeError("refit recipe model identity drifted from freeze")
        if recipe.get("optimizer") != architecture_payload.get("optimizer"):
            raise MultiSeedFreezeError("refit recipe optimizer policy drifted from freeze")
        created = _mapping(root.get("created"), "freeze creation")
        sweep_sources = _mapping(root.get("sweep_sources"), "freeze sweep sources")
        source_provenance = _mapping(
            sweep_sources.get("source_provenance"), "sweep source provenance"
        )
        expected_downstream = {
            "project_root": source_provenance.get("project_root"),
            "code_revision": created.get("code_revision"),
            "dependency_lock_sha256": created.get("dependency_lock_sha256"),
        }
        if recipe.get("downstream_provenance") != expected_downstream:
            raise MultiSeedFreezeError("refit recipe downstream provenance drifted")
    expected = {
        (architecture, seed)
        for architecture in ARCHITECTURES
        for seed in CONFIRMATION_SEEDS
    }
    if seen != expected:
        raise MultiSeedFreezeError("freeze refit recipes do not cover the fixed six members")


def _validate_internal_freeze(root: Mapping[str, object], protocol: ExperimentProtocol) -> None:
    _keys(root, required=_FREEZE_KEYS, context="multi-seed freeze")
    if root["schema_version"] != FREEZE_SCHEMA_VERSION:
        raise MultiSeedFreezeError("freeze schema_version is unsupported")
    if root["artifact_type"] != FREEZE_ARTIFACT_TYPE:
        raise MultiSeedFreezeError("freeze artifact_type is invalid")
    verify_self_hash(root, "multi-seed freeze")
    if _normalized_sha256(root["protocol_hash"], "freeze protocol hash") != protocol.protocol_hash:
        raise MultiSeedFreezeError("freeze protocol hash disagrees with supplied protocol")
    if root["label_order"] != list(LABEL_ORDER) or root["lead_order"] != list(LEADS):
        raise MultiSeedFreezeError("freeze label or lead order is not canonical")
    if root["input_shape"] != [len(LEADS), 1000]:
        raise MultiSeedFreezeError("freeze input shape is not canonical")
    plan = _mapping(root["confirmation_plan"], "freeze confirmation plan")
    expected_plan: dict[str, object] = {
        "seeds": list(CONFIRMATION_SEEDS),
        "objective": OBJECTIVE,
        "direction": "maximize",
        "practical_margin": PRACTICAL_MARGIN,
        "tie_policy": TIE_POLICY,
        "epoch_budget_rule": EPOCH_BUDGET_RULE,
        "carry_forward_policy": CARRY_FORWARD_POLICY,
    }
    if dict(plan) != expected_plan:
        raise MultiSeedFreezeError("freeze confirmation plan drifted from the fixed contract")
    architectures = _mapping(root["architectures"], "freeze architectures")
    if set(architectures) != set(ARCHITECTURES):
        raise MultiSeedFreezeError("freeze architectures must be ResNet and Transformer")
    recomputed_means: dict[str, float] = {}
    for architecture in ARCHITECTURES:
        payload = _mapping(architectures[architecture], f"{architecture} freeze")
        members = _sequence(payload.get("confirmation_members"), "confirmation members")
        if len(members) != 3:
            raise MultiSeedFreezeError("each architecture requires exactly three members")
        seeds = [_integer(_mapping(item, "member").get("seed"), "member seed") for item in members]
        if tuple(seeds) != CONFIRMATION_SEEDS:
            raise MultiSeedFreezeError("architecture members must use ordered fixed seeds")
        for raw_member in members:
            member = _mapping(raw_member, "confirmation member")
            if member.get("manifest_sha256") != root.get("manifest_hash"):
                raise MultiSeedFreezeError("confirmation member manifest hash drifted")
            if member.get("normalization_sha256") != root.get("normalization_hash"):
                raise MultiSeedFreezeError("confirmation member normalization hash drifted")
        scores = [
            _finite_float(_mapping(item, "member").get("macro_auroc"), "member score")
            for item in members
        ]
        mean_score = math.fsum(scores) / len(scores)
        _same_float(payload.get("mean_macro_auroc"), mean_score, "architecture mean")
        per_seed = _mapping(payload.get("per_seed_scores"), "per-seed scores")
        if set(per_seed) != {str(seed) for seed in CONFIRMATION_SEEDS}:
            raise MultiSeedFreezeError("per-seed scores do not cover fixed seeds")
        for seed, score in zip(CONFIRMATION_SEEDS, scores, strict=True):
            _same_float(per_seed[str(seed)], score, "per-seed score")
        counts = [
            _integer(_mapping(item, "member").get("selected_epoch_count"), "epoch count", minimum=1)
            for item in members
        ]
        if payload.get("selected_epoch_counts") != counts:
            raise MultiSeedFreezeError("architecture selected epoch counts drifted")
        warmup = _integer(payload.get("warmup_epochs"), "architecture warmup epochs")
        expected_epochs = max(warmup + 1, int(median(counts)))
        if expected_epochs > 30 or payload.get("frozen_refit_epochs") != expected_epochs:
            raise MultiSeedFreezeError("architecture frozen refit epoch budget is invalid")
        recomputed_means[architecture] = mean_score
    delta = recomputed_means["ecg_transformer"] - recomputed_means["resnet1d"]
    status, primary = _architecture_status(delta)
    decision = _mapping(root["decision"], "freeze decision")
    _same_float(
        decision.get("delta_transformer_minus_resnet1d"), delta, "architecture delta"
    )
    if decision.get("status") != status or decision.get("primary_architecture") != primary:
        raise MultiSeedFreezeError("freeze architecture decision is not reproducible")
    if decision.get("frozen_comparators") != list(ARCHITECTURES):
        raise MultiSeedFreezeError("both architectures must remain frozen comparators")
    if decision.get("practical_margin") != PRACTICAL_MARGIN:
        raise MultiSeedFreezeError("freeze practical margin drifted")
    _validate_recipe_templates(root)


@dataclass(frozen=True, slots=True)
class MultiSeedFreezeArtifact:
    """Loaded immutable freeze and its verified canonical payload."""

    path: Path
    artifact_sha256: str
    _canonical_payload: str

    @property
    def payload(self) -> dict[str, object]:
        decoded: object = json.loads(self._canonical_payload)
        return cast(dict[str, object], decoded)

    @property
    def comparison_id(self) -> str:
        return cast(str, self.payload["comparison_id"])

    @property
    def manifest_sha256(self) -> str:
        return _normalized_sha256(self.payload["manifest_hash"], "freeze manifest hash")

    @property
    def normalization_sha256(self) -> str:
        return _normalized_sha256(
            self.payload["normalization_hash"], "freeze normalization hash"
        )

    def recipe_template(self, architecture: str, seed: int) -> dict[str, object]:
        for raw in cast(list[object], self.payload["refit_recipes"]):
            recipe = _dict(raw, "refit recipe")
            if recipe.get("architecture") == architecture and recipe.get(
                "confirmation_seed"
            ) == seed:
                return recipe
        raise MultiSeedFreezeError(f"freeze has no refit recipe for {architecture}/seed{seed}")


def load_multiseed_freeze(
    path: str | Path,
    *,
    protocol: ExperimentProtocol,
    verify_sources: bool = True,
) -> MultiSeedFreezeArtifact:
    """Load, self-verify, and optionally re-verify every source artifact."""

    freeze_path = Path(path).resolve()
    root = _read_json(freeze_path, "multi-seed freeze")
    _validate_internal_freeze(root, protocol)
    if verify_sources:
        sources = _mapping(root["sweep_sources"], "freeze sweep sources")
        sweep_path = _path(
            sources.get("sweep_summary_path"), "freeze sweep summary", base_dir=freeze_path.parent
        )
        _verify_file(sweep_path, sources.get("sweep_summary_sha256"), "freeze sweep summary")
        architectures = _mapping(root["architectures"], "freeze architectures")
        completions: list[Path] = []
        for architecture in ARCHITECTURES:
            members = _sequence(
                _mapping(architectures[architecture], "architecture freeze").get(
                    "confirmation_members"
                ),
                "confirmation members",
            )
            for raw in members:
                member = _mapping(raw, "confirmation member")
                completion = _path(
                    member.get("member_completion_path"),
                    "member completion path",
                    base_dir=freeze_path.parent,
                )
                _verify_file(
                    completion,
                    member.get("member_completion_sha256"),
                    "member completion",
                )
                completions.append(completion)
        created_map = _mapping(root["created"], "freeze creation")
        versions_raw = _mapping(created_map.get("software_versions"), "freeze versions")
        versions = {
            key: _string(value, f"software version {key}")
            for key, value in versions_raw.items()
        }
        creation = FreezeCreation(
            timestamp_utc=_string(created_map.get("timestamp_utc"), "freeze timestamp"),
            code_revision=_string(created_map.get("code_revision"), "freeze code revision"),
            dependency_lock_sha256=_normalized_sha256(
                created_map.get("dependency_lock_sha256"), "freeze lock hash"
            ),
            software_versions=versions,
        )
        rebuilt = create_multiseed_freeze_payload(
            sweep_summary_path=sweep_path,
            member_completion_paths=completions,
            protocol=protocol,
            refit_output_root=_path(
                sources.get("refit_output_root"),
                "freeze refit output root",
                base_dir=freeze_path.parent,
            ),
            creation=creation,
        )
        if rebuilt != root:
            raise MultiSeedFreezeError("freeze payload no longer matches its source evidence")
    return MultiSeedFreezeArtifact(
        path=freeze_path,
        artifact_sha256=_normalized_sha256(root["artifact_sha256"], "freeze hash"),
        _canonical_payload=_canonical_json(root),
    )


def write_multiseed_freeze(
    output_path: str | Path,
    *,
    sweep_summary_path: str | Path,
    member_completion_paths: Sequence[str | Path],
    protocol: ExperimentProtocol,
    refit_output_root: str | Path,
    creation: FreezeCreation | None = None,
) -> MultiSeedFreezeArtifact:
    """Verify all evidence and atomically commit a new immutable freeze."""

    output = Path(output_path).resolve()
    payload = create_multiseed_freeze_payload(
        sweep_summary_path=sweep_summary_path,
        member_completion_paths=member_completion_paths,
        protocol=protocol,
        refit_output_root=refit_output_root,
        creation=creation,
    )
    write_new_json(output, payload)
    return load_multiseed_freeze(output, protocol=protocol, verify_sources=True)


def materialize_refit_recipes(
    freeze: MultiSeedFreezeArtifact,
    output_dir: str | Path,
) -> tuple[Path, ...]:
    """Recoverably materialize six exact recipes without overwriting any file."""

    directory = Path(output_dir).resolve()
    planned: list[tuple[Path, dict[str, object]]] = []
    for architecture in ARCHITECTURES:
        for seed in CONFIRMATION_SEEDS:
            recipe = freeze.recipe_template(architecture, seed)
            recipe["freeze_artifact"] = str(freeze.path)
            recipe["freeze_artifact_sha256"] = freeze.artifact_sha256
            target = directory / f"{architecture}-seed{seed}-refit.json"
            planned.append((target, recipe))
    _preflight_recipe_destinations(planned)
    committed: list[Path] = []
    try:
        for target, recipe in planned:
            if not target.exists():
                write_new_json(target, recipe)
                committed.append(target)
    except (OSError, MultiSeedFreezeError):
        for target in committed:
            target.unlink(missing_ok=True)
        raise
    return tuple(target for target, _ in planned)


def _preflight_recipe_destinations(
    planned: Sequence[tuple[Path, Mapping[str, object]]],
) -> None:
    for target, expected in planned:
        if target.exists():
            observed = _read_json(target, "staged refit recipe")
            if observed != dict(expected):
                raise MultiSeedFreezeError(
                    f"refit recipe destination contains different content: {target}"
                )


def publish_multiseed_freeze_bundle(
    output_path: str | Path,
    *,
    recipes_dir: str | Path,
    sweep_summary_path: str | Path,
    member_completion_paths: Sequence[str | Path],
    protocol: ExperimentProtocol,
    refit_output_root: str | Path,
    creation: FreezeCreation | None = None,
) -> tuple[MultiSeedFreezeArtifact, tuple[Path, ...]]:
    """Publish six recoverable recipes first and the authoritative freeze last."""

    output = Path(output_path).resolve()
    if output.exists():
        raise MultiSeedFreezeError(f"immutable artifact already exists: {output}")
    payload = create_multiseed_freeze_payload(
        sweep_summary_path=sweep_summary_path,
        member_completion_paths=member_completion_paths,
        protocol=protocol,
        refit_output_root=refit_output_root,
        creation=creation,
    )
    freeze_hash = _normalized_sha256(payload["artifact_sha256"], "freeze hash")
    directory = Path(recipes_dir).resolve()
    planned: list[tuple[Path, dict[str, object]]] = []
    raw_recipes = _sequence(payload["refit_recipes"], "freeze refit recipes")
    by_identity: dict[tuple[str, int], dict[str, object]] = {}
    for raw in raw_recipes:
        recipe = _dict(raw, "freeze refit recipe")
        identity = (
            _string(recipe.get("architecture"), "recipe architecture"),
            _integer(recipe.get("confirmation_seed"), "recipe seed"),
        )
        by_identity[identity] = recipe
    for architecture in ARCHITECTURES:
        for seed in CONFIRMATION_SEEDS:
            recipe = dict(by_identity[(architecture, seed)])
            recipe["freeze_artifact"] = str(output)
            recipe["freeze_artifact_sha256"] = freeze_hash
            planned.append(
                (directory / f"{architecture}-seed{seed}-refit.json", recipe)
            )
    destinations = [output, *(target for target, _ in planned)]
    if len(set(destinations)) != len(destinations):
        raise MultiSeedFreezeError("freeze and recipe destinations must be distinct")
    _preflight_recipe_destinations(planned)
    committed: list[Path] = []
    try:
        for target, recipe in planned:
            if not target.exists():
                write_new_json(target, recipe)
                committed.append(target)
        write_new_json(output, payload)
    except (OSError, MultiSeedFreezeError):
        if not output.exists():
            for target in committed:
                target.unlink(missing_ok=True)
        raise
    freeze = load_multiseed_freeze(output, protocol=protocol, verify_sources=True)
    return freeze, tuple(target for target, _ in planned)


def normalized_recipe_template(recipe: Mapping[str, object]) -> dict[str, object]:
    """Replace materialized freeze bindings with canonical membership tokens."""

    normalized = dict(recipe)
    normalized["freeze_artifact"] = FREEZE_PATH_PLACEHOLDER
    normalized["freeze_artifact_sha256"] = FREEZE_HASH_PLACEHOLDER
    return normalized
