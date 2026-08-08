"""Sealed six-member release gates between refit, calibration, and final testing.

This module deliberately sits above the individual training and prediction
APIs.  A caller cannot export fold 9 from a convenient single checkpoint: it
must first prove that the complete, preregistered two-architecture by
three-seed refit grid exists.  The resulting bundles are immutable,
content-addressed JSON records whose source files are re-hashed whenever a
downstream gate is opened.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from ecg_trust.constants import PTBXL_VERSION
from ecg_trust.decisioning import (
    CalibrationDecisionArtifact,
    fit_calibration_decisions,
    load_calibration_decisions,
    save_calibration_decisions,
)
from ecg_trust.multiseed_freeze import (
    load_multiseed_freeze,
    normalized_recipe_template,
)
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
    CALIBRATION_FOLDS,
    LABEL_ORDER,
    TRAIN_FOLDS,
    ExperimentProtocol,
    FoldRole,
)
from ecg_trust.refit_runner import load_refit_completion

REFIT_BUNDLE_SCHEMA_VERSION = 1
REFIT_BUNDLE_TYPE = "ecg_trust.refit_release_bundle"
CALIBRATION_BUNDLE_SCHEMA_VERSION = 1
CALIBRATION_BUNDLE_TYPE = "ecg_trust.calibration_release_bundle"
FOLD9_EXPORT_PLAN_TYPE = "ecg_trust.fold9_export_batch_plan"
CALIBRATION_FIT_PLAN_TYPE = "ecg_trust.calibration_fit_batch_plan"
EXPECTED_ARCHITECTURES: tuple[str, ...] = ("resnet1d", "ecg_transformer")
EXPECTED_SEEDS: tuple[int, ...] = (2026, 2027, 2028)
REFIT_FOLDS: tuple[int, ...] = tuple(range(1, 9))

JsonValue = object
PredictionExporter = Callable[..., PredictionExportResult]


class ReleaseGateError(ValueError):
    """Raised when a release stage violates the sealed scientific contract."""


class ReleaseIntegrityError(ReleaseGateError):
    """Raised when an artifact or one of its bound source files was changed."""


class ReleaseStateError(ReleaseGateError):
    """Raised when a lifecycle action is attempted in the wrong order."""


def canonical_sha256(payload: Mapping[str, object]) -> str:
    """Return the prefixed SHA-256 of finite canonical JSON."""

    try:
        serialized = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise ReleaseGateError("artifact payload must be finite JSON") from error
    return "sha256:" + hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash a required regular file without following a caller-supplied digest."""

    source = Path(path)
    if not source.is_file():
        raise ReleaseIntegrityError(f"required release source is missing: {source}")
    digest = hashlib.sha256()
    try:
        with source.open("rb") as handle:
            while chunk := handle.read(chunk_size):
                digest.update(chunk)
    except OSError as error:
        raise ReleaseIntegrityError(f"could not hash release source {source}: {error}") from error
    return digest.hexdigest()


def _prefixed_file_sha256(path: str | Path) -> str:
    return "sha256:" + sha256_file(path)


def read_json_mapping(path: str | Path, *, context: str) -> Mapping[str, object]:
    """Read one bounded JSON object."""

    source = Path(path)
    if not source.is_file():
        raise ReleaseIntegrityError(f"{context} is missing: {source}")
    if source.stat().st_size > 100_000_000:
        raise ReleaseIntegrityError(f"{context} is unreasonably large")
    try:
        decoded: object = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ReleaseIntegrityError(f"could not decode {context}: {error}") from error
    return _mapping(decoded, context)


def write_new_hashed_json(
    path: str | Path,
    payload: Mapping[str, object],
    *,
    hash_field: str,
) -> tuple[Path, str]:
    """Commit a new, non-overwriting JSON artifact with a canonical hash."""

    destination = Path(path)
    if destination.suffix.casefold() != ".json":
        raise ReleaseGateError("release artifact path must end in .json")
    if hash_field in payload:
        raise ReleaseGateError(f"unhashed payload must not contain {hash_field!r}")
    digest = canonical_sha256(payload)
    committed = dict(payload)
    committed[hash_field] = digest
    _write_json(destination, committed, replace=False)
    return destination, digest


@dataclass(frozen=True, slots=True)
class RefitMember:
    """Verified lineage for one authoritative folds-1–8 refit."""

    member_id: str
    comparison_id: str
    architecture: str
    seed: int
    run_name: str
    run_dir: Path
    completion_path: Path
    completion_sha256: str
    freeze_artifact_path: Path
    freeze_artifact_sha256: str
    recipe_sha256: str
    source_member_completion_path: Path
    source_member_completion_sha256: str
    final_checkpoint_path: Path
    final_checkpoint_sha256: str
    resolved_config_path: Path
    resolved_config_file_sha256: str
    resolved_config_hash: str
    metadata_path: Path
    metadata_sha256: str
    protocol_path: Path
    protocol_file_sha256: str
    history_path: Path
    history_sha256: str
    protocol_hash: str
    manifest_path: Path
    manifest_sha256: str
    normalization_path: Path
    normalization_sha256: str
    source_checkpoint_path: Path
    source_checkpoint_sha256: str
    frozen_epochs: int
    selection_provenance: Mapping[str, object]
    selection_lineage_sha256: str
    lineage_sha256: str

    def to_payload(self) -> dict[str, object]:
        return {
            "member_id": self.member_id,
            "comparison_id": self.comparison_id,
            "architecture": self.architecture,
            "seed": self.seed,
            "run_name": self.run_name,
            "run_dir": str(self.run_dir),
            "refit_completion": {
                "path": str(self.completion_path),
                "artifact_sha256": self.completion_sha256,
            },
            "files": {
                "final_checkpoint": {
                    "path": str(self.final_checkpoint_path),
                    "sha256": self.final_checkpoint_sha256,
                },
                "resolved_config": {
                    "path": str(self.resolved_config_path),
                    "sha256": self.resolved_config_file_sha256,
                    "config_hash": self.resolved_config_hash,
                },
                "metadata": {
                    "path": str(self.metadata_path),
                    "sha256": self.metadata_sha256,
                },
                "protocol": {
                    "path": str(self.protocol_path),
                    "sha256": self.protocol_file_sha256,
                },
                "history": {
                    "path": str(self.history_path),
                    "sha256": self.history_sha256,
                },
                "manifest": {
                    "path": str(self.manifest_path),
                    "sha256": self.manifest_sha256,
                },
                "normalization": {
                    "path": str(self.normalization_path),
                    "sha256": self.normalization_sha256,
                },
                "source_checkpoint": {
                    "path": str(self.source_checkpoint_path),
                    "sha256": self.source_checkpoint_sha256,
                },
            },
            "freeze": {
                "refit_folds": list(REFIT_FOLDS),
                "normalization_folds": list(TRAIN_FOLDS),
                "frozen_epochs": self.frozen_epochs,
                "freeze_artifact_path": str(self.freeze_artifact_path),
                "freeze_artifact_sha256": self.freeze_artifact_sha256,
                "recipe_sha256": self.recipe_sha256,
                "source_member_completion_path": str(
                    self.source_member_completion_path
                ),
                "source_member_completion_sha256": (
                    self.source_member_completion_sha256
                ),
                "selection_provenance": dict(self.selection_provenance),
                "selection_lineage_sha256": self.selection_lineage_sha256,
            },
            "protocol_hash": self.protocol_hash,
            "lineage_sha256": self.lineage_sha256,
        }


@dataclass(frozen=True, slots=True)
class RefitBundle:
    """The exact six-member prerequisite for any fold-9 export."""

    protocol_hash: str
    manifest_sha256: str
    normalization_sha256: str
    label_order: tuple[str, ...]
    members: tuple[RefitMember, ...]
    created_at_utc: str
    artifact_sha256: str | None

    def to_payload(self, *, include_integrity: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": REFIT_BUNDLE_SCHEMA_VERSION,
            "artifact_type": REFIT_BUNDLE_TYPE,
            "protocol_hash": self.protocol_hash,
            "manifest_sha256": self.manifest_sha256,
            "normalization_sha256": self.normalization_sha256,
            "label_order": list(self.label_order),
            "release_grid": {
                "architectures": list(EXPECTED_ARCHITECTURES),
                "seeds": list(EXPECTED_SEEDS),
                "member_count": 6,
                "refit_folds": list(REFIT_FOLDS),
                "normalization_folds": list(TRAIN_FOLDS),
            },
            "members": [member.to_payload() for member in self.members],
            "created_at_utc": self.created_at_utc,
        }
        if include_integrity and self.artifact_sha256 is not None:
            payload["artifact_sha256"] = self.artifact_sha256
        return payload


@dataclass(frozen=True, slots=True)
class CalibrationMember:
    """One independently fitted fold-9 policy and its complete freeze lineage."""

    member_id: str
    architecture: str
    seed: int
    model_name: str
    refit_lineage_sha256: str
    checkpoint_path: Path
    resolved_config_hash: str
    checkpoint_sha256: str
    resolved_config_path: Path
    resolved_config_file_sha256: str
    normalization_path: Path
    normalization_sha256: str
    prediction_path: Path
    prediction_sidecar_path: Path
    prediction_npz_sha256: str
    prediction_sidecar_sha256: str
    prediction_artifact_sha256: str
    prediction_alignment_sha256: str
    decision_path: Path
    decision_file_sha256: str
    decision_artifact_sha256: str
    temperature: float
    thresholds: tuple[float, ...]
    entropy_gates: tuple[Mapping[str, object], ...]
    independent_fit_sha256: str

    def to_payload(self) -> dict[str, object]:
        return {
            "member_id": self.member_id,
            "architecture": self.architecture,
            "seed": self.seed,
            "model_name": self.model_name,
            "freeze_lineage": {
                "refit_lineage_sha256": self.refit_lineage_sha256,
                "checkpoint_path": str(self.checkpoint_path),
                "resolved_config_hash": self.resolved_config_hash,
                "checkpoint_sha256": self.checkpoint_sha256,
                "resolved_config_path": str(self.resolved_config_path),
                "resolved_config_file_sha256": self.resolved_config_file_sha256,
                "normalization_path": str(self.normalization_path),
                "normalization_sha256": self.normalization_sha256,
            },
            "fold9_prediction": {
                "path": str(self.prediction_path),
                "sidecar_path": str(self.prediction_sidecar_path),
                "npz_sha256": self.prediction_npz_sha256,
                "sidecar_sha256": self.prediction_sidecar_sha256,
                "artifact_sha256": self.prediction_artifact_sha256,
                "alignment_sha256": self.prediction_alignment_sha256,
                "folds": list(CALIBRATION_FOLDS),
            },
            "decision": {
                "path": str(self.decision_path),
                "file_sha256": self.decision_file_sha256,
                "artifact_sha256": self.decision_artifact_sha256,
                "temperature": self.temperature,
                "thresholds": list(self.thresholds),
                "entropy_method": "mean_normalized_binary_entropy",
                "entropy_gates": [dict(gate) for gate in self.entropy_gates],
            },
            "independent_fit_sha256": self.independent_fit_sha256,
        }


@dataclass(frozen=True, slots=True)
class CalibrationBundle:
    """Six independently fitted policies frozen before fold 10 is opened."""

    refit_bundle_sha256: str
    protocol_hash: str
    manifest_sha256: str
    normalization_sha256: str
    label_order: tuple[str, ...]
    members: tuple[CalibrationMember, ...]
    created_at_utc: str
    artifact_sha256: str | None

    def to_payload(self, *, include_integrity: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": CALIBRATION_BUNDLE_SCHEMA_VERSION,
            "artifact_type": CALIBRATION_BUNDLE_TYPE,
            "refit_bundle_sha256": self.refit_bundle_sha256,
            "protocol_hash": self.protocol_hash,
            "manifest_sha256": self.manifest_sha256,
            "normalization_sha256": self.normalization_sha256,
            "label_order": list(self.label_order),
            "release_grid": {
                "architectures": list(EXPECTED_ARCHITECTURES),
                "seeds": list(EXPECTED_SEEDS),
                "member_count": 6,
                "calibration_folds": list(CALIBRATION_FOLDS),
                "temperature_fits": 6,
                "thresholds_per_fit": len(LABEL_ORDER),
                "temperature_scope": "one_global_temperature_per_member",
                "retuning_after_freeze": False,
            },
            "members": [member.to_payload() for member in self.members],
            "created_at_utc": self.created_at_utc,
        }
        if include_integrity and self.artifact_sha256 is not None:
            payload["artifact_sha256"] = self.artifact_sha256
        return payload


def create_refit_bundle(
    completion_paths: Sequence[str | Path],
    *,
    protocol: ExperimentProtocol,
    created_at_utc: str | None = None,
) -> RefitBundle:
    """Verify the complete refit grid and return an unsaved release bundle."""

    if len(completion_paths) != 6:
        raise ReleaseGateError("refit bundle requires exactly six completion receipts")
    members = tuple(
        sorted(
            (
                _verify_refit_completion(Path(path), protocol=protocol)
                for path in completion_paths
            ),
            key=lambda member: (EXPECTED_ARCHITECTURES.index(member.architecture), member.seed),
        )
    )
    _validate_member_grid(members)
    manifests = {member.manifest_sha256 for member in members}
    normalizations = {member.normalization_sha256 for member in members}
    protocols = {member.protocol_hash for member in members}
    comparison_ids = {member.comparison_id for member in members}
    freeze_hashes = {member.freeze_artifact_sha256 for member in members}
    if manifests != {members[0].manifest_sha256}:
        raise ReleaseGateError("all refits must bind the same manifest")
    if normalizations != {members[0].normalization_sha256}:
        raise ReleaseGateError("all refits must bind the same normalization artifact")
    if protocols != {protocol.protocol_hash}:
        raise ReleaseGateError("all refits must bind the supplied protocol")
    if len(comparison_ids) != 1 or len(freeze_hashes) != 1:
        raise ReleaseGateError("all refits must bind one comparison and one freeze")
    if len({member.recipe_sha256 for member in members}) != 6:
        raise ReleaseGateError("each refit member requires its own frozen recipe")
    for architecture in EXPECTED_ARCHITECTURES:
        budgets = {
            member.frozen_epochs
            for member in members
            if member.architecture == architecture
        }
        if len(budgets) != 1:
            raise ReleaseGateError(
                f"{architecture} refits do not share the frozen median epoch budget"
            )
    return RefitBundle(
        protocol_hash=protocol.protocol_hash,
        manifest_sha256=members[0].manifest_sha256,
        normalization_sha256=members[0].normalization_sha256,
        label_order=LABEL_ORDER,
        members=members,
        created_at_utc=_timestamp(created_at_utc),
        artifact_sha256=None,
    )


def save_refit_bundle(bundle: RefitBundle, path: str | Path) -> tuple[Path, str]:
    """Save a newly verified refit bundle without allowing overwrite."""

    _validate_refit_bundle(bundle)
    return write_new_hashed_json(
        path,
        bundle.to_payload(include_integrity=False),
        hash_field="artifact_sha256",
    )


def load_refit_bundle(
    path: str | Path,
    *,
    protocol: ExperimentProtocol,
    verify_sources: bool = True,
) -> RefitBundle:
    """Load, hash-check, and optionally re-verify every refit source file."""

    payload = read_json_mapping(path, context="refit release bundle")
    bundle = _parse_refit_bundle(payload, protocol=protocol)
    if verify_sources:
        for member in bundle.members:
            _verify_refit_member_sources(member, protocol=protocol)
    return bundle


def export_fold9_predictions(
    refit_bundle_path: str | Path,
    output_directory: str | Path,
    *,
    protocol: ExperimentProtocol,
    batch_size: int | None = None,
    num_workers: int | None = None,
    device: str = "auto",
    bf16: bool = True,
    exporter: PredictionExporter = export_checkpoint_predictions,
) -> Mapping[str, Path]:
    """Export all six fold-9 artifacts only after the refit bundle re-verifies."""

    bundle = load_refit_bundle(refit_bundle_path, protocol=protocol, verify_sources=True)
    destination = Path(output_directory).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    planned = {
        member.member_id: destination / f"{member.member_id}.fold9.npz"
        for member in bundle.members
    }
    stage_plan: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": FOLD9_EXPORT_PLAN_TYPE,
        "refit_bundle_sha256": bundle.artifact_sha256,
        "protocol_hash": bundle.protocol_hash,
        "manifest_sha256": bundle.manifest_sha256,
        "output_directory": str(destination),
        "inference": {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "device": device,
            "bf16": bf16,
        },
        "members": [
            {
                "member_id": member.member_id,
                "lineage_sha256": member.lineage_sha256,
                "checkpoint_sha256": member.final_checkpoint_sha256,
                "resolved_config_hash": member.resolved_config_hash,
                "prediction_path": str(planned[member.member_id]),
            }
            for member in bundle.members
        ],
    }
    plan_path = destination / "fold9-export-plan.json"
    if not plan_path.exists():
        collisions = [
            path
            for prediction_path in planned.values()
            for path in (prediction_path, prediction_path.with_suffix(".json"))
            if path.exists()
        ]
        if collisions:
            raise ReleaseStateError(
                "fold-9 artifacts predate the batch plan: "
                + ", ".join(str(path) for path in collisions)
            )
    _ensure_exact_stage_plan(plan_path, stage_plan)
    completed: dict[str, Path] = {}
    for member in bundle.members:
        prediction_path = planned[member.member_id]
        sidecar_path = prediction_path.with_suffix(".json")
        if prediction_path.exists() or sidecar_path.exists():
            if not prediction_path.is_file() or not sidecar_path.is_file():
                raise ReleaseStateError(
                    f"partial fold-9 artifact cannot be adopted: {member.member_id}"
                )
            prediction = load_prediction_artifact(
                prediction_path,
                protocol=protocol,
                expected_config_hash=member.resolved_config_hash,
                expected_manifest_hash=member.manifest_sha256,
            )
            _validate_prediction_lineage(prediction, member)
            completed[member.member_id] = prediction_path
            continue
        request = PredictionExportRequest(
            checkpoint_path=member.final_checkpoint_path,
            resolved_config_path=member.resolved_config_path,
            run_metadata_path=member.metadata_path,
            output_path=planned[member.member_id],
            fold_role=FoldRole.CALIBRATION,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            bf16=bf16,
        )
        result = exporter(request, protocol=protocol)
        if result.fold_role is not FoldRole.CALIBRATION or result.folds != CALIBRATION_FOLDS:
            raise ReleaseIntegrityError("fold-9 exporter returned the wrong fold role")
        if result.model_seed != member.seed or result.config_hash != member.resolved_config_hash:
            raise ReleaseIntegrityError("fold-9 exporter returned mismatched member lineage")
        if result.files.npz_path.resolve() != prediction_path.resolve():
            raise ReleaseIntegrityError("fold-9 exporter wrote outside the batch plan")
        prediction = load_prediction_artifact(
            prediction_path,
            protocol=protocol,
            expected_config_hash=member.resolved_config_hash,
            expected_manifest_hash=member.manifest_sha256,
        )
        _validate_prediction_lineage(prediction, member)
        completed[member.member_id] = prediction_path
    if set(completed) != {member.member_id for member in bundle.members}:
        raise ReleaseStateError("fold-9 export did not complete the exact six-member batch")
    return completed


def create_calibration_bundle(
    refit_bundle: RefitBundle,
    prediction_paths: Mapping[str, str | Path],
    decision_paths: Mapping[str, str | Path],
    *,
    protocol: ExperimentProtocol,
    created_at_utc: str | None = None,
) -> CalibrationBundle:
    """Verify six independently fitted fold-9 policies and bind their lineage."""

    _validate_refit_bundle(refit_bundle, require_integrity=True)
    expected_ids = {member.member_id for member in refit_bundle.members}
    if set(prediction_paths) != expected_ids or set(decision_paths) != expected_ids:
        raise ReleaseGateError("calibration inputs must map the exact six refit member IDs")
    members: list[CalibrationMember] = []
    for refit_member in refit_bundle.members:
        prediction_path = Path(prediction_paths[refit_member.member_id]).resolve()
        decision_path = Path(decision_paths[refit_member.member_id]).resolve()
        prediction = load_prediction_artifact(
            prediction_path,
            protocol=protocol,
            expected_config_hash=refit_member.resolved_config_hash,
            expected_manifest_hash=refit_member.manifest_sha256,
        )
        decision = load_calibration_decisions(decision_path, protocol=protocol)
        members.append(
            _build_calibration_member(
                refit_member,
                prediction,
                prediction_path=prediction_path,
                decision=decision,
                decision_path=decision_path,
            )
        )
    ordered = tuple(members)
    if len({member.independent_fit_sha256 for member in ordered}) != 6:
        raise ReleaseGateError("calibration policies are not six independent member fits")
    if len({member.prediction_artifact_sha256 for member in ordered}) != 6:
        raise ReleaseGateError("each calibration fit requires its own prediction artifact")
    bundle = CalibrationBundle(
        refit_bundle_sha256=cast(str, refit_bundle.artifact_sha256),
        protocol_hash=refit_bundle.protocol_hash,
        manifest_sha256=refit_bundle.manifest_sha256,
        normalization_sha256=refit_bundle.normalization_sha256,
        label_order=refit_bundle.label_order,
        members=ordered,
        created_at_utc=_timestamp(created_at_utc),
        artifact_sha256=None,
    )
    _validate_calibration_bundle(bundle)
    return bundle


def fit_calibration_bundle(
    refit_bundle_path: str | Path,
    prediction_paths: Mapping[str, str | Path],
    output_directory: str | Path,
    *,
    protocol: ExperimentProtocol,
    coverage_targets: Sequence[float] = (1.0, 0.9, 0.8, 0.7, 0.5),
    created_at_utc: str | None = None,
) -> CalibrationBundle:
    """Fit six fold-9 policies with the existing leakage-safe API."""

    refit_bundle = load_refit_bundle(
        refit_bundle_path, protocol=protocol, verify_sources=True
    )
    expected_ids = {member.member_id for member in refit_bundle.members}
    if set(prediction_paths) != expected_ids:
        raise ReleaseGateError("fold-9 predictions must map the exact six refit member IDs")
    destination = Path(output_directory).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    loaded_predictions: dict[str, PredictionArtifact] = {}
    normalized_prediction_paths: dict[str, Path] = {}
    for member in refit_bundle.members:
        prediction_path = Path(prediction_paths[member.member_id]).resolve()
        prediction = load_prediction_artifact(
            prediction_path,
            protocol=protocol,
            expected_config_hash=member.resolved_config_hash,
            expected_manifest_hash=member.manifest_sha256,
        )
        _validate_prediction_lineage(prediction, member)
        loaded_predictions[member.member_id] = prediction
        normalized_prediction_paths[member.member_id] = prediction_path
    decision_outputs = {
        member.member_id: destination / f"{member.member_id}.fold9.decisions.json"
        for member in refit_bundle.members
    }
    fit_plan: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": CALIBRATION_FIT_PLAN_TYPE,
        "refit_bundle_sha256": refit_bundle.artifact_sha256,
        "protocol_hash": refit_bundle.protocol_hash,
        "manifest_sha256": refit_bundle.manifest_sha256,
        "coverage_targets": [float(value) for value in coverage_targets],
        "members": [
            {
                "member_id": member.member_id,
                "refit_lineage_sha256": member.lineage_sha256,
                "prediction_path": str(normalized_prediction_paths[member.member_id]),
                "prediction_artifact_sha256": loaded_predictions[
                    member.member_id
                ].integrity_sha256,
                "decision_path": str(decision_outputs[member.member_id]),
            }
            for member in refit_bundle.members
        ],
    }
    fit_plan_path = destination / "calibration-fit-plan.json"
    if not fit_plan_path.exists():
        collisions = [path for path in decision_outputs.values() if path.exists()]
        if collisions:
            raise ReleaseStateError(
                "calibration decisions predate the fit plan: "
                + ", ".join(str(path) for path in collisions)
            )
    _ensure_exact_stage_plan(fit_plan_path, fit_plan)
    decision_paths: dict[str, Path] = {}
    for member in refit_bundle.members:
        prediction = loaded_predictions[member.member_id]
        output = decision_outputs[member.member_id]
        if output.exists():
            load_calibration_decisions(output, protocol=protocol)
        else:
            decisions = fit_calibration_decisions(
                prediction,
                protocol=protocol,
                coverage_targets=coverage_targets,
                created_at_utc=created_at_utc,
            )
            save_calibration_decisions(decisions, output)
        decision_paths[member.member_id] = output
    return create_calibration_bundle(
        refit_bundle,
        normalized_prediction_paths,
        decision_paths,
        protocol=protocol,
        created_at_utc=created_at_utc,
    )


def _ensure_exact_stage_plan(
    path: Path, payload: Mapping[str, object]
) -> None:
    if not path.exists():
        write_new_hashed_json(path, payload, hash_field="plan_sha256")
        return
    existing = read_json_mapping(path, context=f"existing stage plan {path.name}")
    stored = _hash_string(existing.get("plan_sha256"), "plan_sha256")
    unhashed = dict(existing)
    del unhashed["plan_sha256"]
    if canonical_sha256(unhashed) != stored or unhashed != dict(payload):
        raise ReleaseIntegrityError(
            f"existing stage plan differs from requested exact batch: {path}"
        )


def save_calibration_bundle(
    bundle: CalibrationBundle, path: str | Path
) -> tuple[Path, str]:
    """Save the frozen pre-fold-10 calibration bundle."""

    _validate_calibration_bundle(bundle)
    return write_new_hashed_json(
        path,
        bundle.to_payload(include_integrity=False),
        hash_field="artifact_sha256",
    )


def load_calibration_bundle(
    path: str | Path,
    *,
    protocol: ExperimentProtocol,
    verify_sources: bool = True,
) -> CalibrationBundle:
    """Load the complete pre-fold-10 freeze and optionally re-hash all sources."""

    payload = read_json_mapping(path, context="calibration release bundle")
    bundle = _parse_calibration_bundle(payload, protocol=protocol)
    if verify_sources:
        _verify_calibration_sources(bundle, protocol=protocol)
    return bundle


def materialize_demo_policy_payload(
    bundle: CalibrationBundle,
    member_id: str,
    *,
    target_coverage: float = 0.8,
) -> dict[str, object]:
    """Materialize one provenance-complete demo policy without refitting.

    ``target_coverage`` must name an entropy gate already frozen in the bundle;
    this function never interpolates or learns a new cutoff.
    """

    _validate_calibration_bundle(bundle, require_integrity=True)
    matches = [member for member in bundle.members if member.member_id == member_id]
    if len(matches) != 1:
        raise ReleaseGateError(f"calibration bundle has no unique member {member_id!r}")
    member = matches[0]
    gates = [
        gate
        for gate in member.entropy_gates
        if math.isclose(
            _finite_float(
                gate.get("target_coverage"),
                "entropy gate target_coverage",
                minimum=0.0,
            ),
            target_coverage,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ]
    if len(gates) != 1:
        raise ReleaseGateError(
            "demo target coverage must match exactly one frozen entropy gate"
        )
    gate_threshold = _finite_float(
        gates[0].get("maximum_entropy"),
        "entropy gate maximum_entropy",
        minimum=0.0,
    )
    if gate_threshold > 1.0:
        raise ReleaseGateError("entropy gate maximum must lie in [0, 1]")
    return {
        "schema_version": 1,
        "label_order": list(bundle.label_order),
        "calibration": {
            "method": "temperature_scaling",
            "temperature": member.temperature,
        },
        "classification_thresholds": list(member.thresholds),
        "gate": {
            "method": "mean_normalized_binary_entropy",
            "uncertainty_threshold": gate_threshold,
        },
        "provenance": {
            "dataset_version": PTBXL_VERSION,
            "protocol_hash": bundle.protocol_hash,
            "manifest_hash": bundle.manifest_sha256.removeprefix("sha256:"),
            "checkpoint_config_hash": member.resolved_config_hash,
            "checkpoint_sha256": member.checkpoint_sha256.removeprefix("sha256:"),
            "resolved_config_sha256": (
                member.resolved_config_file_sha256.removeprefix("sha256:")
            ),
            "normalization_sha256": member.normalization_sha256.removeprefix(
                "sha256:"
            ),
            "calibration_folds": list(CALIBRATION_FOLDS),
        },
    }


_POST_SWEEP_RECIPE_FIELDS = {
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
_POST_SWEEP_RESOLUTION_FIELDS = {
    "selection_provenance",
    "freeze_binding",
    "attempt_index",
    "effective_data",
    "checkpoint_roles",
}
_SOURCE_MEMBER_RECEIPT_FIELDS = {
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


def _resolved_post_sweep_recipe(
    resolved_config: Mapping[str, object],
) -> dict[str, object]:
    """Project an enriched run config back to its exact immutable recipe."""

    _exact_keys(
        resolved_config,
        _POST_SWEEP_RECIPE_FIELDS | _POST_SWEEP_RESOLUTION_FIELDS,
        "resolved post-sweep refit config",
    )
    model_metadata = _mapping(
        resolved_config.get("model"), "resolved post-sweep model metadata"
    )
    model_identity = _mapping(
        resolved_config.get("model_identity"),
        "resolved post-sweep model_identity",
    )
    if dict(model_metadata) != dict(model_identity):
        raise ReleaseIntegrityError(
            "resolved post-sweep model metadata drifted from frozen identity"
        )
    binding = _mapping(
        resolved_config.get("freeze_binding"),
        "resolved post-sweep freeze_binding",
    )
    _exact_keys(
        binding,
        {"path", "artifact_sha256", "comparison_id", "recipe_sha256"},
        "resolved post-sweep freeze_binding",
    )
    expected_binding = {
        "path": resolved_config.get("freeze_artifact"),
        "artifact_sha256": resolved_config.get("freeze_artifact_sha256"),
        "comparison_id": resolved_config.get("comparison_id"),
        "recipe_sha256": resolved_config.get("recipe_sha256"),
    }
    if dict(binding) != expected_binding:
        raise ReleaseIntegrityError(
            "resolved post-sweep freeze binding is internally inconsistent"
        )
    _integer(resolved_config.get("attempt_index"), "attempt_index", minimum=0)
    recipe = {
        key: resolved_config[key]
        for key in _POST_SWEEP_RECIPE_FIELDS
    }
    recipe["model"] = {
        "architecture": model_metadata.get("architecture"),
        "preset": model_metadata.get("preset"),
    }
    return normalized_recipe_template(recipe)


def _expected_selection_provenance(
    source: Mapping[str, object],
    *,
    freeze_hash: str,
    recipe_hash: str,
    source_completion_hash: str,
    seed: int,
) -> dict[str, object]:
    selected_epoch = _integer(
        source.get("best_epoch"), "frozen source best_epoch", minimum=0
    )
    return {
        "checkpoint": _string(
            source.get("best_checkpoint"), "frozen source best_checkpoint"
        ),
        "checkpoint_sha256": _hash_string(
            source.get("best_checkpoint_sha256"),
            "frozen source best_checkpoint_sha256",
        ).removeprefix("sha256:"),
        "checkpoint_config_hash": _hash_string(
            source.get("resolved_config_hash"),
            "frozen source resolved_config_hash",
        ),
        "selected_epoch": selected_epoch,
        "selected_epoch_count": selected_epoch + 1,
        "selected_macro_auroc": _finite_float(
            source.get("best_validation_macro_auroc"),
            "frozen source best_validation_macro_auroc",
            minimum=0.0,
        ),
        "source_seed": seed,
        "member_completion_sha256": source_completion_hash,
        "freeze_artifact_sha256": freeze_hash,
        "recipe_sha256": recipe_hash,
    }


def _verify_source_member_receipt(
    path: Path,
    *,
    expected_file_hash: str,
    expected_source: Mapping[str, object],
    comparison_id: str,
    architecture: str,
    seed: int,
    protocol_hash: str,
    manifest_hash: str,
    normalization_hash: str,
) -> Mapping[str, object]:
    """Verify both byte identity and the source receipt's canonical self-hash."""

    if _prefixed_file_sha256(path) != expected_file_hash:
        raise ReleaseIntegrityError("source member completion file hash mismatch")
    receipt = read_json_mapping(path, context="source member completion")
    _exact_keys(
        receipt,
        _SOURCE_MEMBER_RECEIPT_FIELDS,
        "source member completion",
    )
    stored = _hash_string(
        receipt.get("artifact_sha256"),
        "source member completion artifact_sha256",
    )
    unhashed = dict(receipt)
    del unhashed["artifact_sha256"]
    if canonical_sha256(unhashed) != stored:
        raise ReleaseIntegrityError("source member completion self-hash mismatch")
    semantic = {
        "schema_version": 1,
        "artifact_type": "ecg_trust.multiseed_member_completion",
        "status": "complete",
        "comparison_id": comparison_id,
        "architecture": architecture,
        "seed": seed,
        "protocol_hash": protocol_hash,
        "manifest_hash": manifest_hash,
        "normalization_sha256": normalization_hash,
    }
    drift = [
        field for field, expected in semantic.items() if receipt.get(field) != expected
    ]
    if drift:
        raise ReleaseIntegrityError(
            "source member receipt semantic mismatch: " + ", ".join(drift)
        )
    source_to_receipt = {
        "run_metadata_sha256": "run_metadata_sha256",
        "resolved_config_file_sha256": "resolved_config_sha256",
        "resolved_config_hash": "config_hash",
        "history_sha256": "history_sha256",
        "best_checkpoint_sha256": "best_checkpoint_sha256",
        "prediction_npz_sha256": "prediction_npz_sha256",
        "prediction_artifact_sha256": "prediction_artifact_sha256",
        "best_epoch": "best_epoch",
        "best_validation_macro_auroc": "best_validation_macro_auroc",
    }
    for source_field, receipt_field in source_to_receipt.items():
        if expected_source.get(source_field) != receipt.get(receipt_field):
            raise ReleaseIntegrityError(
                f"source member receipt {receipt_field} drifted from freeze recipe"
            )
    path_bindings = {
        "run_metadata": "run_metadata_path",
        "resolved_config": "resolved_config_path",
        "history": "history_path",
        "best_checkpoint": "best_checkpoint_path",
        "prediction": "prediction_path",
        "prediction_json": "prediction_json_path",
    }
    for source_field, receipt_field in path_bindings.items():
        expected_path = Path(
            _string(expected_source.get(source_field), f"frozen source {source_field}")
        ).resolve()
        raw_receipt_path = Path(
            _string(receipt.get(receipt_field), f"source receipt {receipt_field}")
        )
        observed_path = (
            raw_receipt_path.resolve()
            if raw_receipt_path.is_absolute()
            else (path.parent / raw_receipt_path).resolve()
        )
        if observed_path != expected_path:
            raise ReleaseIntegrityError(
                f"source member receipt {receipt_field} drifted from freeze recipe"
            )
    return receipt


def _verify_refit_completion(
    completion_path: Path, *, protocol: ExperimentProtocol
) -> RefitMember:
    """Delegate receipt/source verification, then normalize it into the bundle."""

    try:
        completion = load_refit_completion(
            completion_path, protocol=protocol, verify_sources=True
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise ReleaseIntegrityError(f"invalid refit completion: {error}") from error
    files = _mapping(completion.get("files"), "refit completion.files")
    checkpoint = _completion_file_entry(files.get("final_checkpoint"), "final_checkpoint")
    resolved = _completion_file_entry(
        files.get("resolved_config"), "resolved_config", config_hash=True
    )
    metadata = _completion_file_entry(files.get("metadata"), "metadata")
    protocol_file = _completion_file_entry(files.get("protocol"), "protocol")
    history = _completion_file_entry(files.get("history"), "history")
    manifest = _completion_file_entry(files.get("manifest"), "manifest")
    normalization = _completion_file_entry(files.get("normalization"), "normalization")
    source_checkpoint = _completion_file_entry(
        files.get("source_checkpoint"), "source_checkpoint"
    )
    architecture = _string(completion.get("architecture"), "architecture")
    if architecture not in EXPECTED_ARCHITECTURES:
        raise ReleaseGateError(f"unsupported release architecture {architecture!r}")
    seed = _integer(completion.get("seed"), "seed", minimum=0)
    member_id = f"{architecture}-seed{seed}"
    if _integer_tuple(completion.get("refit_folds"), "refit_folds") != REFIT_FOLDS:
        raise ReleaseGateError("refit completion must bind exactly folds 1-8")
    if (
        _integer_tuple(completion.get("normalization_folds"), "normalization_folds")
        != TRAIN_FOLDS
    ):
        raise ReleaseGateError("refit completion must retain folds-1-7 normalization")
    protocol_hash = _hash_string(completion.get("protocol_hash"), "protocol_hash")
    if protocol_hash != protocol.protocol_hash:
        raise ReleaseIntegrityError("refit completion protocol mismatch")
    manifest_hash = _hash_string(completion.get("manifest_hash"), "manifest_hash")
    normalization_hash = _hash_string(
        completion.get("normalization_hash"), "normalization_hash"
    )
    if manifest[1] != manifest_hash or normalization[1] != normalization_hash:
        raise ReleaseIntegrityError("refit completion common data hashes disagree")
    selection = _mapping(
        completion.get("selection_provenance"), "selection_provenance"
    )
    selection_hash = _hash_string(
        completion.get("selection_lineage_sha256"), "selection_lineage_sha256"
    )
    if canonical_sha256(selection) != selection_hash:
        raise ReleaseIntegrityError("refit completion selection lineage hash mismatch")
    completion_hash = _hash_string(
        completion.get("artifact_sha256"), "artifact_sha256"
    )
    freeze_hash = _hash_string(
        completion.get("freeze_artifact_sha256"), "freeze_artifact_sha256"
    )
    recipe_hash = _hash_string(completion.get("recipe_sha256"), "recipe_sha256")
    source_completion_hash = _hash_string(
        completion.get("source_member_completion_sha256"),
        "source_member_completion_sha256",
    )
    freeze_path = Path(
        _string(completion.get("freeze_artifact_path"), "freeze_artifact_path")
    )
    try:
        freeze = load_multiseed_freeze(
            freeze_path, protocol=protocol, verify_sources=True
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise ReleaseIntegrityError(
            f"could not verify multi-seed freeze: {error}"
        ) from error
    if freeze.artifact_sha256 != freeze_hash:
        raise ReleaseIntegrityError("refit completion freeze artifact hash mismatch")
    freeze_payload = _mapping(freeze.payload, "multi-seed freeze payload")
    freeze_protocol_hash = _hash_string(
        freeze_payload.get("protocol_hash"), "freeze protocol_hash"
    )
    freeze_manifest_hash = _hash_string(
        freeze_payload.get("manifest_hash"), "freeze manifest_hash"
    )
    freeze_normalization_hash = _hash_string(
        freeze_payload.get("normalization_hash"), "freeze normalization_hash"
    )
    freeze_comparison_id = _string(
        freeze_payload.get("comparison_id"), "freeze comparison_id"
    )
    if freeze_protocol_hash != protocol_hash:
        raise ReleaseIntegrityError("refit completion protocol drifted from freeze root")
    if manifest_hash != freeze_manifest_hash:
        raise ReleaseIntegrityError("refit manifest drifted from freeze root")
    if normalization_hash != freeze_normalization_hash:
        raise ReleaseIntegrityError("refit normalization drifted from freeze root")
    comparison_id = _string(completion.get("comparison_id"), "comparison_id")
    if comparison_id != freeze_comparison_id:
        raise ReleaseIntegrityError("refit comparison_id drifted from freeze root")
    try:
        expected_recipe = _mapping(
            freeze.recipe_template(architecture, seed),
            "frozen member recipe",
        )
    except (KeyError, OSError, RuntimeError, ValueError) as error:
        raise ReleaseIntegrityError(
            f"freeze has no verified recipe for {member_id}: {error}"
        ) from error
    expected_recipe_hash = _hash_string(
        expected_recipe.get("recipe_sha256"), "frozen recipe_sha256"
    )
    if recipe_hash != expected_recipe_hash:
        raise ReleaseIntegrityError("refit recipe hash drifted from freeze member")
    expected_source = _mapping(
        expected_recipe.get("source"), "frozen member recipe.source"
    )
    source_completion_path = Path(
        _string(
            completion.get("source_member_completion_path"),
            "source_member_completion_path",
        )
    )
    if _prefixed_file_sha256(source_completion_path) != source_completion_hash:
        raise ReleaseIntegrityError("source member completion hash mismatch")
    expected_source_path = Path(
        _string(
            expected_source.get("member_completion"),
            "frozen source member_completion",
        )
    )
    expected_source_hash = _hash_string(
        expected_source.get("member_completion_sha256"),
        "frozen source member_completion_sha256",
    )
    if source_completion_path.resolve() != expected_source_path.resolve():
        raise ReleaseIntegrityError("source member path drifted from freeze recipe")
    if source_completion_hash != expected_source_hash:
        raise ReleaseIntegrityError("source member hash drifted from freeze recipe")
    source_receipt = _verify_source_member_receipt(
        source_completion_path,
        expected_file_hash=source_completion_hash,
        expected_source=expected_source,
        comparison_id=comparison_id,
        architecture=architecture,
        seed=seed,
        protocol_hash=protocol_hash,
        manifest_hash=freeze_manifest_hash,
        normalization_hash=freeze_normalization_hash,
    )
    resolved_wrapper = read_json_mapping(
        resolved[0], context="completion resolved refit config"
    )
    if set(resolved_wrapper) != {"config_hash", "config"}:
        raise ReleaseIntegrityError("resolved refit config keys are not canonical")
    if resolved_wrapper["config_hash"] != resolved[2]:
        raise ReleaseIntegrityError("resolved refit config hash field mismatch")
    resolved_config = _mapping(
        resolved_wrapper["config"], "completion resolved refit config.config"
    )
    if canonical_sha256(resolved_config) != resolved[2]:
        raise ReleaseIntegrityError("resolved refit config content hash mismatch")
    frozen_epochs = _integer(
        completion.get("frozen_epochs"), "frozen_epochs", minimum=1
    )
    normalized_resolved_recipe = _resolved_post_sweep_recipe(resolved_config)
    if normalized_resolved_recipe != dict(expected_recipe):
        raise ReleaseIntegrityError(
            "resolved post-sweep config is not the exact frozen member recipe"
        )
    expected_selection = _expected_selection_provenance(
        expected_source,
        freeze_hash=freeze_hash,
        recipe_hash=recipe_hash,
        source_completion_hash=source_completion_hash,
        seed=seed,
    )
    if dict(selection) != expected_selection:
        raise ReleaseIntegrityError(
            "selection provenance drifted from frozen source evidence"
        )
    expected_frozen_epochs = _integer(
        _mapping(expected_recipe.get("selection"), "frozen recipe.selection").get(
            "frozen_epochs"
        ),
        "frozen recipe selection.frozen_epochs",
        minimum=1,
    )
    if frozen_epochs != expected_frozen_epochs:
        raise ReleaseIntegrityError("refit epoch budget drifted from freeze recipe")
    source_checkpoint_path_expected = Path(
        _string(expected_source.get("best_checkpoint"), "frozen source best_checkpoint")
    )
    source_checkpoint_hash_expected = _hash_string(
        expected_source.get("best_checkpoint_sha256"),
        "frozen source best_checkpoint_sha256",
    )
    if source_checkpoint[0].resolve() != source_checkpoint_path_expected.resolve():
        raise ReleaseIntegrityError("source checkpoint path drifted from freeze recipe")
    if source_checkpoint[1] != source_checkpoint_hash_expected:
        raise ReleaseIntegrityError("source checkpoint hash drifted from freeze recipe")
    if source_receipt.get("best_checkpoint_sha256") != source_checkpoint_hash_expected:
        raise ReleaseIntegrityError("source receipt checkpoint drifted from freeze recipe")
    lineage_payload: dict[str, object] = {
        "member_id": member_id,
        "comparison_id": comparison_id,
        "architecture": architecture,
        "seed": seed,
        "run_name": _string(completion.get("run_name"), "run_name"),
        "refit_completion_sha256": completion_hash,
        "checkpoint_sha256": checkpoint[1],
        "resolved_config_hash": resolved[2],
        "metadata_sha256": metadata[1],
        "history_sha256": history[1],
        "protocol_hash": protocol_hash,
        "manifest_sha256": manifest_hash,
        "normalization_sha256": normalization_hash,
        "freeze_artifact_sha256": freeze_hash,
        "recipe_sha256": recipe_hash,
        "source_member_completion_sha256": source_completion_hash,
        "selection_lineage_sha256": selection_hash,
        "frozen_epochs": frozen_epochs,
    }
    return RefitMember(
        member_id=member_id,
        comparison_id=cast(str, lineage_payload["comparison_id"]),
        architecture=architecture,
        seed=seed,
        run_name=cast(str, lineage_payload["run_name"]),
        run_dir=Path(_string(completion.get("run_dir"), "run_dir")),
        completion_path=completion_path.resolve(),
        completion_sha256=completion_hash,
        freeze_artifact_path=freeze_path,
        freeze_artifact_sha256=freeze_hash,
        recipe_sha256=recipe_hash,
        source_member_completion_path=source_completion_path,
        source_member_completion_sha256=source_completion_hash,
        final_checkpoint_path=checkpoint[0],
        final_checkpoint_sha256=checkpoint[1],
        resolved_config_path=resolved[0],
        resolved_config_file_sha256=resolved[1],
        resolved_config_hash=resolved[2],
        metadata_path=metadata[0],
        metadata_sha256=metadata[1],
        protocol_path=protocol_file[0],
        protocol_file_sha256=protocol_file[1],
        history_path=history[0],
        history_sha256=history[1],
        protocol_hash=protocol_hash,
        manifest_path=manifest[0],
        manifest_sha256=manifest_hash,
        normalization_path=normalization[0],
        normalization_sha256=normalization_hash,
        source_checkpoint_path=source_checkpoint[0],
        source_checkpoint_sha256=source_checkpoint[1],
        frozen_epochs=frozen_epochs,
        selection_provenance=selection,
        selection_lineage_sha256=selection_hash,
        lineage_sha256=canonical_sha256(lineage_payload),
    )


def _verify_refit_member_sources(
    member: RefitMember, *, protocol: ExperimentProtocol
) -> None:
    observed = _verify_refit_completion(member.completion_path, protocol=protocol)
    if observed.to_payload() != member.to_payload():
        raise ReleaseIntegrityError(
            f"refit member changed after bundle creation: {member.member_id}"
        )


def _build_calibration_member(
    refit: RefitMember,
    prediction: PredictionArtifact,
    *,
    prediction_path: Path,
    decision: CalibrationDecisionArtifact,
    decision_path: Path,
) -> CalibrationMember:
    _validate_prediction_lineage(prediction, refit)
    if decision.integrity_sha256 is None:
        raise ReleaseIntegrityError("calibration decisions must be loaded from disk")
    expected = {
        "model_name": prediction.model_name,
        "model_seed": prediction.model_seed,
        "protocol_hash": prediction.protocol_hash,
        "config_hash": prediction.config_hash,
        "manifest_hash": prediction.manifest_hash,
        "label_order": prediction.label_order,
        "source_prediction_sha256": prediction.integrity_sha256,
        "source_alignment_sha256": prediction.alignment_sha256,
    }
    observed = {
        "model_name": decision.model_name,
        "model_seed": decision.model_seed,
        "protocol_hash": decision.protocol_hash,
        "config_hash": decision.config_hash,
        "manifest_hash": decision.manifest_hash,
        "label_order": decision.label_order,
        "source_prediction_sha256": decision.source_prediction_sha256,
        "source_alignment_sha256": decision.source_alignment_sha256,
    }
    mismatches = [name for name in expected if expected[name] != observed[name]]
    if mismatches:
        raise ReleaseIntegrityError(
            "fold-9 decision lineage mismatch: " + ", ".join(mismatches)
        )
    if decision.temperature_scaling.source_folds != CALIBRATION_FOLDS:
        raise ReleaseGateError("temperature must be fitted on fold 9 only")
    if decision.threshold_optimization.source_folds != CALIBRATION_FOLDS:
        raise ReleaseGateError("thresholds must be fitted on fold 9 only")
    thresholds = decision.threshold_optimization.thresholds
    if len(thresholds) != len(LABEL_ORDER):
        raise ReleaseGateError("each calibration member requires five thresholds")
    if not decision.coverage_gates:
        raise ReleaseGateError("each calibration member requires entropy gates")
    sidecar = prediction_path.with_suffix(".json")
    fit_payload: dict[str, object] = {
        "member_id": refit.member_id,
        "refit_lineage_sha256": refit.lineage_sha256,
        "prediction_artifact_sha256": prediction.integrity_sha256,
        "decision_artifact_sha256": decision.integrity_sha256,
    }
    return CalibrationMember(
        member_id=refit.member_id,
        architecture=refit.architecture,
        seed=refit.seed,
        model_name=prediction.model_name,
        refit_lineage_sha256=refit.lineage_sha256,
        checkpoint_path=refit.final_checkpoint_path,
        resolved_config_hash=refit.resolved_config_hash,
        checkpoint_sha256=refit.final_checkpoint_sha256,
        resolved_config_path=refit.resolved_config_path,
        resolved_config_file_sha256=refit.resolved_config_file_sha256,
        normalization_path=refit.normalization_path,
        normalization_sha256=refit.normalization_sha256,
        prediction_path=prediction_path,
        prediction_sidecar_path=sidecar,
        prediction_npz_sha256=sha256_file(prediction_path),
        prediction_sidecar_sha256=sha256_file(sidecar),
        prediction_artifact_sha256=cast(str, prediction.integrity_sha256),
        prediction_alignment_sha256=prediction.alignment_sha256,
        decision_path=decision_path,
        decision_file_sha256=sha256_file(decision_path),
        decision_artifact_sha256=decision.integrity_sha256,
        temperature=decision.temperature_scaling.temperature,
        thresholds=thresholds,
        entropy_gates=tuple(gate.to_dict() for gate in decision.coverage_gates),
        independent_fit_sha256=canonical_sha256(fit_payload),
    )


def _validate_prediction_lineage(prediction: PredictionArtifact, refit: RefitMember) -> None:
    if prediction.fold_role is not FoldRole.CALIBRATION or prediction.folds != CALIBRATION_FOLDS:
        raise ReleaseGateError("calibration bundle accepts fold-9 predictions only")
    expected: dict[str, object] = {
        "model_seed": refit.seed,
        "protocol_hash": refit.protocol_hash,
        "config_hash": refit.resolved_config_hash,
        "manifest_hash": refit.manifest_sha256,
        "label_order": LABEL_ORDER,
    }
    observed: dict[str, object] = {
        "model_seed": prediction.model_seed,
        "protocol_hash": prediction.protocol_hash,
        "config_hash": prediction.config_hash,
        "manifest_hash": prediction.manifest_hash,
        "label_order": prediction.label_order,
    }
    mismatches = [name for name in expected if expected[name] != observed[name]]
    if mismatches:
        raise ReleaseIntegrityError(
            "fold-9 prediction/refit mismatch: " + ", ".join(mismatches)
        )
    extra = prediction.extra_metadata
    if extra.get("lineage") != "frozen_refit":
        raise ReleaseGateError("fold-9 prediction must come from frozen_refit lineage")
    if (
        _normalized_hash(extra.get("checkpoint_sha256"), "checkpoint_sha256")
        != refit.final_checkpoint_sha256
    ):
        raise ReleaseIntegrityError("fold-9 prediction checkpoint hash mismatch")
    if (
        _normalized_hash(extra.get("normalization_sha256"), "normalization_sha256")
        != refit.normalization_sha256
    ):
        raise ReleaseIntegrityError("fold-9 prediction normalization hash mismatch")


def _verify_calibration_sources(
    bundle: CalibrationBundle, *, protocol: ExperimentProtocol
) -> None:
    loaded_predictions: list[PredictionArtifact] = []
    for member in bundle.members:
        if _prefixed_file_sha256(member.checkpoint_path) != member.checkpoint_sha256:
            raise ReleaseIntegrityError(f"refit checkpoint changed: {member.member_id}")
        if (
            _prefixed_file_sha256(member.resolved_config_path)
            != member.resolved_config_file_sha256
        ):
            raise ReleaseIntegrityError(f"resolved config changed: {member.member_id}")
        if (
            _prefixed_file_sha256(member.normalization_path)
            != member.normalization_sha256
        ):
            raise ReleaseIntegrityError(f"normalization changed: {member.member_id}")
        if sha256_file(member.prediction_path) != member.prediction_npz_sha256:
            raise ReleaseIntegrityError(f"fold-9 prediction changed: {member.member_id}")
        if sha256_file(member.prediction_sidecar_path) != member.prediction_sidecar_sha256:
            raise ReleaseIntegrityError(f"fold-9 sidecar changed: {member.member_id}")
        if sha256_file(member.decision_path) != member.decision_file_sha256:
            raise ReleaseIntegrityError(f"fold-9 decision changed: {member.member_id}")
        prediction = load_prediction_artifact(
            member.prediction_path,
            protocol=protocol,
            expected_config_hash=member.resolved_config_hash,
            expected_manifest_hash=bundle.manifest_sha256,
        )
        decision = load_calibration_decisions(member.decision_path, protocol=protocol)
        if prediction.integrity_sha256 != member.prediction_artifact_sha256:
            raise ReleaseIntegrityError("fold-9 prediction artifact hash mismatch")
        if decision.integrity_sha256 != member.decision_artifact_sha256:
            raise ReleaseIntegrityError("fold-9 decision artifact hash mismatch")
        _verify_loaded_calibration_semantics(
            bundle,
            member,
            prediction,
            decision,
        )
        loaded_predictions.append(prediction)
    if len({item.integrity_sha256 for item in loaded_predictions}) != 6:
        raise ReleaseIntegrityError(
            "calibration reload does not contain six distinct predictions"
        )
    first = loaded_predictions[0]
    for prediction in loaded_predictions[1:]:
        try:
            assert_prediction_artifacts_aligned(first, prediction)
        except (RuntimeError, ValueError) as error:
            raise ReleaseIntegrityError(
                f"fold-9 predictions are not aligned across the six-member batch: {error}"
            ) from error


def _verify_loaded_calibration_semantics(
    bundle: CalibrationBundle,
    member: CalibrationMember,
    prediction: PredictionArtifact,
    decision: CalibrationDecisionArtifact,
) -> None:
    """Recompute all semantic member bindings during a bundle reload."""

    if prediction.fold_role is not FoldRole.CALIBRATION or (
        prediction.folds != CALIBRATION_FOLDS
    ):
        raise ReleaseIntegrityError("reloaded calibration prediction is not fold 9")
    expected_prediction: dict[str, object] = {
        "model_name": member.model_name,
        "model_seed": member.seed,
        "protocol_hash": bundle.protocol_hash,
        "config_hash": member.resolved_config_hash,
        "manifest_hash": bundle.manifest_sha256,
        "label_order": bundle.label_order,
        "artifact_sha256": member.prediction_artifact_sha256,
        "alignment_sha256": member.prediction_alignment_sha256,
    }
    observed_prediction: dict[str, object] = {
        "model_name": prediction.model_name,
        "model_seed": prediction.model_seed,
        "protocol_hash": prediction.protocol_hash,
        "config_hash": prediction.config_hash,
        "manifest_hash": prediction.manifest_hash,
        "label_order": prediction.label_order,
        "artifact_sha256": prediction.integrity_sha256,
        "alignment_sha256": prediction.alignment_sha256,
    }
    mismatches = [
        field
        for field, expected in expected_prediction.items()
        if observed_prediction[field] != expected
    ]
    if mismatches:
        raise ReleaseIntegrityError(
            "reloaded fold-9 prediction differs from bundle member: "
            + ", ".join(mismatches)
        )
    if member.prediction_sidecar_path != member.prediction_path.with_suffix(".json"):
        raise ReleaseIntegrityError("fold-9 sidecar is not the prediction same-stem JSON")
    extra = prediction.extra_metadata
    if extra.get("lineage") != "frozen_refit":
        raise ReleaseIntegrityError("reloaded fold-9 prediction lineage is not frozen_refit")
    if (
        _normalized_hash(extra.get("checkpoint_sha256"), "checkpoint_sha256")
        != member.checkpoint_sha256
    ):
        raise ReleaseIntegrityError("reloaded fold-9 checkpoint binding differs")
    if (
        _normalized_hash(extra.get("normalization_sha256"), "normalization_sha256")
        != member.normalization_sha256
    ):
        raise ReleaseIntegrityError("reloaded fold-9 normalization binding differs")

    expected_decision: dict[str, object] = {
        "model_name": prediction.model_name,
        "model_seed": prediction.model_seed,
        "protocol_hash": prediction.protocol_hash,
        "config_hash": prediction.config_hash,
        "manifest_hash": prediction.manifest_hash,
        "label_order": prediction.label_order,
        "source_prediction_sha256": prediction.integrity_sha256,
        "source_alignment_sha256": prediction.alignment_sha256,
        "artifact_sha256": member.decision_artifact_sha256,
    }
    observed_decision: dict[str, object] = {
        "model_name": decision.model_name,
        "model_seed": decision.model_seed,
        "protocol_hash": decision.protocol_hash,
        "config_hash": decision.config_hash,
        "manifest_hash": decision.manifest_hash,
        "label_order": decision.label_order,
        "source_prediction_sha256": decision.source_prediction_sha256,
        "source_alignment_sha256": decision.source_alignment_sha256,
        "artifact_sha256": decision.integrity_sha256,
    }
    mismatches = [
        field
        for field, expected in expected_decision.items()
        if observed_decision[field] != expected
    ]
    if mismatches:
        raise ReleaseIntegrityError(
            "reloaded fold-9 decision differs from prediction/bundle member: "
            + ", ".join(mismatches)
        )
    if decision.temperature_scaling.source_folds != CALIBRATION_FOLDS or (
        decision.threshold_optimization.source_folds != CALIBRATION_FOLDS
    ):
        raise ReleaseIntegrityError("reloaded calibration policy was not fitted on fold 9")
    if not math.isclose(
        decision.temperature_scaling.temperature,
        member.temperature,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ReleaseIntegrityError("reloaded calibration temperature differs from bundle")
    if decision.threshold_optimization.thresholds != member.thresholds:
        raise ReleaseIntegrityError("reloaded calibration thresholds differ from bundle")
    loaded_gates = tuple(gate.to_dict() for gate in decision.coverage_gates)
    if loaded_gates != tuple(dict(gate) for gate in member.entropy_gates):
        raise ReleaseIntegrityError("reloaded entropy gates differ from bundle")


def _parse_refit_bundle(
    payload: Mapping[str, object], *, protocol: ExperimentProtocol
) -> RefitBundle:
    required = {
        "schema_version",
        "artifact_type",
        "protocol_hash",
        "manifest_sha256",
        "normalization_sha256",
        "label_order",
        "release_grid",
        "members",
        "created_at_utc",
        "artifact_sha256",
    }
    _exact_keys(payload, required, "refit release bundle")
    if payload["schema_version"] != REFIT_BUNDLE_SCHEMA_VERSION:
        raise ReleaseIntegrityError("unsupported refit bundle schema")
    if payload["artifact_type"] != REFIT_BUNDLE_TYPE:
        raise ReleaseIntegrityError("unexpected refit bundle type")
    stored = _hash_string(payload["artifact_sha256"], "artifact_sha256")
    unhashed = dict(payload)
    del unhashed["artifact_sha256"]
    if canonical_sha256(unhashed) != stored:
        raise ReleaseIntegrityError("refit bundle artifact hash mismatch")
    if payload["protocol_hash"] != protocol.protocol_hash:
        raise ReleaseIntegrityError("refit bundle protocol mismatch")
    _validate_release_grid(payload["release_grid"], calibration=False)
    raw_members = _sequence(payload["members"], "members")
    members = tuple(_parse_refit_member(item) for item in raw_members)
    bundle = RefitBundle(
        protocol_hash=_hash_string(payload["protocol_hash"], "protocol_hash"),
        manifest_sha256=_hash_string(payload["manifest_sha256"], "manifest_sha256"),
        normalization_sha256=_hash_string(
            payload["normalization_sha256"], "normalization_sha256"
        ),
        label_order=_label_order(payload["label_order"]),
        members=members,
        created_at_utc=_timestamp(_string(payload["created_at_utc"], "created_at_utc")),
        artifact_sha256=stored,
    )
    _validate_refit_bundle(bundle, require_integrity=True)
    return bundle


def _parse_refit_member(value: object) -> RefitMember:
    root = _mapping(value, "refit member")
    required = {
        "member_id",
        "comparison_id",
        "architecture",
        "seed",
        "run_name",
        "run_dir",
        "refit_completion",
        "files",
        "freeze",
        "protocol_hash",
        "lineage_sha256",
    }
    _exact_keys(root, required, "refit member")
    receipt = _mapping(root["refit_completion"], "refit_completion")
    _exact_keys(receipt, {"path", "artifact_sha256"}, "refit_completion")
    files = _mapping(root["files"], "refit member.files")
    _exact_keys(
        files,
        {
            "final_checkpoint",
            "resolved_config",
            "metadata",
            "protocol",
            "history",
            "manifest",
            "normalization",
            "source_checkpoint",
        },
        "refit member.files",
    )
    checkpoint = _completion_file_entry(files["final_checkpoint"], "final_checkpoint")
    resolved = _completion_file_entry(
        files["resolved_config"], "resolved_config", config_hash=True
    )
    metadata = _completion_file_entry(files["metadata"], "metadata")
    protocol_file = _completion_file_entry(files["protocol"], "protocol")
    history = _completion_file_entry(files["history"], "history")
    manifest = _completion_file_entry(files["manifest"], "manifest")
    normalization = _completion_file_entry(files["normalization"], "normalization")
    source_checkpoint = _completion_file_entry(
        files["source_checkpoint"], "source_checkpoint"
    )
    freeze = _mapping(root["freeze"], "refit member.freeze")
    _exact_keys(
        freeze,
        {
            "refit_folds",
            "normalization_folds",
            "frozen_epochs",
            "freeze_artifact_path",
            "freeze_artifact_sha256",
            "recipe_sha256",
            "source_member_completion_path",
            "source_member_completion_sha256",
            "selection_provenance",
            "selection_lineage_sha256",
        },
        "refit member.freeze",
    )
    if _integer_tuple(freeze["refit_folds"], "refit_folds") != REFIT_FOLDS:
        raise ReleaseIntegrityError("refit member has unsafe refit folds")
    if _integer_tuple(freeze["normalization_folds"], "normalization_folds") != TRAIN_FOLDS:
        raise ReleaseIntegrityError("refit member has unsafe normalization folds")
    selection = _mapping(freeze["selection_provenance"], "selection_provenance")
    selection_hash = _hash_string(
        freeze["selection_lineage_sha256"], "selection_lineage_sha256"
    )
    if canonical_sha256(selection) != selection_hash:
        raise ReleaseIntegrityError("selection lineage hash mismatch")
    member = RefitMember(
        member_id=_string(root["member_id"], "member_id"),
        comparison_id=_string(root["comparison_id"], "comparison_id"),
        architecture=_string(root["architecture"], "architecture"),
        seed=_integer(root["seed"], "seed", minimum=0),
        run_name=_string(root["run_name"], "run_name"),
        run_dir=Path(_string(root["run_dir"], "run_dir")),
        completion_path=Path(_string(receipt["path"], "refit_completion.path")),
        completion_sha256=_hash_string(
            receipt["artifact_sha256"], "refit_completion.artifact_sha256"
        ),
        freeze_artifact_path=Path(
            _string(freeze["freeze_artifact_path"], "freeze_artifact_path")
        ),
        freeze_artifact_sha256=_hash_string(
            freeze["freeze_artifact_sha256"], "freeze_artifact_sha256"
        ),
        recipe_sha256=_hash_string(freeze["recipe_sha256"], "recipe_sha256"),
        source_member_completion_path=Path(
            _string(
                freeze["source_member_completion_path"],
                "source_member_completion_path",
            )
        ),
        source_member_completion_sha256=_hash_string(
            freeze["source_member_completion_sha256"],
            "source_member_completion_sha256",
        ),
        final_checkpoint_path=checkpoint[0],
        final_checkpoint_sha256=checkpoint[1],
        resolved_config_path=resolved[0],
        resolved_config_file_sha256=resolved[1],
        resolved_config_hash=cast(str, resolved[2]),
        metadata_path=metadata[0],
        metadata_sha256=metadata[1],
        protocol_path=protocol_file[0],
        protocol_file_sha256=protocol_file[1],
        history_path=history[0],
        history_sha256=history[1],
        protocol_hash=_hash_string(root["protocol_hash"], "protocol_hash"),
        manifest_path=manifest[0],
        manifest_sha256=manifest[1],
        normalization_path=normalization[0],
        normalization_sha256=normalization[1],
        source_checkpoint_path=source_checkpoint[0],
        source_checkpoint_sha256=source_checkpoint[1],
        frozen_epochs=_integer(freeze["frozen_epochs"], "frozen_epochs", minimum=1),
        selection_provenance=selection,
        selection_lineage_sha256=selection_hash,
        lineage_sha256=_hash_string(root["lineage_sha256"], "lineage_sha256"),
    )
    if canonical_sha256(_refit_lineage_payload(member)) != member.lineage_sha256:
        raise ReleaseIntegrityError("refit member lineage hash mismatch")
    return member

def _parse_calibration_bundle(
    payload: Mapping[str, object], *, protocol: ExperimentProtocol
) -> CalibrationBundle:
    required = {
        "schema_version",
        "artifact_type",
        "refit_bundle_sha256",
        "protocol_hash",
        "manifest_sha256",
        "normalization_sha256",
        "label_order",
        "release_grid",
        "members",
        "created_at_utc",
        "artifact_sha256",
    }
    _exact_keys(payload, required, "calibration release bundle")
    if payload["schema_version"] != CALIBRATION_BUNDLE_SCHEMA_VERSION:
        raise ReleaseIntegrityError("unsupported calibration bundle schema")
    if payload["artifact_type"] != CALIBRATION_BUNDLE_TYPE:
        raise ReleaseIntegrityError("unexpected calibration bundle type")
    stored = _hash_string(payload["artifact_sha256"], "artifact_sha256")
    unhashed = dict(payload)
    del unhashed["artifact_sha256"]
    if canonical_sha256(unhashed) != stored:
        raise ReleaseIntegrityError("calibration bundle artifact hash mismatch")
    if payload["protocol_hash"] != protocol.protocol_hash:
        raise ReleaseIntegrityError("calibration bundle protocol mismatch")
    _validate_release_grid(payload["release_grid"], calibration=True)
    members = tuple(
        _parse_calibration_member(item) for item in _sequence(payload["members"], "members")
    )
    bundle = CalibrationBundle(
        refit_bundle_sha256=_hash_string(
            payload["refit_bundle_sha256"], "refit_bundle_sha256"
        ),
        protocol_hash=_hash_string(payload["protocol_hash"], "protocol_hash"),
        manifest_sha256=_hash_string(payload["manifest_sha256"], "manifest_sha256"),
        normalization_sha256=_hash_string(
            payload["normalization_sha256"], "normalization_sha256"
        ),
        label_order=_label_order(payload["label_order"]),
        members=members,
        created_at_utc=_timestamp(_string(payload["created_at_utc"], "created_at_utc")),
        artifact_sha256=stored,
    )
    _validate_calibration_bundle(bundle, require_integrity=True)
    return bundle


def _parse_calibration_member(value: object) -> CalibrationMember:
    root = _mapping(value, "calibration member")
    _exact_keys(
        root,
        {
            "member_id",
            "architecture",
            "seed",
            "model_name",
            "freeze_lineage",
            "fold9_prediction",
            "decision",
            "independent_fit_sha256",
        },
        "calibration member",
    )
    freeze = _mapping(root["freeze_lineage"], "freeze_lineage")
    _exact_keys(
        freeze,
        {
            "refit_lineage_sha256",
            "checkpoint_path",
            "checkpoint_sha256",
            "resolved_config_path",
            "resolved_config_file_sha256",
            "resolved_config_hash",
            "normalization_path",
            "normalization_sha256",
        },
        "freeze_lineage",
    )
    prediction = _mapping(root["fold9_prediction"], "fold9_prediction")
    _exact_keys(
        prediction,
        {
            "path",
            "sidecar_path",
            "npz_sha256",
            "sidecar_sha256",
            "artifact_sha256",
            "alignment_sha256",
            "folds",
        },
        "fold9_prediction",
    )
    if _integer_tuple(prediction["folds"], "fold9_prediction.folds") != CALIBRATION_FOLDS:
        raise ReleaseIntegrityError("calibration member must bind fold 9 only")
    decision = _mapping(root["decision"], "decision")
    _exact_keys(
        decision,
        {
            "path",
            "file_sha256",
            "artifact_sha256",
            "temperature",
            "thresholds",
            "entropy_method",
            "entropy_gates",
        },
        "decision",
    )
    if decision["entropy_method"] != "mean_normalized_binary_entropy":
        raise ReleaseIntegrityError("unsupported entropy gate method")
    gates = tuple(
        _mapping(item, "entropy gate")
        for item in _sequence(decision["entropy_gates"], "entropy_gates")
    )
    member = CalibrationMember(
        member_id=_string(root["member_id"], "member_id"),
        architecture=_string(root["architecture"], "architecture"),
        seed=_integer(root["seed"], "seed", minimum=0),
        model_name=_string(root["model_name"], "model_name"),
        refit_lineage_sha256=_hash_string(
            freeze["refit_lineage_sha256"], "refit_lineage_sha256"
        ),
        checkpoint_path=Path(
            _string(freeze["checkpoint_path"], "checkpoint_path")
        ),
        resolved_config_hash=_hash_string(
            freeze["resolved_config_hash"], "resolved_config_hash"
        ),
        checkpoint_sha256=_hash_string(
            freeze["checkpoint_sha256"], "checkpoint_sha256"
        ),
        resolved_config_path=Path(
            _string(freeze["resolved_config_path"], "resolved_config_path")
        ),
        resolved_config_file_sha256=_hash_string(
            freeze["resolved_config_file_sha256"],
            "resolved_config_file_sha256",
        ),
        normalization_path=Path(
            _string(freeze["normalization_path"], "normalization_path")
        ),
        normalization_sha256=_hash_string(
            freeze["normalization_sha256"], "normalization_sha256"
        ),
        prediction_path=Path(_string(prediction["path"], "prediction.path")),
        prediction_sidecar_path=Path(
            _string(prediction["sidecar_path"], "prediction.sidecar_path")
        ),
        prediction_npz_sha256=_raw_hash_string(
            prediction["npz_sha256"], "prediction.npz_sha256"
        ),
        prediction_sidecar_sha256=_raw_hash_string(
            prediction["sidecar_sha256"], "prediction.sidecar_sha256"
        ),
        prediction_artifact_sha256=_hash_string(
            prediction["artifact_sha256"], "prediction.artifact_sha256"
        ),
        prediction_alignment_sha256=_hash_string(
            prediction["alignment_sha256"], "prediction.alignment_sha256"
        ),
        decision_path=Path(_string(decision["path"], "decision.path")),
        decision_file_sha256=_raw_hash_string(
            decision["file_sha256"], "decision.file_sha256"
        ),
        decision_artifact_sha256=_hash_string(
            decision["artifact_sha256"], "decision.artifact_sha256"
        ),
        temperature=_finite_float(decision["temperature"], "temperature", minimum=0.0, strict=True),
        thresholds=_float_tuple(decision["thresholds"], "thresholds"),
        entropy_gates=gates,
        independent_fit_sha256=_hash_string(
            root["independent_fit_sha256"], "independent_fit_sha256"
        ),
    )
    expected_fit = canonical_sha256(
        {
            "member_id": member.member_id,
            "refit_lineage_sha256": member.refit_lineage_sha256,
            "prediction_artifact_sha256": member.prediction_artifact_sha256,
            "decision_artifact_sha256": member.decision_artifact_sha256,
        }
    )
    if expected_fit != member.independent_fit_sha256:
        raise ReleaseIntegrityError("independent calibration fit hash mismatch")
    return member


def _validate_refit_bundle(bundle: RefitBundle, *, require_integrity: bool = False) -> None:
    if require_integrity and bundle.artifact_sha256 is None:
        raise ReleaseStateError("refit bundle must be loaded from an integrity-bound artifact")
    if bundle.label_order != LABEL_ORDER:
        raise ReleaseGateError("refit bundle label order is not canonical")
    _validate_member_grid(bundle.members)
    if any(member.protocol_hash != bundle.protocol_hash for member in bundle.members):
        raise ReleaseIntegrityError("refit member protocol hashes differ")
    if any(member.manifest_sha256 != bundle.manifest_sha256 for member in bundle.members):
        raise ReleaseIntegrityError("refit member manifest hashes differ")
    if any(
        member.normalization_sha256 != bundle.normalization_sha256
        for member in bundle.members
    ):
        raise ReleaseIntegrityError("refit member normalization hashes differ")
    if len({member.comparison_id for member in bundle.members}) != 1:
        raise ReleaseIntegrityError("refit members do not share one comparison")
    if len({member.freeze_artifact_sha256 for member in bundle.members}) != 1:
        raise ReleaseIntegrityError("refit members do not share one freeze")
    if len({member.recipe_sha256 for member in bundle.members}) != 6:
        raise ReleaseIntegrityError("refit recipes are not one-to-one with members")
    for architecture in EXPECTED_ARCHITECTURES:
        budgets = {
            member.frozen_epochs
            for member in bundle.members
            if member.architecture == architecture
        }
        if len(budgets) != 1:
            raise ReleaseIntegrityError(
                f"{architecture} median epoch budget differs across refit seeds"
            )


def _validate_calibration_bundle(
    bundle: CalibrationBundle, *, require_integrity: bool = False
) -> None:
    if require_integrity and bundle.artifact_sha256 is None:
        raise ReleaseStateError("calibration bundle must be integrity-bound")
    if bundle.label_order != LABEL_ORDER:
        raise ReleaseGateError("calibration bundle label order is not canonical")
    _validate_member_grid(bundle.members)
    for member in bundle.members:
        if len(member.thresholds) != len(LABEL_ORDER):
            raise ReleaseGateError("each calibration member must contain five thresholds")
        if any(not 0.0 <= value <= 1.0 for value in member.thresholds):
            raise ReleaseGateError("calibration thresholds must lie in [0, 1]")
        if not math.isfinite(member.temperature) or member.temperature <= 0.0:
            raise ReleaseGateError("calibration temperatures must be finite and positive")
        if not member.entropy_gates:
            raise ReleaseGateError("each calibration member must contain entropy gates")
        if member.normalization_sha256 != bundle.normalization_sha256:
            raise ReleaseIntegrityError("calibration member normalization hash differs")
    if len({member.independent_fit_sha256 for member in bundle.members}) != 6:
        raise ReleaseGateError("calibration bundle requires six independent fits")


def _validate_member_grid(members: Sequence[object]) -> None:
    if len(members) != 6:
        raise ReleaseGateError("release grid requires exactly six members")
    observed: set[tuple[str, int]] = set()
    for member in members:
        architecture = getattr(member, "architecture", None)
        seed = getattr(member, "seed", None)
        if not isinstance(architecture, str) or isinstance(seed, bool) or not isinstance(seed, int):
            raise ReleaseGateError("release members require architecture and integer seed")
        pair = (architecture, seed)
        if pair in observed:
            raise ReleaseGateError(f"duplicate release member {pair!r}")
        observed.add(pair)
    expected = {
        (architecture, seed)
        for architecture in EXPECTED_ARCHITECTURES
        for seed in EXPECTED_SEEDS
    }
    if observed != expected:
        raise ReleaseGateError(
            f"release grid mismatch: missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}"
        )


def _validate_release_grid(value: object, *, calibration: bool) -> None:
    grid = _mapping(value, "release_grid")
    common = {"architectures", "seeds", "member_count"}
    extra = (
        {
            "calibration_folds",
            "temperature_fits",
            "thresholds_per_fit",
            "temperature_scope",
            "retuning_after_freeze",
        }
        if calibration
        else {"refit_folds", "normalization_folds"}
    )
    _exact_keys(grid, common | extra, "release_grid")
    if tuple(_string_sequence(grid["architectures"], "architectures")) != EXPECTED_ARCHITECTURES:
        raise ReleaseIntegrityError("release architectures are not canonical")
    if _integer_tuple(grid["seeds"], "seeds") != EXPECTED_SEEDS or grid["member_count"] != 6:
        raise ReleaseIntegrityError("release seed grid is not canonical")
    if calibration:
        if _integer_tuple(grid["calibration_folds"], "calibration_folds") != CALIBRATION_FOLDS:
            raise ReleaseIntegrityError("calibration grid must use fold 9")
        if grid["temperature_fits"] != 6 or grid["thresholds_per_fit"] != len(LABEL_ORDER):
            raise ReleaseIntegrityError("calibration fit cardinality mismatch")
        if grid["temperature_scope"] != "one_global_temperature_per_member":
            raise ReleaseIntegrityError("temperature scope mismatch")
        if grid["retuning_after_freeze"] is not False:
            raise ReleaseIntegrityError("retuning must remain disabled")
    else:
        if _integer_tuple(grid["refit_folds"], "refit_folds") != REFIT_FOLDS:
            raise ReleaseIntegrityError("release refit folds mismatch")
        if _integer_tuple(grid["normalization_folds"], "normalization_folds") != TRAIN_FOLDS:
            raise ReleaseIntegrityError("release normalization folds mismatch")


def _refit_lineage_payload(member: RefitMember) -> dict[str, object]:
    return {
        "member_id": member.member_id,
        "comparison_id": member.comparison_id,
        "architecture": member.architecture,
        "seed": member.seed,
        "run_name": member.run_name,
        "refit_completion_sha256": member.completion_sha256,
        "checkpoint_sha256": member.final_checkpoint_sha256,
        "resolved_config_hash": member.resolved_config_hash,
        "metadata_sha256": member.metadata_sha256,
        "history_sha256": member.history_sha256,
        "protocol_hash": member.protocol_hash,
        "manifest_sha256": member.manifest_sha256,
        "normalization_sha256": member.normalization_sha256,
        "freeze_artifact_sha256": member.freeze_artifact_sha256,
        "recipe_sha256": member.recipe_sha256,
        "source_member_completion_sha256": (
            member.source_member_completion_sha256
        ),
        "selection_lineage_sha256": member.selection_lineage_sha256,
        "frozen_epochs": member.frozen_epochs,
    }


def _completion_file_entry(
    value: object, context: str, *, config_hash: bool = False
) -> tuple[Path, str, str | None]:
    root = _mapping(value, context)
    required = {"path", "sha256"} | ({"config_hash"} if config_hash else set())
    _exact_keys(root, required, context)
    return (
        Path(_string(root["path"], f"{context}.path")),
        _hash_string(root["sha256"], f"{context}.sha256"),
        _hash_string(root["config_hash"], f"{context}.config_hash")
        if config_hash
        else None,
    )


def _write_json(path: Path, payload: Mapping[str, object], *, replace: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise FileExistsError(f"immutable release artifact already exists: {path}")
    descriptor, raw_temp = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(raw_temp)
    try:
        serialized = json.dumps(
            payload, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True
        )
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(serialized)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if not replace and path.exists():
            raise FileExistsError(f"immutable release artifact already exists: {path}")
        os.replace(temporary, path)
    finally:
        with suppress(OSError):
            temporary.unlink(missing_ok=True)


def _canonical_object(value: Mapping[str, object]) -> object:
    try:
        return json.loads(
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ReleaseGateError("release lineage must be finite JSON") from error


def _mapping(value: object, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ReleaseIntegrityError(f"{context} must be a string-keyed mapping")
    return cast(Mapping[str, Any], value)


def _sequence(value: object, context: str) -> list[object]:
    if not isinstance(value, list):
        raise ReleaseIntegrityError(f"{context} must be an array")
    return cast(list[object], value)


def _exact_keys(value: Mapping[str, object], required: set[str], context: str) -> None:
    missing = sorted(required.difference(value))
    unexpected = sorted(set(value).difference(required))
    if missing or unexpected:
        raise ReleaseIntegrityError(
            f"{context} keys mismatch: missing={missing}, unexpected={unexpected}"
        )


def _string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReleaseIntegrityError(f"{context} must be a non-empty string")
    return value


def _integer(value: object, context: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ReleaseIntegrityError(f"{context} must be an integer >= {minimum}")
    return value


def _integer_tuple(value: object, context: str) -> tuple[int, ...]:
    return tuple(_integer(item, f"{context} item", minimum=1) for item in _sequence(value, context))


def _string_sequence(value: object, context: str) -> tuple[str, ...]:
    return tuple(_string(item, f"{context} item") for item in _sequence(value, context))


def _float_tuple(value: object, context: str) -> tuple[float, ...]:
    return tuple(
        _finite_float(item, f"{context} item", minimum=0.0) for item in _sequence(value, context)
    )


def _finite_float(
    value: object,
    context: str,
    *,
    minimum: float,
    strict: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReleaseIntegrityError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < minimum or (strict and result == minimum):
        raise ReleaseIntegrityError(f"{context} must be finite and valid")
    return result


def _hash_string(value: object, context: str) -> str:
    text = _string(value, context)
    if not text.startswith("sha256:"):
        raise ReleaseIntegrityError(f"{context} must use the sha256: prefix")
    _validate_hex(text.removeprefix("sha256:"), context)
    return text


def _normalized_hash(value: object, context: str) -> str:
    text = _string(value, context)
    digest = text.removeprefix("sha256:")
    _validate_hex(digest, context)
    return "sha256:" + digest


def _raw_hash_string(value: object, context: str) -> str:
    text = _string(value, context)
    if text.startswith("sha256:"):
        raise ReleaseIntegrityError(f"{context} must be an unprefixed file hash")
    _validate_hex(text, context)
    return text


def _validate_hex(value: str, context: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ReleaseIntegrityError(f"{context} must contain a lowercase SHA-256 digest")


def _label_order(value: object) -> tuple[str, ...]:
    labels = _string_sequence(value, "label_order")
    if labels != LABEL_ORDER:
        raise ReleaseIntegrityError("label_order must be canonical")
    return labels


def _timestamp(value: str | None) -> str:
    candidate = value or datetime.now(UTC).isoformat()
    try:
        parsed = datetime.fromisoformat(candidate.replace("Z", "+00:00"))
    except ValueError as error:
        raise ReleaseGateError("timestamp must be ISO-8601") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ReleaseGateError("timestamp must include a UTC offset")
    return parsed.astimezone(UTC).isoformat()


__all__ = [
    "CALIBRATION_BUNDLE_SCHEMA_VERSION",
    "CALIBRATION_BUNDLE_TYPE",
    "EXPECTED_ARCHITECTURES",
    "EXPECTED_SEEDS",
    "REFIT_BUNDLE_SCHEMA_VERSION",
    "REFIT_BUNDLE_TYPE",
    "CalibrationBundle",
    "CalibrationMember",
    "RefitBundle",
    "RefitMember",
    "ReleaseGateError",
    "ReleaseIntegrityError",
    "ReleaseStateError",
    "canonical_sha256",
    "create_calibration_bundle",
    "create_refit_bundle",
    "export_fold9_predictions",
    "fit_calibration_bundle",
    "load_calibration_bundle",
    "load_refit_bundle",
    "materialize_demo_policy_payload",
    "read_json_mapping",
    "save_calibration_bundle",
    "save_refit_bundle",
    "sha256_file",
    "write_new_hashed_json",
]
