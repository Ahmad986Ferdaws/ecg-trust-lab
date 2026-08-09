"""Persistent, one-time, exact-six execution for the sealed fold-10 release.

The opening ledger is committed before a final-test token is issued or a
fold-10 prediction is loaded/exported.  A crashed process may resume only when
the complete content-addressed batch plan is identical.  This module never
fits calibration parameters; it only applies the six policies frozen in a
verified :class:`~ecg_trust.release_gates.CalibrationBundle`.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import socket
import statistics
import tempfile
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import psutil  # type: ignore[import-untyped]
from filelock import FileLock, Timeout
from numpy.typing import NDArray

from ecg_trust.audit import paired_model_difference_intervals
from ecg_trust.decisioning import (
    generate_final_report,
    load_calibration_decisions,
    save_final_report,
    verify_final_report,
)
from ecg_trust.final_evaluation_spec import (
    FinalEvaluationSpecError,
    load_final_evaluation_spec,
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
    FINAL_TEST_CONFIRMATION,
    FINAL_TEST_FOLDS,
    LABEL_ORDER,
    ExperimentProtocol,
    FinalTestAccessToken,
    FoldRole,
    authorize_final_test_access,
)
from ecg_trust.release_gates import (
    EXPECTED_ARCHITECTURES,
    EXPECTED_SEEDS,
    CalibrationBundle,
    CalibrationMember,
    RefitBundle,
    RefitMember,
    ReleaseGateError,
    ReleaseIntegrityError,
    ReleaseStateError,
    canonical_sha256,
    load_calibration_bundle,
    load_refit_bundle,
    read_json_mapping,
    sha256_file,
    write_new_hashed_json,
)
from ecg_trust.subgroup_artifact import (
    SubgroupArtifactError,
    load_subgroup_artifact,
)

FINAL_BATCH_PLAN_SCHEMA_VERSION = 2
FINAL_BATCH_PLAN_TYPE = "ecg_trust.final_test_batch_plan"
FINAL_LEDGER_SCHEMA_VERSION = 2
FINAL_LEDGER_TYPE = "ecg_trust.final_test_opening_ledger"
FINAL_OPENING_MARKER_SCHEMA_VERSION = 2
FINAL_OPENING_MARKER_TYPE = "ecg_trust.final_test_canonical_opening_marker"
ARCHITECTURE_REPORT_TYPE = "ecg_trust.final_architecture_aggregate"
PAIRED_BOOTSTRAP_MANIFEST_TYPE = "ecg_trust.paired_patient_bootstrap_manifest"
FINAL_BATCH_SUMMARY_TYPE = "ecg_trust.final_batch_summary"
FINAL_LEDGER_LOCK_SCHEMA_VERSION = 2
FINAL_LEDGER_LOCK_TYPE = "ecg_trust.final_test_ledger_writer_lock"
FINAL_PREDICTION_ORPHAN_DIRECTORY = ".final-prediction-orphans"

PredictionExporter = Callable[..., PredictionExportResult]


@dataclass(frozen=True, slots=True)
class FinalBatchSettings:
    """Preregistered non-tuning settings that must match on resume."""

    output_directory: Path
    subgroup_path: Path
    subgroup_sha256: str
    batch_size: int | None = None
    num_workers: int | None = None
    device: str = "auto"
    bf16: bool = True
    bootstrap_resamples: int = 1_000
    bootstrap_seed: int = 20_260_808
    bootstrap_confidence: float = 0.95
    bootstrap_minimum_valid: int | None = None
    minimum_group_samples: int = 30
    minimum_group_patients: int = 20
    ece_bins: int = 15

    def __post_init__(self) -> None:
        if self.batch_size is not None and (
            isinstance(self.batch_size, bool) or self.batch_size < 1
        ):
            raise ReleaseGateError("batch_size must be a positive integer")
        if self.num_workers is not None and (
            isinstance(self.num_workers, bool) or self.num_workers < 0
        ):
            raise ReleaseGateError("num_workers must be a non-negative integer")
        if not self.device.strip():
            raise ReleaseGateError("device must be non-empty")
        _positive_int(self.bootstrap_resamples, "bootstrap_resamples", minimum=2)
        _positive_int(self.bootstrap_seed, "bootstrap_seed", minimum=0)
        _positive_int(
            self.minimum_group_samples, "minimum_group_samples", minimum=1
        )
        _positive_int(
            self.minimum_group_patients, "minimum_group_patients", minimum=1
        )
        _positive_int(self.ece_bins, "ece_bins", minimum=2)
        if self.bootstrap_minimum_valid is not None:
            minimum = _positive_int(
                self.bootstrap_minimum_valid,
                "bootstrap_minimum_valid",
                minimum=1,
            )
            if minimum > self.bootstrap_resamples:
                raise ReleaseGateError(
                    "bootstrap_minimum_valid cannot exceed bootstrap_resamples"
                )
        if not math.isfinite(self.bootstrap_confidence) or not (
            0.0 < self.bootstrap_confidence < 1.0
        ):
            raise ReleaseGateError("bootstrap_confidence must lie strictly in (0, 1)")
        if len(self.subgroup_sha256) != 64:
            raise ReleaseGateError("subgroup_sha256 must be an unprefixed SHA-256")

    @classmethod
    def create(
        cls,
        *,
        output_directory: str | Path,
        subgroup_path: str | Path,
        batch_size: int | None = None,
        num_workers: int | None = None,
        device: str = "auto",
        bf16: bool = True,
        bootstrap_resamples: int = 1_000,
        bootstrap_seed: int = 20_260_808,
        bootstrap_confidence: float = 0.95,
        bootstrap_minimum_valid: int | None = None,
        minimum_group_samples: int = 30,
        minimum_group_patients: int = 20,
        ece_bins: int = 15,
    ) -> FinalBatchSettings:
        subgroup = Path(subgroup_path).resolve()
        return cls(
            output_directory=Path(output_directory).resolve(),
            subgroup_path=subgroup,
            subgroup_sha256=sha256_file(subgroup),
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            bf16=bf16,
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_seed=bootstrap_seed,
            bootstrap_confidence=bootstrap_confidence,
            bootstrap_minimum_valid=bootstrap_minimum_valid,
            minimum_group_samples=minimum_group_samples,
            minimum_group_patients=minimum_group_patients,
            ece_bins=ece_bins,
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "output_directory": str(self.output_directory),
            "subgroups": {
                "path": str(self.subgroup_path),
                "sha256": self.subgroup_sha256,
            },
            "inference": {
                "batch_size": self.batch_size,
                "num_workers": self.num_workers,
                "device": self.device,
                "bf16": self.bf16,
            },
            "evaluation": {
                "bootstrap_resamples": self.bootstrap_resamples,
                "bootstrap_seed": self.bootstrap_seed,
                "bootstrap_confidence": self.bootstrap_confidence,
                "bootstrap_minimum_valid": self.bootstrap_minimum_valid,
                "minimum_group_samples": self.minimum_group_samples,
                "minimum_group_patients": self.minimum_group_patients,
                "ece_bins": self.ece_bins,
                "paired_bootstrap_seed_strategy": "base_plus_model_seed",
            },
            "retuning_allowed": False,
        }


@dataclass(frozen=True, slots=True)
class FinalBatchPlan:
    """Exact source and destination identity for the one-time batch."""

    protocol_hash: str
    refit_bundle_sha256: str
    calibration_bundle_sha256: str
    manifest_sha256: str
    normalization_sha256: str
    label_order: tuple[str, ...]
    final_evaluation_spec: Mapping[str, object]
    opening_marker_path: Path
    settings: FinalBatchSettings
    members: tuple[Mapping[str, object], ...]
    batch_sha256: str

    def to_payload(self, *, include_batch_hash: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": FINAL_BATCH_PLAN_SCHEMA_VERSION,
            "artifact_type": FINAL_BATCH_PLAN_TYPE,
            "protocol_hash": self.protocol_hash,
            "refit_bundle_sha256": self.refit_bundle_sha256,
            "calibration_bundle_sha256": self.calibration_bundle_sha256,
            "manifest_sha256": self.manifest_sha256,
            "normalization_sha256": self.normalization_sha256,
            "label_order": list(self.label_order),
            "final_evaluation_spec": dict(self.final_evaluation_spec),
            "opening_marker_path": str(self.opening_marker_path),
            "settings": self.settings.to_payload(),
            "members": [dict(member) for member in self.members],
            "member_count": 6,
            "final_folds": list(FINAL_TEST_FOLDS),
            "retuning_allowed": False,
        }
        if include_batch_hash:
            payload["batch_sha256"] = self.batch_sha256
        return payload


@dataclass(frozen=True, slots=True)
class FinalOpeningLedger:
    """Tamper-evident mutable journal proving when fold 10 was opened."""

    plan: FinalBatchPlan
    purpose: str
    operator: str
    confirmation_sha256: str
    opening_intent_sha256: str
    state: str
    members: Mapping[str, Mapping[str, object]]
    outputs: Mapping[str, object]
    events: tuple[Mapping[str, object], ...]
    created_at_utc: str
    updated_at_utc: str
    ledger_sha256: str | None

    def to_payload(self, *, include_integrity: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": FINAL_LEDGER_SCHEMA_VERSION,
            "artifact_type": FINAL_LEDGER_TYPE,
            "plan": self.plan.to_payload(),
            "opening": {
                "purpose": self.purpose,
                "operator": self.operator,
                "confirmation_sha256": self.confirmation_sha256,
                "opening_intent_sha256": self.opening_intent_sha256,
                "created_at_utc": self.created_at_utc,
                "ledger_precedes_fold10_access": True,
            },
            "state": self.state,
            "members": {
                key: dict(value) for key, value in sorted(self.members.items())
            },
            "outputs": dict(self.outputs),
            "events": [dict(event) for event in self.events],
            "updated_at_utc": self.updated_at_utc,
        }
        if include_integrity and self.ledger_sha256 is not None:
            payload["ledger_sha256"] = self.ledger_sha256
        return payload


@dataclass(frozen=True, slots=True)
class FinalBatchResult:
    ledger_path: Path
    batch_summary_path: Path
    paired_manifest_path: Path
    architecture_report_paths: Mapping[str, Path]
    batch_sha256: str


def build_final_batch_plan(
    refit_bundle: RefitBundle,
    calibration_bundle: CalibrationBundle,
    settings: FinalBatchSettings,
) -> FinalBatchPlan:
    """Build the only batch identity accepted by the opening ledger."""

    if sha256_file(settings.subgroup_path) != settings.subgroup_sha256:
        raise ReleaseIntegrityError("subgroup artifact changed before final plan freeze")
    if refit_bundle.artifact_sha256 is None:
        raise ReleaseStateError("final batch requires an integrity-bound refit bundle")
    if calibration_bundle.artifact_sha256 is None:
        raise ReleaseStateError(
            "final batch requires an integrity-bound calibration bundle"
        )
    if calibration_bundle.refit_bundle_sha256 != refit_bundle.artifact_sha256:
        raise ReleaseIntegrityError("calibration bundle does not bind this refit bundle")
    common = {
        "protocol_hash": (
            refit_bundle.protocol_hash,
            calibration_bundle.protocol_hash,
        ),
        "manifest_sha256": (
            refit_bundle.manifest_sha256,
            calibration_bundle.manifest_sha256,
        ),
        "normalization_sha256": (
            refit_bundle.normalization_sha256,
            calibration_bundle.normalization_sha256,
        ),
        "label_order": (refit_bundle.label_order, calibration_bundle.label_order),
    }
    mismatches = [name for name, values in common.items() if values[0] != values[1]]
    if mismatches:
        raise ReleaseIntegrityError(
            "refit/calibration bundle mismatch: " + ", ".join(mismatches)
        )
    final_spec_binding, frozen_device, opening_registry = (
        _final_evaluation_spec_context(
            calibration_bundle,
            refit_bundle=refit_bundle,
            settings=settings,
        )
    )
    refits = {member.member_id: member for member in refit_bundle.members}
    calibrations = {
        member.member_id: member for member in calibration_bundle.members
    }
    expected_ids = {
        f"{architecture}-seed{seed}"
        for architecture in EXPECTED_ARCHITECTURES
        for seed in EXPECTED_SEEDS
    }
    if set(refits) != expected_ids or set(calibrations) != expected_ids:
        raise ReleaseGateError("final batch requires the exact six-member release grid")
    opening_marker_path = (
        opening_registry
        / (
            _hash_string(
                final_spec_binding["artifact_sha256"],
                "final evaluation spec artifact_sha256",
            ).removeprefix("sha256:")
            + ".opening.json"
        )
    )
    members: list[Mapping[str, object]] = []
    for architecture in EXPECTED_ARCHITECTURES:
        for seed in EXPECTED_SEEDS:
            member_id = f"{architecture}-seed{seed}"
            refit = refits[member_id]
            calibration = calibrations[member_id]
            _validate_pair(refit, calibration)
            inference = _resolved_member_inference(
                refit,
                settings,
                frozen_device=frozen_device,
            )
            members.append(
                {
                    "member_id": member_id,
                    "architecture": architecture,
                    "seed": seed,
                    "model_name": calibration.model_name,
                    "refit_lineage_sha256": refit.lineage_sha256,
                    "checkpoint_sha256": refit.final_checkpoint_sha256,
                    "resolved_config_hash": refit.resolved_config_hash,
                    "calibration_decision_sha256": (
                        calibration.decision_artifact_sha256
                    ),
                    "fold9_prediction_sha256": (
                        calibration.prediction_artifact_sha256
                    ),
                    "inference": inference,
                    "final_prediction_path": str(
                        settings.output_directory / f"{member_id}.fold10.npz"
                    ),
                    "final_report_path": str(
                        settings.output_directory / f"{member_id}.final-report.json"
                    ),
                }
            )
    unhashed: dict[str, object] = {
        "schema_version": FINAL_BATCH_PLAN_SCHEMA_VERSION,
        "artifact_type": FINAL_BATCH_PLAN_TYPE,
        "protocol_hash": refit_bundle.protocol_hash,
        "refit_bundle_sha256": refit_bundle.artifact_sha256,
        "calibration_bundle_sha256": calibration_bundle.artifact_sha256,
        "manifest_sha256": refit_bundle.manifest_sha256,
        "normalization_sha256": refit_bundle.normalization_sha256,
        "label_order": list(refit_bundle.label_order),
        "final_evaluation_spec": dict(final_spec_binding),
        "opening_marker_path": str(opening_marker_path),
        "settings": settings.to_payload(),
        "members": [dict(member) for member in members],
        "member_count": 6,
        "final_folds": list(FINAL_TEST_FOLDS),
        "retuning_allowed": False,
    }
    return FinalBatchPlan(
        protocol_hash=refit_bundle.protocol_hash,
        refit_bundle_sha256=refit_bundle.artifact_sha256,
        calibration_bundle_sha256=calibration_bundle.artifact_sha256,
        manifest_sha256=refit_bundle.manifest_sha256,
        normalization_sha256=refit_bundle.normalization_sha256,
        label_order=refit_bundle.label_order,
        final_evaluation_spec=final_spec_binding,
        opening_marker_path=opening_marker_path,
        settings=settings,
        members=tuple(members),
        batch_sha256=canonical_sha256(unhashed),
    )


def _final_evaluation_spec_context(
    calibration_bundle: CalibrationBundle,
    *,
    refit_bundle: RefitBundle,
    settings: FinalBatchSettings,
) -> tuple[Mapping[str, object], str, Path]:
    """Validate the preregistration transitively bound by calibration."""

    if calibration_bundle.stage_provenance is None:
        raise ReleaseStateError(
            "final batch requires calibration provenance bound to the "
            "final-evaluation specification"
        )
    provenance = _mapping(
        calibration_bundle.stage_provenance, "calibration stage_provenance"
    )
    binding = _mapping(
        provenance.get("final_evaluation_spec"),
        "stage_provenance.final_evaluation_spec",
    )
    _exact_keys(
        binding,
        {"path", "file_sha256", "artifact_sha256"},
        "stage_provenance.final_evaluation_spec",
    )
    spec_path = Path(_string(binding["path"], "final evaluation spec path")).resolve()
    expected_file_hash = _hash_string(
        binding["file_sha256"], "final evaluation spec file_sha256"
    )
    if "sha256:" + sha256_file(spec_path) != expected_file_hash:
        raise ReleaseIntegrityError("bound final-evaluation specification file changed")
    payload = read_json_mapping(spec_path, context="final-evaluation specification")
    artifact_hash = _hash_string(
        payload.get("artifact_sha256"), "final evaluation spec artifact_sha256"
    )
    if artifact_hash != _hash_string(
        binding["artifact_sha256"], "bound final evaluation spec artifact_sha256"
    ):
        raise ReleaseIntegrityError("final-evaluation specification binding changed")
    unhashed = dict(payload)
    del unhashed["artifact_sha256"]
    if canonical_sha256(unhashed) != artifact_hash:
        raise ReleaseIntegrityError("final-evaluation specification hash mismatch")
    verified_payload = _load_verified_final_evaluation_spec(
        spec_path, protocol_hash=refit_bundle.protocol_hash
    )
    if dict(verified_payload) != dict(payload):
        raise ReleaseIntegrityError(
            "verified final-evaluation specification payload changed"
        )

    spec_refit = _mapping(payload.get("refit_bundle"), "spec refit_bundle")
    if spec_refit.get("artifact_sha256") != refit_bundle.artifact_sha256 or (
        spec_refit.get("manifest_sha256") != refit_bundle.manifest_sha256
    ):
        raise ReleaseIntegrityError("final-evaluation spec binds another refit root")
    subgroup = _mapping(payload.get("subgroup_artifact"), "spec subgroup_artifact")
    if Path(_string(subgroup.get("path"), "spec subgroup path")).resolve() != (
        settings.subgroup_path.resolve()
    ):
        raise ReleaseIntegrityError("final settings use a different subgroup artifact")
    if subgroup.get("file_sha256") != "sha256:" + settings.subgroup_sha256:
        raise ReleaseIntegrityError("final subgroup file differs from preregistration")

    evaluation = _mapping(payload.get("final_evaluation"), "spec final_evaluation")
    frozen_evaluation: dict[str, object] = {
        "final_folds": list(FINAL_TEST_FOLDS),
        "patient_resampling": "patient_cluster_percentile_bootstrap",
        "bootstrap_resamples": settings.bootstrap_resamples,
        "bootstrap_base_seed": settings.bootstrap_seed,
        "bootstrap_confidence": settings.bootstrap_confidence,
        "bootstrap_minimum_valid": settings.bootstrap_minimum_valid,
        "bootstrap_seed_strategy": "base_plus_model_seed",
        "ece_bins": settings.ece_bins,
        "minimum_group_samples": settings.minimum_group_samples,
        "minimum_group_patients": settings.minimum_group_patients,
        "retuning_allowed": False,
    }
    if dict(evaluation) != frozen_evaluation:
        raise ReleaseStateError(
            "final-batch scientific settings differ from the preregistered specification"
        )
    if settings.batch_size is not None or settings.num_workers is not None:
        raise ReleaseStateError(
            "final inference must inherit each frozen refit loader; batch/worker "
            "overrides are forbidden"
        )
    runtime = _mapping(payload.get("runtime_envelope"), "spec runtime_envelope")
    project_root = Path(
        _string(runtime.get("project_root"), "spec runtime project_root")
    ).resolve()
    hardware = _mapping(runtime.get("hardware"), "spec runtime hardware")
    requested_device = _string(
        hardware.get("requested_device"), "spec requested_device"
    ).casefold()
    resolved_device = _string(
        hardware.get("resolved_device"), "spec resolved_device"
    ).casefold()
    if settings.device.strip().casefold() not in {requested_device, resolved_device}:
        raise ReleaseStateError(
            "final inference device differs from the preregistered runtime"
        )
    if settings.device.strip().casefold() == "auto":
        raise ReleaseStateError("device=auto is forbidden for final evaluation")
    if settings.bf16 is not True or hardware.get("bf16_supported") is not True:
        raise ReleaseStateError("BF16 is required by the final-evaluation freeze")
    opening_registry = project_root / "runs" / "release" / ".final-test-openings"
    return dict(binding), resolved_device, opening_registry


def _load_verified_final_evaluation_spec(
    path: Path, *, protocol_hash: str
) -> Mapping[str, object]:
    protocol = ExperimentProtocol.canonical()
    if protocol.protocol_hash != protocol_hash:
        raise ReleaseIntegrityError(
            "final batch protocol is not the canonical preregistered protocol"
        )
    try:
        spec = load_final_evaluation_spec(
            path,
            protocol=protocol,
            verify_sources=True,
            verify_runtime=True,
        )
    except (OSError, RuntimeError, ValueError, FinalEvaluationSpecError) as error:
        raise ReleaseIntegrityError(
            f"final-evaluation specification failed full verification: {error}"
        ) from error
    return spec.payload


def _verify_plan_preregistration(
    plan: FinalBatchPlan, *, protocol: ExperimentProtocol
) -> None:
    if plan.protocol_hash != protocol.protocol_hash:
        raise ReleaseIntegrityError("final plan protocol differs from active protocol")
    binding = plan.final_evaluation_spec
    _exact_keys(
        binding,
        {"path", "file_sha256", "artifact_sha256"},
        "final plan evaluation specification",
    )
    spec_path = Path(_string(binding["path"], "final spec path")).resolve()
    if "sha256:" + sha256_file(spec_path) != _hash_string(
        binding["file_sha256"], "final spec file_sha256"
    ):
        raise ReleaseIntegrityError("final plan specification file changed")
    payload = _load_verified_final_evaluation_spec(
        spec_path, protocol_hash=protocol.protocol_hash
    )
    if payload.get("artifact_sha256") != binding["artifact_sha256"]:
        raise ReleaseIntegrityError("final plan specification artifact changed")
    spec_refit = _mapping(payload.get("refit_bundle"), "spec refit_bundle")
    if spec_refit.get("artifact_sha256") != plan.refit_bundle_sha256 or (
        spec_refit.get("manifest_sha256") != plan.manifest_sha256
    ):
        raise ReleaseIntegrityError("final plan refit root differs from specification")
    subgroup = _mapping(payload.get("subgroup_artifact"), "spec subgroup artifact")
    if Path(_string(subgroup.get("path"), "spec subgroup path")).resolve() != (
        plan.settings.subgroup_path.resolve()
    ) or subgroup.get("file_sha256") != "sha256:" + plan.settings.subgroup_sha256:
        raise ReleaseIntegrityError("final plan subgroup differs from specification")
    evaluation = _mapping(payload.get("final_evaluation"), "spec final evaluation")
    expected_evaluation: dict[str, object] = {
        "final_folds": list(FINAL_TEST_FOLDS),
        "patient_resampling": "patient_cluster_percentile_bootstrap",
        "bootstrap_resamples": plan.settings.bootstrap_resamples,
        "bootstrap_base_seed": plan.settings.bootstrap_seed,
        "bootstrap_confidence": plan.settings.bootstrap_confidence,
        "bootstrap_minimum_valid": plan.settings.bootstrap_minimum_valid,
        "bootstrap_seed_strategy": "base_plus_model_seed",
        "ece_bins": plan.settings.ece_bins,
        "minimum_group_samples": plan.settings.minimum_group_samples,
        "minimum_group_patients": plan.settings.minimum_group_patients,
        "retuning_allowed": False,
    }
    if dict(evaluation) != expected_evaluation:
        raise ReleaseIntegrityError("final plan settings differ from specification")
    runtime = _mapping(payload.get("runtime_envelope"), "spec runtime envelope")
    hardware = _mapping(runtime.get("hardware"), "spec runtime hardware")
    resolved_device = _string(
        hardware.get("resolved_device"), "spec resolved device"
    ).casefold()
    if plan.settings.batch_size is not None or plan.settings.num_workers is not None:
        raise ReleaseIntegrityError("final plan contains forbidden loader overrides")
    for member in plan.members:
        inference = _mapping(member.get("inference"), "planned member inference")
        if inference.get("device") != resolved_device or inference.get("bf16") is not True:
            raise ReleaseIntegrityError(
                "planned inference runtime differs from frozen specification"
            )
    project_root = Path(
        _string(runtime.get("project_root"), "spec project_root")
    ).resolve()
    expected_marker = (
        project_root
        / "runs"
        / "release"
        / ".final-test-openings"
        / (
            _hash_string(binding["artifact_sha256"], "final spec artifact_sha256")
            .removeprefix("sha256:")
            + ".opening.json"
        )
    )
    if plan.opening_marker_path.resolve() != expected_marker:
        raise ReleaseIntegrityError("final plan opening registry path is not canonical")


def _resolved_member_inference(
    refit: RefitMember,
    settings: FinalBatchSettings,
    *,
    frozen_device: str | None,
) -> dict[str, object]:
    batch_size = settings.batch_size
    num_workers = settings.num_workers
    if batch_size is None or num_workers is None:
        wrapper = read_json_mapping(
            refit.resolved_config_path,
            context=f"resolved config for {refit.member_id}",
        )
        config = _mapping(wrapper.get("config"), "resolved config")
        if wrapper.get("config_hash") != refit.resolved_config_hash or (
            canonical_sha256(config) != refit.resolved_config_hash
        ):
            raise ReleaseIntegrityError("resolved config hash differs from refit bundle")
        loader = _mapping(config.get("loader"), "resolved config loader")
        if batch_size is None:
            batch_size = _positive_int(
                loader.get("batch_size"), "resolved batch_size", minimum=1
            )
        if num_workers is None:
            num_workers = _positive_int(
                loader.get("num_workers"), "resolved num_workers", minimum=0
            )
    resolved: dict[str, object] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "device": frozen_device or settings.device.strip().casefold(),
        "bf16": settings.bf16,
    }
    _validate_inference_mapping(resolved)
    return resolved


def _validate_inference_mapping(inference: Mapping[str, object]) -> None:
    _exact_keys(
        inference,
        {"batch_size", "num_workers", "device", "bf16"},
        "member inference",
    )
    _positive_int(inference["batch_size"], "inference batch_size", minimum=1)
    _positive_int(inference["num_workers"], "inference num_workers", minimum=0)
    _string(inference["device"], "inference device")
    _boolean(inference["bf16"], "inference bf16")


def _existing_planned_final_artifacts(plan: FinalBatchPlan) -> tuple[Path, ...]:
    candidates: list[Path] = []
    for member in plan.members:
        prediction = Path(
            _string(member["final_prediction_path"], "final_prediction_path")
        )
        candidates.extend((prediction, prediction.with_suffix(".json")))
        candidates.append(
            Path(_string(member["final_report_path"], "final_report_path"))
        )
    destination = plan.settings.output_directory
    candidates.extend(
        destination / f"{architecture}.architecture-summary.json"
        for architecture in EXPECTED_ARCHITECTURES
    )
    candidates.extend(
        destination / f"paired-seed{seed}.bootstrap.json"
        for seed in EXPECTED_SEEDS
    )
    candidates.extend(
        (
            destination / "paired-patient-bootstrap.manifest.json",
            destination / "final-batch-summary.json",
        )
    )
    return tuple(path for path in candidates if _path_lexists(path))


def _opening_marker_payload(
    *,
    plan: FinalBatchPlan,
    ledger_path: Path,
    created_at_utc: str,
    opening_intent_sha256: str,
) -> dict[str, object]:
    return {
        "schema_version": FINAL_OPENING_MARKER_SCHEMA_VERSION,
        "artifact_type": FINAL_OPENING_MARKER_TYPE,
        "batch_sha256": plan.batch_sha256,
        "refit_bundle_sha256": plan.refit_bundle_sha256,
        "calibration_bundle_sha256": plan.calibration_bundle_sha256,
        "ledger_path": str(ledger_path.resolve()),
        "output_directory": str(plan.settings.output_directory.resolve()),
        "created_at_utc": created_at_utc,
        "opening_intent_sha256": opening_intent_sha256,
        "marker_precedes_fold10_access": True,
    }


def _opening_intent_sha256(
    *,
    plan: FinalBatchPlan,
    ledger_path: Path,
    purpose: str,
    operator: str,
    confirmation_sha256: str,
    created_at_utc: str,
) -> str:
    return canonical_sha256(
        {
            "batch_sha256": plan.batch_sha256,
            "ledger_path": str(ledger_path.resolve()),
            "purpose": purpose,
            "operator": operator,
            "confirmation_sha256": confirmation_sha256,
            "created_at_utc": created_at_utc,
            "ledger_precedes_fold10_access": True,
        }
    )


def _create_canonical_opening_marker(
    path: Path,
    *,
    plan: FinalBatchPlan,
    ledger_path: Path,
    created_at_utc: str,
    opening_intent_sha256: str,
) -> None:
    if path.resolve() != plan.opening_marker_path.resolve():
        raise ReleaseIntegrityError("opening marker path differs from final batch plan")
    payload = _opening_marker_payload(
        plan=plan,
        ledger_path=ledger_path,
        created_at_utc=created_at_utc,
        opening_intent_sha256=opening_intent_sha256,
    )
    committed = dict(payload)
    committed["marker_sha256"] = canonical_sha256(payload)
    try:
        _atomic_json(path, committed, replace=False)
    except FileExistsError as error:
        raise ReleaseStateError(
            "this frozen calibration/refit release has already opened fold 10"
        ) from error


def _verify_canonical_opening_marker(
    path: Path,
    *,
    plan: FinalBatchPlan,
    ledger_path: Path,
    created_at_utc: str,
    opening_intent_sha256: str,
) -> None:
    payload = read_json_mapping(path, context="canonical final-test opening marker")
    required = {
        "schema_version",
        "artifact_type",
        "batch_sha256",
        "refit_bundle_sha256",
        "calibration_bundle_sha256",
        "ledger_path",
        "output_directory",
        "created_at_utc",
        "opening_intent_sha256",
        "marker_precedes_fold10_access",
        "marker_sha256",
    }
    _exact_keys(payload, required, "canonical final-test opening marker")
    stored = _hash_string(payload["marker_sha256"], "marker_sha256")
    unhashed = dict(payload)
    del unhashed["marker_sha256"]
    if canonical_sha256(unhashed) != stored:
        raise ReleaseIntegrityError("canonical opening marker hash mismatch")
    expected = {
        "schema_version": FINAL_OPENING_MARKER_SCHEMA_VERSION,
        "artifact_type": FINAL_OPENING_MARKER_TYPE,
        "batch_sha256": plan.batch_sha256,
        "refit_bundle_sha256": plan.refit_bundle_sha256,
        "calibration_bundle_sha256": plan.calibration_bundle_sha256,
        "ledger_path": str(ledger_path.resolve()),
        "output_directory": str(plan.settings.output_directory.resolve()),
        "created_at_utc": created_at_utc,
        "opening_intent_sha256": opening_intent_sha256,
        "marker_precedes_fold10_access": True,
    }
    drift = [
        field for field, value in expected.items() if payload.get(field) != value
    ]
    if drift:
        raise ReleaseIntegrityError(
            "canonical opening marker differs from final batch: "
            + ", ".join(drift)
        )
    _timestamp(_string(payload["created_at_utc"], "marker created_at_utc"))


def _ensure_canonical_opening_marker(
    path: Path,
    *,
    plan: FinalBatchPlan,
    ledger_path: Path,
    created_at_utc: str,
    opening_intent_sha256: str,
) -> None:
    """Create, or adopt after a race, only the exact pending-opening marker."""

    if _path_lexists(path):
        _verify_canonical_opening_marker(
            path,
            plan=plan,
            ledger_path=ledger_path,
            created_at_utc=created_at_utc,
            opening_intent_sha256=opening_intent_sha256,
        )
        return
    try:
        _create_canonical_opening_marker(
            path,
            plan=plan,
            ledger_path=ledger_path,
            created_at_utc=created_at_utc,
            opening_intent_sha256=opening_intent_sha256,
        )
    except ReleaseStateError:
        # Another exact resume may have won the non-overwriting marker race.
        if not _path_lexists(path):
            raise
        _verify_canonical_opening_marker(
            path,
            plan=plan,
            ledger_path=ledger_path,
            created_at_utc=created_at_utc,
            opening_intent_sha256=opening_intent_sha256,
        )


def canonical_final_ledger_path(plan: FinalBatchPlan) -> Path:
    """Return the one authoritative ledger path for a frozen release."""

    marker = plan.opening_marker_path.resolve()
    suffix = ".opening.json"
    if not marker.name.endswith(suffix):
        raise ReleaseIntegrityError("canonical opening marker has an invalid filename")
    return marker.with_name(marker.name[: -len(suffix)] + ".opening-ledger.json")


def _canonical_ledger_destination(
    path: str | Path, plan: FinalBatchPlan
) -> Path:
    destination = Path(path).resolve()
    canonical = canonical_final_ledger_path(plan)
    if _path_identity(destination) != _path_identity(canonical):
        raise ReleaseStateError(
            "final-test ledger path must equal the canonical release ledger: "
            f"{canonical}"
        )
    protected = {
        _path_identity(plan.opening_marker_path),
        _path_identity(
            plan.opening_marker_path.with_name(
                plan.opening_marker_path.name + ".writer.lock"
            )
        ),
        *(_path_identity(item) for item in _planned_artifact_paths(plan)),
    }
    ledger_paths = {
        _path_identity(destination),
        _path_identity(destination.with_name(destination.name + ".writer.lock")),
    }
    if protected.intersection(ledger_paths):
        raise ReleaseIntegrityError(
            "canonical final-test ledger or its lock aliases a protected output"
        )
    return destination


def _planned_artifact_paths(plan: FinalBatchPlan) -> tuple[Path, ...]:
    candidates: list[Path] = []
    for member in plan.members:
        prediction = Path(
            _string(member["final_prediction_path"], "final_prediction_path")
        )
        candidates.extend((prediction, prediction.with_suffix(".json")))
        candidates.append(
            Path(_string(member["final_report_path"], "final_report_path"))
        )
    destination = plan.settings.output_directory
    candidates.extend(
        destination / f"{architecture}.architecture-summary.json"
        for architecture in EXPECTED_ARCHITECTURES
    )
    candidates.extend(
        destination / f"paired-seed{seed}.bootstrap.json"
        for seed in EXPECTED_SEEDS
    )
    candidates.extend(
        (
            destination / "paired-patient-bootstrap.manifest.json",
            destination / "final-batch-summary.json",
        )
    )
    return tuple(path.resolve() for path in candidates)


def create_final_opening_ledger(
    path: str | Path,
    plan: FinalBatchPlan,
    *,
    purpose: str,
    operator: str,
    confirmation: str,
    created_at_utc: str | None = None,
) -> FinalOpeningLedger:
    """Commit pending ledger, canonical marker, then the open state in order."""

    _verify_plan_preregistration(plan, protocol=ExperimentProtocol.canonical())
    destination = _canonical_ledger_destination(path, plan)
    with _ledger_writer_lock(plan.opening_marker_path):
        return _create_final_opening_ledger_unlocked(
            destination,
            plan,
            purpose=purpose,
            operator=operator,
            confirmation=confirmation,
            created_at_utc=created_at_utc,
        )


def _create_final_opening_ledger_unlocked(
    destination: Path,
    plan: FinalBatchPlan,
    *,
    purpose: str,
    operator: str,
    confirmation: str,
    created_at_utc: str | None,
) -> FinalOpeningLedger:
    if _path_lexists(destination):
        raise ReleaseStateError(
            "fold-10 opening ledger already exists; use exact-batch resume"
        )
    if _path_lexists(plan.opening_marker_path):
        raise ReleaseStateError(
            "this frozen calibration/refit release has already opened fold 10"
        )
    normalized_purpose = purpose.strip()
    normalized_operator = operator.strip()
    if not normalized_purpose or not normalized_operator:
        raise ReleaseGateError("opening purpose and operator must be non-empty")
    if confirmation != FINAL_TEST_CONFIRMATION:
        raise ReleaseGateError("final-test confirmation is not exact")
    collisions = _existing_planned_final_artifacts(plan)
    if collisions:
        raise ReleaseStateError(
            "final opening requires no pre-existing planned fold-10 artifacts: "
            + ", ".join(str(item) for item in collisions)
        )
    timestamp = _timestamp(created_at_utc)
    member_states = {
        _string(member["member_id"], "member_id"): {
            "state": "planned",
            "final_prediction_path": member["final_prediction_path"],
            "final_prediction_artifact_sha256": None,
            "final_prediction_file_sha256": None,
            "final_prediction_sidecar_sha256": None,
            "final_report_path": member["final_report_path"],
            "final_report_sha256": None,
        }
        for member in plan.members
    }
    confirmation_sha256 = hashlib.sha256(
        confirmation.encode("utf-8")
    ).hexdigest()
    opening_intent_sha256 = _opening_intent_sha256(
        plan=plan,
        ledger_path=destination,
        purpose=normalized_purpose,
        operator=normalized_operator,
        confirmation_sha256=confirmation_sha256,
        created_at_utc=timestamp,
    )
    ledger = FinalOpeningLedger(
        plan=plan,
        purpose=normalized_purpose,
        operator=normalized_operator,
        confirmation_sha256=confirmation_sha256,
        opening_intent_sha256=opening_intent_sha256,
        state="opening_pending",
        members=member_states,
        outputs={},
        events=(
            {
                "sequence": 0,
                "timestamp_utc": timestamp,
                "event": "ledger_created_before_fold10_access",
                "batch_sha256": plan.batch_sha256,
                "opening_state": "opening_pending",
                "opening_intent_sha256": opening_intent_sha256,
            },
        ),
        created_at_utc=timestamp,
        updated_at_utc=timestamp,
        ledger_sha256=None,
    )
    try:
        pending = _commit_ledger(destination, ledger, replace=False)
    except FileExistsError as error:
        raise ReleaseStateError(
            "fold-10 opening ledger already exists; use exact-batch resume"
        ) from error
    _ensure_canonical_opening_marker(
        plan.opening_marker_path,
        plan=plan,
        ledger_path=destination,
        created_at_utc=timestamp,
        opening_intent_sha256=opening_intent_sha256,
    )
    return _transition_pending_opening_to_open(destination, pending)


def _transition_pending_opening_to_open(
    path: Path, ledger: FinalOpeningLedger
) -> FinalOpeningLedger:
    if ledger.state != "opening_pending":
        raise ReleaseStateError("only a pending opening may transition to open")
    return _append_event(
        path,
        ledger,
        members=ledger.members,
        state="open",
        event={
            "event": "canonical_opening_marker_committed",
            "marker_path": str(ledger.plan.opening_marker_path.resolve()),
            "marker_precedes_fold10_access": True,
            "opening_intent_sha256": ledger.opening_intent_sha256,
        },
    )


def load_final_opening_ledger(
    path: str | Path, *, protocol: ExperimentProtocol
) -> FinalOpeningLedger:
    """Load and fully verify the persistent fold-10 journal."""

    payload = read_json_mapping(path, context="final-test opening ledger")
    required = {
        "schema_version",
        "artifact_type",
        "plan",
        "opening",
        "state",
        "members",
        "outputs",
        "events",
        "updated_at_utc",
        "ledger_sha256",
    }
    _exact_keys(payload, required, "final-test opening ledger")
    if payload["schema_version"] != FINAL_LEDGER_SCHEMA_VERSION:
        raise ReleaseIntegrityError("unsupported opening ledger schema")
    if payload["artifact_type"] != FINAL_LEDGER_TYPE:
        raise ReleaseIntegrityError("unexpected opening ledger type")
    stored = _hash_string(payload["ledger_sha256"], "ledger_sha256")
    unhashed = dict(payload)
    del unhashed["ledger_sha256"]
    if canonical_sha256(unhashed) != stored:
        raise ReleaseIntegrityError("opening ledger hash mismatch")
    plan = _parse_plan(payload["plan"])
    if plan.protocol_hash != protocol.protocol_hash:
        raise ReleaseIntegrityError("opening ledger protocol mismatch")
    _verify_plan_preregistration(plan, protocol=protocol)
    opening = _mapping(payload["opening"], "opening")
    _exact_keys(
        opening,
        {
            "purpose",
            "operator",
            "confirmation_sha256",
            "opening_intent_sha256",
            "created_at_utc",
            "ledger_precedes_fold10_access",
        },
        "opening",
    )
    if opening["ledger_precedes_fold10_access"] is not True:
        raise ReleaseIntegrityError("ledger does not assert pre-access creation")
    created_at_utc = _timestamp(
        _string(opening["created_at_utc"], "opening.created_at_utc")
    )
    purpose = _string(opening["purpose"], "opening.purpose")
    operator = _string(opening["operator"], "opening.operator")
    confirmation_sha256 = _raw_hash_string(
        opening["confirmation_sha256"], "confirmation_sha256"
    )
    opening_intent_sha256 = _hash_string(
        opening["opening_intent_sha256"], "opening_intent_sha256"
    )
    expected_intent = _opening_intent_sha256(
        plan=plan,
        ledger_path=Path(path),
        purpose=purpose,
        operator=operator,
        confirmation_sha256=confirmation_sha256,
        created_at_utc=created_at_utc,
    )
    if opening_intent_sha256 != expected_intent:
        raise ReleaseIntegrityError("opening intent differs from immutable ledger fields")
    state_value = _string(payload["state"], "state")
    if state_value not in {"opening_pending", "open", "complete"}:
        raise ReleaseIntegrityError("opening ledger has an unsupported state")
    if _path_lexists(plan.opening_marker_path):
        _verify_canonical_opening_marker(
            plan.opening_marker_path,
            plan=plan,
            ledger_path=Path(path),
            created_at_utc=created_at_utc,
            opening_intent_sha256=opening_intent_sha256,
        )
    elif state_value != "opening_pending":
        raise ReleaseIntegrityError("open ledger is missing its canonical marker")
    member_states = _mapping(payload["members"], "members")
    expected_ids = {_string(item["member_id"], "member_id") for item in plan.members}
    if set(member_states) != expected_ids:
        raise ReleaseIntegrityError("opening ledger member set differs from plan")
    parsed_states: dict[str, Mapping[str, object]] = {}
    for member_id, value in member_states.items():
        state = _mapping(value, f"members.{member_id}")
        _validate_member_state(state, member_id=member_id, plan=plan)
        parsed_states[member_id] = state
    events = tuple(
        _mapping(item, "ledger event") for item in _sequence(payload["events"], "events")
    )
    for index, event in enumerate(events):
        if event.get("sequence") != index:
            raise ReleaseIntegrityError("ledger event sequence is not contiguous")
    if state_value == "opening_pending":
        if any(state["state"] != "planned" for state in parsed_states.values()):
            raise ReleaseIntegrityError("pending opening contains member progress")
        if _mapping(payload["outputs"], "outputs"):
            raise ReleaseIntegrityError("pending opening contains final outputs")
    return FinalOpeningLedger(
        plan=plan,
        purpose=purpose,
        operator=operator,
        confirmation_sha256=confirmation_sha256,
        opening_intent_sha256=opening_intent_sha256,
        state=state_value,
        members=parsed_states,
        outputs=_mapping(payload["outputs"], "outputs"),
        events=events,
        created_at_utc=created_at_utc,
        updated_at_utc=_timestamp(
            _string(payload["updated_at_utc"], "updated_at_utc")
        ),
        ledger_sha256=stored,
    )


def open_or_resume_final_batch(
    ledger_path: str | Path,
    plan: FinalBatchPlan,
    *,
    protocol: ExperimentProtocol,
    purpose: str,
    operator: str,
    confirmation: str,
    resume: bool,
    created_at_utc: str | None = None,
) -> FinalOpeningLedger:
    """Create once, or resume only the byte-identical scientific batch plan."""

    _verify_plan_preregistration(plan, protocol=protocol)
    destination = _canonical_ledger_destination(ledger_path, plan)
    with _ledger_writer_lock(plan.opening_marker_path):
        return _open_or_resume_final_batch_unlocked(
            destination,
            plan,
            protocol=protocol,
            purpose=purpose,
            operator=operator,
            confirmation=confirmation,
            resume=resume,
            created_at_utc=created_at_utc,
        )


def _open_or_resume_final_batch_unlocked(
    destination: Path,
    plan: FinalBatchPlan,
    *,
    protocol: ExperimentProtocol,
    purpose: str,
    operator: str,
    confirmation: str,
    resume: bool,
    created_at_utc: str | None,
) -> FinalOpeningLedger:
    if not resume:
        return _create_final_opening_ledger_unlocked(
            destination,
            plan,
            purpose=purpose,
            operator=operator,
            confirmation=confirmation,
            created_at_utc=created_at_utc,
        )
    ledger = load_final_opening_ledger(destination, protocol=protocol)
    if ledger.plan.batch_sha256 != plan.batch_sha256:
        raise ReleaseStateError(
            "resume rejected: requested final batch differs from persisted opening"
        )
    if ledger.plan.to_payload() != plan.to_payload():
        raise ReleaseIntegrityError("resume plan content differs despite batch hash")
    if ledger.purpose != purpose.strip() or ledger.operator != operator.strip():
        raise ReleaseStateError("resume purpose/operator must match the opening ledger")
    expected_confirmation = hashlib.sha256(confirmation.encode("utf-8")).hexdigest()
    if confirmation != FINAL_TEST_CONFIRMATION or (
        ledger.confirmation_sha256 != expected_confirmation
    ):
        raise ReleaseStateError("resume confirmation does not match the opening ledger")
    if ledger.state == "opening_pending":
        collisions = _existing_planned_final_artifacts(plan)
        if collisions:
            raise ReleaseStateError(
                "pending opening cannot authorize pre-existing fold-10 artifacts: "
                + ", ".join(str(item) for item in collisions)
            )
        _ensure_canonical_opening_marker(
            plan.opening_marker_path,
            plan=plan,
            ledger_path=destination,
            created_at_utc=ledger.created_at_utc,
            opening_intent_sha256=ledger.opening_intent_sha256,
        )
        ledger = _transition_pending_opening_to_open(destination, ledger)
    return ledger


def authorize_ledgered_final_test(
    ledger_path: str | Path,
    plan: FinalBatchPlan,
    *,
    protocol: ExperimentProtocol,
    purpose: str,
    confirmation: str,
) -> FinalTestAccessToken:
    """Issue a token only after the persistent, identical ledger is visible."""

    destination = _canonical_ledger_destination(ledger_path, plan)
    ledger = load_final_opening_ledger(destination, protocol=protocol)
    if ledger.plan.batch_sha256 != plan.batch_sha256:
        raise ReleaseStateError("ledger does not authorize this final batch")
    if ledger.purpose != purpose.strip():
        raise ReleaseStateError("ledger purpose does not authorize this final batch")
    if ledger.state not in {"open", "complete"}:
        raise ReleaseStateError("final-test opening is not durably open")
    if confirmation != FINAL_TEST_CONFIRMATION:
        raise ReleaseStateError("final-test confirmation is not exact")
    return authorize_final_test_access(
        protocol,
        purpose=purpose,
        confirmation=confirmation,
    )


def run_final_batch(
    *,
    refit_bundle_path: str | Path,
    calibration_bundle_path: str | Path,
    settings: FinalBatchSettings,
    ledger_path: str | Path | None = None,
    protocol: ExperimentProtocol,
    purpose: str,
    operator: str,
    confirmation: str,
    resume: bool = False,
    exporter: PredictionExporter = export_checkpoint_predictions,
) -> FinalBatchResult:
    """Run/resume the exact six-member final batch without fitting anything."""

    refit_bundle = load_refit_bundle(
        refit_bundle_path, protocol=protocol, verify_sources=True
    )
    calibration_bundle = load_calibration_bundle(
        calibration_bundle_path, protocol=protocol, verify_sources=True
    )
    # Validate the complete, self-hashed, label-free subgroup artifact before
    # creating the irreversible opening marker or ledger. A malformed subgroup
    # file must never consume the one-time final-test opening.
    try:
        subgroup_artifact = load_subgroup_artifact(
            settings.subgroup_path,
            protocol=protocol,
            expected_manifest_sha256=refit_bundle.manifest_sha256,
            verify_source=True,
        )
    except SubgroupArtifactError as error:
        raise ReleaseIntegrityError(
            f"invalid frozen subgroup artifact: {error}"
        ) from error
    subgroup_ids = np.asarray(subgroup_artifact.ecg_id, dtype=np.int64)
    subgroups = {
        "sex": np.asarray(subgroup_artifact.sex, dtype=object),
        "age_band": np.asarray(subgroup_artifact.age_band, dtype=object),
    }
    plan = build_final_batch_plan(refit_bundle, calibration_bundle, settings)
    requested_ledger = (
        canonical_final_ledger_path(plan) if ledger_path is None else ledger_path
    )
    destination = _canonical_ledger_destination(requested_ledger, plan)
    # The release-scoped lock stays held through every prediction, report, and
    # aggregate commit.  A second process cannot interleave a fresh or resumed
    # execution, even when it supplied a different caller path.
    with _ledger_writer_lock(plan.opening_marker_path):
        return _run_final_batch_locked(
            destination=destination,
            plan=plan,
            refit_bundle=refit_bundle,
            calibration_bundle=calibration_bundle,
            settings=settings,
            protocol=protocol,
            purpose=purpose,
            operator=operator,
            confirmation=confirmation,
            resume=resume,
            exporter=exporter,
            subgroup_ids=subgroup_ids,
            subgroups=subgroups,
        )


def _run_final_batch_locked(
    *,
    destination: Path,
    plan: FinalBatchPlan,
    refit_bundle: RefitBundle,
    calibration_bundle: CalibrationBundle,
    settings: FinalBatchSettings,
    protocol: ExperimentProtocol,
    purpose: str,
    operator: str,
    confirmation: str,
    resume: bool,
    exporter: PredictionExporter,
    subgroup_ids: NDArray[np.integer[Any]],
    subgroups: Mapping[str, NDArray[np.object_]],
) -> FinalBatchResult:
    ledger = _open_or_resume_final_batch_unlocked(
        destination,
        plan,
        protocol=protocol,
        purpose=purpose,
        operator=operator,
        confirmation=confirmation,
        resume=resume,
        created_at_utc=None,
    )
    # The durable ledger exists and has been re-read before this token is issued.
    token = authorize_ledgered_final_test(
        destination,
        plan,
        protocol=protocol,
        purpose=purpose,
        confirmation=confirmation,
    )
    refits = {member.member_id: member for member in refit_bundle.members}
    calibrations = {
        member.member_id: member for member in calibration_bundle.members
    }
    preregistration = _publication_preregistration(plan)
    if ledger.state == "complete":
        return _verify_completed_final_batch(
            destination=destination,
            ledger=ledger,
            refits=refits,
            calibrations=calibrations,
            settings=settings,
            protocol=protocol,
            test_access=token,
            exporter=exporter,
        )
    try:
        for planned in plan.members:
            member_id = _string(planned["member_id"], "member_id")
            state = ledger.members[member_id]
            refit = refits[member_id]
            calibration = calibrations[member_id]
            prediction_path = Path(
                _string(planned["final_prediction_path"], "final_prediction_path")
            )
            report_path = Path(
                _string(planned["final_report_path"], "final_report_path")
            )
            if state["state"] == "planned":
                ledger = _reconcile_planned_orphan_evidence(
                    destination,
                    ledger,
                    refit=refit,
                    prediction_path=prediction_path,
                )
                state = ledger.members[member_id]
            prediction = _export_or_resume_prediction(
                refit,
                calibration,
                prediction_path,
                state=state,
                settings=settings,
                inference=_mapping(planned["inference"], "planned inference"),
                protocol=protocol,
                test_access=token,
                exporter=exporter,
            )
            if state["state"] == "planned":
                ledger = _record_prediction(
                    destination, ledger, member_id, prediction_path, prediction
                )
                state = ledger.members[member_id]
            if state["state"] == "report_saved":
                existing_report = verify_final_report(
                    report_path, protocol=protocol, test_access=token
                )
                _validate_member_report(
                    existing_report,
                    refit=refit,
                    calibration=calibration,
                    prediction=prediction,
                    settings=settings,
                    plan=plan,
                )
                if existing_report["report_sha256"] != state["final_report_sha256"]:
                    raise ReleaseIntegrityError(
                        "resumed final report hash differs from ledger"
                    )
                continue
            if _path_lexists(report_path):
                existing_report = verify_final_report(
                    report_path, protocol=protocol, test_access=token
                )
                _validate_member_report(
                    existing_report,
                    refit=refit,
                    calibration=calibration,
                    prediction=prediction,
                    settings=settings,
                    plan=plan,
                )
                ledger = _record_report(
                    destination,
                    ledger,
                    member_id,
                    _hash_string(existing_report["report_sha256"], "report_sha256"),
                )
                continue
            decisions = load_calibration_decisions(
                calibration.decision_path, protocol=protocol
            )
            report = generate_final_report(
                decisions,
                prediction,
                protocol=protocol,
                test_access=token,
                subgroup_ecg_id=subgroup_ids,
                subgroups=subgroups,
                final_evaluation_spec=plan.final_evaluation_spec,
                protocol_deviations=_mapping(
                    preregistration["protocol_deviations"],
                    "protocol deviations",
                ),
                bootstrap_resamples=settings.bootstrap_resamples,
                bootstrap_seed=settings.bootstrap_seed + calibration.seed,
                bootstrap_confidence=settings.bootstrap_confidence,
                bootstrap_minimum_valid=settings.bootstrap_minimum_valid,
                minimum_group_samples=settings.minimum_group_samples,
                minimum_group_patients=settings.minimum_group_patients,
                ece_bins=settings.ece_bins,
            )
            saved = save_final_report(
                report, report_path, protocol=protocol, test_access=token
            )
            ledger = _record_report(
                destination, ledger, member_id, saved.sha256
            )
        outputs = _build_aggregate_outputs(
            plan,
            calibration_bundle,
            protocol=protocol,
            test_access=token,
        )
        ledger = _complete_ledger(destination, ledger, outputs)
    except Exception as error:
        _record_failure(destination, ledger, error)
        raise
    return _final_batch_result(destination, ledger, outputs)


def _verify_completed_final_batch(
    *,
    destination: Path,
    ledger: FinalOpeningLedger,
    refits: Mapping[str, RefitMember],
    calibrations: Mapping[str, CalibrationMember],
    settings: FinalBatchSettings,
    protocol: ExperimentProtocol,
    test_access: FinalTestAccessToken,
    exporter: PredictionExporter,
) -> FinalBatchResult:
    if any(state["state"] != "report_saved" for state in ledger.members.values()):
        raise ReleaseIntegrityError("complete ledger contains an incomplete member")
    for planned in ledger.plan.members:
        member_id = _string(planned["member_id"], "member_id")
        state = ledger.members[member_id]
        prediction_path = Path(
            _string(planned["final_prediction_path"], "final_prediction_path")
        )
        prediction = _export_or_resume_prediction(
            refits[member_id],
            calibrations[member_id],
            prediction_path,
            state=state,
            settings=settings,
            inference=_mapping(planned["inference"], "planned inference"),
            protocol=protocol,
            test_access=test_access,
            exporter=exporter,
        )
        report_path = Path(
            _string(planned["final_report_path"], "final_report_path")
        )
        report = verify_final_report(
            report_path, protocol=protocol, test_access=test_access
        )
        _validate_member_report(
            report,
            refit=refits[member_id],
            calibration=calibrations[member_id],
            prediction=prediction,
            settings=settings,
            plan=ledger.plan,
        )
        if report["report_sha256"] != state["final_report_sha256"]:
            raise ReleaseIntegrityError("complete ledger report hash mismatch")
    _verify_completed_outputs(ledger)
    return _final_batch_result(destination, ledger, ledger.outputs)


def _verify_completed_outputs(ledger: FinalOpeningLedger) -> None:
    expected_keys = {
        "batch_summary_path",
        "batch_summary_sha256",
        "paired_manifest_path",
        "paired_manifest_sha256",
        *(
            f"architecture_{architecture}_{suffix}"
            for architecture in EXPECTED_ARCHITECTURES
            for suffix in ("path", "sha256")
        ),
    }
    _exact_keys(ledger.outputs, expected_keys, "complete ledger outputs")
    output_directory = ledger.plan.settings.output_directory.resolve()
    preregistration = _publication_preregistration(ledger.plan)
    architecture_bindings: dict[str, dict[str, object]] = {}
    for architecture in EXPECTED_ARCHITECTURES:
        path = Path(
            _string(
                ledger.outputs[f"architecture_{architecture}_path"],
                f"architecture_{architecture}_path",
            )
        ).resolve()
        expected_path = (
            output_directory / f"{architecture}.architecture-summary.json"
        )
        if path != expected_path:
            raise ReleaseIntegrityError("complete architecture output path changed")
        expected_hash = _hash_string(
            ledger.outputs[f"architecture_{architecture}_sha256"],
            f"architecture_{architecture}_sha256",
        )
        payload = _read_verified_output(path, expected_hash)
        if payload.get("batch_sha256") != ledger.plan.batch_sha256 or (
            payload.get("architecture") != architecture
        ) or payload.get("preregistration") != preregistration:
            raise ReleaseIntegrityError("architecture output differs from final plan")
        architecture_bindings[architecture] = {
            "path": str(path),
            "artifact_sha256": expected_hash,
        }

    paired_path = Path(
        _string(ledger.outputs["paired_manifest_path"], "paired_manifest_path")
    ).resolve()
    if paired_path != output_directory / "paired-patient-bootstrap.manifest.json":
        raise ReleaseIntegrityError("paired manifest path changed")
    paired_hash = _hash_string(
        ledger.outputs["paired_manifest_sha256"], "paired_manifest_sha256"
    )
    paired_manifest = _read_verified_output(paired_path, paired_hash)
    if paired_manifest.get("batch_sha256") != ledger.plan.batch_sha256:
        raise ReleaseIntegrityError("paired manifest final plan hash changed")
    if paired_manifest.get("preregistration") != preregistration:
        raise ReleaseIntegrityError("paired manifest preregistration changed")
    entries = _sequence(paired_manifest.get("entries"), "paired manifest entries")
    if len(entries) != len(EXPECTED_SEEDS):
        raise ReleaseIntegrityError("paired manifest must contain three seeds")
    for expected_seed, raw_entry in zip(EXPECTED_SEEDS, entries, strict=True):
        entry = _mapping(raw_entry, "paired manifest entry")
        if entry.get("seed") != expected_seed:
            raise ReleaseIntegrityError("paired manifest seed order changed")
        path = Path(_string(entry.get("path"), "paired report path")).resolve()
        if path != output_directory / f"paired-seed{expected_seed}.bootstrap.json":
            raise ReleaseIntegrityError("paired report path changed")
        report_hash = _hash_string(
            entry.get("artifact_sha256"), "paired report artifact_sha256"
        )
        report = _read_verified_output(path, report_hash)
        if report.get("batch_sha256") != ledger.plan.batch_sha256 or (
            report.get("seed") != expected_seed
        ) or report.get("preregistration") != preregistration:
            raise ReleaseIntegrityError("paired report differs from final plan")

    summary_path = Path(
        _string(ledger.outputs["batch_summary_path"], "batch_summary_path")
    ).resolve()
    if summary_path != output_directory / "final-batch-summary.json":
        raise ReleaseIntegrityError("batch summary path changed")
    summary_hash = _hash_string(
        ledger.outputs["batch_summary_sha256"], "batch_summary_sha256"
    )
    summary = _read_verified_output(summary_path, summary_hash)
    if summary.get("batch_sha256") != ledger.plan.batch_sha256 or (
        summary.get("architecture_reports") != architecture_bindings
    ) or summary.get("preregistration") != preregistration:
        raise ReleaseIntegrityError("batch summary differs from complete outputs")
    if summary.get("paired_bootstrap_manifest") != {
        "path": str(paired_path),
        "artifact_sha256": paired_hash,
    }:
        raise ReleaseIntegrityError("batch summary paired-manifest binding changed")


def _read_verified_output(path: Path, expected_hash: str) -> Mapping[str, object]:
    try:
        payload = read_json_mapping(path, context=f"completed final output {path.name}")
    except (OSError, ReleaseGateError) as error:
        raise ReleaseIntegrityError(
            f"completed final output is missing or unreadable: {path}"
        ) from error
    stored = _hash_string(payload.get("artifact_sha256"), "artifact_sha256")
    unhashed = dict(payload)
    del unhashed["artifact_sha256"]
    if stored != expected_hash or canonical_sha256(unhashed) != stored:
        raise ReleaseIntegrityError(f"completed final output hash mismatch: {path}")
    return payload


def _final_batch_result(
    destination: Path,
    ledger: FinalOpeningLedger,
    outputs: Mapping[str, object],
) -> FinalBatchResult:
    architecture_paths = {
        architecture: Path(
            _string(outputs[f"architecture_{architecture}_path"], "architecture path")
        )
        for architecture in EXPECTED_ARCHITECTURES
    }
    return FinalBatchResult(
        ledger_path=destination,
        batch_summary_path=Path(
            _string(outputs["batch_summary_path"], "batch_summary_path")
        ),
        paired_manifest_path=Path(
            _string(outputs["paired_manifest_path"], "paired_manifest_path")
        ),
        architecture_report_paths=architecture_paths,
        batch_sha256=ledger.plan.batch_sha256,
    )


def _export_or_resume_prediction(
    refit: RefitMember,
    calibration: CalibrationMember,
    prediction_path: Path,
    *,
    state: Mapping[str, object],
    settings: FinalBatchSettings,
    inference: Mapping[str, object] | None = None,
    protocol: ExperimentProtocol,
    test_access: FinalTestAccessToken,
    exporter: PredictionExporter,
) -> PredictionArtifact:
    resolved_inference = (
        dict(inference)
        if inference is not None
        else _resolved_member_inference(refit, settings, frozen_device=None)
    )
    _validate_inference_mapping(resolved_inference)
    if state["state"] == "planned":
        _quarantine_planned_partial_prediction(refit, prediction_path)
    if state["state"] == "planned" and not (
        prediction_path.exists() or prediction_path.with_suffix(".json").exists()
    ):
        result = exporter(
            PredictionExportRequest(
                checkpoint_path=refit.final_checkpoint_path,
                resolved_config_path=refit.resolved_config_path,
                run_metadata_path=refit.metadata_path,
                refit_completion_path=refit.completion_path,
                output_path=prediction_path,
                fold_role=FoldRole.FINAL_TEST,
                batch_size=_positive_int(
                    resolved_inference["batch_size"], "batch_size", minimum=1
                ),
                num_workers=_positive_int(
                    resolved_inference["num_workers"], "num_workers", minimum=0
                ),
                device=_string(resolved_inference["device"], "device"),
                bf16=_boolean(resolved_inference["bf16"], "bf16"),
            ),
            protocol=protocol,
            test_access=test_access,
        )
        _validate_final_export_result(
            result,
            refit=refit,
            prediction_path=prediction_path,
            inference=resolved_inference,
        )
    prediction = load_prediction_artifact(
        prediction_path,
        protocol=protocol,
        test_access=test_access,
        expected_config_hash=refit.resolved_config_hash,
        expected_manifest_hash=refit.manifest_sha256,
    )
    _validate_final_prediction(
        prediction,
        refit,
        calibration,
        inference=resolved_inference,
        purpose=test_access.purpose,
    )
    expected_artifact = state.get("final_prediction_artifact_sha256")
    if expected_artifact is not None and expected_artifact != prediction.integrity_sha256:
        raise ReleaseIntegrityError("resumed final prediction hash differs from ledger")
    expected_file = state.get("final_prediction_file_sha256")
    if expected_file is not None and expected_file != sha256_file(prediction_path):
        raise ReleaseIntegrityError("resumed final prediction file differs from ledger")
    expected_sidecar = state.get("final_prediction_sidecar_sha256")
    sidecar_path = prediction_path.with_suffix(".json")
    if expected_sidecar is not None and expected_sidecar != sha256_file(sidecar_path):
        raise ReleaseIntegrityError(
            "resumed final prediction sidecar differs from ledger"
        )
    return prediction


def _quarantine_planned_partial_prediction(
    refit: RefitMember, prediction_path: Path
) -> Mapping[str, object] | None:
    """Move one orphaned half-pair aside before a planned deterministic retry."""

    pair = (prediction_path, prediction_path.with_suffix(".json"))
    existing = tuple(path for path in pair if _path_lexists(path))
    if len(existing) != 1:
        return None
    orphan = existing[0]
    if not orphan.is_file():
        raise ReleaseIntegrityError(
            f"partial final prediction path is not a regular file: {orphan}"
        )
    digest = sha256_file(orphan)
    quarantine = (
        prediction_path.parent
        / FINAL_PREDICTION_ORPHAN_DIRECTORY
        / refit.member_id
    )
    quarantine.mkdir(parents=True, exist_ok=True)
    destination = quarantine / f"{orphan.name}.{digest}.orphan"
    if _path_lexists(destination):
        if not destination.is_file() or sha256_file(destination) != digest:
            raise ReleaseIntegrityError(
                "existing deterministic orphan evidence differs from fragment"
            )
    else:
        try:
            os.link(orphan, destination)
        except OSError as error:
            raise ReleaseStateError(
                f"could not preserve partial final prediction {orphan}: {error}"
            ) from error
    try:
        orphan.unlink()
    except OSError as error:
        # The evidence hard link is already durable. Fail closed and leave both
        # names rather than risking loss of the fragment.
        raise ReleaseStateError(
            f"could not remove original partial prediction {orphan}: {error}"
        ) from error
    return {
        "member_id": refit.member_id,
        "original_path": str(orphan.resolve()),
        "quarantine_path": str(destination.resolve()),
        "fragment_kind": "npz" if orphan.suffix.casefold() == ".npz" else "json",
        "file_sha256": digest,
        "reason": "planned_member_had_exactly_one_prediction_pair_fragment",
    }


def _reconcile_planned_orphan_evidence(
    path: Path,
    ledger: FinalOpeningLedger,
    *,
    refit: RefitMember,
    prediction_path: Path,
) -> FinalOpeningLedger:
    _quarantine_planned_partial_prediction(refit, prediction_path)
    recorded = {
        event.get("quarantine_path")
        for event in ledger.events
        if event.get("event") == "partial_final_prediction_quarantined"
    }
    quarantine = (
        prediction_path.parent
        / FINAL_PREDICTION_ORPHAN_DIRECTORY
        / refit.member_id
    )
    if not quarantine.exists():
        return ledger
    candidates: list[tuple[Path, str, Path]] = []
    for fragment_kind, original in (
        ("npz", prediction_path),
        ("json", prediction_path.with_suffix(".json")),
    ):
        candidates.extend(
            (candidate, fragment_kind, original)
            for candidate in sorted(quarantine.glob(f"{original.name}.*.orphan"))
        )
    for candidate, fragment_kind, original in candidates:
        resolved = str(candidate.resolve())
        if resolved in recorded:
            continue
        if not candidate.is_file():
            raise ReleaseIntegrityError("orphan evidence is not a regular file")
        record: dict[str, object] = {
            "member_id": refit.member_id,
            "original_path": str(original.resolve()),
            "quarantine_path": resolved,
            "fragment_kind": fragment_kind,
            "file_sha256": sha256_file(candidate),
            "reason": "reconciled_planned_prediction_pair_fragment",
        }
        ledger = _record_orphan_quarantine(path, ledger, record)
        recorded.add(resolved)
    return ledger


def _record_orphan_quarantine(
    path: Path,
    ledger: FinalOpeningLedger,
    record: Mapping[str, object],
) -> FinalOpeningLedger:
    return _append_event(
        path,
        ledger,
        members=ledger.members,
        event={"event": "partial_final_prediction_quarantined", **dict(record)},
    )


def _validate_final_prediction(
    prediction: PredictionArtifact,
    refit: RefitMember,
    calibration: CalibrationMember,
    *,
    inference: Mapping[str, object],
    purpose: str,
) -> None:
    if prediction.fold_role is not FoldRole.FINAL_TEST or prediction.folds != FINAL_TEST_FOLDS:
        raise ReleaseIntegrityError("final prediction must contain fold 10 only")
    expected: dict[str, object] = {
        "model_name": refit.run_name,
        "model_seed": refit.seed,
        "protocol_hash": refit.protocol_hash,
        "config_hash": refit.resolved_config_hash,
        "manifest_hash": refit.manifest_sha256,
        "label_order": LABEL_ORDER,
    }
    observed: dict[str, object] = {
        "model_name": prediction.model_name,
        "model_seed": prediction.model_seed,
        "protocol_hash": prediction.protocol_hash,
        "config_hash": prediction.config_hash,
        "manifest_hash": prediction.manifest_hash,
        "label_order": prediction.label_order,
    }
    mismatches = [name for name in expected if expected[name] != observed[name]]
    if mismatches:
        raise ReleaseIntegrityError(
            "final prediction lineage mismatch: " + ", ".join(mismatches)
        )
    extra = prediction.extra_metadata
    if extra.get("lineage") != "frozen_refit":
        raise ReleaseIntegrityError("final prediction must use frozen_refit lineage")
    if (
        _normalized_hash(extra.get("checkpoint_sha256"), "checkpoint_sha256")
        != refit.final_checkpoint_sha256
    ):
        raise ReleaseIntegrityError("final prediction checkpoint hash mismatch")
    if (
        _normalized_hash(extra.get("normalization_sha256"), "normalization_sha256")
        != refit.normalization_sha256
    ):
        raise ReleaseIntegrityError("final prediction normalization hash mismatch")
    if calibration.resolved_config_hash != prediction.config_hash:
        raise ReleaseIntegrityError("final prediction differs from frozen calibration config")
    post_sweep_expected: dict[str, object] = {
        "refit_run_kind": "post_sweep_frozen_refit",
        "refit_completion_sha256": refit.completion_sha256,
        "freeze_artifact_sha256": refit.freeze_artifact_sha256,
        "recipe_sha256": refit.recipe_sha256,
        "checkpoint_epoch": refit.frozen_epochs - 1,
        "final_test_purpose": purpose,
        "inference_batch_size": inference["batch_size"],
        "inference_num_workers": inference["num_workers"],
        "inference_device": inference["device"],
        "inference_bf16": inference["bf16"],
    }
    post_sweep_drift = [
        field
        for field, expected_value in post_sweep_expected.items()
        if extra.get(field) != expected_value
    ]
    if post_sweep_drift:
        raise ReleaseIntegrityError(
            "final prediction frozen execution mismatch: "
            + ", ".join(post_sweep_drift)
        )


def _validate_final_export_result(
    result: PredictionExportResult,
    *,
    refit: RefitMember,
    prediction_path: Path,
    inference: Mapping[str, object],
) -> None:
    expected: dict[str, object] = {
        "fold_role": FoldRole.FINAL_TEST,
        "folds": FINAL_TEST_FOLDS,
        "lineage": "frozen_refit",
        "model_name": refit.run_name,
        "model_seed": refit.seed,
        "checkpoint_sha256": refit.final_checkpoint_sha256,
        "config_hash": refit.resolved_config_hash,
        "manifest_hash": refit.manifest_sha256,
        "normalization_sha256": refit.normalization_sha256,
        "device": inference["device"],
        "bf16_enabled": inference["bf16"],
    }
    observed: dict[str, object] = {
        "fold_role": result.fold_role,
        "folds": result.folds,
        "lineage": result.lineage,
        "model_name": result.model_name,
        "model_seed": result.model_seed,
        "checkpoint_sha256": _normalized_hash(
            result.checkpoint_sha256, "export checkpoint_sha256"
        ),
        "config_hash": result.config_hash,
        "manifest_hash": result.manifest_hash,
        "normalization_sha256": _normalized_hash(
            result.normalization_sha256, "export normalization_sha256"
        ),
        "device": result.device.strip().casefold(),
        "bf16_enabled": result.bf16_enabled,
    }
    mismatches = [name for name in expected if expected[name] != observed[name]]
    if mismatches:
        raise ReleaseIntegrityError(
            "final exporter returned artifacts outside the frozen plan: "
            + ", ".join(mismatches)
        )
    if result.files.npz_path.resolve() != prediction_path.resolve() or (
        result.files.json_path.resolve() != prediction_path.with_suffix(".json").resolve()
    ):
        raise ReleaseIntegrityError("final exporter wrote outside the planned path")
    if _normalized_hash(
        result.files.npz_sha256, "export npz_sha256"
    ) != "sha256:" + sha256_file(prediction_path):
        raise ReleaseIntegrityError("final exporter NPZ hash differs from saved file")


def _record_prediction(
    path: Path,
    ledger: FinalOpeningLedger,
    member_id: str,
    prediction_path: Path,
    prediction: PredictionArtifact,
) -> FinalOpeningLedger:
    states = {key: dict(value) for key, value in ledger.members.items()}
    state = states[member_id]
    state.update(
        {
            "state": "prediction_saved",
            "final_prediction_artifact_sha256": prediction.integrity_sha256,
            "final_prediction_file_sha256": sha256_file(prediction_path),
            "final_prediction_sidecar_sha256": sha256_file(
                prediction_path.with_suffix(".json")
            ),
        }
    )
    return _append_event(
        path,
        ledger,
        members=states,
        event={
            "event": "fold10_prediction_saved",
            "member_id": member_id,
            "artifact_sha256": prediction.integrity_sha256,
        },
    )


def _record_report(
    path: Path, ledger: FinalOpeningLedger, member_id: str, report_sha256: str
) -> FinalOpeningLedger:
    states = {key: dict(value) for key, value in ledger.members.items()}
    states[member_id].update(
        {"state": "report_saved", "final_report_sha256": report_sha256}
    )
    return _append_event(
        path,
        ledger,
        members=states,
        event={
            "event": "final_member_report_saved",
            "member_id": member_id,
            "report_sha256": report_sha256,
        },
    )


def _validate_member_report(
    report: Mapping[str, object],
    *,
    refit: RefitMember,
    calibration: CalibrationMember,
    prediction: PredictionArtifact,
    settings: FinalBatchSettings,
    plan: FinalBatchPlan,
) -> None:
    expected_top: dict[str, object] = {
        "model": {"name": refit.run_name, "seed": refit.seed},
        "protocol_hash": refit.protocol_hash,
        "config_hash": refit.resolved_config_hash,
        "manifest_hash": refit.manifest_sha256,
        "label_order": list(LABEL_ORDER),
    }
    preregistration = _publication_preregistration(plan)
    expected_top["final_evaluation_spec"] = plan.final_evaluation_spec
    expected_top["protocol_deviations"] = preregistration["protocol_deviations"]
    top_drift = [
        field
        for field, expected in expected_top.items()
        if report.get(field) != expected
    ]
    if top_drift:
        raise ReleaseIntegrityError(
            "final report member lineage mismatch: " + ", ".join(top_drift)
        )
    sources = _mapping(report.get("sources"), "final report sources")
    expected_sources: dict[str, object] = {
        "calibration_artifact_sha256": calibration.decision_artifact_sha256,
        "final_prediction_sha256": prediction.integrity_sha256,
        "final_alignment_sha256": prediction.alignment_sha256,
        "final_fold_role": FoldRole.FINAL_TEST.value,
        "final_folds": list(FINAL_TEST_FOLDS),
    }
    if dict(sources) != expected_sources:
        raise ReleaseIntegrityError("final report source artifacts differ from plan")
    frozen = _mapping(report.get("frozen_decisions"), "frozen_decisions")
    expected_frozen: dict[str, object] = {
        "temperature": calibration.temperature,
        "thresholds": list(calibration.thresholds),
        "uncertainty_method": "mean_normalized_binary_entropy",
        "coverage_gates": [dict(gate) for gate in calibration.entropy_gates],
    }
    if dict(frozen) != expected_frozen:
        raise ReleaseIntegrityError("final report did not apply exact frozen decisions")
    bootstrap = _mapping(report.get("patient_bootstrap"), "patient_bootstrap")
    expected_bootstrap: dict[str, object] = {
        "method": "patient_cluster_percentile_bootstrap",
        "requested_resamples": settings.bootstrap_resamples,
        "seed": settings.bootstrap_seed + refit.seed,
        "confidence_level": settings.bootstrap_confidence,
        "minimum_valid_resamples": settings.bootstrap_minimum_valid,
        "ece_bins": settings.ece_bins,
        "label_order": list(LABEL_ORDER),
    }
    bootstrap_drift = [
        field
        for field, expected in expected_bootstrap.items()
        if bootstrap.get(field) != expected
    ]
    if bootstrap_drift:
        raise ReleaseIntegrityError(
            "final report bootstrap policy differs from preregistration: "
            + ", ".join(bootstrap_drift)
        )
    subgroup = _mapping(report.get("subgroup_audit"), "subgroup_audit")
    expected_subgroup: dict[str, object] = {
        "label_order": list(LABEL_ORDER),
        "ece_bins": settings.ece_bins,
        "minimum_group_samples": settings.minimum_group_samples,
        "minimum_group_patients": settings.minimum_group_patients,
    }
    subgroup_drift = [
        field
        for field, expected in expected_subgroup.items()
        if subgroup.get(field) != expected
    ]
    if subgroup_drift:
        raise ReleaseIntegrityError(
            "final report subgroup policy differs from preregistration: "
            + ", ".join(subgroup_drift)
        )


def _build_aggregate_outputs(
    plan: FinalBatchPlan,
    calibration_bundle: CalibrationBundle,
    *,
    protocol: ExperimentProtocol,
    test_access: FinalTestAccessToken,
) -> dict[str, object]:
    destination = plan.settings.output_directory
    destination.mkdir(parents=True, exist_ok=True)
    preregistration = _publication_preregistration(plan)
    calibration = {member.member_id: member for member in calibration_bundle.members}
    member_reports: dict[str, Mapping[str, object]] = {}
    final_predictions: dict[str, PredictionArtifact] = {}
    for member in plan.members:
        member_id = _string(member["member_id"], "member_id")
        report_path = Path(_string(member["final_report_path"], "final_report_path"))
        prediction_path = Path(
            _string(member["final_prediction_path"], "final_prediction_path")
        )
        member_reports[member_id] = verify_final_report(
            report_path, protocol=protocol, test_access=test_access
        )
        final_predictions[member_id] = load_prediction_artifact(
            prediction_path,
            protocol=protocol,
            test_access=test_access,
            expected_config_hash=calibration[member_id].resolved_config_hash,
            expected_manifest_hash=plan.manifest_sha256,
        )

    architecture_outputs: dict[str, tuple[Path, str]] = {}
    for architecture in EXPECTED_ARCHITECTURES:
        rows = []
        for seed in EXPECTED_SEEDS:
            member_id = f"{architecture}-seed{seed}"
            report = member_reports[member_id]
            metrics = _mapping(report["metrics"], "metrics")
            macro = _mapping(metrics["macro"], "metrics.macro")
            rows.append(
                {
                    "seed": seed,
                    "member_id": member_id,
                    "report_sha256": report["report_sha256"],
                    "macro": dict(macro),
                }
            )
        metrics_summary = _aggregate_macro_rows(rows)
        payload: dict[str, object] = {
            "schema_version": 1,
            "artifact_type": ARCHITECTURE_REPORT_TYPE,
            "batch_sha256": plan.batch_sha256,
            "preregistration": preregistration,
            "architecture": architecture,
            "seeds": list(EXPECTED_SEEDS),
            "members": rows,
            "cross_seed_summary": metrics_summary,
            "interpretation": "descriptive_summary_across_three_preregistered_seeds",
        }
        path = destination / f"{architecture}.architecture-summary.json"
        architecture_outputs[architecture] = _write_or_verify(
            path, payload, hash_field="artifact_sha256"
        )

    paired_entries: list[dict[str, object]] = []
    for seed in EXPECTED_SEEDS:
        resnet_id = f"resnet1d-seed{seed}"
        transformer_id = f"ecg_transformer-seed{seed}"
        resnet_prediction = final_predictions[resnet_id]
        transformer_prediction = final_predictions[transformer_id]
        assert_prediction_artifacts_aligned(resnet_prediction, transformer_prediction)
        resnet_decisions = load_calibration_decisions(
            calibration[resnet_id].decision_path, protocol=protocol
        )
        transformer_decisions = load_calibration_decisions(
            calibration[transformer_id].decision_path, protocol=protocol
        )
        resnet_probabilities = resnet_decisions.temperature_scaling.predict_proba(
            resnet_prediction.raw_logits, label_order=resnet_prediction.label_order
        )
        transformer_probabilities = (
            transformer_decisions.temperature_scaling.predict_proba(
                transformer_prediction.raw_logits,
                label_order=transformer_prediction.label_order,
            )
        )
        paired = paired_model_difference_intervals(
            resnet_prediction.targets,
            transformer_probabilities,
            resnet_probabilities,
            resnet_prediction.patient_id,
            model_a=transformer_id,
            model_b=resnet_id,
            n_resamples=plan.settings.bootstrap_resamples,
            seed=plan.settings.bootstrap_seed + seed,
            confidence_level=plan.settings.bootstrap_confidence,
            minimum_valid_resamples=plan.settings.bootstrap_minimum_valid,
            label_order=plan.label_order,
            ece_bins=plan.settings.ece_bins,
        )
        paired_payload: dict[str, object] = {
            "schema_version": 1,
            "artifact_type": "ecg_trust.paired_patient_bootstrap_report",
            "batch_sha256": plan.batch_sha256,
            "preregistration": preregistration,
            "seed": seed,
            "sources": {
                "transformer_prediction_sha256": (
                    transformer_prediction.integrity_sha256
                ),
                "resnet_prediction_sha256": resnet_prediction.integrity_sha256,
                "transformer_decision_sha256": (
                    transformer_decisions.integrity_sha256
                ),
                "resnet_decision_sha256": resnet_decisions.integrity_sha256,
                "alignment_sha256": resnet_prediction.alignment_sha256,
            },
            "comparison": paired.to_dict(),
        }
        paired_path = destination / f"paired-seed{seed}.bootstrap.json"
        saved_path, saved_hash = _write_or_verify(
            paired_path, paired_payload, hash_field="artifact_sha256"
        )
        paired_entries.append(
            {
                "seed": seed,
                "path": str(saved_path),
                "artifact_sha256": saved_hash,
                "direction": "ecg_transformer_minus_resnet1d",
                "alignment_sha256": resnet_prediction.alignment_sha256,
            }
        )

    paired_manifest_payload: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": PAIRED_BOOTSTRAP_MANIFEST_TYPE,
        "batch_sha256": plan.batch_sha256,
        "preregistration": preregistration,
        "pairing": "within_seed_same_fold10_patients",
        "patient_resampling": "patient_cluster_percentile_bootstrap",
        "entries": paired_entries,
    }
    paired_manifest = _write_or_verify(
        destination / "paired-patient-bootstrap.manifest.json",
        paired_manifest_payload,
        hash_field="artifact_sha256",
    )
    summary_payload: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": FINAL_BATCH_SUMMARY_TYPE,
        "batch_sha256": plan.batch_sha256,
        "preregistration": preregistration,
        "member_reports": [
            {
                "member_id": member_id,
                "report_sha256": report["report_sha256"],
            }
            for member_id, report in sorted(member_reports.items())
        ],
        "architecture_reports": {
            architecture: {"path": str(saved[0]), "artifact_sha256": saved[1]}
            for architecture, saved in architecture_outputs.items()
        },
        "paired_bootstrap_manifest": {
            "path": str(paired_manifest[0]),
            "artifact_sha256": paired_manifest[1],
        },
        "retuning_performed": False,
    }
    batch_summary = _write_or_verify(
        destination / "final-batch-summary.json",
        summary_payload,
        hash_field="artifact_sha256",
    )
    outputs: dict[str, object] = {
        "batch_summary_path": str(batch_summary[0]),
        "batch_summary_sha256": batch_summary[1],
        "paired_manifest_path": str(paired_manifest[0]),
        "paired_manifest_sha256": paired_manifest[1],
    }
    for architecture, saved in architecture_outputs.items():
        outputs[f"architecture_{architecture}_path"] = str(saved[0])
        outputs[f"architecture_{architecture}_sha256"] = saved[1]
    return outputs


def _publication_preregistration(plan: FinalBatchPlan) -> dict[str, object]:
    spec_binding = dict(plan.final_evaluation_spec)
    spec_path = Path(
        _string(spec_binding["path"], "final evaluation spec path")
    ).resolve()
    payload = read_json_mapping(spec_path, context="published final evaluation spec")
    if payload.get("artifact_sha256") != spec_binding["artifact_sha256"] or (
        "sha256:" + sha256_file(spec_path) != spec_binding["file_sha256"]
    ):
        raise ReleaseIntegrityError("published final-evaluation spec binding changed")
    deviation = _mapping(
        payload.get("protocol_deviations"), "protocol deviations binding"
    )
    _exact_keys(
        deviation,
        {"path", "file_sha256", "required_in_final_reporting"},
        "protocol deviations binding",
    )
    deviation_path = Path(
        _string(deviation["path"], "protocol deviations path")
    ).resolve()
    if deviation.get("required_in_final_reporting") is not True or (
        "sha256:" + sha256_file(deviation_path) != deviation.get("file_sha256")
    ):
        raise ReleaseIntegrityError("protocol-deviation disclosure binding changed")
    return {
        "final_evaluation_spec": spec_binding,
        "protocol_deviations": dict(deviation),
    }


def _aggregate_macro_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    summary: dict[str, object] = {}
    for metric in ("roc_auc", "average_precision", "brier_score", "ece"):
        values: list[float] = []
        for row in rows:
            macro = _mapping(row["macro"], "macro")
            value = macro.get(metric)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                numeric = float(value)
                if math.isfinite(numeric):
                    values.append(numeric)
        summary[metric] = {
            "values": values,
            "mean": statistics.fmean(values) if values else None,
            "sample_standard_deviation": statistics.stdev(values)
            if len(values) > 1
            else None,
            "median": statistics.median(values) if values else None,
            "minimum": min(values) if values else None,
            "maximum": max(values) if values else None,
            "valid_seeds": len(values),
        }
    return summary


def _complete_ledger(
    path: Path, ledger: FinalOpeningLedger, outputs: Mapping[str, object]
) -> FinalOpeningLedger:
    if any(state["state"] != "report_saved" for state in ledger.members.values()):
        raise ReleaseStateError("cannot complete ledger before all six reports exist")
    return _append_event(
        path,
        ledger,
        members=ledger.members,
        outputs=outputs,
        state="complete",
        event={"event": "exact_six_member_final_batch_complete"},
    )


def _record_failure(path: Path, ledger: FinalOpeningLedger, error: Exception) -> None:
    with suppress(Exception):
        _append_event(
            path,
            ledger,
            members=ledger.members,
            event={
                "event": "batch_interrupted",
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )


def _append_event(
    path: Path,
    ledger: FinalOpeningLedger,
    *,
    members: Mapping[str, Mapping[str, object]],
    event: Mapping[str, object],
    outputs: Mapping[str, object] | None = None,
    state: str | None = None,
) -> FinalOpeningLedger:
    timestamp = datetime.now(UTC).isoformat()
    appended = {
        "sequence": len(ledger.events),
        "timestamp_utc": timestamp,
        **dict(event),
    }
    updated = FinalOpeningLedger(
        plan=ledger.plan,
        purpose=ledger.purpose,
        operator=ledger.operator,
        confirmation_sha256=ledger.confirmation_sha256,
        opening_intent_sha256=ledger.opening_intent_sha256,
        state=state or ledger.state,
        members=members,
        outputs=outputs if outputs is not None else ledger.outputs,
        events=ledger.events + (appended,),
        created_at_utc=ledger.created_at_utc,
        updated_at_utc=timestamp,
        ledger_sha256=None,
    )
    return _commit_ledger(path, updated, replace=True)


def _commit_ledger(
    path: Path, ledger: FinalOpeningLedger, *, replace: bool
) -> FinalOpeningLedger:
    payload = ledger.to_payload(include_integrity=False)
    digest = canonical_sha256(payload)
    committed = dict(payload)
    committed["ledger_sha256"] = digest
    with _ledger_writer_lock(path):
        if replace:
            existing = read_json_mapping(path, context="current opening ledger")
            stored = _hash_string(existing.get("ledger_sha256"), "ledger_sha256")
            unhashed = dict(existing)
            del unhashed["ledger_sha256"]
            if canonical_sha256(unhashed) != stored:
                raise ReleaseIntegrityError("current opening ledger hash mismatch")
            previous_events = [dict(event) for event in ledger.events[:-1]]
            if existing.get("events") != previous_events:
                raise ReleaseStateError(
                    "concurrent opening-ledger update detected; reload before retrying"
                )
        _atomic_json(path, committed, replace=replace)
    return FinalOpeningLedger(
        plan=ledger.plan,
        purpose=ledger.purpose,
        operator=ledger.operator,
        confirmation_sha256=ledger.confirmation_sha256,
        opening_intent_sha256=ledger.opening_intent_sha256,
        state=ledger.state,
        members=ledger.members,
        outputs=ledger.outputs,
        events=ledger.events,
        created_at_utc=ledger.created_at_utc,
        updated_at_utc=ledger.updated_at_utc,
        ledger_sha256=digest,
    )


@contextmanager
def _ledger_writer_lock(path: Path) -> Iterator[None]:
    """Reserve one kernel-locked, crash-recoverable exact-owner writer."""

    lock_path = path.with_name(path.name + ".writer.lock")
    guard_path = lock_path.with_name(lock_path.name + ".guard")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    nonce = uuid.uuid4().hex
    host = socket.gethostname()
    process_started_at = psutil.Process(os.getpid()).create_time()
    payload: dict[str, object] = {
        "schema_version": FINAL_LEDGER_LOCK_SCHEMA_VERSION,
        "artifact_type": FINAL_LEDGER_LOCK_TYPE,
        "host": host,
        "pid": os.getpid(),
        "process_started_at": process_started_at,
        "nonce": nonce,
    }
    kernel_lock = FileLock(str(guard_path))
    try:
        kernel_lock.acquire(timeout=0)
    except Timeout as error:
        raise ReleaseStateError(
            f"opening ledger already has an active writer: path={lock_path}"
        ) from error
    try:
        if lock_path.exists():
            existing = read_json_mapping(
                lock_path, context="opening ledger writer lock"
            )
            _exact_keys(
                existing,
                {
                    "schema_version",
                    "artifact_type",
                    "host",
                    "pid",
                    "process_started_at",
                    "nonce",
                },
                "opening ledger writer lock",
            )
            if (
                existing["schema_version"] != FINAL_LEDGER_LOCK_SCHEMA_VERSION
                or existing["artifact_type"] != FINAL_LEDGER_LOCK_TYPE
            ):
                raise ReleaseIntegrityError(
                    "opening ledger writer lock has an unsupported identity"
                )
            lock_host = _string(existing.get("host"), "ledger lock host")
            lock_pid = _positive_int(
                existing.get("pid"), "ledger lock pid", minimum=1
            )
            lock_started_at = _float(
                existing.get("process_started_at"),
                "ledger lock process_started_at",
            )
            _string(existing.get("nonce"), "ledger lock nonce")
            if lock_host != host or _process_identity_is_alive(
                lock_pid, lock_started_at
            ):
                raise ReleaseStateError(
                    "opening ledger already has an active writer: "
                    f"host={lock_host}, pid={lock_pid}, path={lock_path}"
                )
        _atomic_json(lock_path, payload, replace=lock_path.exists())
        try:
            yield
        finally:
            with suppress(OSError, ReleaseGateError):
                current = read_json_mapping(
                    lock_path, context="opening ledger writer lock"
                )
                if current.get("nonce") == nonce:
                    lock_path.unlink(missing_ok=True)
    finally:
        kernel_lock.release()


def _process_identity_is_alive(pid: int, process_started_at: float) -> bool:
    try:
        observed = psutil.Process(pid).create_time()
    except psutil.NoSuchProcess:
        return False
    except psutil.AccessDenied:
        return True
    return math.isclose(observed, process_started_at, rel_tol=0.0, abs_tol=1e-3)


def _pid_is_alive(pid: int) -> bool:
    """Compatibility helper used by diagnostics and focused unit tests."""

    try:
        psutil.Process(pid)
    except psutil.NoSuchProcess:
        return False
    except psutil.AccessDenied:
        return True
    return True


def _parse_plan(value: object) -> FinalBatchPlan:
    root = _mapping(value, "final batch plan")
    required = {
        "schema_version",
        "artifact_type",
        "protocol_hash",
        "refit_bundle_sha256",
        "calibration_bundle_sha256",
        "manifest_sha256",
        "normalization_sha256",
        "label_order",
        "final_evaluation_spec",
        "opening_marker_path",
        "settings",
        "members",
        "member_count",
        "final_folds",
        "retuning_allowed",
        "batch_sha256",
    }
    _exact_keys(root, required, "final batch plan")
    if root["schema_version"] != FINAL_BATCH_PLAN_SCHEMA_VERSION:
        raise ReleaseIntegrityError("unsupported final batch plan schema")
    if root["artifact_type"] != FINAL_BATCH_PLAN_TYPE:
        raise ReleaseIntegrityError("unexpected final batch plan type")
    if root["member_count"] != 6 or root["retuning_allowed"] is not False:
        raise ReleaseIntegrityError("final batch cardinality or tuning policy changed")
    if _integer_tuple(root["final_folds"], "final_folds") != FINAL_TEST_FOLDS:
        raise ReleaseIntegrityError("final batch must contain fold 10 only")
    settings = _parse_settings(root["settings"])
    members = tuple(
        _mapping(item, "final batch member")
        for item in _sequence(root["members"], "members")
    )
    _validate_plan_members(members)
    stored = _hash_string(root["batch_sha256"], "batch_sha256")
    unhashed = dict(root)
    del unhashed["batch_sha256"]
    if canonical_sha256(unhashed) != stored:
        raise ReleaseIntegrityError("final batch plan hash mismatch")
    final_spec = _mapping(root["final_evaluation_spec"], "final_evaluation_spec")
    _exact_keys(
        final_spec,
        {"path", "file_sha256", "artifact_sha256"},
        "final_evaluation_spec",
    )
    _string(final_spec["path"], "final_evaluation_spec.path")
    _hash_string(final_spec["file_sha256"], "final_evaluation_spec.file_sha256")
    _hash_string(
        final_spec["artifact_sha256"],
        "final_evaluation_spec.artifact_sha256",
    )
    return FinalBatchPlan(
        protocol_hash=_hash_string(root["protocol_hash"], "protocol_hash"),
        refit_bundle_sha256=_hash_string(
            root["refit_bundle_sha256"], "refit_bundle_sha256"
        ),
        calibration_bundle_sha256=_hash_string(
            root["calibration_bundle_sha256"], "calibration_bundle_sha256"
        ),
        manifest_sha256=_hash_string(root["manifest_sha256"], "manifest_sha256"),
        normalization_sha256=_hash_string(
            root["normalization_sha256"], "normalization_sha256"
        ),
        label_order=_label_order(root["label_order"]),
        final_evaluation_spec=dict(final_spec),
        opening_marker_path=Path(
            _string(root["opening_marker_path"], "opening_marker_path")
        ),
        settings=settings,
        members=members,
        batch_sha256=stored,
    )


def _parse_settings(value: object) -> FinalBatchSettings:
    root = _mapping(value, "settings")
    _exact_keys(
        root,
        {"output_directory", "subgroups", "inference", "evaluation", "retuning_allowed"},
        "settings",
    )
    if root["retuning_allowed"] is not False:
        raise ReleaseIntegrityError("final settings cannot enable retuning")
    subgroups = _mapping(root["subgroups"], "settings.subgroups")
    _exact_keys(subgroups, {"path", "sha256"}, "settings.subgroups")
    inference = _mapping(root["inference"], "settings.inference")
    _exact_keys(inference, {"batch_size", "num_workers", "device", "bf16"}, "settings.inference")
    evaluation = _mapping(root["evaluation"], "settings.evaluation")
    _exact_keys(
        evaluation,
        {
            "bootstrap_resamples",
            "bootstrap_seed",
            "bootstrap_confidence",
            "bootstrap_minimum_valid",
            "minimum_group_samples",
            "minimum_group_patients",
            "ece_bins",
            "paired_bootstrap_seed_strategy",
        },
        "settings.evaluation",
    )
    if evaluation["paired_bootstrap_seed_strategy"] != "base_plus_model_seed":
        raise ReleaseIntegrityError("paired bootstrap seed strategy changed")
    return FinalBatchSettings(
        output_directory=Path(_string(root["output_directory"], "output_directory")),
        subgroup_path=Path(_string(subgroups["path"], "subgroups.path")),
        subgroup_sha256=_raw_hash_string(subgroups["sha256"], "subgroups.sha256"),
        batch_size=_optional_int(inference["batch_size"], "batch_size", minimum=1),
        num_workers=_optional_int(inference["num_workers"], "num_workers", minimum=0),
        device=_string(inference["device"], "device"),
        bf16=_boolean(inference["bf16"], "bf16"),
        bootstrap_resamples=_positive_int(
            evaluation["bootstrap_resamples"], "bootstrap_resamples", minimum=2
        ),
        bootstrap_seed=_positive_int(
            evaluation["bootstrap_seed"], "bootstrap_seed", minimum=0
        ),
        bootstrap_confidence=_float(
            evaluation["bootstrap_confidence"], "bootstrap_confidence"
        ),
        bootstrap_minimum_valid=_optional_int(
            evaluation["bootstrap_minimum_valid"],
            "bootstrap_minimum_valid",
            minimum=1,
        ),
        minimum_group_samples=_positive_int(
            evaluation["minimum_group_samples"], "minimum_group_samples", minimum=1
        ),
        minimum_group_patients=_positive_int(
            evaluation["minimum_group_patients"],
            "minimum_group_patients",
            minimum=1,
        ),
        ece_bins=_positive_int(evaluation["ece_bins"], "ece_bins", minimum=2),
    )


def _validate_plan_members(members: tuple[Mapping[str, object], ...]) -> None:
    if len(members) != 6:
        raise ReleaseIntegrityError("final batch plan requires six members")
    observed: set[tuple[str, int]] = set()
    required = {
        "member_id",
        "architecture",
        "seed",
        "model_name",
        "refit_lineage_sha256",
        "checkpoint_sha256",
        "resolved_config_hash",
        "calibration_decision_sha256",
        "fold9_prediction_sha256",
        "inference",
        "final_prediction_path",
        "final_report_path",
    }
    for member in members:
        _exact_keys(member, required, "final batch member")
        architecture = _string(member["architecture"], "architecture")
        seed = _positive_int(member["seed"], "seed", minimum=0)
        member_id = _string(member["member_id"], "member_id")
        _string(member["model_name"], "model_name")
        if member_id != f"{architecture}-seed{seed}":
            raise ReleaseIntegrityError("final batch member ID mismatch")
        observed.add((architecture, seed))
        _hash_string(member["refit_lineage_sha256"], "refit_lineage_sha256")
        _hash_string(member["checkpoint_sha256"], "checkpoint_sha256")
        _hash_string(member["resolved_config_hash"], "resolved_config_hash")
        _hash_string(
            member["calibration_decision_sha256"], "calibration_decision_sha256"
        )
        _hash_string(member["fold9_prediction_sha256"], "fold9_prediction_sha256")
        inference = _mapping(member["inference"], "member inference")
        _validate_inference_mapping(inference)
        _string(member["final_prediction_path"], "final_prediction_path")
        _string(member["final_report_path"], "final_report_path")
    expected = {
        (architecture, seed)
        for architecture in EXPECTED_ARCHITECTURES
        for seed in EXPECTED_SEEDS
    }
    if observed != expected:
        raise ReleaseIntegrityError("final batch plan is not the exact release grid")


def _validate_member_state(
    state: Mapping[str, object], *, member_id: str, plan: FinalBatchPlan
) -> None:
    required = {
        "state",
        "final_prediction_path",
        "final_prediction_artifact_sha256",
        "final_prediction_file_sha256",
        "final_prediction_sidecar_sha256",
        "final_report_path",
        "final_report_sha256",
    }
    _exact_keys(state, required, f"members.{member_id}")
    value = _string(state["state"], f"members.{member_id}.state")
    if value not in {"planned", "prediction_saved", "report_saved"}:
        raise ReleaseIntegrityError("unsupported final member lifecycle state")
    planned = next(
        item for item in plan.members if item["member_id"] == member_id
    )
    if state["final_prediction_path"] != planned["final_prediction_path"]:
        raise ReleaseIntegrityError("ledger prediction path differs from plan")
    if state["final_report_path"] != planned["final_report_path"]:
        raise ReleaseIntegrityError("ledger report path differs from plan")
    if value in {"prediction_saved", "report_saved"}:
        _hash_string(
            state["final_prediction_artifact_sha256"],
            "final_prediction_artifact_sha256",
        )
        _raw_hash_string(
            state["final_prediction_file_sha256"],
            "final_prediction_file_sha256",
        )
        _raw_hash_string(
            state["final_prediction_sidecar_sha256"],
            "final_prediction_sidecar_sha256",
        )
    else:
        for field in (
            "final_prediction_artifact_sha256",
            "final_prediction_file_sha256",
            "final_prediction_sidecar_sha256",
        ):
            if state[field] is not None:
                raise ReleaseIntegrityError(
                    f"planned member must keep {field} null"
                )
    if value == "report_saved":
        _hash_string(state["final_report_sha256"], "final_report_sha256")
    elif state["final_report_sha256"] is not None:
        raise ReleaseIntegrityError(
            "member report hash must remain null before report_saved"
        )


def _validate_pair(refit: RefitMember, calibration: CalibrationMember) -> None:
    expected = {
        "member_id": refit.member_id,
        "architecture": refit.architecture,
        "seed": refit.seed,
        "model_name": refit.run_name,
        "refit_lineage_sha256": refit.lineage_sha256,
        "checkpoint_path": refit.final_checkpoint_path,
        "resolved_config_hash": refit.resolved_config_hash,
        "checkpoint_sha256": refit.final_checkpoint_sha256,
        "resolved_config_path": refit.resolved_config_path,
        "resolved_config_file_sha256": refit.resolved_config_file_sha256,
        "normalization_path": refit.normalization_path,
        "normalization_sha256": refit.normalization_sha256,
    }
    observed = {
        "member_id": calibration.member_id,
        "architecture": calibration.architecture,
        "seed": calibration.seed,
        "model_name": calibration.model_name,
        "refit_lineage_sha256": calibration.refit_lineage_sha256,
        "checkpoint_path": calibration.checkpoint_path,
        "resolved_config_hash": calibration.resolved_config_hash,
        "checkpoint_sha256": calibration.checkpoint_sha256,
        "resolved_config_path": calibration.resolved_config_path,
        "resolved_config_file_sha256": calibration.resolved_config_file_sha256,
        "normalization_path": calibration.normalization_path,
        "normalization_sha256": calibration.normalization_sha256,
    }
    mismatches = [name for name in expected if expected[name] != observed[name]]
    if mismatches:
        raise ReleaseIntegrityError(
            "refit/calibration member mismatch: " + ", ".join(mismatches)
        )


def _load_subgroups(
    path: Path,
) -> tuple[NDArray[np.object_], dict[str, NDArray[np.object_]]]:
    sha256_file(path)
    root = read_json_mapping(path, context="subgroup artifact")
    _exact_keys(root, {"ecg_id", "attributes"}, "subgroup artifact")
    ids = _sequence(root["ecg_id"], "subgroup ecg_id")
    attributes = _mapping(root["attributes"], "subgroup attributes")
    result: dict[str, NDArray[np.object_]] = {}
    for name, values in attributes.items():
        if not isinstance(name, str):
            raise ReleaseIntegrityError("subgroup names must be strings")
        result[name] = np.asarray(_sequence(values, f"subgroup {name}"), dtype=object)
    return np.asarray(ids, dtype=object), result


def _write_or_verify(
    path: Path, payload: Mapping[str, object], *, hash_field: str
) -> tuple[Path, str]:
    if not path.exists():
        saved = write_new_hashed_json(path, payload, hash_field=hash_field)
        return Path(saved[0]), str(saved[1])
    existing = read_json_mapping(path, context=f"existing aggregate {path.name}")
    stored = _hash_string(existing.get(hash_field), hash_field)
    unhashed = dict(existing)
    del unhashed[hash_field]
    if canonical_sha256(unhashed) != stored or unhashed != dict(payload):
        raise ReleaseIntegrityError(f"existing aggregate differs on resume: {path}")
    return path, stored


def _atomic_json(path: Path, payload: Mapping[str, object], *, replace: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise FileExistsError(f"opening ledger already exists: {path}")
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
        if replace:
            os.replace(temporary, path)
        else:
            # A same-directory hard link is a true no-replace commit.  Unlike
            # exists()+replace(), it cannot overwrite a file created between
            # the check and the commit.
            try:
                os.link(temporary, path)
            except FileExistsError as error:
                raise FileExistsError(
                    f"immutable final-test artifact already exists: {path}"
                ) from error
    finally:
        with suppress(OSError):
            temporary.unlink(missing_ok=True)


def _path_identity(path: Path) -> str:
    """Return a symlink-resolved, platform-normalized path identity."""

    return os.path.normcase(str(path.resolve()))


def _path_lexists(path: Path) -> bool:
    """Return true for files, directories, symlinks, and broken symlinks."""

    return os.path.lexists(path)


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


def _positive_int(value: object, context: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ReleaseIntegrityError(f"{context} must be an integer >= {minimum}")
    return value


def _optional_int(value: object, context: str, *, minimum: int) -> int | None:
    if value is None:
        return None
    return _positive_int(value, context, minimum=minimum)


def _integer_tuple(value: object, context: str) -> tuple[int, ...]:
    return tuple(
        _positive_int(item, f"{context} item", minimum=1)
        for item in _sequence(value, context)
    )


def _float(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReleaseIntegrityError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ReleaseIntegrityError(f"{context} must be finite")
    return result


def _boolean(value: object, context: str) -> bool:
    if not isinstance(value, bool):
        raise ReleaseIntegrityError(f"{context} must be boolean")
    return value


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
    labels = tuple(_string(item, "label_order item") for item in _sequence(value, "label_order"))
    if labels != LABEL_ORDER:
        raise ReleaseIntegrityError("final batch label order is not canonical")
    return labels


def _timestamp(value: str | None) -> str:
    candidate = value or datetime.now(UTC).isoformat()
    try:
        parsed = datetime.fromisoformat(candidate.replace("Z", "+00:00"))
    except ValueError as error:
        raise ReleaseGateError("timestamp must be ISO-8601") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ReleaseGateError("timestamp must contain a UTC offset")
    return parsed.astimezone(UTC).isoformat()


__all__ = [
    "ARCHITECTURE_REPORT_TYPE",
    "FINAL_BATCH_PLAN_SCHEMA_VERSION",
    "FINAL_BATCH_PLAN_TYPE",
    "FINAL_BATCH_SUMMARY_TYPE",
    "FINAL_LEDGER_SCHEMA_VERSION",
    "FINAL_LEDGER_TYPE",
    "PAIRED_BOOTSTRAP_MANIFEST_TYPE",
    "FinalBatchPlan",
    "FinalBatchResult",
    "FinalBatchSettings",
    "FinalOpeningLedger",
    "authorize_ledgered_final_test",
    "build_final_batch_plan",
    "canonical_final_ledger_path",
    "create_final_opening_ledger",
    "load_final_opening_ledger",
    "open_or_resume_final_batch",
    "run_final_batch",
]
