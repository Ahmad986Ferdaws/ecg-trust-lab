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
import statistics
import tempfile
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from ecg_trust.audit import paired_model_difference_intervals
from ecg_trust.decisioning import (
    generate_final_report,
    load_calibration_decisions,
    save_final_report,
    verify_final_report,
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

FINAL_BATCH_PLAN_SCHEMA_VERSION = 1
FINAL_BATCH_PLAN_TYPE = "ecg_trust.final_test_batch_plan"
FINAL_LEDGER_SCHEMA_VERSION = 1
FINAL_LEDGER_TYPE = "ecg_trust.final_test_opening_ledger"
FINAL_OPENING_MARKER_SCHEMA_VERSION = 1
FINAL_OPENING_MARKER_TYPE = "ecg_trust.final_test_canonical_opening_marker"
ARCHITECTURE_REPORT_TYPE = "ecg_trust.final_architecture_aggregate"
PAIRED_BOOTSTRAP_MANIFEST_TYPE = "ecg_trust.paired_patient_bootstrap_manifest"
FINAL_BATCH_SUMMARY_TYPE = "ecg_trust.final_batch_summary"

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
    marker_anchor = min(
        calibration_bundle.members, key=lambda member: member.member_id
    ).decision_path.resolve().parent
    opening_marker_path = (
        marker_anchor
        / ".final-test-openings"
        / (
            calibration_bundle.artifact_sha256.removeprefix("sha256:")
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
        opening_marker_path=opening_marker_path,
        settings=settings,
        members=tuple(members),
        batch_sha256=canonical_sha256(unhashed),
    )


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
    return tuple(path for path in candidates if path.exists())


def _opening_marker_payload(
    *,
    plan: FinalBatchPlan,
    ledger_path: Path,
    created_at_utc: str,
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
        "marker_precedes_fold10_access": True,
    }


def _create_canonical_opening_marker(
    path: Path,
    *,
    plan: FinalBatchPlan,
    ledger_path: Path,
    created_at_utc: str,
) -> None:
    if path.resolve() != plan.opening_marker_path.resolve():
        raise ReleaseIntegrityError("opening marker path differs from final batch plan")
    try:
        write_new_hashed_json(
            path,
            _opening_marker_payload(
                plan=plan,
                ledger_path=ledger_path,
                created_at_utc=created_at_utc,
            ),
            hash_field="marker_sha256",
        )
    except (FileExistsError, ReleaseGateError) as error:
        raise ReleaseStateError(
            "this frozen calibration/refit release has already opened fold 10"
        ) from error


def _verify_canonical_opening_marker(
    path: Path,
    *,
    plan: FinalBatchPlan,
    ledger_path: Path,
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


def create_final_opening_ledger(
    path: str | Path,
    plan: FinalBatchPlan,
    *,
    purpose: str,
    operator: str,
    confirmation: str,
    created_at_utc: str | None = None,
) -> FinalOpeningLedger:
    """Persist the one-time opening record before any fold-10 authorization."""

    destination = Path(path)
    if destination.exists():
        raise ReleaseStateError(
            "fold-10 opening ledger already exists; use exact-batch resume"
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
            "final_report_path": member["final_report_path"],
            "final_report_sha256": None,
        }
        for member in plan.members
    }
    ledger = FinalOpeningLedger(
        plan=plan,
        purpose=normalized_purpose,
        operator=normalized_operator,
        confirmation_sha256=hashlib.sha256(
            confirmation.encode("utf-8")
        ).hexdigest(),
        state="opened",
        members=member_states,
        outputs={},
        events=(
            {
                "sequence": 0,
                "timestamp_utc": timestamp,
                "event": "ledger_created_before_fold10_access",
                "batch_sha256": plan.batch_sha256,
            },
        ),
        created_at_utc=timestamp,
        updated_at_utc=timestamp,
        ledger_sha256=None,
    )
    _create_canonical_opening_marker(
        plan.opening_marker_path,
        plan=plan,
        ledger_path=destination,
        created_at_utc=timestamp,
    )
    return _commit_ledger(destination, ledger, replace=False)


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
    _verify_canonical_opening_marker(
        plan.opening_marker_path,
        plan=plan,
        ledger_path=Path(path),
    )
    opening = _mapping(payload["opening"], "opening")
    _exact_keys(
        opening,
        {
            "purpose",
            "operator",
            "confirmation_sha256",
            "created_at_utc",
            "ledger_precedes_fold10_access",
        },
        "opening",
    )
    if opening["ledger_precedes_fold10_access"] is not True:
        raise ReleaseIntegrityError("ledger does not assert pre-access creation")
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
    state_value = _string(payload["state"], "state")
    if state_value not in {"opened", "complete"}:
        raise ReleaseIntegrityError("opening ledger has an unsupported state")
    return FinalOpeningLedger(
        plan=plan,
        purpose=_string(opening["purpose"], "opening.purpose"),
        operator=_string(opening["operator"], "opening.operator"),
        confirmation_sha256=_raw_hash_string(
            opening["confirmation_sha256"], "confirmation_sha256"
        ),
        state=state_value,
        members=parsed_states,
        outputs=_mapping(payload["outputs"], "outputs"),
        events=events,
        created_at_utc=_timestamp(
            _string(opening["created_at_utc"], "opening.created_at_utc")
        ),
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

    destination = Path(ledger_path)
    if not resume:
        return create_final_opening_ledger(
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

    ledger = load_final_opening_ledger(ledger_path, protocol=protocol)
    if ledger.plan.batch_sha256 != plan.batch_sha256:
        raise ReleaseStateError("ledger does not authorize this final batch")
    if ledger.purpose != purpose.strip():
        raise ReleaseStateError("ledger purpose does not authorize this final batch")
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
    ledger_path: str | Path,
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
    plan = build_final_batch_plan(refit_bundle, calibration_bundle, settings)
    ledger = open_or_resume_final_batch(
        ledger_path,
        plan,
        protocol=protocol,
        purpose=purpose,
        operator=operator,
        confirmation=confirmation,
        resume=resume,
    )
    # The durable ledger exists and has been re-read before this token is issued.
    token = authorize_ledgered_final_test(
        ledger_path,
        plan,
        protocol=protocol,
        purpose=purpose,
        confirmation=confirmation,
    )
    subgroup_ids, subgroups = _load_subgroups(settings.subgroup_path)
    refits = {member.member_id: member for member in refit_bundle.members}
    calibrations = {
        member.member_id: member for member in calibration_bundle.members
    }
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
            prediction = _export_or_resume_prediction(
                refit,
                calibration,
                prediction_path,
                state=state,
                settings=settings,
                protocol=protocol,
                test_access=token,
                exporter=exporter,
            )
            if state["state"] == "planned":
                ledger = _record_prediction(
                    Path(ledger_path), ledger, member_id, prediction_path, prediction
                )
                state = ledger.members[member_id]
            if state["state"] == "report_saved":
                verify_final_report(report_path, protocol=protocol, test_access=token)
                continue
            if report_path.exists():
                existing_report = verify_final_report(
                    report_path, protocol=protocol, test_access=token
                )
                ledger = _record_report(
                    Path(ledger_path),
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
                Path(ledger_path), ledger, member_id, saved.sha256
            )
        outputs = _build_aggregate_outputs(
            plan,
            calibration_bundle,
            protocol=protocol,
            test_access=token,
        )
        ledger = _complete_ledger(Path(ledger_path), ledger, outputs)
    except Exception as error:
        _record_failure(Path(ledger_path), ledger, error)
        raise
    architecture_paths = {
        architecture: Path(
            _string(outputs[f"architecture_{architecture}_path"], "architecture path")
        )
        for architecture in EXPECTED_ARCHITECTURES
    }
    return FinalBatchResult(
        ledger_path=Path(ledger_path),
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
    protocol: ExperimentProtocol,
    test_access: FinalTestAccessToken,
    exporter: PredictionExporter,
) -> PredictionArtifact:
    if state["state"] == "planned" and not prediction_path.exists():
        result = exporter(
            PredictionExportRequest(
                checkpoint_path=refit.final_checkpoint_path,
                resolved_config_path=refit.resolved_config_path,
                run_metadata_path=refit.metadata_path,
                output_path=prediction_path,
                fold_role=FoldRole.FINAL_TEST,
                batch_size=settings.batch_size,
                num_workers=settings.num_workers,
                device=settings.device,
                bf16=settings.bf16,
            ),
            protocol=protocol,
            test_access=test_access,
        )
        if result.fold_role is not FoldRole.FINAL_TEST or result.folds != FINAL_TEST_FOLDS:
            raise ReleaseIntegrityError("final exporter returned the wrong fold role")
    prediction = load_prediction_artifact(
        prediction_path,
        protocol=protocol,
        test_access=test_access,
        expected_config_hash=refit.resolved_config_hash,
        expected_manifest_hash=refit.manifest_sha256,
    )
    _validate_final_prediction(prediction, refit, calibration)
    expected_artifact = state.get("final_prediction_artifact_sha256")
    if expected_artifact is not None and expected_artifact != prediction.integrity_sha256:
        raise ReleaseIntegrityError("resumed final prediction hash differs from ledger")
    expected_file = state.get("final_prediction_file_sha256")
    if expected_file is not None and expected_file != sha256_file(prediction_path):
        raise ReleaseIntegrityError("resumed final prediction file differs from ledger")
    return prediction


def _validate_final_prediction(
    prediction: PredictionArtifact,
    refit: RefitMember,
    calibration: CalibrationMember,
) -> None:
    if prediction.fold_role is not FoldRole.FINAL_TEST or prediction.folds != FINAL_TEST_FOLDS:
        raise ReleaseIntegrityError("final prediction must contain fold 10 only")
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


def _build_aggregate_outputs(
    plan: FinalBatchPlan,
    calibration_bundle: CalibrationBundle,
    *,
    protocol: ExperimentProtocol,
    test_access: FinalTestAccessToken,
) -> dict[str, object]:
    destination = plan.settings.output_directory
    destination.mkdir(parents=True, exist_ok=True)
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
    """Reserve one exclusive writer for a ledger mutation."""

    lock_path = path.with_name(path.name + ".writer.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise ReleaseStateError(
            f"opening ledger already has an active writer: {lock_path}"
        ) from error
    try:
        os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
        os.fsync(descriptor)
        yield
    finally:
        os.close(descriptor)
        with suppress(OSError):
            lock_path.unlink()


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
        "final_prediction_path",
        "final_report_path",
    }
    for member in members:
        _exact_keys(member, required, "final batch member")
        architecture = _string(member["architecture"], "architecture")
        seed = _positive_int(member["seed"], "seed", minimum=0)
        member_id = _string(member["member_id"], "member_id")
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
    if value == "report_saved":
        _hash_string(state["final_report_sha256"], "final_report_sha256")


def _validate_pair(refit: RefitMember, calibration: CalibrationMember) -> None:
    expected = {
        "member_id": refit.member_id,
        "architecture": refit.architecture,
        "seed": refit.seed,
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
        if not replace and path.exists():
            raise FileExistsError(f"opening ledger already exists: {path}")
        os.replace(temporary, path)
    finally:
        with suppress(OSError):
            temporary.unlink(missing_ok=True)


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
    "create_final_opening_ledger",
    "load_final_opening_ledger",
    "open_or_resume_final_batch",
    "run_final_batch",
]
