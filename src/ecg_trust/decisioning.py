"""Leakage-safe calibration decisions and locked final-report orchestration."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ecg_trust import __version__
from ecg_trust.audit import (
    PatientClusterBootstrapResult,
    SubgroupAuditEntry,
    SubgroupAuditResult,
    SubgroupSelectiveCoverage,
    audit_subgroups,
    bootstrap_multilabel_metrics,
)
from ecg_trust.evaluation import (
    MultilabelMetrics,
    PerLabelThreshold,
    SelectiveCoveragePoint,
    SelectivePredictionResult,
    TemperatureScalingResult,
    ThresholdOptimizationResult,
    compute_multilabel_metrics,
    fit_temperature_scaling,
    optimize_thresholds,
)
from ecg_trust.predictions import PredictionArtifact
from ecg_trust.protocol import (
    CALIBRATION_FOLDS,
    FINAL_TEST_FOLDS,
    LABEL_ORDER,
    ExperimentProtocol,
    FinalTestAccessToken,
    FoldRole,
)

DECISION_SCHEMA_VERSION = 1
DECISION_ARTIFACT_TYPE = "ecg_trust.calibration_decisions"
FINAL_REPORT_SCHEMA_VERSION = 1
FINAL_REPORT_TYPE = "ecg_trust.final_evaluation_report"
ENTROPY_METHOD = "mean_normalized_binary_entropy"

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
MetadataScalar = str | int | float | bool | None


class DecisioningError(ValueError):
    """Raised when calibration or final-report contracts are violated."""


class DecisionIntegrityError(DecisioningError):
    """Raised when an integrity-bound JSON artifact is inconsistent."""


class FinalReportProvenanceError(DecisioningError):
    """Raised when calibration and final prediction provenance do not match."""


@dataclass(frozen=True, slots=True)
class JsonArtifactFile:
    """Path and SHA-256 returned after an immutable JSON commit."""

    path: Path
    sha256: str

    def to_dict(self) -> dict[str, object]:
        return {"path": str(self.path), "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class CoverageGate:
    """Entropy cutoff learned on fold 9 for one requested coverage."""

    target_coverage: float
    maximum_entropy: float
    calibration_coverage: float
    selected_count: int
    calibration_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "target_coverage": self.target_coverage,
            "maximum_entropy": self.maximum_entropy,
            "calibration_coverage": self.calibration_coverage,
            "selected_count": self.selected_count,
            "calibration_count": self.calibration_count,
        }


@dataclass(frozen=True, slots=True)
class CalibrationDecisionArtifact:
    """Frozen fold-9 transforms and their exact source provenance."""

    model_name: str
    model_seed: int
    protocol_hash: str
    config_hash: str
    manifest_hash: str
    label_order: tuple[str, ...]
    source_prediction_sha256: str
    source_alignment_sha256: str
    temperature_scaling: TemperatureScalingResult
    threshold_optimization: ThresholdOptimizationResult
    coverage_gates: tuple[CoverageGate, ...]
    created_at_utc: str
    software_versions: Mapping[str, str]
    integrity_sha256: str | None

    def to_payload(self, *, include_integrity: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": DECISION_SCHEMA_VERSION,
            "artifact_type": DECISION_ARTIFACT_TYPE,
            "model": {"name": self.model_name, "seed": self.model_seed},
            "protocol_hash": self.protocol_hash,
            "config_hash": self.config_hash,
            "manifest_hash": self.manifest_hash,
            "label_order": list(self.label_order),
            "source": {
                "fold_role": FoldRole.CALIBRATION.value,
                "folds": list(CALIBRATION_FOLDS),
                "prediction_sha256": self.source_prediction_sha256,
                "alignment_sha256": self.source_alignment_sha256,
            },
            "temperature_scaling": self.temperature_scaling.to_dict(),
            "threshold_optimization": self.threshold_optimization.to_dict(),
            "selective_gates": {
                "uncertainty_method": ENTROPY_METHOD,
                "gates": [gate.to_dict() for gate in self.coverage_gates],
            },
            "created": {
                "timestamp_utc": self.created_at_utc,
                "software_versions": dict(self.software_versions),
            },
        }
        if include_integrity and self.integrity_sha256 is not None:
            payload["artifact_sha256"] = self.integrity_sha256
        return payload


@dataclass(frozen=True, slots=True)
class FinalEvaluationReport:
    """Final fold-10 results obtained with only frozen fold-9 decisions."""

    model_name: str
    model_seed: int
    protocol_hash: str
    config_hash: str
    manifest_hash: str
    label_order: tuple[str, ...]
    calibration_artifact_sha256: str
    final_prediction_sha256: str
    final_alignment_sha256: str
    applied_temperature: float
    applied_thresholds: tuple[float, ...]
    applied_coverage_gates: tuple[CoverageGate, ...]
    metrics: MultilabelMetrics
    selective_prediction: SelectivePredictionResult
    subgroup_audit: SubgroupAuditResult
    patient_bootstrap: PatientClusterBootstrapResult
    created_at_utc: str

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": FINAL_REPORT_SCHEMA_VERSION,
            "report_type": FINAL_REPORT_TYPE,
            "model": {"name": self.model_name, "seed": self.model_seed},
            "protocol_hash": self.protocol_hash,
            "config_hash": self.config_hash,
            "manifest_hash": self.manifest_hash,
            "label_order": list(self.label_order),
            "sources": {
                "calibration_artifact_sha256": self.calibration_artifact_sha256,
                "final_prediction_sha256": self.final_prediction_sha256,
                "final_alignment_sha256": self.final_alignment_sha256,
                "final_fold_role": FoldRole.FINAL_TEST.value,
                "final_folds": list(FINAL_TEST_FOLDS),
            },
            "frozen_decisions": {
                "temperature": self.applied_temperature,
                "thresholds": list(self.applied_thresholds),
                "uncertainty_method": ENTROPY_METHOD,
                "coverage_gates": [
                    gate.to_dict() for gate in self.applied_coverage_gates
                ],
            },
            "metrics": self.metrics.to_dict(),
            "selective_prediction": self.selective_prediction.to_dict(),
            "subgroup_audit": self.subgroup_audit.to_dict(),
            "patient_bootstrap": self.patient_bootstrap.to_dict(),
            "created": {
                "timestamp_utc": self.created_at_utc,
                "software_versions": _software_versions(),
            },
        }


def fit_calibration_decisions(
    prediction: PredictionArtifact,
    *,
    protocol: ExperimentProtocol,
    coverage_targets: Sequence[float] = (1.0, 0.9, 0.8, 0.7, 0.5),
    created_at_utc: str | None = None,
) -> CalibrationDecisionArtifact:
    """Fit all post-processing choices on an integrity-checked fold-9 artifact."""

    _validate_calibration_prediction(prediction, protocol)
    temperature = fit_temperature_scaling(
        logits=prediction.raw_logits,
        y_true=prediction.targets,
        calibration_fold_ids=prediction.strat_fold,
        label_order=prediction.label_order,
    )
    probabilities = temperature.predict_proba(
        prediction.raw_logits, label_order=prediction.label_order
    )
    thresholds = optimize_thresholds(
        y_true=prediction.targets,
        probabilities=probabilities,
        calibration_fold_ids=prediction.strat_fold,
        label_order=prediction.label_order,
    )
    gates = _fit_coverage_gates(probabilities, coverage_targets)
    return CalibrationDecisionArtifact(
        model_name=prediction.model_name,
        model_seed=prediction.model_seed,
        protocol_hash=prediction.protocol_hash,
        config_hash=prediction.config_hash,
        manifest_hash=prediction.manifest_hash,
        label_order=prediction.label_order,
        source_prediction_sha256=_required_integrity(
            prediction.integrity_sha256, "calibration prediction"
        ),
        source_alignment_sha256=prediction.alignment_sha256,
        temperature_scaling=temperature,
        threshold_optimization=thresholds,
        coverage_gates=gates,
        created_at_utc=_timestamp(created_at_utc),
        software_versions=MappingProxyType(_software_versions()),
        integrity_sha256=None,
    )


def save_calibration_decisions(
    artifact: CalibrationDecisionArtifact,
    path: str | Path,
) -> JsonArtifactFile:
    """Integrity-bind and atomically save a new calibration JSON artifact."""

    destination = _json_destination(path)
    payload = artifact.to_payload(include_integrity=False)
    digest = _payload_hash(payload)
    payload["artifact_sha256"] = digest
    _write_new_json(destination, payload)
    return JsonArtifactFile(path=destination, sha256=digest)


def load_calibration_decisions(
    path: str | Path,
    *,
    protocol: ExperimentProtocol,
) -> CalibrationDecisionArtifact:
    """Load and verify a fold-9 calibration decision artifact."""

    payload = _read_json(Path(path), context="calibration decision artifact")
    _expect_keys(
        payload,
        required={
            "schema_version",
            "artifact_type",
            "model",
            "protocol_hash",
            "config_hash",
            "manifest_hash",
            "label_order",
            "source",
            "temperature_scaling",
            "threshold_optimization",
            "selective_gates",
            "created",
            "artifact_sha256",
        },
        context="calibration decision artifact",
    )
    if _integer(payload["schema_version"], "schema_version", minimum=1) != 1:
        raise DecisioningError("unsupported calibration decision schema_version")
    if _string(payload["artifact_type"], "artifact_type") != DECISION_ARTIFACT_TYPE:
        raise DecisioningError("unexpected calibration decision artifact_type")
    stored_hash = _hash_string(payload["artifact_sha256"], "artifact_sha256")
    unhashed_payload = dict(payload)
    del unhashed_payload["artifact_sha256"]
    if stored_hash != _payload_hash(unhashed_payload):
        raise DecisionIntegrityError("calibration decision artifact SHA-256 mismatch")

    protocol_hash = _hash_string(payload["protocol_hash"], "protocol_hash")
    if protocol_hash != protocol.protocol_hash:
        raise DecisionIntegrityError(
            "calibration artifact protocol_hash does not match supplied protocol"
        )
    model = _mapping(payload["model"], "model")
    _expect_keys(model, required={"name", "seed"}, context="model")
    source = _mapping(payload["source"], "source")
    _expect_keys(
        source,
        required={"fold_role", "folds", "prediction_sha256", "alignment_sha256"},
        context="source",
    )
    if _string(source["fold_role"], "source.fold_role") != FoldRole.CALIBRATION.value:
        raise DecisionIntegrityError("calibration source fold_role must be calibration")
    if _integer_tuple(source["folds"], "source.folds") != CALIBRATION_FOLDS:
        raise DecisionIntegrityError("calibration source must contain fold 9 only")

    temperature = _parse_temperature_result(payload["temperature_scaling"])
    thresholds = _parse_threshold_result(payload["threshold_optimization"])
    if temperature.n_samples != thresholds.n_samples:
        raise DecisionIntegrityError(
            "temperature and threshold fits have different calibration sample counts"
        )
    gates_mapping = _mapping(payload["selective_gates"], "selective_gates")
    _expect_keys(
        gates_mapping,
        required={"uncertainty_method", "gates"},
        context="selective_gates",
    )
    if _string(
        gates_mapping["uncertainty_method"], "selective_gates.uncertainty_method"
    ) != ENTROPY_METHOD:
        raise DecisionIntegrityError("unsupported selective uncertainty method")
    gates = _parse_gates(
        gates_mapping["gates"], expected_count=temperature.n_samples
    )
    labels = _label_order(payload["label_order"])
    if temperature.label_order != labels or thresholds.label_order != labels:
        raise DecisionIntegrityError("calibration transform label orders do not align")
    created = _mapping(payload["created"], "created")
    _expect_keys(
        created,
        required={"timestamp_utc", "software_versions"},
        context="created",
    )
    return CalibrationDecisionArtifact(
        model_name=_string(model["name"], "model.name"),
        model_seed=_integer(model["seed"], "model.seed", minimum=0),
        protocol_hash=protocol_hash,
        config_hash=_hash_string(payload["config_hash"], "config_hash"),
        manifest_hash=_hash_string(payload["manifest_hash"], "manifest_hash"),
        label_order=labels,
        source_prediction_sha256=_hash_string(
            source["prediction_sha256"], "source.prediction_sha256"
        ),
        source_alignment_sha256=_hash_string(
            source["alignment_sha256"], "source.alignment_sha256"
        ),
        temperature_scaling=temperature,
        threshold_optimization=thresholds,
        coverage_gates=gates,
        created_at_utc=_timestamp(
            _string(created["timestamp_utc"], "created.timestamp_utc")
        ),
        software_versions=MappingProxyType(
            _string_mapping(created["software_versions"], "created.software_versions")
        ),
        integrity_sha256=stored_hash,
    )


def generate_final_report(
    decisions: CalibrationDecisionArtifact,
    final_prediction: PredictionArtifact,
    *,
    protocol: ExperimentProtocol,
    test_access: FinalTestAccessToken,
    subgroup_ecg_id: ArrayLike,
    subgroups: Mapping[str, ArrayLike],
    bootstrap_resamples: int = 1_000,
    bootstrap_seed: int = 20_260_808,
    bootstrap_confidence: float = 0.95,
    bootstrap_minimum_valid: int | None = None,
    minimum_group_samples: int = 30,
    minimum_group_patients: int = 20,
    ece_bins: int = 15,
    created_at_utc: str | None = None,
) -> FinalEvaluationReport:
    """Evaluate fold 10 once with frozen fold-9 transforms and gates.

    No fitting or threshold selection occurs in this function. The explicit
    protocol token is checked again even if the prediction artifact was already
    loaded through a guarded API.
    """

    protocol.folds_for(FoldRole.FINAL_TEST, test_access=test_access)
    _validate_final_provenance(decisions, final_prediction, protocol)
    _validate_subgroup_alignment(subgroup_ecg_id, final_prediction.ecg_id)

    probabilities = decisions.temperature_scaling.predict_proba(
        final_prediction.raw_logits, label_order=final_prediction.label_order
    )
    metrics = compute_multilabel_metrics(
        final_prediction.targets,
        probabilities,
        label_order=final_prediction.label_order,
        ece_bins=ece_bins,
    )
    selective = _apply_frozen_gates(
        final_prediction.targets,
        probabilities,
        thresholds=decisions.threshold_optimization.thresholds,
        gates=decisions.coverage_gates,
    )
    subgroup_audit = _frozen_subgroup_audit(
        final_prediction,
        probabilities,
        subgroups=subgroups,
        selective=selective,
        minimum_group_samples=minimum_group_samples,
        minimum_group_patients=minimum_group_patients,
        ece_bins=ece_bins,
    )
    bootstrap = bootstrap_multilabel_metrics(
        final_prediction.targets,
        probabilities,
        final_prediction.patient_id,
        n_resamples=bootstrap_resamples,
        seed=bootstrap_seed,
        confidence_level=bootstrap_confidence,
        minimum_valid_resamples=bootstrap_minimum_valid,
        label_order=final_prediction.label_order,
        ece_bins=ece_bins,
    )
    return FinalEvaluationReport(
        model_name=final_prediction.model_name,
        model_seed=final_prediction.model_seed,
        protocol_hash=final_prediction.protocol_hash,
        config_hash=final_prediction.config_hash,
        manifest_hash=final_prediction.manifest_hash,
        label_order=final_prediction.label_order,
        calibration_artifact_sha256=_required_integrity(
            decisions.integrity_sha256, "calibration decision artifact"
        ),
        final_prediction_sha256=_required_integrity(
            final_prediction.integrity_sha256, "final prediction artifact"
        ),
        final_alignment_sha256=final_prediction.alignment_sha256,
        applied_temperature=decisions.temperature_scaling.temperature,
        applied_thresholds=decisions.threshold_optimization.thresholds,
        applied_coverage_gates=decisions.coverage_gates,
        metrics=metrics,
        selective_prediction=selective,
        subgroup_audit=subgroup_audit,
        patient_bootstrap=bootstrap,
        created_at_utc=_timestamp(created_at_utc),
    )


def save_final_report(
    report: FinalEvaluationReport,
    path: str | Path,
    *,
    protocol: ExperimentProtocol,
    test_access: FinalTestAccessToken,
) -> JsonArtifactFile:
    """Atomically write a non-overwriting, integrity-bound final report."""

    protocol.folds_for(FoldRole.FINAL_TEST, test_access=test_access)
    if report.protocol_hash != protocol.protocol_hash:
        raise FinalReportProvenanceError("final report protocol_hash mismatch")
    destination = _json_destination(path)
    payload = report.to_payload()
    digest = _payload_hash(payload)
    payload["report_sha256"] = digest
    _write_new_json(destination, payload)
    return JsonArtifactFile(path=destination, sha256=digest)


def verify_final_report(
    path: str | Path,
    *,
    protocol: ExperimentProtocol,
    test_access: FinalTestAccessToken,
) -> Mapping[str, object]:
    """Verify a saved final report and require explicit fold-10 authorization."""

    protocol.folds_for(FoldRole.FINAL_TEST, test_access=test_access)
    payload = _read_json(Path(path), context="final evaluation report")
    _expect_keys(
        payload,
        required={
            "schema_version",
            "report_type",
            "model",
            "protocol_hash",
            "config_hash",
            "manifest_hash",
            "label_order",
            "sources",
            "frozen_decisions",
            "metrics",
            "selective_prediction",
            "subgroup_audit",
            "patient_bootstrap",
            "created",
            "report_sha256",
        },
        context="final evaluation report",
    )
    if _integer(payload.get("schema_version"), "schema_version", minimum=1) != 1:
        raise DecisionIntegrityError("unsupported final report schema_version")
    if _string(payload.get("report_type"), "report_type") != FINAL_REPORT_TYPE:
        raise DecisionIntegrityError("unexpected final report type")
    if _hash_string(payload.get("protocol_hash"), "protocol_hash") != protocol.protocol_hash:
        raise DecisionIntegrityError("final report protocol_hash mismatch")
    stored = _hash_string(payload.get("report_sha256"), "report_sha256")
    unhashed = dict(payload)
    del unhashed["report_sha256"]
    if stored != _payload_hash(unhashed):
        raise DecisionIntegrityError("final report SHA-256 mismatch")
    return payload


def _validate_calibration_prediction(
    prediction: PredictionArtifact,
    protocol: ExperimentProtocol,
) -> None:
    if prediction.protocol_hash != protocol.protocol_hash:
        raise DecisioningError("calibration prediction protocol_hash mismatch")
    if prediction.fold_role is not FoldRole.CALIBRATION or prediction.folds != (9,):
        raise DecisioningError("calibration fitting requires a fold-9-only prediction artifact")
    protocol.folds_for(FoldRole.CALIBRATION)
    _required_integrity(prediction.integrity_sha256, "calibration prediction")


def _validate_final_provenance(
    decisions: CalibrationDecisionArtifact,
    prediction: PredictionArtifact,
    protocol: ExperimentProtocol,
) -> None:
    if decisions.integrity_sha256 is None:
        raise FinalReportProvenanceError(
            "calibration decisions must be loaded from an integrity-bound artifact"
        )
    if prediction.integrity_sha256 is None:
        raise FinalReportProvenanceError(
            "final predictions must be loaded from an integrity-bound artifact"
        )
    if prediction.fold_role is not FoldRole.FINAL_TEST or prediction.folds != (10,):
        raise FinalReportProvenanceError(
            "final reporting requires a fold-10-only prediction artifact"
        )
    expected: dict[str, object] = {
        "model_name": decisions.model_name,
        "model_seed": decisions.model_seed,
        "protocol_hash": decisions.protocol_hash,
        "config_hash": decisions.config_hash,
        "manifest_hash": decisions.manifest_hash,
        "label_order": decisions.label_order,
    }
    observed: dict[str, object] = {
        "model_name": prediction.model_name,
        "model_seed": prediction.model_seed,
        "protocol_hash": prediction.protocol_hash,
        "config_hash": prediction.config_hash,
        "manifest_hash": prediction.manifest_hash,
        "label_order": prediction.label_order,
    }
    mismatches = [key for key in expected if expected[key] != observed[key]]
    if mismatches:
        raise FinalReportProvenanceError(
            "calibration/final provenance mismatch: " + ", ".join(mismatches)
        )
    if prediction.protocol_hash != protocol.protocol_hash:
        raise FinalReportProvenanceError("final prediction protocol_hash mismatch")


def _fit_coverage_gates(
    probabilities: FloatArray,
    coverage_targets: Sequence[float],
) -> tuple[CoverageGate, ...]:
    targets = _coverage_targets(coverage_targets)
    uncertainty = _mean_binary_entropy(probabilities)
    sorted_uncertainty = np.sort(uncertainty, kind="stable")
    total = int(uncertainty.size)
    gates: list[CoverageGate] = []
    for target in targets:
        requested_count = min(total, math.ceil(target * total))
        cutoff = (
            1.0
            if target == 1.0
            else float(sorted_uncertainty[requested_count - 1])
        )
        selected_count = int(np.sum(uncertainty <= cutoff))
        gates.append(
            CoverageGate(
                target_coverage=target,
                maximum_entropy=cutoff,
                calibration_coverage=float(selected_count / total),
                selected_count=selected_count,
                calibration_count=total,
            )
        )
    return tuple(gates)


def _apply_frozen_gates(
    targets: NDArray[np.generic],
    probabilities: FloatArray,
    *,
    thresholds: Sequence[float],
    gates: Sequence[CoverageGate],
) -> SelectivePredictionResult:
    threshold_array = np.asarray(thresholds, dtype=np.float64)
    predictions = probabilities >= threshold_array[None, :]
    binary_targets = np.asarray(targets, dtype=np.bool_)
    errors = predictions != binary_targets
    uncertainty = _mean_binary_entropy(probabilities)
    all_indices = np.arange(probabilities.shape[0], dtype=np.int64)
    points: list[SelectiveCoveragePoint] = []
    for gate in gates:
        selected = np.flatnonzero(uncertainty <= gate.maximum_entropy)
        selected_mask = np.zeros(probabilities.shape[0], dtype=np.bool_)
        selected_mask[selected] = True
        abstained = all_indices[~selected_mask]
        if selected.size:
            selected_errors = errors[selected]
            hamming: float | None = float(selected_errors.mean())
            exact: float | None = float((~selected_errors.any(axis=1)).mean())
            per_label: tuple[float | None, ...] = tuple(
                float(value) for value in selected_errors.mean(axis=0)
            )
            mean_uncertainty: float | None = float(uncertainty[selected].mean())
        else:
            hamming = None
            exact = None
            per_label = tuple(None for _ in LABEL_ORDER)
            mean_uncertainty = None
        points.append(
            SelectiveCoveragePoint(
                target_coverage=gate.target_coverage,
                achieved_coverage=float(selected.size / probabilities.shape[0]),
                selected_count=int(selected.size),
                abstained_count=int(abstained.size),
                hamming_risk=hamming,
                exact_match_accuracy=exact,
                per_label_error_rate=per_label,
                mean_selected_uncertainty=mean_uncertainty,
                selected_indices=tuple(int(index) for index in selected),
                abstained_indices=tuple(int(index) for index in abstained),
            )
        )
    return SelectivePredictionResult(
        n_samples=probabilities.shape[0],
        label_order=LABEL_ORDER,
        thresholds=tuple(float(value) for value in threshold_array),
        uncertainty_method=ENTROPY_METHOD,
        coverage_points=tuple(points),
    )


def _frozen_subgroup_audit(
    prediction: PredictionArtifact,
    probabilities: FloatArray,
    *,
    subgroups: Mapping[str, ArrayLike],
    selective: SelectivePredictionResult,
    minimum_group_samples: int,
    minimum_group_patients: int,
    ece_bins: int,
) -> SubgroupAuditResult:
    # A full-coverage call obtains the established group validation, prevalence,
    # discrimination, calibration, and small-group status without choosing any
    # final-set abstention cutoff.
    base = audit_subgroups(
        prediction.targets,
        probabilities,
        prediction.patient_id,
        subgroups,
        thresholds=selective.thresholds,
        coverage_targets=(1.0,),
        minimum_group_samples=minimum_group_samples,
        minimum_group_patients=minimum_group_patients,
        label_order=prediction.label_order,
        ece_bins=ece_bins,
    )
    normalized_groups = _normalized_group_arrays(subgroups, prediction.n_samples)
    predictions = probabilities >= np.asarray(selective.thresholds)[None, :]
    errors = predictions != prediction.targets.astype(np.bool_, copy=False)
    entries: list[SubgroupAuditEntry] = []
    for entry in base.groups:
        keys = normalized_groups[entry.attribute]
        member_indices = np.asarray(
            [
                index
                for index, key in enumerate(keys)
                if key == (entry.group_value_type, entry.group_value)
            ],
            dtype=np.int64,
        )
        member_mask = np.zeros(prediction.n_samples, dtype=np.bool_)
        member_mask[member_indices] = True
        group_points: list[SubgroupSelectiveCoverage] = []
        for global_point in selective.coverage_points:
            global_mask = np.zeros(prediction.n_samples, dtype=np.bool_)
            global_mask[list(global_point.selected_indices)] = True
            selected_members = np.flatnonzero(member_mask & global_mask)
            if selected_members.size:
                selected_errors = errors[selected_members]
                risk: float | None = float(selected_errors.mean())
                exact: float | None = float((~selected_errors.any(axis=1)).mean())
                status = "ok"
            else:
                risk = None
                exact = None
                status = "no_selected_samples"
            group_points.append(
                SubgroupSelectiveCoverage(
                    target_global_coverage=global_point.target_coverage,
                    achieved_global_coverage=global_point.achieved_coverage,
                    subgroup_coverage=float(selected_members.size / member_indices.size),
                    selected_count=int(selected_members.size),
                    abstained_count=int(member_indices.size - selected_members.size),
                    hamming_risk=risk,
                    exact_match_accuracy=exact,
                    status=status,
                )
            )
        entries.append(
            SubgroupAuditEntry(
                attribute=entry.attribute,
                group_value=entry.group_value,
                group_value_type=entry.group_value_type,
                n_samples=entry.n_samples,
                n_patients=entry.n_patients,
                status=entry.status,
                metrics=entry.metrics,
                selective_coverage=tuple(group_points),
            )
        )
    return SubgroupAuditResult(
        n_samples=base.n_samples,
        n_patients=base.n_patients,
        label_order=base.label_order,
        ece_bins=base.ece_bins,
        minimum_group_samples=base.minimum_group_samples,
        minimum_group_patients=base.minimum_group_patients,
        overall_metrics=base.overall_metrics,
        global_selective_prediction=selective,
        groups=tuple(entries),
    )


def _mean_binary_entropy(probabilities: FloatArray) -> FloatArray:
    epsilon = np.finfo(np.float64).eps
    clipped = np.clip(probabilities, epsilon, 1.0 - epsilon)
    entropy = -(
        clipped * np.log(clipped) + (1.0 - clipped) * np.log(1.0 - clipped)
    ) / math.log(2.0)
    return entropy.mean(axis=1)


def _coverage_targets(values: Sequence[float]) -> tuple[float, ...]:
    targets = tuple(float(value) for value in values)
    if not targets:
        raise DecisioningError("at least one positive coverage target is required")
    if any(not math.isfinite(value) or not 0.0 < value <= 1.0 for value in targets):
        raise DecisioningError("coverage targets must be finite and in (0, 1]")
    if len(set(targets)) != len(targets):
        raise DecisioningError("coverage targets must not contain duplicates")
    return targets


def _validate_subgroup_alignment(expected_ids: ArrayLike, observed_ids: ArrayLike) -> None:
    expected = np.asarray(expected_ids)
    observed = np.asarray(observed_ids)
    if expected.ndim != 1 or expected.shape != observed.shape:
        raise FinalReportProvenanceError(
            "subgroup ecg_id must align one-to-one with final predictions"
        )
    expected_kind = "string" if expected.dtype.kind in {"U", "S"} else "numeric"
    observed_kind = "string" if observed.dtype.kind in {"U", "S"} else "numeric"
    if expected_kind != observed_kind or not np.array_equal(expected, observed):
        raise FinalReportProvenanceError(
            "subgroup ecg_id order does not match final prediction alignment"
        )


def _normalized_group_arrays(
    subgroups: Mapping[str, ArrayLike],
    n_samples: int,
) -> dict[str, tuple[tuple[str, str], ...]]:
    normalized: dict[str, tuple[tuple[str, str], ...]] = {}
    for raw_attribute, values in subgroups.items():
        attribute = raw_attribute.strip()
        if not attribute:
            raise DecisioningError("subgroup names must be non-empty")
        array = np.asarray(values, dtype=object)
        if array.ndim != 1 or array.shape[0] != n_samples:
            raise DecisioningError(
                f"subgroup {attribute!r} must have one value per final prediction"
            )
        normalized[attribute] = tuple(_normalize_group_value(value) for value in array)
    return normalized


def _normalize_group_value(value: object) -> tuple[str, str]:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ("missing", "<MISSING>")
    if isinstance(value, bool):
        return ("boolean", "true" if value else "false")
    if isinstance(value, int):
        return ("integer", str(value))
    if isinstance(value, float):
        if not math.isfinite(value):
            raise DecisioningError("subgroup values cannot contain infinities")
        return ("number", repr(value))
    if isinstance(value, str):
        return ("string", value)
    raise DecisioningError("subgroup values must be strings, numbers, booleans, or missing")


def _parse_temperature_result(value: object) -> TemperatureScalingResult:
    raw = _mapping(value, "temperature_scaling")
    required = {
        "temperature",
        "label_order",
        "n_samples",
        "source_folds",
        "fitted_labels",
        "excluded_degenerate_labels",
        "nll_before",
        "nll_after",
        "status",
        "converged",
        "optimization_steps",
        "temperature_bounds",
    }
    _expect_keys(raw, required=required, context="temperature_scaling")
    labels = _label_order(raw["label_order"])
    source_folds = _integer_tuple(raw["source_folds"], "temperature.source_folds")
    if source_folds != CALIBRATION_FOLDS:
        raise DecisionIntegrityError("temperature scaling must originate from fold 9")
    temperature = _finite_float(raw["temperature"], "temperature", minimum=0.0)
    if temperature <= 0.0:
        raise DecisionIntegrityError("temperature must be positive")
    bounds_values = _float_tuple(raw["temperature_bounds"], "temperature_bounds")
    if len(bounds_values) != 2 or not 0 < bounds_values[0] <= 1 <= bounds_values[1]:
        raise DecisionIntegrityError("invalid temperature bounds")
    fitted = _string_tuple(raw["fitted_labels"], "fitted_labels")
    excluded = _string_tuple(
        raw["excluded_degenerate_labels"], "excluded_degenerate_labels"
    )
    if tuple(label for label in labels if label in fitted) != fitted:
        raise DecisionIntegrityError("fitted temperature labels are out of order")
    if set(fitted).intersection(excluded) or set(fitted).union(excluded) != set(labels):
        raise DecisionIntegrityError("fitted/excluded temperature labels are inconsistent")
    return TemperatureScalingResult(
        temperature=temperature,
        label_order=labels,
        n_samples=_integer(raw["n_samples"], "temperature.n_samples", minimum=1),
        source_folds=source_folds,
        fitted_labels=fitted,
        excluded_degenerate_labels=excluded,
        nll_before=_optional_float(raw["nll_before"], "nll_before"),
        nll_after=_optional_float(raw["nll_after"], "nll_after"),
        status=_string(raw["status"], "temperature.status"),
        converged=_boolean(raw["converged"], "temperature.converged"),
        optimization_steps=_integer(
            raw["optimization_steps"], "temperature.optimization_steps", minimum=0
        ),
        temperature_bounds=(bounds_values[0], bounds_values[1]),
    )


def _parse_threshold_result(value: object) -> ThresholdOptimizationResult:
    raw = _mapping(value, "threshold_optimization")
    required = {
        "label_order",
        "thresholds",
        "objective",
        "macro_objective",
        "default_threshold",
        "n_samples",
        "source_folds",
        "per_label",
    }
    _expect_keys(raw, required=required, context="threshold_optimization")
    labels = _label_order(raw["label_order"])
    thresholds = _float_tuple(raw["thresholds"], "thresholds")
    if len(thresholds) != len(labels) or any(not 0 <= value <= 1 for value in thresholds):
        raise DecisionIntegrityError("thresholds must contain five values in [0, 1]")
    if _string(raw["objective"], "threshold.objective") != "f1":
        raise DecisionIntegrityError("threshold objective must be f1")
    source_folds = _integer_tuple(raw["source_folds"], "threshold.source_folds")
    if source_folds != CALIBRATION_FOLDS:
        raise DecisionIntegrityError("threshold optimization must originate from fold 9")
    n_samples = _integer(raw["n_samples"], "threshold.n_samples", minimum=1)
    per_label_values = raw["per_label"]
    if not isinstance(per_label_values, list) or len(per_label_values) != len(labels):
        raise DecisionIntegrityError("threshold per_label must contain five entries")
    per_label: list[PerLabelThreshold] = []
    for index, item in enumerate(per_label_values):
        entry = _mapping(item, f"threshold.per_label[{index}]")
        _expect_keys(
            entry,
            required={
                "label",
                "threshold",
                "objective",
                "objective_value",
                "positives",
                "negatives",
                "status",
            },
            context=f"threshold.per_label[{index}]",
        )
        label = _string(entry["label"], "threshold label")
        threshold = _finite_float(entry["threshold"], "threshold", minimum=0.0)
        positives = _integer(entry["positives"], "positives", minimum=0)
        negatives = _integer(entry["negatives"], "negatives", minimum=0)
        if label != labels[index] or threshold != thresholds[index]:
            raise DecisionIntegrityError("per-label threshold alignment mismatch")
        if positives + negatives != n_samples:
            raise DecisionIntegrityError("per-label threshold counts do not sum to n_samples")
        per_label.append(
            PerLabelThreshold(
                label=label,
                threshold=threshold,
                objective=_string(entry["objective"], "per-label objective"),
                objective_value=_optional_float(
                    entry["objective_value"], "objective_value"
                ),
                positives=positives,
                negatives=negatives,
                status=_string(entry["status"], "threshold status"),
            )
        )
    macro_objective = _optional_float(raw["macro_objective"], "macro_objective")
    default_threshold = _finite_float(
        raw["default_threshold"], "default_threshold", minimum=0.0
    )
    if macro_objective is not None and macro_objective > 1.0:
        raise DecisionIntegrityError("macro threshold objective must be in [0, 1]")
    if default_threshold > 1.0:
        raise DecisionIntegrityError("default threshold must be in [0, 1]")
    if any(item.objective != "f1" for item in per_label):
        raise DecisionIntegrityError("every per-label threshold objective must be f1")
    if any(
        item.objective_value is not None and item.objective_value > 1.0
        for item in per_label
    ):
        raise DecisionIntegrityError("per-label threshold objectives must be in [0, 1]")
    return ThresholdOptimizationResult(
        label_order=labels,
        thresholds=thresholds,
        objective="f1",
        macro_objective=macro_objective,
        default_threshold=default_threshold,
        n_samples=n_samples,
        source_folds=source_folds,
        per_label=tuple(per_label),
    )


def _parse_gates(value: object, *, expected_count: int) -> tuple[CoverageGate, ...]:
    if not isinstance(value, list) or not value:
        raise DecisionIntegrityError("selective gates must be a non-empty list")
    gates: list[CoverageGate] = []
    for index, item in enumerate(value):
        raw = _mapping(item, f"gates[{index}]")
        _expect_keys(
            raw,
            required={
                "target_coverage",
                "maximum_entropy",
                "calibration_coverage",
                "selected_count",
                "calibration_count",
            },
            context=f"gates[{index}]",
        )
        target = _finite_float(raw["target_coverage"], "target_coverage", minimum=0.0)
        entropy = _finite_float(raw["maximum_entropy"], "maximum_entropy", minimum=0.0)
        achieved = _finite_float(
            raw["calibration_coverage"], "calibration_coverage", minimum=0.0
        )
        selected = _integer(raw["selected_count"], "selected_count", minimum=1)
        count = _integer(raw["calibration_count"], "calibration_count", minimum=1)
        if not 0 < target <= 1 or not 0 <= entropy <= 1 or not target <= achieved <= 1:
            raise DecisionIntegrityError("invalid selective gate coverage or entropy")
        if count != expected_count or selected > count or achieved != selected / count:
            raise DecisionIntegrityError("selective gate counts are inconsistent")
        gates.append(CoverageGate(target, entropy, achieved, selected, count))
    if len({gate.target_coverage for gate in gates}) != len(gates):
        raise DecisionIntegrityError("selective gate coverage targets contain duplicates")
    return tuple(gates)


def _json_destination(path: str | Path) -> Path:
    destination = Path(path)
    if destination.suffix.casefold() != ".json":
        raise DecisioningError("JSON artifact path must end in .json")
    return destination


def _write_new_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"immutable JSON artifact already exists: {path}")
    descriptor, raw_temp = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(raw_temp)
    try:
        serialized = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(serialized)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        with suppress(OSError):
            temporary.unlink(missing_ok=True)


def _read_json(path: Path, *, context: str) -> Mapping[str, object]:
    if not path.is_file():
        raise DecisionIntegrityError(f"{context} is missing: {path}")
    if path.stat().st_size > 100_000_000:
        raise DecisionIntegrityError(f"{context} is unreasonably large")
    try:
        decoded: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DecisionIntegrityError(f"could not decode {context}: {exc}") from exc
    return _mapping(decoded, context)


def _payload_hash(payload: Mapping[str, object]) -> str:
    serialized = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return "sha256:" + hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _required_integrity(value: str | None, name: str) -> str:
    if value is None:
        raise DecisioningError(f"{name} must be loaded from an integrity-bound artifact")
    return _hash_string(value, f"{name} integrity")


def _timestamp(value: str | None) -> str:
    if value is None:
        return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except (TypeError, ValueError) as exc:
        raise DecisioningError(
            "timestamp must use canonical UTC format YYYY-MM-DDTHH:MM:SSZ"
        ) from exc
    return parsed.strftime("%Y-%m-%dT%H:%M:%SZ")


def _software_versions() -> dict[str, str]:
    return {
        "ecg_trust": __version__,
        "numpy": np.__version__,
        "python": platform.python_version(),
    }


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise DecisionIntegrityError(f"{context} must be a string-keyed mapping")
    return cast(Mapping[str, object], value)


def _expect_keys(
    mapping: Mapping[str, object],
    *,
    required: set[str],
    context: str,
) -> None:
    missing = sorted(required.difference(mapping))
    unexpected = sorted(set(mapping).difference(required))
    if missing or unexpected:
        raise DecisionIntegrityError(
            f"{context} has invalid keys; missing={missing}, unexpected={unexpected}"
        )


def _string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DecisionIntegrityError(f"{context} must be a non-empty string")
    return value


def _integer(value: object, context: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise DecisionIntegrityError(f"{context} must be an integer >= {minimum}")
    return value


def _boolean(value: object, context: str) -> bool:
    if not isinstance(value, bool):
        raise DecisionIntegrityError(f"{context} must be boolean")
    return value


def _finite_float(value: object, context: str, *, minimum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DecisionIntegrityError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise DecisionIntegrityError(f"{context} must be finite and >= {minimum}")
    return result


def _optional_float(value: object, context: str) -> float | None:
    return None if value is None else _finite_float(value, context, minimum=0.0)


def _integer_tuple(value: object, context: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise DecisionIntegrityError(f"{context} must be a list of integers")
    return tuple(_integer(item, context, minimum=0) for item in value)


def _float_tuple(value: object, context: str) -> tuple[float, ...]:
    if not isinstance(value, list):
        raise DecisionIntegrityError(f"{context} must be a list of numbers")
    return tuple(_finite_float(item, context, minimum=0.0) for item in value)


def _string_tuple(value: object, context: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise DecisionIntegrityError(f"{context} must be a list of strings")
    return tuple(_string(item, context) for item in value)


def _label_order(value: object) -> tuple[str, ...]:
    labels = _string_tuple(value, "label_order")
    if labels != LABEL_ORDER:
        raise DecisionIntegrityError(
            f"label_order must be exactly {LABEL_ORDER!r}; received {labels!r}"
        )
    return labels


def _hash_string(value: object, context: str) -> str:
    text = _string(value, context)
    prefix = "sha256:"
    digest = text[len(prefix) :] if text.startswith(prefix) else text
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise DecisionIntegrityError(f"{context} must be a lower-case SHA-256 string")
    return prefix + digest


def _string_mapping(value: object, context: str) -> dict[str, str]:
    raw = _mapping(value, context)
    return {
        _string(key, f"{context} key"): _string(item, f"{context}.{key}")
        for key, item in raw.items()
    }
