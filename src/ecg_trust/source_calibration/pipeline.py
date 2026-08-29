"""Leakage-resistant preparation of the frozen Trust Sentinel source policy.

The public result contains aggregate evidence only. Row-level identifiers and
prediction arrays remain transient in memory and are never written by this module.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Literal, cast

import numpy as np
import yaml  # type: ignore[import-untyped]
from numpy.typing import NDArray
from pydantic import ValidationError

from ecg_trust.conformal import (
    LabelwiseBinaryConformal,
    evaluate_prediction_sets,
)
from ecg_trust.evaluation import (
    TemperatureScalingResult,
    ThresholdOptimizationResult,
    compute_multilabel_metrics,
    fit_temperature_scaling,
    optimize_thresholds,
)
from ecg_trust.open_world.scores import normalized_bernoulli_entropy
from ecg_trust.predictions import (
    PredictionArtifact,
    PredictionArtifactError,
    load_prediction_artifact,
)
from ecg_trust.protocol import ExperimentProtocol, FoldRole
from ecg_trust.source_calibration.models import (
    FAILURE_RECEIPT_FILENAME,
    LABEL_ORDER,
    RESULT_FILENAME,
    SPLIT_ALGORITHM_ID,
    ClaimBoundary,
    ConformalFitSummary,
    ConformalValidationSummary,
    EntropyGateSummary,
    EntropyValidationSummary,
    FailureCode,
    FailureReceiptBody,
    FrozenComponents,
    FrozenComponentsBody,
    LabelConformalCoverage,
    LabelConformalThreshold,
    LabelMetricSummary,
    LabelThresholdSummary,
    MacroMetricSummary,
    OpenWorldPendingSummary,
    PositiveRecords,
    RoleCounts,
    SourceCalibrationConfig,
    SourceCalibrationConfigError,
    SourceCalibrationIntegrityError,
    SourceCalibrationOutputError,
    SourceCalibrationResult,
    SourceCalibrationResultBody,
    SourceProvenance,
    SourceRole,
    SourceValidationSummary,
    SplitEvidence,
    TemperatureFitSummary,
    ThresholdFitSummary,
    ThresholdValidationSummary,
    canonical_json_bytes,
    canonical_sha256,
    failure_receipt_json_bytes,
    result_json_bytes,
    seal_failure_receipt,
    seal_frozen_components,
    seal_source_calibration_result,
)

FloatArray = NDArray[np.float64]
Int64Array = NDArray[np.int64]
Int8Array = NDArray[np.int8]
BoolArray = NDArray[np.bool_]
LabelName = Literal["NORM", "MI", "STTC", "CD", "HYP"]

_CONFIG_MAX_BYTES = 1_000_000
_RESULT_MAX_BYTES = 2_000_000
_SOURCE_NPZ_MAX_BYTES = 256_000_000
_SIDECAR_MAX_BYTES = 1_000_000
_BOUND_JSON_MAX_BYTES = 10_000_000
_NPY_HEADER_ALLOWANCE = 65_536
_EXACT_ARRAY_NAMES = ("ecg_id", "patient_id", "strat_fold", "targets", "raw_logits")
_SENSITIVE_RESULT_KEYS = frozenset(
    {
        "ecg_id",
        "patient_id",
        "raw_logits",
        "targets",
        "probabilities",
        "selected_indices",
        "abstained_indices",
        "rows",
        "records_array",
        "path",
        "npz_path",
        "sidecar_path",
    }
)
_SENSITIVE_KEY_FRAGMENTS = ("password", "secret", "access_token", "private_key")
_GIT_REVISION_PATTERN = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")


@dataclass(frozen=True, slots=True)
class SourcePredictionArrays:
    """Validated in-memory source predictions; never serialize this object."""

    ecg_id: Int64Array
    patient_id: Int64Array
    strat_fold: Int8Array
    targets: Int8Array
    raw_logits: FloatArray
    prediction_artifact_sha256: str
    source_alignment_sha256: str

    @property
    def n_samples(self) -> int:
        return int(self.ecg_id.shape[0])


@dataclass(frozen=True, slots=True)
class RoleData:
    """One patient-exclusive role held transiently in memory."""

    role: SourceRole
    patient_id: Int64Array
    targets: Int8Array
    raw_logits: FloatArray

    @property
    def records(self) -> int:
        return int(self.targets.shape[0])

    @property
    def patients(self) -> int:
        return int(np.unique(self.patient_id).size)


@dataclass(frozen=True, slots=True)
class SourcePartitions:
    """The three frozen patient-level roles."""

    decision_fit: RoleData
    conformal_and_ood_threshold_fit: RoleData
    source_validation: RoleData
    evidence: SplitEvidence

    def __post_init__(self) -> None:
        observed_roles = (
            self.decision_fit.role,
            self.conformal_and_ood_threshold_fit.role,
            self.source_validation.role,
        )
        if observed_roles != tuple(SourceRole):
            raise SourceCalibrationIntegrityError("partition roles are not in frozen order")
        patient_sets = tuple(
            {int(value) for value in role.patient_id}
            for role in (
                self.decision_fit,
                self.conformal_and_ood_threshold_fit,
                self.source_validation,
            )
        )
        if (
            patient_sets[0].intersection(patient_sets[1])
            or patient_sets[0].intersection(patient_sets[2])
            or patient_sets[1].intersection(patient_sets[2])
        ):
            raise SourceCalibrationIntegrityError("patient identities overlap frozen roles")

    def for_role(self, role: SourceRole) -> RoleData:
        if role is SourceRole.DECISION_FIT:
            return self.decision_fit
        if role is SourceRole.CONFORMAL_AND_OOD_THRESHOLD_FIT:
            return self.conformal_and_ood_threshold_fit
        return self.source_validation


@dataclass(frozen=True, slots=True)
class FittedSourceComponents:
    """Runtime fitted objects plus their aggregate immutable manifest."""

    temperature: TemperatureScalingResult
    thresholds: ThresholdOptimizationResult
    entropy_cutoff: float
    conformal: LabelwiseBinaryConformal
    summary: FrozenComponents


@dataclass(frozen=True, slots=True)
class FittedEntropyGate:
    """Tie-correct frozen entropy cutoff fitted on one declared role."""

    target_coverage: float
    maximum_entropy: float
    selected_count: int
    fit_count: int

    @property
    def achieved_coverage(self) -> float:
        return self.selected_count / self.fit_count


@dataclass(frozen=True, slots=True)
class VerifiedSourceInputs:
    """Verified paths and hashes retained only for the duration of execution."""

    npz_path: Path
    sidecar_path: Path
    source_npz_sha256: str
    source_sidecar_sha256: str
    demo_binding_file_sha256: str
    historical_policy_file_sha256: str


def patient_split_fraction(*, patient_id: int, salt: str) -> float:
    """Apply the exact frozen patient-level SHA-256 split function."""

    if isinstance(patient_id, bool) or not isinstance(patient_id, int) or patient_id <= 0:
        raise SourceCalibrationIntegrityError("patient identifiers must be positive integers")
    if not isinstance(salt, str) or not salt:
        raise SourceCalibrationIntegrityError("patient split salt must be a non-empty string")
    digest = hashlib.sha256(f"{salt}|{patient_id:d}".encode()).digest()
    numerator = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return numerator / float(1 << 64)


def patient_split_role(*, patient_id: int, salt: str) -> SourceRole:
    """Assign one patient to exactly one frozen role."""

    fraction = patient_split_fraction(patient_id=patient_id, salt=salt)
    if fraction < 0.4:
        return SourceRole.DECISION_FIT
    if fraction < 0.8:
        return SourceRole.CONFORMAL_AND_OOD_THRESHOLD_FIT
    return SourceRole.SOURCE_VALIDATION


def load_source_calibration_config(
    path: str | Path,
) -> tuple[SourceCalibrationConfig, str]:
    """Read and strictly validate a frozen YAML configuration and raw file hash."""

    config_path = Path(path)
    try:
        size = config_path.stat().st_size
        if size <= 0 or size > _CONFIG_MAX_BYTES:
            raise SourceCalibrationConfigError("configuration file size is invalid")
        raw = config_path.read_bytes()
    except SourceCalibrationConfigError:
        raise
    except OSError as error:
        raise SourceCalibrationConfigError("configuration file is unavailable") from error
    try:
        serialized = raw.decode("utf-8")
        _reject_duplicate_yaml_keys(serialized)
        decoded: object = yaml.safe_load(serialized)
    except (UnicodeError, yaml.YAMLError) as error:
        raise SourceCalibrationConfigError("configuration is not valid UTF-8 YAML") from error
    if not isinstance(decoded, Mapping) or not all(isinstance(key, str) for key in decoded):
        raise SourceCalibrationConfigError("configuration root must be a string-keyed mapping")
    try:
        config = SourceCalibrationConfig.model_validate(dict(decoded))
    except ValidationError as error:
        raise SourceCalibrationConfigError("configuration violates the frozen schema") from error
    return config, _sha256_bytes(raw)


def verify_source_inputs(
    config: SourceCalibrationConfig,
    *,
    project_root: str | Path,
) -> VerifiedSourceInputs:
    """Verify every declared file hash before decoding any source artifact."""

    root = Path(project_root).resolve(strict=True)
    references = (
        (
            "source_npz",
            config.source_prediction.npz_path,
            config.source_prediction.npz_sha256,
        ),
        (
            "source_sidecar",
            config.source_prediction.sidecar_path,
            config.source_prediction.sidecar_sha256,
        ),
        ("demo_binding", config.model.demo_binding.path, config.model.demo_binding.file_sha256),
        (
            "historical_policy",
            config.model.historical_policy.path,
            config.model.historical_policy.file_sha256,
        ),
    )
    resolved: dict[str, Path] = {}
    observed: dict[str, str] = {}
    maximum_sizes = {
        "source_npz": _SOURCE_NPZ_MAX_BYTES,
        "source_sidecar": _SIDECAR_MAX_BYTES,
        "demo_binding": _BOUND_JSON_MAX_BYTES,
        "historical_policy": _BOUND_JSON_MAX_BYTES,
    }
    for name, relative_path, expected_hash in references:
        path = _resolve_project_path(root, relative_path, require_file=True)
        try:
            size = path.stat().st_size
        except OSError as error:
            raise SourceCalibrationIntegrityError(f"{name} file is unavailable") from error
        if size <= 0 or size > maximum_sizes[name]:
            raise SourceCalibrationIntegrityError(f"{name} file size is invalid")
        digest = _sha256_file(path)
        if digest != f"sha256:{expected_hash}":
            raise SourceCalibrationIntegrityError(f"{name} file hash does not match config")
        resolved[name] = path
        observed[name] = digest

    return VerifiedSourceInputs(
        npz_path=resolved["source_npz"],
        sidecar_path=resolved["source_sidecar"],
        source_npz_sha256=observed["source_npz"],
        source_sidecar_sha256=observed["source_sidecar"],
        demo_binding_file_sha256=observed["demo_binding"],
        historical_policy_file_sha256=observed["historical_policy"],
    )


def load_verified_source_predictions(
    config: SourceCalibrationConfig,
    verified: VerifiedSourceInputs,
) -> SourcePredictionArrays:
    """Load a hash-verified exact NPZ and cross-check its integrity sidecar."""

    _assert_source_pair_unchanged(verified)
    safe_arrays = _safe_load_npz(
        verified.npz_path,
        expected_records=config.source_prediction.expected_records,
    )
    protocol = ExperimentProtocol.canonical()
    try:
        artifact = load_prediction_artifact(verified.npz_path, protocol=protocol)
    except PredictionArtifactError as error:
        raise SourceCalibrationIntegrityError(
            "source prediction sidecar or archive contract failed"
        ) from error
    _verify_prediction_artifact(artifact, safe_arrays=safe_arrays, config=config)
    _assert_source_pair_unchanged(verified)
    integrity = artifact.integrity_sha256
    if integrity is None:
        raise SourceCalibrationIntegrityError("loaded prediction artifact lacks integrity identity")
    return SourcePredictionArrays(
        ecg_id=_readonly_copy(cast(Int64Array, safe_arrays["ecg_id"])),
        patient_id=_readonly_copy(cast(Int64Array, safe_arrays["patient_id"])),
        strat_fold=_readonly_copy(cast(Int8Array, safe_arrays["strat_fold"])),
        targets=_readonly_copy(cast(Int8Array, safe_arrays["targets"])),
        raw_logits=_readonly_copy(cast(FloatArray, safe_arrays["raw_logits"])),
        prediction_artifact_sha256=integrity,
        source_alignment_sha256=artifact.alignment_sha256,
    )


def partition_source_predictions(
    source: SourcePredictionArrays,
    config: SourceCalibrationConfig,
) -> SourcePartitions:
    """Split by patient, then enforce the frozen record/patient/positive counts."""

    patient_values = tuple(int(value) for value in np.unique(source.patient_id))
    assignments = {
        patient: patient_split_role(patient_id=patient, salt=config.patient_split.salt)
        for patient in patient_values
    }
    masks = {
        role: np.asarray(
            [assignments[int(patient)] is role for patient in source.patient_id],
            dtype=np.bool_,
        )
        for role in SourceRole
    }
    roles = {role: _role_data(source=source, role=role, mask=masks[role]) for role in SourceRole}
    expected_by_role = {
        SourceRole.DECISION_FIT: config.patient_split.expected.decision_fit,
        SourceRole.CONFORMAL_AND_OOD_THRESHOLD_FIT: (
            config.patient_split.expected.conformal_and_ood_threshold_fit
        ),
        SourceRole.SOURCE_VALIDATION: config.patient_split.expected.source_validation,
    }
    role_counts: list[RoleCounts] = []
    for role in SourceRole:
        data = roles[role]
        counts = _role_counts(data)
        if counts.model_dump(mode="json", exclude={"role"}) != expected_by_role[role].model_dump(
            mode="json"
        ):
            raise SourceCalibrationIntegrityError(
                f"observed {role.value} counts differ from the frozen split"
            )
        role_counts.append(counts)

    assignment_payload: dict[str, object] = {
        "schema_version": 1,
        "algorithm": SPLIT_ALGORITHM_ID,
        "assignments": [
            {"patient_id_base10": str(patient), "role": assignments[patient].value}
            for patient in sorted(assignments)
        ],
    }
    evidence = SplitEvidence(
        unit="patient",
        algorithm=SPLIT_ALGORITHM_ID,
        salt_sha256=_sha256_bytes(config.patient_split.salt.encode("utf-8")),
        assignment_sha256=canonical_sha256(assignment_payload),
        roles=(role_counts[0], role_counts[1], role_counts[2]),
    )
    return SourcePartitions(
        decision_fit=roles[SourceRole.DECISION_FIT],
        conformal_and_ood_threshold_fit=roles[SourceRole.CONFORMAL_AND_OOD_THRESHOLD_FIT],
        source_validation=roles[SourceRole.SOURCE_VALIDATION],
        evidence=evidence,
    )


def fit_source_components(
    partitions: SourcePartitions,
    config: SourceCalibrationConfig,
) -> FittedSourceComponents:
    """Fit only on the two declared fitting roles; validation is never read."""

    decision = partitions.decision_fit
    conformal_role = partitions.conformal_and_ood_threshold_fit
    decision_folds = np.full(decision.records, 9, dtype=np.int8)
    temperature = fit_temperature_scaling(
        logits=decision.raw_logits,
        y_true=decision.targets,
        calibration_fold_ids=decision_folds,
        label_order=LABEL_ORDER,
    )
    decision_probabilities = temperature.predict_proba(decision.raw_logits)
    thresholds = optimize_thresholds(
        y_true=decision.targets,
        probabilities=decision_probabilities,
        calibration_fold_ids=decision_folds,
        label_order=LABEL_ORDER,
    )
    entropy_gate = fit_entropy_gate(
        decision_probabilities,
        target_coverage=config.decision_fit.legacy_entropy_gate.target_coverage,
    )

    conformal_probabilities = temperature.predict_proba(conformal_role.raw_logits)
    conformal = LabelwiseBinaryConformal.fit(
        conformal_probabilities,
        conformal_role.targets,
        label_names=LABEL_ORDER,
        alpha=config.conformal.alpha,
    )

    component_body = FrozenComponentsBody(
        temperature=_temperature_summary(temperature),
        thresholds=_threshold_summary(thresholds),
        entropy_gate=EntropyGateSummary(
            method="mean_normalized_binary_entropy",
            fit_role=SourceRole.DECISION_FIT,
            target_coverage=0.8,
            tie_rule="retain_all_scores_less_than_or_equal_to_frozen_order_statistic",
            maximum_entropy=entropy_gate.maximum_entropy,
            selected_count=entropy_gate.selected_count,
            fit_count=entropy_gate.fit_count,
            achieved_coverage=entropy_gate.achieved_coverage,
        ),
        conformal=_conformal_summary(conformal),
    )
    return FittedSourceComponents(
        temperature=temperature,
        thresholds=thresholds,
        entropy_cutoff=entropy_gate.maximum_entropy,
        conformal=conformal,
        summary=seal_frozen_components(component_body),
    )


def evaluate_source_validation(
    partitions: SourcePartitions,
    fitted: FittedSourceComponents,
    config: SourceCalibrationConfig,
) -> SourceValidationSummary:
    """Evaluate frozen components on source-validation rows without tuning."""

    validation = partitions.source_validation
    probabilities = fitted.temperature.predict_proba(validation.raw_logits)
    metrics = compute_multilabel_metrics(
        validation.targets,
        probabilities,
        label_order=LABEL_ORDER,
        ece_bins=15,
    )
    decisions = fitted.thresholds.apply(probabilities, label_order=LABEL_ORDER)
    hamming, exact = _decision_metrics(validation.targets, decisions)

    entropy = normalized_bernoulli_entropy(probabilities)
    retained = entropy <= fitted.entropy_cutoff
    retained_count = int(np.count_nonzero(retained))
    if retained_count:
        retained_hamming, retained_exact = _decision_metrics(
            validation.targets[retained], decisions[retained]
        )
    else:
        retained_hamming, retained_exact = None, None

    conformal_metrics = evaluate_prediction_sets(
        fitted.conformal.predict(probabilities), validation.targets
    )
    per_label = tuple(
        LabelMetricSummary(
            label=_label_name(item.label),
            positives=item.positives,
            negatives=item.negatives,
            minimum_positive_records=(
                config.source_validation.minimum_positive_records_for_label_statement
            ),
            statement_status=(
                "SUFFICIENT_EVIDENCE"
                if item.positives
                >= config.source_validation.minimum_positive_records_for_label_statement
                else "INSUFFICIENT_EVIDENCE"
            ),
            roc_auc=item.roc_auc,
            average_precision=item.average_precision,
            brier_score=item.brier_score,
            ece15=item.ece,
            degenerate_reason=item.degenerate_reason,
        )
        for item in metrics.per_label
    )
    conformal_per_label = tuple(
        LabelConformalCoverage(
            label=_label_name(label),
            empirical_coverage=conformal_metrics.labelwise_coverage[index],
            mean_set_size=conformal_metrics.labelwise_mean_set_size[index],
        )
        for index, label in enumerate(LABEL_ORDER)
    )
    component_hash = fitted.summary.component_sha256
    return SourceValidationSummary(
        evaluation_role=SourceRole.SOURCE_VALIDATION,
        tuning_allowed=False,
        records=validation.records,
        patients=validation.patients,
        ece_bins=15,
        per_label=cast(
            tuple[
                LabelMetricSummary,
                LabelMetricSummary,
                LabelMetricSummary,
                LabelMetricSummary,
                LabelMetricSummary,
            ],
            per_label,
        ),
        macro=MacroMetricSummary(
            roc_auc=metrics.macro.roc_auc,
            average_precision=metrics.macro.average_precision,
            brier_score=metrics.macro.brier_score,
            ece15=metrics.macro.ece,
            roc_auc_labels=metrics.macro.roc_auc_labels,
            average_precision_labels=metrics.macro.average_precision_labels,
        ),
        threshold_decisions=ThresholdValidationSummary(
            frozen_component_sha256=component_hash,
            hamming_loss=hamming,
            exact_match_accuracy=exact,
        ),
        entropy_gate=EntropyValidationSummary(
            frozen_component_sha256=component_hash,
            maximum_entropy=fitted.entropy_cutoff,
            selected_count=retained_count,
            validation_count=validation.records,
            achieved_coverage=float(retained_count / validation.records),
            retained_hamming_loss=retained_hamming,
            retained_exact_match_accuracy=retained_exact,
        ),
        conformal=ConformalValidationSummary(
            frozen_component_sha256=component_hash,
            coverage_scope="labelwise_marginal_under_exchangeability",
            individual_certainty_guarantee=False,
            marginal_coverage=conformal_metrics.marginal_coverage,
            joint_sample_coverage=conformal_metrics.joint_sample_coverage,
            mean_set_size=conformal_metrics.mean_set_size,
            singleton_fraction=conformal_metrics.singleton_fraction,
            empty_fraction=conformal_metrics.empty_fraction,
            both_fraction=conformal_metrics.both_fraction,
            per_label=cast(
                tuple[
                    LabelConformalCoverage,
                    LabelConformalCoverage,
                    LabelConformalCoverage,
                    LabelConformalCoverage,
                    LabelConformalCoverage,
                ],
                conformal_per_label,
            ),
        ),
    )


def build_source_calibration_result(
    *,
    config: SourceCalibrationConfig,
    source: SourcePredictionArrays,
    verified: VerifiedSourceInputs,
    config_file_sha256: str,
    code_revision: str,
) -> SourceCalibrationResult:
    """Partition, fit, evaluate, and seal one aggregate-only deterministic result."""

    partitions = partition_source_predictions(source, config)
    fitted = fit_source_components(partitions, config)
    validation = evaluate_source_validation(partitions, fitted, config)
    protocol = ExperimentProtocol.canonical()
    provenance_hashes: dict[str, object] = {
        "config_file_sha256": config_file_sha256,
        "source_npz_sha256": verified.source_npz_sha256,
        "source_sidecar_sha256": verified.source_sidecar_sha256,
        "prediction_artifact_sha256": source.prediction_artifact_sha256,
        "source_alignment_sha256": source.source_alignment_sha256,
        "checkpoint_sha256": f"sha256:{config.model.checkpoint_sha256}",
        "demo_binding_file_sha256": verified.demo_binding_file_sha256,
        "historical_policy_file_sha256": verified.historical_policy_file_sha256,
        "experiment_protocol_sha256": protocol.protocol_hash,
    }
    body = SourceCalibrationResultBody(
        schema_version=1,
        artifact_type="ecg_trust.source_calibration_result",
        protocol_id="trust-sentinel-source-calibration-v1",
        status="PREPARED_NOT_RELEASE_READY",
        frozen_at_utc=config.frozen_at_utc,
        provenance=SourceProvenance(
            config_file_sha256=config_file_sha256,
            source_npz_sha256=verified.source_npz_sha256,
            source_sidecar_sha256=verified.source_sidecar_sha256,
            prediction_artifact_sha256=source.prediction_artifact_sha256,
            source_alignment_sha256=source.source_alignment_sha256,
            checkpoint_sha256=f"sha256:{config.model.checkpoint_sha256}",
            demo_binding_file_sha256=verified.demo_binding_file_sha256,
            historical_policy_file_sha256=verified.historical_policy_file_sha256,
            experiment_protocol_sha256=protocol.protocol_hash,
            source_bundle_sha256=canonical_sha256(provenance_hashes),
            code_revision=code_revision,
            model_member_id=config.model.member_id,
            source_artifact_model_name=config.model.source_artifact_model_name,
            architecture="resnet1d",
            seed=config.model.seed,
            source_fold=9,
        ),
        split=partitions.evidence,
        frozen_components=fitted.summary,
        source_validation=validation,
        open_world=OpenWorldPendingSummary(
            method="shrinkage_mahalanobis_embedding_distance",
            status="PENDING",
            artifact_sha256=None,
            threshold_fitted=False,
            source_false_rejection_evaluated=False,
            release_ready=False,
            reference_alignment_verified=False,
            embedding_device=None,
            embedding_precision=None,
            reason_code="REFERENCE_AND_THRESHOLD_EMBEDDINGS_NOT_PROVIDED",
        ),
        claims=ClaimBoundary(
            scope="retrospective_source_domain_development_only",
            research_only=True,
            clinical_validation=False,
            limitations=config.claims.limitations,
        ),
    )
    result = seal_source_calibration_result(body)
    assert_aggregate_only_result(result)
    return result


def prepare_source_calibration(
    *,
    config_path: str | Path,
    project_root: str | Path,
    code_revision: str,
) -> SourceCalibrationResult:
    """Prepare a new immutable local result, leaving a sanitized receipt on failure."""

    _validate_code_revision(code_revision)
    config, config_file_sha256 = load_source_calibration_config(config_path)
    verified = verify_source_inputs(config, project_root=project_root)
    root = Path(project_root).resolve(strict=True)
    output_root = _resolve_project_path(root, config.execution.output_root, require_file=False)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    if output_root.exists():
        raise SourceCalibrationOutputError("immutable output root already exists")
    try:
        raw_staging_root = tempfile.mkdtemp(
            prefix=f".{output_root.name}.staging-", dir=output_root.parent
        )
    except OSError as error:
        raise SourceCalibrationOutputError("could not create the output staging root") from error
    staging_root = Path(raw_staging_root).resolve(strict=True)
    committed = False

    try:
        source = load_verified_source_predictions(config, verified)
        result = build_source_calibration_result(
            config=config,
            source=source,
            verified=verified,
            config_file_sha256=config_file_sha256,
            code_revision=code_revision,
        )
        _atomic_write_new(staging_root / RESULT_FILENAME, result_json_bytes(result))
        load_source_calibration_result_bytes((staging_root / RESULT_FILENAME).read_bytes())
        _commit_staged_directory(staging_root, output_root)
        committed = True
        load_source_calibration_result_bytes((output_root / RESULT_FILENAME).read_bytes())
        return result
    except Exception as error:
        if committed:
            receipt = seal_failure_receipt(
                FailureReceiptBody(
                    schema_version=1,
                    artifact_type="ecg_trust.source_calibration_failure",
                    protocol_id="trust-sentinel-source-calibration-v1",
                    status="FAILED",
                    frozen_at_utc=config.frozen_at_utc,
                    config_file_sha256=config_file_sha256,
                    code_revision=code_revision,
                    failure_code=_failure_code(error),
                    contains_raw_ids_or_rows=False,
                    retry_requires_new_output_root=True,
                )
            )
            try:
                _atomic_write_new(
                    output_root / FAILURE_RECEIPT_FILENAME,
                    failure_receipt_json_bytes(receipt),
                )
            except Exception as receipt_error:
                raise SourceCalibrationOutputError(
                    "preparation failed and sanitized failure receipt could not be committed"
                ) from receipt_error
        else:
            _remove_staging_root(staging_root, expected_parent=output_root.parent)
        raise


def verify_clean_git_revision(project_root: str | Path) -> str:
    """Fail closed unless the project root is the clean committed Git worktree."""

    root = Path(project_root).resolve(strict=True)
    top_level = Path(_run_git(root, "rev-parse", "--show-toplevel")).resolve(strict=True)
    if top_level != root:
        raise SourceCalibrationIntegrityError("project root is not the Git worktree root")
    status = _run_git(root, "status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise SourceCalibrationIntegrityError("source calibration requires a clean worktree")
    revision = _run_git(root, "rev-parse", "HEAD").casefold()
    _validate_code_revision(revision)
    return revision


def assert_complete_release_ready(result: SourceCalibrationResult) -> None:
    """Reject any attempt to treat this preparation artifact as a complete release."""

    if not result.open_world.release_ready:
        raise SourceCalibrationIntegrityError(
            "source preparation is not a complete release while OOD evidence is pending"
        )


def load_source_calibration_result_bytes(payload: bytes) -> SourceCalibrationResult:
    """Verify canonical serialization, strict schema, self hash, and privacy boundary."""

    if not payload or len(payload) > _RESULT_MAX_BYTES or not payload.endswith(b"\n"):
        raise SourceCalibrationIntegrityError("result JSON byte contract is invalid")
    if payload.endswith(b"\n\n") or b"\r" in payload:
        raise SourceCalibrationIntegrityError("result JSON is not canonically terminated")
    try:
        decoded: object = json.loads(payload[:-1].decode("ascii"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise SourceCalibrationIntegrityError("result JSON cannot be decoded") from error
    if not isinstance(decoded, dict) or not all(isinstance(key, str) for key in decoded):
        raise SourceCalibrationIntegrityError("result JSON root is invalid")
    try:
        result = SourceCalibrationResult.model_validate(decoded)
    except ValidationError as error:
        raise SourceCalibrationIntegrityError("result JSON violates its schema") from error
    if result_json_bytes(result) != payload:
        raise SourceCalibrationIntegrityError("result JSON is not canonical")
    assert_aggregate_only_result(result)
    return result


def assert_aggregate_only_result(result: SourceCalibrationResult) -> None:
    """Fail closed if a serialized result contains rows, identifiers, paths, or secrets."""

    payload = result.model_dump(mode="json")

    def visit(value: object) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                normalized = key.casefold()
                if normalized in _SENSITIVE_RESULT_KEYS or any(
                    fragment in normalized for fragment in _SENSITIVE_KEY_FRAGMENTS
                ):
                    raise SourceCalibrationIntegrityError(
                        "result contains a forbidden sensitive field"
                    )
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)
        elif isinstance(value, str) and _looks_absolute_path(value):
            raise SourceCalibrationIntegrityError("result contains an absolute path")

    visit(payload)
    canonical_json_bytes(payload)


def replace_validation_role(
    partitions: SourcePartitions,
    *,
    raw_logits: FloatArray | None = None,
    targets: Int8Array | None = None,
) -> SourcePartitions:
    """Test helper for proving validation mutations cannot influence fitted components."""

    current = partitions.source_validation
    replacement = replace(
        current,
        raw_logits=current.raw_logits if raw_logits is None else _readonly_copy(raw_logits),
        targets=current.targets if targets is None else _readonly_copy(targets),
    )
    return replace(partitions, source_validation=replacement)


def fit_entropy_gate(
    probabilities: FloatArray,
    *,
    target_coverage: float,
) -> FittedEntropyGate:
    """Fit the order statistic while retaining every score tied at the cutoff."""

    if not math.isfinite(target_coverage) or not 0.0 < target_coverage <= 1.0:
        raise SourceCalibrationIntegrityError("entropy target coverage must lie in (0, 1]")
    uncertainty = normalized_bernoulli_entropy(probabilities)
    total = int(uncertainty.size)
    requested_count = min(total, max(1, math.ceil(target_coverage * total)))
    cutoff = float(np.sort(uncertainty, kind="stable")[requested_count - 1])
    selected_count = int(np.count_nonzero(uncertainty <= cutoff))
    return FittedEntropyGate(
        target_coverage=target_coverage,
        maximum_entropy=cutoff,
        selected_count=selected_count,
        fit_count=total,
    )


def _safe_load_npz(path: Path, *, expected_records: int) -> dict[str, NDArray[np.generic]]:
    try:
        source_size = path.stat().st_size
    except OSError as error:
        raise SourceCalibrationIntegrityError("source NPZ is unavailable") from error
    if source_size <= 0 or source_size > _SOURCE_NPZ_MAX_BYTES:
        raise SourceCalibrationIntegrityError("source NPZ file size is invalid")
    expected_shapes = {
        "ecg_id": (expected_records,),
        "patient_id": (expected_records,),
        "strat_fold": (expected_records,),
        "targets": (expected_records, len(LABEL_ORDER)),
        "raw_logits": (expected_records, len(LABEL_ORDER)),
    }
    expected_dtypes: dict[str, np.dtype[np.generic]] = {
        "ecg_id": np.dtype(np.int64),
        "patient_id": np.dtype(np.int64),
        "strat_fold": np.dtype(np.int8),
        "targets": np.dtype(np.int8),
        "raw_logits": np.dtype(np.float64),
    }
    try:
        with zipfile.ZipFile(path, mode="r") as archive:
            members = archive.infolist()
            expected_member_names = {f"{name}.npy" for name in _EXACT_ARRAY_NAMES}
            if (
                len(members) != len(expected_member_names)
                or {member.filename for member in members} != expected_member_names
            ):
                raise SourceCalibrationIntegrityError("source NPZ has an invalid member inventory")
            for member in members:
                if member.flag_bits & 0x1:
                    raise SourceCalibrationIntegrityError("encrypted NPZ members are forbidden")
                name = member.filename.removesuffix(".npy")
                payload_size = math.prod(expected_shapes[name]) * expected_dtypes[name].itemsize
                if not payload_size < member.file_size <= payload_size + _NPY_HEADER_ALLOWANCE:
                    raise SourceCalibrationIntegrityError("source NPZ member size is invalid")
        with np.load(path, allow_pickle=False) as loaded:
            if tuple(loaded.files) != _EXACT_ARRAY_NAMES:
                raise SourceCalibrationIntegrityError(
                    "source NPZ key order or inventory is invalid"
                )
            arrays = {name: np.asarray(loaded[name]).copy() for name in _EXACT_ARRAY_NAMES}
    except SourceCalibrationIntegrityError:
        raise
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        raise SourceCalibrationIntegrityError("source NPZ cannot be decoded safely") from error

    for name in _EXACT_ARRAY_NAMES:
        array = arrays[name]
        if array.dtype != expected_dtypes[name] or array.shape != expected_shapes[name]:
            raise SourceCalibrationIntegrityError("source NPZ dtype or shape is invalid")
    ecg_id = cast(Int64Array, arrays["ecg_id"])
    patient_id = cast(Int64Array, arrays["patient_id"])
    folds = cast(Int8Array, arrays["strat_fold"])
    targets = cast(Int8Array, arrays["targets"])
    logits = cast(FloatArray, arrays["raw_logits"])
    if np.any(ecg_id <= 0) or np.unique(ecg_id).size != expected_records:
        raise SourceCalibrationIntegrityError("source ECG identifiers are invalid")
    if np.any(patient_id <= 0):
        raise SourceCalibrationIntegrityError("source patient identifiers are invalid")
    if not np.all(folds == 9):
        raise SourceCalibrationIntegrityError("source NPZ must contain fold 9 only")
    if not np.all((targets == 0) | (targets == 1)):
        raise SourceCalibrationIntegrityError("source targets must be exactly binary")
    if not np.all(np.isfinite(logits)):
        raise SourceCalibrationIntegrityError("source logits must be finite")
    return {name: _readonly_copy(array) for name, array in arrays.items()}


def _verify_prediction_artifact(
    artifact: PredictionArtifact,
    *,
    safe_arrays: Mapping[str, NDArray[np.generic]],
    config: SourceCalibrationConfig,
) -> None:
    if artifact.fold_role is not FoldRole.CALIBRATION or artifact.folds != (9,):
        raise SourceCalibrationIntegrityError("prediction artifact is not fold-9 calibration data")
    if artifact.n_samples != config.source_prediction.expected_records:
        raise SourceCalibrationIntegrityError("prediction artifact record count is invalid")
    if artifact.label_order != LABEL_ORDER:
        raise SourceCalibrationIntegrityError("prediction artifact label order is invalid")
    if (
        artifact.model_name != config.model.source_artifact_model_name
        or artifact.model_seed != config.model.seed
    ):
        raise SourceCalibrationIntegrityError("prediction artifact model identity is invalid")
    if artifact.calibrated_probabilities is not None:
        raise SourceCalibrationIntegrityError("pre-calibrated source probabilities are forbidden")
    observed = {
        "ecg_id": artifact.ecg_id,
        "patient_id": artifact.patient_id,
        "strat_fold": artifact.strat_fold,
        "targets": artifact.targets,
        "raw_logits": artifact.raw_logits,
    }
    if any(not np.array_equal(observed[name], safe_arrays[name]) for name in _EXACT_ARRAY_NAMES):
        raise SourceCalibrationIntegrityError("safe NPZ and prediction artifact arrays disagree")


def _assert_source_pair_unchanged(verified: VerifiedSourceInputs) -> None:
    if _sha256_file(verified.npz_path) != verified.source_npz_sha256:
        raise SourceCalibrationIntegrityError("source NPZ changed after input verification")
    if _sha256_file(verified.sidecar_path) != verified.source_sidecar_sha256:
        raise SourceCalibrationIntegrityError("source sidecar changed after input verification")


def _role_data(*, source: SourcePredictionArrays, role: SourceRole, mask: BoolArray) -> RoleData:
    if int(np.count_nonzero(mask)) == 0:
        raise SourceCalibrationIntegrityError(f"frozen role {role.value} is empty")
    return RoleData(
        role=role,
        patient_id=_readonly_copy(source.patient_id[mask]),
        targets=_readonly_copy(source.targets[mask]),
        raw_logits=_readonly_copy(source.raw_logits[mask]),
    )


def _role_counts(data: RoleData) -> RoleCounts:
    positives = tuple(int(value) for value in data.targets.sum(axis=0))
    return RoleCounts(
        role=data.role,
        records=data.records,
        patients=data.patients,
        positive_records=PositiveRecords(
            NORM=positives[0],
            MI=positives[1],
            STTC=positives[2],
            CD=positives[3],
            HYP=positives[4],
        ),
    )


def _temperature_summary(result: TemperatureScalingResult) -> TemperatureFitSummary:
    return TemperatureFitSummary(
        method="single_positive_temperature_binary_nll",
        fit_role=SourceRole.DECISION_FIT,
        n_samples=result.n_samples,
        temperature=result.temperature,
        nll_before=result.nll_before,
        nll_after=result.nll_after,
        status=result.status,
        converged=result.converged,
        optimization_steps=result.optimization_steps,
        fitted_labels=result.fitted_labels,
        excluded_degenerate_labels=result.excluded_degenerate_labels,
    )


def _threshold_summary(result: ThresholdOptimizationResult) -> ThresholdFitSummary:
    per_label = tuple(
        LabelThresholdSummary(
            label=_label_name(item.label),
            threshold=item.threshold,
            objective="f1",
            objective_value=item.objective_value,
            positives=item.positives,
            negatives=item.negatives,
            status=item.status,
        )
        for item in result.per_label
    )
    return ThresholdFitSummary(
        method="per_label_maximum_f1",
        tie_rule="maximum_f1_then_closest_to_0.5_then_higher_threshold",
        fit_role=SourceRole.DECISION_FIT,
        n_samples=result.n_samples,
        macro_objective=result.macro_objective,
        per_label=cast(
            tuple[
                LabelThresholdSummary,
                LabelThresholdSummary,
                LabelThresholdSummary,
                LabelThresholdSummary,
                LabelThresholdSummary,
            ],
            per_label,
        ),
    )


def _conformal_summary(result: LabelwiseBinaryConformal) -> ConformalFitSummary:
    per_label = tuple(
        LabelConformalThreshold(label=_label_name(label), threshold=result.thresholds[index])
        for index, label in enumerate(LABEL_ORDER)
    )
    return ConformalFitSummary(
        method="labelwise_binary_split_conformal",
        fit_role=SourceRole.CONFORMAL_AND_OOD_THRESHOLD_FIT,
        alpha=result.alpha,
        n_samples=result.n_calibration_samples,
        quantile_rank=result.quantile_rank,
        quantile_level=result.quantile_level,
        coverage_scope="labelwise_marginal_under_exchangeability",
        individual_certainty_guarantee=False,
        per_label=cast(
            tuple[
                LabelConformalThreshold,
                LabelConformalThreshold,
                LabelConformalThreshold,
                LabelConformalThreshold,
                LabelConformalThreshold,
            ],
            per_label,
        ),
    )


def _decision_metrics(targets: Int8Array, decisions: BoolArray) -> tuple[float, float]:
    expected = targets.astype(np.bool_, copy=False)
    if expected.shape != decisions.shape or expected.shape[0] == 0:
        raise SourceCalibrationIntegrityError("decision metric arrays are invalid")
    return (
        float(np.not_equal(expected, decisions).mean()),
        float(np.all(expected == decisions, axis=1).mean()),
    )


def _resolve_project_path(root: Path, relative_path: str, *, require_file: bool) -> Path:
    if PurePosixPath(relative_path).is_absolute() or PureWindowsPath(relative_path).is_absolute():
        raise SourceCalibrationIntegrityError("project path must be relative")
    candidate = (root / Path(*PurePosixPath(relative_path).parts)).resolve(strict=False)
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise SourceCalibrationIntegrityError("project path escapes the project root") from error
    if require_file and not candidate.is_file():
        raise SourceCalibrationIntegrityError("required project file is missing")
    return candidate


def _reject_duplicate_yaml_keys(serialized: str) -> None:
    try:
        root = yaml.compose(serialized, Loader=yaml.SafeLoader)
    except yaml.YAMLError as error:
        raise SourceCalibrationConfigError("configuration is not valid YAML") from error
    visited: set[int] = set()

    def visit(node: object) -> None:
        identity = id(node)
        if identity in visited:
            raise SourceCalibrationConfigError("YAML aliases are forbidden")
        visited.add(identity)
        if isinstance(node, yaml.MappingNode):
            seen: set[tuple[str, str]] = set()
            for key_node, value_node in node.value:
                if not isinstance(key_node, yaml.ScalarNode):
                    raise SourceCalibrationConfigError("YAML mapping keys must be scalar")
                if str(key_node.tag).endswith(":merge") or str(key_node.value) == "<<":
                    raise SourceCalibrationConfigError("YAML merge keys are forbidden")
                identity_key = (str(key_node.tag), str(key_node.value))
                if identity_key in seen:
                    raise SourceCalibrationConfigError("YAML mapping keys must be unique")
                seen.add(identity_key)
                visit(value_node)
        elif isinstance(node, yaml.SequenceNode):
            for item in node.value:
                visit(item)

    if root is not None:
        visit(root)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise SourceCalibrationIntegrityError("required project file cannot be read") from error
    return f"sha256:{digest.hexdigest()}"


def _sha256_bytes(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _readonly_copy[ArrayDType: np.generic](
    array: NDArray[ArrayDType],
) -> NDArray[ArrayDType]:
    result = np.asarray(array).copy()
    result.flags.writeable = False
    return result


def _atomic_write_new(path: Path, payload: bytes) -> None:
    if path.exists():
        raise SourceCalibrationOutputError("immutable artifact already exists")
    descriptor, raw_temp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp = Path(raw_temp)
    linked = False
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temp, path)
        linked = True
    except FileExistsError as error:
        raise SourceCalibrationOutputError("immutable artifact already exists") from error
    except OSError as error:
        raise SourceCalibrationOutputError("atomic artifact commit failed") from error
    finally:
        with suppress(OSError):
            temp.unlink()
    if not linked:  # pragma: no cover - defensive invariant
        raise SourceCalibrationOutputError("atomic artifact commit did not complete")


def _commit_staged_directory(staging_root: Path, output_root: Path) -> None:
    if output_root.exists():
        raise SourceCalibrationOutputError("immutable output root already exists")
    try:
        os.rename(staging_root, output_root)
    except FileExistsError as error:
        raise SourceCalibrationOutputError("immutable output root already exists") from error
    except OSError as error:
        raise SourceCalibrationOutputError("atomic output-root commit failed") from error


def _remove_staging_root(staging_root: Path, *, expected_parent: Path) -> None:
    resolved = staging_root.resolve(strict=False)
    if resolved.parent != expected_parent.resolve(strict=True) or not resolved.name.startswith("."):
        raise SourceCalibrationOutputError("refusing to remove an unexpected staging root")
    try:
        shutil.rmtree(resolved, ignore_errors=False)
    except FileNotFoundError:
        return
    except OSError as error:
        raise SourceCalibrationOutputError("could not remove failed output staging root") from error


def _run_git(project_root: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=project_root,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="strict",
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError, UnicodeError) as error:
        raise SourceCalibrationIntegrityError("Git preflight could not be executed") from error
    if completed.returncode != 0:
        raise SourceCalibrationIntegrityError("Git preflight command failed")
    return completed.stdout.strip()


def _validate_code_revision(revision: str) -> None:
    if not isinstance(revision, str) or _GIT_REVISION_PATTERN.fullmatch(revision) is None:
        raise SourceCalibrationIntegrityError("code revision must be a committed Git SHA")


def _failure_code(error: Exception) -> FailureCode:
    if isinstance(error, SourceCalibrationOutputError):
        return FailureCode.OUTPUT_COMMIT_FAILED
    if isinstance(
        error,
        (
            SourceCalibrationConfigError,
            SourceCalibrationIntegrityError,
            PredictionArtifactError,
            ValidationError,
        ),
    ):
        return FailureCode.SOURCE_CONTRACT_FAILED
    if isinstance(error, ValueError):
        return FailureCode.FIT_FAILED
    return FailureCode.INTERNAL_FAILURE


def _looks_absolute_path(value: str) -> bool:
    return PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute()


def _label_name(value: str) -> LabelName:
    if value not in LABEL_ORDER:
        raise SourceCalibrationIntegrityError("label order differs from the frozen contract")
    return cast(LabelName, value)


__all__ = [
    "FittedEntropyGate",
    "FittedSourceComponents",
    "RoleData",
    "SourcePartitions",
    "SourcePredictionArrays",
    "VerifiedSourceInputs",
    "assert_aggregate_only_result",
    "assert_complete_release_ready",
    "build_source_calibration_result",
    "evaluate_source_validation",
    "fit_entropy_gate",
    "fit_source_components",
    "load_source_calibration_config",
    "load_source_calibration_result_bytes",
    "load_verified_source_predictions",
    "partition_source_predictions",
    "patient_split_fraction",
    "patient_split_role",
    "prepare_source_calibration",
    "replace_validation_role",
    "verify_source_inputs",
    "verify_clean_git_revision",
]
