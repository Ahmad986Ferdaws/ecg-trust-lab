"""Strict configuration and aggregate-only result contracts for source calibration."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from enum import StrEnum
from pathlib import PurePosixPath, PureWindowsPath
from typing import Annotated, Final, Literal, Self

from pydantic import (
    AwareDatetime,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

SOURCE_CALIBRATION_SCHEMA_VERSION: Final = 1
SOURCE_CALIBRATION_ARTIFACT_TYPE: Final = "ecg_trust.source_calibration_result"
FAILURE_RECEIPT_ARTIFACT_TYPE: Final = "ecg_trust.source_calibration_failure"
LABEL_ORDER: Final = ("NORM", "MI", "STTC", "CD", "HYP")
SPLIT_ALGORITHM_TEXT: Final = (
    'sha256(utf8(salt + "|" + base10_patient_id)); take the first 8 digest bytes '
    "as an unsigned big-endian integer; divide by 2^64"
)
SPLIT_ALGORITHM_ID: Final = "sha256_first8_uint64_fraction_v1"
RESULT_FILENAME: Final = "source-calibration-result.json"
FAILURE_RECEIPT_FILENAME: Final = "failure-receipt.json"

StrictText = Annotated[str, StringConstraints(strict=True, min_length=1, max_length=2000)]
Identifier = Annotated[
    str,
    StringConstraints(
        strict=True,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
        min_length=1,
        max_length=160,
    ),
]
HexSha256 = Annotated[
    str,
    StringConstraints(strict=True, pattern=r"^[0-9a-f]{64}$", min_length=64, max_length=64),
]
Sha256Digest = Annotated[
    str,
    StringConstraints(
        strict=True,
        pattern=r"^sha256:[0-9a-f]{64}$",
        min_length=71,
        max_length=71,
    ),
]
GitRevision = Annotated[
    str,
    StringConstraints(strict=True, pattern=r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$"),
]
PositiveInt = Annotated[int, Field(strict=True, ge=1)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
FiniteFloat = Annotated[float, Field(strict=True, allow_inf_nan=False)]
UnitFloat = Annotated[float, Field(strict=True, ge=0.0, le=1.0, allow_inf_nan=False)]
PositiveFloat = Annotated[float, Field(strict=True, gt=0.0, allow_inf_nan=False)]


def _require_true(value: object) -> bool:
    if value is not True:
        raise ValueError("value must be the boolean true")
    return True


def _require_false(value: object) -> bool:
    if value is not False:
        raise ValueError("value must be the boolean false")
    return False


def _require_one(value: object) -> int:
    if type(value) is not int or value != 1:
        raise ValueError("value must be the integer 1")
    return 1


def _require_nine(value: object) -> int:
    if type(value) is not int or value != 9:
        raise ValueError("value must be the integer 9")
    return 9


def _require_fifteen(value: object) -> int:
    if type(value) is not int or value != 15:
        raise ValueError("value must be the integer 15")
    return 15


StrictTrue = Annotated[Literal[True], BeforeValidator(_require_true)]
StrictFalse = Annotated[Literal[False], BeforeValidator(_require_false)]
StrictOne = Annotated[Literal[1], BeforeValidator(_require_one)]
StrictNine = Annotated[Literal[9], BeforeValidator(_require_nine)]
StrictFifteen = Annotated[Literal[15], BeforeValidator(_require_fifteen)]
StrictBool = Annotated[bool, Field(strict=True)]


class SourceCalibrationError(ValueError):
    """Base class for source-calibration contract and integrity failures."""


class SourceCalibrationConfigError(SourceCalibrationError):
    """Raised when the frozen YAML configuration is invalid."""


class SourceCalibrationIntegrityError(SourceCalibrationError):
    """Raised when an input or emitted artifact fails integrity verification."""


class SourceCalibrationOutputError(SourceCalibrationError):
    """Raised when immutable local output cannot be committed."""


class StrictFrozenModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        revalidate_instances="always",
        validate_default=True,
        use_enum_values=False,
    )


def _validate_relative_path(value: str) -> str:
    if "\\" in value:
        raise ValueError("paths must use project-relative POSIX separators")
    path = PurePosixPath(value)
    if path.is_absolute() or PureWindowsPath(value).is_absolute() or ".." in path.parts:
        raise ValueError("paths must stay relative to the project root")
    if not path.parts or any(part in {"", "."} for part in path.parts):
        raise ValueError("paths must be non-empty normalized relative paths")
    return value


class PositiveRecords(StrictFrozenModel):
    NORM: NonNegativeInt
    MI: NonNegativeInt
    STTC: NonNegativeInt
    CD: NonNegativeInt
    HYP: NonNegativeInt


class ExpectedRoleCounts(StrictFrozenModel):
    records: PositiveInt
    patients: PositiveInt
    positive_records: PositiveRecords

    @model_validator(mode="after")
    def _counts_are_possible(self) -> Self:
        if self.patients > self.records:
            raise ValueError("patients cannot exceed records")
        if any(value > self.records for value in self.positive_records.model_dump().values()):
            raise ValueError("positive record counts cannot exceed role records")
        return self


class FileHashReference(StrictFrozenModel):
    path: StrictText
    file_sha256: HexSha256

    @field_validator("path")
    @classmethod
    def _path_is_relative(cls, value: str) -> str:
        return _validate_relative_path(value)


class HistoricalPolicyReference(FileHashReference):
    use: Literal["comparison_only_not_reused_for_vnext_fitting"]


class ModelConfig(StrictFrozenModel):
    member_id: Identifier
    source_artifact_model_name: Identifier
    architecture: Literal["resnet1d"]
    seed: NonNegativeInt
    selection_rule: Literal["development_selected_primary_architecture_and_first_fixed_seed"]
    checkpoint_sha256: HexSha256
    demo_binding: FileHashReference
    historical_policy: HistoricalPolicyReference


class SourcePredictionConfig(StrictFrozenModel):
    role: Literal["ptbxl_fold9_source_calibration_pool"]
    npz_path: StrictText
    npz_sha256: HexSha256
    sidecar_path: StrictText
    sidecar_sha256: HexSha256
    expected_records: PositiveInt
    label_order: tuple[Literal["NORM", "MI", "STTC", "CD", "HYP"], ...]

    @field_validator("npz_path", "sidecar_path")
    @classmethod
    def _paths_are_relative(cls, value: str) -> str:
        return _validate_relative_path(value)

    @model_validator(mode="after")
    def _prediction_pair_is_canonical(self) -> Self:
        if self.label_order != LABEL_ORDER:
            raise ValueError(f"label_order must be exactly {LABEL_ORDER!r}")
        npz = PurePosixPath(self.npz_path)
        sidecar = PurePosixPath(self.sidecar_path)
        if npz.suffix != ".npz" or sidecar.suffix != ".json":
            raise ValueError("source prediction paths must end in .npz and .json")
        if npz.with_suffix("") != sidecar.with_suffix(""):
            raise ValueError("source NPZ and sidecar must have the same stem")
        return self


class SplitRanges(StrictFrozenModel):
    decision_fit: Literal["[0.0,0.4)"]
    conformal_and_ood_threshold_fit: Literal["[0.4,0.8)"]
    source_validation: Literal["[0.8,1.0)"]


class ExpectedSplits(StrictFrozenModel):
    decision_fit: ExpectedRoleCounts
    conformal_and_ood_threshold_fit: ExpectedRoleCounts
    source_validation: ExpectedRoleCounts


class PatientSplitConfig(StrictFrozenModel):
    unit: Literal["patient_id"]
    algorithm: Literal[
        'sha256(utf8(salt + "|" + base10_patient_id)); take the first 8 digest bytes '
        "as an unsigned big-endian integer; divide by 2^64"
    ]
    salt: Identifier
    ranges: SplitRanges
    expected: ExpectedSplits
    design_provenance: StrictText


class LegacyEntropyGateConfig(StrictFrozenModel):
    method: Literal["mean_normalized_binary_entropy"]
    target_coverage: UnitFloat
    tie_rule: Literal["retain_all_scores_less_than_or_equal_to_frozen_order_statistic"]

    @field_validator("target_coverage")
    @classmethod
    def _coverage_is_frozen(cls, value: float) -> float:
        if value != 0.8:
            raise ValueError("legacy entropy target coverage must be 0.8")
        return value


class DecisionFitConfig(StrictFrozenModel):
    temperature_scaling: Literal["single_positive_temperature_binary_nll"]
    classification_thresholds: Literal["per_label_maximum_f1"]
    classification_threshold_tie_rule: Literal[
        "maximum_f1_then_closest_to_0.5_then_higher_threshold"
    ]
    legacy_entropy_gate: LegacyEntropyGateConfig


class ConformalConfig(StrictFrozenModel):
    method: Literal["labelwise_binary_split_conformal"]
    alpha: UnitFloat
    fit_role: Literal["conformal_and_ood_threshold_fit"]
    coverage_scope: Literal["labelwise_marginal_under_exchangeability"]
    individual_certainty_guarantee: StrictFalse

    @model_validator(mode="after")
    def _alpha_is_frozen(self) -> Self:
        if self.alpha != 0.1:
            raise ValueError("conformal alpha must be frozen at 0.1")
        return self


class OpenWorldConfig(StrictFrozenModel):
    primary_method: Literal["shrinkage_mahalanobis_embedding_distance"]
    embedding: Literal["frozen_resnet_preclassifier_global_average_pool"]
    reference_role: Literal["ptbxl_folds_1_to_8_training_reference"]
    threshold_role: Literal["conformal_and_ood_threshold_fit"]
    threshold_inlier_coverage: UnitFloat
    shrinkage: UnitFloat
    ridge: PositiveFloat
    target_site_fitting: Literal["forbidden"]

    @model_validator(mode="after")
    def _parameters_are_frozen(self) -> Self:
        if self.threshold_inlier_coverage != 0.95:
            raise ValueError("OOD threshold inlier coverage must be 0.95")
        if self.shrinkage != 0.1 or self.ridge != 0.000001:
            raise ValueError("OOD shrinkage and ridge differ from the frozen protocol")
        return self


ValidationReportItem = Literal[
    "labelwise_conformal_coverage_and_set_size",
    "legacy_gate_coverage",
    "hamming_loss_and_exact_match_at_frozen_thresholds",
    "auroc_average_precision_brier_ece15",
    "ood_source_false_rejection",
]


class SourceValidationConfig(StrictFrozenModel):
    tuning_allowed: StrictFalse
    report: tuple[ValidationReportItem, ...]
    minimum_positive_records_for_label_statement: PositiveInt

    @model_validator(mode="after")
    def _report_is_exact(self) -> Self:
        expected: tuple[ValidationReportItem, ...] = (
            "labelwise_conformal_coverage_and_set_size",
            "legacy_gate_coverage",
            "hamming_loss_and_exact_match_at_frozen_thresholds",
            "auroc_average_precision_brier_ece15",
            "ood_source_false_rejection",
        )
        if self.report != expected:
            raise ValueError("source_validation.report differs from the frozen report")
        return self


ForbiddenSource = Literal[
    "ptbxl_fold10",
    "sph",
    "future_external_observed_sites",
    "future_external_lockbox_sites",
]


class ExecutionConfig(StrictFrozenModel):
    require_clean_committed_revision: StrictTrue
    require_verified_input_hashes: StrictTrue
    output_root: StrictText
    output_root_must_be_absent: StrictTrue
    automatic_publication: StrictFalse
    raw_ids_or_row_arrays_public: StrictFalse

    @field_validator("output_root")
    @classmethod
    def _output_is_relative(cls, value: str) -> str:
        return _validate_relative_path(value)


class ClaimsConfig(StrictFrozenModel):
    scope: Literal["retrospective_source_domain_development_only"]
    limitations: tuple[Identifier, ...]

    @model_validator(mode="after")
    def _limitations_are_exact(self) -> Self:
        expected = (
            "no_external_lockbox_evaluation",
            "no_clinical_validation",
            "conformal_coverage_is_marginal_not_individual",
            "ood_score_does_not_identify_unknown_disease",
            "thresholds_are_provisional_research_values",
        )
        if self.limitations != expected:
            raise ValueError("claims.limitations differs from the frozen safety boundary")
        return self


class SourceCalibrationConfig(StrictFrozenModel):
    schema_version: StrictOne
    protocol_id: Literal["trust-sentinel-source-calibration-v1"]
    status: Literal["frozen_pre_execution"]
    frozen_at_utc: AwareDatetime
    research_only: StrictTrue
    purpose: StrictText
    model: ModelConfig
    source_prediction: SourcePredictionConfig
    patient_split: PatientSplitConfig
    decision_fit: DecisionFitConfig
    conformal: ConformalConfig
    open_world: OpenWorldConfig
    source_validation: SourceValidationConfig
    forbidden_fit_or_selection_sources: tuple[ForbiddenSource, ...]
    execution: ExecutionConfig
    claims: ClaimsConfig

    @field_validator("frozen_at_utc")
    @classmethod
    def _frozen_timestamp_is_utc(cls, value: datetime) -> datetime:
        offset = value.utcoffset()
        if offset is None or offset.total_seconds() != 0:
            raise ValueError("frozen_at_utc must use UTC")
        return value

    @model_validator(mode="after")
    def _cross_contract_is_frozen(self) -> Self:
        if self.forbidden_fit_or_selection_sources != (
            "ptbxl_fold10",
            "sph",
            "future_external_observed_sites",
            "future_external_lockbox_sites",
        ):
            raise ValueError("forbidden sources differ from the frozen protocol")
        expected = self.patient_split.expected
        total = (
            expected.decision_fit.records
            + expected.conformal_and_ood_threshold_fit.records
            + expected.source_validation.records
        )
        if total != self.source_prediction.expected_records:
            raise ValueError("expected split records do not sum to source expected_records")
        return self


class SourceRole(StrEnum):
    DECISION_FIT = "decision_fit"
    CONFORMAL_AND_OOD_THRESHOLD_FIT = "conformal_and_ood_threshold_fit"
    SOURCE_VALIDATION = "source_validation"


class RoleCounts(StrictFrozenModel):
    role: SourceRole
    records: PositiveInt
    patients: PositiveInt
    positive_records: PositiveRecords


class SplitEvidence(StrictFrozenModel):
    unit: Literal["patient"]
    algorithm: Literal["sha256_first8_uint64_fraction_v1"]
    salt_sha256: Sha256Digest
    assignment_sha256: Sha256Digest
    roles: tuple[RoleCounts, RoleCounts, RoleCounts]

    @model_validator(mode="after")
    def _roles_are_canonical(self) -> Self:
        if tuple(item.role for item in self.roles) != tuple(SourceRole):
            raise ValueError("split roles must appear once in canonical order")
        return self


class SourceProvenance(StrictFrozenModel):
    config_file_sha256: Sha256Digest
    source_npz_sha256: Sha256Digest
    source_sidecar_sha256: Sha256Digest
    prediction_artifact_sha256: Sha256Digest
    source_alignment_sha256: Sha256Digest
    source_bundle_sha256: Sha256Digest
    checkpoint_sha256: Sha256Digest
    demo_binding_file_sha256: Sha256Digest
    historical_policy_file_sha256: Sha256Digest
    experiment_protocol_sha256: Sha256Digest
    code_revision: GitRevision
    model_member_id: Identifier
    source_artifact_model_name: Identifier
    architecture: Literal["resnet1d"]
    seed: NonNegativeInt
    source_fold: StrictNine


class TemperatureFitSummary(StrictFrozenModel):
    method: Literal["single_positive_temperature_binary_nll"]
    fit_role: Literal[SourceRole.DECISION_FIT]
    n_samples: PositiveInt
    temperature: PositiveFloat
    nll_before: FiniteFloat | None
    nll_after: FiniteFloat | None
    status: Identifier
    converged: StrictBool
    optimization_steps: NonNegativeInt
    fitted_labels: tuple[Identifier, ...]
    excluded_degenerate_labels: tuple[Identifier, ...]


class LabelThresholdSummary(StrictFrozenModel):
    label: Literal["NORM", "MI", "STTC", "CD", "HYP"]
    threshold: UnitFloat
    objective: Literal["f1"]
    objective_value: UnitFloat | None
    positives: NonNegativeInt
    negatives: NonNegativeInt
    status: Identifier


class ThresholdFitSummary(StrictFrozenModel):
    method: Literal["per_label_maximum_f1"]
    tie_rule: Literal["maximum_f1_then_closest_to_0.5_then_higher_threshold"]
    fit_role: Literal[SourceRole.DECISION_FIT]
    n_samples: PositiveInt
    macro_objective: UnitFloat | None
    per_label: tuple[
        LabelThresholdSummary,
        LabelThresholdSummary,
        LabelThresholdSummary,
        LabelThresholdSummary,
        LabelThresholdSummary,
    ]

    @model_validator(mode="after")
    def _labels_are_canonical(self) -> Self:
        if tuple(item.label for item in self.per_label) != LABEL_ORDER:
            raise ValueError("threshold labels must use canonical order")
        return self


class EntropyGateSummary(StrictFrozenModel):
    method: Literal["mean_normalized_binary_entropy"]
    fit_role: Literal[SourceRole.DECISION_FIT]
    target_coverage: UnitFloat
    tie_rule: Literal["retain_all_scores_less_than_or_equal_to_frozen_order_statistic"]
    maximum_entropy: UnitFloat
    selected_count: PositiveInt
    fit_count: PositiveInt
    achieved_coverage: UnitFloat

    @field_validator("target_coverage")
    @classmethod
    def _coverage_is_frozen(cls, value: float) -> float:
        if value != 0.8:
            raise ValueError("entropy gate target coverage must be 0.8")
        return value


class LabelConformalThreshold(StrictFrozenModel):
    label: Literal["NORM", "MI", "STTC", "CD", "HYP"]
    threshold: UnitFloat


class ConformalFitSummary(StrictFrozenModel):
    method: Literal["labelwise_binary_split_conformal"]
    fit_role: Literal[SourceRole.CONFORMAL_AND_OOD_THRESHOLD_FIT]
    alpha: UnitFloat
    n_samples: PositiveInt
    quantile_rank: PositiveInt
    quantile_level: UnitFloat
    coverage_scope: Literal["labelwise_marginal_under_exchangeability"]
    individual_certainty_guarantee: StrictFalse
    per_label: tuple[
        LabelConformalThreshold,
        LabelConformalThreshold,
        LabelConformalThreshold,
        LabelConformalThreshold,
        LabelConformalThreshold,
    ]

    @model_validator(mode="after")
    def _labels_are_canonical(self) -> Self:
        if tuple(item.label for item in self.per_label) != LABEL_ORDER:
            raise ValueError("conformal labels must use canonical order")
        return self


class FrozenComponentsBody(StrictFrozenModel):
    temperature: TemperatureFitSummary
    thresholds: ThresholdFitSummary
    entropy_gate: EntropyGateSummary
    conformal: ConformalFitSummary


class FrozenComponents(FrozenComponentsBody):
    component_sha256: Sha256Digest

    @model_validator(mode="after")
    def _hash_verifies(self) -> Self:
        observed = canonical_sha256(self.model_dump(mode="json", exclude={"component_sha256"}))
        if observed != self.component_sha256:
            raise ValueError("component_sha256 does not match frozen components")
        return self


class LabelMetricSummary(StrictFrozenModel):
    label: Literal["NORM", "MI", "STTC", "CD", "HYP"]
    positives: NonNegativeInt
    negatives: NonNegativeInt
    minimum_positive_records: PositiveInt
    statement_status: Literal["SUFFICIENT_EVIDENCE", "INSUFFICIENT_EVIDENCE"]
    roc_auc: UnitFloat | None
    average_precision: UnitFloat | None
    brier_score: UnitFloat
    ece15: UnitFloat
    degenerate_reason: StrictText | None

    @model_validator(mode="after")
    def _statement_status_is_derived(self) -> Self:
        expected = (
            "SUFFICIENT_EVIDENCE"
            if self.positives >= self.minimum_positive_records
            else "INSUFFICIENT_EVIDENCE"
        )
        if self.statement_status != expected:
            raise ValueError("statement_status does not follow minimum positive evidence")
        return self


class MacroMetricSummary(StrictFrozenModel):
    roc_auc: UnitFloat | None
    average_precision: UnitFloat | None
    brier_score: UnitFloat
    ece15: UnitFloat
    roc_auc_labels: NonNegativeInt
    average_precision_labels: NonNegativeInt


class ThresholdValidationSummary(StrictFrozenModel):
    frozen_component_sha256: Sha256Digest
    hamming_loss: UnitFloat
    exact_match_accuracy: UnitFloat


class EntropyValidationSummary(StrictFrozenModel):
    frozen_component_sha256: Sha256Digest
    maximum_entropy: UnitFloat
    selected_count: NonNegativeInt
    validation_count: PositiveInt
    achieved_coverage: UnitFloat
    retained_hamming_loss: UnitFloat | None
    retained_exact_match_accuracy: UnitFloat | None


class LabelConformalCoverage(StrictFrozenModel):
    label: Literal["NORM", "MI", "STTC", "CD", "HYP"]
    empirical_coverage: UnitFloat
    mean_set_size: Annotated[float, Field(strict=True, ge=0.0, le=2.0, allow_inf_nan=False)]


class ConformalValidationSummary(StrictFrozenModel):
    frozen_component_sha256: Sha256Digest
    coverage_scope: Literal["labelwise_marginal_under_exchangeability"]
    individual_certainty_guarantee: StrictFalse
    marginal_coverage: UnitFloat
    joint_sample_coverage: UnitFloat
    mean_set_size: Annotated[float, Field(strict=True, ge=0.0, le=2.0, allow_inf_nan=False)]
    singleton_fraction: UnitFloat
    empty_fraction: UnitFloat
    both_fraction: UnitFloat
    per_label: tuple[
        LabelConformalCoverage,
        LabelConformalCoverage,
        LabelConformalCoverage,
        LabelConformalCoverage,
        LabelConformalCoverage,
    ]


class SourceValidationSummary(StrictFrozenModel):
    evaluation_role: Literal[SourceRole.SOURCE_VALIDATION]
    tuning_allowed: StrictFalse
    records: PositiveInt
    patients: PositiveInt
    ece_bins: StrictFifteen
    per_label: tuple[
        LabelMetricSummary,
        LabelMetricSummary,
        LabelMetricSummary,
        LabelMetricSummary,
        LabelMetricSummary,
    ]
    macro: MacroMetricSummary
    threshold_decisions: ThresholdValidationSummary
    entropy_gate: EntropyValidationSummary
    conformal: ConformalValidationSummary

    @model_validator(mode="after")
    def _labels_are_canonical(self) -> Self:
        if tuple(item.label for item in self.per_label) != LABEL_ORDER:
            raise ValueError("validation metric labels must use canonical order")
        if tuple(item.label for item in self.conformal.per_label) != LABEL_ORDER:
            raise ValueError("validation conformal labels must use canonical order")
        return self


class OpenWorldPendingSummary(StrictFrozenModel):
    method: Literal["shrinkage_mahalanobis_embedding_distance"]
    status: Literal["PENDING"]
    artifact_sha256: Literal[None]
    threshold_fitted: StrictFalse
    source_false_rejection_evaluated: StrictFalse
    release_ready: StrictFalse
    reference_alignment_verified: StrictFalse
    embedding_device: Literal[None]
    embedding_precision: Literal[None]
    reason_code: Literal["REFERENCE_AND_THRESHOLD_EMBEDDINGS_NOT_PROVIDED"]


class ClaimBoundary(StrictFrozenModel):
    scope: Literal["retrospective_source_domain_development_only"]
    research_only: StrictTrue
    clinical_validation: StrictFalse
    limitations: tuple[Identifier, ...]


class SourceCalibrationResultBody(StrictFrozenModel):
    schema_version: StrictOne
    artifact_type: Literal["ecg_trust.source_calibration_result"]
    protocol_id: Literal["trust-sentinel-source-calibration-v1"]
    status: Literal["PREPARED_NOT_RELEASE_READY"]
    frozen_at_utc: AwareDatetime
    provenance: SourceProvenance
    split: SplitEvidence
    frozen_components: FrozenComponents
    source_validation: SourceValidationSummary
    open_world: OpenWorldPendingSummary
    claims: ClaimBoundary


class SourceCalibrationResult(SourceCalibrationResultBody):
    artifact_sha256: Sha256Digest

    @model_validator(mode="after")
    def _self_hash_verifies(self) -> Self:
        observed = canonical_sha256(self.model_dump(mode="json", exclude={"artifact_sha256"}))
        if observed != self.artifact_sha256:
            raise ValueError("artifact_sha256 does not match source calibration result")
        return self


class FailureCode(StrEnum):
    SOURCE_CONTRACT_FAILED = "SOURCE_CONTRACT_FAILED"
    FIT_FAILED = "FIT_FAILED"
    OUTPUT_COMMIT_FAILED = "OUTPUT_COMMIT_FAILED"
    INTERNAL_FAILURE = "INTERNAL_FAILURE"


class FailureReceiptBody(StrictFrozenModel):
    schema_version: StrictOne
    artifact_type: Literal["ecg_trust.source_calibration_failure"]
    protocol_id: Literal["trust-sentinel-source-calibration-v1"]
    status: Literal["FAILED"]
    frozen_at_utc: AwareDatetime
    config_file_sha256: Sha256Digest
    code_revision: GitRevision
    failure_code: FailureCode
    contains_raw_ids_or_rows: StrictFalse
    retry_requires_new_output_root: StrictTrue


class FailureReceipt(FailureReceiptBody):
    artifact_sha256: Sha256Digest

    @model_validator(mode="after")
    def _self_hash_verifies(self) -> Self:
        observed = canonical_sha256(self.model_dump(mode="json", exclude={"artifact_sha256"}))
        if observed != self.artifact_sha256:
            raise ValueError("artifact_sha256 does not match failure receipt")
        return self


def canonical_json_bytes(value: dict[str, object]) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise SourceCalibrationIntegrityError("value is not finite canonical JSON") from error


def canonical_sha256(value: dict[str, object]) -> str:
    return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def seal_frozen_components(body: FrozenComponentsBody) -> FrozenComponents:
    payload = body.model_dump(mode="python")
    json_payload = body.model_dump(mode="json")
    return FrozenComponents.model_validate(
        {**payload, "component_sha256": canonical_sha256(json_payload)}
    )


def seal_source_calibration_result(body: SourceCalibrationResultBody) -> SourceCalibrationResult:
    payload = body.model_dump(mode="python")
    json_payload = body.model_dump(mode="json")
    return SourceCalibrationResult.model_validate(
        {**payload, "artifact_sha256": canonical_sha256(json_payload)}
    )


def seal_failure_receipt(body: FailureReceiptBody) -> FailureReceipt:
    payload = body.model_dump(mode="python")
    json_payload = body.model_dump(mode="json")
    return FailureReceipt.model_validate(
        {**payload, "artifact_sha256": canonical_sha256(json_payload)}
    )


def result_json_bytes(result: SourceCalibrationResult) -> bytes:
    return canonical_json_bytes(result.model_dump(mode="json")) + b"\n"


def failure_receipt_json_bytes(receipt: FailureReceipt) -> bytes:
    return canonical_json_bytes(receipt.model_dump(mode="json")) + b"\n"


__all__ = [
    "FAILURE_RECEIPT_ARTIFACT_TYPE",
    "FAILURE_RECEIPT_FILENAME",
    "LABEL_ORDER",
    "RESULT_FILENAME",
    "SOURCE_CALIBRATION_ARTIFACT_TYPE",
    "SOURCE_CALIBRATION_SCHEMA_VERSION",
    "SPLIT_ALGORITHM_ID",
    "SPLIT_ALGORITHM_TEXT",
    "ClaimBoundary",
    "ConformalFitSummary",
    "ConformalValidationSummary",
    "EntropyGateSummary",
    "EntropyValidationSummary",
    "FailureCode",
    "FailureReceipt",
    "FailureReceiptBody",
    "FrozenComponents",
    "FrozenComponentsBody",
    "LabelConformalCoverage",
    "LabelConformalThreshold",
    "LabelMetricSummary",
    "LabelThresholdSummary",
    "MacroMetricSummary",
    "OpenWorldPendingSummary",
    "PositiveRecords",
    "RoleCounts",
    "SourceCalibrationConfig",
    "SourceCalibrationConfigError",
    "SourceCalibrationError",
    "SourceCalibrationIntegrityError",
    "SourceCalibrationOutputError",
    "SourceCalibrationResult",
    "SourceCalibrationResultBody",
    "SourceProvenance",
    "SourceRole",
    "SourceValidationSummary",
    "SplitEvidence",
    "TemperatureFitSummary",
    "ThresholdFitSummary",
    "ThresholdValidationSummary",
    "canonical_json_bytes",
    "canonical_sha256",
    "failure_receipt_json_bytes",
    "result_json_bytes",
    "seal_failure_receipt",
    "seal_frozen_components",
    "seal_source_calibration_result",
]
