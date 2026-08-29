"""Strict immutable contracts for Trust Sentinel OOD-completion evidence.

This module deliberately does not perform waveform inference or private-array
I/O.  It defines the canonical public evidence that may be produced after a
separate, frozen OOD-completion execution.  The original source-calibration v1
artifact is nested without changing its permanently pending schema.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime
from enum import StrEnum
from itertools import pairwise
from pathlib import PurePosixPath, PureWindowsPath
from typing import Annotated, Final, Literal, Self

from pydantic import (
    AwareDatetime,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)

from ecg_trust.open_world import (
    MahalanobisValidationError,
    ShrinkageMahalanobisDetector,
)
from ecg_trust.source_calibration.models import SourceCalibrationResult

OOD_COMPLETION_SCHEMA_VERSION: Final = 1
OOD_COMPLETION_PROTOCOL_ID: Final = "trust-sentinel-ood-completion-v1"
OOD_COMPLETION_ARTIFACT_TYPE: Final = "ecg_trust.ood_completion_result"
DISTRIBUTION_POLICY_ARTIFACT_TYPE: Final = "ecg_trust.distribution_policy"
OOD_COMPLETION_FAILURE_ARTIFACT_TYPE: Final = "ecg_trust.ood_completion_failure"
OOD_COMPLETION_SUCCESS_ARTIFACT_TYPE: Final = "ecg_trust.ood_completion_success_manifest"

OOD_COMPLETION_RESULT_FILENAME: Final = "ood-completion-result.json"
DISTRIBUTION_POLICY_FILENAME: Final = "distribution-policy.json"
OOD_COMPLETION_FAILURE_FILENAME: Final = "failure-receipt.json"
OOD_COMPLETION_SUCCESS_FILENAME: Final = "success-manifest.json"

MAX_DISTRIBUTION_POLICY_BYTES: Final = 64 * 1024 * 1024
MAX_OOD_COMPLETION_RESULT_BYTES: Final = 4 * 1024 * 1024
MAX_OOD_COMPLETION_FAILURE_BYTES: Final = 64 * 1024
MAX_OOD_COMPLETION_SUCCESS_BYTES: Final = 64 * 1024

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
NonNegativeFloat = Annotated[float, Field(strict=True, ge=0.0, allow_inf_nan=False)]
PositiveFloat = Annotated[float, Field(strict=True, gt=0.0, allow_inf_nan=False)]
UnitFloat = Annotated[float, Field(strict=True, ge=0.0, le=1.0, allow_inf_nan=False)]
OpenUnitFloat = Annotated[float, Field(strict=True, gt=0.0, lt=1.0, allow_inf_nan=False)]
StrictBool = Annotated[bool, Field(strict=True)]


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


def _require_point_nine_five(value: object) -> float:
    if type(value) is not float or value != 0.95:
        raise ValueError("value must be exactly 0.95")
    return 0.95


def _require_point_zero_five(value: object) -> float:
    if type(value) is not float or value != 0.05:
        raise ValueError("value must be exactly 0.05")
    return 0.05


def _require_integer(value: object, *, expected: int) -> int:
    if type(value) is not int or value != expected:
        raise ValueError(f"value must be the integer {expected}")
    return expected


StrictTrue = Annotated[Literal[True], BeforeValidator(_require_true)]
StrictFalse = Annotated[Literal[False], BeforeValidator(_require_false)]
StrictOne = Annotated[Literal[1], BeforeValidator(_require_one)]
StrictPointNineFive = Annotated[
    float,
    BeforeValidator(_require_point_nine_five),
    Field(strict=True, allow_inf_nan=False),
]
StrictPointZeroFive = Annotated[
    float,
    BeforeValidator(_require_point_zero_five),
    Field(strict=True, allow_inf_nan=False),
]
StrictTwo = Annotated[
    Literal[2], BeforeValidator(lambda value: _require_integer(value, expected=2))
]
StrictFour = Annotated[
    Literal[4], BeforeValidator(lambda value: _require_integer(value, expected=4))
]
StrictOneTwentyEight = Annotated[
    Literal[128], BeforeValidator(lambda value: _require_integer(value, expected=128))
]
StrictFiveTwelve = Annotated[
    Literal[512], BeforeValidator(lambda value: _require_integer(value, expected=512))
]
StrictTenThousand = Annotated[
    Literal[10000], BeforeValidator(lambda value: _require_integer(value, expected=10000))
]
StrictBootstrapSeed = Annotated[
    Literal[20260829],
    BeforeValidator(lambda value: _require_integer(value, expected=20260829)),
]
StrictModelSeed = Annotated[
    Literal[2026], BeforeValidator(lambda value: _require_integer(value, expected=2026))
]
StrictCudnnVersion = Annotated[
    Literal[92000], BeforeValidator(lambda value: _require_integer(value, expected=92000))
]
StrictSelectedRecordCount = Annotated[
    Literal[18383], BeforeValidator(lambda value: _require_integer(value, expected=18383))
]
StrictSelectedFileCount = Annotated[
    Literal[36766], BeforeValidator(lambda value: _require_integer(value, expected=36766))
]


class OODCompletionError(ValueError):
    """Base class for OOD-completion contract and integrity failures."""


class OODCompletionIntegrityError(OODCompletionError):
    """Raised when canonical OOD evidence fails integrity verification."""


class StrictFrozenModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        revalidate_instances="always",
        validate_default=True,
        use_enum_values=False,
    )


class MahalanobisDetectorArtifact(StrictFrozenModel):
    """Strict JSON form of :class:`ShrinkageMahalanobisDetector`."""

    schema_version: StrictOne
    artifact_type: Literal["ecg_trust.shrinkage_mahalanobis_detector"]
    score_direction: Literal["higher_is_more_out_of_distribution"]
    fit_scope: Literal["development_reference_plus_source_calibration_only"]
    mean: tuple[FiniteFloat, ...] = Field(min_length=1)
    precision: tuple[tuple[FiniteFloat, ...], ...] = Field(min_length=1)
    threshold: NonNegativeFloat
    embedding_dim: PositiveInt
    shrinkage: UnitFloat
    ridge: PositiveFloat
    inlier_coverage: OpenUnitFloat
    n_fit_samples: PositiveInt
    n_threshold_samples: PositiveInt
    quantile_rank: PositiveInt
    threshold_rule: Literal["ceil((n+1)*inlier_coverage)_order_statistic"]

    @model_validator(mode="after")
    def _detector_is_semantically_valid(self) -> Self:
        try:
            ShrinkageMahalanobisDetector.from_dict(self.model_dump(mode="python"))
        except MahalanobisValidationError as error:
            raise ValueError("detector payload violates the Mahalanobis contract") from error
        return self

    @classmethod
    def from_detector(cls, detector: ShrinkageMahalanobisDetector) -> Self:
        if not isinstance(detector, ShrinkageMahalanobisDetector):
            raise TypeError("detector must be a ShrinkageMahalanobisDetector")
        return cls.model_validate(detector.to_dict())

    def to_detector(self) -> ShrinkageMahalanobisDetector:
        """Restore the runtime detector after strict contract validation."""

        try:
            return ShrinkageMahalanobisDetector.from_dict(self.model_dump(mode="python"))
        except MahalanobisValidationError as error:  # pragma: no cover - model already validates
            raise OODCompletionIntegrityError("detector payload became invalid") from error


class EmbeddingContract(StrictFrozenModel):
    """Frozen representation boundary shared by fitting and runtime scoring."""

    architecture: Literal["resnet1d"]
    extraction_point: Literal["frozen_resnet_preclassifier_global_average_pool"]
    pooling: Literal["adaptive_average_pool_1d"]
    embedding_dimension: StrictFiveTwelve
    tensor_dtype: Literal["float32"]
    detector_numeric_dtype: Literal["float64"]
    lead_order: tuple[
        Literal["I"],
        Literal["II"],
        Literal["III"],
        Literal["aVR"],
        Literal["aVL"],
        Literal["aVF"],
        Literal["V1"],
        Literal["V2"],
        Literal["V3"],
        Literal["V4"],
        Literal["V5"],
        Literal["V6"],
    ]
    sampling_frequency_hz: Annotated[float, Field(strict=True, allow_inf_nan=False, gt=0.0)]
    samples_per_lead: PositiveInt
    physical_units: Literal["mV"]

    @model_validator(mode="after")
    def _canonical_input_contract(self) -> Self:
        if self.sampling_frequency_hz != 100.0 or self.samples_per_lead != 1000:
            raise ValueError("embedding input must be the canonical 100 Hz, 1000-sample ECG")
        return self


class OODLineageProvenance(StrictFrozenModel):
    """Path-free hash chain joining source calibration, model, data, and code."""

    ood_config_file_sha256: Sha256Digest
    source_calibration_artifact_sha256: Sha256Digest
    source_calibration_file_sha256: Sha256Digest
    source_calibration_config_file_sha256: Sha256Digest
    refit_completion_artifact_sha256: Sha256Digest
    refit_completion_file_sha256: Sha256Digest
    checkpoint_file_sha256: Sha256Digest
    resolved_config_sha256: Sha256Digest
    resolved_config_file_sha256: Sha256Digest
    dataset_manifest_file_sha256: Sha256Digest
    normalization_file_sha256: Sha256Digest
    experiment_protocol_sha256: Sha256Digest
    environment_lock_file_sha256: Sha256Digest
    project_manifest_file_sha256: Literal[
        "sha256:e1de755829678d588784bbcc34becc8c031c742d9b3f05458e76e67f577da3cd"
    ]
    raw_checksum_inventory_file_sha256: Sha256Digest
    raw_selected_inventory_sha256: Sha256Digest
    selected_record_count: StrictSelectedRecordCount
    selected_file_count: StrictSelectedFileCount
    raw_reference_inventory_sha256: Sha256Digest
    raw_source_inventory_sha256: Sha256Digest
    code_revision: GitRevision
    model_member_id: Identifier
    architecture: Literal["resnet1d"]
    seed: NonNegativeInt


class DistributionPolicyBody(StrictFrozenModel):
    """Self-contained runtime policy body before its logical hash is attached."""

    schema_version: StrictOne
    artifact_type: Literal["ecg_trust.distribution_policy"]
    protocol_id: Literal["trust-sentinel-ood-completion-v1"]
    method: Literal["shrinkage_mahalanobis_embedding_distance"]
    score_direction: Literal["higher_is_more_out_of_distribution"]
    threshold_comparison: Literal["score_strictly_greater_than_threshold"]
    embedding_contract: EmbeddingContract
    detector: MahalanobisDetectorArtifact
    provenance: OODLineageProvenance
    research_only: StrictTrue

    @model_validator(mode="after")
    def _policy_is_frozen(self) -> Self:
        detector = self.detector
        if detector.embedding_dim != self.embedding_contract.embedding_dimension:
            raise ValueError("detector dimension differs from the embedding contract")
        if (
            detector.shrinkage != 0.1
            or detector.ridge != 0.000001
            or detector.inlier_coverage != 0.95
        ):
            raise ValueError("detector parameters differ from the frozen OOD protocol")
        return self


class DistributionPolicy(DistributionPolicyBody):
    """Canonical, self-hashed distribution policy used by runtime scoring."""

    artifact_sha256: Sha256Digest

    @model_validator(mode="after")
    def _self_hash_verifies(self) -> Self:
        observed = canonical_sha256(self.model_dump(mode="json", exclude={"artifact_sha256"}))
        if observed != self.artifact_sha256:
            raise ValueError("artifact_sha256 does not match distribution policy")
        return self

    def to_detector(self) -> ShrinkageMahalanobisDetector:
        return self.detector.to_detector()


class EmbeddingRuntimeSummary(StrictFrozenModel):
    """Deterministic CUDA settings used for both representation passes."""

    requested_device: Literal["cuda:0"]
    resolved_device: Literal["cuda:0"]
    device_type: Literal["cuda"]
    device_name: Literal["NVIDIA GeForce RTX 5070 Ti Laptop GPU"]
    compute_capability: Literal["12.0"]
    python_version: Literal["3.12.13"]
    torch_version: Literal["2.13.0+cu130"]
    cuda_runtime_version: Literal["13.0"]
    cudnn_version: StrictCudnnVersion
    nvidia_driver_version: Literal["596.49"]
    tensor_precision: Literal["float32"]
    autocast_enabled: StrictFalse
    bf16_enabled: StrictFalse
    tf32_enabled: StrictFalse
    deterministic_algorithms: StrictTrue
    cudnn_deterministic: StrictTrue
    cudnn_benchmark: StrictFalse
    cublas_workspace_config: Literal[":4096:8"]
    inference_mode: StrictTrue
    model_eval_mode: StrictTrue
    shuffled: StrictFalse
    batch_size: StrictOneTwentyEight
    num_workers: StrictFour
    pin_memory: StrictTrue
    persistent_workers: StrictTrue
    extraction_passes: StrictTwo
    torch_compile_enabled: StrictFalse
    seed: StrictModelSeed

    @model_validator(mode="after")
    def _device_is_explicit_cuda(self) -> Self:
        if not self.requested_device.startswith("cuda:") or not self.resolved_device.startswith(
            "cuda:"
        ):
            raise ValueError("embedding execution must bind an explicit CUDA device index")
        return self


class _EmbeddingExecutionEvidence(StrictFrozenModel):
    records: PositiveInt
    patients: PositiveInt
    embedding_dimension: StrictFiveTwelve
    alignment_sha256: Sha256Digest
    embedding_tensor_sha256: Sha256Digest
    repeated_embedding_tensor_sha256: Sha256Digest
    embedding_artifact_sha256: Sha256Digest
    embedding_npz_file_sha256: Sha256Digest
    embedding_sidecar_file_sha256: Sha256Digest
    runtime_sha256: Sha256Digest
    exact_repeat_verified: StrictTrue
    public_contains_row_arrays: StrictFalse

    @model_validator(mode="after")
    def _execution_is_consistent(self) -> Self:
        if self.patients > self.records:
            raise ValueError("embedding cohort patients cannot exceed records")
        if self.embedding_tensor_sha256 != self.repeated_embedding_tensor_sha256:
            raise ValueError("repeated embedding tensor hash differs from the first pass")
        return self


class ReferenceEmbeddingExecutionSummary(_EmbeddingExecutionEvidence):
    role: Literal["ptbxl_folds_1_to_8_training_reference"]
    folds: tuple[
        Literal[1],
        Literal[2],
        Literal[3],
        Literal[4],
        Literal[5],
        Literal[6],
        Literal[7],
        Literal[8],
    ]

    @model_validator(mode="after")
    def _reference_counts_are_frozen(self) -> Self:
        if self.records != 17084 or self.patients != 14823:
            raise ValueError("reference execution counts differ from the frozen R cohort")
        return self


class ThresholdEmbeddingExecutionSummary(_EmbeddingExecutionEvidence):
    role: Literal["conformal_and_ood_threshold_fit"]
    folds: tuple[Literal[9]]
    source_assignment_sha256: Sha256Digest

    @model_validator(mode="after")
    def _threshold_counts_are_frozen(self) -> Self:
        if self.records != 834 or self.patients != 757:
            raise ValueError("threshold execution counts differ from the frozen B cohort")
        return self


class ReferenceAndThresholdExecutionSummary(StrictFrozenModel):
    """Execution evidence for the reference (R) and threshold-fit (B) cohorts."""

    runtime: EmbeddingRuntimeSummary
    reference: ReferenceEmbeddingExecutionSummary
    threshold_fit: ThresholdEmbeddingExecutionSummary

    @model_validator(mode="after")
    def _dimensions_match(self) -> Self:
        if self.reference.embedding_dimension != self.threshold_fit.embedding_dimension:
            raise ValueError("reference and threshold embedding dimensions differ")
        expected_runtime_sha256 = canonical_sha256(self.runtime.model_dump(mode="json"))
        if (
            self.reference.runtime_sha256 != expected_runtime_sha256
            or self.threshold_fit.runtime_sha256 != expected_runtime_sha256
        ):
            raise ValueError("R/B embedding evidence does not bind the declared runtime")
        return self


class ScoreQuantiles(StrictFrozenModel):
    """Finite aggregate quantiles for a nonnegative Mahalanobis score."""

    min: NonNegativeFloat
    p01: NonNegativeFloat
    p05: NonNegativeFloat
    p25: NonNegativeFloat
    p50: NonNegativeFloat
    p75: NonNegativeFloat
    p90: NonNegativeFloat
    p95: NonNegativeFloat
    p99: NonNegativeFloat
    max: NonNegativeFloat

    @model_validator(mode="after")
    def _quantiles_are_monotone(self) -> Self:
        values = (
            self.min,
            self.p01,
            self.p05,
            self.p25,
            self.p50,
            self.p75,
            self.p90,
            self.p95,
            self.p99,
            self.max,
        )
        if any(right < left for left, right in pairwise(values)):
            raise ValueError("score quantiles must be monotonically nondecreasing")
        return self


class ThresholdFitSummary(StrictFrozenModel):
    """Frozen B-role threshold fit, separate from C-role evaluation."""

    method: Literal["shrinkage_mahalanobis_embedding_distance"]
    fit_role: Literal["conformal_and_ood_threshold_fit"]
    n_reference_samples: PositiveInt
    n_threshold_samples: PositiveInt
    embedding_dimension: StrictFiveTwelve
    shrinkage: UnitFloat
    ridge: PositiveFloat
    target_inlier_coverage: StrictPointNineFive
    quantile_rank: PositiveInt
    threshold: NonNegativeFloat
    threshold_comparison: Literal["score_strictly_greater_than_threshold"]
    observed_inlier_count: PositiveInt
    observed_rejection_count: NonNegativeInt
    observed_inlier_fraction: UnitFloat
    observed_false_rejection_rate: UnitFloat
    score_quantiles: ScoreQuantiles

    @model_validator(mode="after")
    def _threshold_fit_is_consistent(self) -> Self:
        expected_rank = math.ceil((self.n_threshold_samples + 1) * 0.95)
        if self.quantile_rank != expected_rank or self.quantile_rank > self.n_threshold_samples:
            raise ValueError("quantile_rank does not follow the finite-sample rule")
        if self.shrinkage != 0.1 or self.ridge != 0.000001:
            raise ValueError("threshold fit parameters differ from the frozen protocol")
        if self.observed_inlier_count + self.observed_rejection_count != self.n_threshold_samples:
            raise ValueError("threshold fit counts do not sum to n_threshold_samples")
        expected_inlier = self.observed_inlier_count / self.n_threshold_samples
        expected_rejection = self.observed_rejection_count / self.n_threshold_samples
        if not math.isclose(self.observed_inlier_fraction, expected_inlier, abs_tol=1e-15):
            raise ValueError("observed_inlier_fraction does not match threshold counts")
        if not math.isclose(self.observed_false_rejection_rate, expected_rejection, abs_tol=1e-15):
            raise ValueError("observed_false_rejection_rate does not match threshold counts")
        if self.score_quantiles.max < self.threshold:
            raise ValueError("threshold cannot exceed the maximum threshold-fit score")
        return self


class PatientClusterBootstrapInterval(StrictFrozenModel):
    """Frozen cluster-bootstrap uncertainty interval for record false rejection."""

    method: Literal["patient_cluster_percentile_bootstrap"]
    estimator: Literal["record_false_rejection_rate"]
    resampling_unit: Literal["patient"]
    sampling_with_replacement: StrictTrue
    seed: StrictBootstrapSeed
    replicates: StrictTenThousand
    random_generator: Literal["numpy.random.Generator_PCG64"] = "numpy.random.Generator_PCG64"
    patient_order: Literal["ascending_numeric_patient_id"] = "ascending_numeric_patient_id"
    record_order_within_patient: Literal["ascending_ecg_id"] = "ascending_ecg_id"
    percentile_function: Literal["numpy.quantile"] = "numpy.quantile"
    quantile_method: Literal["linear"] = "linear"
    two_sided_confidence_level: StrictPointNineFive
    two_sided_lower: UnitFloat
    two_sided_upper: UnitFloat
    one_sided_confidence_level: StrictPointNineFive
    one_sided_upper: UnitFloat

    @model_validator(mode="after")
    def _interval_is_ordered(self) -> Self:
        if self.two_sided_lower > self.two_sided_upper:
            raise ValueError("two-sided bootstrap interval is reversed")
        if self.one_sided_upper > self.two_sided_upper:
            raise ValueError("one-sided upper bound exceeds the two-sided upper bound")
        return self


class SourceOODValidationSummary(_EmbeddingExecutionEvidence):
    """Untuned C-role clean-source false-rejection evaluation."""

    evaluation_role: Literal["source_validation"]
    tuning_allowed: StrictFalse
    rejected_records: NonNegativeInt
    accepted_records: NonNegativeInt
    rejected_patients_any: NonNegativeInt
    record_false_rejection_rate: UnitFloat
    source_record_support_coverage: UnitFloat
    patient_equalized_false_rejection_rate: UnitFloat
    patient_any_false_rejection_rate: UnitFloat
    maximum_allowed_record_false_rejection_rate: StrictPointZeroFive
    cluster_bootstrap: PatientClusterBootstrapInterval
    score_quantiles: ScoreQuantiles
    threshold: NonNegativeFloat
    threshold_tie_count: NonNegativeInt
    threshold_comparison: Literal["score_strictly_greater_than_threshold"]
    source_assignment_sha256: Sha256Digest
    target_met: StrictBool

    @model_validator(mode="after")
    def _validation_is_consistent(self) -> Self:
        if self.records != 465 or self.patients != 409:
            raise ValueError("validation counts differ from the frozen C cohort")
        if self.patients > self.records:
            raise ValueError("validation patients cannot exceed validation records")
        if self.rejected_records > self.records:
            raise ValueError("rejected_records cannot exceed records")
        if self.rejected_patients_any > self.patients:
            raise ValueError("rejected_patients_any cannot exceed patients")
        if self.threshold_tie_count > self.records:
            raise ValueError("threshold_tie_count cannot exceed records")
        if self.rejected_records + self.accepted_records != self.records:
            raise ValueError("validation record counts do not sum to records")
        expected_record_rate = self.rejected_records / self.records
        expected_patient_any = self.rejected_patients_any / self.patients
        if not math.isclose(self.record_false_rejection_rate, expected_record_rate, abs_tol=1e-15):
            raise ValueError("record_false_rejection_rate does not match validation counts")
        expected_support_coverage = self.accepted_records / self.records
        if not math.isclose(
            self.source_record_support_coverage,
            expected_support_coverage,
            abs_tol=1e-15,
        ):
            raise ValueError("source_record_support_coverage does not match validation counts")
        if not math.isclose(
            self.source_record_support_coverage + self.record_false_rejection_rate,
            1.0,
            abs_tol=1e-15,
        ):
            raise ValueError("source support coverage and false-rejection rate must sum to one")
        if not math.isclose(
            self.patient_any_false_rejection_rate, expected_patient_any, abs_tol=1e-15
        ):
            raise ValueError("patient_any_false_rejection_rate does not match patient counts")
        expected_target_met = self.cluster_bootstrap.one_sided_upper <= 0.05
        if self.target_met is not expected_target_met:
            raise ValueError("target_met must be derived from the one-sided upper bound")
        return self


class OODPositiveEvaluationSummary(StrictFrozenModel):
    """Explicit statement that no positive-OOD discrimination claim was evaluated."""

    status: Literal["NOT_EVALUATED"]
    evaluation_sources: tuple[()] = ()
    records: Literal[0]
    semantic_ood_recall: Literal["NOT_EVALUATED"]
    severe_ood_recall: Literal["NOT_EVALUATED"]
    ood_auroc: Literal["NOT_EVALUATED"]
    ood_average_precision: Literal["NOT_EVALUATED"]
    unseen_site_or_device_performance: Literal["NOT_EVALUATED"]
    target_site_fitting_performed: StrictFalse
    reason_code: Literal["NO_FROZEN_OOD_POSITIVE_EVALUATION_SOURCE"]


class OODIntegritySummary(StrictFrozenModel):
    """All integrity predicates required before a result artifact can exist."""

    complete: StrictTrue
    verified_input_hashes: StrictTrue
    source_calibration_verified: StrictTrue
    refit_lineage_verified: StrictTrue
    waveform_checksums_verified_before_and_after: StrictTrue
    source_alignment_verified: StrictTrue
    patient_roles_disjoint: StrictTrue
    forbidden_sources_not_accessed: StrictTrue
    model_state_unchanged: StrictTrue
    deterministic_repeat_verified: StrictTrue
    distribution_policy_verified: StrictTrue
    aggregate_only_result_verified: StrictTrue


class DistributionPolicyBinding(StrictFrozenModel):
    """Logical and physical identity of the separately stored runtime policy."""

    filename: Literal["distribution-policy.json"]
    artifact_sha256: Sha256Digest
    file_sha256: Sha256Digest
    size_bytes: PositiveInt


class OODClaimBoundary(StrictFrozenModel):
    scope: Literal["retrospective_ptbxl_source_domain_development_only"]
    research_only: StrictTrue
    clinical_validation: StrictFalse
    external_ood_positive_validation: StrictFalse
    limitations: tuple[Identifier, ...]

    @model_validator(mode="after")
    def _limitations_are_exact(self) -> Self:
        expected = (
            "no_external_lockbox_evaluation",
            "no_ood_positive_evaluation",
            "no_clinical_validation",
            "ood_score_does_not_identify_unknown_disease",
            "source_false_rejection_target_is_provisional",
            "research_bundle_eligibility_is_not_clinical_release_readiness",
        )
        if self.limitations != expected:
            raise ValueError("OOD completion limitations differ from the safety boundary")
        return self


OODCompletionStatus = Literal[
    "SOURCE_SUPPORT_GATE_COMPLETE",
    "SOURCE_SUPPORT_GATE_TARGET_MISSED",
]


class OODCompletionResultBody(StrictFrozenModel):
    """Aggregate-only composite evidence before its logical hash is attached."""

    schema_version: StrictOne
    artifact_type: Literal["ecg_trust.ood_completion_result"]
    protocol_id: Literal["trust-sentinel-ood-completion-v1"]
    status: OODCompletionStatus
    frozen_at_utc: AwareDatetime
    source_calibration: SourceCalibrationResult
    provenance: OODLineageProvenance
    distribution_policy: DistributionPolicyBinding
    reference_and_threshold_execution: ReferenceAndThresholdExecutionSummary
    threshold_fit: ThresholdFitSummary
    source_validation: SourceOODValidationSummary
    ood_positive_evaluation: OODPositiveEvaluationSummary
    integrity: OODIntegritySummary
    research_bundle_eligible: StrictBool
    claims: OODClaimBoundary

    @field_validator("frozen_at_utc")
    @classmethod
    def _timestamp_is_utc(cls, value: datetime) -> datetime:
        offset = value.utcoffset()
        if offset is None or offset.total_seconds() != 0:
            raise ValueError("frozen_at_utc must use UTC")
        return value

    @model_validator(mode="after")
    def _composite_contract_is_consistent(self) -> Self:
        source = self.source_calibration
        if (
            source.protocol_id != "trust-sentinel-source-calibration-v1"
            or source.status != "PREPARED_NOT_RELEASE_READY"
            or source.open_world.status != "PENDING"
            or source.open_world.release_ready is not False
        ):
            raise ValueError("nested source calibration must remain the frozen pending v1 result")

        provenance = self.provenance
        if provenance.source_calibration_artifact_sha256 != source.artifact_sha256:
            raise ValueError("provenance does not bind the nested source calibration")
        if (
            provenance.source_calibration_config_file_sha256 != source.provenance.config_file_sha256
            or provenance.checkpoint_file_sha256 != source.provenance.checkpoint_sha256
            or provenance.experiment_protocol_sha256 != source.provenance.experiment_protocol_sha256
            or provenance.model_member_id != source.provenance.model_member_id
            or provenance.architecture != source.provenance.architecture
            or provenance.seed != source.provenance.seed
        ):
            raise ValueError("OOD provenance differs from the nested source lineage")

        execution = self.reference_and_threshold_execution
        reference = execution.reference
        threshold_execution = execution.threshold_fit
        threshold_fit = self.threshold_fit
        validation = self.source_validation
        source_roles = {item.role.value: item for item in source.split.roles}
        source_threshold = source_roles["conformal_and_ood_threshold_fit"]
        source_validation = source_roles["source_validation"]
        if (
            threshold_execution.records != source_threshold.records
            or threshold_execution.patients != source_threshold.patients
            or validation.records != source_validation.records
            or validation.patients != source_validation.patients
        ):
            raise ValueError("OOD B/C cohort counts differ from frozen source roles")
        if (
            threshold_execution.source_assignment_sha256 != source.split.assignment_sha256
            or validation.source_assignment_sha256 != source.split.assignment_sha256
        ):
            raise ValueError("OOD B/C evidence differs from the frozen patient assignment")
        if (
            threshold_fit.n_reference_samples != reference.records
            or threshold_fit.n_threshold_samples != threshold_execution.records
            or threshold_fit.embedding_dimension != reference.embedding_dimension
            or validation.embedding_dimension != reference.embedding_dimension
            or validation.threshold != threshold_fit.threshold
        ):
            raise ValueError("OOD fit, execution, and validation summaries do not align")
        if validation.runtime_sha256 != reference.runtime_sha256:
            raise ValueError("C embedding evidence does not bind the R/B runtime")
        if execution.runtime.seed != provenance.seed:
            raise ValueError("embedding runtime seed differs from the frozen model seed")
        selected_records = reference.records + threshold_execution.records + validation.records
        if selected_records != provenance.selected_record_count:
            raise ValueError("R/B/C execution counts differ from the selected checksum subset")
        if provenance.selected_file_count != 2 * selected_records:
            raise ValueError("selected checksum subset must bind one DAT and HEA file per record")

        eligible = (
            self.integrity.complete
            and self.source_validation.cluster_bootstrap.one_sided_upper <= 0.05
        )
        expected_status: OODCompletionStatus = (
            "SOURCE_SUPPORT_GATE_COMPLETE"
            if eligible
            else "SOURCE_SUPPORT_GATE_TARGET_MISSED"
        )
        if self.research_bundle_eligible is not eligible:
            raise ValueError(
                "research_bundle_eligible is not derived from integrity and C upper bound"
            )
        if self.status != expected_status:
            raise ValueError("OOD completion status is inconsistent with bundle eligibility")
        return self


class OODCompletionResult(OODCompletionResultBody):
    """Canonical, self-hashed aggregate OOD-completion evidence."""

    artifact_sha256: Sha256Digest

    @model_validator(mode="after")
    def _self_hash_verifies(self) -> Self:
        observed = canonical_sha256(self.model_dump(mode="json", exclude={"artifact_sha256"}))
        if observed != self.artifact_sha256:
            raise ValueError("artifact_sha256 does not match OOD completion result")
        return self


class OODCompletionFailureCode(StrEnum):
    INPUT_CONTRACT_FAILED = "INPUT_CONTRACT_FAILED"
    EMBEDDING_EXTRACTION_FAILED = "EMBEDDING_EXTRACTION_FAILED"
    DETERMINISM_FAILED = "DETERMINISM_FAILED"
    FIT_FAILED = "FIT_FAILED"
    VALIDATION_FAILED = "VALIDATION_FAILED"
    OUTPUT_COMMIT_FAILED = "OUTPUT_COMMIT_FAILED"
    INTERNAL_FAILURE = "INTERNAL_FAILURE"


class OODCompletionFailureReceiptBody(StrictFrozenModel):
    """Sanitized failure identity; it intentionally carries no error text."""

    schema_version: StrictOne
    artifact_type: Literal["ecg_trust.ood_completion_failure"]
    protocol_id: Literal["trust-sentinel-ood-completion-v1"]
    status: Literal["FAILED"]
    frozen_at_utc: AwareDatetime
    config_file_sha256: Sha256Digest
    code_revision: GitRevision
    failure_code: OODCompletionFailureCode
    contains_raw_ids_or_rows: StrictFalse
    contains_embeddings: StrictFalse
    contains_filesystem_paths: StrictFalse
    retry_requires_new_output_root: StrictTrue

    @field_validator("frozen_at_utc")
    @classmethod
    def _timestamp_is_utc(cls, value: datetime) -> datetime:
        offset = value.utcoffset()
        if offset is None or offset.total_seconds() != 0:
            raise ValueError("frozen_at_utc must use UTC")
        return value


class OODCompletionFailureReceipt(OODCompletionFailureReceiptBody):
    artifact_sha256: Sha256Digest

    @model_validator(mode="after")
    def _self_hash_verifies(self) -> Self:
        observed = canonical_sha256(self.model_dump(mode="json", exclude={"artifact_sha256"}))
        if observed != self.artifact_sha256:
            raise ValueError("artifact_sha256 does not match OOD completion failure receipt")
        return self


OODBundleRelativePath = Literal[
    "distribution-policy.json",
    "ood-completion-result.json",
    "private/reference-embeddings.json",
    "private/reference-embeddings.npz",
    "private/source-validation-embeddings.json",
    "private/source-validation-embeddings.npz",
    "private/threshold-fit-embeddings.json",
    "private/threshold-fit-embeddings.npz",
    "source-validation-access-armed.json",
]

_OOD_BUNDLE_MEMBER_PATHS: Final = (
    "distribution-policy.json",
    "ood-completion-result.json",
    "private/reference-embeddings.json",
    "private/reference-embeddings.npz",
    "private/source-validation-embeddings.json",
    "private/source-validation-embeddings.npz",
    "private/threshold-fit-embeddings.json",
    "private/threshold-fit-embeddings.npz",
    "source-validation-access-armed.json",
)


class OODBundleMember(StrictFrozenModel):
    """One exact regular file bound by the terminal success manifest."""

    relative_path: OODBundleRelativePath
    size_bytes: PositiveInt
    file_sha256: Sha256Digest


class OODCompletionSuccessManifestBody(StrictFrozenModel):
    """Terminal whole-output identity, written only after every final check."""

    schema_version: StrictOne
    artifact_type: Literal["ecg_trust.ood_completion_success_manifest"]
    protocol_id: Literal["trust-sentinel-ood-completion-v1"]
    status: Literal["SUCCESS"]
    frozen_at_utc: AwareDatetime
    config_file_sha256: Sha256Digest
    code_revision: GitRevision
    result_artifact_sha256: Sha256Digest
    distribution_policy_artifact_sha256: Sha256Digest
    validation_access_claim_filename: Literal[
        ".ood_completion_v1.source-validation-one-shot-claim.json"
    ]
    validation_access_claim_file_sha256: Sha256Digest
    member_count: Literal[9]
    members: tuple[OODBundleMember, ...] = Field(min_length=9, max_length=9)
    terminal_checks_complete: StrictTrue
    failure_receipt_present: StrictFalse

    @field_validator("frozen_at_utc")
    @classmethod
    def _success_timestamp_is_utc(cls, value: datetime) -> datetime:
        offset = value.utcoffset()
        if offset is None or offset.total_seconds() != 0:
            raise ValueError("frozen_at_utc must use UTC")
        return value

    @model_validator(mode="after")
    def _members_are_exact_and_ordered(self) -> Self:
        paths = tuple(member.relative_path for member in self.members)
        if paths != _OOD_BUNDLE_MEMBER_PATHS:
            raise ValueError("success manifest members differ from the exact bundle inventory")
        if self.member_count != len(self.members):
            raise ValueError("success manifest member_count differs from members")
        return self


class OODCompletionSuccessManifest(OODCompletionSuccessManifestBody):
    artifact_sha256: Sha256Digest

    @model_validator(mode="after")
    def _success_self_hash_verifies(self) -> Self:
        observed = canonical_sha256(self.model_dump(mode="json", exclude={"artifact_sha256"}))
        if observed != self.artifact_sha256:
            raise ValueError("artifact_sha256 does not match OOD completion success manifest")
        return self


def canonical_json_bytes(value: dict[str, object]) -> bytes:
    """Serialize finite canonical JSON shared by all OOD artifacts."""

    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise OODCompletionIntegrityError("value is not finite canonical JSON") from error


def canonical_sha256(value: dict[str, object]) -> str:
    return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def seal_distribution_policy(body: DistributionPolicyBody) -> DistributionPolicy:
    if not isinstance(body, DistributionPolicyBody):
        raise TypeError("body must be a DistributionPolicyBody")
    python_payload = body.model_dump(mode="python")
    json_payload = body.model_dump(mode="json")
    return DistributionPolicy.model_validate(
        {**python_payload, "artifact_sha256": canonical_sha256(json_payload)}
    )


def seal_ood_completion_result(body: OODCompletionResultBody) -> OODCompletionResult:
    if not isinstance(body, OODCompletionResultBody):
        raise TypeError("body must be an OODCompletionResultBody")
    python_payload = body.model_dump(mode="python")
    json_payload = body.model_dump(mode="json")
    result = OODCompletionResult.model_validate(
        {**python_payload, "artifact_sha256": canonical_sha256(json_payload)}
    )
    assert_aggregate_only_ood_result(result)
    return result


def seal_ood_completion_failure_receipt(
    body: OODCompletionFailureReceiptBody,
) -> OODCompletionFailureReceipt:
    if not isinstance(body, OODCompletionFailureReceiptBody):
        raise TypeError("body must be an OODCompletionFailureReceiptBody")
    python_payload = body.model_dump(mode="python")
    json_payload = body.model_dump(mode="json")
    return OODCompletionFailureReceipt.model_validate(
        {**python_payload, "artifact_sha256": canonical_sha256(json_payload)}
    )


def seal_ood_completion_success_manifest(
    body: OODCompletionSuccessManifestBody,
) -> OODCompletionSuccessManifest:
    if not isinstance(body, OODCompletionSuccessManifestBody):
        raise TypeError("body must be an OODCompletionSuccessManifestBody")
    python_payload = body.model_dump(mode="python")
    json_payload = body.model_dump(mode="json")
    return OODCompletionSuccessManifest.model_validate(
        {**python_payload, "artifact_sha256": canonical_sha256(json_payload)}
    )


def distribution_policy_json_bytes(policy: DistributionPolicy) -> bytes:
    if not isinstance(policy, DistributionPolicy):
        raise TypeError("policy must be a DistributionPolicy")
    validated = DistributionPolicy.model_validate(policy.model_dump(mode="python"))
    _assert_no_paths_secrets_or_row_arrays(validated.model_dump(mode="json"), allow_detector=True)
    return canonical_json_bytes(validated.model_dump(mode="json")) + b"\n"


def ood_completion_result_json_bytes(result: OODCompletionResult) -> bytes:
    if not isinstance(result, OODCompletionResult):
        raise TypeError("result must be an OODCompletionResult")
    validated = OODCompletionResult.model_validate(result.model_dump(mode="python"))
    assert_aggregate_only_ood_result(validated)
    return canonical_json_bytes(validated.model_dump(mode="json")) + b"\n"


def ood_completion_failure_json_bytes(receipt: OODCompletionFailureReceipt) -> bytes:
    if not isinstance(receipt, OODCompletionFailureReceipt):
        raise TypeError("receipt must be an OODCompletionFailureReceipt")
    validated = OODCompletionFailureReceipt.model_validate(receipt.model_dump(mode="python"))
    _assert_no_paths_secrets_or_row_arrays(validated.model_dump(mode="json"), allow_detector=False)
    return canonical_json_bytes(validated.model_dump(mode="json")) + b"\n"


def ood_completion_success_json_bytes(manifest: OODCompletionSuccessManifest) -> bytes:
    if not isinstance(manifest, OODCompletionSuccessManifest):
        raise TypeError("manifest must be an OODCompletionSuccessManifest")
    validated = OODCompletionSuccessManifest.model_validate(manifest.model_dump(mode="python"))
    _assert_no_paths_secrets_or_row_arrays(validated.model_dump(mode="json"), allow_detector=False)
    return canonical_json_bytes(validated.model_dump(mode="json")) + b"\n"


def _load_canonical_model[ModelValue: BaseModel](
    payload: bytes,
    *,
    model_type: type[ModelValue],
    maximum_bytes: int,
    context: str,
) -> ModelValue:
    if not payload or len(payload) > maximum_bytes or not payload.endswith(b"\n"):
        raise OODCompletionIntegrityError(f"{context} JSON byte contract is invalid")
    if payload.endswith(b"\n\n") or b"\r" in payload:
        raise OODCompletionIntegrityError(f"{context} JSON is not canonically terminated")
    try:
        decoded: object = json.loads(payload[:-1].decode("ascii"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise OODCompletionIntegrityError(f"{context} JSON cannot be decoded") from error
    if not isinstance(decoded, dict) or not all(isinstance(key, str) for key in decoded):
        raise OODCompletionIntegrityError(f"{context} JSON root is invalid")
    try:
        value = model_type.model_validate(decoded)
    except ValidationError as error:
        raise OODCompletionIntegrityError(f"{context} JSON violates its schema") from error
    expected = canonical_json_bytes(value.model_dump(mode="json")) + b"\n"
    if expected != payload:
        raise OODCompletionIntegrityError(f"{context} JSON is not canonical")
    return value


def load_distribution_policy_bytes(payload: bytes) -> DistributionPolicy:
    policy = _load_canonical_model(
        payload,
        model_type=DistributionPolicy,
        maximum_bytes=MAX_DISTRIBUTION_POLICY_BYTES,
        context="distribution policy",
    )
    _assert_no_paths_secrets_or_row_arrays(policy.model_dump(mode="json"), allow_detector=True)
    return policy


def load_ood_completion_result_bytes(payload: bytes) -> OODCompletionResult:
    result = _load_canonical_model(
        payload,
        model_type=OODCompletionResult,
        maximum_bytes=MAX_OOD_COMPLETION_RESULT_BYTES,
        context="OOD completion result",
    )
    assert_aggregate_only_ood_result(result)
    return result


def load_ood_completion_failure_bytes(payload: bytes) -> OODCompletionFailureReceipt:
    receipt = _load_canonical_model(
        payload,
        model_type=OODCompletionFailureReceipt,
        maximum_bytes=MAX_OOD_COMPLETION_FAILURE_BYTES,
        context="OOD completion failure receipt",
    )
    _assert_no_paths_secrets_or_row_arrays(receipt.model_dump(mode="json"), allow_detector=False)
    return receipt


def load_ood_completion_success_bytes(payload: bytes) -> OODCompletionSuccessManifest:
    manifest = _load_canonical_model(
        payload,
        model_type=OODCompletionSuccessManifest,
        maximum_bytes=MAX_OOD_COMPLETION_SUCCESS_BYTES,
        context="OOD completion success manifest",
    )
    _assert_no_paths_secrets_or_row_arrays(manifest.model_dump(mode="json"), allow_detector=False)
    return manifest


def assert_aggregate_only_ood_result(result: OODCompletionResult) -> None:
    """Reject identifiers, paths, per-row arrays, raw scores, or secrets."""

    if not isinstance(result, OODCompletionResult):
        raise TypeError("result must be an OODCompletionResult")
    validated = OODCompletionResult.model_validate(result.model_dump(mode="python"))
    _assert_no_paths_secrets_or_row_arrays(validated.model_dump(mode="json"), allow_detector=False)
    canonical_json_bytes(validated.model_dump(mode="json"))


def _assert_result_source_support_eligible(result: OODCompletionResult) -> None:
    """Fail closed unless a result's derived source-support eligibility is satisfied."""

    if not isinstance(result, OODCompletionResult):
        raise TypeError("result must be an OODCompletionResult")
    result = OODCompletionResult.model_validate(result.model_dump(mode="python"))
    if (
        result.status != "SOURCE_SUPPORT_GATE_COMPLETE"
        or result.research_bundle_eligible is not True
        or result.integrity.complete is not True
        or result.source_validation.cluster_bootstrap.one_sided_upper > 0.05
    ):
        raise OODCompletionIntegrityError("OOD completion is not eligible for a research bundle")


_FORBIDDEN_PUBLIC_KEYS = frozenset(
    {
        "ecg_id",
        "patient_id",
        "record_path",
        "path",
        "npz_path",
        "sidecar_path",
        "raw_logits",
        "targets",
        "probabilities",
        "embeddings",
        "embedding_array",
        "scores",
        "score_array",
        "rows",
        "records_array",
        "selected_indices",
        "abstained_indices",
    }
)
_SECRET_KEY_FRAGMENTS = ("password", "secret", "access_token", "private_key")


def _assert_no_paths_secrets_or_row_arrays(value: object, *, allow_detector: bool) -> None:
    def visit(item: object, *, inside_detector: bool = False) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                normalized = key.casefold()
                child_inside_detector = inside_detector or (allow_detector and key == "detector")
                if normalized in _FORBIDDEN_PUBLIC_KEYS or any(
                    fragment in normalized for fragment in _SECRET_KEY_FRAGMENTS
                ):
                    raise OODCompletionIntegrityError(
                        "artifact contains a forbidden sensitive field"
                    )
                if not child_inside_detector and normalized in {"mean", "precision"}:
                    raise OODCompletionIntegrityError(
                        "aggregate result contains detector parameter arrays"
                    )
                visit(child, inside_detector=child_inside_detector)
        elif isinstance(item, list):
            for child in item:
                visit(child, inside_detector=inside_detector)
        elif isinstance(item, str) and _looks_absolute_path(item):
            raise OODCompletionIntegrityError("artifact contains an absolute filesystem path")

    visit(value)


def _looks_absolute_path(value: str) -> bool:
    return PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute()


__all__ = [
    "DISTRIBUTION_POLICY_ARTIFACT_TYPE",
    "DISTRIBUTION_POLICY_FILENAME",
    "MAX_DISTRIBUTION_POLICY_BYTES",
    "MAX_OOD_COMPLETION_FAILURE_BYTES",
    "MAX_OOD_COMPLETION_RESULT_BYTES",
    "MAX_OOD_COMPLETION_SUCCESS_BYTES",
    "OOD_COMPLETION_ARTIFACT_TYPE",
    "OOD_COMPLETION_FAILURE_ARTIFACT_TYPE",
    "OOD_COMPLETION_FAILURE_FILENAME",
    "OOD_COMPLETION_PROTOCOL_ID",
    "OOD_COMPLETION_RESULT_FILENAME",
    "OOD_COMPLETION_SUCCESS_ARTIFACT_TYPE",
    "OOD_COMPLETION_SUCCESS_FILENAME",
    "OOD_COMPLETION_SCHEMA_VERSION",
    "DistributionPolicy",
    "DistributionPolicyBinding",
    "DistributionPolicyBody",
    "EmbeddingContract",
    "EmbeddingRuntimeSummary",
    "MahalanobisDetectorArtifact",
    "OODClaimBoundary",
    "OODCompletionError",
    "OODCompletionFailureCode",
    "OODCompletionFailureReceipt",
    "OODCompletionFailureReceiptBody",
    "OODCompletionIntegrityError",
    "OODCompletionResult",
    "OODCompletionResultBody",
    "OODCompletionStatus",
    "OODBundleMember",
    "OODBundleRelativePath",
    "OODCompletionSuccessManifest",
    "OODCompletionSuccessManifestBody",
    "OODIntegritySummary",
    "OODLineageProvenance",
    "OODPositiveEvaluationSummary",
    "PatientClusterBootstrapInterval",
    "ReferenceAndThresholdExecutionSummary",
    "ReferenceEmbeddingExecutionSummary",
    "ScoreQuantiles",
    "SourceOODValidationSummary",
    "ThresholdEmbeddingExecutionSummary",
    "ThresholdFitSummary",
    "assert_aggregate_only_ood_result",
    "canonical_json_bytes",
    "canonical_sha256",
    "distribution_policy_json_bytes",
    "load_distribution_policy_bytes",
    "load_ood_completion_failure_bytes",
    "load_ood_completion_result_bytes",
    "load_ood_completion_success_bytes",
    "ood_completion_failure_json_bytes",
    "ood_completion_result_json_bytes",
    "ood_completion_success_json_bytes",
    "seal_distribution_policy",
    "seal_ood_completion_failure_receipt",
    "seal_ood_completion_result",
    "seal_ood_completion_success_manifest",
]
