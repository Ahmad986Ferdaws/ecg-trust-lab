"""Immutable aggregate-only evidence for the frozen external OOD v2 protocol.

The v2 artifact is intentionally independent of :mod:`ecg_trust.ood_completion`.
Nothing in this module imports, rewrites, or relaxes the sealed v1 contracts.
Only cohort-level counts, rates, uncertainty intervals, and cryptographic lineage
may enter the public result.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime
from enum import StrEnum
from pathlib import PurePosixPath, PureWindowsPath
from typing import Annotated, Final, Literal, Self, cast

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

OOD_V2_SCHEMA_VERSION: Final = 1
OOD_V2_ORIGINAL_PROTOCOL_ID: Final = "trust-sentinel-ood-external-v2-parent"
OOD_V2_PROTOCOL_ID: Final = "trust-sentinel-ood-external-v2-1-parent"
OOD_V2_ARTIFACT_TYPE: Final = "ecg_trust.ood_external_v2_1_result"
OOD_V2_RESULT_FILENAME: Final = "ood-external-v2-1-result.json"
OOD_V2_PARENT_CONFIG_SHA256: Final = (
    "sha256:2b6696d07c1fbab1e31eccb3d8d48fdc6251d12301df6ca604b8af1d02b7dd10"
)
_V1_RESULT_FILE_SHA256: Final = (
    "sha256:844bbe7f2a85b229f553cd12df14f7db712b9e0090fe6fd6823319a557777c12"
)
_V1_DISTRIBUTION_POLICY_FILE_SHA256: Final = (
    "sha256:817d6e5c4a3058c064cdc7bdceafb774c7ea4bb0b6cf725be1b8f12c7aae9c1c"
)
_V1_ONE_SHOT_CLAIM_FILE_SHA256: Final = (
    "sha256:956c16e6d9ce4575274f040e44a822e7c8952b98642cc243f165a262f1b5a2f8"
)
_V1_SOURCE_SPLIT_ASSIGNMENT_SHA256: Final = (
    "sha256:87992206fcbfc2b091d8f8dd08998a5d9bae3d55a2d2056f1ab674a316b0675b"
)
MAX_OOD_V2_RESULT_BYTES: Final = 2 * 1024 * 1024


def _require_true(value: object) -> bool:
    if value is not True:
        raise ValueError("value must be the boolean true")
    return True


def _require_false(value: object) -> bool:
    if value is not False:
        raise ValueError("value must be the boolean false")
    return False


def _require_float(value: object) -> float:
    if type(value) is not float:
        raise ValueError("value must be a JSON floating-point number")
    return value


StrictTrue = Annotated[Literal[True], BeforeValidator(_require_true)]
StrictFalse = Annotated[Literal[False], BeforeValidator(_require_false)]
StrictBool = Annotated[bool, Field(strict=True)]
PositiveInt = Annotated[int, Field(strict=True, ge=1)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
UnitFloat = Annotated[
    float,
    BeforeValidator(_require_float),
    Field(strict=True, ge=0.0, le=1.0, allow_inf_nan=False),
]
Identifier = Annotated[
    str,
    StringConstraints(
        strict=True,
        min_length=1,
        max_length=160,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
    ),
]
StrictText = Annotated[str, StringConstraints(strict=True, min_length=1, max_length=500)]
Sha256Digest = Annotated[
    str,
    StringConstraints(
        strict=True,
        min_length=71,
        max_length=71,
        pattern=r"^sha256:[0-9a-f]{64}$",
    ),
]
GitRevision = Annotated[
    str,
    StringConstraints(strict=True, pattern=r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$"),
]


class OODV2Error(ValueError):
    """Base error for v2 evidence contracts."""


class OODV2IntegrityError(OODV2Error):
    """Raised when serialized v2 evidence is non-canonical or fails integrity."""


class StrictFrozenModel(BaseModel):
    """Base for immutable, exact-schema, always-revalidated public evidence."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        revalidate_instances="always",
        validate_default=True,
        use_enum_values=False,
    )


class ResamplingUnit(StrEnum):
    """Independent unit used by the deterministic bootstrap."""

    PATIENT_CLUSTER = "patient_cluster"
    RECORD = "record"


class ExternalCohortRole(StrEnum):
    """Predeclared role of an external cohort."""

    PHYSIONET_CHALLENGE_2011_SET_A = "physionet_challenge_2011_set_a"
    ZZU_PECG_V1 = "zzu_pecg_v1"


class OODAxis(StrEnum):
    """Claim axis evaluated by one external OOD-positive cohort."""

    EXTERNAL_ACQUISITION_AND_POPULATION = "external_acquisition_and_population_domain"
    PEDIATRIC_POPULATION_AND_ACQUISITION = (
        "pediatric_population_and_external_acquisition_domain"
    )


class TechnicalQualityEventDefinition(StrEnum):
    """Predeclared binary success event for a technical quality endpoint."""

    BLOCK_UNACCEPTABLE = "block_unacceptable"
    PASS_ACCEPTABLE = "pass_acceptable"


class OODV2Status(StrEnum):
    """Only terminal status vocabulary permitted by the v2 result."""

    EXTERNAL_OOD_EVIDENCE_COMPLETE = "EXTERNAL_OOD_EVIDENCE_COMPLETE"
    EXTERNAL_OOD_TARGET_MISSED = "EXTERNAL_OOD_TARGET_MISSED"
    EXTERNAL_OOD_INSUFFICIENT_EVIDENCE = "EXTERNAL_OOD_INSUFFICIENT_EVIDENCE"


class HistoricalSourceBootstrapInterval(StrictFrozenModel):
    """Exact aggregate interval published by the sealed v1 source study.

    V1 preregistered and published a two-sided 95% interval and a one-sided
    95% upper bound. It did not publish a one-sided lower bound or bootstrap
    replicates. V2 therefore represents only the values that actually exist;
    it must not reconstruct or invent a missing lower bound from private C
    identifiers or decisions.
    """

    method: Literal["historical_patient_cluster_percentile_bootstrap"]
    estimator: Literal["record_weighted_event_rate"]
    resampling_unit: Literal[ResamplingUnit.PATIENT_CLUSTER]
    sampling_with_replacement: StrictTrue
    random_generator: Literal["numpy.random.Generator_PCG64"]
    seed: NonNegativeInt
    replicates: Annotated[int, Field(strict=True, ge=1_000)]
    percentile_function: Literal["numpy.quantile"]
    quantile_method: Literal["linear"]
    confidence_level: Annotated[
        float,
        BeforeValidator(_require_float),
        Field(strict=True, gt=0.0, lt=1.0, allow_inf_nan=False),
    ]
    records: PositiveInt
    resampling_units: PositiveInt
    event_count: NonNegativeInt
    point_estimate: UnitFloat
    two_sided_lower: UnitFloat
    two_sided_upper: UnitFloat
    one_sided_upper: UnitFloat
    one_sided_lower_published: StrictFalse

    @model_validator(mode="after")
    def _historical_interval_is_exactly_bounded(self) -> Self:
        if self.confidence_level != 0.95:
            raise ValueError("historical source confidence_level must be exactly 0.95")
        if self.event_count > self.records:
            raise ValueError("event_count cannot exceed records")
        if self.resampling_units > self.records:
            raise ValueError("patient clusters cannot outnumber records")
        expected = self.event_count / self.records
        if not math.isclose(self.point_estimate, expected, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError("point_estimate does not match event_count / records")
        if not (
            self.two_sided_lower
            <= self.point_estimate
            <= self.two_sided_upper
            and self.point_estimate <= self.one_sided_upper
        ):
            raise ValueError("historical source interval bounds are inconsistent")
        return self


class ProportionBootstrapInterval(StrictFrozenModel):
    """Deterministic percentile interval for a record-weighted proportion.

    For ``record`` resampling, records are sampled with replacement. For
    ``patient_cluster`` resampling, patients are sampled with replacement and
    all their records are carried into the replicate. The public artifact never
    contains the private cluster labels.
    """

    method: Literal["percentile_bootstrap"]
    estimator: Literal["record_weighted_event_rate"]
    resampling_unit: ResamplingUnit
    sampling_with_replacement: StrictTrue
    random_generator: Literal["numpy.random.Generator_PCG64"]
    seed: NonNegativeInt
    replicates: Annotated[int, Field(strict=True, ge=1_000)]
    percentile_function: Literal["numpy.quantile"]
    quantile_method: Literal["linear"]
    confidence_level: Annotated[
        float,
        BeforeValidator(_require_float),
        Field(strict=True, ge=0.5, lt=1.0, allow_inf_nan=False),
    ]
    records: PositiveInt
    resampling_units: PositiveInt
    event_count: NonNegativeInt
    point_estimate: UnitFloat
    two_sided_lower: UnitFloat
    two_sided_upper: UnitFloat
    one_sided_lower: UnitFloat
    one_sided_upper: UnitFloat

    @model_validator(mode="after")
    def _interval_is_consistent(self) -> Self:
        if self.event_count > self.records:
            raise ValueError("event_count cannot exceed records")
        expected = self.event_count / self.records
        if not math.isclose(self.point_estimate, expected, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError("point_estimate does not match event_count / records")
        if self.resampling_unit is ResamplingUnit.RECORD:
            if self.resampling_units != self.records:
                raise ValueError("record bootstrap must have one resampling unit per record")
        elif self.resampling_units > self.records:
            raise ValueError("patient clusters cannot outnumber records")
        if not (
            self.two_sided_lower
            <= self.one_sided_lower
            <= self.one_sided_upper
            <= self.two_sided_upper
        ):
            raise ValueError("bootstrap interval bounds are not consistently ordered")
        return self


class SourceGateSummary(StrictFrozenModel):
    """Aggregate source-retention gate evaluated without sealed-v1 C tuning."""

    cohort_key: Identifier
    cohort_manifest_sha256: Sha256Digest
    evaluation_role: Literal["source_retention"]
    records: PositiveInt
    subjects: PositiveInt
    rejected_records: NonNegativeInt
    retained_records: NonNegativeInt
    false_rejection_rate: UnitFloat
    support_coverage: UnitFloat
    maximum_false_rejection_rate: UnitFloat
    interval: ProportionBootstrapInterval | HistoricalSourceBootstrapInterval
    gate_passed: StrictBool
    sealed_v1_source_validation_used_for_tuning: StrictFalse
    public_contains_record_level_outputs: StrictFalse

    @model_validator(mode="after")
    def _source_gate_is_consistent(self) -> Self:
        if self.subjects > self.records:
            raise ValueError("source subjects cannot exceed records")
        if self.rejected_records + self.retained_records != self.records:
            raise ValueError("source gate counts do not sum to records")
        expected_rate = self.rejected_records / self.records
        if not math.isclose(self.false_rejection_rate, expected_rate, abs_tol=1e-15):
            raise ValueError("false_rejection_rate does not match source counts")
        expected_coverage = self.retained_records / self.records
        if not math.isclose(self.support_coverage, expected_coverage, abs_tol=1e-15):
            raise ValueError("support_coverage does not match source counts")
        if self.interval.records != self.records:
            raise ValueError("source interval record count differs from its cohort")
        if self.interval.event_count != self.rejected_records:
            raise ValueError("source interval event count differs from rejected_records")
        if self.interval.resampling_unit is ResamplingUnit.RECORD:
            if self.subjects != self.records:
                raise ValueError("record-resampled source subjects must equal records")
        elif self.subjects != self.interval.resampling_units:
            raise ValueError("source subjects must equal patient resampling units")
        expected_passed = self.interval.one_sided_upper <= self.maximum_false_rejection_rate
        if self.gate_passed is not expected_passed:
            raise ValueError("source gate must use the one-sided upper confidence bound")
        return self


class ExternalCohortSummary(StrictFrozenModel):
    """Aggregate-only OOD-positive evaluation for one predeclared cohort."""

    endpoint_key: Identifier
    cohort_key: Identifier
    dataset_name: StrictText
    dataset_version: StrictText
    license_identifier: StrictText
    cohort_manifest_sha256: Sha256Digest
    role_assignment_sha256: Sha256Digest
    evaluation_role: ExternalCohortRole
    ood_axis: OODAxis
    records: PositiveInt
    subjects: PositiveInt
    detected_records: NonNegativeInt
    missed_records: NonNegativeInt
    ood_recall: UnitFloat
    minimum_ood_recall: UnitFloat
    interval: ProportionBootstrapInterval
    gate_passed: StrictBool
    target_site_fitting_performed: StrictFalse
    public_contains_record_level_outputs: StrictFalse

    @field_validator("dataset_name", "dataset_version", "license_identifier")
    @classmethod
    def _metadata_is_not_a_path(cls, value: str) -> str:
        if _looks_absolute_path(value):
            raise ValueError("external cohort metadata cannot contain filesystem paths")
        return value

    @model_validator(mode="after")
    def _external_gate_is_consistent(self) -> Self:
        if self.subjects > self.records:
            raise ValueError("external cohort subjects cannot exceed records")
        if self.detected_records + self.missed_records != self.records:
            raise ValueError("external OOD counts do not sum to records")
        expected_recall = self.detected_records / self.records
        if not math.isclose(self.ood_recall, expected_recall, abs_tol=1e-15):
            raise ValueError("ood_recall does not match external cohort counts")
        if self.interval.records != self.records:
            raise ValueError("external interval record count differs from its cohort")
        if self.interval.event_count != self.detected_records:
            raise ValueError("external interval event count differs from detected_records")
        if self.interval.resampling_unit is ResamplingUnit.RECORD:
            if self.subjects != self.records:
                raise ValueError("record-resampled external subjects must equal records")
        elif self.subjects != self.interval.resampling_units:
            raise ValueError("external subjects must equal patient resampling units")
        expected_passed = self.interval.one_sided_lower >= self.minimum_ood_recall
        if self.gate_passed is not expected_passed:
            raise ValueError("OOD gate must use the one-sided lower confidence bound")
        return self


class TechnicalQualityEndpointSummary(StrictFrozenModel):
    """A co-primary technical-quality success rate, separate from OOD recall."""

    endpoint_key: Identifier
    cohort_key: Identifier
    event_definition: TechnicalQualityEventDefinition
    records: PositiveInt
    subjects: PositiveInt
    events: NonNegativeInt
    non_events: NonNegativeInt
    point_rate: UnitFloat
    minimum_rate: UnitFloat
    interval: ProportionBootstrapInterval
    gate_passed: StrictBool
    public_contains_record_level_outputs: StrictFalse

    @model_validator(mode="after")
    def _technical_quality_gate_is_consistent(self) -> Self:
        if self.subjects > self.records:
            raise ValueError("technical-quality subjects cannot exceed records")
        if self.events + self.non_events != self.records:
            raise ValueError("technical-quality counts do not sum to records")
        expected_rate = self.events / self.records
        if not math.isclose(self.point_rate, expected_rate, abs_tol=1e-15):
            raise ValueError("point_rate does not match technical-quality counts")
        if self.interval.records != self.records or self.interval.event_count != self.events:
            raise ValueError("technical-quality interval differs from endpoint counts")
        if self.interval.resampling_unit is ResamplingUnit.RECORD:
            if self.subjects != self.records:
                raise ValueError("record-resampled technical subjects must equal records")
        elif self.subjects != self.interval.resampling_units:
            raise ValueError("technical subjects must equal patient resampling units")
        expected_passed = self.interval.one_sided_lower >= self.minimum_rate
        if self.gate_passed is not expected_passed:
            raise ValueError("technical-quality gate must use the one-sided lower bound")
        return self


class EvidenceRequirements(StrictFrozenModel):
    """Exact multiplicity and bootstrap constants in the frozen parent protocol."""

    family_wise_alpha: UnitFloat
    multiplicity_method: Literal["bonferroni"]
    co_primary_endpoint_count: PositiveInt
    one_sided_alpha_per_endpoint: UnitFloat
    co_primary_confidence_level: Annotated[
        float,
        BeforeValidator(_require_float),
        Field(strict=True, ge=0.5, lt=1.0, allow_inf_nan=False),
    ]
    bootstrap_replicates: PositiveInt
    challenge_bootstrap_seed: NonNegativeInt
    zzu_bootstrap_seed: NonNegativeInt

    @model_validator(mode="after")
    def _requirements_match_frozen_parent(self) -> Self:
        expected = (
            self.family_wise_alpha == 0.05
            and self.co_primary_endpoint_count == 4
            and self.one_sided_alpha_per_endpoint == 0.0125
            and self.co_primary_confidence_level == 0.9875
            and self.bootstrap_replicates == 10_000
            and self.challenge_bootstrap_seed == 20_260_901
            and self.zzu_bootstrap_seed == 20_260_902
        )
        if not expected:
            raise ValueError("evidence requirements differ from the frozen parent protocol")
        return self


class ExternalOODHardGates(StrictFrozenModel):
    """All non-statistical gates frozen by the external v2 parent protocol."""

    challenge_reference_label_alignment_complete: StrictBool
    challenge_invalid_input_count: NonNegativeInt
    challenge_quality_pass_records: NonNegativeInt
    zzu_invalid_input_count: NonNegativeInt
    zzu_selected_records: PositiveInt
    zzu_quality_pass_records: NonNegativeInt
    zzu_quality_pass_record_coverage: UnitFloat
    zzu_selected_patients: PositiveInt
    zzu_quality_pass_patients: NonNegativeInt
    zzu_quality_pass_patient_coverage: UnitFloat
    challenge_group3_prediction_allowed_count: NonNegativeInt
    skipped_selected_records: NonNegativeInt
    target_site_fitting_performed: StrictFalse
    v1_policy_bytes_unchanged_before_and_after: StrictBool
    exact_v1_whole_bundle_verifier_passes: StrictBool
    external_raw_sources_verified_before_and_after: StrictBool
    exact_dataset_roots_verified: StrictBool
    exact_selected_input_inventory_verified_before_and_after: StrictBool
    semantic_roles_rederived_before_and_after: StrictBool
    raw_canonical_lead_and_data_file_bindings_verified: StrictBool
    active_scientific_package_versions_match_child: StrictBool
    deterministic_repeated_embeddings_match: StrictBool
    raw_source_to_canonical_signal_replay_matches: StrictBool
    canonical_signal_to_full_backbone_embedding_replay_matches: StrictBool
    aggregate_only_publication_verified: StrictBool
    immutable_success_bundle_verifies: StrictBool
    failure_receipt_exists: StrictBool
    all_passed: StrictBool

    @model_validator(mode="after")
    def _all_passed_is_derived(self) -> Self:
        if self.zzu_quality_pass_records > self.zzu_selected_records:
            raise ValueError("ZZU quality-pass records cannot exceed selected records")
        if self.zzu_quality_pass_patients > self.zzu_selected_patients:
            raise ValueError("ZZU quality-pass patients cannot exceed selected patients")
        expected_record_coverage = self.zzu_quality_pass_records / self.zzu_selected_records
        expected_patient_coverage = self.zzu_quality_pass_patients / self.zzu_selected_patients
        if not math.isclose(
            self.zzu_quality_pass_record_coverage,
            expected_record_coverage,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError("ZZU quality-pass record coverage differs from its counts")
        if not math.isclose(
            self.zzu_quality_pass_patient_coverage,
            expected_patient_coverage,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError("ZZU quality-pass patient coverage differs from its counts")
        expected = all(
            (
                self.challenge_reference_label_alignment_complete,
                self.challenge_invalid_input_count == 0,
                self.challenge_quality_pass_records >= 1,
                self.zzu_invalid_input_count == 0,
                self.zzu_quality_pass_record_coverage >= 0.80,
                self.zzu_quality_pass_patient_coverage >= 0.80,
                self.challenge_group3_prediction_allowed_count == 0,
                self.skipped_selected_records == 0,
                self.v1_policy_bytes_unchanged_before_and_after,
                self.exact_v1_whole_bundle_verifier_passes,
                self.external_raw_sources_verified_before_and_after,
                self.exact_dataset_roots_verified,
                self.exact_selected_input_inventory_verified_before_and_after,
                self.semantic_roles_rederived_before_and_after,
                self.raw_canonical_lead_and_data_file_bindings_verified,
                self.active_scientific_package_versions_match_child,
                self.deterministic_repeated_embeddings_match,
                self.raw_source_to_canonical_signal_replay_matches,
                self.canonical_signal_to_full_backbone_embedding_replay_matches,
                self.aggregate_only_publication_verified,
                self.immutable_success_bundle_verifies,
                not self.failure_receipt_exists,
            )
        )
        if self.all_passed is not expected:
            raise ValueError("all_passed is not derived from every frozen hard gate")
        return self


class AggregateRouteCounts(StrictFrozenModel):
    """Privacy-safe exact counts for the frozen five-state operational route."""

    INVALID_INPUT: NonNegativeInt
    REACQUIRE: NonNegativeInt
    UNSUPPORTED_INPUT: NonNegativeInt
    ABSTAIN: NonNegativeInt
    PREDICTION_ALLOWED: NonNegativeInt
    total_records: Literal[13_328]

    @model_validator(mode="after")
    def _total_is_derived(self) -> Self:
        observed = (
            self.INVALID_INPUT
            + self.REACQUIRE
            + self.UNSUPPORTED_INPUT
            + self.ABSTAIN
            + self.PREDICTION_ALLOWED
        )
        if observed != self.total_records:
            raise ValueError("five-state route counts do not sum to total_records")
        return self


class OODV2IntegritySummary(StrictFrozenModel):
    """Integrity predicates that distinguish missing evidence from a gate miss."""

    preregistration_frozen_before_evaluation: StrictBool
    cohort_roles_frozen_before_model_outputs: StrictBool
    dataset_hashes_verified: StrictBool
    overlap_exclusions_verified: StrictBool
    frozen_detector_verified: StrictBool
    evaluation_alignment_verified: StrictBool
    aggregate_only_result_verified: StrictBool
    sealed_v1_unchanged_verified: StrictBool
    sealed_v1_source_validation_used_for_tuning: StrictFalse
    target_site_fitting_performed: StrictFalse
    complete: StrictBool

    @model_validator(mode="after")
    def _complete_is_derived(self) -> Self:
        expected = all(
            (
                self.preregistration_frozen_before_evaluation,
                self.cohort_roles_frozen_before_model_outputs,
                self.dataset_hashes_verified,
                self.overlap_exclusions_verified,
                self.frozen_detector_verified,
                self.evaluation_alignment_verified,
                self.aggregate_only_result_verified,
                self.sealed_v1_unchanged_verified,
            )
        )
        if self.complete is not expected:
            raise ValueError("integrity complete flag is not derived from all predicates")
        return self


_CHALLENGE_COHORT_KEY: Final = "physionet-challenge-2011-set-a"
_ZZU_COHORT_KEY: Final = "zzu-pecg-v1"
_CHALLENGE_DISTRIBUTION_ENDPOINT: Final = "challenge_external_distribution_recall"
_ZZU_DISTRIBUTION_ENDPOINT: Final = "zzu_external_distribution_recall"
_GROUP3_TECHNICAL_ENDPOINT: Final = "challenge_group3_technical_block_sensitivity"
_GROUP1_TECHNICAL_ENDPOINT: Final = "challenge_group1_quality_pass_rate"
_EXTERNAL_ENDPOINTS: Final = {
    _CHALLENGE_DISTRIBUTION_ENDPOINT,
    _ZZU_DISTRIBUTION_ENDPOINT,
}
_TECHNICAL_ENDPOINTS: Final = {
    _GROUP3_TECHNICAL_ENDPOINT,
    _GROUP1_TECHNICAL_ENDPOINT,
}


class OODV2ResultBody(StrictFrozenModel):
    """Aggregate external-v2 evidence with fail-closed derived status.

    The historical v1 source gate is mandatory disclosure context, but it is
    not one of this protocol's four co-primary endpoints. Consequently it does
    not turn an otherwise complete external result into a target miss. It does,
    however, keep whole-system integration permanently closed in this artifact.
    """

    schema_version: Literal[1]
    artifact_type: Literal["ecg_trust.ood_external_v2_1_result"]
    protocol_id: Literal["trust-sentinel-ood-external-v2-1-parent"]
    frozen_at_utc: AwareDatetime
    status: OODV2Status
    preregistration_sha256: Sha256Digest
    cohort_role_manifest_sha256: Sha256Digest
    detector_policy_sha256: Sha256Digest
    sealed_v1_result_sha256: Sha256Digest
    sealed_v1_claim_sha256: Sha256Digest
    code_revision: GitRevision
    evidence_requirements: EvidenceRequirements
    source_gate: SourceGateSummary
    external_cohorts: tuple[ExternalCohortSummary, ...] = Field(max_length=2)
    technical_quality_endpoints: tuple[TechnicalQualityEndpointSummary, ...] = Field(
        max_length=2
    )
    final_route_counts: AggregateRouteCounts
    hard_gates: ExternalOODHardGates
    integrity: OODV2IntegritySummary
    external_evidence_eligible: StrictBool
    integration_permitted: StrictFalse
    aggregate_only: StrictTrue
    research_only: StrictTrue
    clinical_validation: StrictFalse

    @field_validator("frozen_at_utc")
    @classmethod
    def _timestamp_is_utc(cls, value: datetime) -> datetime:
        offset = value.utcoffset()
        if offset is None or offset.total_seconds() != 0:
            raise ValueError("frozen_at_utc must use UTC")
        return value

    @model_validator(mode="after")
    def _result_status_is_derived(self) -> Self:
        requirements = self.evidence_requirements
        if self.preregistration_sha256 != OOD_V2_PARENT_CONFIG_SHA256:
            raise ValueError("preregistration_sha256 differs from the frozen parent file")
        if (
            self.detector_policy_sha256 != _V1_DISTRIBUTION_POLICY_FILE_SHA256
            or self.sealed_v1_result_sha256 != _V1_RESULT_FILE_SHA256
            or self.sealed_v1_claim_sha256 != _V1_ONE_SHOT_CLAIM_FILE_SHA256
        ):
            raise ValueError("v1 file bindings differ from the frozen parent protocol")
        self._validate_historical_source_context()

        external_keys = tuple(cohort.endpoint_key for cohort in self.external_cohorts)
        if len(external_keys) != len(set(external_keys)):
            raise ValueError("external endpoint keys must be unique")
        if not set(external_keys).issubset(_EXTERNAL_ENDPOINTS):
            raise ValueError("result contains an endpoint not declared by the frozen parent")
        declared_external_order = (
            _CHALLENGE_DISTRIBUTION_ENDPOINT,
            _ZZU_DISTRIBUTION_ENDPOINT,
        )
        if external_keys != tuple(key for key in declared_external_order if key in external_keys):
            raise ValueError("external endpoints are not in frozen declaration order")
        technical_keys = tuple(
            endpoint.endpoint_key for endpoint in self.technical_quality_endpoints
        )
        if len(technical_keys) != len(set(technical_keys)):
            raise ValueError("technical-quality endpoint keys must be unique")
        if not set(technical_keys).issubset(_TECHNICAL_ENDPOINTS):
            raise ValueError("result contains a technical endpoint absent from the parent")
        declared_technical_order = (
            _GROUP3_TECHNICAL_ENDPOINT,
            _GROUP1_TECHNICAL_ENDPOINT,
        )
        expected_technical_order = tuple(
            key for key in declared_technical_order if key in technical_keys
        )
        if technical_keys != expected_technical_order:
            raise ValueError("technical endpoints are not in frozen declaration order")

        for cohort in self.external_cohorts:
            if cohort.role_assignment_sha256 != self.cohort_role_manifest_sha256:
                raise ValueError("external endpoint role assignment differs from child contract")
            self._validate_external_endpoint(cohort, requirements=requirements)
        for endpoint in self.technical_quality_endpoints:
            self._validate_technical_endpoint(endpoint, requirements=requirements)

        zzu = next(
            (
                cohort
                for cohort in self.external_cohorts
                if cohort.endpoint_key == _ZZU_DISTRIBUTION_ENDPOINT
            ),
            None,
        )
        if zzu is not None and (
            self.hard_gates.zzu_quality_pass_records != zzu.records
            or self.hard_gates.zzu_quality_pass_patients != zzu.subjects
        ):
            raise ValueError("ZZU quality-pass hard-gate counts differ from its endpoint")
        challenge = next(
            (
                cohort
                for cohort in self.external_cohorts
                if cohort.endpoint_key == _CHALLENGE_DISTRIBUTION_ENDPOINT
            ),
            None,
        )
        if (
            challenge is not None
            and self.hard_gates.challenge_quality_pass_records != challenge.records
        ):
            raise ValueError("Challenge quality-pass hard-gate count differs from its endpoint")
        if (
            self.hard_gates.challenge_invalid_input_count
            + self.hard_gates.zzu_invalid_input_count
            != self.final_route_counts.INVALID_INPUT
        ):
            raise ValueError("public INVALID_INPUT count differs from dataset hard gates")
        four_endpoints_complete = (
            set(external_keys) == _EXTERNAL_ENDPOINTS
            and set(technical_keys) == _TECHNICAL_ENDPOINTS
        )
        sufficient = self.integrity.complete and four_endpoints_complete
        external_gates_pass = (
            all(cohort.gate_passed for cohort in self.external_cohorts)
            and all(endpoint.gate_passed for endpoint in self.technical_quality_endpoints)
            and self.hard_gates.all_passed
        )
        expected_status = (
            OODV2Status.EXTERNAL_OOD_INSUFFICIENT_EVIDENCE
            if not sufficient
            else (
                OODV2Status.EXTERNAL_OOD_EVIDENCE_COMPLETE
                if external_gates_pass
                else OODV2Status.EXTERNAL_OOD_TARGET_MISSED
            )
        )
        expected_eligible = expected_status is OODV2Status.EXTERNAL_OOD_EVIDENCE_COMPLETE
        if self.status is not expected_status:
            raise ValueError("status is not derived from external evidence and hard gates")
        if self.external_evidence_eligible is not expected_eligible:
            raise ValueError("external_evidence_eligible is not derived from external status")
        if self.integration_permitted:
            raise ValueError("historically ineligible v1 source evidence forbids integration")
        return self

    def _validate_historical_source_context(self) -> None:
        source = self.source_gate
        interval = source.interval
        if not isinstance(interval, HistoricalSourceBootstrapInterval):
            raise ValueError("source_gate must disclose the published historical v1 interval")
        expected_counts = (
            source.cohort_key == "sealed-v1-source-validation"
            and source.cohort_manifest_sha256
            == _V1_SOURCE_SPLIT_ASSIGNMENT_SHA256
            and source.records == 465
            and source.subjects == 409
            and source.rejected_records == 25
            and source.retained_records == 440
            and source.maximum_false_rejection_rate == 0.05
            and source.gate_passed is False
            and interval.seed == 20_260_829
            and interval.replicates == 10_000
            and interval.resampling_units == 409
            and math.isclose(
                interval.two_sided_lower,
                0.03239566265733224,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            and math.isclose(
                interval.two_sided_upper,
                0.07725321888412018,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            and math.isclose(
                interval.one_sided_upper,
                0.07296137339055794,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
        )
        if not expected_counts:
            raise ValueError("source_gate differs from the published v1 aggregate context")

    @staticmethod
    def _validate_external_endpoint(
        cohort: ExternalCohortSummary,
        *,
        requirements: EvidenceRequirements,
    ) -> None:
        interval = cohort.interval
        shared = (
            cohort.minimum_ood_recall == 0.90
            and interval.confidence_level == requirements.co_primary_confidence_level
            and interval.replicates == requirements.bootstrap_replicates
        )
        if cohort.endpoint_key == _CHALLENGE_DISTRIBUTION_ENDPOINT:
            exact = (
                cohort.cohort_key == _CHALLENGE_COHORT_KEY
                and cohort.dataset_name == "PhysioNet Challenge 2011 Set A"
                and cohort.dataset_version == "1.0.0"
                and cohort.license_identifier == "ODC-By-1.0"
                and cohort.evaluation_role
                is ExternalCohortRole.PHYSIONET_CHALLENGE_2011_SET_A
                and cohort.ood_axis is OODAxis.EXTERNAL_ACQUISITION_AND_POPULATION
                and interval.resampling_unit is ResamplingUnit.RECORD
                and interval.seed == requirements.challenge_bootstrap_seed
            )
        else:
            exact = (
                cohort.cohort_key == _ZZU_COHORT_KEY
                and cohort.dataset_name == "ZZU pediatric ECG"
                and cohort.dataset_version == "1"
                and cohort.license_identifier == "CC-BY-4.0"
                and cohort.evaluation_role is ExternalCohortRole.ZZU_PECG_V1
                and cohort.ood_axis is OODAxis.PEDIATRIC_POPULATION_AND_ACQUISITION
                and interval.resampling_unit is ResamplingUnit.PATIENT_CLUSTER
                and interval.seed == requirements.zzu_bootstrap_seed
            )
        if not shared or not exact:
            raise ValueError("external endpoint differs from its frozen parent definition")

    @staticmethod
    def _validate_technical_endpoint(
        endpoint: TechnicalQualityEndpointSummary,
        *,
        requirements: EvidenceRequirements,
    ) -> None:
        interval = endpoint.interval
        shared = (
            endpoint.cohort_key == _CHALLENGE_COHORT_KEY
            and interval.resampling_unit is ResamplingUnit.RECORD
            and interval.seed == requirements.challenge_bootstrap_seed
            and interval.replicates == requirements.bootstrap_replicates
            and interval.confidence_level == requirements.co_primary_confidence_level
        )
        if endpoint.endpoint_key == _GROUP3_TECHNICAL_ENDPOINT:
            exact = (
                endpoint.event_definition
                is TechnicalQualityEventDefinition.BLOCK_UNACCEPTABLE
                and endpoint.minimum_rate == 0.95
            )
        else:
            exact = (
                endpoint.event_definition is TechnicalQualityEventDefinition.PASS_ACCEPTABLE
                and endpoint.minimum_rate == 0.90
            )
        if not shared or not exact:
            raise ValueError("technical endpoint differs from its frozen parent definition")


class OODV2Result(OODV2ResultBody):
    """Canonical self-hashed v2 result."""

    artifact_sha256: Sha256Digest

    @model_validator(mode="after")
    def _self_hash_and_privacy_verify(self) -> Self:
        observed = canonical_sha256(self.model_dump(mode="json", exclude={"artifact_sha256"}))
        if observed != self.artifact_sha256:
            raise ValueError("artifact_sha256 does not match OOD v2 result")
        assert_aggregate_only_ood_v2_result(self)
        return self


def canonical_json_bytes(value: dict[str, object]) -> bytes:
    """Serialize finite JSON with stable UTF-8 key ordering and no whitespace."""

    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise OODV2IntegrityError("value is not finite canonical JSON") from error


def canonical_sha256(value: dict[str, object]) -> str:
    """Return the prefixed SHA-256 of canonical JSON bytes."""

    return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def seal_ood_v2_result(body: OODV2ResultBody) -> OODV2Result:
    """Validate aggregate privacy and attach a logical content hash."""

    if not isinstance(body, OODV2ResultBody):
        raise TypeError("body must be an OODV2ResultBody")
    assert_aggregate_only_ood_v2_result(body)
    python_payload = body.model_dump(mode="python")
    json_payload = body.model_dump(mode="json")
    return OODV2Result.model_validate(
        {**python_payload, "artifact_sha256": canonical_sha256(json_payload)}
    )


def ood_v2_result_json_bytes(result: OODV2Result) -> bytes:
    """Return the exact newline-terminated transport form of a v2 result."""

    if not isinstance(result, OODV2Result):
        raise TypeError("result must be an OODV2Result")
    validated = OODV2Result.model_validate(result.model_dump(mode="python"))
    assert_aggregate_only_ood_v2_result(validated)
    return canonical_json_bytes(validated.model_dump(mode="json")) + b"\n"


def load_ood_v2_result_bytes(payload: bytes) -> OODV2Result:
    """Load only an exact canonical v2 JSON artifact and verify its self-hash."""

    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes")
    if not payload or len(payload) > MAX_OOD_V2_RESULT_BYTES:
        raise OODV2IntegrityError("OOD v2 result has an invalid byte length")
    if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
        raise OODV2IntegrityError("OOD v2 result must have exactly one trailing newline")
    raw = payload[:-1]
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise OODV2IntegrityError("OOD v2 result is not UTF-8 JSON") from error

    def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise OODV2IntegrityError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        parsed = json.loads(text, object_pairs_hook=reject_duplicate_keys)
    except OODV2IntegrityError:
        raise
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        raise OODV2IntegrityError("OOD v2 result is not valid JSON") from error
    if not isinstance(parsed, dict):
        raise OODV2IntegrityError("OOD v2 result must be a JSON object")
    object_payload = cast(dict[str, object], parsed)
    if canonical_json_bytes(object_payload) != raw:
        raise OODV2IntegrityError("OOD v2 result is not canonical JSON")
    try:
        result = OODV2Result.model_validate(object_payload)
    except ValidationError as error:
        raise OODV2IntegrityError("OOD v2 result violates its strict contract") from error
    assert_aggregate_only_ood_v2_result(result)
    return result


_FORBIDDEN_PUBLIC_KEYS: Final = frozenset(
    {
        "patient_id",
        "patient_ids",
        "subject_id",
        "subject_ids",
        "record_id",
        "record_ids",
        "ecg_id",
        "ecg_ids",
        "row",
        "rows",
        "waveform",
        "waveforms",
        "embedding",
        "embeddings",
        "score",
        "scores",
        "logit",
        "logits",
        "probability",
        "probabilities",
        "prediction",
        "predictions",
        "file_path",
        "filepath",
    }
)


def assert_aggregate_only_ood_v2_result(
    result: OODV2ResultBody | OODV2Result,
) -> None:
    """Defense-in-depth scan for private row data and filesystem paths."""

    if not isinstance(result, OODV2ResultBody):
        raise TypeError("result must be an OODV2ResultBody or OODV2Result")
    value = result.model_dump(mode="json")

    def visit(item: object) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                normalized = key.casefold()
                if normalized in _FORBIDDEN_PUBLIC_KEYS or normalized.endswith("_path"):
                    raise OODV2IntegrityError(
                        f"aggregate result contains forbidden public field: {key}"
                    )
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)
        elif isinstance(item, str) and _looks_absolute_path(item):
            raise OODV2IntegrityError("aggregate result contains an absolute filesystem path")

    visit(value)


def _looks_absolute_path(value: str) -> bool:
    return PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute()


__all__ = [
    "MAX_OOD_V2_RESULT_BYTES",
    "OOD_V2_ARTIFACT_TYPE",
    "OOD_V2_PARENT_CONFIG_SHA256",
    "OOD_V2_ORIGINAL_PROTOCOL_ID",
    "OOD_V2_PROTOCOL_ID",
    "OOD_V2_RESULT_FILENAME",
    "OOD_V2_SCHEMA_VERSION",
    "AggregateRouteCounts",
    "EvidenceRequirements",
    "ExternalCohortRole",
    "ExternalCohortSummary",
    "ExternalOODHardGates",
    "HistoricalSourceBootstrapInterval",
    "OODAxis",
    "OODV2Error",
    "OODV2IntegrityError",
    "OODV2IntegritySummary",
    "OODV2Result",
    "OODV2ResultBody",
    "OODV2Status",
    "ProportionBootstrapInterval",
    "ResamplingUnit",
    "SourceGateSummary",
    "TechnicalQualityEndpointSummary",
    "TechnicalQualityEventDefinition",
    "assert_aggregate_only_ood_v2_result",
    "canonical_json_bytes",
    "canonical_sha256",
    "load_ood_v2_result_bytes",
    "ood_v2_result_json_bytes",
    "seal_ood_v2_result",
]
