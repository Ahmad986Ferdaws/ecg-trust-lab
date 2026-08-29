"""Versioned, fail-closed contracts for ECG Trust Sentinel.

The API contracts deliberately carry metadata and content identities only. Raw
waveforms and attribution arrays belong in separately verified binary artifacts.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Final, Literal, Self

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from ecg_trust.constants import LEADS

SIGNAL_ENVELOPE_SCHEMA_VERSION: Final[Literal["ecg_trust.signal_envelope.v1"]] = (
    "ecg_trust.signal_envelope.v1"
)
QUALITY_FINDING_SCHEMA_VERSION: Final[Literal["ecg_trust.quality_finding.v1"]] = (
    "ecg_trust.quality_finding.v1"
)
QUALITY_REPORT_SCHEMA_VERSION: Final[Literal["ecg_trust.quality_report.v1"]] = (
    "ecg_trust.quality_report.v1"
)
DISTRIBUTION_METRIC_SCHEMA_VERSION: Final[Literal["ecg_trust.distribution_metric.v1"]] = (
    "ecg_trust.distribution_metric.v1"
)
DISTRIBUTION_REPORT_SCHEMA_VERSION: Final[Literal["ecg_trust.distribution_report.v1"]] = (
    "ecg_trust.distribution_report.v1"
)
CASE_DISTRIBUTION_ASSESSMENT_SCHEMA_VERSION: Final[
    Literal["ecg_trust.case_distribution_assessment.v1"]
] = "ecg_trust.case_distribution_assessment.v1"
LABEL_UNCERTAINTY_SCHEMA_VERSION: Final[Literal["ecg_trust.label_uncertainty_decision.v1"]] = (
    "ecg_trust.label_uncertainty_decision.v1"
)
LABEL_PREDICTION_SET_DECISION_SCHEMA_VERSION: Final[
    Literal["ecg_trust.label_prediction_set_decision.v1"]
] = "ecg_trust.label_prediction_set_decision.v1"
EXPLANATION_ENVELOPE_SCHEMA_VERSION: Final[Literal["ecg_trust.explanation_envelope.v1"]] = (
    "ecg_trust.explanation_envelope.v1"
)
AUDIT_EVENT_SCHEMA_VERSION: Final[Literal["ecg_trust.audit_event.v1"]] = "ecg_trust.audit_event.v1"
EVIDENCE_BUNDLE_SCHEMA_VERSION: Final[Literal["ecg_trust.evidence_bundle.v1"]] = (
    "ecg_trust.evidence_bundle.v1"
)
ARTIFACT_REFERENCE_SCHEMA_VERSION: Final[Literal["ecg_trust.artifact_reference.v1"]] = (
    "ecg_trust.artifact_reference.v1"
)
EVIDENCE_COHORT_SCHEMA_VERSION: Final[Literal["ecg_trust.evidence_cohort.v1"]] = (
    "ecg_trust.evidence_cohort.v1"
)
EVIDENCE_METRIC_SCHEMA_VERSION: Final[Literal["ecg_trust.evidence_metric.v1"]] = (
    "ecg_trust.evidence_metric.v1"
)
CONFIDENCE_INTERVAL_SCHEMA_VERSION: Final[Literal["ecg_trust.confidence_interval.v1"]] = (
    "ecg_trust.confidence_interval.v1"
)

CANONICAL_SAMPLING_FREQUENCY_HZ = 100.0
CANONICAL_SAMPLES_PER_LEAD = 1000
CANONICAL_DURATION_SECONDS = 10.0
CANONICAL_SIGNAL_BYTES = len(LEADS) * CANONICAL_SAMPLES_PER_LEAD * 4
CANONICAL_SIGNAL_MEDIA_TYPE = "application/vnd.ecg-trust.signal+float32"
CONFORMAL_COVERAGE_SCOPE: Final[Literal["labelwise_marginal_under_exchangeability"]] = (
    "labelwise_marginal_under_exchangeability"
)
CONFORMAL_COVERAGE_SCOPE_TEXT: Final[
    Literal[
        "Coverage is label-wise marginal under exchangeability; "
        "it is not an individual certainty guarantee."
    ]
] = (
    "Coverage is label-wise marginal under exchangeability; "
    "it is not an individual certainty guarantee."
)

Sha256Digest = Annotated[
    str,
    StringConstraints(
        strict=True,
        pattern=r"^sha256:[0-9a-f]{64}$",
        min_length=71,
        max_length=71,
    ),
]
OpaqueIdentifier = Annotated[
    str,
    StringConstraints(
        strict=True,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
        min_length=1,
        max_length=160,
    ),
]
ReasonCode = Annotated[
    str,
    StringConstraints(
        strict=True,
        pattern=r"^[A-Z][A-Z0-9_]*$",
        min_length=1,
        max_length=96,
    ),
]
NonEmptyText = Annotated[str, StringConstraints(strict=True, min_length=1, max_length=1000)]
MediaType = Annotated[
    str,
    StringConstraints(
        strict=True,
        pattern=r"^[a-z0-9][a-z0-9.+-]*/[a-z0-9][a-z0-9.+-]*$",
        min_length=3,
        max_length=128,
    ),
]
FiniteFloat = Annotated[float, Field(strict=True, allow_inf_nan=False)]
Probability = Annotated[
    float,
    Field(strict=True, ge=0.0, le=1.0, allow_inf_nan=False),
]
PositiveInt = Annotated[int, Field(strict=True, ge=1)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
LeadName = Literal["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
SuperclassLabel = Literal["NORM", "MI", "STTC", "CD", "HYP"]


class StrictContract(BaseModel):
    """Shared behavior for immutable boundary objects."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_default=True,
        use_enum_values=False,
    )


class TrustDecision(StrEnum):
    """The complete and intentionally small Sentinel decision vocabulary."""

    INVALID_INPUT = "INVALID_INPUT"
    REACQUIRE = "REACQUIRE"
    UNSUPPORTED_INPUT = "UNSUPPORTED_INPUT"
    ABSTAIN = "ABSTAIN"
    PREDICTION_ALLOWED = "PREDICTION_ALLOWED"


class FindingSeverity(StrEnum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class DistributionStatus(StrEnum):
    WITHIN_REFERENCE = "WITHIN_REFERENCE"
    SHIFT_DETECTED = "SHIFT_DETECTED"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class CaseDistributionStatus(StrEnum):
    WITHIN_REFERENCE = "WITHIN_REFERENCE"
    OUTSIDE_REFERENCE = "OUTSIDE_REFERENCE"
    UNAVAILABLE = "UNAVAILABLE"


class PredictionSetDecision(StrEnum):
    SUPPORTED = "SUPPORTED"
    NOT_SUPPORTED = "NOT_SUPPORTED"
    UNCERTAIN = "UNCERTAIN"


class PredictionSetUncertaintyKind(StrEnum):
    NONE = "NONE"
    BOTH = "BOTH"
    EMPTY = "EMPTY"


class ThresholdDirection(StrEnum):
    ABOVE = "ABOVE"
    BELOW = "BELOW"
    ABSOLUTE_ABOVE = "ABSOLUTE_ABOVE"


class SignalSourceKind(StrEnum):
    APPROVED_EXAMPLE = "APPROVED_EXAMPLE"
    SYNTHETIC = "SYNTHETIC"
    AUTHORIZED_UPLOAD = "AUTHORIZED_UPLOAD"


class ExplanationMethod(StrEnum):
    GRAD_CAM_1D = "GRAD_CAM_1D"
    INTEGRATED_GRADIENTS = "INTEGRATED_GRADIENTS"


class AuditAction(StrEnum):
    """Closed action vocabulary shared by audit producers and durable storage."""

    CASE_VALIDATION = "CASE_VALIDATION"
    INFERENCE = "INFERENCE"
    RELEASE_VERIFICATION = "RELEASE_VERIFICATION"
    POLICY_CHANGE = "POLICY_CHANGE"
    MONITORING_ACTION = "MONITORING_ACTION"


class AuditActorType(StrEnum):
    USER = "USER"
    SERVICE = "SERVICE"
    SYSTEM = "SYSTEM"


class AuditOutcome(StrEnum):
    SUCCESS = "SUCCESS"
    DENIED = "DENIED"
    FAILED = "FAILED"


class MetricDirection(StrEnum):
    HIGHER_IS_BETTER = "HIGHER_IS_BETTER"
    LOWER_IS_BETTER = "LOWER_IS_BETTER"
    DESCRIPTIVE = "DESCRIPTIVE"


class ArtifactReference(StrictContract):
    """Content identity for a blob resolved by a trusted storage adapter."""

    schema_version: Literal["ecg_trust.artifact_reference.v1"]
    artifact_id: OpaqueIdentifier
    file_sha256: Sha256Digest
    size_bytes: PositiveInt
    media_type: MediaType
    sensitive: bool = Field(strict=True)


class QualityFinding(StrictContract):
    """One machine-readable input-quality finding."""

    schema_version: Literal["ecg_trust.quality_finding.v1"]
    code: ReasonCode
    severity: FindingSeverity
    message: NonEmptyText
    affected_leads: tuple[LeadName, ...] = ()
    sample_start: Annotated[int, Field(strict=True, ge=0, lt=CANONICAL_SAMPLES_PER_LEAD)] | None = (
        None
    )
    sample_end: Annotated[int, Field(strict=True, gt=0, le=CANONICAL_SAMPLES_PER_LEAD)] | None = (
        None
    )

    @field_validator("affected_leads")
    @classmethod
    def _canonical_affected_leads(cls, value: tuple[LeadName, ...]) -> tuple[LeadName, ...]:
        if len(set(value)) != len(value):
            raise ValueError("affected_leads must not contain duplicates")
        positions = [LEADS.index(lead) for lead in value]
        if positions != sorted(positions):
            raise ValueError("affected_leads must use canonical lead order")
        return value

    @model_validator(mode="after")
    def _complete_sample_interval(self) -> Self:
        if (self.sample_start is None) != (self.sample_end is None):
            raise ValueError(
                "sample_start and sample_end must either both be set or both be absent"
            )
        if (
            self.sample_start is not None
            and self.sample_end is not None
            and self.sample_start >= self.sample_end
        ):
            raise ValueError("sample_start must be less than sample_end")
        return self


class QualityReport(StrictContract):
    """Input-quality gate; uncertainty abstention is represented separately."""

    schema_version: Literal["ecg_trust.quality_report.v1"]
    evaluated_at: AwareDatetime
    passed: bool = Field(strict=True)
    decision: TrustDecision
    findings: tuple[QualityFinding, ...] = ()

    @model_validator(mode="after")
    def _decision_matches_findings(self) -> Self:
        has_error = any(finding.severity is FindingSeverity.ERROR for finding in self.findings)
        if self.passed:
            if self.decision is not TrustDecision.PREDICTION_ALLOWED or has_error:
                raise ValueError(
                    "a passed quality report must allow prediction and contain no errors"
                )
        elif (
            self.decision not in {TrustDecision.INVALID_INPUT, TrustDecision.REACQUIRE}
            or not self.findings
        ):
            raise ValueError(
                "a failed quality report requires findings and an input-quality decision"
            )
        return self


class DistributionMetric(StrictContract):
    """One thresholded distribution comparison."""

    schema_version: Literal["ecg_trust.distribution_metric.v1"]
    feature_name: OpaqueIdentifier
    metric_name: OpaqueIdentifier
    value: FiniteFloat
    threshold: FiniteFloat
    threshold_direction: ThresholdDirection
    threshold_exceeded: bool = Field(strict=True)

    @model_validator(mode="after")
    def _threshold_result_is_derived(self) -> Self:
        if self.threshold_direction is ThresholdDirection.ABOVE:
            observed = self.value > self.threshold
        elif self.threshold_direction is ThresholdDirection.BELOW:
            observed = self.value < self.threshold
        else:
            if self.threshold < 0.0:
                raise ValueError("an ABSOLUTE_ABOVE threshold must be non-negative")
            observed = abs(self.value) > self.threshold
        if observed is not self.threshold_exceeded:
            raise ValueError("threshold_exceeded does not match value, threshold, and direction")
        return self


class DistributionReport(StrictContract):
    """Aggregate shift report with an explicit sample-sufficiency state."""

    schema_version: Literal["ecg_trust.distribution_report.v1"]
    generated_at: AwareDatetime
    reference_cohort_id: OpaqueIdentifier
    observed_cohort_id: OpaqueIdentifier
    reference_count: PositiveInt
    observed_count: NonNegativeInt
    minimum_observed_count: PositiveInt
    status: DistributionStatus
    metrics: tuple[DistributionMetric, ...]

    @field_validator("metrics")
    @classmethod
    def _unique_metrics(
        cls, value: tuple[DistributionMetric, ...]
    ) -> tuple[DistributionMetric, ...]:
        identities = [(metric.feature_name, metric.metric_name) for metric in value]
        if len(set(identities)) != len(identities):
            raise ValueError("distribution metrics must have unique feature/metric identities")
        return value

    @model_validator(mode="after")
    def _status_is_derived(self) -> Self:
        if self.observed_count < self.minimum_observed_count:
            if self.status is not DistributionStatus.INSUFFICIENT_DATA:
                raise ValueError("an undersized observed cohort must be INSUFFICIENT_DATA")
            return self
        if self.status is DistributionStatus.INSUFFICIENT_DATA:
            raise ValueError("a sufficiently large cohort cannot be INSUFFICIENT_DATA")
        if not self.metrics:
            raise ValueError("a sufficient distribution report requires at least one metric")
        expected = (
            DistributionStatus.SHIFT_DETECTED
            if any(metric.threshold_exceeded for metric in self.metrics)
            else DistributionStatus.WITHIN_REFERENCE
        )
        if self.status is not expected:
            raise ValueError("distribution status does not match thresholded metrics")
        return self


class CaseDistributionAssessment(StrictContract):
    """Case-level OOD assessment, distinct from aggregate cohort drift."""

    schema_version: Literal["ecg_trust.case_distribution_assessment.v1"]
    assessment_id: OpaqueIdentifier
    signal_id: OpaqueIdentifier
    release_id: OpaqueIdentifier
    method: OpaqueIdentifier
    method_artifact: ArtifactReference
    expected_method_schema_version: PositiveInt
    observed_method_schema_version: PositiveInt | None
    artifact_available: bool = Field(strict=True)
    score_direction: Literal["HIGHER_IS_MORE_OUT_OF_DISTRIBUTION"]
    score: FiniteFloat | None
    threshold: FiniteFloat | None
    is_out_of_distribution: bool | None = Field(strict=True)
    status: CaseDistributionStatus
    reason_codes: tuple[ReasonCode, ...] = Field(min_length=1)

    @field_validator("reason_codes")
    @classmethod
    def _unique_case_reason_codes(cls, value: tuple[ReasonCode, ...]) -> tuple[ReasonCode, ...]:
        if len(set(value)) != len(value):
            raise ValueError("reason_codes must not contain duplicates")
        return value

    @model_validator(mode="after")
    def _fail_closed_case_assessment(self) -> Self:
        result_values = (self.score, self.threshold, self.is_out_of_distribution)
        if not self.artifact_available:
            if self.observed_method_schema_version is not None:
                raise ValueError("an unavailable method artifact cannot have an observed version")
            self._require_unavailable_results("ARTIFACT_UNAVAILABLE", result_values)
            return self

        if self.observed_method_schema_version is None:
            raise ValueError("an available method artifact requires its observed schema version")
        if self.observed_method_schema_version != self.expected_method_schema_version:
            self._require_unavailable_results("ARTIFACT_VERSION_MISMATCH", result_values)
            return self

        if self.status is CaseDistributionStatus.UNAVAILABLE:
            self._require_unavailable_results("SCORING_UNAVAILABLE", result_values)
            return self
        if any(value is None for value in result_values):
            raise ValueError("an available case assessment requires score, threshold, and flag")

        score = self.score
        threshold = self.threshold
        observed_flag = self.is_out_of_distribution
        if score is None or threshold is None or observed_flag is None:
            raise ValueError("an available case assessment is incomplete")
        expected_flag = score > threshold
        if observed_flag is not expected_flag:
            raise ValueError("is_out_of_distribution must be derived as score > threshold")
        expected_status = (
            CaseDistributionStatus.OUTSIDE_REFERENCE
            if expected_flag
            else CaseDistributionStatus.WITHIN_REFERENCE
        )
        if self.status is not expected_status:
            raise ValueError("case distribution status does not match the derived flag")
        required_reason = (
            "SCORE_ABOVE_FROZEN_THRESHOLD" if expected_flag else "SCORE_WITHIN_FROZEN_THRESHOLD"
        )
        if required_reason not in self.reason_codes:
            raise ValueError(f"reason_codes must include {required_reason}")
        return self

    def _require_unavailable_results(
        self,
        required_reason: str,
        result_values: tuple[float | None, float | None, bool | None],
    ) -> None:
        if self.status is not CaseDistributionStatus.UNAVAILABLE:
            raise ValueError("artifact unavailability or incompatibility must fail closed")
        if any(value is not None for value in result_values):
            raise ValueError("an unavailable assessment cannot expose score, threshold, or flag")
        if required_reason not in self.reason_codes:
            raise ValueError(f"reason_codes must include {required_reason}")


class LabelUncertaintyDecision(StrictContract):
    """A calibrated, thresholded, per-label uncertainty decision."""

    schema_version: Literal["ecg_trust.label_uncertainty_decision.v1"]
    label: SuperclassLabel
    calibrated_probability: Probability
    classification_threshold: Probability
    predicted_positive: bool = Field(strict=True)
    uncertainty_score: Probability
    uncertainty_threshold: Probability
    decision: TrustDecision
    reason_codes: tuple[ReasonCode, ...] = Field(min_length=1)

    @field_validator("reason_codes")
    @classmethod
    def _unique_reason_codes(cls, value: tuple[ReasonCode, ...]) -> tuple[ReasonCode, ...]:
        if len(set(value)) != len(value):
            raise ValueError("reason_codes must not contain duplicates")
        return value

    @model_validator(mode="after")
    def _derived_prediction_and_decision(self) -> Self:
        expected_positive = self.calibrated_probability >= self.classification_threshold
        if self.predicted_positive is not expected_positive:
            raise ValueError("predicted_positive does not match the calibrated probability")
        if self.decision not in {
            TrustDecision.ABSTAIN,
            TrustDecision.PREDICTION_ALLOWED,
        }:
            raise ValueError("per-label uncertainty decision must be ABSTAIN or PREDICTION_ALLOWED")
        expected_decision = (
            TrustDecision.PREDICTION_ALLOWED
            if self.uncertainty_score <= self.uncertainty_threshold
            else TrustDecision.ABSTAIN
        )
        if self.decision is not expected_decision:
            raise ValueError("decision does not match the uncertainty gate")
        return self


class LabelPredictionSetDecision(StrictContract):
    """One label-wise conformal set translated into an explicit three-state result."""

    schema_version: Literal["ecg_trust.label_prediction_set_decision.v1"]
    label: SuperclassLabel
    calibrated_probability: Probability
    include_supported: bool = Field(strict=True)
    include_not_supported: bool = Field(strict=True)
    decision: PredictionSetDecision
    uncertainty_kind: PredictionSetUncertaintyKind
    calibration_artifact: ArtifactReference
    calibration_artifact_type: Literal["ecg_trust.labelwise_binary_conformal"]
    calibration_artifact_schema_version: Literal[1]
    coverage_scope: Literal["labelwise_marginal_under_exchangeability"]
    coverage_scope_text: Literal[
        "Coverage is label-wise marginal under exchangeability; "
        "it is not an individual certainty guarantee."
    ]

    @model_validator(mode="after")
    def _prediction_set_membership_is_authoritative(self) -> Self:
        membership = (self.include_not_supported, self.include_supported)
        if membership == (False, True):
            expected_decision = PredictionSetDecision.SUPPORTED
            expected_uncertainty = PredictionSetUncertaintyKind.NONE
        elif membership == (True, False):
            expected_decision = PredictionSetDecision.NOT_SUPPORTED
            expected_uncertainty = PredictionSetUncertaintyKind.NONE
        elif membership == (True, True):
            expected_decision = PredictionSetDecision.UNCERTAIN
            expected_uncertainty = PredictionSetUncertaintyKind.BOTH
        else:
            expected_decision = PredictionSetDecision.UNCERTAIN
            expected_uncertainty = PredictionSetUncertaintyKind.EMPTY
        if self.decision is not expected_decision:
            raise ValueError("decision contradicts conformal prediction-set membership")
        if self.uncertainty_kind is not expected_uncertainty:
            raise ValueError("uncertainty_kind contradicts conformal prediction-set membership")
        return self


class SignalEnvelope(StrictContract):
    """Canonical ECG metadata plus an opaque binary artifact identity."""

    schema_version: Literal["ecg_trust.signal_envelope.v1"]
    signal_id: OpaqueIdentifier
    source_kind: SignalSourceKind
    payload: ArtifactReference
    lead_order: tuple[LeadName, ...]
    sampling_frequency_hz: FiniteFloat
    samples_per_lead: PositiveInt
    duration_seconds: FiniteFloat
    units: Literal["mV"]
    dtype: Literal["float32"]
    shape: tuple[PositiveInt, PositiveInt]
    quality: QualityReport

    @model_validator(mode="after")
    def _canonical_signal_contract(self) -> Self:
        if self.lead_order != LEADS:
            raise ValueError("lead_order must exactly match the canonical 12-lead order")
        if self.sampling_frequency_hz != CANONICAL_SAMPLING_FREQUENCY_HZ:
            raise ValueError("sampling_frequency_hz must be exactly 100 Hz")
        if self.samples_per_lead != CANONICAL_SAMPLES_PER_LEAD:
            raise ValueError("samples_per_lead must be exactly 1000")
        if self.duration_seconds != CANONICAL_DURATION_SECONDS:
            raise ValueError("duration_seconds must be exactly 10 seconds")
        if self.shape != (len(LEADS), CANONICAL_SAMPLES_PER_LEAD):
            raise ValueError("shape must be exactly [12, 1000]")
        if self.payload.media_type != CANONICAL_SIGNAL_MEDIA_TYPE:
            raise ValueError("payload media_type is incompatible with the canonical signal")
        if self.payload.size_bytes != CANONICAL_SIGNAL_BYTES:
            raise ValueError("canonical float32 signal payload must contain exactly 48000 bytes")
        if not self.payload.sensitive:
            raise ValueError("ECG signal payloads must be marked sensitive")
        return self


class ExplanationEnvelope(StrictContract):
    """Metadata for a separately stored explanation array."""

    schema_version: Literal["ecg_trust.explanation_envelope.v1"]
    explanation_id: OpaqueIdentifier
    release_id: OpaqueIdentifier
    signal_id: OpaqueIdentifier
    method: ExplanationMethod
    target_label: SuperclassLabel
    attribution: ArtifactReference
    shape: tuple[PositiveInt, PositiveInt]
    coordinate_space: Literal["FROZEN_NORMALIZED_MODEL_INPUT"]
    signed: Literal[True]
    normalized_to: Literal["UNIT_MAX_MAGNITUDE"]
    controls_passed: bool = Field(strict=True)
    control_findings: tuple[QualityFinding, ...] = ()

    @model_validator(mode="after")
    def _explanation_is_consistent(self) -> Self:
        if self.shape not in {
            (1, CANONICAL_SAMPLES_PER_LEAD),
            (len(LEADS), CANONICAL_SAMPLES_PER_LEAD),
        }:
            raise ValueError("explanation shape must be [1, 1000] or [12, 1000]")
        if not self.attribution.sensitive:
            raise ValueError("explanation artifacts must be marked sensitive")
        has_error = any(
            finding.severity is FindingSeverity.ERROR for finding in self.control_findings
        )
        if self.controls_passed is has_error:
            raise ValueError("controls_passed does not match control findings")
        return self


class AuditEvent(StrictContract):
    """Canonical redacted audit event accepted by every durable audit sink."""

    schema_version: Literal["ecg_trust.audit_event.v1"]
    event_id: OpaqueIdentifier
    occurred_at: AwareDatetime
    request_id: OpaqueIdentifier
    actor_type: AuditActorType
    actor_id: OpaqueIdentifier
    action: AuditAction
    resource_type: ReasonCode
    resource_id: OpaqueIdentifier
    release_id: OpaqueIdentifier | None = None
    outcome: AuditOutcome
    reason_code: ReasonCode | None = None
    decision: TrustDecision | None = None

    @model_validator(mode="after")
    def _coherent_outcome_and_decision(self) -> Self:
        if self.outcome is not AuditOutcome.SUCCESS and self.reason_code is None:
            raise ValueError("denied and failed audit events require a reason_code")
        decision_actions = {AuditAction.CASE_VALIDATION, AuditAction.INFERENCE}
        if (
            self.outcome is AuditOutcome.SUCCESS
            and self.action in decision_actions
            and (self.release_id is None or self.decision is None or self.reason_code is None)
        ):
            raise ValueError(
                "successful validation and inference require release, decision, and reason"
            )
        if self.action is AuditAction.RELEASE_VERIFICATION and (
            self.outcome is AuditOutcome.SUCCESS and self.release_id is None
        ):
            raise ValueError("successful release verification requires a release_id")
        if self.action not in decision_actions and self.decision is not None:
            raise ValueError("only validation and inference events may carry a decision")
        if self.decision is TrustDecision.PREDICTION_ALLOWED:
            if self.reason_code != "ALL_TRUST_GATES_PASSED":
                raise ValueError("PREDICTION_ALLOWED requires reason_code ALL_TRUST_GATES_PASSED")
        elif self.reason_code == "ALL_TRUST_GATES_PASSED":
            raise ValueError("ALL_TRUST_GATES_PASSED is reserved for PREDICTION_ALLOWED")
        return self


class ConfidenceInterval(StrictContract):
    schema_version: Literal["ecg_trust.confidence_interval.v1"]
    lower: FiniteFloat
    upper: FiniteFloat
    confidence_level: Probability
    method: OpaqueIdentifier

    @model_validator(mode="after")
    def _ordered_interval(self) -> Self:
        if self.lower > self.upper:
            raise ValueError("confidence interval lower bound must not exceed upper bound")
        if self.confidence_level <= 0.0:
            raise ValueError("confidence_level must be greater than zero")
        return self


class EvidenceCohort(StrictContract):
    schema_version: Literal["ecg_trust.evidence_cohort.v1"]
    cohort_id: OpaqueIdentifier
    dataset_name: NonEmptyText
    dataset_version: NonEmptyText
    role: Literal["SEALED_TEST", "EXTERNAL_TRANSPORT", "STRESS_TEST", "SUBGROUP_AUDIT"]
    record_count: PositiveInt
    patient_count: PositiveInt | None = None
    definition: NonEmptyText

    @model_validator(mode="after")
    def _patient_count_is_bounded(self) -> Self:
        if self.patient_count is not None and self.patient_count > self.record_count:
            raise ValueError("patient_count cannot exceed record_count")
        return self


class EvidenceMetric(StrictContract):
    schema_version: Literal["ecg_trust.evidence_metric.v1"]
    metric_id: OpaqueIdentifier
    display_name: NonEmptyText
    definition: NonEmptyText
    direction: MetricDirection
    value: FiniteFloat
    unit: NonEmptyText
    sample_count: PositiveInt
    confidence_interval: ConfidenceInterval | None = None
    per_seed_values: tuple[FiniteFloat, ...] = ()

    @model_validator(mode="after")
    def _point_is_inside_interval(self) -> Self:
        interval = self.confidence_interval
        if interval is not None and not interval.lower <= self.value <= interval.upper:
            raise ValueError("metric value must lie within its confidence interval")
        return self


class EvidenceBundle(StrictContract):
    """Aggregate-only evidence that can be explicitly approved for publication."""

    schema_version: Literal["ecg_trust.evidence_bundle.v1"]
    evidence_id: OpaqueIdentifier
    release_id: OpaqueIdentifier
    created_at: AwareDatetime
    title: NonEmptyText
    cohort: EvidenceCohort
    metrics: tuple[EvidenceMetric, ...] = Field(min_length=1)
    artifacts: tuple[ArtifactReference, ...] = ()
    source_artifact_sha256: tuple[Sha256Digest, ...] = Field(min_length=1)
    limitations: tuple[NonEmptyText, ...] = Field(min_length=1)
    aggregate_only: Literal[True]
    contains_direct_identifiers: Literal[False]
    research_only: Literal[True]
    publishable: bool = Field(strict=True)

    @model_validator(mode="after")
    def _publication_boundary(self) -> Self:
        metric_ids = [metric.metric_id for metric in self.metrics]
        if len(set(metric_ids)) != len(metric_ids):
            raise ValueError("metric_id values must be unique")
        if len(set(self.source_artifact_sha256)) != len(self.source_artifact_sha256):
            raise ValueError("source_artifact_sha256 values must be unique")
        if self.publishable and any(artifact.sensitive for artifact in self.artifacts):
            raise ValueError("publishable evidence cannot reference sensitive artifacts")
        return self


__all__ = [
    "ARTIFACT_REFERENCE_SCHEMA_VERSION",
    "AUDIT_EVENT_SCHEMA_VERSION",
    "CANONICAL_DURATION_SECONDS",
    "CANONICAL_SAMPLES_PER_LEAD",
    "CANONICAL_SAMPLING_FREQUENCY_HZ",
    "CANONICAL_SIGNAL_BYTES",
    "CANONICAL_SIGNAL_MEDIA_TYPE",
    "CASE_DISTRIBUTION_ASSESSMENT_SCHEMA_VERSION",
    "CONFIDENCE_INTERVAL_SCHEMA_VERSION",
    "CONFORMAL_COVERAGE_SCOPE",
    "CONFORMAL_COVERAGE_SCOPE_TEXT",
    "DISTRIBUTION_METRIC_SCHEMA_VERSION",
    "DISTRIBUTION_REPORT_SCHEMA_VERSION",
    "EVIDENCE_BUNDLE_SCHEMA_VERSION",
    "EVIDENCE_COHORT_SCHEMA_VERSION",
    "EVIDENCE_METRIC_SCHEMA_VERSION",
    "EXPLANATION_ENVELOPE_SCHEMA_VERSION",
    "LABEL_PREDICTION_SET_DECISION_SCHEMA_VERSION",
    "LABEL_UNCERTAINTY_SCHEMA_VERSION",
    "QUALITY_FINDING_SCHEMA_VERSION",
    "QUALITY_REPORT_SCHEMA_VERSION",
    "SIGNAL_ENVELOPE_SCHEMA_VERSION",
    "ArtifactReference",
    "AuditAction",
    "AuditActorType",
    "AuditEvent",
    "AuditOutcome",
    "CaseDistributionAssessment",
    "CaseDistributionStatus",
    "ConfidenceInterval",
    "DistributionMetric",
    "DistributionReport",
    "DistributionStatus",
    "EvidenceBundle",
    "EvidenceCohort",
    "EvidenceMetric",
    "ExplanationEnvelope",
    "ExplanationMethod",
    "FindingSeverity",
    "LabelPredictionSetDecision",
    "LabelUncertaintyDecision",
    "MetricDirection",
    "PredictionSetDecision",
    "PredictionSetUncertaintyKind",
    "QualityFinding",
    "QualityReport",
    "Sha256Digest",
    "SignalEnvelope",
    "SignalSourceKind",
    "ThresholdDirection",
    "TrustDecision",
]
