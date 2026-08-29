"""Privacy-safe, aggregate-only model passport contracts."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Final, Literal, Self

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)

from ecg_trust.constants import LEADS, SUPERCLASSES
from ecg_trust.contracts import MetricDirection

MODEL_PASSPORT_SCHEMA_VERSION: Final[Literal["ecg_trust.model_passport.v1"]] = (
    "ecg_trust.model_passport.v1"
)
INPUT_CONTRACT_SCHEMA_VERSION: Final[Literal["ecg_trust.passport.input_contract.v1"]] = (
    "ecg_trust.passport.input_contract.v1"
)
COHORT_EVIDENCE_SCHEMA_VERSION: Final[Literal["ecg_trust.passport.cohort.v1"]] = (
    "ecg_trust.passport.cohort.v1"
)
CONFIDENCE_INTERVAL_SCHEMA_VERSION: Final[Literal["ecg_trust.passport.confidence_interval.v1"]] = (
    "ecg_trust.passport.confidence_interval.v1"
)
AGGREGATE_METRIC_SCHEMA_VERSION: Final[Literal["ecg_trust.passport.metric.v1"]] = (
    "ecg_trust.passport.metric.v1"
)
LABEL_PERFORMANCE_SCHEMA_VERSION: Final[Literal["ecg_trust.passport.label_performance.v1"]] = (
    "ecg_trust.passport.label_performance.v1"
)
MACRO_PERFORMANCE_SCHEMA_VERSION: Final[Literal["ecg_trust.passport.macro_performance.v1"]] = (
    "ecg_trust.passport.macro_performance.v1"
)
SUBGROUP_EVIDENCE_SCHEMA_VERSION: Final[Literal["ecg_trust.passport.subgroup_evidence.v1"]] = (
    "ecg_trust.passport.subgroup_evidence.v1"
)
EXTERNAL_TRANSPORT_SCHEMA_VERSION: Final[Literal["ecg_trust.passport.external_transport.v1"]] = (
    "ecg_trust.passport.external_transport.v1"
)
OOD_EVIDENCE_SCHEMA_VERSION: Final[Literal["ecg_trust.passport.ood_evidence.v1"]] = (
    "ecg_trust.passport.ood_evidence.v1"
)
QUALITY_EVIDENCE_SCHEMA_VERSION: Final[Literal["ecg_trust.passport.quality_evidence.v1"]] = (
    "ecg_trust.passport.quality_evidence.v1"
)
SELECTIVE_EVIDENCE_SCHEMA_VERSION: Final[Literal["ecg_trust.passport.selective_evidence.v1"]] = (
    "ecg_trust.passport.selective_evidence.v1"
)
CONFORMAL_EVIDENCE_SCHEMA_VERSION: Final[Literal["ecg_trust.passport.conformal_evidence.v1"]] = (
    "ecg_trust.passport.conformal_evidence.v1"
)

RESEARCH_ONLY_NOTICE: Final[
    Literal[
        "Research use only. Not for diagnosis or clinical decision-making. "
        "This passport is not clinical validation."
    ]
] = (
    "Research use only. Not for diagnosis or clinical decision-making. "
    "This passport is not clinical validation."
)
CONFORMAL_SCOPE_TEXT: Final[
    Literal[
        "Coverage is label-wise marginal under exchangeability; "
        "it is not an individual certainty guarantee."
    ]
] = (
    "Coverage is label-wise marginal under exchangeability; "
    "it is not an individual certainty guarantee."
)
MAX_CANONICAL_PASSPORT_BYTES = 2 * 1024 * 1024

Sha256Digest = Annotated[
    str,
    StringConstraints(
        strict=True,
        pattern=r"^sha256:[0-9a-f]{64}$",
        min_length=71,
        max_length=71,
    ),
]
Identifier = Annotated[
    str,
    StringConstraints(
        strict=True,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
        min_length=1,
        max_length=160,
    ),
]
SafeText = Annotated[str, StringConstraints(strict=True, min_length=1, max_length=1200)]
FiniteFloat = Annotated[float, Field(strict=True, allow_inf_nan=False)]
Probability = Annotated[
    float,
    Field(strict=True, ge=0.0, le=1.0, allow_inf_nan=False),
]
PositiveInt = Annotated[int, Field(strict=True, ge=1)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
LeadName = Literal["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
SuperclassLabel = Literal["NORM", "MI", "STTC", "CD", "HYP"]

_FORBIDDEN_KEY = re.compile(
    r"(?:^|_)(?:patient|record|case|signal|ecg)_(?:id|ids|identifier|identifiers)(?:_|$)"
    r"|(?:^|_)(?:per_(?:patient|record|case)|raw_(?:ecg|samples?|signals?|waveforms?)|"
    r"signals?|waveforms?|rows?)(?:_|$)"
    r"|(?:^|_)(?:secret|secrets|password|passwords|credential|credentials|api_key|"
    r"access_token|refresh_token|private_key|authorization)(?:_|$)",
    flags=re.IGNORECASE,
)
_ABSOLUTE_PATH = re.compile(
    r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]"
    r"|\\\\[^\\\s]+[\\/]"
    r"|file://"
    r"|(?:^|[\s\"'=(])/(?!/)[^\s]+",
    flags=re.IGNORECASE,
)
_SECRET_VALUE = re.compile(
    r"(?:-----BEGIN [A-Z ]*PRIVATE KEY-----|\bsk-[A-Za-z0-9_-]{16,}\b|"
    r"\bghp_[A-Za-z0-9]{16,}\b|\bAKIA[0-9A-Z]{16}\b|\bBearer\s+[A-Za-z0-9._~-]{12,})"
    r"|\beyJ[A-Za-z0-9_-]{16,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"
)


class ModelPassportError(Exception):
    """Base class for passport validation failures."""


class ModelPassportPrivacyError(ModelPassportError):
    """Raised when row-level, identifying, path, or secret material is supplied."""


class ModelPassportIntegrityError(ModelPassportError):
    """Raised when canonical content or an expected identity does not verify."""


class PassportContract(BaseModel):
    """Immutable contract with a recursive privacy boundary."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        revalidate_instances="always",
        validate_default=True,
        use_enum_values=False,
    )

    @model_validator(mode="before")
    @classmethod
    def _privacy_boundary(cls, value: object) -> object:
        _scan_private_material(value, context=cls.__name__)
        return value


class CohortRole(StrEnum):
    DEVELOPMENT = "DEVELOPMENT"
    SELECTION = "SELECTION"
    CALIBRATION = "CALIBRATION"
    SEALED_TEST = "SEALED_TEST"
    EXTERNAL_TRANSPORT = "EXTERNAL_TRANSPORT"
    QUALITY_AUDIT = "QUALITY_AUDIT"
    ROBUSTNESS_AUDIT = "ROBUSTNESS_AUDIT"


class EvidenceStatus(StrEnum):
    AVAILABLE = "AVAILABLE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    NOT_EVALUATED = "NOT_EVALUATED"


class MinimumEvidenceStatus(StrEnum):
    SUFFICIENT_EVIDENCE = "SUFFICIENT_EVIDENCE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class SupportedInputContract(PassportContract):
    """The only signal shape supported by this release family."""

    schema_version: Literal["ecg_trust.passport.input_contract.v1"]
    recording_type: Literal["RESTING_12_LEAD_ECG"]
    lead_order: tuple[LeadName, ...]
    sampling_frequency_hz: FiniteFloat
    samples_per_lead: PositiveInt
    duration_seconds: FiniteFloat
    physical_units: Literal["mV"]
    dtype: Literal["float32"]

    @model_validator(mode="after")
    def _canonical_input(self) -> Self:
        if self.lead_order != LEADS:
            raise ValueError("lead_order must exactly match the canonical 12-lead order")
        if self.sampling_frequency_hz != 100.0:
            raise ValueError("sampling_frequency_hz must be exactly 100 Hz")
        if self.samples_per_lead != 1000 or self.duration_seconds != 10.0:
            raise ValueError("supported inputs must contain exactly 1000 samples / 10 seconds")
        return self

    @classmethod
    def canonical(cls) -> Self:
        return cls(
            schema_version=INPUT_CONTRACT_SCHEMA_VERSION,
            recording_type="RESTING_12_LEAD_ECG",
            lead_order=(
                "I",
                "II",
                "III",
                "aVR",
                "aVL",
                "aVF",
                "V1",
                "V2",
                "V3",
                "V4",
                "V5",
                "V6",
            ),
            sampling_frequency_hz=100.0,
            samples_per_lead=1000,
            duration_seconds=10.0,
            physical_units="mV",
            dtype="float32",
        )


class CohortEvidence(PassportContract):
    """Aggregate identity and counts for one dataset/site cohort."""

    schema_version: Literal["ecg_trust.passport.cohort.v1"]
    cohort_id: Identifier
    dataset_name: SafeText
    dataset_version: SafeText
    site_name: SafeText
    role: CohortRole
    sample_count: PositiveInt
    patient_count: PositiveInt
    manifest_sha256: Sha256Digest
    definition: SafeText

    @model_validator(mode="after")
    def _counts_are_consistent(self) -> Self:
        if self.patient_count > self.sample_count:
            raise ValueError("patient_count cannot exceed sample_count")
        return self


class AggregateConfidenceInterval(PassportContract):
    schema_version: Literal["ecg_trust.passport.confidence_interval.v1"]
    lower: FiniteFloat
    upper: FiniteFloat
    confidence_level: Probability
    method: Identifier

    @model_validator(mode="after")
    def _ordered_interval(self) -> Self:
        if self.lower > self.upper:
            raise ValueError("confidence interval lower must not exceed upper")
        if self.confidence_level <= 0.0:
            raise ValueError("confidence_level must be greater than zero")
        return self


class AggregateMetric(PassportContract):
    """One scalar aggregate; arrays and row-level observations are unsupported."""

    schema_version: Literal["ecg_trust.passport.metric.v1"]
    metric_id: Identifier
    display_name: SafeText
    definition: SafeText
    direction: MetricDirection
    status: EvidenceStatus
    value: FiniteFloat | None
    unit: SafeText
    sample_count: NonNegativeInt
    confidence_interval: AggregateConfidenceInterval | None = None

    @model_validator(mode="after")
    def _availability_is_explicit(self) -> Self:
        if self.status is EvidenceStatus.AVAILABLE:
            if self.value is None or self.sample_count < 1:
                raise ValueError("an available metric requires a value and positive sample_count")
            interval = self.confidence_interval
            if interval is not None and not interval.lower <= self.value <= interval.upper:
                raise ValueError("metric value must lie inside its confidence interval")
        elif self.value is not None or self.confidence_interval is not None:
            raise ValueError("an unavailable metric cannot publish a value or interval")
        return self


class LabelPerformance(PassportContract):
    schema_version: Literal["ecg_trust.passport.label_performance.v1"]
    label: SuperclassLabel
    discrimination: AggregateMetric
    calibration: AggregateMetric


class MacroPerformance(PassportContract):
    schema_version: Literal["ecg_trust.passport.macro_performance.v1"]
    discrimination: AggregateMetric
    calibration: AggregateMetric


class SubgroupEvidence(PassportContract):
    """Aggregate subgroup evidence hidden when minimum counts are not met."""

    schema_version: Literal["ecg_trust.passport.subgroup_evidence.v1"]
    evidence_id: Identifier
    cohort_id: Identifier
    attribute: Identifier
    group_name: SafeText
    sample_count: NonNegativeInt
    patient_count: NonNegativeInt
    minimum_sample_count: PositiveInt
    minimum_patient_count: PositiveInt
    status: MinimumEvidenceStatus
    metrics: tuple[AggregateMetric, ...] = ()
    reason_codes: tuple[Identifier, ...] = Field(min_length=1)

    @field_validator("metrics")
    @classmethod
    def _ordered_unique_subgroup_metrics(
        cls, value: tuple[AggregateMetric, ...]
    ) -> tuple[AggregateMetric, ...]:
        _require_unique_sorted([metric.metric_id for metric in value], context="subgroup metric_id")
        return value

    @model_validator(mode="after")
    def _minimum_evidence_is_derived(self) -> Self:
        if self.patient_count > self.sample_count:
            raise ValueError("patient_count cannot exceed sample_count")
        sufficient = (
            self.sample_count >= self.minimum_sample_count
            and self.patient_count >= self.minimum_patient_count
        )
        expected = (
            MinimumEvidenceStatus.SUFFICIENT_EVIDENCE
            if sufficient
            else MinimumEvidenceStatus.INSUFFICIENT_EVIDENCE
        )
        if self.status is not expected:
            raise ValueError("subgroup status does not match the minimum-evidence rule")
        required_reason = "MINIMUM_EVIDENCE_MET" if sufficient else "MINIMUM_EVIDENCE_NOT_MET"
        if required_reason not in self.reason_codes:
            raise ValueError(f"reason_codes must include {required_reason}")
        if sufficient and not self.metrics:
            raise ValueError("sufficient subgroup evidence requires aggregate metrics")
        if not sufficient and self.metrics:
            raise ValueError("insufficient subgroup evidence must suppress metrics")
        return self


class EvidenceSummaryBase(PassportContract):
    """Shared aggregate evidence envelope without a polymorphic schema version."""

    evidence_id: Identifier
    status: EvidenceStatus
    method: Identifier
    artifact_sha256: Sha256Digest | None
    cohort_ids: tuple[Identifier, ...] = Field(min_length=1)
    sample_count: NonNegativeInt
    patient_count: NonNegativeInt
    metrics: tuple[AggregateMetric, ...] = ()
    summary: SafeText
    limitations: tuple[SafeText, ...] = Field(min_length=1)

    @field_validator("cohort_ids")
    @classmethod
    def _ordered_unique_cohorts(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        _require_unique_sorted(list(value), context="evidence cohort_id")
        return value

    @field_validator("metrics")
    @classmethod
    def _ordered_unique_metrics(
        cls, value: tuple[AggregateMetric, ...]
    ) -> tuple[AggregateMetric, ...]:
        _require_unique_sorted([metric.metric_id for metric in value], context="evidence metric_id")
        return value

    @model_validator(mode="after")
    def _summary_availability(self) -> Self:
        if self.patient_count > self.sample_count:
            raise ValueError("patient_count cannot exceed sample_count")
        if self.status is EvidenceStatus.AVAILABLE:
            if self.artifact_sha256 is None:
                raise ValueError("available evidence requires an artifact_sha256")
            if self.sample_count < 1 or self.patient_count < 1 or not self.metrics:
                raise ValueError("available evidence requires positive counts and metrics")
        elif self.metrics:
            raise ValueError("unavailable evidence summaries cannot publish metrics")
        return self


class ExternalTransportEvidence(EvidenceSummaryBase):
    schema_version: Literal["ecg_trust.passport.external_transport.v1"]
    frozen_source_model: Literal[True]
    target_adaptation: Literal["NONE"]


class OODEvidenceSummary(EvidenceSummaryBase):
    schema_version: Literal["ecg_trust.passport.ood_evidence.v1"]
    score_direction: Literal["HIGHER_IS_MORE_OUT_OF_DISTRIBUTION"]
    threshold_scope: Literal["SOURCE_CALIBRATION_ONLY"]


class QualityEvidenceSummary(EvidenceSummaryBase):
    schema_version: Literal["ecg_trust.passport.quality_evidence.v1"]
    policy_scope: Literal["DETERMINISTIC_SIGNAL_QUALITY"]
    frozen_policy: Literal[True]


class SelectiveEvidenceSummary(EvidenceSummaryBase):
    schema_version: Literal["ecg_trust.passport.selective_evidence.v1"]
    policy_scope: Literal["FROZEN_ABSTENTION_GATE"]
    frozen_policy: Literal[True]


class ConformalEvidenceSummary(EvidenceSummaryBase):
    schema_version: Literal["ecg_trust.passport.conformal_evidence.v1"]
    coverage_scope: Literal["labelwise_marginal_under_exchangeability"]
    coverage_scope_text: Literal[
        "Coverage is label-wise marginal under exchangeability; "
        "it is not an individual certainty guarantee."
    ]
    calibration_scope: Literal["SOURCE_CALIBRATION_ONLY"]


class ModelPassportBody(PassportContract):
    """Canonical aggregate evidence body prior to self-hashing."""

    schema_version: Literal["ecg_trust.model_passport.v1"]
    passport_id: Identifier
    generated_at: AwareDatetime
    release_id: Identifier
    release_sha256: Sha256Digest
    bundle_sha256: Sha256Digest
    protocol_sha256: Sha256Digest
    supported_input: SupportedInputContract
    cohorts: tuple[CohortEvidence, ...] = Field(min_length=1)
    label_performance: tuple[LabelPerformance, ...] = Field(min_length=5, max_length=5)
    macro_performance: MacroPerformance
    subgroup_evidence: tuple[SubgroupEvidence, ...] = Field(min_length=1)
    external_transport: tuple[ExternalTransportEvidence, ...] = Field(min_length=1)
    ood_evidence: OODEvidenceSummary
    quality_evidence: QualityEvidenceSummary
    selective_evidence: SelectiveEvidenceSummary
    conformal_evidence: ConformalEvidenceSummary
    limitations: tuple[SafeText, ...] = Field(min_length=1)
    research_only: Literal[True]
    clinically_validated: Literal[False]
    clinical_use_permitted: Literal[False]
    safety_notice: Literal[
        "Research use only. Not for diagnosis or clinical decision-making. "
        "This passport is not clinical validation."
    ]

    @field_validator("generated_at")
    @classmethod
    def _generated_at_is_utc(cls, value: datetime) -> datetime:
        offset = value.utcoffset()
        if offset is None or offset.total_seconds() != 0:
            raise ValueError("generated_at must use UTC")
        return value

    @field_validator("cohorts")
    @classmethod
    def _ordered_unique_declared_cohorts(
        cls, value: tuple[CohortEvidence, ...]
    ) -> tuple[CohortEvidence, ...]:
        _require_unique_sorted([cohort.cohort_id for cohort in value], context="cohort_id")
        return value

    @field_validator("label_performance")
    @classmethod
    def _canonical_label_performance(
        cls, value: tuple[LabelPerformance, ...]
    ) -> tuple[LabelPerformance, ...]:
        if tuple(item.label for item in value) != SUPERCLASSES:
            raise ValueError("label_performance must contain the five labels in canonical order")
        return value

    @field_validator("subgroup_evidence")
    @classmethod
    def _ordered_unique_subgroups(
        cls, value: tuple[SubgroupEvidence, ...]
    ) -> tuple[SubgroupEvidence, ...]:
        _require_unique_sorted([item.evidence_id for item in value], context="subgroup evidence_id")
        return value

    @field_validator("external_transport")
    @classmethod
    def _ordered_unique_transport(
        cls, value: tuple[ExternalTransportEvidence, ...]
    ) -> tuple[ExternalTransportEvidence, ...]:
        _require_unique_sorted(
            [item.evidence_id for item in value], context="transport evidence_id"
        )
        return value

    @model_validator(mode="after")
    def _all_evidence_references_declared_cohorts(self) -> Self:
        declared = {cohort.cohort_id for cohort in self.cohorts}
        referenced: set[str] = {item.cohort_id for item in self.subgroup_evidence}
        summaries: tuple[EvidenceSummaryBase, ...] = (
            *self.external_transport,
            self.ood_evidence,
            self.quality_evidence,
            self.selective_evidence,
            self.conformal_evidence,
        )
        for summary in summaries:
            referenced.update(summary.cohort_ids)
        unknown = referenced - declared
        if unknown:
            raise ValueError(f"evidence references undeclared cohorts: {sorted(unknown)!r}")
        roles = {cohort.cohort_id: cohort.role for cohort in self.cohorts}
        for transport in self.external_transport:
            if not any(
                roles[cohort_id] is CohortRole.EXTERNAL_TRANSPORT
                for cohort_id in transport.cohort_ids
            ):
                raise ValueError(
                    "external transport evidence must reference an EXTERNAL_TRANSPORT cohort"
                )
        if len(set(self.limitations)) != len(self.limitations):
            raise ValueError("limitations must not contain duplicates")
        return self


class ModelPassport(ModelPassportBody):
    """Immutable, self-hashed model passport."""

    passport_sha256: Sha256Digest

    @model_validator(mode="after")
    def _self_hash_verifies(self) -> Self:
        observed = canonical_sha256(self.model_dump(mode="json", exclude={"passport_sha256"}))
        if observed != self.passport_sha256:
            raise ValueError("passport_sha256 does not match the canonical passport body")
        return self


def build_model_passport(
    *,
    passport_id: str,
    generated_at: datetime,
    release_id: str,
    release_sha256: str,
    bundle_sha256: str,
    protocol_sha256: str,
    supported_input: SupportedInputContract,
    cohorts: Sequence[CohortEvidence],
    label_performance: Sequence[LabelPerformance],
    macro_performance: MacroPerformance,
    subgroup_evidence: Sequence[SubgroupEvidence],
    external_transport: Sequence[ExternalTransportEvidence],
    ood_evidence: OODEvidenceSummary,
    quality_evidence: QualityEvidenceSummary,
    selective_evidence: SelectiveEvidenceSummary,
    conformal_evidence: ConformalEvidenceSummary,
    limitations: Sequence[str],
) -> ModelPassport:
    """Sort set-like evidence, validate privacy, and seal a deterministic passport."""

    label_rank = {label: index for index, label in enumerate(SUPERCLASSES)}
    body = ModelPassportBody(
        schema_version=MODEL_PASSPORT_SCHEMA_VERSION,
        passport_id=passport_id,
        generated_at=generated_at,
        release_id=release_id,
        release_sha256=release_sha256,
        bundle_sha256=bundle_sha256,
        protocol_sha256=protocol_sha256,
        supported_input=supported_input,
        cohorts=tuple(sorted(cohorts, key=lambda item: item.cohort_id)),
        label_performance=tuple(sorted(label_performance, key=lambda item: label_rank[item.label])),
        macro_performance=macro_performance,
        subgroup_evidence=tuple(sorted(subgroup_evidence, key=lambda item: item.evidence_id)),
        external_transport=tuple(sorted(external_transport, key=lambda item: item.evidence_id)),
        ood_evidence=ood_evidence,
        quality_evidence=quality_evidence,
        selective_evidence=selective_evidence,
        conformal_evidence=conformal_evidence,
        limitations=tuple(sorted(limitations)),
        research_only=True,
        clinically_validated=False,
        clinical_use_permitted=False,
        safety_notice=RESEARCH_ONLY_NOTICE,
    )
    canonical_body = body.model_dump(mode="json")
    python_body = body.model_dump(mode="python")
    return ModelPassport.model_validate(
        {**python_body, "passport_sha256": canonical_sha256(canonical_body)}
    )


def validate_model_passport(
    passport: ModelPassport,
    *,
    expected_release_sha256: str,
    expected_bundle_sha256: str,
    expected_protocol_sha256: str,
    expected_release_id: str | None = None,
) -> ModelPassport:
    """Revalidate content and fail closed on any expected release identity mismatch."""

    try:
        validated = ModelPassport.model_validate(passport.model_dump(mode="python"))
    except ValidationError as error:
        raise ModelPassportIntegrityError(f"invalid model passport: {error}") from error
    expected = {
        "release_sha256": expected_release_sha256,
        "bundle_sha256": expected_bundle_sha256,
        "protocol_sha256": expected_protocol_sha256,
    }
    for field, identity in expected.items():
        if getattr(validated, field) != identity:
            raise ModelPassportIntegrityError(f"{field} differs from expectation")
    if expected_release_id is not None and validated.release_id != expected_release_id:
        raise ModelPassportIntegrityError("release_id differs from expectation")
    return validated


def canonical_json_bytes(value: Mapping[str, object]) -> bytes:
    """Serialize finite aggregate JSON deterministically."""

    _scan_private_material(value, context="canonical JSON")
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ModelPassportIntegrityError("passport is not finite canonical JSON") from error


def canonical_sha256(value: Mapping[str, object]) -> str:
    return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def model_passport_to_json_bytes(passport: ModelPassport) -> bytes:
    """Return the one accepted serialized representation."""

    return canonical_json_bytes(passport.model_dump(mode="json")) + b"\n"


def model_passport_from_json_bytes(
    payload: bytes,
    *,
    expected_release_sha256: str,
    expected_bundle_sha256: str,
    expected_protocol_sha256: str,
    expected_release_id: str | None = None,
) -> ModelPassport:
    """Load only canonical, privacy-safe, identity-compatible passport JSON."""

    if not 1 < len(payload) <= MAX_CANONICAL_PASSPORT_BYTES:
        raise ModelPassportIntegrityError("model passport payload size is invalid")
    try:
        passport = ModelPassport.model_validate_json(payload)
    except (ValidationError, UnicodeError) as error:
        raise ModelPassportIntegrityError(f"could not decode model passport: {error}") from error
    if payload != model_passport_to_json_bytes(passport):
        raise ModelPassportIntegrityError("model passport JSON is not canonical")
    return validate_model_passport(
        passport,
        expected_release_sha256=expected_release_sha256,
        expected_bundle_sha256=expected_bundle_sha256,
        expected_protocol_sha256=expected_protocol_sha256,
        expected_release_id=expected_release_id,
    )


def _require_unique_sorted(values: list[str], *, context: str) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"{context} values must be unique")
    if values != sorted(values):
        raise ValueError(f"{context} values must be sorted")


def _scan_private_material(value: object, *, context: str) -> None:
    if isinstance(value, BaseModel):
        return
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            if not isinstance(raw_key, str):
                raise ModelPassportPrivacyError(f"{context} contains a non-string key")
            normalized = raw_key.strip().replace("-", "_").replace(" ", "_")
            if _FORBIDDEN_KEY.search(normalized):
                raise ModelPassportPrivacyError(
                    f"{context} contains forbidden row-level or secret field {raw_key!r}"
                )
            _scan_private_material(item, context=f"{context}.{raw_key}")
        return
    if isinstance(value, str):
        if _ABSOLUTE_PATH.search(value):
            raise ModelPassportPrivacyError(f"{context} contains an absolute path")
        if _SECRET_VALUE.search(value):
            raise ModelPassportPrivacyError(f"{context} contains secret-like material")
        if any(ord(character) < 32 for character in value):
            raise ModelPassportPrivacyError(f"{context} contains control characters")
        return
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        if value and all(isinstance(item, (int, float, bool)) for item in value):
            raise ModelPassportPrivacyError(f"{context} contains a forbidden numeric row array")
        for index, item in enumerate(value):
            _scan_private_material(item, context=f"{context}[{index}]")


__all__ = [
    "AGGREGATE_METRIC_SCHEMA_VERSION",
    "COHORT_EVIDENCE_SCHEMA_VERSION",
    "CONFIDENCE_INTERVAL_SCHEMA_VERSION",
    "CONFORMAL_EVIDENCE_SCHEMA_VERSION",
    "CONFORMAL_SCOPE_TEXT",
    "EXTERNAL_TRANSPORT_SCHEMA_VERSION",
    "INPUT_CONTRACT_SCHEMA_VERSION",
    "LABEL_PERFORMANCE_SCHEMA_VERSION",
    "MACRO_PERFORMANCE_SCHEMA_VERSION",
    "MAX_CANONICAL_PASSPORT_BYTES",
    "MODEL_PASSPORT_SCHEMA_VERSION",
    "OOD_EVIDENCE_SCHEMA_VERSION",
    "QUALITY_EVIDENCE_SCHEMA_VERSION",
    "RESEARCH_ONLY_NOTICE",
    "SELECTIVE_EVIDENCE_SCHEMA_VERSION",
    "SUBGROUP_EVIDENCE_SCHEMA_VERSION",
    "AggregateConfidenceInterval",
    "AggregateMetric",
    "CohortEvidence",
    "CohortRole",
    "ConformalEvidenceSummary",
    "EvidenceStatus",
    "ExternalTransportEvidence",
    "LabelPerformance",
    "MacroPerformance",
    "MetricDirection",
    "MinimumEvidenceStatus",
    "ModelPassport",
    "ModelPassportBody",
    "ModelPassportError",
    "ModelPassportIntegrityError",
    "ModelPassportPrivacyError",
    "OODEvidenceSummary",
    "QualityEvidenceSummary",
    "SelectiveEvidenceSummary",
    "SubgroupEvidence",
    "SupportedInputContract",
    "build_model_passport",
    "canonical_json_bytes",
    "canonical_sha256",
    "model_passport_from_json_bytes",
    "model_passport_to_json_bytes",
    "validate_model_passport",
]
