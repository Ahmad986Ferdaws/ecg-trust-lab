"""Privacy-preserving contracts for a preregistered Sentinel interface study.

This module defines research infrastructure only.  It neither runs a study nor
contains clinical data.  Raw responses are deliberately accepted only through
a closed, pseudonymous schema; the sole evaluation output is aggregate-only.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import re
import statistics
import unicodedata
from bisect import bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from itertools import pairwise
from typing import Annotated, Final, Literal, Self, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)

from ecg_trust.contracts.models import TrustDecision

STUDY_SCHEMA_VERSION: Final[Literal[1]] = 1
STUDY_ARTIFACT_TYPE: Final[Literal["ecg_trust.human_factors_preregistration"]] = (
    "ecg_trust.human_factors_preregistration"
)
STUDY_DESIGN: Final[Literal["randomized_two_period_crossover"]] = "randomized_two_period_crossover"
RANDOMIZATION_ALGORITHM: Final[Literal["hmac_sha256_domain_separated_sort_v1"]] = (
    "hmac_sha256_domain_separated_sort_v1"
)
RESEARCH_USE_LIMIT: Final[Literal["research_only_not_for_clinical_decisions"]] = (
    "research_only_not_for_clinical_decisions"
)
NON_CLINICAL_LIMIT: Final[Literal["not_a_medical_device_no_clinical_use"]] = (
    "not_a_medical_device_no_clinical_use"
)
USABILITY_CLAIM_LIMIT: Final[
    Literal["no_usability_claim_before_preregistered_minimum_evidence_and_effect_thresholds"]
] = "no_usability_claim_before_preregistered_minimum_evidence_and_effect_thresholds"
AGGREGATE_ONLY_LIMIT: Final[Literal["aggregate_only_no_participant_or_scenario_linkage"]] = (
    "aggregate_only_no_participant_or_scenario_linkage"
)

_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_HMAC_RE = re.compile(r"^hmac-sha256:[0-9a-f]{64}$")

StrictIdentifier = Annotated[
    str,
    StringConstraints(
        strict=True,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,159}$",
        min_length=1,
        max_length=160,
    ),
]
ScenarioIdentifier = Annotated[
    str,
    StringConstraints(
        strict=True,
        pattern=r"^scenario_[A-Za-z0-9][A-Za-z0-9._:-]{0,143}$",
        min_length=10,
        max_length=153,
    ),
]
ParticipantToken = Annotated[
    str,
    StringConstraints(strict=True, pattern=r"^p_[0-9a-f]{64}$", min_length=66, max_length=66),
]
CaseToken = Annotated[
    str,
    StringConstraints(strict=True, pattern=r"^c_[0-9a-f]{64}$", min_length=66, max_length=66),
]
TrialToken = Annotated[
    str,
    StringConstraints(strict=True, pattern=r"^t_[0-9a-f]{64}$", min_length=66, max_length=66),
]
Sha256Digest = Annotated[
    str,
    StringConstraints(strict=True, pattern=r"^sha256:[0-9a-f]{64}$", min_length=71, max_length=71),
]
HmacDigest = Annotated[
    str,
    StringConstraints(
        strict=True,
        pattern=r"^hmac-sha256:[0-9a-f]{64}$",
        min_length=76,
        max_length=76,
    ),
]
Probability = Annotated[float, Field(strict=True, ge=0.0, le=1.0, allow_inf_nan=False)]
FiniteFloat = Annotated[float, Field(strict=True, allow_inf_nan=False)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
PositiveInt = Annotated[int, Field(strict=True, ge=1)]


class HumanFactorsError(ValueError):
    """Base error for invalid study operations."""


class StudyIntegrityError(HumanFactorsError):
    """Raised when a self-hash, schedule MAC, or protocol binding is invalid."""


class StudyDataError(HumanFactorsError):
    """Raised when response data cannot be safely aggregated."""


class StrictStudyModel(BaseModel):
    """Shared immutable, closed-schema behavior for study boundary objects."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
        use_enum_values=False,
    )


class StudyArm(StrEnum):
    """The two preregistered interface conditions."""

    TRUST_SENTINEL = "TRUST_SENTINEL_FIVE_STATE"
    PROBABILITIES_ONLY = "PROBABILITIES_ONLY_RESEARCH_BASELINE"


class CaseSource(StrEnum):
    """Permitted non-clinical study material sources."""

    SYNTHETIC = "SYNTHETIC"
    APPROVED_EXAMPLE = "APPROVED_EXAMPLE"


class ParticipantAction(StrEnum):
    """Closed action vocabulary; no free-text response path exists."""

    USE_MODEL_OUTPUT = "USE_MODEL_OUTPUT"
    REACQUIRE_SIGNAL = "REACQUIRE_SIGNAL"
    SEEK_ADDITIONAL_REVIEW = "SEEK_ADDITIONAL_REVIEW"
    DO_NOT_USE = "DO_NOT_USE"


class EvidenceStatus(StrEnum):
    """Whether the preregistered minimum paired evidence is present."""

    UNDERPOWERED = "UNDERPOWERED"
    MINIMUM_EVIDENCE_MET = "MINIMUM_EVIDENCE_MET"


class UsabilityClaimStatus(StrEnum):
    """Fail-closed interpretation of the preregistered research thresholds."""

    PROHIBITED_UNDERPOWERED = "PROHIBITED_UNDERPOWERED"
    PROHIBITED_THRESHOLDS_NOT_MET = "PROHIBITED_THRESHOLDS_NOT_MET"
    THRESHOLDS_MET_CONFIRMATORY_REVIEW_REQUIRED = "THRESHOLDS_MET_CONFIRMATORY_REVIEW_REQUIRED"


_EXPECTED_ACTION: Final[Mapping[TrustDecision, ParticipantAction]] = {
    TrustDecision.INVALID_INPUT: ParticipantAction.DO_NOT_USE,
    TrustDecision.REACQUIRE: ParticipantAction.REACQUIRE_SIGNAL,
    TrustDecision.UNSUPPORTED_INPUT: ParticipantAction.SEEK_ADDITIONAL_REVIEW,
    TrustDecision.ABSTAIN: ParticipantAction.SEEK_ADDITIONAL_REVIEW,
    TrustDecision.PREDICTION_ALLOWED: ParticipantAction.USE_MODEL_OUTPUT,
}
_BLOCKED_DECISIONS: Final[frozenset[TrustDecision]] = frozenset(
    decision for decision in TrustDecision if decision is not TrustDecision.PREDICTION_ALLOWED
)


def expected_action(decision: TrustDecision) -> ParticipantAction:
    """Return the preregistered correct action for one exact Sentinel state."""

    if not isinstance(decision, TrustDecision):
        raise HumanFactorsError("decision must be an exact TrustDecision")
    return _EXPECTED_ACTION[decision]


class StudyCase(StrictStudyModel):
    """Metadata-only study case; patient and waveform fields do not exist."""

    scenario_id: ScenarioIdentifier
    source: CaseSource
    sentinel_decision: TrustDecision


class StudyThresholds(StrictStudyModel):
    """Preregistered evidence and effect thresholds."""

    minimum_paired_participants: Annotated[int, Field(strict=True, ge=2)]
    minimum_trials_per_arm_per_participant: PositiveInt
    minimum_action_accuracy_difference: Annotated[
        float, Field(strict=True, ge=-1.0, le=1.0, allow_inf_nan=False)
    ]
    maximum_overreliance_rate_difference: Annotated[
        float, Field(strict=True, ge=-1.0, le=1.0, allow_inf_nan=False)
    ]
    minimum_sentinel_comprehension_accuracy: Probability


class StudyPreregistration(StrictStudyModel):
    """Self-hashed, immutable randomized-crossover analysis contract."""

    schema_version: Literal[1] = STUDY_SCHEMA_VERSION
    artifact_type: Literal["ecg_trust.human_factors_preregistration"] = STUDY_ARTIFACT_TYPE
    study_id: StrictIdentifier
    design: Literal["randomized_two_period_crossover"] = STUDY_DESIGN
    arms: tuple[StudyArm, StudyArm] = (
        StudyArm.TRUST_SENTINEL,
        StudyArm.PROBABILITIES_ONLY,
    )
    cases: tuple[StudyCase, ...] = Field(min_length=2)
    thresholds: StudyThresholds
    confidence_bin_edges_percent: tuple[Annotated[int, Field(strict=True, ge=0, le=100)], ...] = (
        0,
        20,
        40,
        60,
        80,
        100,
    )
    randomization_algorithm: Literal["hmac_sha256_domain_separated_sort_v1"] = (
        RANDOMIZATION_ALGORITHM
    )
    randomization_secret_in_artifact: Literal[False] = False
    case_use_limit: Literal["synthetic_or_approved_examples_only"] = (
        "synthetic_or_approved_examples_only"
    )
    privacy_contract: Literal["aggregate_only_no_participant_or_scenario_linkage"] = (
        AGGREGATE_ONLY_LIMIT
    )
    research_use_limit: Literal["research_only_not_for_clinical_decisions"] = RESEARCH_USE_LIMIT
    non_clinical_limit: Literal["not_a_medical_device_no_clinical_use"] = NON_CLINICAL_LIMIT
    usability_claim_limit: Literal[
        "no_usability_claim_before_preregistered_minimum_evidence_and_effect_thresholds"
    ] = USABILITY_CLAIM_LIMIT

    @field_validator("cases")
    @classmethod
    def _canonical_cases(cls, value: tuple[StudyCase, ...]) -> tuple[StudyCase, ...]:
        identifiers = tuple(item.scenario_id for item in value)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("scenario identifiers must be unique")
        if identifiers != tuple(sorted(identifiers)):
            raise ValueError("cases must be sorted by scenario_id")
        decisions = {item.sentinel_decision for item in value}
        if TrustDecision.PREDICTION_ALLOWED not in decisions:
            raise ValueError("at least one PREDICTION_ALLOWED case is required")
        if not decisions.intersection(_BLOCKED_DECISIONS):
            raise ValueError("at least one blocked-state case is required")
        return value

    @field_validator("confidence_bin_edges_percent")
    @classmethod
    def _valid_confidence_edges(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if len(value) < 3 or value[0] != 0 or value[-1] != 100:
            raise ValueError("confidence bins must contain at least [0, ..., 100]")
        if any(right <= left for left, right in pairwise(value)):
            raise ValueError("confidence bin edges must be strictly increasing")
        return value

    @model_validator(mode="after")
    def _valid_design(self) -> Self:
        expected_arms = (
            StudyArm.TRUST_SENTINEL,
            StudyArm.PROBABILITIES_ONLY,
        )
        if self.arms != expected_arms:
            raise ValueError("preregistration arms must use the canonical two-arm order")
        if self.thresholds.minimum_trials_per_arm_per_participant > len(self.cases):
            raise ValueError("minimum trials per arm cannot exceed the registered case count")
        return self

    def _body(self) -> dict[str, object]:
        return cast(dict[str, object], self.model_dump(mode="json"))

    @property
    def preregistration_sha256(self) -> str:
        """Content identity over every preregistered field."""

        return _sha256(self._body())

    def to_dict(self) -> dict[str, object]:
        """Serialize with a self-hash while never serializing randomization secrets."""

        return {**self._body(), "preregistration_sha256": self.preregistration_sha256}

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> StudyPreregistration:
        """Restore a preregistration only after strict schema and hash verification."""

        if not isinstance(payload, Mapping) or any(not isinstance(key, str) for key in payload):
            raise StudyIntegrityError("preregistration must be an object with string keys")
        expected_keys = set(cls.model_fields) | {"preregistration_sha256"}
        if set(payload) != expected_keys:
            raise StudyIntegrityError("preregistration fields do not match the closed schema")
        stored_hash = payload["preregistration_sha256"]
        if not isinstance(stored_hash, str) or _SHA256_RE.fullmatch(stored_hash) is None:
            raise StudyIntegrityError("preregistration SHA-256 has an invalid format")
        body = {key: value for key, value in payload.items() if key != "preregistration_sha256"}
        try:
            restored = cls.model_validate_json(_canonical_json(body))
        except ValidationError as error:
            raise StudyIntegrityError("preregistration violates the study contract") from error
        if not hmac.compare_digest(stored_hash, restored.preregistration_sha256):
            raise StudyIntegrityError("preregistration SHA-256 mismatch")
        return restored


class TrialAssignment(StrictStudyModel):
    """One pseudonymous backend assignment; it contains no raw case identifier."""

    trial_token: TrialToken
    case_token: CaseToken
    arm: StudyArm
    period: Annotated[int, Field(strict=True, ge=1, le=2)]
    ordinal_in_period: PositiveInt
    underlying_decision: TrustDecision


class ParticipantSchedule(StrictStudyModel):
    """Authenticated randomized schedule containing pseudonyms only."""

    study_id: StrictIdentifier
    preregistration_sha256: Sha256Digest
    participant_token: ParticipantToken
    arm_order: tuple[StudyArm, StudyArm]
    assignments: tuple[TrialAssignment, ...] = Field(min_length=4)
    schedule_mac: HmacDigest

    @model_validator(mode="after")
    def _valid_schedule_shape(self) -> Self:
        if set(self.arm_order) != {StudyArm.TRUST_SENTINEL, StudyArm.PROBABILITIES_ONLY}:
            raise ValueError("arm_order must contain each registered arm exactly once")
        trial_tokens = tuple(item.trial_token for item in self.assignments)
        case_tokens = tuple(item.case_token for item in self.assignments)
        if len(set(trial_tokens)) != len(trial_tokens):
            raise ValueError("trial tokens must be unique")
        if len(set(case_tokens)) != len(case_tokens):
            raise ValueError("case tokens must be unique")
        for period, arm in enumerate(self.arm_order, start=1):
            period_items = tuple(item for item in self.assignments if item.period == period)
            if not period_items or any(item.arm is not arm for item in period_items):
                raise ValueError("assignment periods must match arm_order")
            ordinals = tuple(item.ordinal_in_period for item in period_items)
            if ordinals != tuple(range(1, len(period_items) + 1)):
                raise ValueError("assignment ordinals must be contiguous within each period")
        expected_order = tuple(
            sorted(self.assignments, key=lambda item: (item.period, item.ordinal_in_period))
        )
        if self.assignments != expected_order:
            raise ValueError("assignments must use canonical period and ordinal order")
        return self

    def _mac_body(self) -> dict[str, object]:
        return {
            "study_id": self.study_id,
            "preregistration_sha256": self.preregistration_sha256,
            "participant_token": self.participant_token,
            "arm_order": [arm.value for arm in self.arm_order],
            "assignments": [item.model_dump(mode="json") for item in self.assignments],
        }


class StudyResponse(StrictStudyModel):
    """Closed response schema with no free text, direct IDs, or signal payloads."""

    participant_token: ParticipantToken
    trial_token: TrialToken
    arm: StudyArm
    selected_action: ParticipantAction
    interpreted_decision: TrustDecision | None
    confidence_percent: Annotated[int, Field(strict=True, ge=0, le=100)]
    decision_time_ms: Annotated[int, Field(strict=True, ge=1, le=1_800_000)]

    @model_validator(mode="after")
    def _arm_specific_comprehension_response(self) -> Self:
        if self.arm is StudyArm.TRUST_SENTINEL and self.interpreted_decision is None:
            raise ValueError("the Sentinel arm requires a closed state interpretation")
        if self.arm is StudyArm.PROBABILITIES_ONLY and self.interpreted_decision is not None:
            raise ValueError("the probabilities-only arm cannot report a hidden Sentinel state")
        return self


class ArmAggregate(StrictStudyModel):
    """Aggregate metrics for one arm; all metrics are suppressed when underpowered."""

    arm: StudyArm
    evidence_status: EvidenceStatus
    paired_participant_count: NonNegativeInt
    response_count: NonNegativeInt
    blocked_response_count: NonNegativeInt
    action_accuracy: Probability | None
    comprehension_accuracy: Probability | None
    overreliance_rate: Probability | None
    mean_decision_time_ms: FiniteFloat | None
    median_decision_time_ms: FiniteFloat | None
    mean_confidence: Probability | None
    confidence_expected_calibration_error: Probability | None
    confidence_brier_score: Probability | None

    @model_validator(mode="after")
    def _underpowered_metrics_are_suppressed(self) -> Self:
        metrics = (
            self.action_accuracy,
            self.comprehension_accuracy,
            self.overreliance_rate,
            self.mean_decision_time_ms,
            self.median_decision_time_ms,
            self.mean_confidence,
            self.confidence_expected_calibration_error,
            self.confidence_brier_score,
        )
        if self.evidence_status is EvidenceStatus.UNDERPOWERED and any(
            value is not None for value in metrics
        ):
            raise ValueError("underpowered arm metrics must be suppressed")
        if self.evidence_status is EvidenceStatus.MINIMUM_EVIDENCE_MET:
            required = tuple(value for index, value in enumerate(metrics) if index != 1)
            if any(value is None for value in required):
                raise ValueError("a sufficiently powered arm requires aggregate metrics")
            if self.arm is StudyArm.TRUST_SENTINEL and self.comprehension_accuracy is None:
                raise ValueError("the Sentinel arm requires comprehension accuracy")
            if self.arm is StudyArm.PROBABILITIES_ONLY and self.comprehension_accuracy is not None:
                raise ValueError("baseline comprehension accuracy is not applicable")
        return self


class PairedAggregate(StrictStudyModel):
    """Mean within-participant Sentinel-minus-baseline differences."""

    evidence_status: EvidenceStatus
    paired_participant_count: NonNegativeInt
    action_accuracy_difference: FiniteFloat | None
    overreliance_rate_difference: FiniteFloat | None
    mean_decision_time_ms_difference: FiniteFloat | None
    confidence_expected_calibration_error_difference: FiniteFloat | None

    @model_validator(mode="after")
    def _minimum_evidence_controls_metrics(self) -> Self:
        metrics = (
            self.action_accuracy_difference,
            self.overreliance_rate_difference,
            self.mean_decision_time_ms_difference,
            self.confidence_expected_calibration_error_difference,
        )
        if self.evidence_status is EvidenceStatus.UNDERPOWERED and any(
            value is not None for value in metrics
        ):
            raise ValueError("underpowered paired metrics must be suppressed")
        if self.evidence_status is EvidenceStatus.MINIMUM_EVIDENCE_MET and any(
            value is None for value in metrics
        ):
            raise ValueError("minimum paired evidence requires all registered differences")
        return self


class StudySummary(StrictStudyModel):
    """Public, aggregate-only study result with fail-closed claims language."""

    preregistration_sha256: Sha256Digest
    scheduled_participant_count: NonNegativeInt
    paired_complete_participant_count: NonNegativeInt
    excluded_incomplete_participant_count: NonNegativeInt
    received_response_count: NonNegativeInt
    evidence_status: EvidenceStatus
    usability_claim_status: UsabilityClaimStatus
    arm_aggregates: tuple[ArmAggregate, ArmAggregate]
    paired_aggregate: PairedAggregate
    privacy_contract: Literal["aggregate_only_no_participant_or_scenario_linkage"] = (
        AGGREGATE_ONLY_LIMIT
    )
    research_use_limit: Literal["research_only_not_for_clinical_decisions"] = RESEARCH_USE_LIMIT
    non_clinical_limit: Literal["not_a_medical_device_no_clinical_use"] = NON_CLINICAL_LIMIT
    usability_claim_limit: Literal[
        "no_usability_claim_before_preregistered_minimum_evidence_and_effect_thresholds"
    ] = USABILITY_CLAIM_LIMIT
    standalone_usability_claim_authorized: Literal[False] = False

    @model_validator(mode="after")
    def _summary_is_consistent(self) -> Self:
        if self.paired_complete_participant_count + self.excluded_incomplete_participant_count != (
            self.scheduled_participant_count
        ):
            raise ValueError("participant counts are inconsistent")
        if tuple(item.arm for item in self.arm_aggregates) != (
            StudyArm.TRUST_SENTINEL,
            StudyArm.PROBABILITIES_ONLY,
        ):
            raise ValueError("arm aggregates must use canonical order")
        if self.paired_aggregate.evidence_status is not self.evidence_status:
            raise ValueError("paired and overall evidence statuses must match")
        if any(item.evidence_status is not self.evidence_status for item in self.arm_aggregates):
            raise ValueError("arm and overall evidence statuses must match")
        if self.evidence_status is EvidenceStatus.UNDERPOWERED and (
            self.usability_claim_status is not UsabilityClaimStatus.PROHIBITED_UNDERPOWERED
        ):
            raise ValueError("underpowered evidence must prohibit a usability claim")
        return self

    def to_public_dict(self) -> dict[str, object]:
        """Return only cohort aggregates, never participant/trial/scenario linkages."""

        return cast(dict[str, object], self.model_dump(mode="json"))


@dataclass(frozen=True, slots=True)
class _MetricValues:
    action_accuracy: float
    comprehension_accuracy: float | None
    overreliance_rate: float
    mean_decision_time_ms: float
    median_decision_time_ms: float
    mean_confidence: float
    confidence_expected_calibration_error: float
    confidence_brier_score: float


def randomize_participant_schedule(
    preregistration: StudyPreregistration,
    *,
    participant_private_key: str,
    randomization_secret: bytes,
) -> ParticipantSchedule:
    """Create a deterministic HMAC-pseudonymous, counterbalanced schedule.

    The private participant key and randomization secret are consumed only as
    HMAC inputs.  Neither value nor a reversible derivative is retained.
    """

    if not isinstance(preregistration, StudyPreregistration):
        raise HumanFactorsError("preregistration must be a validated study contract")
    private_key = _private_participant_key(participant_private_key)
    secret = _secret(randomization_secret)
    participant_token = "p_" + _hmac_hex(
        secret,
        "participant-token",
        preregistration.study_id,
        private_key,
    )
    order_digest = _hmac_bytes(
        secret,
        "arm-order",
        preregistration.study_id,
        private_key,
    )
    canonical_arms = (
        StudyArm.TRUST_SENTINEL,
        StudyArm.PROBABILITIES_ONLY,
    )
    arm_order: tuple[StudyArm, StudyArm]
    if order_digest[0] % 2 == 0:
        arm_order = canonical_arms
    else:
        arm_order = (canonical_arms[1], canonical_arms[0])

    assignments: list[TrialAssignment] = []
    for period, arm in enumerate(arm_order, start=1):
        ordered_cases = sorted(
            preregistration.cases,
            key=lambda item: (
                _hmac_bytes(
                    secret,
                    "case-order",
                    preregistration.study_id,
                    private_key,
                    arm.value,
                    item.scenario_id,
                ),
                item.scenario_id,
            ),
        )
        for ordinal, case in enumerate(ordered_cases, start=1):
            domain_parts = (
                preregistration.study_id,
                private_key,
                arm.value,
                case.scenario_id,
            )
            assignments.append(
                TrialAssignment(
                    trial_token="t_" + _hmac_hex(secret, "trial-token", *domain_parts),
                    case_token="c_" + _hmac_hex(secret, "case-token", *domain_parts),
                    arm=arm,
                    period=period,
                    ordinal_in_period=ordinal,
                    underlying_decision=case.sentinel_decision,
                )
            )

    body: dict[str, object] = {
        "study_id": preregistration.study_id,
        "preregistration_sha256": preregistration.preregistration_sha256,
        "participant_token": participant_token,
        "arm_order": [arm.value for arm in arm_order],
        "assignments": [item.model_dump(mode="json") for item in assignments],
    }
    schedule_mac = _schedule_mac(secret, body)
    return ParticipantSchedule(
        study_id=preregistration.study_id,
        preregistration_sha256=preregistration.preregistration_sha256,
        participant_token=participant_token,
        arm_order=arm_order,
        assignments=tuple(assignments),
        schedule_mac=schedule_mac,
    )


def verify_participant_schedule(
    schedule: ParticipantSchedule,
    preregistration: StudyPreregistration,
    *,
    randomization_secret: bytes,
) -> None:
    """Verify HMAC authenticity and binding to the preregistered design."""

    if not isinstance(schedule, ParticipantSchedule):
        raise StudyIntegrityError("schedule must satisfy the schedule contract")
    if not isinstance(preregistration, StudyPreregistration):
        raise StudyIntegrityError("preregistration must satisfy the study contract")
    secret = _secret(randomization_secret)
    if schedule.study_id != preregistration.study_id or (
        schedule.preregistration_sha256 != preregistration.preregistration_sha256
    ):
        raise StudyIntegrityError("schedule is not bound to this preregistration")
    expected_mac = _schedule_mac(secret, schedule._mac_body())
    if not hmac.compare_digest(schedule.schedule_mac, expected_mac):
        raise StudyIntegrityError("schedule HMAC verification failed")
    case_count = len(preregistration.cases)
    if len(schedule.assignments) != case_count * len(preregistration.arms):
        raise StudyIntegrityError("schedule does not contain every case in both arms")
    expected_decisions = sorted(case.sentinel_decision.value for case in preregistration.cases)
    for arm in preregistration.arms:
        arm_decisions = sorted(
            item.underlying_decision.value for item in schedule.assignments if item.arm is arm
        )
        if arm_decisions != expected_decisions:
            raise StudyIntegrityError("schedule case composition differs from preregistration")


def evaluate_study(
    preregistration: StudyPreregistration,
    schedules: Sequence[ParticipantSchedule],
    responses: Sequence[StudyResponse],
    *,
    randomization_secret: bytes,
) -> StudySummary:
    """Validate joins and return aggregate-only randomized-crossover results.

    Only participants with a response for every assigned trial enter either arm
    or paired metrics.  All descriptive metrics are suppressed until the
    preregistered minimum number of complete paired participants is reached.
    """

    if not isinstance(preregistration, StudyPreregistration):
        raise StudyDataError("preregistration must be a validated study contract")
    if isinstance(schedules, (str, bytes)) or not isinstance(schedules, Sequence):
        raise StudyDataError("schedules must be a sequence")
    if isinstance(responses, (str, bytes)) or not isinstance(responses, Sequence):
        raise StudyDataError("responses must be a sequence")
    if any(not isinstance(item, ParticipantSchedule) for item in schedules):
        raise StudyDataError("schedules contains an invalid object")
    if any(not isinstance(item, StudyResponse) for item in responses):
        raise StudyDataError("responses contains an invalid object")

    schedule_by_participant: dict[str, ParticipantSchedule] = {}
    assignments_by_participant: dict[str, dict[str, TrialAssignment]] = {}
    for schedule in schedules:
        verify_participant_schedule(
            schedule,
            preregistration,
            randomization_secret=randomization_secret,
        )
        if schedule.participant_token in schedule_by_participant:
            raise StudyDataError("participant schedules must be unique")
        schedule_by_participant[schedule.participant_token] = schedule
        assignments_by_participant[schedule.participant_token] = {
            item.trial_token: item for item in schedule.assignments
        }

    response_by_key: dict[tuple[str, str], StudyResponse] = {}
    for response in responses:
        participant_assignments = assignments_by_participant.get(response.participant_token)
        if participant_assignments is None:
            raise StudyDataError("response references an unknown participant schedule")
        assignment = participant_assignments.get(response.trial_token)
        if assignment is None:
            raise StudyDataError("response references an unknown trial")
        if response.arm is not assignment.arm:
            raise StudyDataError("response arm does not match its authenticated assignment")
        key = (response.participant_token, response.trial_token)
        if key in response_by_key:
            raise StudyDataError("duplicate response for a participant trial")
        response_by_key[key] = response

    complete_tokens: list[str] = []
    for participant_token, assignment_map in assignments_by_participant.items():
        answered = sum(
            (participant_token, trial_token) in response_by_key for trial_token in assignment_map
        )
        if answered == len(assignment_map):
            complete_tokens.append(participant_token)

    complete_tokens.sort()
    complete_count = len(complete_tokens)
    evidence_status = (
        EvidenceStatus.MINIMUM_EVIDENCE_MET
        if complete_count >= preregistration.thresholds.minimum_paired_participants
        else EvidenceStatus.UNDERPOWERED
    )

    joined_by_arm: dict[StudyArm, list[tuple[TrialAssignment, StudyResponse]]] = {
        StudyArm.TRUST_SENTINEL: [],
        StudyArm.PROBABILITIES_ONLY: [],
    }
    joined_by_participant: dict[
        str, dict[StudyArm, list[tuple[TrialAssignment, StudyResponse]]]
    ] = {}
    for participant_token in complete_tokens:
        participant_arms: dict[StudyArm, list[tuple[TrialAssignment, StudyResponse]]] = {
            StudyArm.TRUST_SENTINEL: [],
            StudyArm.PROBABILITIES_ONLY: [],
        }
        for assignment in schedule_by_participant[participant_token].assignments:
            response = response_by_key[(participant_token, assignment.trial_token)]
            pair = (assignment, response)
            participant_arms[assignment.arm].append(pair)
            joined_by_arm[assignment.arm].append(pair)
        joined_by_participant[participant_token] = participant_arms

    arm_aggregates = tuple(
        _aggregate_arm(
            arm,
            joined_by_arm[arm],
            paired_participant_count=complete_count,
            evidence_status=evidence_status,
            confidence_edges=preregistration.confidence_bin_edges_percent,
        )
        for arm in (StudyArm.TRUST_SENTINEL, StudyArm.PROBABILITIES_ONLY)
    )
    paired_aggregate = _aggregate_paired(
        joined_by_participant,
        evidence_status=evidence_status,
        confidence_edges=preregistration.confidence_bin_edges_percent,
    )

    if evidence_status is EvidenceStatus.UNDERPOWERED:
        claim_status = UsabilityClaimStatus.PROHIBITED_UNDERPOWERED
    else:
        sentinel = arm_aggregates[0]
        action_difference = paired_aggregate.action_accuracy_difference
        overreliance_difference = paired_aggregate.overreliance_rate_difference
        comprehension = sentinel.comprehension_accuracy
        if action_difference is None or overreliance_difference is None or comprehension is None:
            raise StudyDataError("powered results unexpectedly lack registered metrics")
        thresholds_met = (
            action_difference >= preregistration.thresholds.minimum_action_accuracy_difference
            and overreliance_difference
            <= preregistration.thresholds.maximum_overreliance_rate_difference
            and comprehension >= preregistration.thresholds.minimum_sentinel_comprehension_accuracy
        )
        claim_status = (
            UsabilityClaimStatus.THRESHOLDS_MET_CONFIRMATORY_REVIEW_REQUIRED
            if thresholds_met
            else UsabilityClaimStatus.PROHIBITED_THRESHOLDS_NOT_MET
        )

    return StudySummary(
        preregistration_sha256=preregistration.preregistration_sha256,
        scheduled_participant_count=len(schedules),
        paired_complete_participant_count=complete_count,
        excluded_incomplete_participant_count=len(schedules) - complete_count,
        received_response_count=len(responses),
        evidence_status=evidence_status,
        usability_claim_status=claim_status,
        arm_aggregates=cast(tuple[ArmAggregate, ArmAggregate], arm_aggregates),
        paired_aggregate=paired_aggregate,
    )


def _aggregate_arm(
    arm: StudyArm,
    joined: Sequence[tuple[TrialAssignment, StudyResponse]],
    *,
    paired_participant_count: int,
    evidence_status: EvidenceStatus,
    confidence_edges: tuple[int, ...],
) -> ArmAggregate:
    blocked_count = sum(item[0].underlying_decision in _BLOCKED_DECISIONS for item in joined)
    if evidence_status is EvidenceStatus.UNDERPOWERED:
        return ArmAggregate(
            arm=arm,
            evidence_status=evidence_status,
            paired_participant_count=paired_participant_count,
            response_count=len(joined),
            blocked_response_count=blocked_count,
            action_accuracy=None,
            comprehension_accuracy=None,
            overreliance_rate=None,
            mean_decision_time_ms=None,
            median_decision_time_ms=None,
            mean_confidence=None,
            confidence_expected_calibration_error=None,
            confidence_brier_score=None,
        )
    values = _metric_values(joined, arm=arm, confidence_edges=confidence_edges)
    return ArmAggregate(
        arm=arm,
        evidence_status=evidence_status,
        paired_participant_count=paired_participant_count,
        response_count=len(joined),
        blocked_response_count=blocked_count,
        action_accuracy=values.action_accuracy,
        comprehension_accuracy=values.comprehension_accuracy,
        overreliance_rate=values.overreliance_rate,
        mean_decision_time_ms=values.mean_decision_time_ms,
        median_decision_time_ms=values.median_decision_time_ms,
        mean_confidence=values.mean_confidence,
        confidence_expected_calibration_error=values.confidence_expected_calibration_error,
        confidence_brier_score=values.confidence_brier_score,
    )


def _aggregate_paired(
    joined_by_participant: Mapping[
        str, Mapping[StudyArm, Sequence[tuple[TrialAssignment, StudyResponse]]]
    ],
    *,
    evidence_status: EvidenceStatus,
    confidence_edges: tuple[int, ...],
) -> PairedAggregate:
    if evidence_status is EvidenceStatus.UNDERPOWERED:
        return PairedAggregate(
            evidence_status=evidence_status,
            paired_participant_count=len(joined_by_participant),
            action_accuracy_difference=None,
            overreliance_rate_difference=None,
            mean_decision_time_ms_difference=None,
            confidence_expected_calibration_error_difference=None,
        )
    action_differences: list[float] = []
    overreliance_differences: list[float] = []
    time_differences: list[float] = []
    calibration_differences: list[float] = []
    for participant_token in sorted(joined_by_participant):
        arms = joined_by_participant[participant_token]
        sentinel = _metric_values(
            arms[StudyArm.TRUST_SENTINEL],
            arm=StudyArm.TRUST_SENTINEL,
            confidence_edges=confidence_edges,
        )
        baseline = _metric_values(
            arms[StudyArm.PROBABILITIES_ONLY],
            arm=StudyArm.PROBABILITIES_ONLY,
            confidence_edges=confidence_edges,
        )
        action_differences.append(sentinel.action_accuracy - baseline.action_accuracy)
        overreliance_differences.append(sentinel.overreliance_rate - baseline.overreliance_rate)
        time_differences.append(sentinel.mean_decision_time_ms - baseline.mean_decision_time_ms)
        calibration_differences.append(
            sentinel.confidence_expected_calibration_error
            - baseline.confidence_expected_calibration_error
        )
    return PairedAggregate(
        evidence_status=evidence_status,
        paired_participant_count=len(joined_by_participant),
        action_accuracy_difference=_mean(action_differences),
        overreliance_rate_difference=_mean(overreliance_differences),
        mean_decision_time_ms_difference=_mean(time_differences),
        confidence_expected_calibration_error_difference=_mean(calibration_differences),
    )


def _metric_values(
    joined: Sequence[tuple[TrialAssignment, StudyResponse]],
    *,
    arm: StudyArm,
    confidence_edges: tuple[int, ...],
) -> _MetricValues:
    if not joined:
        raise StudyDataError("complete participant metrics cannot be empty")
    correct = [
        response.selected_action is expected_action(assignment.underlying_decision)
        for assignment, response in joined
    ]
    blocked = [
        (assignment, response)
        for assignment, response in joined
        if assignment.underlying_decision in _BLOCKED_DECISIONS
    ]
    if not blocked:
        raise StudyDataError("the registered design requires blocked-state observations")
    overreliance = sum(
        response.selected_action is ParticipantAction.USE_MODEL_OUTPUT for _, response in blocked
    ) / len(blocked)
    confidence = [response.confidence_percent / 100.0 for _, response in joined]
    correctness = [float(item) for item in correct]
    comprehension: float | None = None
    if arm is StudyArm.TRUST_SENTINEL:
        comprehension = sum(
            response.interpreted_decision is assignment.underlying_decision
            for assignment, response in joined
        ) / len(joined)
    return _MetricValues(
        action_accuracy=sum(correct) / len(correct),
        comprehension_accuracy=comprehension,
        overreliance_rate=overreliance,
        mean_decision_time_ms=_mean([float(response.decision_time_ms) for _, response in joined]),
        median_decision_time_ms=float(
            statistics.median(response.decision_time_ms for _, response in joined)
        ),
        mean_confidence=_mean(confidence),
        confidence_expected_calibration_error=_expected_calibration_error(
            confidence,
            correctness,
            confidence_edges,
        ),
        confidence_brier_score=_mean(
            [
                (confidence_value - correct_value) ** 2
                for confidence_value, correct_value in zip(confidence, correctness, strict=True)
            ]
        ),
    )


def _expected_calibration_error(
    confidence: Sequence[float],
    correctness: Sequence[float],
    edges_percent: tuple[int, ...],
) -> float:
    bins: list[list[int]] = [[] for _ in range(len(edges_percent) - 1)]
    for index, confidence_value in enumerate(confidence):
        percent = confidence_value * 100.0
        bin_index = min(bisect_right(edges_percent, percent) - 1, len(bins) - 1)
        bins[bin_index].append(index)
    total = len(confidence)
    return sum(
        (len(indices) / total)
        * abs(
            _mean([confidence[index] for index in indices])
            - _mean([correctness[index] for index in indices])
        )
        for indices in bins
        if indices
    )


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise StudyDataError("cannot average an empty metric")
    result = math.fsum(values) / len(values)
    if not math.isfinite(result):
        raise StudyDataError("aggregate metric is not finite")
    return float(result)


def _private_participant_key(value: str) -> str:
    if not isinstance(value, str):
        raise HumanFactorsError("participant_private_key must be a string")
    normalized = unicodedata.normalize("NFC", value)
    encoded = normalized.encode("utf-8")
    if (
        not normalized
        or len(encoded) > 512
        or any(unicodedata.category(char) == "Cc" for char in normalized)
    ):
        raise HumanFactorsError(
            "participant_private_key is empty, too long, or contains control data"
        )
    return normalized


def _secret(value: bytes) -> bytes:
    if not isinstance(value, bytes) or len(value) < 32:
        raise HumanFactorsError("randomization_secret must be at least 32 bytes")
    return value


def _domain_message(domain: str, parts: Sequence[str]) -> bytes:
    encoded_parts = [domain.encode("utf-8"), *(part.encode("utf-8") for part in parts)]
    return b"\x00".join(encoded_parts)


def _hmac_bytes(secret: bytes, domain: str, *parts: str) -> bytes:
    return hmac.new(secret, _domain_message(domain, parts), hashlib.sha256).digest()


def _hmac_hex(secret: bytes, domain: str, *parts: str) -> str:
    return _hmac_bytes(secret, domain, *parts).hex()


def _schedule_mac(secret: bytes, body: Mapping[str, object]) -> str:
    return (
        "hmac-sha256:"
        + hmac.new(
            secret,
            _domain_message("schedule-mac", (_canonical_json(body),)),
            hashlib.sha256,
        ).hexdigest()
    )


def _canonical_json(value: Mapping[str, object]) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise HumanFactorsError("study artifact is not canonical JSON data") from error


def _sha256(value: Mapping[str, object]) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


__all__ = [
    "AGGREGATE_ONLY_LIMIT",
    "NON_CLINICAL_LIMIT",
    "RESEARCH_USE_LIMIT",
    "USABILITY_CLAIM_LIMIT",
    "ArmAggregate",
    "CaseSource",
    "EvidenceStatus",
    "HumanFactorsError",
    "PairedAggregate",
    "ParticipantAction",
    "ParticipantSchedule",
    "StudyArm",
    "StudyCase",
    "StudyDataError",
    "StudyIntegrityError",
    "StudyPreregistration",
    "StudyResponse",
    "StudySummary",
    "StudyThresholds",
    "TrialAssignment",
    "UsabilityClaimStatus",
    "evaluate_study",
    "expected_action",
    "randomize_participant_schedule",
    "verify_participant_schedule",
]
