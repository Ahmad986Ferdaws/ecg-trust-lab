"""Overlap-aware comparison plans for external foundation representations."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

from ecg_trust.foundation.contracts import (
    EXTERNAL_ONLY_LIMIT,
    RESEARCH_USE_LIMIT,
    EvaluationMode,
    FoundationError,
    FoundationModelSpec,
    OverlapStatus,
    TrainabilityPolicy,
    strict_identifier,
)

_PRIVATE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,199}$")


class ComparisonRole(StrEnum):
    """Predeclared role of a local patient/site cohort."""

    DEVELOPMENT = "development"
    CALIBRATION = "calibration"
    PREVIOUSLY_OBSERVED = "previously_observed"
    UNTOUCHED_LOCKBOX = "untouched_lockbox"


class IndependenceAssessment(StrEnum):
    """Evidence-supported independence level after overlap review."""

    NON_INDEPENDENT_KNOWN_OVERLAP = "non_independent_known_overlap"
    LIMITED_POSSIBLE_OVERLAP = "limited_possible_overlap"
    LIMITED_UNDISCLOSED_OVERLAP = "limited_undisclosed_overlap"
    NO_DISCLOSED_OVERLAP = "no_disclosed_overlap_not_proof_of_independence"


@dataclass(frozen=True, slots=True, init=False)
class ComparisonCohortMember:
    """Private patient/site assignment used only to prove role isolation."""

    dataset_name: str
    dataset_version: str
    site_key: str
    patient_key: str
    role: ComparisonRole

    @classmethod
    def create(
        cls,
        *,
        dataset_name: str,
        dataset_version: str,
        site_key: str,
        patient_key: str,
        role: ComparisonRole | str,
    ) -> ComparisonCohortMember:
        try:
            parsed_role = ComparisonRole(role)
        except (TypeError, ValueError) as exc:
            raise FoundationError("comparison role is invalid") from exc
        if not isinstance(site_key, str) or _PRIVATE_IDENTIFIER.fullmatch(site_key) is None:
            raise FoundationError("site_key must be a safe private identifier")
        if not isinstance(patient_key, str) or _PRIVATE_IDENTIFIER.fullmatch(patient_key) is None:
            raise FoundationError("patient_key must be a safe private identifier")
        instance = object.__new__(cls)
        object.__setattr__(
            instance, "dataset_name", strict_identifier(dataset_name, "dataset_name")
        )
        object.__setattr__(
            instance,
            "dataset_version",
            strict_identifier(dataset_version, "dataset_version"),
        )
        object.__setattr__(instance, "site_key", site_key)
        object.__setattr__(instance, "patient_key", patient_key)
        object.__setattr__(instance, "role", parsed_role)
        return instance

    @property
    def patient_identity(self) -> tuple[str, str, str]:
        return (self.dataset_name, self.dataset_version, self.patient_key)

    @property
    def site_identity(self) -> tuple[str, str, str]:
        return (self.dataset_name, self.dataset_version, self.site_key)

    @property
    def member_identity(self) -> tuple[str, str, str, str]:
        return (
            self.dataset_name,
            self.dataset_version,
            self.site_key,
            self.patient_key,
        )


@dataclass(frozen=True, slots=True)
class FoundationComparisonPlan:
    """Aggregate overlap decision; contains no patient or site identifiers."""

    model_spec_sha256: str
    evaluation_mode: EvaluationMode
    independence_assessment: IndependenceAssessment
    independence_claim: str
    overlap_reason_codes: tuple[str, ...]
    local_dataset_count: int
    patient_count: int
    site_count: int
    role_counts: tuple[tuple[str, int], ...]
    allowed_trainable_parameter_names: tuple[str, ...]

    def to_public_dict(self) -> dict[str, object]:
        return {
            "model_spec_sha256": self.model_spec_sha256,
            "evaluation_mode": self.evaluation_mode.value,
            "independence_assessment": self.independence_assessment.value,
            "independence_claim": self.independence_claim,
            "overlap_reason_codes": list(self.overlap_reason_codes),
            "local_dataset_count": self.local_dataset_count,
            "patient_count": self.patient_count,
            "site_count": self.site_count,
            "role_counts": dict(self.role_counts),
            "allowed_trainable_parameter_names": list(self.allowed_trainable_parameter_names),
            "local_pretraining_performed": False,
            "scope_limit": EXTERNAL_ONLY_LIMIT,
            "research_use_limit": RESEARCH_USE_LIMIT,
            "privacy_contract": "aggregate_only_no_patient_or_site_identifiers",
        }


def assert_comparison_role_isolation(members: Iterable[ComparisonCohortMember]) -> None:
    """Require every scoped patient and source site to have exactly one role."""

    patient_roles: dict[tuple[str, str, str], ComparisonRole] = {}
    site_roles: dict[tuple[str, str, str], ComparisonRole] = {}
    for member in members:
        prior_patient_role = patient_roles.setdefault(member.patient_identity, member.role)
        if prior_patient_role is not member.role:
            raise FoundationError("a patient appears in multiple comparison roles")
        prior_site_role = site_roles.setdefault(member.site_identity, member.role)
        if prior_site_role is not member.role:
            raise FoundationError("a site appears in multiple comparison roles")


def validate_foundation_comparison_plan(
    spec: FoundationModelSpec,
    *,
    trainability_policy: TrainabilityPolicy,
    cohort_members: Iterable[ComparisonCohortMember],
    local_pretraining_claimed: bool = False,
) -> FoundationComparisonPlan:
    """Validate role isolation and downgrade unsupported independence claims.

    This function cannot establish absence of patient overlap. Even exhaustive
    ``none_known`` disclosures therefore produce a qualified, not absolute,
    independence statement.
    """

    if not isinstance(spec, FoundationModelSpec):
        raise TypeError("spec must be a FoundationModelSpec")
    if not isinstance(trainability_policy, TrainabilityPolicy):
        raise TypeError("trainability_policy must be a TrainabilityPolicy")
    if not isinstance(local_pretraining_claimed, bool):
        raise FoundationError("local_pretraining_claimed must be boolean")
    if local_pretraining_claimed:
        raise FoundationError("local foundation-model pretraining claims are prohibited")
    members = tuple(cohort_members)
    if not members:
        raise FoundationError("comparison plan requires cohort members")
    if any(not isinstance(item, ComparisonCohortMember) for item in members):
        raise FoundationError("cohort_members contains an invalid value")
    member_ids = [item.member_identity for item in members]
    if len(set(member_ids)) != len(member_ids):
        raise FoundationError("comparison cohort contains duplicate patient/site rows")
    assert_comparison_role_isolation(members)

    local_datasets = sorted({(item.dataset_name, item.dataset_version) for item in members})
    pretraining_datasets = [item.dataset_identity for item in spec.pretraining_datasets]
    disclosures = {item.disclosure_key: item.status for item in spec.known_overlaps}
    confirmed = False
    possible = False
    undisclosed = False
    reasons: set[str] = set()
    for local_name, local_version in local_datasets:
        for pretraining_name, pretraining_version in pretraining_datasets:
            key = (local_name, local_version, pretraining_name, pretraining_version)
            disclosed = disclosures.get(key)
            same_family = _dataset_token(local_name) == _dataset_token(pretraining_name)
            if same_family and local_version == pretraining_version:
                confirmed = True
                reasons.add("exact_dataset_version_in_pretraining")
                if disclosed is OverlapStatus.NONE_KNOWN:
                    reasons.add("none_known_disclosure_overridden_by_exact_dataset_match")
            elif disclosed is OverlapStatus.CONFIRMED:
                confirmed = True
                reasons.add("provider_disclosed_confirmed_overlap")
            elif same_family:
                possible = True
                reasons.add("dataset_family_overlap_version_differs")
            elif disclosed is OverlapStatus.POSSIBLE:
                possible = True
                reasons.add("provider_disclosed_possible_overlap")
            elif disclosed is None:
                undisclosed = True
                reasons.add("dataset_pair_has_no_overlap_disclosure")

    if confirmed:
        assessment = IndependenceAssessment.NON_INDEPENDENT_KNOWN_OVERLAP
        claim = "non_independent_external_representation_comparison"
    elif possible:
        assessment = IndependenceAssessment.LIMITED_POSSIBLE_OVERLAP
        claim = "independence_not_established_possible_pretraining_overlap"
    elif undisclosed:
        assessment = IndependenceAssessment.LIMITED_UNDISCLOSED_OVERLAP
        claim = "independence_not_established_incomplete_overlap_disclosure"
    else:
        assessment = IndependenceAssessment.NO_DISCLOSED_OVERLAP
        claim = "no_disclosed_overlap_not_proof_of_independence"
        reasons.add("all_dataset_pairs_disclosed_none_known")

    roles = Counter(item.role.value for item in members)
    return FoundationComparisonPlan(
        model_spec_sha256=spec.spec_sha256,
        evaluation_mode=trainability_policy.mode,
        independence_assessment=assessment,
        independence_claim=claim,
        overlap_reason_codes=tuple(sorted(reasons)),
        local_dataset_count=len(local_datasets),
        patient_count=len({item.patient_identity for item in members}),
        site_count=len({item.site_identity for item in members}),
        role_counts=tuple(sorted(roles.items())),
        allowed_trainable_parameter_names=trainability_policy.allowed_trainable_parameter_names,
    )


def _dataset_token(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())
