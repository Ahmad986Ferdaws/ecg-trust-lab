from __future__ import annotations

import json
from collections.abc import Iterable

import pytest
from pydantic import ValidationError

from ecg_trust.contracts.models import TrustDecision
from ecg_trust.human_factors import (
    AGGREGATE_ONLY_LIMIT,
    NON_CLINICAL_LIMIT,
    RESEARCH_USE_LIMIT,
    USABILITY_CLAIM_LIMIT,
    CaseSource,
    EvidenceStatus,
    ParticipantAction,
    ParticipantSchedule,
    StudyArm,
    StudyCase,
    StudyDataError,
    StudyIntegrityError,
    StudyPreregistration,
    StudyResponse,
    StudyThresholds,
    UsabilityClaimStatus,
    evaluate_study,
    expected_action,
    randomize_participant_schedule,
    verify_participant_schedule,
)

SECRET = b"human-factors-randomization-key-v1"
OTHER_SECRET = b"different-human-factors-key-v1!!"


def _preregistration(*, minimum_participants: int = 2) -> StudyPreregistration:
    cases = tuple(
        StudyCase(
            scenario_id=f"scenario_{index:02d}",
            source=CaseSource.SYNTHETIC if index % 2 else CaseSource.APPROVED_EXAMPLE,
            sentinel_decision=decision,
        )
        for index, decision in enumerate(TrustDecision, start=1)
    )
    return StudyPreregistration(
        study_id="sentinel_interface_crossover_v1",
        cases=cases,
        thresholds=StudyThresholds(
            minimum_paired_participants=minimum_participants,
            minimum_trials_per_arm_per_participant=len(cases),
            minimum_action_accuracy_difference=0.1,
            maximum_overreliance_rate_difference=-0.1,
            minimum_sentinel_comprehension_accuracy=0.8,
        ),
    )


def _schedule(participant: str, *, secret: bytes = SECRET) -> ParticipantSchedule:
    return randomize_participant_schedule(
        _preregistration(),
        participant_private_key=participant,
        randomization_secret=secret,
    )


def _responses(
    schedule: ParticipantSchedule,
    *,
    sentinel_correct: bool = True,
    baseline_always_uses_output: bool = True,
) -> list[StudyResponse]:
    responses: list[StudyResponse] = []
    for assignment in schedule.assignments:
        if assignment.arm is StudyArm.TRUST_SENTINEL:
            action = (
                expected_action(assignment.underlying_decision)
                if sentinel_correct
                else ParticipantAction.USE_MODEL_OUTPUT
            )
            interpreted = assignment.underlying_decision
            confidence = 100
            decision_time = 1_000
        else:
            action = (
                ParticipantAction.USE_MODEL_OUTPUT
                if baseline_always_uses_output
                else expected_action(assignment.underlying_decision)
            )
            interpreted = None
            confidence = 80
            decision_time = 1_500
        responses.append(
            StudyResponse(
                participant_token=schedule.participant_token,
                trial_token=assignment.trial_token,
                arm=assignment.arm,
                selected_action=action,
                interpreted_decision=interpreted,
                confidence_percent=confidence,
                decision_time_ms=decision_time,
            )
        )
    return responses


def _flatten(rows: Iterable[Iterable[StudyResponse]]) -> list[StudyResponse]:
    return [response for participant_rows in rows for response in participant_rows]


def test_preregistration_is_self_hashed_strict_and_tamper_evident() -> None:
    preregistration = _preregistration()
    artifact = preregistration.to_dict()

    assert artifact["preregistration_sha256"] == preregistration.preregistration_sha256
    assert StudyPreregistration.from_dict(artifact) == preregistration
    assert artifact["randomization_secret_in_artifact"] is False
    assert "secret" not in json.dumps(artifact).lower().replace(
        '"randomization_secret_in_artifact"', ""
    )

    tampered = dict(artifact)
    thresholds_payload = tampered["thresholds"]
    assert isinstance(thresholds_payload, dict)
    thresholds = dict(thresholds_payload)
    thresholds["minimum_paired_participants"] = 3
    tampered["thresholds"] = thresholds
    with pytest.raises(StudyIntegrityError, match="SHA-256 mismatch"):
        StudyPreregistration.from_dict(tampered)

    injected = dict(artifact)
    injected["randomization_secret"] = "forbidden"
    with pytest.raises(StudyIntegrityError, match="closed schema"):
        StudyPreregistration.from_dict(injected)


def test_preregistration_requires_safe_cases_and_canonical_design() -> None:
    preregistration = _preregistration()
    with pytest.raises(ValidationError, match="scenario_id"):
        StudyCase(
            scenario_id="patient_123",
            source=CaseSource.SYNTHETIC,
            sentinel_decision=TrustDecision.ABSTAIN,
        )
    with pytest.raises(ValidationError, match="canonical two-arm order"):
        StudyPreregistration(
            study_id="bad_order",
            arms=(StudyArm.PROBABILITIES_ONLY, StudyArm.TRUST_SENTINEL),
            cases=preregistration.cases,
            thresholds=preregistration.thresholds,
        )
    with pytest.raises(ValidationError, match="sorted"):
        StudyPreregistration(
            study_id="bad_cases",
            cases=tuple(reversed(preregistration.cases)),
            thresholds=preregistration.thresholds,
        )


def test_randomization_is_deterministic_domain_separated_and_secret_free() -> None:
    preregistration = _preregistration()
    first = randomize_participant_schedule(
        preregistration,
        participant_private_key="private-participant-A",
        randomization_secret=SECRET,
    )
    repeated = randomize_participant_schedule(
        preregistration,
        participant_private_key="private-participant-A",
        randomization_secret=SECRET,
    )
    other_key = randomize_participant_schedule(
        preregistration,
        participant_private_key="private-participant-A",
        randomization_secret=OTHER_SECRET,
    )

    assert first == repeated
    assert first.participant_token != other_key.participant_token
    assert {item.trial_token for item in first.assignments}.isdisjoint(
        item.trial_token for item in other_key.assignments
    )
    assert len(first.assignments) == 2 * len(preregistration.cases)
    assert {assignment.arm for assignment in first.assignments} == {
        StudyArm.TRUST_SENTINEL,
        StudyArm.PROBABILITIES_ONLY,
    }
    serialized = json.dumps(first.model_dump(mode="json"), sort_keys=True)
    assert "private-participant-A" not in serialized
    assert SECRET.hex() not in serialized
    verify_participant_schedule(first, preregistration, randomization_secret=SECRET)
    with pytest.raises(StudyIntegrityError, match="HMAC"):
        verify_participant_schedule(first, preregistration, randomization_secret=OTHER_SECRET)


def test_arm_order_is_counterbalanced_across_pseudonymous_participants() -> None:
    preregistration = _preregistration()
    first_arms = {
        randomize_participant_schedule(
            preregistration,
            participant_private_key=f"private-participant-{index}",
            randomization_secret=SECRET,
        ).arm_order[0]
        for index in range(32)
    }
    assert first_arms == {StudyArm.TRUST_SENTINEL, StudyArm.PROBABILITIES_ONLY}


def test_schedule_authentication_detects_backend_assignment_tampering() -> None:
    preregistration = _preregistration()
    schedule = _schedule("participant-A")
    tampered = schedule.model_copy(update={"schedule_mac": "hmac-sha256:" + "0" * 64})

    with pytest.raises(StudyIntegrityError, match="HMAC"):
        verify_participant_schedule(tampered, preregistration, randomization_secret=SECRET)


def test_response_schema_structurally_excludes_free_text_identifiers_and_waveforms() -> None:
    assignment = _schedule("participant-A").assignments[0]
    base: dict[str, object] = {
        "participant_token": _schedule("participant-A").participant_token,
        "trial_token": assignment.trial_token,
        "arm": assignment.arm,
        "selected_action": expected_action(assignment.underlying_decision),
        "interpreted_decision": (
            assignment.underlying_decision if assignment.arm is StudyArm.TRUST_SENTINEL else None
        ),
        "confidence_percent": 80,
        "decision_time_ms": 1_000,
    }
    for forbidden_field, value in (
        ("free_text", "looks dangerous"),
        ("participant_name", "Alice Example"),
        ("patient_id", "patient-7"),
        ("waveform", [[0.1, 0.2]]),
    ):
        with pytest.raises(ValidationError, match="Extra inputs"):
            StudyResponse.model_validate({**base, forbidden_field: value})

    assert set(StudyResponse.model_fields) == {
        "participant_token",
        "trial_token",
        "arm",
        "selected_action",
        "interpreted_decision",
        "confidence_percent",
        "decision_time_ms",
    }


def test_response_rejects_malformed_arm_semantics_types_and_ranges() -> None:
    schedule = _schedule("participant-A")
    sentinel = next(item for item in schedule.assignments if item.arm is StudyArm.TRUST_SENTINEL)
    baseline = next(
        item for item in schedule.assignments if item.arm is StudyArm.PROBABILITIES_ONLY
    )

    with pytest.raises(ValidationError, match="requires"):
        StudyResponse(
            participant_token=schedule.participant_token,
            trial_token=sentinel.trial_token,
            arm=StudyArm.TRUST_SENTINEL,
            selected_action=ParticipantAction.DO_NOT_USE,
            interpreted_decision=None,
            confidence_percent=50,
            decision_time_ms=500,
        )
    with pytest.raises(ValidationError, match="cannot report"):
        StudyResponse(
            participant_token=schedule.participant_token,
            trial_token=baseline.trial_token,
            arm=StudyArm.PROBABILITIES_ONLY,
            selected_action=ParticipantAction.DO_NOT_USE,
            interpreted_decision=TrustDecision.ABSTAIN,
            confidence_percent=50,
            decision_time_ms=500,
        )
    with pytest.raises(ValidationError):
        StudyResponse(
            participant_token=schedule.participant_token,
            trial_token=sentinel.trial_token,
            arm=StudyArm.TRUST_SENTINEL,
            selected_action=ParticipantAction.DO_NOT_USE,
            interpreted_decision=sentinel.underlying_decision,
            confidence_percent=True,
            decision_time_ms=0,
        )


def test_duplicate_unknown_and_arm_mismatched_responses_fail_closed() -> None:
    preregistration = _preregistration()
    schedule = _schedule("participant-A")
    responses = _responses(schedule)

    with pytest.raises(StudyDataError, match="duplicate response"):
        evaluate_study(
            preregistration,
            [schedule],
            [*responses, responses[0]],
            randomization_secret=SECRET,
        )

    foreign = _responses(_schedule("participant-B"))[0]
    with pytest.raises(StudyDataError, match="unknown participant"):
        evaluate_study(
            preregistration,
            [schedule],
            [foreign],
            randomization_secret=SECRET,
        )

    first = responses[0]
    wrong_arm = first.model_copy(
        update={
            "arm": (
                StudyArm.PROBABILITIES_ONLY
                if first.arm is StudyArm.TRUST_SENTINEL
                else StudyArm.TRUST_SENTINEL
            )
        }
    )
    with pytest.raises(StudyDataError, match="arm"):
        evaluate_study(
            preregistration,
            [schedule],
            [wrong_arm],
            randomization_secret=SECRET,
        )


def test_powered_aggregate_metrics_and_paired_differences_match_known_outcomes() -> None:
    preregistration = _preregistration()
    schedules = [_schedule("participant-A"), _schedule("participant-B")]
    responses = _flatten(_responses(schedule) for schedule in schedules)

    summary = evaluate_study(
        preregistration,
        schedules,
        responses,
        randomization_secret=SECRET,
    )

    assert summary.evidence_status is EvidenceStatus.MINIMUM_EVIDENCE_MET
    assert summary.paired_complete_participant_count == 2
    sentinel, baseline = summary.arm_aggregates
    assert sentinel.action_accuracy == pytest.approx(1.0)
    assert sentinel.comprehension_accuracy == pytest.approx(1.0)
    assert sentinel.overreliance_rate == pytest.approx(0.0)
    assert sentinel.mean_decision_time_ms == pytest.approx(1_000.0)
    assert sentinel.confidence_expected_calibration_error == pytest.approx(0.0)
    assert sentinel.confidence_brier_score == pytest.approx(0.0)
    assert baseline.action_accuracy == pytest.approx(0.2)
    assert baseline.comprehension_accuracy is None
    assert baseline.overreliance_rate == pytest.approx(1.0)
    assert baseline.mean_decision_time_ms == pytest.approx(1_500.0)
    assert baseline.confidence_expected_calibration_error == pytest.approx(0.6)
    assert baseline.confidence_brier_score == pytest.approx(0.52)
    paired = summary.paired_aggregate
    assert paired.action_accuracy_difference == pytest.approx(0.8)
    assert paired.overreliance_rate_difference == pytest.approx(-1.0)
    assert paired.mean_decision_time_ms_difference == pytest.approx(-500.0)
    assert paired.confidence_expected_calibration_error_difference == pytest.approx(-0.6)
    assert (
        summary.usability_claim_status
        is UsabilityClaimStatus.THRESHOLDS_MET_CONFIRMATORY_REVIEW_REQUIRED
    )
    assert summary.standalone_usability_claim_authorized is False


def test_incomplete_and_too_small_study_is_underpowered_and_suppresses_metrics() -> None:
    preregistration = _preregistration()
    complete = _schedule("participant-A")
    incomplete = _schedule("participant-B")
    responses = [*_responses(complete), _responses(incomplete)[0]]

    summary = evaluate_study(
        preregistration,
        [complete, incomplete],
        responses,
        randomization_secret=SECRET,
    )

    assert summary.evidence_status is EvidenceStatus.UNDERPOWERED
    assert summary.usability_claim_status is UsabilityClaimStatus.PROHIBITED_UNDERPOWERED
    assert summary.paired_complete_participant_count == 1
    assert summary.excluded_incomplete_participant_count == 1
    assert summary.paired_aggregate.action_accuracy_difference is None
    assert all(item.action_accuracy is None for item in summary.arm_aggregates)
    assert all(item.comprehension_accuracy is None for item in summary.arm_aggregates)


def test_threshold_failure_cannot_be_reported_as_a_usability_success() -> None:
    preregistration = _preregistration()
    schedules = [_schedule("participant-A"), _schedule("participant-B")]
    responses = _flatten(
        _responses(schedule, baseline_always_uses_output=False) for schedule in schedules
    )

    summary = evaluate_study(
        preregistration,
        schedules,
        responses,
        randomization_secret=SECRET,
    )

    assert summary.evidence_status is EvidenceStatus.MINIMUM_EVIDENCE_MET
    assert summary.usability_claim_status is UsabilityClaimStatus.PROHIBITED_THRESHOLDS_NOT_MET
    assert summary.standalone_usability_claim_authorized is False


def test_public_summary_is_aggregate_only_and_carries_all_research_boundaries() -> None:
    preregistration = _preregistration()
    schedules = [_schedule("private-A"), _schedule("private-B")]
    responses = _flatten(_responses(schedule) for schedule in schedules)
    summary = evaluate_study(
        preregistration,
        schedules,
        responses,
        randomization_secret=SECRET,
    )

    public = summary.to_public_dict()
    serialized = json.dumps(public, sort_keys=True)
    assert public["privacy_contract"] == AGGREGATE_ONLY_LIMIT
    assert public["research_use_limit"] == RESEARCH_USE_LIMIT
    assert public["non_clinical_limit"] == NON_CLINICAL_LIMIT
    assert public["usability_claim_limit"] == USABILITY_CLAIM_LIMIT
    assert public["standalone_usability_claim_authorized"] is False
    assert "participant_token" not in serialized
    assert "trial_token" not in serialized
    assert "case_token" not in serialized
    assert all(case.scenario_id not in serialized for case in preregistration.cases)
    assert all(schedule.participant_token not in serialized for schedule in schedules)


def test_contracts_are_immutable_and_use_the_canonical_five_decision_states() -> None:
    preregistration = _preregistration()
    assert {case.sentinel_decision for case in preregistration.cases} == set(TrustDecision)
    assert len(TrustDecision) == 5
    with pytest.raises(ValidationError, match="frozen"):
        preregistration.study_id = "changed"
