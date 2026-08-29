"""Leakage-safe chronological cohort and future-event target construction."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import timedelta

from ecg_trust.longitudinal.contracts import (
    AGGREGATE_ONLY_LIMIT,
    NON_CAUSAL_TARGET_LIMIT,
    RESEARCH_USE_LIMIT,
    BinaryFutureTarget,
    CensoringPolicy,
    ECGEncounter,
    FollowUpStatus,
    FutureEventDefinition,
    LongitudinalError,
    RoleIsolationError,
    SourceRole,
    TargetStatus,
    TemporalLeakageError,
    TimelineConfig,
    format_utc,
    parse_utc,
)


@dataclass(frozen=True, slots=True)
class PatientTimeline:
    """Private, chronologically ordered encounters for one scoped patient."""

    patient_identity: tuple[str, str, str]
    source_identities: tuple[tuple[str, str, str], ...]
    source_role: SourceRole
    index_time_utc: str
    horizon_end_utc: str
    observation_end_utc: str
    history_encounters: tuple[ECGEncounter, ...]
    future_encounters: tuple[ECGEncounter, ...]
    follow_up_status: FollowUpStatus

    @property
    def assignment_time_utc(self) -> str:
        """Last available input encounter time, used for temporal partitioning."""

        return self.history_encounters[-1].occurred_at_utc


@dataclass(frozen=True, slots=True)
class LongitudinalCohortSummary:
    """Aggregate-only construction counts; contains no linkage identifiers."""

    input_patient_count: int
    included_patient_count: int
    excluded_no_history_count: int
    excluded_by_censoring_count: int
    complete_horizon_count: int
    right_censored_count: int
    insufficient_follow_up_count: int

    def to_public_dict(self) -> dict[str, object]:
        return {
            "input_patient_count": self.input_patient_count,
            "included_patient_count": self.included_patient_count,
            "excluded_no_history_count": self.excluded_no_history_count,
            "excluded_by_censoring_count": self.excluded_by_censoring_count,
            "complete_horizon_count": self.complete_horizon_count,
            "right_censored_count": self.right_censored_count,
            "insufficient_follow_up_count": self.insufficient_follow_up_count,
            "privacy_contract": AGGREGATE_ONLY_LIMIT,
            "research_use_limit": RESEARCH_USE_LIMIT,
        }


@dataclass(frozen=True, slots=True)
class LongitudinalCohort:
    """Private timelines paired with their aggregate public summary."""

    config: TimelineConfig
    timelines: tuple[PatientTimeline, ...]
    summary: LongitudinalCohortSummary


@dataclass(frozen=True, slots=True)
class PatientFutureTargets:
    """Private linkage identity plus binary or multilabel target observations."""

    patient_identity: tuple[str, str, str]
    targets: tuple[BinaryFutureTarget, ...]
    interpretation: str = NON_CAUSAL_TARGET_LIMIT


def assert_patient_source_role_isolation(encounters: Iterable[ECGEncounter]) -> None:
    """Require every patient and source scope to occur in exactly one source role."""

    patient_roles: dict[tuple[str, str, str], SourceRole] = {}
    source_roles: dict[tuple[str, str, str], SourceRole] = {}
    for encounter in encounters:
        prior_patient_role = patient_roles.setdefault(
            encounter.patient_identity,
            encounter.source_role,
        )
        if prior_patient_role is not encounter.source_role:
            raise RoleIsolationError("a patient appears in multiple source roles")
        prior_source_role = source_roles.setdefault(
            encounter.source_identity,
            encounter.source_role,
        )
        if prior_source_role is not encounter.source_role:
            raise RoleIsolationError("a source appears in multiple source roles")


def assert_no_temporal_leakage(timeline: PatientTimeline) -> None:
    """Prove inputs end at index and targets begin strictly after index."""

    if not timeline.history_encounters:
        raise TemporalLeakageError("timeline has no history encounters")
    index = parse_utc(timeline.index_time_utc)
    horizon_end = parse_utc(timeline.horizon_end_utc)
    history_ids = {item.encounter_identity for item in timeline.history_encounters}
    future_ids = {item.encounter_identity for item in timeline.future_encounters}
    if history_ids & future_ids:
        raise TemporalLeakageError("an encounter appears in both history and future")
    if any(parse_utc(item.occurred_at_utc) > index for item in timeline.history_encounters):
        raise TemporalLeakageError("a future encounter appears in model history")
    if any(
        not index < parse_utc(item.occurred_at_utc) <= horizon_end
        for item in timeline.future_encounters
    ):
        raise TemporalLeakageError("a same-time or out-of-horizon encounter appears in targets")
    history_order = tuple(item.occurred_at_utc for item in timeline.history_encounters)
    future_order = tuple(item.occurred_at_utc for item in timeline.future_encounters)
    if history_order != tuple(sorted(history_order)) or future_order != tuple(sorted(future_order)):
        raise TemporalLeakageError("timeline encounters are not chronological")


def build_patient_timelines(
    encounters: Iterable[ECGEncounter],
    config: TimelineConfig,
) -> LongitudinalCohort:
    """Build patient timelines at one predeclared fixed UTC index time.

    Input encounters occupy ``[index-history, index]``. Target encounters occupy
    ``(index, index+horizon]``. Consequently an ECG at the index can be an input
    but can never also define its own future-event target.
    """

    if not isinstance(config, TimelineConfig):
        raise TypeError("config must be a TimelineConfig")
    materialized = tuple(encounters)
    if not materialized:
        raise LongitudinalError("at least one ECG encounter is required")
    if any(not isinstance(item, ECGEncounter) for item in materialized):
        raise LongitudinalError("encounters must contain only ECGEncounter values")
    identities = [item.encounter_identity for item in materialized]
    if len(set(identities)) != len(identities):
        raise LongitudinalError("duplicate encounter identity")
    assert_patient_source_role_isolation(materialized)

    grouped: dict[tuple[str, str, str], list[ECGEncounter]] = defaultdict(list)
    for encounter in materialized:
        grouped[encounter.patient_identity].append(encounter)

    index = parse_utc(config.index_time_utc)
    history_start = index - timedelta(days=config.history_window_days)
    horizon_end = index + timedelta(days=config.prediction_horizon_days)
    timelines: list[PatientTimeline] = []
    excluded_no_history = 0
    excluded_censoring = 0

    for patient_identity in sorted(grouped):
        patient_encounters = tuple(
            sorted(
                grouped[patient_identity],
                key=lambda item: (item.occurred_at_utc, item.encounter_identity),
            )
        )
        observation_ends = {item.observed_through_utc for item in patient_encounters}
        if len(observation_ends) != 1:
            raise LongitudinalError("patient encounters disagree on observation end")
        roles = {item.source_role for item in patient_encounters}
        if len(roles) != 1:
            raise RoleIsolationError("a patient appears in multiple source roles")
        observation_end_utc = next(iter(observation_ends))
        observation_end = parse_utc(observation_end_utc)
        latest_encounter = parse_utc(patient_encounters[-1].occurred_at_utc)
        if observation_end < latest_encounter:
            raise LongitudinalError("patient observation end precedes a recorded encounter")

        history = tuple(
            item
            for item in patient_encounters
            if history_start <= parse_utc(item.occurred_at_utc) <= index
        )
        if not history:
            excluded_no_history += 1
            continue
        future = tuple(
            item
            for item in patient_encounters
            if index < parse_utc(item.occurred_at_utc) <= horizon_end
        )
        available_seconds = max(0.0, (observation_end - index).total_seconds())
        minimum_seconds = float(config.minimum_follow_up_days * 86_400)
        if observation_end >= horizon_end:
            status = FollowUpStatus.COMPLETE_HORIZON
        elif available_seconds < minimum_seconds:
            status = FollowUpStatus.INSUFFICIENT_FOLLOW_UP
        else:
            status = FollowUpStatus.RIGHT_CENSORED

        should_exclude = (
            config.censoring_policy is CensoringPolicy.EXCLUDE_BELOW_MINIMUM
            and status is FollowUpStatus.INSUFFICIENT_FOLLOW_UP
        ) or (
            config.censoring_policy is CensoringPolicy.REQUIRE_COMPLETE_HORIZON
            and status is not FollowUpStatus.COMPLETE_HORIZON
        )
        if should_exclude:
            excluded_censoring += 1
            continue

        sources = tuple(sorted({item.source_identity for item in patient_encounters}))
        timeline = PatientTimeline(
            patient_identity=patient_identity,
            source_identities=sources,
            source_role=next(iter(roles)),
            index_time_utc=config.index_time_utc,
            horizon_end_utc=format_utc(horizon_end),
            observation_end_utc=observation_end_utc,
            history_encounters=history,
            future_encounters=future,
            follow_up_status=status,
        )
        assert_no_temporal_leakage(timeline)
        timelines.append(timeline)

    canonical = tuple(sorted(timelines, key=lambda item: item.patient_identity))
    counts = {status: 0 for status in FollowUpStatus}
    for timeline in canonical:
        counts[timeline.follow_up_status] += 1
    summary = LongitudinalCohortSummary(
        input_patient_count=len(grouped),
        included_patient_count=len(canonical),
        excluded_no_history_count=excluded_no_history,
        excluded_by_censoring_count=excluded_censoring,
        complete_horizon_count=counts[FollowUpStatus.COMPLETE_HORIZON],
        right_censored_count=counts[FollowUpStatus.RIGHT_CENSORED],
        insufficient_follow_up_count=counts[FollowUpStatus.INSUFFICIENT_FOLLOW_UP],
    )
    return LongitudinalCohort(config=config, timelines=canonical, summary=summary)


def derive_future_event_targets(
    cohort: LongitudinalCohort,
    definitions: Sequence[FutureEventDefinition],
) -> tuple[PatientFutureTargets, ...]:
    """Derive observed or explicitly censored binary/multilabel targets.

    These labels encode later documented events within a fixed retrospective
    window. They do not represent an intervention effect or causal estimate.
    """

    if not isinstance(cohort, LongitudinalCohort):
        raise TypeError("cohort must be a LongitudinalCohort")
    if isinstance(definitions, (str, bytes)) or not definitions:
        raise LongitudinalError("at least one future-event definition is required")
    if any(not isinstance(item, FutureEventDefinition) for item in definitions):
        raise LongitudinalError("definitions must contain FutureEventDefinition values")
    names = [item.target_name for item in definitions]
    if len(set(names)) != len(names):
        raise LongitudinalError("future-event target names must be unique")
    ordered_definitions = tuple(sorted(definitions, key=lambda item: item.target_name))

    output: list[PatientFutureTargets] = []
    for timeline in cohort.timelines:
        assert_no_temporal_leakage(timeline)
        future_labels = {
            label for encounter in timeline.future_encounters for label in encounter.event_labels
        }
        target_values: list[BinaryFutureTarget] = []
        for definition in ordered_definitions:
            event_observed = any(label in future_labels for label in definition.event_any_of)
            if event_observed:
                value: int | None = 1
                status = TargetStatus.OBSERVED
            elif timeline.follow_up_status is FollowUpStatus.COMPLETE_HORIZON:
                value = 0
                status = TargetStatus.OBSERVED
            elif timeline.follow_up_status is FollowUpStatus.RIGHT_CENSORED:
                value = None
                status = TargetStatus.RIGHT_CENSORED
            else:
                value = None
                status = TargetStatus.INSUFFICIENT_FOLLOW_UP
            target_values.append(
                BinaryFutureTarget(
                    target_name=definition.target_name,
                    value=value,
                    status=status,
                )
            )
        output.append(
            PatientFutureTargets(
                patient_identity=timeline.patient_identity,
                targets=tuple(target_values),
            )
        )
    return tuple(output)
