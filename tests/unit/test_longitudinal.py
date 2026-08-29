from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace

import pytest

from ecg_trust.longitudinal import (
    AGGREGATE_ONLY_LIMIT,
    NON_CAUSAL_TARGET_LIMIT,
    RESEARCH_USE_LIMIT,
    BinaryFutureTarget,
    CalibrationBinStatus,
    CensoringPolicy,
    ECGEncounter,
    EvaluationStatus,
    FollowUpStatus,
    FutureEventDefinition,
    LongitudinalError,
    LongitudinalIntegrityError,
    MetricStatus,
    RiskEvaluationConfig,
    RiskObservation,
    RoleIsolationError,
    SourceRole,
    TargetStatus,
    TemporalLeakageError,
    TemporalPartition,
    TemporalSplitManifest,
    TimelineConfig,
    assert_no_temporal_leakage,
    build_patient_timelines,
    build_temporal_split_manifest,
    derive_future_event_targets,
    evaluate_time_dependent_binary_risk,
)


def _encounter(
    patient: str,
    encounter: str,
    occurred: str,
    *,
    observed_through: str = "2026-05-01T00:00:00Z",
    labels: tuple[str, ...] = (),
    dataset: str = "cohort_a",
    version: str = "1.0",
    site: str = "site_a",
    role: SourceRole = SourceRole.DEVELOPMENT,
) -> ECGEncounter:
    return ECGEncounter.create(
        source_dataset=dataset,
        source_version=version,
        source_site=site,
        source_role=role,
        patient_key=patient,
        encounter_key=encounter,
        occurred_at_utc=occurred,
        observed_through_utc=observed_through,
        event_labels=labels,
    )


def _config(
    policy: CensoringPolicy = CensoringPolicy.RETAIN_WITH_STATUS,
) -> TimelineConfig:
    return TimelineConfig.create(
        index_time_utc="2026-01-01T00:00:00Z",
        history_window_days=365,
        prediction_horizon_days=90,
        minimum_follow_up_days=30,
        censoring_policy=policy,
    )


def _mixed_follow_up_encounters() -> list[ECGEncounter]:
    return [
        _encounter("complete", "old", "2024-12-01T00:00:00Z"),
        _encounter("complete", "history", "2025-10-01T00:00:00Z"),
        _encounter("complete", "index", "2026-01-01T00:00:00Z", labels=("AF",)),
        _encounter("complete", "future", "2026-02-01T00:00:00Z", labels=("MI",)),
        _encounter("complete", "late", "2026-04-15T00:00:00Z", labels=("AF",)),
        _encounter(
            "insufficient",
            "history",
            "2025-11-01T00:00:00Z",
            observed_through="2026-01-20T00:00:00Z",
        ),
        _encounter(
            "insufficient",
            "future",
            "2026-01-10T00:00:00Z",
            observed_through="2026-01-20T00:00:00Z",
            labels=("AF",),
        ),
        _encounter(
            "right_censored",
            "history",
            "2025-12-01T00:00:00Z",
            observed_through="2026-02-15T00:00:00Z",
            labels=("MI",),
        ),
        _encounter("no_history", "future", "2026-01-15T00:00:00Z", labels=("MI",)),
    ]


def test_encounter_is_strict_immutable_and_private_serialization_round_trips() -> None:
    encounter = _encounter(
        "patient_1",
        "record_1",
        "2025-12-01T00:00:00Z",
        labels=("AF", "MI"),
    )

    restored = ECGEncounter.from_private_dict(encounter.to_private_dict())

    assert restored == encounter
    assert restored.patient_identity == ("cohort_a", "1.0", "patient_1")
    assert restored.encounter_identity[-1] == "record_1"
    with pytest.raises(FrozenInstanceError):
        encounter.patient_key = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"patient_key": "patient with spaces"},
        {"occurred_at_utc": "2025-12-01"},
        {
            "occurred_at_utc": "2026-02-01T00:00:00Z",
            "observed_through_utc": "2026-01-01T00:00:00Z",
        },
        {"event_labels": ("MI", "AF")},
        {"event_labels": ("MI", "MI")},
    ],
)
def test_malformed_encounter_metadata_is_rejected(kwargs: dict[str, object]) -> None:
    base: dict[str, object] = {
        "source_dataset": "cohort",
        "source_version": "1",
        "source_site": "site",
        "source_role": SourceRole.DEVELOPMENT,
        "patient_key": "patient",
        "encounter_key": "record",
        "occurred_at_utc": "2025-12-01T00:00:00Z",
        "observed_through_utc": "2026-05-01T00:00:00Z",
        "event_labels": (),
    }
    base.update(kwargs)
    with pytest.raises(LongitudinalError):
        ECGEncounter.create(**base)  # type: ignore[arg-type]


def test_timeline_config_is_versioned_hash_bound_and_rejects_bad_windows() -> None:
    config = _config()
    artifact = config.to_dict()

    assert TimelineConfig.from_dict(artifact) == config
    assert artifact["research_use_limit"] == RESEARCH_USE_LIMIT
    assert artifact["target_interpretation"] == NON_CAUSAL_TARGET_LIMIT
    assert str(artifact["config_sha256"]).startswith("sha256:")

    tampered = dict(artifact)
    tampered["history_window_days"] = 30
    with pytest.raises(LongitudinalIntegrityError, match="SHA-256"):
        TimelineConfig.from_dict(tampered)
    with pytest.raises(LongitudinalError, match="cannot exceed"):
        TimelineConfig.create(
            index_time_utc="2026-01-01T00:00:00Z",
            history_window_days=30,
            prediction_horizon_days=10,
            minimum_follow_up_days=11,
            censoring_policy=CensoringPolicy.RETAIN_WITH_STATUS,
        )


def test_timeline_construction_is_chronological_windowed_and_leakage_safe() -> None:
    cohort = build_patient_timelines(reversed(_mixed_follow_up_encounters()), _config())

    assert tuple(item.patient_identity[-1] for item in cohort.timelines) == (
        "complete",
        "insufficient",
        "right_censored",
    )
    complete = cohort.timelines[0]
    assert tuple(item.encounter_key for item in complete.history_encounters) == (
        "history",
        "index",
    )
    assert tuple(item.encounter_key for item in complete.future_encounters) == ("future",)
    assert complete.follow_up_status is FollowUpStatus.COMPLETE_HORIZON
    assert cohort.timelines[1].follow_up_status is FollowUpStatus.INSUFFICIENT_FOLLOW_UP
    assert cohort.timelines[2].follow_up_status is FollowUpStatus.RIGHT_CENSORED
    for timeline in cohort.timelines:
        assert_no_temporal_leakage(timeline)

    summary = cohort.summary.to_public_dict()
    assert summary == {
        "input_patient_count": 4,
        "included_patient_count": 3,
        "excluded_no_history_count": 1,
        "excluded_by_censoring_count": 0,
        "complete_horizon_count": 1,
        "right_censored_count": 1,
        "insufficient_follow_up_count": 1,
        "privacy_contract": AGGREGATE_ONLY_LIMIT,
        "research_use_limit": RESEARCH_USE_LIMIT,
    }
    serialized_summary = json.dumps(summary)
    assert "patient_key" not in serialized_summary
    assert 'complete"' not in serialized_summary


def test_censoring_policies_exclude_only_the_declared_groups() -> None:
    encounters = _mixed_follow_up_encounters()

    minimum = build_patient_timelines(
        encounters,
        _config(CensoringPolicy.EXCLUDE_BELOW_MINIMUM),
    )
    complete = build_patient_timelines(
        encounters,
        _config(CensoringPolicy.REQUIRE_COMPLETE_HORIZON),
    )

    assert tuple(item.patient_identity[-1] for item in minimum.timelines) == (
        "complete",
        "right_censored",
    )
    assert minimum.summary.excluded_by_censoring_count == 1
    assert tuple(item.patient_identity[-1] for item in complete.timelines) == ("complete",)
    assert complete.summary.excluded_by_censoring_count == 2


def test_binary_and_multilabel_targets_use_only_strictly_future_events() -> None:
    cohort = build_patient_timelines(_mixed_follow_up_encounters(), _config())
    definitions = [
        FutureEventDefinition.create(target_name="future_mi", event_any_of=("MI",)),
        FutureEventDefinition.create(target_name="future_af", event_any_of=("AF",)),
    ]

    targets = derive_future_event_targets(cohort, definitions)
    by_patient = {
        row.patient_identity[-1]: {target.target_name: target for target in row.targets}
        for row in targets
    }

    assert by_patient["complete"]["future_mi"].value == 1
    assert by_patient["complete"]["future_af"].value == 0
    assert by_patient["insufficient"]["future_af"].value == 1
    insufficient_mi = by_patient["insufficient"]["future_mi"]
    assert insufficient_mi.value is None
    assert insufficient_mi.status is TargetStatus.INSUFFICIENT_FOLLOW_UP
    censored = by_patient["right_censored"]
    assert censored["future_mi"].status is TargetStatus.RIGHT_CENSORED
    assert censored["future_af"].status is TargetStatus.RIGHT_CENSORED
    assert all(row.interpretation == NON_CAUSAL_TARGET_LIMIT for row in targets)


def test_duplicate_targets_and_invalid_binary_target_states_are_rejected() -> None:
    cohort = build_patient_timelines(_mixed_follow_up_encounters(), _config())
    definition = FutureEventDefinition.create(target_name="future_mi", event_any_of=("MI",))
    with pytest.raises(LongitudinalError, match="unique"):
        derive_future_event_targets(cohort, [definition, definition])
    with pytest.raises(LongitudinalError, match="value=null"):
        BinaryFutureTarget(
            target_name="future_mi",
            value=0,
            status=TargetStatus.RIGHT_CENSORED,
        )


def test_patient_and_source_role_leakage_and_duplicate_encounters_are_rejected() -> None:
    patient_role_collision = [
        _encounter("same", "a", "2025-10-01T00:00:00Z"),
        _encounter(
            "same",
            "b",
            "2025-11-01T00:00:00Z",
            role=SourceRole.CALIBRATION,
            site="site_b",
        ),
    ]
    with pytest.raises(RoleIsolationError, match="patient"):
        build_patient_timelines(patient_role_collision, _config())

    source_role_collision = [
        _encounter("one", "a", "2025-10-01T00:00:00Z"),
        _encounter(
            "two",
            "b",
            "2025-11-01T00:00:00Z",
            role=SourceRole.CALIBRATION,
        ),
    ]
    with pytest.raises(RoleIsolationError, match="source"):
        build_patient_timelines(source_role_collision, _config())

    duplicate = _encounter("one", "same", "2025-10-01T00:00:00Z")
    with pytest.raises(LongitudinalError, match="duplicate"):
        build_patient_timelines([duplicate, duplicate], _config())


def test_inconsistent_patient_observation_end_is_rejected() -> None:
    encounters = [
        _encounter(
            "one",
            "a",
            "2025-10-01T00:00:00Z",
            observed_through="2026-03-01T00:00:00Z",
        ),
        _encounter(
            "one",
            "b",
            "2025-11-01T00:00:00Z",
            observed_through="2026-04-01T00:00:00Z",
        ),
    ]
    with pytest.raises(LongitudinalError, match="observation end"):
        build_patient_timelines(encounters, _config())


def test_explicit_leakage_validator_rejects_overlap_and_same_time_future() -> None:
    cohort = build_patient_timelines(_mixed_follow_up_encounters(), _config())
    valid = cohort.timelines[0]
    overlapped = replace(valid, future_encounters=(valid.history_encounters[-1],))
    with pytest.raises(TemporalLeakageError):
        assert_no_temporal_leakage(overlapped)

    same_time = _encounter(
        "complete",
        "same_time_target",
        "2026-01-01T00:00:00Z",
    )
    invalid = replace(valid, future_encounters=(same_time,))
    with pytest.raises(TemporalLeakageError, match="same-time"):
        assert_no_temporal_leakage(invalid)


def _split_encounters() -> list[ECGEncounter]:
    times = (
        "2025-02-01T00:00:00Z",
        "2025-03-01T00:00:00Z",
        "2025-06-01T00:00:00Z",
        "2025-07-01T00:00:00Z",
        "2025-10-01T00:00:00Z",
        "2025-12-01T00:00:00Z",
    )
    return [_encounter(f"private_{index}", f"e{index}", time) for index, time in enumerate(times)]


def _build_split(key: bytes = b"k" * 32) -> TemporalSplitManifest:
    cohort = build_patient_timelines(_split_encounters(), _config())
    return build_temporal_split_manifest(
        cohort,
        development_end_utc="2025-04-30T00:00:00Z",
        calibration_end_utc="2025-08-31T00:00:00Z",
        pseudonymization_key=key,
    )


def test_temporal_split_is_patient_disjoint_deterministic_private_and_hash_bound() -> None:
    first = _build_split()
    shuffled_cohort = build_patient_timelines(reversed(_split_encounters()), _config())
    second = build_temporal_split_manifest(
        shuffled_cohort,
        development_end_utc="2025-04-30T00:00:00Z",
        calibration_end_utc="2025-08-31T00:00:00Z",
        pseudonymization_key=b"k" * 32,
    )

    assert first.to_dict() == second.to_dict()
    assert len({item.patient_token for item in first.assignments}) == 6
    assert [item.partition for item in first.assignments] == [
        TemporalPartition.DEVELOPMENT,
        TemporalPartition.DEVELOPMENT,
        TemporalPartition.CALIBRATION,
        TemporalPartition.CALIBRATION,
        TemporalPartition.TEMPORAL_EVALUATION,
        TemporalPartition.TEMPORAL_EVALUATION,
    ]
    serialized = json.dumps(first.to_dict(), sort_keys=True)
    assert "private_" not in serialized
    assert "cohort_a" not in serialized
    assert TemporalSplitManifest.from_dict(first.to_dict()).to_dict() == first.to_dict()
    assert first.manifest_sha256.startswith("sha256:")

    summary = first.public_summary()
    assert summary["development_count"] == 2
    assert summary["calibration_count"] == 2
    assert summary["temporal_evaluation_count"] == 2
    assert summary["privacy_contract"] == AGGREGATE_ONLY_LIMIT
    assert "token" not in json.dumps(summary)


def test_temporal_split_key_changes_pseudonyms_and_integrity_tampering_fails() -> None:
    first = _build_split(b"a" * 32)
    second = _build_split(b"b" * 32)
    assert first.manifest_sha256 != second.manifest_sha256
    assert first.assignments[0].patient_token != second.assignments[0].patient_token

    tampered = first.to_dict()
    rows = [dict(row) for row in tampered["assignments"]]  # type: ignore[union-attr]
    rows[0]["assignment_time_utc"] = "2025-04-01T00:00:00Z"
    tampered["assignments"] = rows
    with pytest.raises(LongitudinalIntegrityError):
        TemporalSplitManifest.from_dict(tampered)


def test_serialized_split_rechecks_patient_and_source_isolation() -> None:
    manifest = _build_split()
    source_collision = manifest.to_dict()
    source_rows = [dict(row) for row in source_collision["assignments"]]  # type: ignore[union-attr]
    source_rows[1]["source_role"] = SourceRole.CALIBRATION.value
    source_collision["assignments"] = source_rows
    with pytest.raises(LongitudinalIntegrityError, match="source scope"):
        TemporalSplitManifest.from_dict(source_collision)

    patient_collision = manifest.to_dict()
    patient_rows = [dict(row) for row in patient_collision["assignments"]]  # type: ignore[union-attr]
    patient_rows[1]["patient_token"] = patient_rows[0]["patient_token"]
    patient_collision["assignments"] = patient_rows
    with pytest.raises(LongitudinalIntegrityError, match="patient token"):
        TemporalSplitManifest.from_dict(patient_collision)


def test_temporal_split_rejects_weak_key_bad_cutoffs_and_empty_partition() -> None:
    cohort = build_patient_timelines(_split_encounters(), _config())
    with pytest.raises(LongitudinalError, match="32 bytes"):
        build_temporal_split_manifest(
            cohort,
            development_end_utc="2025-04-30T00:00:00Z",
            calibration_end_utc="2025-08-31T00:00:00Z",
            pseudonymization_key=b"short",
        )
    with pytest.raises(LongitudinalError, match="must be bytes"):
        build_temporal_split_manifest(
            cohort,
            development_end_utc="2025-04-30T00:00:00Z",
            calibration_end_utc="2025-08-31T00:00:00Z",
            pseudonymization_key="not-bytes",  # type: ignore[arg-type]
        )
    with pytest.raises(LongitudinalError, match="precede"):
        build_temporal_split_manifest(
            cohort,
            development_end_utc="2025-09-01T00:00:00Z",
            calibration_end_utc="2025-08-31T00:00:00Z",
            pseudonymization_key=b"k" * 32,
        )
    with pytest.raises(LongitudinalIntegrityError, match="all temporal partitions"):
        build_temporal_split_manifest(
            cohort,
            development_end_utc="2025-01-15T00:00:00Z",
            calibration_end_utc="2025-01-20T00:00:00Z",
            pseudonymization_key=b"k" * 32,
        )


def _risk(risk: float, outcome: int) -> RiskObservation:
    return RiskObservation.create(
        predicted_risk=risk,
        outcome=outcome,
        target_status=TargetStatus.OBSERVED,
    )


def _evaluation_config(
    *, minimum: int = 4, bins: int = 2, minimum_bin: int = 2
) -> RiskEvaluationConfig:
    return RiskEvaluationConfig.create(
        horizon_days=90,
        minimum_evaluable_count=minimum,
        calibration_bin_count=bins,
        minimum_calibration_bin_count=minimum_bin,
    )


def test_time_dependent_metrics_match_known_binary_example() -> None:
    observations = [_risk(0.1, 0), _risk(0.4, 0), _risk(0.35, 1), _risk(0.8, 1)]

    result = evaluate_time_dependent_binary_risk(
        "future_mi",
        observations,
        _evaluation_config(),
    )

    assert result.status is EvaluationStatus.OK
    assert result.auroc.status is MetricStatus.OK
    assert result.auroc.value == pytest.approx(0.75)
    assert result.average_precision.value == pytest.approx(5.0 / 6.0)
    assert result.brier_score.value == pytest.approx(0.158125)
    assert result.observed_outcome_count == 4
    assert result.positive_count == 2
    assert result.negative_count == 2
    assert result.calibration_bins[0].status is CalibrationBinStatus.OK
    assert result.calibration_bins[1].status is CalibrationBinStatus.SUPPRESSED_LOW_COUNT
    assert result.calibration_bins[1].count is None


def test_censored_rows_are_counted_but_never_treated_as_negative() -> None:
    observations = [
        _risk(0.1, 0),
        _risk(0.2, 0),
        _risk(0.8, 1),
        _risk(0.9, 1),
        RiskObservation.create(
            predicted_risk=0.7,
            outcome=None,
            target_status=TargetStatus.RIGHT_CENSORED,
        ),
        RiskObservation.create(
            predicted_risk=0.6,
            outcome=None,
            target_status=TargetStatus.INSUFFICIENT_FOLLOW_UP,
        ),
    ]

    result = evaluate_time_dependent_binary_risk(
        "future_mi",
        observations,
        _evaluation_config(),
    )

    assert result.total_prediction_count == 6
    assert result.observed_outcome_count == 4
    assert result.negative_count == 2
    assert dict(result.excluded_status_counts) == {
        "insufficient_follow_up": 1,
        "right_censored": 1,
    }


def test_metric_validity_reports_partial_or_insufficient_evidence() -> None:
    all_negative = [_risk(0.1, 0), _risk(0.2, 0), _risk(0.3, 0), _risk(0.4, 0)]
    partial = evaluate_time_dependent_binary_risk(
        "future_mi",
        all_negative,
        _evaluation_config(),
    )
    assert partial.status is EvaluationStatus.PARTIAL_EVIDENCE
    assert partial.auroc.status is MetricStatus.NO_POSITIVE_EVENTS
    assert partial.average_precision.status is MetricStatus.NO_POSITIVE_EVENTS
    assert partial.brier_score.status is MetricStatus.OK

    too_small = evaluate_time_dependent_binary_risk(
        "future_mi",
        [_risk(0.8, 1)],
        _evaluation_config(),
    )
    assert too_small.status is EvaluationStatus.INSUFFICIENT_EVIDENCE
    assert too_small.auroc.status is MetricStatus.INSUFFICIENT_SAMPLE_SIZE
    assert too_small.brier_score.value is None

    no_observed = evaluate_time_dependent_binary_risk(
        "future_mi",
        [
            RiskObservation.create(
                predicted_risk=0.5,
                outcome=None,
                target_status=TargetStatus.RIGHT_CENSORED,
            )
        ],
        _evaluation_config(),
    )
    assert no_observed.auroc.status is MetricStatus.NO_OBSERVED_OUTCOMES


def test_calibration_bins_are_fixed_width_include_one_and_suppress_small_cells() -> None:
    observations = [
        _risk(0.0, 0),
        _risk(0.1, 0),
        _risk(0.2, 1),
        _risk(0.8, 1),
        _risk(0.9, 1),
        _risk(1.0, 0),
    ]
    result = evaluate_time_dependent_binary_risk(
        "future_event",
        observations,
        _evaluation_config(minimum=2, bins=2, minimum_bin=3),
    )

    assert [item.count for item in result.calibration_bins] == [3, 3]
    assert result.calibration_bins[0].mean_predicted_risk == pytest.approx(0.1)
    assert result.calibration_bins[0].observed_event_rate == pytest.approx(1 / 3)
    assert result.calibration_bins[1].includes_upper_bound is True
    assert result.calibration_bins[1].mean_predicted_risk == pytest.approx(0.9)


def test_risk_observation_contract_and_target_adapter_are_strict() -> None:
    target = BinaryFutureTarget(
        target_name="future_mi",
        value=1,
        status=TargetStatus.OBSERVED,
    )
    observation = RiskObservation.from_target(target, predicted_risk=0.8)
    assert observation.outcome == 1
    assert observation.predicted_risk == 0.8

    with pytest.raises(LongitudinalError, match=r"\[0, 1\]"):
        RiskObservation.create(
            predicted_risk=1.1,
            outcome=1,
            target_status=TargetStatus.OBSERVED,
        )
    with pytest.raises(LongitudinalError, match="outcome=null"):
        RiskObservation.create(
            predicted_risk=0.5,
            outcome=0,
            target_status=TargetStatus.RIGHT_CENSORED,
        )


def test_public_evaluation_is_aggregate_only_and_carries_research_limits() -> None:
    result = evaluate_time_dependent_binary_risk(
        "future_mi",
        [_risk(0.1, 0), _risk(0.2, 0), _risk(0.8, 1), _risk(0.9, 1)],
        _evaluation_config(),
    )
    public = result.to_public_dict()
    serialized = json.dumps(public, sort_keys=True)

    assert public["privacy_contract"] == AGGREGATE_ONLY_LIMIT
    assert public["research_use_limit"] == RESEARCH_USE_LIMIT
    assert public["target_interpretation"] == NON_CAUSAL_TARGET_LIMIT
    assert public["censoring_handling"] == "observed_targets_only_no_ipcw"
    assert "patient_key" not in serialized
    assert "encounter_key" not in serialized
    assert "patient_token" not in serialized
