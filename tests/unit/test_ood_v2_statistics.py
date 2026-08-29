from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from ecg_trust.ood_v2 import (
    ExternalCohortRole,
    OODAxis,
    ResamplingUnit,
    TechnicalQualityEventDefinition,
    bootstrap_proportion_interval,
    evaluate_external_ood_gate,
    evaluate_source_gate,
    evaluate_technical_quality_gate,
)


def _digest(character: str) -> str:
    return "sha256:" + character * 64


def test_record_bootstrap_is_reproducible_and_reports_both_interval_types() -> None:
    events = np.asarray([True] * 73 + [False] * 27, dtype=np.bool_)

    first = bootstrap_proportion_interval(
        events,
        resampling_unit=ResamplingUnit.RECORD,
        seed=71,
        replicates=2_000,
        confidence_level=0.9875,
    )
    second = bootstrap_proportion_interval(
        events,
        resampling_unit="record",
        seed=71,
        replicates=2_000,
        confidence_level=0.9875,
    )

    assert first == second
    assert first.point_estimate == 0.73
    assert first.confidence_level == 0.9875
    assert first.resampling_units == 100
    assert first.two_sided_lower <= first.one_sided_lower
    assert first.one_sided_lower <= first.one_sided_upper
    assert first.one_sided_upper <= first.two_sided_upper


def test_record_bootstrap_uses_exact_draw_n_indices_with_replacement_rule() -> None:
    events = np.asarray([True, False, True, False, False], dtype=np.bool_)
    seed = 7_301
    replicates = 1_000
    confidence_level = 0.95

    observed = bootstrap_proportion_interval(
        events,
        resampling_unit="record",
        seed=seed,
        replicates=replicates,
        confidence_level=confidence_level,
    )

    generator = np.random.Generator(np.random.PCG64(seed))
    sampled = generator.integers(
        0,
        events.shape[0],
        size=(replicates, events.shape[0]),
        endpoint=False,
    )
    rates = events[sampled].mean(axis=1, dtype=np.float64)
    alpha = 0.05
    expected = np.quantile(
        rates,
        [alpha / 2.0, 1.0 - alpha / 2.0, alpha, 1.0 - alpha],
        method="linear",
    )

    assert (
        observed.two_sided_lower,
        observed.two_sided_upper,
        observed.one_sided_lower,
        observed.one_sided_upper,
    ) == tuple(float(value) for value in expected)


def test_preregistered_confidence_uses_exact_decimal_tail_quantiles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_quantiles: list[np.ndarray] = []
    original_quantile = np.quantile

    def capture_quantiles(*args: Any, **kwargs: Any) -> Any:
        observed_quantiles.append(np.asarray(args[1], dtype=np.float64))
        return original_quantile(*args, **kwargs)

    monkeypatch.setattr("ecg_trust.ood_v2.statistics.np.quantile", capture_quantiles)
    bootstrap_proportion_interval(
        np.asarray([True, False] * 50, dtype=np.bool_),
        resampling_unit=ResamplingUnit.RECORD,
        seed=73,
        replicates=1_000,
        confidence_level=0.9875,
    )

    assert len(observed_quantiles) == 1
    assert np.array_equal(
        observed_quantiles[0],
        np.asarray([0.00625, 0.99375, 0.0125, 0.9875], dtype=np.float64),
    )


def test_patient_cluster_bootstrap_is_order_invariant_and_carries_all_records() -> None:
    labels = np.asarray([30, 10, 30, 20, 30, 20], dtype=np.int64)
    events = np.asarray([True, False, False, True, True, False], dtype=np.bool_)
    permutation = np.asarray([5, 0, 3, 2, 1, 4], dtype=np.int64)

    first = bootstrap_proportion_interval(
        events,
        resampling_unit="patient_cluster",
        cluster_labels=labels,
        seed=72,
        replicates=2_000,
        confidence_level=0.95,
    )
    reordered = bootstrap_proportion_interval(
        events[permutation],
        resampling_unit="patient_cluster",
        cluster_labels=labels[permutation],
        seed=72,
        replicates=2_000,
        confidence_level=0.95,
    )

    assert first == reordered
    assert first.records == 6
    assert first.resampling_units == 3
    assert first.event_count == 3
    assert "cluster" not in first.model_dump(mode="json")


def test_record_and_patient_cluster_are_explicitly_distinct_estimators() -> None:
    labels = np.asarray([1] * 20 + [2] + [3], dtype=np.int64)
    events = np.asarray([True] * 20 + [False, False], dtype=np.bool_)
    record = bootstrap_proportion_interval(
        events,
        resampling_unit="record",
        seed=73,
        replicates=2_000,
        confidence_level=0.95,
    )
    patient = bootstrap_proportion_interval(
        events,
        resampling_unit="patient_cluster",
        cluster_labels=labels,
        seed=73,
        replicates=2_000,
        confidence_level=0.95,
    )

    assert record.point_estimate == patient.point_estimate == 20 / 22
    assert record.resampling_units == 22
    assert patient.resampling_units == 3
    assert (record.two_sided_lower, record.two_sided_upper) != (
        patient.two_sided_lower,
        patient.two_sided_upper,
    )


def test_source_gate_uses_upper_bound_not_only_the_point_estimate() -> None:
    passing = evaluate_source_gate(
        np.zeros(200, dtype=np.bool_),
        cohort_key="source-dev-holdout",
        cohort_manifest_sha256=_digest("a"),
        resampling_unit="record",
        maximum_false_rejection_rate=0.05,
        seed=74,
        replicates=1_000,
        confidence_level=0.95,
    )
    missed = evaluate_source_gate(
        np.asarray([True] * 12 + [False] * 188, dtype=np.bool_),
        cohort_key="source-dev-holdout",
        cohort_manifest_sha256=_digest("a"),
        resampling_unit="record",
        maximum_false_rejection_rate=0.05,
        seed=74,
        replicates=1_000,
        confidence_level=0.95,
    )

    assert passing.gate_passed is True
    assert passing.support_coverage == 1.0
    assert missed.false_rejection_rate == 0.06
    assert missed.interval.one_sided_upper > 0.05
    assert missed.gate_passed is False


def test_external_gate_uses_lower_bound_and_retains_only_aggregates() -> None:
    summary = evaluate_external_ood_gate(
        np.asarray([True] * 190 + [False] * 10, dtype=np.bool_),
        endpoint_key="challenge_external_distribution_recall",
        cohort_key="physionet-challenge-2011-set-a",
        dataset_name="PhysioNet Challenge 2011 Set A",
        dataset_version="1.0.0",
        license_identifier="ODC-By-1.0",
        cohort_manifest_sha256=_digest("b"),
        role_assignment_sha256=_digest("c"),
        evaluation_role=ExternalCohortRole.PHYSIONET_CHALLENGE_2011_SET_A,
        ood_axis=OODAxis.EXTERNAL_ACQUISITION_AND_POPULATION,
        resampling_unit="record",
        minimum_ood_recall=0.90,
        seed=75,
        replicates=2_000,
        confidence_level=0.9875,
    )

    assert summary.ood_recall == 0.95
    assert summary.gate_passed is (summary.interval.one_sided_lower >= 0.90)
    serialized = summary.model_dump(mode="json")
    assert "cluster_labels" not in serialized
    assert "detected" not in serialized
    assert summary.target_site_fitting_performed is False


@pytest.mark.parametrize(
    "event_definition",
    [
        TechnicalQualityEventDefinition.BLOCK_UNACCEPTABLE,
        TechnicalQualityEventDefinition.PASS_ACCEPTABLE,
    ],
)
def test_technical_quality_gate_is_semantically_separate(
    event_definition: TechnicalQualityEventDefinition,
) -> None:
    summary = evaluate_technical_quality_gate(
        np.ones(160, dtype=np.bool_),
        endpoint_key=f"quality-{event_definition.value}",
        cohort_key="quality-challenge",
        event_definition=event_definition,
        resampling_unit="record",
        minimum_rate=0.95,
        seed=76,
        replicates=1_000,
        confidence_level=0.9875,
    )

    assert summary.event_definition is event_definition
    assert summary.point_rate == 1.0
    assert summary.gate_passed is True
    assert not hasattr(summary, "ood_recall")


@pytest.mark.parametrize(
    ("events", "kwargs", "match"),
    [
        ([], {"resampling_unit": "record"}, "non-empty"),
        ([0, 1], {"resampling_unit": "record"}, "strict boolean"),
        ([False, True], {"resampling_unit": "patient_cluster"}, "required"),
        (
            [False, True],
            {"resampling_unit": "record", "cluster_labels": [1, 2]},
            "omitted",
        ),
        (
            [False, True],
            {"resampling_unit": "patient_cluster", "cluster_labels": [1]},
            "align",
        ),
        (
            [False, True],
            {"resampling_unit": "patient_cluster", "cluster_labels": [1, 0]},
            "positive",
        ),
    ],
)
def test_bootstrap_rejects_ambiguous_or_invalid_inputs(
    events: Any,
    kwargs: dict[str, Any],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        bootstrap_proportion_interval(
            events,
            seed=77,
            replicates=1_000,
            confidence_level=0.95,
            **kwargs,
        )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("seed", True, "seed"),
        ("seed", -1, "seed"),
        ("replicates", 999, "at least 1000"),
        ("confidence_level", 1, "confidence_level"),
        ("confidence_level", 1.0, "confidence_level"),
    ],
)
def test_bootstrap_parameters_are_strict(field: str, value: object, match: str) -> None:
    kwargs: dict[str, object] = {
        "seed": 78,
        "replicates": 1_000,
        "confidence_level": 0.95,
    }
    kwargs[field] = value
    with pytest.raises(ValueError, match=match):
        bootstrap_proportion_interval(
            np.asarray([False, True], dtype=np.bool_),
            resampling_unit="record",
            **kwargs,  # type: ignore[arg-type]
        )
