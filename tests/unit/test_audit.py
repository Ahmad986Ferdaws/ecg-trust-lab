from __future__ import annotations

import json

import numpy as np
import pytest

from ecg_trust.audit import (
    AuditValidationError,
    audit_subgroups,
    bootstrap_multilabel_metrics,
    paired_model_difference_intervals,
)


def _clustered_binary_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Each patient contributes two records and has one outcome for every label.
    patient_ids = np.repeat(np.arange(12), 2)
    patient_targets = (np.arange(12) % 2).astype(np.int64)
    targets = np.tile(np.repeat(patient_targets, 2)[:, None], (1, 5))
    base_scores = np.where(targets[:, 0] == 1, 0.78, 0.22)
    # Make a few patient-level errors so intervals have real variation.
    base_scores[np.isin(patient_ids, [2, 7])] = 1.0 - base_scores[
        np.isin(patient_ids, [2, 7])
    ]
    probabilities = np.tile(base_scores[:, None], (1, 5))
    return targets, probabilities, patient_ids


def test_patient_cluster_bootstrap_is_deterministic_and_serializable() -> None:
    targets, probabilities, patient_ids = _clustered_binary_data()

    first = bootstrap_multilabel_metrics(
        targets,
        probabilities,
        patient_ids,
        n_resamples=80,
        seed=41,
        minimum_valid_resamples=20,
        ece_bins=5,
    )
    second = bootstrap_multilabel_metrics(
        targets,
        probabilities,
        patient_ids,
        n_resamples=80,
        seed=41,
        minimum_valid_resamples=20,
        ece_bins=5,
    )

    assert first == second
    assert first.n_samples == 24
    assert first.n_patients == 12
    assert first.completed_resamples == 80
    assert first.macro.roc_auc.lower is not None
    assert first.macro.roc_auc.upper is not None
    assert first.macro.roc_auc.lower <= first.macro.roc_auc.estimate <= first.macro.roc_auc.upper
    assert first.per_label[0].prevalence.valid_resamples == 80
    json.dumps(first.to_dict(), allow_nan=False)


def test_cluster_bootstrap_counts_degenerate_patient_draws_explicitly() -> None:
    patient_ids = np.asarray(["negative", "positive"])
    targets = np.tile(np.asarray([[0], [1]]), (1, 5))
    probabilities = np.tile(np.asarray([[0.1], [0.9]]), (1, 5))

    result = bootstrap_multilabel_metrics(
        targets,
        probabilities,
        patient_ids,
        n_resamples=200,
        seed=7,
        minimum_valid_resamples=50,
    )

    interval = result.per_label[0].roc_auc
    assert 0 < interval.invalid_resamples < 200
    assert interval.valid_resamples + interval.invalid_resamples == 200
    assert interval.status == "ok_with_degenerate_replicates"
    assert interval.estimate == 1.0
    assert interval.lower == 1.0
    assert interval.upper == 1.0


def test_macro_discrimination_replicates_keep_the_point_estimate_label_set() -> None:
    patient_ids = np.arange(8)
    targets = np.zeros((8, 5), dtype=np.int64)
    targets[0, 0] = 1  # The first label is rare and may disappear from a patient draw.
    alternating = np.arange(8) % 2
    targets[:, 1:] = alternating[:, None]
    probabilities = np.where(targets == 1, 0.9, 0.1)

    result = bootstrap_multilabel_metrics(
        targets,
        probabilities,
        patient_ids,
        n_resamples=120,
        seed=13,
        minimum_valid_resamples=20,
    )

    rare_label = result.per_label[0].roc_auc
    assert rare_label.invalid_resamples > 0
    # A macro replicate with the rare label missing is rejected rather than
    # silently averaging a different set of labels.
    assert result.macro.roc_auc.invalid_resamples >= rare_label.invalid_resamples
    assert result.macro.roc_auc.status == "ok_with_degenerate_replicates"


def test_one_patient_returns_explicit_small_sample_status_without_fake_interval() -> None:
    targets = np.asarray([[0, 1, 0, 1, 0], [1, 0, 1, 0, 1]])
    probabilities = np.where(targets == 1, 0.8, 0.2)

    result = bootstrap_multilabel_metrics(
        targets,
        probabilities,
        patient_ids=np.asarray([100, 100]),
        n_resamples=20,
        minimum_valid_resamples=10,
    )

    assert result.status == "insufficient_patient_clusters"
    assert result.completed_resamples == 0
    interval = result.macro.brier_score
    assert interval.estimate == pytest.approx(0.04)
    assert interval.lower is None
    assert interval.upper is None
    assert interval.valid_resamples == 0
    assert interval.invalid_resamples == 20
    assert interval.status == "insufficient_patient_clusters"


def test_paired_model_difference_uses_shared_patient_draws() -> None:
    targets, probabilities_a, patient_ids = _clustered_binary_data()
    probabilities_b = 1.0 - probabilities_a

    result = paired_model_difference_intervals(
        targets,
        probabilities_a,
        probabilities_b,
        patient_ids,
        model_a="resnet",
        model_b="transformer",
        n_resamples=80,
        seed=19,
        minimum_valid_resamples=20,
        ece_bins=5,
    )
    repeated = paired_model_difference_intervals(
        targets,
        probabilities_a,
        probabilities_b,
        patient_ids,
        model_a="resnet",
        model_b="transformer",
        n_resamples=80,
        seed=19,
        minimum_valid_resamples=20,
        ece_bins=5,
    )

    assert result == repeated
    assert result.to_dict()["difference_direction"] == "model_a_minus_model_b"
    assert result.per_label[0].roc_auc.estimate is not None
    assert result.per_label[0].roc_auc.estimate > 0
    assert result.per_label[0].roc_auc.higher_is_better is True
    assert result.per_label[0].brier_score.estimate is not None
    assert result.per_label[0].brier_score.estimate < 0
    assert result.per_label[0].brier_score.higher_is_better is False
    json.dumps(result.to_dict(), allow_nan=False)


def test_identical_models_have_zero_paired_difference_interval() -> None:
    targets, probabilities, patient_ids = _clustered_binary_data()

    result = paired_model_difference_intervals(
        targets,
        probabilities,
        probabilities.copy(),
        patient_ids,
        model_a="seed-a",
        model_b="seed-b",
        n_resamples=40,
        seed=5,
        minimum_valid_resamples=10,
    )

    assert result.macro.brier_score.estimate == 0.0
    assert result.macro.brier_score.lower == 0.0
    assert result.macro.brier_score.upper == 0.0


def test_subgroup_audit_reports_prevalence_metrics_and_shared_gate_coverage() -> None:
    targets = np.asarray(
        [
            [0, 1, 0, 1, 0],
            [1, 0, 1, 0, 1],
            [0, 1, 1, 0, 0],
            [1, 0, 0, 1, 1],
            [0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 0, 1, 0, 0],
            [1, 1, 1, 1, 1],
        ],
        dtype=np.int64,
    )
    # Group A is confident, B is uncertain, and the missing group is smallest.
    probabilities = np.vstack(
        [
            np.where(targets[:4] == 1, 0.99, 0.01),
            np.where(targets[4:7] == 1, 0.55, 0.45),
            np.where(targets[7:] == 1, 0.51, 0.49),
        ]
    )
    patient_ids = np.arange(8)
    group_values = np.asarray(["A", "A", "A", "A", "B", "B", "B", None], dtype=object)

    result = audit_subgroups(
        targets,
        probabilities,
        patient_ids,
        {"site": group_values},
        thresholds=(0.5,) * 5,
        coverage_targets=(0.5, 1.0),
        minimum_group_samples=2,
        minimum_group_patients=2,
        ece_bins=5,
    )

    groups = {entry.group_value: entry for entry in result.groups}
    assert result.n_samples == 8
    assert result.n_patients == 8
    assert groups["A"].n_samples == 4
    assert groups["A"].n_patients == 4
    assert groups["A"].status == "ok"
    assert groups["A"].selective_coverage[0].subgroup_coverage == 1.0
    assert groups["B"].selective_coverage[0].subgroup_coverage == 0.0
    assert groups["B"].selective_coverage[0].status == "no_selected_samples"
    assert groups["B"].metrics.per_label[0].prevalence == 0.0
    assert groups["B"].metrics.per_label[0].degenerate_reason == "no_positive_examples"
    assert groups["<MISSING>"].status == "small_group_descriptive_only"
    assert groups["<MISSING>"].group_value_type == "missing"
    json.dumps(result.to_dict(), allow_nan=False)


def test_subgroup_audit_accepts_multiple_attributes_in_deterministic_order() -> None:
    targets, probabilities, patient_ids = _clustered_binary_data()
    sex = np.where(patient_ids % 2 == 0, "F", "M")
    age_band = np.where(patient_ids < 6, "younger", "older")

    result = audit_subgroups(
        targets,
        probabilities,
        patient_ids,
        {"sex": sex, "age_band": age_band},
        coverage_targets=(1.0,),
        minimum_group_samples=1,
        minimum_group_patients=1,
    )

    assert [entry.attribute for entry in result.groups] == [
        "age_band",
        "age_band",
        "sex",
        "sex",
    ]
    assert all(entry.selective_coverage[0].subgroup_coverage == 1.0 for entry in result.groups)


def test_audit_validation_rejects_invalid_patient_and_subgroup_contracts() -> None:
    targets = np.tile(np.asarray([[0], [1]]), (1, 5))
    probabilities = np.where(targets == 1, 0.9, 0.1)

    with pytest.raises(AuditValidationError, match="patient_ids"):
        bootstrap_multilabel_metrics(
            targets,
            probabilities,
            patient_ids=np.asarray([1, np.nan]),
            n_resamples=10,
        )
    with pytest.raises(AuditValidationError, match="one value per sample"):
        audit_subgroups(
            targets,
            probabilities,
            patient_ids=np.asarray([1, 2]),
            subgroups={"sex": np.asarray(["F"])},
            minimum_group_samples=1,
            minimum_group_patients=1,
        )
    with pytest.raises(AuditValidationError, match="distinct names"):
        paired_model_difference_intervals(
            targets,
            probabilities,
            probabilities,
            patient_ids=np.asarray([1, 2]),
            model_a="same",
            model_b="same",
            n_resamples=10,
        )
