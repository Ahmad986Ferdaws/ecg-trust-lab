from __future__ import annotations

import json

import numpy as np
import pytest

from ecg_trust.evaluation import EvaluationValidationError
from ecg_trust.sph_transport_metrics import (
    SPHTransportMetricsError,
    evaluate_sph_transport,
    paired_sph_transport_differences,
)


def _inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    patient_ids = np.repeat(
        np.asarray([f"private-patient-{index}" for index in range(12)], dtype=object),
        2,
    )
    patient_targets = (np.arange(12)[:, None] + np.arange(5)[None, :]) % 2
    targets = np.repeat(patient_targets, 2, axis=0).astype(np.int64)
    logits = np.where(targets == 1, 2.0, -2.0).astype(np.float64)
    # Add patient-cluster errors so intervals are nontrivial.
    logits[np.isin(patient_ids, ["private-patient-2", "private-patient-7"])] *= -1.0
    return targets, logits, patient_ids


def _gates() -> tuple[dict[str, float], ...]:
    return (
        {"target_coverage": 1.0, "maximum_entropy": 1.0},
        {"target_coverage": 0.5, "maximum_entropy": 0.0},
    )


def test_member_report_is_deterministic_json_safe_and_identifier_free() -> None:
    targets, logits, patient_ids = _inputs()

    first = evaluate_sph_transport(
        targets,
        logits,
        patient_ids,
        temperature=2.0,
        thresholds=(0.5,) * 5,
        entropy_gates=_gates(),
        n_resamples=30,
        seed=17,
        minimum_valid_resamples=10,
        ece_bins=5,
    )
    second = evaluate_sph_transport(
        targets,
        logits,
        patient_ids,
        temperature=2.0,
        thresholds=(0.5,) * 5,
        entropy_gates=_gates(),
        n_resamples=30,
        seed=17,
        minimum_valid_resamples=10,
        ece_bins=5,
    )

    assert first == second
    serialized = json.dumps(first, allow_nan=False, sort_keys=True)
    assert "private-patient" not in serialized
    assert first["analysis_kind"] == "exploratory_external_transport_stress_test"
    assert first["clinical_validation"] is False
    views = first["probability_views"]
    assert isinstance(views, dict)
    raw = views["raw_sigmoid"]
    calibrated = views["frozen_temperature_calibrated"]
    assert isinstance(raw, dict)
    assert isinstance(calibrated, dict)
    assert raw["metrics"] != calibrated["metrics"]
    assert raw["patient_cluster_bootstrap"] != calibrated["patient_cluster_bootstrap"]

    decisions = first["frozen_threshold_decisions"]
    assert isinstance(decisions, dict)
    assert decisions["hamming_risk"] == pytest.approx(1.0 / 6.0)
    assert decisions["exact_match_accuracy"] == pytest.approx(5.0 / 6.0)

    gates = first["frozen_entropy_gates"]
    assert isinstance(gates, list)
    assert gates[0]["observed_coverage"] == 1.0
    assert gates[0]["hamming_risk"] == pytest.approx(1.0 / 6.0)
    assert gates[1] == {
        "target_coverage": 0.5,
        "maximum_entropy": 0.0,
        "observed_coverage": 0.0,
        "selected_count": 0,
        "abstained_count": 24,
        "hamming_risk": None,
        "exact_match_accuracy": None,
        "status": "no_accepted_samples",
    }


def test_degenerate_labels_are_explicit_in_point_and_bootstrap_reports() -> None:
    targets, logits, patient_ids = _inputs()
    targets[:, 4] = 0
    logits[:, 4] = -2.0

    result = evaluate_sph_transport(
        targets,
        logits,
        patient_ids,
        temperature=1.0,
        thresholds=(0.5,) * 5,
        entropy_gates=({"target_coverage": 1.0, "maximum_entropy": 1.0},),
        n_resamples=20,
        seed=8,
        minimum_valid_resamples=5,
    )

    views = result["probability_views"]
    assert isinstance(views, dict)
    calibrated = views["frozen_temperature_calibrated"]
    assert isinstance(calibrated, dict)
    metrics = calibrated["metrics"]
    bootstrap = calibrated["patient_cluster_bootstrap"]
    assert isinstance(metrics, dict)
    assert isinstance(bootstrap, dict)
    per_label = metrics["per_label"]
    bootstrap_per_label = bootstrap["per_label"]
    assert isinstance(per_label, list)
    assert isinstance(bootstrap_per_label, list)
    assert per_label[4]["degenerate_reason"] == "no_positive_examples"
    assert per_label[4]["roc_auc"] is None
    assert bootstrap_per_label[4]["roc_auc"]["status"] == "undefined_point_estimate"


def test_paired_report_uses_transformer_minus_resnet_shared_draws() -> None:
    targets, resnet_logits, patient_ids = _inputs()
    transformer_logits = resnet_logits.copy()
    transformer_logits[np.isin(patient_ids, ["private-patient-2"])] *= -1.0

    first = paired_sph_transport_differences(
        targets,
        resnet_logits,
        transformer_logits,
        patient_ids,
        resnet_temperature=1.0,
        transformer_temperature=1.0,
        n_resamples=40,
        seed=31,
        minimum_valid_resamples=10,
        ece_bins=5,
    )
    second = paired_sph_transport_differences(
        targets,
        resnet_logits,
        transformer_logits,
        patient_ids,
        resnet_temperature=1.0,
        transformer_temperature=1.0,
        n_resamples=40,
        seed=31,
        minimum_valid_resamples=10,
        ece_bins=5,
    )

    assert first == second
    assert first["difference_direction"] == "ecg_transformer_minus_resnet1d"
    views = first["probability_views"]
    assert isinstance(views, dict)
    raw = views["raw_sigmoid"]
    assert isinstance(raw, dict)
    assert raw["difference_direction"] == "model_a_minus_model_b"
    assert raw["model_a"] == "ecg_transformer"
    assert raw["model_b"] == "resnet1d"
    macro = raw["macro"]
    assert isinstance(macro, dict)
    assert macro["brier_score"]["estimate"] < 0.0
    assert "private-patient" not in json.dumps(first, allow_nan=False)


@pytest.mark.parametrize(
    ("change", "match", "error_type"),
    [
        ("nonbinary_targets", "binary", EvaluationValidationError),
        ("wrong_logit_shape", "shape", EvaluationValidationError),
        ("nonfinite_logits", "finite", EvaluationValidationError),
        ("bad_temperature", "temperature", SPHTransportMetricsError),
        ("bad_thresholds", "thresholds", SPHTransportMetricsError),
        ("boolean_threshold", "thresholds", SPHTransportMetricsError),
        ("bad_gate_order", "strictly decreasing", SPHTransportMetricsError),
        ("missing_gate_cutoff", "maximum_entropy", SPHTransportMetricsError),
    ],
)
def test_member_input_validation(
    change: str,
    match: str,
    error_type: type[Exception],
) -> None:
    targets, logits, patient_ids = _inputs()
    temperature = 1.0
    thresholds: tuple[float, ...] = (0.5,) * 5
    gates: tuple[dict[str, float], ...] = _gates()
    if change == "nonbinary_targets":
        targets = targets.copy()
        targets[0, 0] = 2
    elif change == "wrong_logit_shape":
        logits = logits[:, :4]
    elif change == "nonfinite_logits":
        logits = logits.copy()
        logits[0, 0] = np.nan
    elif change == "bad_temperature":
        temperature = 0.0
    elif change == "bad_thresholds":
        thresholds = (0.5,) * 4
    elif change == "boolean_threshold":
        thresholds = (True, 0.5, 0.5, 0.5, 0.5)
    elif change == "bad_gate_order":
        gates = tuple(reversed(_gates()))
    elif change == "missing_gate_cutoff":
        gates = ({"target_coverage": 1.0},)

    with pytest.raises(error_type, match=match):
        evaluate_sph_transport(
            targets,
            logits,
            patient_ids,
            temperature=temperature,
            thresholds=thresholds,
            entropy_gates=gates,
            n_resamples=10,
            minimum_valid_resamples=5,
        )


def test_paired_report_rejects_misaligned_transformer_logits() -> None:
    targets, logits, patient_ids = _inputs()

    with pytest.raises(EvaluationValidationError, match="samples"):
        paired_sph_transport_differences(
            targets,
            logits,
            logits[:-1],
            patient_ids,
            resnet_temperature=1.0,
            transformer_temperature=1.0,
            n_resamples=10,
            minimum_valid_resamples=5,
        )
