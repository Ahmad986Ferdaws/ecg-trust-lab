from __future__ import annotations

import numpy as np
import pytest

from ecg_trust.post_analysis import (
    PostAnalysisError,
    dense_risk_coverage,
    derive_probability_audit,
    error_detection_metrics,
    frozen_gate_audit,
    mean_normalized_binary_entropy,
    multilabel_log_loss,
    reliability_curves,
)


def _arrays() -> tuple[np.ndarray, np.ndarray]:
    targets = np.asarray(
        [
            [1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 0, 1, 1],
            [1, 0, 0, 0, 1],
        ],
        dtype=np.int64,
    )
    probabilities = np.asarray(
        [
            [0.95, 0.05, 0.05, 0.05, 0.05],
            [0.80, 0.75, 0.10, 0.10, 0.10],
            [0.20, 0.70, 0.65, 0.10, 0.10],
            [0.20, 0.20, 0.70, 0.65, 0.20],
            [0.10, 0.10, 0.10, 0.75, 0.80],
            [0.55, 0.45, 0.45, 0.45, 0.55],
        ],
        dtype=np.float64,
    )
    return targets, probabilities


def test_entropy_and_log_loss_are_finite_and_ordered() -> None:
    targets, probabilities = _arrays()
    entropy = mean_normalized_binary_entropy(probabilities)
    assert entropy.shape == (6,)
    assert entropy[-1] > entropy[0]
    assert multilabel_log_loss(targets, probabilities) > 0.0


def test_equal_mass_reliability_preserves_all_rows() -> None:
    targets, probabilities = _arrays()
    curves = reliability_curves(targets, probabilities, n_bins=4)
    assert len(curves) == 5
    assert all(sum(item.count for item in curve.bins) == 6 for curve in curves)
    assert all(len(curve.bins) == 4 for curve in curves)


def test_dense_risk_coverage_uses_every_prefix() -> None:
    targets, probabilities = _arrays()
    curve = dense_risk_coverage(
        targets,
        probabilities,
        thresholds=(0.5, 0.5, 0.5, 0.5, 0.5),
    )
    assert curve.coverage.tolist() == pytest.approx([1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6, 1])
    assert curve.hamming_risk[0] == 0.0
    assert curve.hamming_risk[-1] == pytest.approx(0.0)
    assert 0.0 <= curve.aurc_log_loss <= 1.0


def test_error_detection_reports_degenerate_perfect_predictions() -> None:
    targets, probabilities = _arrays()
    result = error_detection_metrics(
        targets,
        probabilities,
        thresholds=(0.5, 0.5, 0.5, 0.5, 0.5),
    )
    assert result["status"] == "degenerate_error_target"
    assert result["positives"] == 0


def test_frozen_gates_report_scores_prevalence_and_group_coverage() -> None:
    targets, probabilities = _arrays()
    gates = (
        {"target_coverage": 1.0, "maximum_entropy": 1.0},
        {"target_coverage": 0.5, "maximum_entropy": 0.75},
    )
    results = frozen_gate_audit(
        targets,
        probabilities,
        thresholds=(0.5, 0.5, 0.5, 0.5, 0.5),
        gates=gates,
        subgroups={"sex": np.asarray(["F", "M", "F", "M", "F", "M"])},
    )
    assert len(results) == 2
    assert results[0]["achieved_coverage"] == 1.0
    assert len(results[0]["accepted_prevalence"]) == 5  # type: ignore[arg-type]
    assert "sex" in results[1]["subgroup_coverage"]  # type: ignore[operator]


def test_full_probability_audit_compares_raw_and_calibrated() -> None:
    targets, probabilities = _arrays()
    raw = np.clip((probabilities - 0.5) * 1.4 + 0.5, 0.001, 0.999)
    payload = derive_probability_audit(
        targets,
        raw,
        probabilities,
        thresholds=(0.5, 0.5, 0.5, 0.5, 0.5),
        gates=(
            {"target_coverage": 1.0, "maximum_entropy": 1.0},
            {"target_coverage": 0.5, "maximum_entropy": 0.75},
        ),
    )
    assert payload["n_samples"] == 6
    assert payload["raw"]["log_loss"] != payload["calibrated"]["log_loss"]  # type: ignore[index]
    assert len(payload["dense_risk_coverage"]["coverage"]) == 6  # type: ignore[index,arg-type]


def test_rejects_misaligned_subgroups_and_invalid_gate_order() -> None:
    targets, probabilities = _arrays()
    with pytest.raises(PostAnalysisError, match="align"):
        frozen_gate_audit(
            targets,
            probabilities,
            thresholds=(0.5,) * 5,
            gates=({"target_coverage": 1.0, "maximum_entropy": 1.0},),
            subgroups={"sex": ["F"]},
        )
    with pytest.raises(PostAnalysisError, match="decreasing"):
        frozen_gate_audit(
            targets,
            probabilities,
            thresholds=(0.5,) * 5,
            gates=(
                {"target_coverage": 0.5, "maximum_entropy": 0.5},
                {"target_coverage": 0.8, "maximum_entropy": 0.8},
            ),
        )
