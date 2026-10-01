from __future__ import annotations

import numpy as np
import pytest

from ecg_trust.evaluation import optimize_thresholds


def _brute_force_threshold(
    targets: np.ndarray, scores: np.ndarray, default: float
) -> tuple[float, float]:
    """Independent confusion-count oracle, including the established tie policy."""
    best_threshold, best_f1 = default, -1.0
    for threshold in sorted({0.0, 1.0, default, *scores.tolist()}):
        counts = {(actual, predicted): 0 for actual in (0, 1) for predicted in (0, 1)}
        for actual, score in zip(targets, scores, strict=True):
            counts[(int(actual), int(score >= threshold))] += 1
        tp, fp, fn = counts[(1, 1)], counts[(0, 1)], counts[(1, 0)]
        denominator = 2 * tp + fp + fn
        f1 = 2.0 * tp / denominator if denominator else 0.0
        if f1 > best_f1 + 1e-12 or (
            abs(f1 - best_f1) <= 1e-12
            and (abs(threshold - default), -threshold)
            < (abs(best_threshold - default), -best_threshold)
        ):
            best_threshold, best_f1 = threshold, f1
    return best_threshold, best_f1


@pytest.mark.parametrize("default", [0.0, 0.17, 0.5, 0.83, 1.0])
@pytest.mark.parametrize("quantized", [False, True])
def test_threshold_search_exactly_matches_brute_force(default: float, quantized: bool) -> None:
    rng = np.random.default_rng(403)
    for count in (2, 3, 7, 16, 61, 127):
        targets = rng.integers(0, 2, size=(count, 5))
        targets[:2] = np.asarray([[0] * 5, [1] * 5])
        scores = rng.random(size=(count, 5))
        if quantized:
            scores = np.round(scores * 4) / 4
        result = optimize_thresholds(
            y_true=targets,
            probabilities=scores,
            calibration_fold_ids=np.full(count, 9),
            default_threshold=default,
        )

        for label_index, label_result in enumerate(result.per_label):
            expected_threshold, expected_f1 = _brute_force_threshold(
                targets[:, label_index], scores[:, label_index], default
            )
            assert label_result.threshold == expected_threshold
            assert label_result.objective_value == expected_f1


@pytest.mark.parametrize("probability", [0.0, 0.5, 1.0])
def test_threshold_search_keeps_all_rows_at_shared_probability(probability: float) -> None:
    targets = np.tile(np.asarray([[0], [1], [1], [0]]), (1, 5))
    scores = np.full(targets.shape, probability)
    result = optimize_thresholds(
        y_true=targets, probabilities=scores, calibration_fold_ids=np.full(4, 9)
    )

    expected_threshold, expected_f1 = _brute_force_threshold(targets[:, 0], scores[:, 0], 0.5)
    assert result.thresholds == (expected_threshold,) * 5
    assert all(label.objective_value == expected_f1 for label in result.per_label)
