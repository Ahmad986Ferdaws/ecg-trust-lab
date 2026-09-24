from __future__ import annotations

import json
import math
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from ecg_trust.open_world import (
    MahalanobisValidationError,
    MaxNormalizedBernoulliEntropyScorer,
    NormalizedBernoulliEntropyScorer,
    OODScoreValidationError,
    ShrinkageMahalanobisDetector,
    SymmetricBinaryEnergyScorer,
    max_normalized_bernoulli_entropy,
    normalized_bernoulli_entropy,
    symmetric_binary_energy,
)


def test_normalized_entropy_has_documented_ood_direction_and_bounds() -> None:
    scores = normalized_bernoulli_entropy([[0.0, 1.0], [0.01, 0.99], [0.5, 0.5]])

    assert scores[0] == pytest.approx(0.0)
    assert 0.0 < scores[1] < scores[2]
    assert scores[2] == pytest.approx(1.0)


def _binary_entropy_bits(probability: float) -> float:
    if probability in (0.0, 1.0):
        return 0.0
    return -(
        probability * math.log2(probability) + (1.0 - probability) * math.log2(1.0 - probability)
    )


def _pre_refactor_mean_entropy(probabilities: np.ndarray) -> np.ndarray:
    matrix = np.asarray(probabilities, dtype=np.float64)
    entropy = np.zeros_like(matrix)
    interior = (matrix > 0.0) & (matrix < 1.0)
    selected = matrix[interior]
    entropy[interior] = -(
        selected * np.log(selected) + (1.0 - selected) * np.log1p(-selected)
    ) / math.log(2.0)
    return entropy.mean(axis=1)


def _entropy_fixture() -> np.ndarray:
    generator = np.random.default_rng(20260923)
    probabilities = generator.uniform(0.0, 1.0, size=(256, 5))
    probabilities[:16, 0] = 0.0
    probabilities[16:32, 1] = 1.0
    probabilities[32:48, 2] = 0.5
    probabilities[48] = [0.0, 1.0, 0.0, 1.0, 0.0]
    probabilities[49] = 0.5
    # A label just below one half whose entropy rounds one ulp above one bit.
    probabilities[50, 3] = 0.49999999999999983
    return probabilities


def test_max_entropy_matches_hand_computed_worst_label_values() -> None:
    probabilities = [
        [0.5, 0.001, 0.999, 0.001, 0.001],
        [0.0, 1.0, 0.0, 1.0, 0.0],
        [0.1, 0.9, 0.0, 1.0, 0.25],
        [0.5, 0.5, 0.5, 0.5, 0.5],
    ]
    worst = max_normalized_bernoulli_entropy(probabilities)
    mean = normalized_bernoulli_entropy(probabilities)

    expected_worst = [max(_binary_entropy_bits(value) for value in row) for row in probabilities]
    expected_mean = [
        sum(_binary_entropy_bits(value) for value in row) / len(row) for row in probabilities
    ]
    assert worst.dtype == np.float64 and worst.shape == (4,)
    assert worst.tolist() == pytest.approx(expected_worst, abs=1e-12)
    assert mean.tolist() == pytest.approx(expected_mean, abs=1e-12)
    assert worst[0] == pytest.approx(1.0)
    assert mean[0] == pytest.approx((1.0 + 4.0 * _binary_entropy_bits(0.001)) / 5.0)
    assert mean[0] < 0.21
    assert worst[1] == 0.0
    assert worst[2] == pytest.approx(_binary_entropy_bits(0.25))


def test_max_entropy_is_the_exact_worst_per_label_value_in_unit_interval() -> None:
    probabilities = _entropy_fixture()
    worst = max_normalized_bernoulli_entropy(probabilities)
    per_label = np.column_stack(
        [normalized_bernoulli_entropy(probabilities[:, [label]]) for label in range(5)]
    )

    # The per-label value overshoots one bit by one ulp; the worst-label score is capped.
    assert per_label[50, 3] > 1.0
    assert np.array_equal(worst, np.minimum(per_label.max(axis=1), 1.0))
    assert np.all((worst >= 0.0) & (worst <= 1.0))
    assert np.all(worst >= normalized_bernoulli_entropy(probabilities) - 1e-15)
    assert worst[48] == 0.0
    assert worst[49] == pytest.approx(1.0)
    assert worst[50] == 1.0
    one_borderline_label = [[0.49999999999999983, 0.01, 0.99, 0.01, 0.01]]
    assert max_normalized_bernoulli_entropy(one_borderline_label).tolist() == [1.0]


def test_mean_entropy_output_is_bit_identical_to_the_pre_refactor_formula() -> None:
    probabilities = _entropy_fixture()
    observed = normalized_bernoulli_entropy(probabilities)

    assert observed.dtype == np.float64
    assert observed.tobytes() == _pre_refactor_mean_entropy(probabilities).tobytes()
    assert NormalizedBernoulliEntropyScorer().to_dict() == {
        "schema_version": 1,
        "artifact_type": "ecg_trust.normalized_bernoulli_entropy",
        "score_direction": "higher_is_more_out_of_distribution",
        "aggregation": "mean_across_labels",
    }


def test_worst_label_scorer_round_trips_and_refuses_mean_artifacts() -> None:
    worst = MaxNormalizedBernoulliEntropyScorer()
    probabilities = _entropy_fixture()
    payload = json.loads(json.dumps(worst.to_dict(), allow_nan=False))

    assert payload["artifact_type"] == "ecg_trust.max_normalized_bernoulli_entropy"
    assert payload["aggregation"] == "max_across_labels"
    assert MaxNormalizedBernoulliEntropyScorer.from_dict(payload) == worst
    assert np.array_equal(
        worst.score(probabilities), max_normalized_bernoulli_entropy(probabilities)
    )
    with pytest.raises(OODScoreValidationError, match="artifact_type"):
        MaxNormalizedBernoulliEntropyScorer.from_dict(NormalizedBernoulliEntropyScorer().to_dict())
    with pytest.raises(OODScoreValidationError, match="artifact_type"):
        NormalizedBernoulliEntropyScorer.from_dict(payload)
    relabeled = dict(payload)
    relabeled["aggregation"] = "mean_across_labels"
    with pytest.raises(OODScoreValidationError, match="aggregation"):
        MaxNormalizedBernoulliEntropyScorer.from_dict(relabeled)


def test_symmetric_energy_treats_confident_positive_and_negative_equally() -> None:
    scores = symmetric_binary_energy(
        [[-10.0, -10.0], [10.0, 10.0], [0.0, 0.0]],
        temperature=1.0,
    )

    assert scores[0] == pytest.approx(scores[1])
    assert scores[2] > scores[0]
    assert scores[2] == pytest.approx(-np.log(2.0))


def test_stateless_scorers_are_deterministic_and_json_round_trip() -> None:
    entropy = NormalizedBernoulliEntropyScorer()
    energy = SymmetricBinaryEnergyScorer(temperature=2.5)
    probabilities = np.asarray([[0.2, 0.8], [0.5, 0.5]])
    logits = np.asarray([[-2.0, 2.0], [0.0, 0.0]])

    assert np.array_equal(entropy.score(probabilities), entropy.score(probabilities))
    assert np.array_equal(energy.score(logits), energy.score(logits))
    entropy_payload = json.loads(json.dumps(entropy.to_dict(), allow_nan=False))
    energy_payload = json.loads(json.dumps(energy.to_dict(), allow_nan=False))
    assert NormalizedBernoulliEntropyScorer.from_dict(entropy_payload) == entropy
    assert SymmetricBinaryEnergyScorer.from_dict(energy_payload) == energy


@pytest.mark.parametrize(
    ("function", "values", "match"),
    [
        (normalized_bernoulli_entropy, [0.5, 0.5], "two-dimensional"),
        (normalized_bernoulli_entropy, [[0.5, np.nan]], "finite"),
        (normalized_bernoulli_entropy, [[-0.1, 0.5]], r"\[0, 1\]"),
        (max_normalized_bernoulli_entropy, [0.5, 0.5], "two-dimensional"),
        (max_normalized_bernoulli_entropy, [[0.5, np.nan]], "finite"),
        (max_normalized_bernoulli_entropy, [[0.5, 1.1]], r"\[0, 1\]"),
        (max_normalized_bernoulli_entropy, [[]], "samples and labels"),
        (symmetric_binary_energy, [[0.0, np.inf]], "finite"),
        (symmetric_binary_energy, [], "two-dimensional"),
    ],
)
def test_stateless_scores_reject_malformed_inputs(
    function: object, values: object, match: str
) -> None:
    callable_function = function
    with pytest.raises(OODScoreValidationError, match=match):
        callable_function(values)  # type: ignore[operator]


@pytest.mark.parametrize("temperature", [0.0, -1.0, np.inf, np.nan])
def test_energy_rejects_invalid_temperature(temperature: float) -> None:
    with pytest.raises(OODScoreValidationError, match="positive"):
        SymmetricBinaryEnergyScorer(temperature=temperature)


def _detector() -> tuple[ShrinkageMahalanobisDetector, np.ndarray, np.ndarray]:
    generator = np.random.default_rng(17)
    reference = generator.normal(0.0, 1.0, size=(300, 6))
    calibration = generator.normal(0.0, 1.0, size=(120, 6))
    detector = ShrinkageMahalanobisDetector.fit(
        reference,
        calibration,
        shrinkage=0.2,
        ridge=1e-5,
        inlier_coverage=0.9,
    )
    return detector, reference, calibration


def test_mahalanobis_detector_is_deterministic_frozen_and_directional() -> None:
    detector, reference, calibration = _detector()
    repeated = ShrinkageMahalanobisDetector.fit(
        reference,
        calibration,
        shrinkage=0.2,
        ridge=1e-5,
        inlier_coverage=0.9,
    )
    assert detector == repeated

    generator = np.random.default_rng(19)
    in_distribution = generator.normal(0.0, 1.0, size=(100, 6))
    shifted = generator.normal(6.0, 1.0, size=(100, 6))
    before = detector.to_dict()
    in_scores = detector.score(in_distribution)
    shifted_scores = detector.score(shifted)

    assert np.median(shifted_scores) > np.max(in_scores)
    assert detector.is_ood(shifted).mean() > 0.99
    assert detector.to_dict() == before
    with pytest.raises(FrozenInstanceError):
        detector.threshold = 0.0  # type: ignore[misc]


def test_mahalanobis_threshold_uses_finite_sample_corrected_quantile() -> None:
    detector, _, calibration = _detector()
    scores = detector.score(calibration)

    assert detector.quantile_rank == 109
    assert detector.threshold == pytest.approx(np.sort(scores)[108])
    assert np.mean(scores <= detector.threshold) >= detector.inlier_coverage
    assert np.all(detector.is_ood(calibration) == (scores > detector.threshold))


def test_mahalanobis_artifact_round_trip_is_exact_and_json_compatible() -> None:
    detector, _, _ = _detector()
    payload = json.loads(json.dumps(detector.to_dict(), allow_nan=False))
    restored = ShrinkageMahalanobisDetector.from_dict(payload)

    assert restored == detector
    probe = np.zeros((2, detector.embedding_dim), dtype=np.float64)
    assert np.array_equal(restored.score(probe), detector.score(probe))


def test_shrinkage_and_ridge_support_more_dimensions_than_reference_samples() -> None:
    generator = np.random.default_rng(23)
    reference = generator.normal(size=(5, 12))
    calibration = generator.normal(size=(20, 12))
    detector = ShrinkageMahalanobisDetector.fit(
        reference,
        calibration,
        shrinkage=0.25,
        ridge=1e-4,
        inlier_coverage=0.9,
    )

    scores = detector.score(generator.normal(size=(3, 12)))
    assert scores.shape == (3,)
    assert np.all(np.isfinite(scores))
    assert np.all(scores >= 0.0)


@pytest.mark.parametrize(
    ("reference", "calibration", "kwargs", "match"),
    [
        ([[0.0, 0.0]], [[0.0, 0.0]] * 20, {}, "at least two"),
        ([[0.0, 0.0], [1.0, 1.0]], [[0.0]] * 20, {}, "expected 2"),
        ([[0.0, np.nan], [1.0, 1.0]], [[0.0, 0.0]] * 20, {}, "finite"),
        ([[0.0, 0.0], [1.0, 1.0]], [[0.0, 0.0]] * 20, {"shrinkage": -0.1}, r"\[0, 1\]"),
        ([[0.0, 0.0], [1.0, 1.0]], [[0.0, 0.0]] * 20, {"ridge": 0.0}, "positive"),
        ([[0.0, 0.0], [1.0, 1.0]], [[0.0, 0.0]] * 5, {"inlier_coverage": 0.95}, "too small"),
    ],
)
def test_mahalanobis_fit_rejects_malformed_inputs(
    reference: object,
    calibration: object,
    kwargs: dict[str, float],
    match: str,
) -> None:
    with pytest.raises(MahalanobisValidationError, match=match):
        ShrinkageMahalanobisDetector.fit(reference, calibration, **kwargs)


def test_mahalanobis_score_rejects_dimension_and_nonfinite_values() -> None:
    detector, _, _ = _detector()
    with pytest.raises(MahalanobisValidationError, match="expected 6"):
        detector.score([[0.0, 0.0]])
    with pytest.raises(MahalanobisValidationError, match="finite"):
        detector.score([[0.0, 0.0, 0.0, 0.0, 0.0, np.inf]])


def test_serialized_detector_rejects_semantically_invalid_artifacts() -> None:
    detector, _, _ = _detector()
    payload = detector.to_dict()

    wrong_direction = dict(payload)
    wrong_direction["score_direction"] = "lower_is_more_out_of_distribution"
    with pytest.raises(MahalanobisValidationError, match="score_direction"):
        ShrinkageMahalanobisDetector.from_dict(wrong_direction)

    asymmetric = dict(payload)
    precision = [list(row) for row in payload["precision"]]  # type: ignore[union-attr]
    precision[0][1] += 1.0
    asymmetric["precision"] = precision
    with pytest.raises(MahalanobisValidationError, match="symmetric"):
        ShrinkageMahalanobisDetector.from_dict(asymmetric)

    bad_rank = dict(payload)
    bad_rank["quantile_rank"] = 1
    with pytest.raises(MahalanobisValidationError, match="quantile_rank"):
        ShrinkageMahalanobisDetector.from_dict(bad_rank)


@pytest.mark.parametrize("temperature", [1e-320, 1e-200, 1.0, 1e308])
def test_energy_remains_finite_for_extreme_valid_scales(temperature: float) -> None:
    logits = np.asarray([[1e308, -1e308, 1e308, -1e308]])
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        result = symmetric_binary_energy(logits, temperature=temperature)
    assert np.isfinite(result).all()
    if temperature < 1e307:
        assert result[0] == pytest.approx(-5e307)
    else:
        assert result[0] == pytest.approx(-temperature * np.logaddexp(-0.5, 0.5))


def test_energy_tiny_temperature_has_confident_limit_and_zero_limit() -> None:
    result = symmetric_binary_energy([[1.0, -1.0], [0.0, 0.0]], temperature=1e-320)
    assert result[0] == pytest.approx(-0.5)
    assert result[1] == -1e-320 * np.log(2.0)


@pytest.mark.parametrize(
    "function",
    [normalized_bernoulli_entropy, max_normalized_bernoulli_entropy, symmetric_binary_energy],
)
def test_scores_reject_complex_arrays_without_discarding_imaginary_parts(function: object) -> None:
    with pytest.raises(OODScoreValidationError, match="real"):
        function(np.asarray([[0.5 + 2j]]))


@pytest.mark.parametrize("temperature", [np.nextafter(0.0, 1.0), 1e-323, 1e-320])
def test_energy_preserves_ordering_at_subnormal_temperatures(temperature: float) -> None:
    result = symmetric_binary_energy([[0.0], [temperature]], temperature=temperature)
    expected = -temperature * np.logaddexp(-0.5, 0.5)
    assert result[1] == expected
    assert result[1] <= result[0] < 0.0
