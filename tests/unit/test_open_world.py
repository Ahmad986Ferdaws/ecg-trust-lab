from __future__ import annotations

import json
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from ecg_trust.open_world import (
    MahalanobisValidationError,
    NormalizedBernoulliEntropyScorer,
    OODScoreValidationError,
    ShrinkageMahalanobisDetector,
    SymmetricBinaryEnergyScorer,
    normalized_bernoulli_entropy,
    symmetric_binary_energy,
)


def test_normalized_entropy_has_documented_ood_direction_and_bounds() -> None:
    scores = normalized_bernoulli_entropy([[0.0, 1.0], [0.01, 0.99], [0.5, 0.5]])

    assert scores[0] == pytest.approx(0.0)
    assert 0.0 < scores[1] < scores[2]
    assert scores[2] == pytest.approx(1.0)


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
