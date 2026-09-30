from __future__ import annotations

import math

import numpy as np
import pytest

from ecg_trust.ensemble_uncertainty import (
    EnsembleUncertaintyError,
    decompose_ensemble_uncertainty,
)
from ecg_trust.post_analysis import mean_normalized_binary_entropy


def test_agreeing_members_have_no_epistemic_uncertainty() -> None:
    members = np.full((3, 4, 5), 0.5)

    result = decompose_ensemble_uncertainty(members)

    np.testing.assert_allclose(result.total, 1.0)
    np.testing.assert_allclose(result.aleatoric, 1.0)
    np.testing.assert_allclose(result.epistemic, 0.0, atol=1e-12)
    np.testing.assert_allclose(result.member_variance, 0.0)
    assert result.members == 3


def test_confident_disagreement_is_almost_entirely_epistemic() -> None:
    members = np.stack((np.full((1, 5), 1.0), np.full((1, 5), 0.0)))

    result = decompose_ensemble_uncertainty(members)

    np.testing.assert_allclose(result.total, 1.0)
    np.testing.assert_allclose(result.aleatoric, 0.0, atol=1e-12)
    np.testing.assert_allclose(result.epistemic, 1.0, atol=1e-12)
    np.testing.assert_allclose(result.member_variance, 0.25)


def test_components_are_consistent_and_total_matches_the_frozen_entropy_score() -> None:
    rng = np.random.default_rng(5)
    members = rng.uniform(size=(6, 30, 5))

    result = decompose_ensemble_uncertainty(members)

    np.testing.assert_allclose(result.total, result.aleatoric + result.epistemic, atol=1e-12)
    assert np.all(result.epistemic >= 0.0)
    np.testing.assert_allclose(
        result.per_record("total"), mean_normalized_binary_entropy(members.mean(axis=0))
    )
    np.testing.assert_allclose(
        result.per_record("epistemic", reduction="max"), result.epistemic.max(axis=1)
    )


def test_known_mutual_information_value() -> None:
    members = np.array([[[0.9]], [[0.1]]])

    result = decompose_ensemble_uncertainty(members)
    entropy_09 = -(0.9 * math.log2(0.9) + 0.1 * math.log2(0.1))

    assert float(result.epistemic[0, 0]) == pytest.approx(1.0 - entropy_09)


@pytest.mark.parametrize(
    ("values", "message"),
    [
        (np.zeros((1, 3, 5)), "at least two members"),
        (np.zeros((3, 5)), "shape"),
        (np.zeros((2, 0, 5)), "shape"),
        (np.full((2, 3, 5), 1.5), r"\[0, 1\]"),
        (np.full((2, 3, 5), np.nan), "finite"),
        (np.zeros((2, 3, 5), dtype=bool), "real numeric"),
        (np.zeros((2, 3, 5), dtype=complex), "real-valued"),
        (np.zeros((2, 3, 5), dtype=object), "real numeric"),
    ],
)
def test_malformed_member_probabilities_are_rejected(values: np.ndarray, message: str) -> None:
    with pytest.raises(EnsembleUncertaintyError, match=message):
        decompose_ensemble_uncertainty(values)


def test_unknown_component_and_reduction_are_rejected() -> None:
    result = decompose_ensemble_uncertainty(np.full((2, 1, 5), 0.3))

    with pytest.raises(EnsembleUncertaintyError, match="component"):
        result.per_record("variance")
    with pytest.raises(EnsembleUncertaintyError, match="reduction"):
        result.per_record("total", reduction="median")
