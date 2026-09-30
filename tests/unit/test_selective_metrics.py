from __future__ import annotations

import numpy as np
import pytest

from ecg_trust.post_analysis import dense_risk_coverage
from ecg_trust.selective_metrics import (
    SelectiveMetricError,
    aurc,
    excess_aurc,
    oracle_aurc,
    selective_risk_curve,
)


def test_perfect_and_inverted_rankings_have_known_areas() -> None:
    losses = [0.0, 1.0]

    perfect = excess_aurc(losses, [0.1, 0.9])
    inverted = excess_aurc(losses, [0.9, 0.1])

    assert perfect.aurc == pytest.approx(0.25)
    assert perfect.oracle_aurc == pytest.approx(0.25)
    assert perfect.excess_aurc == pytest.approx(0.0)
    assert inverted.aurc == pytest.approx(0.75)
    assert inverted.excess_aurc == pytest.approx(0.5)
    assert inverted.to_dict()["count"] == 2


def test_ties_use_the_random_tie_break_expectation_and_ignore_input_order() -> None:
    losses = np.array([1.0, 0.0, 0.0, 1.0])
    uncertainty = np.array([0.5, 0.5, 0.2, 0.9])
    permutation = np.array([3, 1, 0, 2])

    curve = selective_risk_curve(losses, uncertainty)

    np.testing.assert_allclose(curve, [0.0, 0.25, 1.0 / 3.0, 0.5])
    assert aurc(losses[permutation], uncertainty[permutation]) == pytest.approx(
        aurc(losses, uncertainty)
    )
    assert aurc([1.0, 0.0], [0.3, 0.3]) == pytest.approx(0.5)


def test_constant_uncertainty_cannot_beat_the_oracle() -> None:
    rng = np.random.default_rng(7)
    losses = rng.uniform(size=50)

    result = excess_aurc(losses, np.zeros(50))

    assert result.aurc == pytest.approx(float(losses.mean()))
    assert result.oracle_aurc == pytest.approx(oracle_aurc(losses))
    assert result.excess_aurc > 0.0


def test_matches_existing_dense_risk_coverage_without_ties() -> None:
    rng = np.random.default_rng(11)
    targets = rng.integers(0, 2, size=(40, 5))
    probabilities = rng.uniform(size=(40, 5))
    uncertainty = rng.permutation(40).astype(np.float64)
    thresholds = [0.5] * 5
    exact_error = np.any((probabilities >= 0.5) != targets.astype(bool), axis=1)

    reference = dense_risk_coverage(
        targets,
        probabilities,
        thresholds=thresholds,
        uncertainty=uncertainty,
    )

    assert aurc(exact_error.astype(np.float64), uncertainty) == pytest.approx(
        reference.aurc_exact_match_error
    )


@pytest.mark.parametrize(
    ("losses", "uncertainty", "message"),
    [
        ([], [], "non-empty"),
        ([[0.0]], [[0.0]], "one-dimensional"),
        ([0.0, 1.0], [0.1], "2 entries"),
        ([0.0, -1.0], [0.1, 0.2], "non-negative"),
        ([0.0, np.nan], [0.1, 0.2], "finite"),
        ([0.0, 1.0], [0.1, np.inf], "finite"),
        ([True, False], [0.1, 0.2], "real numeric"),
        ([0.0, 1.0], [0.1 + 1j, 0.2], "real-valued"),
        (["a", "b"], [0.1, 0.2], "real-valued"),
        (np.array([0.0, 1.0], dtype=object), [0.1, 0.2], "real numeric"),
    ],
)
def test_malformed_inputs_are_rejected(losses: object, uncertainty: object, message: str) -> None:
    with pytest.raises(SelectiveMetricError, match=message):
        excess_aurc(losses, uncertainty)  # type: ignore[arg-type]
