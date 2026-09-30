from __future__ import annotations

import numpy as np
import pytest

from ecg_trust.brier_decomposition import BrierDecompositionError, brier_decomposition
from ecg_trust.evaluation import EvaluationValidationError
from ecg_trust.protocol import LABEL_ORDER


@pytest.mark.parametrize("n_bins", [1, 3, 10, 500])
def test_five_terms_reconstruct_the_brier_score_exactly(n_bins: int) -> None:
    rng = np.random.default_rng(9)
    probabilities = rng.uniform(size=(200, 5))
    targets = (rng.uniform(size=(200, 5)) < probabilities**1.5).astype(np.int64)

    components = brier_decomposition(targets, probabilities, n_bins=n_bins)

    assert [item.label for item in components] == list(LABEL_ORDER)
    for index, item in enumerate(components):
        assert item.reconstructed == pytest.approx(item.brier, abs=1e-12)
        assert item.brier == pytest.approx(
            float(np.mean((probabilities[:, index] - targets[:, index]) ** 2))
        )
        assert item.bins <= min(n_bins, 200)


def test_tied_forecasts_share_a_bin_regardless_of_row_order() -> None:
    grouped = np.tile(np.array([[0]] * 5 + [[1]] * 5), (1, 5))
    interleaved = np.tile(np.array([[0], [1]] * 5), (1, 5))
    probabilities = np.full((10, 5), 0.5)

    first = brier_decomposition(grouped, probabilities, n_bins=2)[0]
    second = brier_decomposition(interleaved, probabilities, n_bins=2)[0]

    assert first.bins == second.bins == 1
    assert first.resolution == pytest.approx(0.0)
    assert first.reliability == pytest.approx(second.reliability) == pytest.approx(0.0)


def test_within_bin_covariance_is_the_population_covariance() -> None:
    targets = np.tile(np.array([[0], [1], [0], [1]]), (1, 5))
    probabilities = np.tile(np.array([[0.1], [0.3], [0.6], [0.8]]), (1, 5))

    item = brier_decomposition(targets, probabilities, n_bins=1)[0]
    expected = float(np.mean((targets[:, 0] - 0.5) * (probabilities[:, 0] - 0.45)))

    assert item.within_bin_covariance == pytest.approx(expected)
    assert item.reconstructed == pytest.approx(item.brier, abs=1e-12)


def test_constant_forecast_at_the_base_rate_has_only_uncertainty() -> None:
    targets = np.zeros((10, 5), dtype=np.int64)
    targets[:3] = 1
    probabilities = np.full((10, 5), 0.3)

    item = brier_decomposition(targets, probabilities, n_bins=1)[0]

    assert item.reliability == pytest.approx(0.0)
    assert item.resolution == pytest.approx(0.0)
    assert item.within_bin_variance == pytest.approx(0.0)
    assert item.uncertainty == pytest.approx(0.21)
    assert item.brier == pytest.approx(0.21)


def test_perfect_confident_forecast_has_full_resolution_and_no_error() -> None:
    targets = np.tile(np.array([[0], [0], [1], [1]]), (1, 5))
    probabilities = targets.astype(np.float64)

    item = brier_decomposition(targets, probabilities, n_bins=2)[0]

    assert item.brier == pytest.approx(0.0)
    assert item.resolution == pytest.approx(item.uncertainty)
    assert item.reliability == pytest.approx(0.0)
    assert item.to_dict()["bins"] == 2


@pytest.mark.parametrize("n_bins", [0, -3, True, 2.5])
def test_invalid_bin_count_is_rejected(n_bins: object) -> None:
    with pytest.raises(BrierDecompositionError):
        brier_decomposition(np.zeros((2, 5)), np.zeros((2, 5)), n_bins=n_bins)  # type: ignore[arg-type]


def test_inputs_use_the_evaluation_contract() -> None:
    with pytest.raises(EvaluationValidationError):
        brier_decomposition(np.full((2, 5), 2), np.zeros((2, 5)))
