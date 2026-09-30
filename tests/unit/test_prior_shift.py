from __future__ import annotations

import numpy as np
import pytest

from ecg_trust.prior_shift import PriorShiftError, adjust_to_priors, estimate_label_shift


def _calibrated_scores(
    prior_source: float, prior_target: float, count: int, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    labels = rng.uniform(size=count) < prior_target
    features = rng.normal(np.where(labels, 1.0, -1.0), 1.0)
    likelihood_ratio = np.exp(2.0 * features)
    odds = likelihood_ratio * prior_source / (1.0 - prior_source)
    return odds / (1.0 + odds)


def test_em_recovers_a_shifted_prevalence_per_label() -> None:
    scores = np.stack(
        (
            _calibrated_scores(0.3, 0.1, 20_000, 1),
            _calibrated_scores(0.5, 0.7, 20_000, 2),
        ),
        axis=1,
    )

    result = estimate_label_shift(scores, source_priors=(0.3, 0.5))

    assert [item.target_prior for item in result.labels] == pytest.approx([0.1, 0.7], abs=0.015)
    assert all(item.converged for item in result.labels)
    assert result.adjusted_probabilities.shape == scores.shape
    assert result.adjusted_probabilities[:, 0].mean() == pytest.approx(
        result.labels[0].target_prior, abs=1e-6
    )


def test_no_shift_leaves_calibrated_scores_nearly_unchanged() -> None:
    scores = _calibrated_scores(0.4, 0.4, 20_000, 3)[:, None]

    result = estimate_label_shift(scores, source_priors=(0.4,))

    assert result.labels[0].target_prior == pytest.approx(0.4, abs=0.015)
    np.testing.assert_allclose(result.adjusted_probabilities, scores, atol=0.03)


def test_known_prior_adjustment_matches_bayes_rule() -> None:
    probabilities = np.array([[0.2], [0.5], [0.9], [0.0], [1.0]])

    adjusted = adjust_to_priors(probabilities, source_priors=(0.5,), target_priors=(0.2,))

    odds = probabilities / (1 - np.where(probabilities < 1, probabilities, np.nan)) * 0.25
    expected = odds / (1 + odds)
    np.testing.assert_allclose(adjusted[:3], expected[:3])
    assert adjusted[3, 0] == 0.0
    assert adjusted[4, 0] == 1.0
    np.testing.assert_allclose(
        adjust_to_priors(probabilities, source_priors=(0.3,), target_priors=(0.3,)), probabilities
    )


def test_subnormal_priors_do_not_produce_nans() -> None:
    probabilities = np.array([[0.0], [0.2], [0.9], [1.0]])

    adjusted = adjust_to_priors(probabilities, source_priors=(1e-310,), target_priors=(0.5,))
    estimated = estimate_label_shift(probabilities, source_priors=(1e-310,))

    assert np.isfinite(adjusted).all()
    assert adjusted[0, 0] == 0.0
    assert adjusted[3, 0] == 1.0
    assert np.isfinite(estimated.adjusted_probabilities).all()
    assert np.isfinite(estimated.labels[0].target_prior)


def test_iteration_budget_is_reported() -> None:
    scores = _calibrated_scores(0.3, 0.05, 2_000, 4)[:, None]

    result = estimate_label_shift(scores, source_priors=(0.3,), max_iterations=2)

    assert result.labels[0].iterations == 2
    assert not result.labels[0].converged
    assert result.to_dict()["labels"][0]["converged"] is False  # type: ignore[index]


@pytest.mark.parametrize(
    ("probabilities", "kwargs", "message"),
    [
        (np.zeros((0, 2)), {"source_priors": (0.5, 0.5)}, "non-empty"),
        (np.zeros(3), {"source_priors": (0.5,)}, "non-empty"),
        (np.full((2, 1), 1.5), {"source_priors": (0.5,)}, r"\[0, 1\]"),
        (np.full((2, 1), np.nan), {"source_priors": (0.5,)}, "finite"),
        (np.zeros((2, 1), dtype=bool), {"source_priors": (0.5,)}, "real numeric"),
        (np.zeros((2, 1), dtype=complex), {"source_priors": (0.5,)}, "real-valued"),
        (np.zeros((2, 2)), {"source_priors": (0.5,)}, "one prior per label"),
        (np.zeros((2, 1)), {"source_priors": (1.0,)}, "strictly between"),
        (np.zeros((2, 1)), {"source_priors": (True,)}, "strictly between"),
        (np.zeros((2, 1)), {"source_priors": (0.5,), "tolerance": 0.0}, "tolerance"),
        (np.zeros((2, 1)), {"source_priors": (0.5,), "max_iterations": 0}, "max_iterations"),
    ],
)
def test_malformed_inputs_are_rejected(
    probabilities: np.ndarray, kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(PriorShiftError, match=message):
        estimate_label_shift(probabilities, **kwargs)  # type: ignore[arg-type]
