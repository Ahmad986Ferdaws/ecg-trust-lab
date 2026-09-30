from __future__ import annotations

import numpy as np
import pytest

from ecg_trust.selective_metrics import (
    SelectiveMetricError,
    augrc,
    generalized_risk_curve,
    selective_risk_curve,
)


def test_generalized_risk_divides_accepted_loss_by_all_cases() -> None:
    losses = [0.0, 1.0, 1.0, 0.0]
    uncertainty = [0.1, 0.2, 0.3, 0.4]

    curve = generalized_risk_curve(losses, uncertainty)

    np.testing.assert_allclose(curve, [0.0, 0.25, 0.5, 0.5])
    np.testing.assert_allclose(
        curve, selective_risk_curve(losses, uncertainty) * np.arange(1, 5) / 4
    )
    assert augrc(losses, uncertainty) == pytest.approx(0.3125)


def test_perfect_ranking_and_constant_scores_have_closed_forms() -> None:
    count, errors = 100, 20
    losses = np.r_[np.zeros(count - errors), np.ones(errors)]

    perfect = augrc(losses, np.arange(count, dtype=np.float64))
    constant = augrc(losses, np.zeros(count))

    expected_perfect = (
        float(np.mean(np.maximum(np.arange(1, count + 1) - (count - errors), 0))) / count
    )
    expected_constant = float(np.mean(np.arange(1, count + 1) / count)) * errors / count
    assert perfect == pytest.approx(expected_perfect)
    assert constant == pytest.approx(expected_constant)
    assert perfect < constant


def test_augrc_is_order_invariant_and_bounded_by_the_error_rate() -> None:
    rng = np.random.default_rng(13)
    losses = rng.integers(0, 2, size=200).astype(np.float64)
    uncertainty = rng.integers(0, 10, size=200).astype(np.float64)
    order = rng.permutation(200)

    value = augrc(losses, uncertainty)

    assert value == pytest.approx(augrc(losses[order], uncertainty[order]))
    assert 0.0 <= value <= losses.mean()


def test_augrc_reuses_the_selective_metric_validation() -> None:
    with pytest.raises(SelectiveMetricError, match="non-negative"):
        augrc([-1.0, 0.0], [0.1, 0.2])
    with pytest.raises(SelectiveMetricError, match="real numeric"):
        augrc([True, 0.0], [0.1, 0.2])
