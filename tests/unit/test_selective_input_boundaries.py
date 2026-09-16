from __future__ import annotations

import numpy as np
import pytest

from ecg_trust.evaluation import EvaluationValidationError, compute_selective_predictions


def _select(**kwargs: object) -> object:
    options = dict(thresholds=(0.5,) * 5, coverage_targets=(0.5,))
    options.update(kwargs)
    return compute_selective_predictions(np.zeros((2, 5)), np.zeros((2, 5)), **options)


@pytest.mark.parametrize("object_dtype", [False, True])
def test_selective_ranking_rejects_complex_uncertainty(object_dtype: bool) -> None:
    uncertainty = np.array(
        [np.complex128(1 + 3j), np.complex128(0 + 4j)],
        dtype=object if object_dtype else complex,
    )
    with pytest.raises(EvaluationValidationError):
        _select(uncertainty=uncertainty)


@pytest.mark.parametrize("invalid", [True, np.bool_(False), np.complex128(0.5 + 2j), 10**1000],
                         ids=["bool", "numpy-bool", "complex", "overflow"])
def test_selective_controls_reject_lossy_conversion(invalid: object) -> None:
    with pytest.raises(EvaluationValidationError):
        _select(coverage_targets=(invalid,))
    with pytest.raises(EvaluationValidationError):
        _select(thresholds=(invalid,) * 5)


def test_uncertainty_overflow_is_a_domain_error() -> None:
    with pytest.raises(EvaluationValidationError):
        _select(uncertainty=[10**1000, 0])


def test_real_uncertainty_ranking_preserves_stable_ties_and_coverage() -> None:
    result = compute_selective_predictions(
        np.zeros((3, 5)), np.zeros((3, 5)), thresholds=(0.5,) * 5,
        coverage_targets=(0.5, 1.0), uncertainty=[0.2, 0.1, 0.1],
    )
    assert result.coverage_points[0].selected_indices == (1, 2)
    assert result.coverage_points[0].achieved_coverage == 2 / 3
