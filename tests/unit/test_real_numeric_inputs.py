from __future__ import annotations

import numpy as np
import pytest

from ecg_trust.conformal import ConformalValidationError, LabelwiseBinaryConformal
from ecg_trust.evaluation import (
    EvaluationValidationError,
    fixed_bin_ece,
    stable_sigmoid,
    validate_logits,
    validate_multilabel_arrays,
)


@pytest.mark.parametrize("imaginary", [0.0, 3.0])
def test_evaluation_rejects_complex_scores_and_logits(imaginary: float) -> None:
    scores = np.full((2, 5), 0.5 + imaginary * 1j)
    targets = np.zeros((2, 5))
    with pytest.raises(EvaluationValidationError):
        validate_multilabel_arrays(targets, scores)
    with pytest.raises(EvaluationValidationError):
        validate_logits(scores)
    with pytest.raises(EvaluationValidationError):
        stable_sigmoid(scores)
    with pytest.raises(EvaluationValidationError):
        fixed_bin_ece(targets[:, 0], scores[:, 0])


def test_evaluation_rejects_complex_binary_targets() -> None:
    targets = np.asarray([[0, 1, 0, 1, 0]], dtype=complex)
    with pytest.raises(EvaluationValidationError):
        validate_multilabel_arrays(targets, np.full((1, 5), 0.5))
    with pytest.raises(EvaluationValidationError):
        fixed_bin_ece(targets[0], np.full(5, 0.5))


@pytest.mark.parametrize("invalid", [10**1000, "not a number"], ids=["overflow", "text"])
def test_evaluation_conversion_failures_use_domain_errors(invalid: object) -> None:
    with pytest.raises(EvaluationValidationError):
        stable_sigmoid([invalid])
    with pytest.raises(EvaluationValidationError):
        fixed_bin_ece([0], [invalid])
    with pytest.raises(EvaluationValidationError):
        validate_logits([[invalid] * 5])


@pytest.mark.parametrize("imaginary", [0.0, 3.0])
def test_conformal_rejects_complex_fit_and_prediction_inputs(imaginary: float) -> None:
    scores = np.full((4, 1), 0.5 + imaginary * 1j)
    targets = np.ones((4, 1))
    with pytest.raises(ConformalValidationError):
        LabelwiseBinaryConformal.fit(scores, targets, label_names=("label",), alpha=0.5)
    with pytest.raises(ConformalValidationError):
        LabelwiseBinaryConformal.fit(
            scores.real, targets.astype(complex), label_names=("label",), alpha=0.5
        )
    calibrator = LabelwiseBinaryConformal.fit(
        scores.real, targets, label_names=("label",), alpha=0.5
    )
    with pytest.raises(ConformalValidationError):
        calibrator.predict(scores)


def test_conformal_conversion_overflow_uses_domain_error() -> None:
    with pytest.raises(ConformalValidationError):
        LabelwiseBinaryConformal.fit([[10**1000]], [[1]], label_names=("label",), alpha=0.5)


def test_object_arrays_cannot_hide_numpy_complex_scalars() -> None:
    scores = np.array([np.complex128(0.5 + 3j)] * 20, dtype=object).reshape(4, 5)
    with pytest.raises(EvaluationValidationError):
        stable_sigmoid(scores)
    with pytest.raises(EvaluationValidationError):
        validate_multilabel_arrays(np.zeros((4, 5)), scores)
    with pytest.raises(EvaluationValidationError):
        validate_logits(scores)
    with pytest.raises(EvaluationValidationError):
        fixed_bin_ece(np.zeros(4), scores[:, 0])
    with pytest.raises(ConformalValidationError):
        LabelwiseBinaryConformal.fit(scores[:, :1], [[1]] * 4, label_names=("label",), alpha=0.5)
