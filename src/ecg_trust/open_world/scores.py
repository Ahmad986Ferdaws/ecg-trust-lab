"""Stateless uncertainty scores for multi-label sigmoid outputs.

Every score in this module follows one convention: **higher means more
uncertain / more OOD-like**. These are reproducible baselines, not calibrated
probabilities of being out of distribution.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]

_SCHEMA_VERSION = 1
_ENTROPY_TYPE = "ecg_trust.normalized_bernoulli_entropy"
_ENERGY_TYPE = "ecg_trust.symmetric_binary_energy"


class OODScoreValidationError(ValueError):
    """Raised when an OOD score input or scorer artifact is invalid."""


def normalized_bernoulli_entropy(probabilities: ArrayLike) -> FloatArray:
    """Return mean per-label Bernoulli entropy normalized to ``[0, 1]``.

    Zero is obtained when all sigmoid probabilities are zero or one; one is
    obtained when every probability is one half. Higher values are therefore
    treated as more uncertain/OOD-like.
    """

    matrix = _float_matrix(probabilities, context="probabilities")
    if np.any((matrix < 0.0) | (matrix > 1.0)):
        raise OODScoreValidationError("probabilities must lie in [0, 1]")
    entropy = np.zeros_like(matrix)
    interior = (matrix > 0.0) & (matrix < 1.0)
    selected = matrix[interior]
    entropy[interior] = -(
        selected * np.log(selected) + (1.0 - selected) * np.log1p(-selected)
    ) / math.log(2.0)
    return entropy.mean(axis=1)


def symmetric_binary_energy(logits: ArrayLike, *, temperature: float = 1.0) -> FloatArray:
    """Return a symmetric energy-style uncertainty score for sigmoid logits.

    Each binary logit ``z`` is represented by centered two-class logits
    ``[-z/2, z/2]``. Its energy is
    ``-T * logsumexp([-z/(2T), z/(2T)])`` and scores are averaged across labels.
    A logit of zero has the highest value ``-T*log(2)``; confident positive or
    negative logits are increasingly more negative. Thus, as elsewhere in this
    package, **higher values are more OOD-like**. This symmetric construction
    avoids treating confident all-negative multi-label predictions as OOD.
    """

    matrix = _float_matrix(logits, context="logits")
    scale = _positive_float(temperature, "temperature")
    centered = matrix / (2.0 * scale)
    per_label_energy = -scale * np.logaddexp(-centered, centered)
    return per_label_energy.mean(axis=1)


@dataclass(frozen=True, slots=True)
class NormalizedBernoulliEntropyScorer:
    """Serializable stateless normalized-entropy scorer."""

    def score(self, probabilities: ArrayLike) -> FloatArray:
        return normalized_bernoulli_entropy(probabilities)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "artifact_type": _ENTROPY_TYPE,
            "score_direction": "higher_is_more_out_of_distribution",
            "aggregation": "mean_across_labels",
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> NormalizedBernoulliEntropyScorer:
        _expect_scorer(
            payload,
            artifact_type=_ENTROPY_TYPE,
            expected={
                "schema_version",
                "artifact_type",
                "score_direction",
                "aggregation",
            },
        )
        return cls()


@dataclass(frozen=True, slots=True)
class SymmetricBinaryEnergyScorer:
    """Serializable symmetric binary-energy scorer."""

    temperature: float = 1.0

    def __post_init__(self) -> None:
        _positive_float(self.temperature, "temperature")

    def score(self, logits: ArrayLike) -> FloatArray:
        return symmetric_binary_energy(logits, temperature=self.temperature)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "artifact_type": _ENERGY_TYPE,
            "score_direction": "higher_is_more_out_of_distribution",
            "aggregation": "mean_across_labels",
            "temperature": self.temperature,
            "binary_logits": "centered_negative_and_positive",
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> SymmetricBinaryEnergyScorer:
        _expect_scorer(
            payload,
            artifact_type=_ENERGY_TYPE,
            expected={
                "schema_version",
                "artifact_type",
                "score_direction",
                "aggregation",
                "temperature",
                "binary_logits",
            },
        )
        if payload["binary_logits"] != "centered_negative_and_positive":
            raise OODScoreValidationError("unsupported binary_logits representation")
        return cls(temperature=_positive_float(payload["temperature"], "temperature"))


def _expect_scorer(
    payload: Mapping[str, object], *, artifact_type: str, expected: set[str]
) -> None:
    actual = set(payload)
    if actual != expected:
        raise OODScoreValidationError(
            f"scorer keys differ: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    if payload["schema_version"] != _SCHEMA_VERSION:
        raise OODScoreValidationError("unsupported scorer schema_version")
    if payload["artifact_type"] != artifact_type:
        raise OODScoreValidationError("unexpected scorer artifact_type")
    if payload["score_direction"] != "higher_is_more_out_of_distribution":
        raise OODScoreValidationError("unsupported score_direction")
    if payload["aggregation"] != "mean_across_labels":
        raise OODScoreValidationError("unsupported score aggregation")


def _float_matrix(values: ArrayLike, *, context: str) -> FloatArray:
    try:
        matrix = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise OODScoreValidationError(f"{context} must be numeric") from error
    if matrix.ndim != 2:
        raise OODScoreValidationError(f"{context} must be a two-dimensional matrix")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise OODScoreValidationError(f"{context} must contain samples and labels")
    if not np.all(np.isfinite(matrix)):
        raise OODScoreValidationError(f"{context} must contain only finite values")
    return matrix


def _positive_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OODScoreValidationError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise OODScoreValidationError(f"{name} must be finite and positive")
    return number
