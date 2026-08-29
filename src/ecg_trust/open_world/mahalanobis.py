"""Frozen shrinkage-Mahalanobis detector for ECG embedding shift.

The detector is fitted exactly once from development/reference embeddings and
a separate source-domain threshold-calibration split. Evaluation-site data are
accepted only by :meth:`score` and :meth:`is_ood`; they never update the frozen
mean, precision matrix, or threshold.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

_SCHEMA_VERSION = 1
_ARTIFACT_TYPE = "ecg_trust.shrinkage_mahalanobis_detector"
_SCORE_DIRECTION = "higher_is_more_out_of_distribution"
_FIT_SCOPE = "development_reference_plus_source_calibration_only"


class MahalanobisValidationError(ValueError):
    """Raised when detector fitting, scoring, or deserialization is invalid."""


@dataclass(frozen=True, slots=True)
class ShrinkageMahalanobisDetector:
    """Serializable one-class embedding detector with a frozen OOD threshold."""

    mean: tuple[float, ...]
    precision: tuple[tuple[float, ...], ...]
    threshold: float
    embedding_dim: int
    shrinkage: float
    ridge: float
    inlier_coverage: float
    n_fit_samples: int
    n_threshold_samples: int
    quantile_rank: int

    @classmethod
    def fit(
        cls,
        reference_embeddings: ArrayLike,
        threshold_calibration_embeddings: ArrayLike,
        *,
        shrinkage: float = 0.1,
        ridge: float = 1e-6,
        inlier_coverage: float = 0.95,
    ) -> ShrinkageMahalanobisDetector:
        """Fit only from development and source-domain calibration embeddings.

        ``threshold_calibration_embeddings`` should be a patient-disjoint source
        split. Passing target-site embeddings would invalidate external/OOD
        evaluation and is deliberately outside this API's stated contract.
        """

        reference = _embedding_matrix(reference_embeddings, context="reference embeddings")
        if reference.shape[0] < 2:
            raise MahalanobisValidationError("reference embeddings require at least two samples")
        calibration = _embedding_matrix(
            threshold_calibration_embeddings,
            context="threshold calibration embeddings",
            expected_dim=reference.shape[1],
        )
        shrinkage_value = _closed_unit_float(shrinkage, "shrinkage")
        ridge_value = _positive_float(ridge, "ridge")
        coverage = _open_unit_float(inlier_coverage, "inlier_coverage")
        rank = math.ceil((calibration.shape[0] + 1) * coverage)
        if rank > calibration.shape[0]:
            raise MahalanobisValidationError(
                "threshold calibration sample count is too small for the requested "
                "finite-sample inlier_coverage"
            )

        mean = reference.mean(axis=0)
        centered = reference - mean
        covariance = centered.T @ centered / (reference.shape[0] - 1)
        dimension = reference.shape[1]
        isotropic_scale = float(np.trace(covariance) / dimension)
        target = np.eye(dimension, dtype=np.float64) * isotropic_scale
        regularized = (1.0 - shrinkage_value) * covariance + shrinkage_value * target
        regularized = regularized + np.eye(dimension, dtype=np.float64) * ridge_value
        eigenvalues, eigenvectors = np.linalg.eigh(regularized)
        if not np.all(np.isfinite(eigenvalues)) or np.any(eigenvalues <= 0.0):
            raise MahalanobisValidationError("regularized covariance is not positive definite")
        precision = (eigenvectors * (1.0 / eigenvalues)) @ eigenvectors.T
        precision = (precision + precision.T) / 2.0

        calibration_scores = _squared_distances(calibration, mean, precision)
        threshold = float(np.sort(calibration_scores)[rank - 1])
        if not math.isfinite(threshold):
            raise MahalanobisValidationError("fitted threshold is not finite")

        return cls(
            mean=tuple(float(value) for value in mean),
            precision=tuple(tuple(float(value) for value in row) for row in precision),
            threshold=threshold,
            embedding_dim=dimension,
            shrinkage=shrinkage_value,
            ridge=ridge_value,
            inlier_coverage=coverage,
            n_fit_samples=reference.shape[0],
            n_threshold_samples=calibration.shape[0],
            quantile_rank=rank,
        )

    def score(self, embeddings: ArrayLike) -> FloatArray:
        """Score without changing the frozen fitted artifact; higher is more OOD."""

        matrix = _embedding_matrix(
            embeddings,
            context="embeddings",
            expected_dim=self.embedding_dim,
        )
        return _squared_distances(
            matrix,
            np.asarray(self.mean, dtype=np.float64),
            np.asarray(self.precision, dtype=np.float64),
        )

    def is_ood(self, embeddings: ArrayLike) -> BoolArray:
        """Flag scores strictly above the frozen source-calibration threshold."""

        return self.score(embeddings) > self.threshold

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible fitted detector artifact."""

        return {
            "schema_version": _SCHEMA_VERSION,
            "artifact_type": _ARTIFACT_TYPE,
            "score_direction": _SCORE_DIRECTION,
            "fit_scope": _FIT_SCOPE,
            "mean": list(self.mean),
            "precision": [list(row) for row in self.precision],
            "threshold": self.threshold,
            "embedding_dim": self.embedding_dim,
            "shrinkage": self.shrinkage,
            "ridge": self.ridge,
            "inlier_coverage": self.inlier_coverage,
            "n_fit_samples": self.n_fit_samples,
            "n_threshold_samples": self.n_threshold_samples,
            "quantile_rank": self.quantile_rank,
            "threshold_rule": "ceil((n+1)*inlier_coverage)_order_statistic",
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> ShrinkageMahalanobisDetector:
        """Strictly validate and restore a fitted detector artifact."""

        expected = {
            "schema_version",
            "artifact_type",
            "score_direction",
            "fit_scope",
            "mean",
            "precision",
            "threshold",
            "embedding_dim",
            "shrinkage",
            "ridge",
            "inlier_coverage",
            "n_fit_samples",
            "n_threshold_samples",
            "quantile_rank",
            "threshold_rule",
        }
        actual = set(payload)
        if actual != expected:
            raise MahalanobisValidationError(
                f"detector keys differ: missing={sorted(expected - actual)}, "
                f"extra={sorted(actual - expected)}"
            )
        if payload["schema_version"] != _SCHEMA_VERSION:
            raise MahalanobisValidationError("unsupported detector schema_version")
        if payload["artifact_type"] != _ARTIFACT_TYPE:
            raise MahalanobisValidationError("unexpected detector artifact_type")
        if payload["score_direction"] != _SCORE_DIRECTION:
            raise MahalanobisValidationError("unsupported detector score_direction")
        if payload["fit_scope"] != _FIT_SCOPE:
            raise MahalanobisValidationError("unsupported detector fit_scope")
        if payload["threshold_rule"] != "ceil((n+1)*inlier_coverage)_order_statistic":
            raise MahalanobisValidationError("unsupported threshold_rule")

        dimension = _positive_integer(payload["embedding_dim"], "embedding_dim")
        mean = _float_tuple(payload["mean"], "mean", expected_length=dimension)
        precision = _square_float_tuple(payload["precision"], dimension=dimension)
        precision_array = np.asarray(precision, dtype=np.float64)
        if not np.allclose(precision_array, precision_array.T, rtol=0.0, atol=1e-10):
            raise MahalanobisValidationError("precision matrix must be symmetric")
        eigenvalues = np.linalg.eigvalsh(precision_array)
        if np.any(eigenvalues <= 0.0):
            raise MahalanobisValidationError("precision matrix must be positive definite")

        coverage = _open_unit_float(payload["inlier_coverage"], "inlier_coverage")
        n_threshold = _positive_integer(payload["n_threshold_samples"], "n_threshold_samples")
        rank = _positive_integer(payload["quantile_rank"], "quantile_rank")
        expected_rank = math.ceil((n_threshold + 1) * coverage)
        if rank != expected_rank or rank > n_threshold:
            raise MahalanobisValidationError(
                "quantile_rank does not match the finite-sample threshold rule"
            )
        n_fit = _positive_integer(payload["n_fit_samples"], "n_fit_samples")
        if n_fit < 2:
            raise MahalanobisValidationError("n_fit_samples must be at least two")

        return cls(
            mean=mean,
            precision=precision,
            threshold=_nonnegative_float(payload["threshold"], "threshold"),
            embedding_dim=dimension,
            shrinkage=_closed_unit_float(payload["shrinkage"], "shrinkage"),
            ridge=_positive_float(payload["ridge"], "ridge"),
            inlier_coverage=coverage,
            n_fit_samples=n_fit,
            n_threshold_samples=n_threshold,
            quantile_rank=rank,
        )


def _squared_distances(
    embeddings: FloatArray,
    mean: FloatArray,
    precision: FloatArray,
) -> FloatArray:
    centered = embeddings - mean[None, :]
    distances = np.einsum("ni,ij,nj->n", centered, precision, centered)
    distances = np.maximum(distances, 0.0)
    if not np.all(np.isfinite(distances)):
        raise MahalanobisValidationError("Mahalanobis scores are not finite")
    return cast(FloatArray, distances)


def _embedding_matrix(
    values: ArrayLike,
    *,
    context: str,
    expected_dim: int | None = None,
) -> FloatArray:
    try:
        matrix = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise MahalanobisValidationError(f"{context} must be numeric") from error
    if matrix.ndim != 2:
        raise MahalanobisValidationError(f"{context} must be a two-dimensional matrix")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise MahalanobisValidationError(f"{context} must contain samples and features")
    if expected_dim is not None and matrix.shape[1] != expected_dim:
        raise MahalanobisValidationError(
            f"{context} has dimension {matrix.shape[1]}; expected {expected_dim}"
        )
    if not np.all(np.isfinite(matrix)):
        raise MahalanobisValidationError(f"{context} must contain only finite values")
    return matrix


def _closed_unit_float(value: object, name: str) -> float:
    number = _finite_float(value, name)
    if not 0.0 <= number <= 1.0:
        raise MahalanobisValidationError(f"{name} must lie in [0, 1]")
    return number


def _open_unit_float(value: object, name: str) -> float:
    number = _finite_float(value, name)
    if not 0.0 < number < 1.0:
        raise MahalanobisValidationError(f"{name} must lie strictly between zero and one")
    return number


def _positive_float(value: object, name: str) -> float:
    number = _finite_float(value, name)
    if number <= 0.0:
        raise MahalanobisValidationError(f"{name} must be positive")
    return number


def _nonnegative_float(value: object, name: str) -> float:
    number = _finite_float(value, name)
    if number < 0.0:
        raise MahalanobisValidationError(f"{name} must be nonnegative")
    return number


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MahalanobisValidationError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise MahalanobisValidationError(f"{name} must be finite")
    return number


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise MahalanobisValidationError(f"{name} must be a positive integer")
    return value


def _float_tuple(value: object, name: str, *, expected_length: int) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise MahalanobisValidationError(f"{name} must be a sequence")
    result = tuple(_finite_float(item, name) for item in value)
    if len(result) != expected_length:
        raise MahalanobisValidationError(f"{name} must contain {expected_length} values")
    return result


def _square_float_tuple(value: object, *, dimension: int) -> tuple[tuple[float, ...], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise MahalanobisValidationError("precision must be a sequence of rows")
    rows = tuple(_float_tuple(row, "precision row", expected_length=dimension) for row in value)
    if len(rows) != dimension:
        raise MahalanobisValidationError(f"precision must contain {dimension} rows")
    return rows
