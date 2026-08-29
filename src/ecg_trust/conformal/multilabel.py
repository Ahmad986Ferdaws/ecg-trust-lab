"""Label-wise split-conformal sets for multi-label sigmoid classifiers.

Each diagnostic label is treated as its own binary prediction problem. For a
probability ``p`` the nonconformity scores are ``p`` for the negative outcome
and ``1 - p`` for the positive outcome. Calibration is performed separately
for every label, so the finite-sample guarantee is *label-wise marginal*
coverage under exchangeability; it is not simultaneous coverage of every label
for a patient.

The finite-sample order statistic is ``ceil((n + 1) * (1 - alpha))``. When the
requested rank is ``n + 1``, the bounded binary score permits a threshold of
one, yielding the conservative set ``{0, 1}`` for every probability.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int8]

_ARTIFACT_TYPE = "ecg_trust.labelwise_binary_conformal"
_PREDICTION_TYPE = "ecg_trust.binary_prediction_sets"
_METRICS_TYPE = "ecg_trust.conformal_metrics"
_SCHEMA_VERSION = 1


class ConformalValidationError(ValueError):
    """Raised when a conformal input or serialized artifact is invalid."""


class BinaryDecision(StrEnum):
    """Human-readable decision derived from a binary prediction set."""

    SUPPORTED = "supported"
    NOT_SUPPORTED = "not_supported"
    UNCERTAIN = "uncertain"


class UncertaintyKind(StrEnum):
    """Reason an output is uncertain rather than a singleton decision."""

    BOTH = "both"
    EMPTY = "empty"


@dataclass(frozen=True, slots=True)
class LabelwiseBinaryConformal:
    """Frozen label-wise split-conformal calibration artifact."""

    label_names: tuple[str, ...]
    alpha: float
    thresholds: tuple[float, ...]
    n_calibration_samples: int
    quantile_rank: int
    quantile_level: float

    @classmethod
    def fit(
        cls,
        probabilities: ArrayLike,
        targets: ArrayLike,
        *,
        label_names: Sequence[str],
        alpha: float = 0.1,
    ) -> LabelwiseBinaryConformal:
        """Fit thresholds on one calibration split and no evaluation data."""

        names = _validate_label_names(label_names)
        probability_matrix = _probability_matrix(
            probabilities,
            expected_labels=len(names),
            context="calibration probabilities",
        )
        target_matrix = _target_matrix(
            targets,
            expected_shape=probability_matrix.shape,
            context="calibration targets",
        )
        miscoverage = _open_unit_float(alpha, "alpha")
        n_samples = probability_matrix.shape[0]
        rank = math.ceil((n_samples + 1) * (1.0 - miscoverage))
        effective_level = min(rank / n_samples, 1.0)

        true_scores = np.where(
            target_matrix == 1,
            1.0 - probability_matrix,
            probability_matrix,
        )
        if rank > n_samples:
            threshold_array = np.ones(len(names), dtype=np.float64)
        else:
            threshold_array = np.sort(true_scores, axis=0)[rank - 1, :]

        return cls(
            label_names=names,
            alpha=miscoverage,
            thresholds=tuple(float(value) for value in threshold_array),
            n_calibration_samples=n_samples,
            quantile_rank=rank,
            quantile_level=effective_level,
        )

    def predict(self, probabilities: ArrayLike) -> BinaryPredictionSets:
        """Apply frozen thresholds without fitting on the prediction cohort."""

        probability_matrix = _probability_matrix(
            probabilities,
            expected_labels=len(self.label_names),
            context="prediction probabilities",
        )
        thresholds = np.asarray(self.thresholds, dtype=np.float64)[None, :]
        include_not_supported = probability_matrix <= thresholds
        include_supported = (1.0 - probability_matrix) <= thresholds
        return BinaryPredictionSets.from_masks(
            label_names=self.label_names,
            include_not_supported=include_not_supported,
            include_supported=include_supported,
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible frozen artifact."""

        return {
            "schema_version": _SCHEMA_VERSION,
            "artifact_type": _ARTIFACT_TYPE,
            "label_names": list(self.label_names),
            "alpha": self.alpha,
            "thresholds": list(self.thresholds),
            "n_calibration_samples": self.n_calibration_samples,
            "quantile_rank": self.quantile_rank,
            "quantile_level": self.quantile_level,
            "coverage_scope": "labelwise_marginal_under_exchangeability",
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> LabelwiseBinaryConformal:
        """Validate and restore a serialized calibration artifact."""

        _expect_exact_keys(
            payload,
            {
                "schema_version",
                "artifact_type",
                "label_names",
                "alpha",
                "thresholds",
                "n_calibration_samples",
                "quantile_rank",
                "quantile_level",
                "coverage_scope",
            },
            context="conformal artifact",
        )
        _expect_identity(payload, artifact_type=_ARTIFACT_TYPE)
        if payload["coverage_scope"] != "labelwise_marginal_under_exchangeability":
            raise ConformalValidationError("unsupported conformal coverage_scope")

        names = _validate_label_names(_string_sequence(payload["label_names"], "label_names"))
        alpha = _open_unit_float(payload["alpha"], "alpha")
        n_samples = _positive_integer(payload["n_calibration_samples"], "n_calibration_samples")
        rank = _positive_integer(payload["quantile_rank"], "quantile_rank")
        expected_rank = math.ceil((n_samples + 1) * (1.0 - alpha))
        if rank != expected_rank:
            raise ConformalValidationError("quantile_rank does not match alpha and sample count")
        if rank > n_samples + 1:
            raise ConformalValidationError("quantile_rank exceeds the finite-sample bound")

        expected_level = min(rank / n_samples, 1.0)
        level = _closed_unit_float(payload["quantile_level"], "quantile_level")
        if not math.isclose(level, expected_level, rel_tol=0.0, abs_tol=1e-12):
            raise ConformalValidationError("quantile_level does not match quantile_rank")
        thresholds = _float_tuple(
            payload["thresholds"],
            "thresholds",
            expected_length=len(names),
            lower=0.0,
            upper=1.0,
        )
        if rank > n_samples and any(value != 1.0 for value in thresholds):
            raise ConformalValidationError(
                "rank n+1 requires conservative threshold one for every label"
            )
        return cls(
            label_names=names,
            alpha=alpha,
            thresholds=thresholds,
            n_calibration_samples=n_samples,
            quantile_rank=rank,
            quantile_level=level,
        )


@dataclass(frozen=True, slots=True)
class BinaryPredictionSets:
    """Immutable membership masks and derived three-state decisions."""

    label_names: tuple[str, ...]
    include_not_supported: tuple[tuple[bool, ...], ...]
    include_supported: tuple[tuple[bool, ...], ...]

    @classmethod
    def from_masks(
        cls,
        *,
        label_names: Sequence[str],
        include_not_supported: ArrayLike,
        include_supported: ArrayLike,
    ) -> BinaryPredictionSets:
        names = _validate_label_names(label_names)
        negative = _boolean_matrix(
            include_not_supported,
            expected_labels=len(names),
            context="include_not_supported",
        )
        positive = _boolean_matrix(
            include_supported,
            expected_labels=len(names),
            context="include_supported",
        )
        if negative.shape != positive.shape:
            raise ConformalValidationError("prediction-set membership masks must align")
        return cls(
            label_names=names,
            include_not_supported=_bool_rows(negative),
            include_supported=_bool_rows(positive),
        )

    @property
    def n_samples(self) -> int:
        return len(self.include_supported)

    @property
    def n_labels(self) -> int:
        return len(self.label_names)

    @property
    def not_supported_mask(self) -> BoolArray:
        return np.asarray(self.include_not_supported, dtype=np.bool_)

    @property
    def supported_mask(self) -> BoolArray:
        return np.asarray(self.include_supported, dtype=np.bool_)

    @property
    def decisions(self) -> tuple[tuple[BinaryDecision, ...], ...]:
        """Return supported, not_supported, or uncertain per sample and label."""

        rows: list[tuple[BinaryDecision, ...]] = []
        for negative_row, positive_row in zip(
            self.include_not_supported,
            self.include_supported,
            strict=True,
        ):
            row: list[BinaryDecision] = []
            for includes_negative, includes_positive in zip(
                negative_row,
                positive_row,
                strict=True,
            ):
                if includes_positive and not includes_negative:
                    row.append(BinaryDecision.SUPPORTED)
                elif includes_negative and not includes_positive:
                    row.append(BinaryDecision.NOT_SUPPORTED)
                else:
                    row.append(BinaryDecision.UNCERTAIN)
            rows.append(tuple(row))
        return tuple(rows)

    @property
    def uncertainty_kinds(self) -> tuple[tuple[UncertaintyKind | None, ...], ...]:
        """Identify whether uncertainty came from an empty or both-valued set."""

        rows: list[tuple[UncertaintyKind | None, ...]] = []
        for negative_row, positive_row in zip(
            self.include_not_supported,
            self.include_supported,
            strict=True,
        ):
            row: list[UncertaintyKind | None] = []
            for includes_negative, includes_positive in zip(
                negative_row,
                positive_row,
                strict=True,
            ):
                if includes_negative and includes_positive:
                    row.append(UncertaintyKind.BOTH)
                elif not includes_negative and not includes_positive:
                    row.append(UncertaintyKind.EMPTY)
                else:
                    row.append(None)
            rows.append(tuple(row))
        return tuple(rows)

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible membership and explicit decision outputs."""

        return {
            "schema_version": _SCHEMA_VERSION,
            "artifact_type": _PREDICTION_TYPE,
            "label_names": list(self.label_names),
            "include_not_supported": [list(row) for row in self.include_not_supported],
            "include_supported": [list(row) for row in self.include_supported],
            "decisions": [[value.value for value in row] for row in self.decisions],
            "uncertainty_kind": [
                [None if value is None else value.value for value in row]
                for row in self.uncertainty_kinds
            ],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> BinaryPredictionSets:
        """Validate and restore serialized prediction sets."""

        _expect_exact_keys(
            payload,
            {
                "schema_version",
                "artifact_type",
                "label_names",
                "include_not_supported",
                "include_supported",
                "decisions",
                "uncertainty_kind",
            },
            context="prediction sets",
        )
        _expect_identity(payload, artifact_type=_PREDICTION_TYPE)
        names = _validate_label_names(_string_sequence(payload["label_names"], "label_names"))
        result = cls.from_masks(
            label_names=names,
            include_not_supported=_nested_bool_sequence(
                payload["include_not_supported"], "include_not_supported"
            ),
            include_supported=_nested_bool_sequence(
                payload["include_supported"], "include_supported"
            ),
        )
        serialized_decisions = _nested_optional_string_sequence(
            payload["decisions"], "decisions", allow_none=False
        )
        expected_decisions = tuple(tuple(value.value for value in row) for row in result.decisions)
        if serialized_decisions != expected_decisions:
            raise ConformalValidationError("serialized decisions contradict set membership")
        serialized_kinds = _nested_optional_string_sequence(
            payload["uncertainty_kind"], "uncertainty_kind", allow_none=True
        )
        expected_kinds = tuple(
            tuple(None if value is None else value.value for value in row)
            for row in result.uncertainty_kinds
        )
        if serialized_kinds != expected_kinds:
            raise ConformalValidationError("serialized uncertainty_kind contradicts set membership")
        return result


@dataclass(frozen=True, slots=True)
class ConformalMetrics:
    """Empirical coverage and set-size diagnostics for one evaluation cohort."""

    n_samples: int
    n_labels: int
    labelwise_coverage: tuple[float, ...]
    marginal_coverage: float
    joint_sample_coverage: float
    mean_set_size: float
    labelwise_mean_set_size: tuple[float, ...]
    singleton_fraction: float
    empty_fraction: float
    both_fraction: float

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "artifact_type": _METRICS_TYPE,
            "n_samples": self.n_samples,
            "n_labels": self.n_labels,
            "labelwise_coverage": list(self.labelwise_coverage),
            "marginal_coverage": self.marginal_coverage,
            "joint_sample_coverage": self.joint_sample_coverage,
            "mean_set_size": self.mean_set_size,
            "labelwise_mean_set_size": list(self.labelwise_mean_set_size),
            "singleton_fraction": self.singleton_fraction,
            "empty_fraction": self.empty_fraction,
            "both_fraction": self.both_fraction,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> ConformalMetrics:
        _expect_exact_keys(
            payload,
            {
                "schema_version",
                "artifact_type",
                "n_samples",
                "n_labels",
                "labelwise_coverage",
                "marginal_coverage",
                "joint_sample_coverage",
                "mean_set_size",
                "labelwise_mean_set_size",
                "singleton_fraction",
                "empty_fraction",
                "both_fraction",
            },
            context="conformal metrics",
        )
        _expect_identity(payload, artifact_type=_METRICS_TYPE)
        n_samples = _positive_integer(payload["n_samples"], "n_samples")
        n_labels = _positive_integer(payload["n_labels"], "n_labels")
        singleton_fraction = _closed_unit_float(payload["singleton_fraction"], "singleton_fraction")
        empty_fraction = _closed_unit_float(payload["empty_fraction"], "empty_fraction")
        both_fraction = _closed_unit_float(payload["both_fraction"], "both_fraction")
        if not math.isclose(
            singleton_fraction + empty_fraction + both_fraction,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ConformalValidationError("singleton, empty, and both fractions must sum to one")
        return cls(
            n_samples=n_samples,
            n_labels=n_labels,
            labelwise_coverage=_float_tuple(
                payload["labelwise_coverage"],
                "labelwise_coverage",
                expected_length=n_labels,
                lower=0.0,
                upper=1.0,
            ),
            marginal_coverage=_closed_unit_float(payload["marginal_coverage"], "marginal_coverage"),
            joint_sample_coverage=_closed_unit_float(
                payload["joint_sample_coverage"], "joint_sample_coverage"
            ),
            mean_set_size=_bounded_float(payload["mean_set_size"], "mean_set_size", 0.0, 2.0),
            labelwise_mean_set_size=_float_tuple(
                payload["labelwise_mean_set_size"],
                "labelwise_mean_set_size",
                expected_length=n_labels,
                lower=0.0,
                upper=2.0,
            ),
            singleton_fraction=singleton_fraction,
            empty_fraction=empty_fraction,
            both_fraction=both_fraction,
        )


def evaluate_prediction_sets(
    prediction_sets: BinaryPredictionSets,
    targets: ArrayLike,
) -> ConformalMetrics:
    """Measure empirical label-wise/joint coverage and prediction-set sizes."""

    expected_shape = (prediction_sets.n_samples, prediction_sets.n_labels)
    target_matrix = _target_matrix(
        targets,
        expected_shape=expected_shape,
        context="evaluation targets",
    )
    negative = prediction_sets.not_supported_mask
    positive = prediction_sets.supported_mask
    covered = np.where(target_matrix == 1, positive, negative)
    set_sizes = negative.astype(np.int8) + positive.astype(np.int8)

    return ConformalMetrics(
        n_samples=prediction_sets.n_samples,
        n_labels=prediction_sets.n_labels,
        labelwise_coverage=tuple(float(value) for value in covered.mean(axis=0)),
        marginal_coverage=float(covered.mean()),
        joint_sample_coverage=float(covered.all(axis=1).mean()),
        mean_set_size=float(set_sizes.mean()),
        labelwise_mean_set_size=tuple(float(value) for value in set_sizes.mean(axis=0)),
        singleton_fraction=float((set_sizes == 1).mean()),
        empty_fraction=float((set_sizes == 0).mean()),
        both_fraction=float((set_sizes == 2).mean()),
    )


def _probability_matrix(
    values: ArrayLike,
    *,
    expected_labels: int,
    context: str,
) -> FloatArray:
    matrix = _float_matrix(values, context)
    if matrix.shape[1] != expected_labels:
        raise ConformalValidationError(
            f"{context} has {matrix.shape[1]} labels; expected {expected_labels}"
        )
    if np.any((matrix < 0.0) | (matrix > 1.0)):
        raise ConformalValidationError(f"{context} must lie in [0, 1]")
    return matrix


def _target_matrix(
    values: ArrayLike,
    *,
    expected_shape: tuple[int, int],
    context: str,
) -> IntArray:
    matrix = _float_matrix(values, context)
    if matrix.shape != expected_shape:
        raise ConformalValidationError(
            f"{context} shape {matrix.shape} does not match {expected_shape}"
        )
    if np.any((matrix != 0.0) & (matrix != 1.0)):
        raise ConformalValidationError(f"{context} must contain only binary 0/1 values")
    return matrix.astype(np.int8)


def _float_matrix(values: ArrayLike, context: str) -> FloatArray:
    try:
        matrix = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ConformalValidationError(f"{context} must be numeric") from error
    if matrix.ndim != 2:
        raise ConformalValidationError(f"{context} must be a two-dimensional matrix")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ConformalValidationError(f"{context} must contain samples and labels")
    if not np.all(np.isfinite(matrix)):
        raise ConformalValidationError(f"{context} must contain only finite values")
    return matrix


def _boolean_matrix(
    values: ArrayLike,
    *,
    expected_labels: int,
    context: str,
) -> BoolArray:
    raw = np.asarray(values)
    if raw.ndim != 2 or raw.shape[0] == 0 or raw.shape[1] == 0:
        raise ConformalValidationError(f"{context} must be a non-empty two-dimensional matrix")
    if raw.shape[1] != expected_labels:
        raise ConformalValidationError(
            f"{context} has {raw.shape[1]} labels; expected {expected_labels}"
        )
    if raw.dtype != np.bool_:
        raise ConformalValidationError(f"{context} must contain booleans")
    return cast(BoolArray, raw.astype(np.bool_, copy=False))


def _validate_label_names(values: Sequence[str]) -> tuple[str, ...]:
    names = tuple(values)
    if not names:
        raise ConformalValidationError("label_names must not be empty")
    if any(not isinstance(value, str) or not value.strip() for value in names):
        raise ConformalValidationError("label_names must contain non-empty strings")
    if len(set(names)) != len(names):
        raise ConformalValidationError("label_names must be unique")
    return names


def _bool_rows(values: BoolArray) -> tuple[tuple[bool, ...], ...]:
    return tuple(tuple(bool(value) for value in row) for row in values)


def _expect_identity(payload: Mapping[str, object], *, artifact_type: str) -> None:
    if payload["schema_version"] != _SCHEMA_VERSION:
        raise ConformalValidationError("unsupported schema_version")
    if payload["artifact_type"] != artifact_type:
        raise ConformalValidationError("unexpected artifact_type")


def _expect_exact_keys(payload: Mapping[str, object], expected: set[str], *, context: str) -> None:
    actual = set(payload)
    if actual != expected:
        raise ConformalValidationError(
            f"{context} keys differ: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _open_unit_float(value: object, name: str) -> float:
    number = _bounded_float(value, name, 0.0, 1.0)
    if number in {0.0, 1.0}:
        raise ConformalValidationError(f"{name} must lie strictly between zero and one")
    return number


def _closed_unit_float(value: object, name: str) -> float:
    return _bounded_float(value, name, 0.0, 1.0)


def _bounded_float(value: object, name: str, lower: float, upper: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConformalValidationError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number) or not lower <= number <= upper:
        raise ConformalValidationError(f"{name} must lie in [{lower}, {upper}]")
    return number


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConformalValidationError(f"{name} must be a positive integer")
    return value


def _float_tuple(
    value: object,
    name: str,
    *,
    expected_length: int,
    lower: float,
    upper: float,
) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ConformalValidationError(f"{name} must be a sequence")
    result = tuple(_bounded_float(item, name, lower, upper) for item in value)
    if len(result) != expected_length:
        raise ConformalValidationError(f"{name} must contain {expected_length} values")
    return result


def _string_sequence(value: object, name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ConformalValidationError(f"{name} must be a sequence")
    if any(not isinstance(item, str) for item in value):
        raise ConformalValidationError(f"{name} must contain strings")
    return tuple(cast(str, item) for item in value)


def _nested_bool_sequence(value: object, name: str) -> tuple[tuple[bool, ...], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ConformalValidationError(f"{name} must be a sequence of rows")
    rows: list[tuple[bool, ...]] = []
    for row in value:
        if isinstance(row, (str, bytes)) or not isinstance(row, Sequence):
            raise ConformalValidationError(f"{name} must contain sequence rows")
        if any(not isinstance(item, bool) for item in row):
            raise ConformalValidationError(f"{name} rows must contain booleans")
        rows.append(tuple(cast(bool, item) for item in row))
    return tuple(rows)


def _nested_optional_string_sequence(
    value: object,
    name: str,
    *,
    allow_none: bool,
) -> tuple[tuple[str | None, ...], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ConformalValidationError(f"{name} must be a sequence of rows")
    rows: list[tuple[str | None, ...]] = []
    for row in value:
        if isinstance(row, (str, bytes)) or not isinstance(row, Sequence):
            raise ConformalValidationError(f"{name} must contain sequence rows")
        parsed: list[str | None] = []
        for item in row:
            if item is None and allow_none:
                parsed.append(None)
            elif isinstance(item, str):
                parsed.append(item)
            else:
                raise ConformalValidationError(f"{name} contains an invalid value")
        rows.append(tuple(parsed))
    return tuple(rows)
