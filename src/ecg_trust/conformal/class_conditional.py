"""Class-conditional label-wise split-conformal sets for sigmoid classifiers.

This opt-in artifact is an alternative to ``LabelwiseBinaryConformal``, which
pools both outcomes of a label under one miss budget. Here every label ``k``
and outcome ``c`` in ``{0, 1}`` forms its own calibration stratum, containing
only the calibration cases with ``y_k == c``, with its own caller-supplied miss
budget ``alpha_kc``. Scores follow the pooled artifact: ``p`` for the negative
outcome and ``1 - p`` for the positive outcome. A stratum with ``n_kc`` cases
uses the order statistic at rank ``ceil((n_kc + 1) * (1 - alpha_kc))``. When
that rank exceeds ``n_kc``, including ``n_kc == 0``, the budget is unattainable
at that sample size and the threshold is one, so the outcome is always
included. Stratum counts and ranks are serialized so this remains visible.

If the calibration cases with ``y_k == c`` and a new case with ``y_k == c`` are
exchangeable, then ``P(c in C_k | y_k == c) >= 1 - alpha_kc``, marginally over
the calibration draw. With almost surely distinct scores and an attainable rank
it is also at most ``1 - alpha_kc + 1 / (n_kc + 1)``. The guarantee holds for
each label and outcome separately. It is not simultaneous across labels or
outcomes, and it does not bound the error among singleton decisions: a
``SUPPORTED`` output for a rare label can still be wrong at any rate. Both
outcomes may be excluded, which ``BinaryPredictionSets`` reports as uncertain.

Covariate shift, or any change in the class-conditional input distribution
``P(x | y_k = c)``, voids the guarantee; in the multi-label setting a change in
co-occurring labels can cause the latter. A change only in the prevalence of
``y_k``, with ``P(x | y_k = c)`` fixed, preserves label ``k``'s two
class-conditional statements but changes its pooled coverage and mix of
decisions. The same change can void the statements of any other label ``j``
whose inputs depend on ``y_k``, because it changes ``P(x | y_j = c)``. Miss
budgets are research parameters, not clinical operating points, and have no
defaults.

Nothing in the Sentinel engine, case contracts, or frozen configurations
selects this artifact. The engine accepts only ``LabelwiseBinaryConformal``.
``predict`` returns plain ``BinaryPredictionSets``, which carry no provenance,
and ``conformal_prediction_sets_to_contracts`` stamps whatever sets it receives
with the pooled artifact type and coverage scope. It cannot detect these sets,
so it must not be used to convert them.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import ArrayLike

from ecg_trust.conformal.multilabel import (
    BinaryPredictionSets,
    ConformalValidationError,
    FloatArray,
    _expect_exact_keys,
    _float_tuple,
    _open_unit_float,
    _positive_integer,
    _probability_matrix,
    _string_sequence,
    _target_matrix,
    _validate_label_names,
)

_ARTIFACT_TYPE = "ecg_trust.class_conditional_labelwise_binary_conformal"
_COVERAGE_SCOPE = "labelwise_class_conditional_under_within_class_exchangeability"
_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class ClassConditionalLabelwiseConformal:
    """Frozen label-wise conformal artifact with a miss budget per outcome."""

    label_names: tuple[str, ...]
    negative_alphas: tuple[float, ...]
    positive_alphas: tuple[float, ...]
    negative_thresholds: tuple[float, ...]
    positive_thresholds: tuple[float, ...]
    n_calibration_samples: int
    negative_counts: tuple[int, ...]
    positive_counts: tuple[int, ...]
    negative_ranks: tuple[int, ...]
    positive_ranks: tuple[int, ...]

    def __post_init__(self) -> None:
        """Enforce the artifact contract for every construction path."""

        names = _validate_label_names(_string_sequence(self.label_names, "label_names"))
        n_samples = _positive_integer(self.n_calibration_samples, "n_calibration_samples")
        negative_alphas, negative_counts, negative_ranks, negative_thresholds = (
            _validated_stratum(
                "negative",
                alphas=self.negative_alphas,
                counts=self.negative_counts,
                ranks=self.negative_ranks,
                thresholds=self.negative_thresholds,
                n_labels=len(names),
            )
        )
        positive_alphas, positive_counts, positive_ranks, positive_thresholds = (
            _validated_stratum(
                "positive",
                alphas=self.positive_alphas,
                counts=self.positive_counts,
                ranks=self.positive_ranks,
                thresholds=self.positive_thresholds,
                n_labels=len(names),
            )
        )
        if any(
            negative_count + positive_count != n_samples
            for negative_count, positive_count in zip(
                negative_counts, positive_counts, strict=True
            )
        ):
            raise ConformalValidationError(
                "negative and positive counts must sum to n_calibration_samples for every label"
            )
        object.__setattr__(self, "label_names", names)
        object.__setattr__(self, "negative_alphas", negative_alphas)
        object.__setattr__(self, "positive_alphas", positive_alphas)
        object.__setattr__(self, "negative_thresholds", negative_thresholds)
        object.__setattr__(self, "positive_thresholds", positive_thresholds)
        object.__setattr__(self, "negative_counts", negative_counts)
        object.__setattr__(self, "positive_counts", positive_counts)
        object.__setattr__(self, "negative_ranks", negative_ranks)
        object.__setattr__(self, "positive_ranks", positive_ranks)

    @classmethod
    def fit(
        cls,
        probabilities: ArrayLike,
        targets: ArrayLike,
        *,
        label_names: Sequence[str],
        negative_alphas: Mapping[str, float],
        positive_alphas: Mapping[str, float],
    ) -> ClassConditionalLabelwiseConformal:
        """Fit per-label, per-outcome thresholds on one calibration split.

        Both budget mappings must name every label exactly once, so a budget
        cannot be silently attached to the wrong label by position.
        """

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
        negative_budgets = _alpha_mapping(negative_alphas, names, "negative_alphas")
        positive_budgets = _alpha_mapping(positive_alphas, names, "positive_alphas")

        negative = [
            _fit_stratum(probability_matrix[target_matrix[:, index] == 0, index], alpha)
            for index, alpha in enumerate(negative_budgets)
        ]
        positive = [
            _fit_stratum(1.0 - probability_matrix[target_matrix[:, index] == 1, index], alpha)
            for index, alpha in enumerate(positive_budgets)
        ]
        return cls(
            label_names=names,
            negative_alphas=negative_budgets,
            positive_alphas=positive_budgets,
            negative_thresholds=tuple(threshold for _, _, threshold in negative),
            positive_thresholds=tuple(threshold for _, _, threshold in positive),
            n_calibration_samples=probability_matrix.shape[0],
            negative_counts=tuple(count for count, _, _ in negative),
            positive_counts=tuple(count for count, _, _ in positive),
            negative_ranks=tuple(rank for _, rank, _ in negative),
            positive_ranks=tuple(rank for _, rank, _ in positive),
        )

    @property
    def negative_budget_attainable(self) -> tuple[bool, ...]:
        """Whether each negative stratum had enough cases for its requested rank."""

        return tuple(
            rank <= count
            for rank, count in zip(self.negative_ranks, self.negative_counts, strict=True)
        )

    @property
    def positive_budget_attainable(self) -> tuple[bool, ...]:
        """Whether each positive stratum had enough cases for its requested rank."""

        return tuple(
            rank <= count
            for rank, count in zip(self.positive_ranks, self.positive_counts, strict=True)
        )

    def predict(self, probabilities: ArrayLike) -> BinaryPredictionSets:
        """Apply frozen thresholds without fitting on the prediction cohort."""

        probability_matrix = _probability_matrix(
            probabilities,
            expected_labels=len(self.label_names),
            context="prediction probabilities",
        )
        negative_thresholds = np.asarray(self.negative_thresholds, dtype=np.float64)[None, :]
        positive_thresholds = np.asarray(self.positive_thresholds, dtype=np.float64)[None, :]
        return BinaryPredictionSets.from_masks(
            label_names=self.label_names,
            include_not_supported=probability_matrix <= negative_thresholds,
            include_supported=(1.0 - probability_matrix) <= positive_thresholds,
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible frozen artifact."""

        return {
            "schema_version": _SCHEMA_VERSION,
            "artifact_type": _ARTIFACT_TYPE,
            "label_names": list(self.label_names),
            "negative_alphas": list(self.negative_alphas),
            "positive_alphas": list(self.positive_alphas),
            "negative_thresholds": list(self.negative_thresholds),
            "positive_thresholds": list(self.positive_thresholds),
            "n_calibration_samples": self.n_calibration_samples,
            "negative_counts": list(self.negative_counts),
            "positive_counts": list(self.positive_counts),
            "negative_ranks": list(self.negative_ranks),
            "positive_ranks": list(self.positive_ranks),
            "coverage_scope": _COVERAGE_SCOPE,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> ClassConditionalLabelwiseConformal:
        """Validate and restore a serialized calibration artifact."""

        _expect_exact_keys(
            payload,
            {
                "schema_version",
                "artifact_type",
                "label_names",
                "negative_alphas",
                "positive_alphas",
                "negative_thresholds",
                "positive_thresholds",
                "n_calibration_samples",
                "negative_counts",
                "positive_counts",
                "negative_ranks",
                "positive_ranks",
                "coverage_scope",
            },
            context="class-conditional conformal artifact",
        )
        version = payload["schema_version"]
        if type(version) is not int or version != _SCHEMA_VERSION:
            raise ConformalValidationError("unsupported schema_version")
        if payload["artifact_type"] != _ARTIFACT_TYPE:
            raise ConformalValidationError("unexpected artifact_type")
        if payload["coverage_scope"] != _COVERAGE_SCOPE:
            raise ConformalValidationError("unsupported conformal coverage_scope")

        return cls(
            label_names=cast(tuple[str, ...], payload["label_names"]),
            negative_alphas=cast(tuple[float, ...], payload["negative_alphas"]),
            positive_alphas=cast(tuple[float, ...], payload["positive_alphas"]),
            negative_thresholds=cast(tuple[float, ...], payload["negative_thresholds"]),
            positive_thresholds=cast(tuple[float, ...], payload["positive_thresholds"]),
            n_calibration_samples=cast(int, payload["n_calibration_samples"]),
            negative_counts=cast(tuple[int, ...], payload["negative_counts"]),
            positive_counts=cast(tuple[int, ...], payload["positive_counts"]),
            negative_ranks=cast(tuple[int, ...], payload["negative_ranks"]),
            positive_ranks=cast(tuple[int, ...], payload["positive_ranks"]),
        )


def _fit_stratum(scores: FloatArray, alpha: float) -> tuple[int, int, float]:
    count = int(scores.shape[0])
    rank = _conformal_rank(count, alpha)
    if rank > count:
        return count, rank, 1.0
    return count, rank, float(np.sort(scores)[rank - 1])


def _conformal_rank(count: int, alpha: float) -> int:
    return math.ceil((count + 1) * (1.0 - alpha))


def _validated_stratum(
    prefix: str,
    *,
    alphas: object,
    counts: object,
    ranks: object,
    thresholds: object,
    n_labels: int,
) -> tuple[tuple[float, ...], tuple[int, ...], tuple[int, ...], tuple[float, ...]]:
    alpha_values = tuple(
        _open_unit_float(value, f"{prefix}_alphas")
        for value in _float_tuple(
            alphas, f"{prefix}_alphas", expected_length=n_labels, lower=0.0, upper=1.0
        )
    )
    count_values = _integer_tuple(counts, f"{prefix}_counts", expected_length=n_labels, minimum=0)
    rank_values = _integer_tuple(ranks, f"{prefix}_ranks", expected_length=n_labels, minimum=1)
    threshold_values = _float_tuple(
        thresholds, f"{prefix}_thresholds", expected_length=n_labels, lower=0.0, upper=1.0
    )
    for alpha, count, rank, threshold in zip(
        alpha_values, count_values, rank_values, threshold_values, strict=True
    ):
        if rank != _conformal_rank(count, alpha):
            raise ConformalValidationError(
                f"{prefix}_ranks do not match {prefix}_alphas and {prefix}_counts"
            )
        if rank > count and threshold != 1.0:
            raise ConformalValidationError(
                f"an unattainable {prefix} rank requires conservative threshold one"
            )
    return alpha_values, count_values, rank_values, threshold_values


def _alpha_mapping(value: object, names: tuple[str, ...], name: str) -> tuple[float, ...]:
    if not isinstance(value, Mapping):
        raise ConformalValidationError(f"{name} must map every label name to a miss budget")
    if any(not isinstance(key, str) for key in value):
        raise ConformalValidationError(f"{name} keys must be label names")
    _expect_exact_keys(cast(Mapping[str, object], value), set(names), context=name)
    return tuple(_open_unit_float(value[label], f"{name}[{label}]") for label in names)


def _integer_tuple(
    value: object,
    name: str,
    *,
    expected_length: int,
    minimum: int,
) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ConformalValidationError(f"{name} must be a sequence")
    if any(
        isinstance(item, bool) or not isinstance(item, int) or item < minimum for item in value
    ):
        raise ConformalValidationError(f"{name} must contain integers of at least {minimum}")
    result = tuple(cast(int, item) for item in value)
    if len(result) != expected_length:
        raise ConformalValidationError(f"{name} must contain {expected_length} values")
    return result


__all__ = ["ClassConditionalLabelwiseConformal"]
