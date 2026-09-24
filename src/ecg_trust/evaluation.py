"""Leakage-resistant evaluation utilities for PTB-XL multilabel models.

All functions in this module enforce the canonical five-label output order.
Operations that *fit* a decision rule (temperature scaling, classwise sigmoid
scaling, or thresholds) also require row-level fold provenance and accept fold 9
only.  Fold 10 therefore remains an evaluation set rather than an implicit source
of fitted parameters.  Classwise sigmoid scaling is an opt-in library capability
that no frozen pipeline calls.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ecg_trust.protocol import CALIBRATION_FOLDS, LABEL_ORDER

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]

_SIGMOID_SCALING_FITTED_STATUSES = frozenset({"optimized", "identity_optimal"})
_ARMIJO_FRACTION = 0.25
_MAX_BACKTRACKING_HALVINGS = 64
# Smallest classwise ridge penalty. On the mean-NLL scale a weaker penalty is
# below the 1e-12 improvement margin, and Newton steps on near-constant logit
# columns lose their precision to rounding.
_MIN_SIGMOID_SCALING_REGULARIZATION = 1e-12


class EvaluationValidationError(ValueError):
    """Raised when evaluation inputs violate the canonical data contract."""


class CalibrationLeakageError(EvaluationValidationError):
    """Raised when a fitted evaluation artifact receives non-calibration rows."""


@dataclass(frozen=True, slots=True)
class PerLabelMetrics:
    """Discrimination and calibration metrics for one output label."""

    label: str
    positives: int
    negatives: int
    prevalence: float
    roc_auc: float | None
    average_precision: float | None
    brier_score: float
    ece: float
    degenerate_reason: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "positives": self.positives,
            "negatives": self.negatives,
            "prevalence": self.prevalence,
            "roc_auc": self.roc_auc,
            "average_precision": self.average_precision,
            "brier_score": self.brier_score,
            "ece": self.ece,
            "degenerate_reason": self.degenerate_reason,
        }


@dataclass(frozen=True, slots=True)
class MacroMetrics:
    """Macro averages and the number of labels that contributed to each."""

    roc_auc: float | None
    average_precision: float | None
    brier_score: float
    ece: float
    roc_auc_labels: int
    average_precision_labels: int

    def to_dict(self) -> dict[str, object]:
        return {
            "roc_auc": self.roc_auc,
            "average_precision": self.average_precision,
            "brier_score": self.brier_score,
            "ece": self.ece,
            "roc_auc_labels": self.roc_auc_labels,
            "average_precision_labels": self.average_precision_labels,
        }


@dataclass(frozen=True, slots=True)
class MultilabelMetrics:
    """Serializable evaluation report for canonical PTB-XL probabilities."""

    n_samples: int
    label_order: tuple[str, ...]
    ece_bins: int
    per_label: tuple[PerLabelMetrics, ...]
    macro: MacroMetrics

    def to_dict(self) -> dict[str, object]:
        return {
            "n_samples": self.n_samples,
            "label_order": list(self.label_order),
            "ece_bins": self.ece_bins,
            "per_label": [metric.to_dict() for metric in self.per_label],
            "macro": self.macro.to_dict(),
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        serialized = json.dumps(self.to_dict(), indent=indent, sort_keys=True)
        return f"{serialized}\n" if indent is not None else serialized


@dataclass(frozen=True, slots=True)
class PerLabelThreshold:
    """One label's threshold-selection outcome."""

    label: str
    threshold: float
    objective: str
    objective_value: float | None
    positives: int
    negatives: int
    status: str

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "threshold": self.threshold,
            "objective": self.objective,
            "objective_value": self.objective_value,
            "positives": self.positives,
            "negatives": self.negatives,
            "status": self.status,
        }


@dataclass(frozen=True, slots=True)
class ThresholdOptimizationResult:
    """Calibration-fold-only thresholds and their provenance."""

    label_order: tuple[str, ...]
    thresholds: tuple[float, ...]
    objective: str
    macro_objective: float | None
    default_threshold: float
    n_samples: int
    source_folds: tuple[int, ...]
    per_label: tuple[PerLabelThreshold, ...]

    def apply(
        self,
        probabilities: ArrayLike,
        *,
        label_order: Sequence[str] = LABEL_ORDER,
    ) -> BoolArray:
        """Apply fitted thresholds after re-validating the output contract."""

        labels = _validate_label_order(label_order)
        scores = _validate_score_matrix(
            probabilities,
            name="probabilities",
            n_labels=len(labels),
            n_samples=None,
        )
        if np.any((scores < 0.0) | (scores > 1.0)):
            raise EvaluationValidationError(
                "probabilities must lie in the closed interval [0, 1]"
            )
        return scores >= np.asarray(self.thresholds, dtype=np.float64)[None, :]

    def to_dict(self) -> dict[str, object]:
        return {
            "label_order": list(self.label_order),
            "thresholds": list(self.thresholds),
            "objective": self.objective,
            "macro_objective": self.macro_objective,
            "default_threshold": self.default_threshold,
            "n_samples": self.n_samples,
            "source_folds": list(self.source_folds),
            "per_label": [item.to_dict() for item in self.per_label],
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        serialized = json.dumps(self.to_dict(), indent=indent, sort_keys=True)
        return f"{serialized}\n" if indent is not None else serialized


@dataclass(frozen=True, slots=True)
class TemperatureScalingResult:
    """A single temperature fitted exclusively on calibration fold 9."""

    temperature: float
    label_order: tuple[str, ...]
    n_samples: int
    source_folds: tuple[int, ...]
    fitted_labels: tuple[str, ...]
    excluded_degenerate_labels: tuple[str, ...]
    nll_before: float | None
    nll_after: float | None
    status: str
    converged: bool
    optimization_steps: int
    temperature_bounds: tuple[float, float]

    def transform_logits(
        self,
        logits: ArrayLike,
        *,
        label_order: Sequence[str] = LABEL_ORDER,
    ) -> FloatArray:
        """Divide logits by the fitted positive temperature."""

        validated = validate_logits(logits, label_order=label_order)
        return validated / self.temperature

    def predict_proba(
        self,
        logits: ArrayLike,
        *,
        label_order: Sequence[str] = LABEL_ORDER,
    ) -> FloatArray:
        """Convert temperature-scaled logits to probabilities."""

        return stable_sigmoid(self.transform_logits(logits, label_order=label_order))

    def to_dict(self) -> dict[str, object]:
        return {
            "temperature": self.temperature,
            "label_order": list(self.label_order),
            "n_samples": self.n_samples,
            "source_folds": list(self.source_folds),
            "fitted_labels": list(self.fitted_labels),
            "excluded_degenerate_labels": list(self.excluded_degenerate_labels),
            "nll_before": self.nll_before,
            "nll_after": self.nll_after,
            "status": self.status,
            "converged": self.converged,
            "optimization_steps": self.optimization_steps,
            "temperature_bounds": list(self.temperature_bounds),
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        serialized = json.dumps(self.to_dict(), indent=indent, sort_keys=True)
        return f"{serialized}\n" if indent is not None else serialized


@dataclass(frozen=True, slots=True)
class PerLabelSigmoidScaling:
    """One label's regularized sigmoid-scaling outcome on calibration fold 9.

    ``nll_before`` and ``nll_after`` are unregularized mean binary NLL on the
    calibration rows. Degenerate labels keep the identity map and report
    ``None`` for both, matching the global temperature policy.
    """

    label: str
    slope: float
    intercept: float
    positives: int
    negatives: int
    nll_before: float | None
    nll_after: float | None
    status: str
    converged: bool
    optimization_steps: int
    active_slope_bound: str | None

    def __post_init__(self) -> None:
        """Enforce one label's calibration-map contract for every construction path."""

        if not isinstance(self.label, str) or self.label not in LABEL_ORDER:
            raise EvaluationValidationError(
                f"sigmoid-scaling label must be one of {LABEL_ORDER!r}"
            )
        slope = _finite_real_number(self.slope, name="slope")
        if slope <= 0.0:
            raise EvaluationValidationError("slope must be positive")
        intercept = _finite_real_number(self.intercept, name="intercept")
        positives = _nonnegative_integer(self.positives, name="positives")
        negatives = _nonnegative_integer(self.negatives, name="negatives")
        steps = _nonnegative_integer(self.optimization_steps, name="optimization_steps")
        if positives + negatives == 0:
            raise EvaluationValidationError("sigmoid scaling requires at least one sample")
        if not isinstance(self.converged, bool):
            raise EvaluationValidationError("converged must be a boolean")
        if self.active_slope_bound not in (None, "lower", "upper"):
            raise EvaluationValidationError("active_slope_bound must be lower, upper, or None")

        identity = slope == 1.0 and intercept == 0.0
        reason = _degenerate_reason(positives, negatives)
        before: float | None = None
        after: float | None = None
        if reason is not None:
            if (
                self.status != reason
                or not identity
                or self.nll_before is not None
                or self.nll_after is not None
                or self.converged
                or steps != 0
                or self.active_slope_bound is not None
            ):
                raise EvaluationValidationError(
                    f"degenerate label {self.label} must keep the identity map "
                    f"and report {reason!r}"
                )
        else:
            if self.status not in _SIGMOID_SCALING_FITTED_STATUSES:
                raise EvaluationValidationError(
                    f"non-degenerate label {self.label} has invalid status {self.status!r}"
                )
            if self.nll_before is None or self.nll_after is None:
                raise EvaluationValidationError(
                    f"non-degenerate label {self.label} must report NLL before and after"
                )
            before = _finite_real_number(self.nll_before, name="nll_before")
            after = _finite_real_number(self.nll_after, name="nll_after")
            if before < 0.0 or after < 0.0:
                raise EvaluationValidationError("binary NLL cannot be negative")
            if after > before:
                raise EvaluationValidationError(
                    "sigmoid scaling must not increase calibration-fold NLL"
                )
            if self.status == "identity_optimal" and (
                not identity or after != before or self.active_slope_bound is not None
            ):
                raise EvaluationValidationError(
                    "identity_optimal labels must keep the identity map and NLL"
                )
        object.__setattr__(self, "slope", slope)
        object.__setattr__(self, "intercept", intercept)
        object.__setattr__(self, "nll_before", before)
        object.__setattr__(self, "nll_after", after)

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "slope": self.slope,
            "intercept": self.intercept,
            "positives": self.positives,
            "negatives": self.negatives,
            "nll_before": self.nll_before,
            "nll_after": self.nll_after,
            "status": self.status,
            "converged": self.converged,
            "optimization_steps": self.optimization_steps,
            "active_slope_bound": self.active_slope_bound,
        }


@dataclass(frozen=True, slots=True)
class ClasswiseSigmoidScalingResult:
    """Per-label positive slopes and intercepts fitted exclusively on fold 9.

    Label ``k`` is mapped to ``sigmoid(slopes[k] * z_k + intercepts[k])``.
    Every slope is strictly positive, so each label's logit ordering is kept.
    """

    slopes: tuple[float, ...]
    intercepts: tuple[float, ...]
    label_order: tuple[str, ...]
    n_samples: int
    source_folds: tuple[int, ...]
    fitted_labels: tuple[str, ...]
    excluded_degenerate_labels: tuple[str, ...]
    nll_before: float | None
    nll_after: float | None
    status: str
    converged: bool
    regularization: float
    fit_intercept: bool
    slope_bounds: tuple[float, float]
    tolerance: float
    max_steps: int
    per_label: tuple[PerLabelSigmoidScaling, ...]

    def __post_init__(self) -> None:
        """Enforce the fitted artifact contract for every construction path."""

        if _field_tuple(self.label_order, name="label_order") != LABEL_ORDER:
            raise EvaluationValidationError(f"label_order must be exactly {LABEL_ORDER!r}")
        labels = LABEL_ORDER
        source_folds = _field_tuple(self.source_folds, name="source_folds")
        if source_folds != CALIBRATION_FOLDS or any(
            isinstance(fold, bool) or not isinstance(fold, int) for fold in source_folds
        ):
            raise CalibrationLeakageError(
                "fitted evaluation artifacts may use calibration fold 9 only; "
                f"received folds {source_folds!r}"
            )
        n_samples = _nonnegative_integer(self.n_samples, name="n_samples")
        if n_samples == 0:
            raise EvaluationValidationError("n_samples must be positive")
        regularization = _validate_sigmoid_scaling_regularization(self.regularization)
        if not isinstance(self.fit_intercept, bool):
            raise EvaluationValidationError("fit_intercept must be a boolean")
        lower_slope, upper_slope = _validate_slope_bounds(self.slope_bounds)
        tolerance = _positive_real_number(self.tolerance, name="tolerance")
        max_steps = _validate_max_steps(self.max_steps)

        entries: list[PerLabelSigmoidScaling] = []
        for item in _field_tuple(self.per_label, name="per_label"):
            if not isinstance(item, PerLabelSigmoidScaling):
                raise EvaluationValidationError(
                    "per_label must contain PerLabelSigmoidScaling items"
                )
            entries.append(item)
        if tuple(item.label for item in entries) != labels:
            raise EvaluationValidationError("per_label entries must follow label_order")
        for item in entries:
            if item.positives + item.negatives != n_samples:
                raise EvaluationValidationError(
                    f"label {item.label} counts do not match n_samples"
                )
            if item.optimization_steps > max_steps:
                raise EvaluationValidationError(
                    f"label {item.label} exceeds max_steps Newton steps"
                )
            if not lower_slope <= item.slope <= upper_slope:
                raise EvaluationValidationError(
                    f"label {item.label} slope lies outside slope_bounds"
                )
            if (item.active_slope_bound == "lower" and item.slope != lower_slope) or (
                item.active_slope_bound == "upper" and item.slope != upper_slope
            ):
                raise EvaluationValidationError(
                    f"label {item.label} active slope bound does not match its slope"
                )
            if not self.fit_intercept and item.intercept != 0.0:
                raise EvaluationValidationError(
                    "slope-only sigmoid scaling must keep every intercept at zero"
                )

        slopes = tuple(item.slope for item in entries)
        intercepts = tuple(item.intercept for item in entries)
        if _field_tuple(self.slopes, name="slopes") != slopes or (
            _field_tuple(self.intercepts, name="intercepts") != intercepts
        ):
            raise EvaluationValidationError("slopes and intercepts must match per_label")
        fitted_labels, excluded_labels, before, after, status, converged = (
            _summarize_sigmoid_scaling(entries)
        )
        if (
            _field_tuple(self.fitted_labels, name="fitted_labels") != fitted_labels
            or _field_tuple(self.excluded_degenerate_labels, name="excluded_degenerate_labels")
            != excluded_labels
            or not _matches_optional_mean(self.nll_before, before)
            or not _matches_optional_mean(self.nll_after, after)
            or self.status != status
            or self.converged is not converged
        ):
            raise EvaluationValidationError(
                "fitted labels, NLL means, status, and converged must summarize per_label"
            )

        object.__setattr__(self, "slopes", slopes)
        object.__setattr__(self, "intercepts", intercepts)
        object.__setattr__(self, "label_order", labels)
        object.__setattr__(self, "source_folds", CALIBRATION_FOLDS)
        object.__setattr__(self, "fitted_labels", fitted_labels)
        object.__setattr__(self, "excluded_degenerate_labels", excluded_labels)
        object.__setattr__(self, "nll_before", before)
        object.__setattr__(self, "nll_after", after)
        object.__setattr__(self, "regularization", regularization)
        object.__setattr__(self, "slope_bounds", (lower_slope, upper_slope))
        object.__setattr__(self, "tolerance", tolerance)
        object.__setattr__(self, "per_label", tuple(entries))

    def transform_logits(
        self,
        logits: ArrayLike,
        *,
        label_order: Sequence[str] = LABEL_ORDER,
    ) -> FloatArray:
        """Apply each label's positive slope and intercept to validated logits."""

        validated = validate_logits(logits, label_order=label_order)
        with np.errstate(over="ignore", invalid="ignore"):
            scaled = (
                validated * np.asarray(self.slopes, dtype=np.float64)[None, :]
                + np.asarray(self.intercepts, dtype=np.float64)[None, :]
            )
        if not np.all(np.isfinite(scaled)):
            raise EvaluationValidationError("scaled logits must contain only finite values")
        return scaled

    def predict_proba(
        self,
        logits: ArrayLike,
        *,
        label_order: Sequence[str] = LABEL_ORDER,
    ) -> FloatArray:
        """Convert classwise-scaled logits to probabilities."""

        return stable_sigmoid(self.transform_logits(logits, label_order=label_order))

    def to_dict(self) -> dict[str, object]:
        return {
            "slopes": list(self.slopes),
            "intercepts": list(self.intercepts),
            "label_order": list(self.label_order),
            "n_samples": self.n_samples,
            "source_folds": list(self.source_folds),
            "fitted_labels": list(self.fitted_labels),
            "excluded_degenerate_labels": list(self.excluded_degenerate_labels),
            "nll_before": self.nll_before,
            "nll_after": self.nll_after,
            "status": self.status,
            "converged": self.converged,
            "regularization": self.regularization,
            "fit_intercept": self.fit_intercept,
            "slope_bounds": list(self.slope_bounds),
            "tolerance": self.tolerance,
            "max_steps": self.max_steps,
            "per_label": [item.to_dict() for item in self.per_label],
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        serialized = json.dumps(self.to_dict(), indent=indent, sort_keys=True)
        return f"{serialized}\n" if indent is not None else serialized


@dataclass(frozen=True, slots=True)
class SelectiveCoveragePoint:
    """Prediction quality after retaining the least-uncertain samples."""

    target_coverage: float
    achieved_coverage: float
    selected_count: int
    abstained_count: int
    hamming_risk: float | None
    exact_match_accuracy: float | None
    per_label_error_rate: tuple[float | None, ...]
    mean_selected_uncertainty: float | None
    selected_indices: tuple[int, ...]
    abstained_indices: tuple[int, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "target_coverage": self.target_coverage,
            "achieved_coverage": self.achieved_coverage,
            "selected_count": self.selected_count,
            "abstained_count": self.abstained_count,
            "hamming_risk": self.hamming_risk,
            "exact_match_accuracy": self.exact_match_accuracy,
            "per_label_error_rate": list(self.per_label_error_rate),
            "mean_selected_uncertainty": self.mean_selected_uncertainty,
            "selected_indices": list(self.selected_indices),
            "abstained_indices": list(self.abstained_indices),
        }


@dataclass(frozen=True, slots=True)
class SelectivePredictionResult:
    """Serializable risk/coverage and abstention decisions."""

    n_samples: int
    label_order: tuple[str, ...]
    thresholds: tuple[float, ...]
    uncertainty_method: str
    coverage_points: tuple[SelectiveCoveragePoint, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "n_samples": self.n_samples,
            "label_order": list(self.label_order),
            "thresholds": list(self.thresholds),
            "uncertainty_method": self.uncertainty_method,
            "coverage_points": [point.to_dict() for point in self.coverage_points],
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        serialized = json.dumps(self.to_dict(), indent=indent, sort_keys=True)
        return f"{serialized}\n" if indent is not None else serialized


def validate_multilabel_arrays(
    y_true: ArrayLike,
    probabilities: ArrayLike,
    *,
    label_order: Sequence[str] = LABEL_ORDER,
) -> tuple[IntArray, FloatArray]:
    """Validate canonical binary targets and probability predictions."""

    labels = _validate_label_order(label_order)
    targets = _validate_targets(y_true, n_labels=len(labels))
    scores = _validate_score_matrix(
        probabilities,
        name="probabilities",
        n_labels=len(labels),
        n_samples=targets.shape[0],
    )
    if np.any((scores < 0.0) | (scores > 1.0)):
        raise EvaluationValidationError("probabilities must lie in the closed interval [0, 1]")
    return targets, scores


def validate_logits(
    logits: ArrayLike,
    *,
    label_order: Sequence[str] = LABEL_ORDER,
    n_samples: int | None = None,
) -> FloatArray:
    """Validate a finite ``[samples, 5]`` canonical logit matrix."""

    labels = _validate_label_order(label_order)
    return _validate_score_matrix(
        logits,
        name="logits",
        n_labels=len(labels),
        n_samples=n_samples,
    )


def compute_multilabel_metrics(
    y_true: ArrayLike,
    probabilities: ArrayLike,
    *,
    label_order: Sequence[str] = LABEL_ORDER,
    ece_bins: int = 15,
) -> MultilabelMetrics:
    """Compute per-label and macro discrimination/calibration metrics.

    ROC-AUC and average precision are reported as ``None`` for a label with
    only one observed class.  Such labels are excluded from their macro means;
    Brier score and ECE remain defined and are always included.
    """

    labels = _validate_label_order(label_order)
    targets, scores = validate_multilabel_arrays(
        y_true, probabilities, label_order=labels
    )
    bins = _validate_ece_bins(ece_bins)

    reports: list[PerLabelMetrics] = []
    for index, label in enumerate(labels):
        label_targets = targets[:, index]
        label_scores = scores[:, index]
        positives = int(label_targets.sum())
        negatives = int(targets.shape[0] - positives)
        reason = _degenerate_reason(positives, negatives)
        roc_auc = None if reason else _binary_roc_auc(label_targets, label_scores)
        average_precision = (
            None if reason else _binary_average_precision(label_targets, label_scores)
        )
        reports.append(
            PerLabelMetrics(
                label=label,
                positives=positives,
                negatives=negatives,
                prevalence=float(positives / targets.shape[0]),
                roc_auc=roc_auc,
                average_precision=average_precision,
                brier_score=float(np.mean(np.square(label_scores - label_targets))),
                ece=fixed_bin_ece(label_targets, label_scores, n_bins=bins),
                degenerate_reason=reason,
            )
        )

    valid_roc = [metric.roc_auc for metric in reports if metric.roc_auc is not None]
    valid_ap = [
        metric.average_precision
        for metric in reports
        if metric.average_precision is not None
    ]
    macro = MacroMetrics(
        roc_auc=_optional_mean(valid_roc),
        average_precision=_optional_mean(valid_ap),
        brier_score=float(np.mean([metric.brier_score for metric in reports])),
        ece=float(np.mean([metric.ece for metric in reports])),
        roc_auc_labels=len(valid_roc),
        average_precision_labels=len(valid_ap),
    )
    return MultilabelMetrics(
        n_samples=targets.shape[0],
        label_order=labels,
        ece_bins=bins,
        per_label=tuple(reports),
        macro=macro,
    )


def fixed_bin_ece(
    y_true: ArrayLike,
    probabilities: ArrayLike,
    *,
    n_bins: int = 15,
) -> float:
    """Compute equal-width expected calibration error for one binary label.

    Bins are ``[0, 1/B)``, ..., ``[(B-1)/B, 1]``. Empty bins contribute zero.
    """

    bins = _validate_ece_bins(n_bins)
    targets = np.asarray(y_true)
    scores = _real_float_array(probabilities, name="probabilities")
    if targets.ndim != 1 or scores.ndim != 1 or targets.shape != scores.shape:
        raise EvaluationValidationError(
            "fixed_bin_ece expects equal-length one-dimensional arrays"
        )
    if targets.size == 0:
        raise EvaluationValidationError("fixed_bin_ece requires at least one sample")
    if not np.all(np.isfinite(scores)):
        raise EvaluationValidationError("probabilities must contain only finite values")
    if np.any((scores < 0.0) | (scores > 1.0)):
        raise EvaluationValidationError("probabilities must lie in the closed interval [0, 1]")
    if not _is_binary_array(targets):
        raise EvaluationValidationError("targets must contain only binary values 0 and 1")

    target_float = targets.astype(np.float64, copy=False)
    bin_indices = np.minimum((scores * bins).astype(np.int64), bins - 1)
    ece = 0.0
    for bin_index in range(bins):
        members = bin_indices == bin_index
        count = int(members.sum())
        if count:
            confidence = float(scores[members].mean())
            accuracy = float(target_float[members].mean())
            ece += (count / scores.size) * abs(accuracy - confidence)
    return float(ece)


def optimize_thresholds(
    *,
    y_true: ArrayLike,
    probabilities: ArrayLike,
    calibration_fold_ids: ArrayLike,
    label_order: Sequence[str] = LABEL_ORDER,
    default_threshold: float = 0.5,
) -> ThresholdOptimizationResult:
    """Select per-label F1 thresholds using calibration-fold rows only.

    Degenerate labels cannot support threshold selection.  They retain
    ``default_threshold``, have a ``None`` objective value, and carry an
    explicit status explaining which class was absent.
    """

    labels = _validate_label_order(label_order)
    targets, scores = validate_multilabel_arrays(
        y_true, probabilities, label_order=labels
    )
    source_folds = _validate_calibration_fold_ids(
        calibration_fold_ids, n_samples=targets.shape[0]
    )
    default = _validate_threshold(default_threshold, name="default_threshold")

    per_label: list[PerLabelThreshold] = []
    objective_values: list[float] = []
    thresholds: list[float] = []
    for index, label in enumerate(labels):
        label_targets = targets[:, index]
        label_scores = scores[:, index]
        positives = int(label_targets.sum())
        negatives = int(targets.shape[0] - positives)
        reason = _degenerate_reason(positives, negatives)
        if reason is not None:
            threshold = default
            objective_value = None
            status = reason
        else:
            threshold, objective_value = _best_f1_threshold(
                label_targets, label_scores, default=default
            )
            objective_values.append(objective_value)
            status = "optimized"
        thresholds.append(threshold)
        per_label.append(
            PerLabelThreshold(
                label=label,
                threshold=threshold,
                objective="f1",
                objective_value=objective_value,
                positives=positives,
                negatives=negatives,
                status=status,
            )
        )

    return ThresholdOptimizationResult(
        label_order=labels,
        thresholds=tuple(thresholds),
        objective="f1",
        macro_objective=_optional_mean(objective_values),
        default_threshold=default,
        n_samples=targets.shape[0],
        source_folds=source_folds,
        per_label=tuple(per_label),
    )


def fit_temperature_scaling(
    *,
    logits: ArrayLike,
    y_true: ArrayLike,
    calibration_fold_ids: ArrayLike,
    label_order: Sequence[str] = LABEL_ORDER,
    temperature_bounds: tuple[float, float] = (0.05, 20.0),
    tolerance: float = 1e-10,
    max_steps: int = 128,
) -> TemperatureScalingResult:
    """Fit one positive temperature on non-degenerate calibration labels.

    Optimization is a deterministic golden-section search over inverse
    temperature. Binary NLL is convex in inverse temperature, making this a
    small and reproducible fit without optimizer state.
    """

    labels = _validate_label_order(label_order)
    targets = _validate_targets(y_true, n_labels=len(labels))
    validated_logits = validate_logits(
        logits, label_order=labels, n_samples=targets.shape[0]
    )
    source_folds = _validate_calibration_fold_ids(
        calibration_fold_ids, n_samples=targets.shape[0]
    )
    lower_temperature, upper_temperature = _validate_temperature_bounds(
        temperature_bounds
    )
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise EvaluationValidationError("tolerance must be finite and positive")
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps <= 0:
        raise EvaluationValidationError("max_steps must be a positive integer")

    fitted_indices: list[int] = []
    excluded_labels: list[str] = []
    for index, label in enumerate(labels):
        positives = int(targets[:, index].sum())
        negatives = int(targets.shape[0] - positives)
        if _degenerate_reason(positives, negatives) is None:
            fitted_indices.append(index)
        else:
            excluded_labels.append(label)

    if not fitted_indices:
        return TemperatureScalingResult(
            temperature=1.0,
            label_order=labels,
            n_samples=targets.shape[0],
            source_folds=source_folds,
            fitted_labels=(),
            excluded_degenerate_labels=tuple(excluded_labels),
            nll_before=None,
            nll_after=None,
            status="no_non_degenerate_labels",
            converged=False,
            optimization_steps=0,
            temperature_bounds=(lower_temperature, upper_temperature),
        )

    fit_logits = validated_logits[:, fitted_indices]
    fit_targets = targets[:, fitted_indices].astype(np.float64, copy=False)

    def objective(inverse_temperature: float) -> float:
        return _binary_nll(fit_logits * inverse_temperature, fit_targets)

    before = objective(1.0)
    lower_inverse = 1.0 / upper_temperature
    upper_inverse = 1.0 / lower_temperature
    inverse_temperature, optimized_nll, steps, converged = _golden_section_minimize(
        objective,
        lower=lower_inverse,
        upper=upper_inverse,
        tolerance=tolerance,
        max_steps=max_steps,
    )
    optimized_temperature = float(1.0 / inverse_temperature)

    if optimized_nll < before - 1e-12:
        temperature = optimized_temperature
        after = optimized_nll
        status = "optimized"
    else:
        temperature = 1.0
        after = before
        status = "identity_optimal"

    return TemperatureScalingResult(
        temperature=temperature,
        label_order=labels,
        n_samples=targets.shape[0],
        source_folds=source_folds,
        fitted_labels=tuple(labels[index] for index in fitted_indices),
        excluded_degenerate_labels=tuple(excluded_labels),
        nll_before=before,
        nll_after=after,
        status=status,
        converged=converged,
        optimization_steps=steps,
        temperature_bounds=(lower_temperature, upper_temperature),
    )


def fit_classwise_sigmoid_scaling(
    *,
    logits: ArrayLike,
    y_true: ArrayLike,
    calibration_fold_ids: ArrayLike,
    label_order: Sequence[str] = LABEL_ORDER,
    regularization: float = 1e-3,
    fit_intercept: bool = True,
    slope_bounds: tuple[float, float] = (0.05, 20.0),
    tolerance: float = 1e-12,
    max_steps: int = 128,
) -> ClasswiseSigmoidScalingResult:
    """Fit regularized per-label sigmoid scaling on non-degenerate calibration labels.

    Each label ``k`` with both classes in fold 9 gets its own map
    ``sigmoid(a_k * z_k + b_k)``. ``(a_k, b_k)`` minimizes that label's mean
    binary NLL plus ``regularization * ((a_k - 1)^2 + b_k^2)``. The penalty
    shrinks toward the identity map and makes the objective strictly convex,
    so the constrained minimizer is unique. It is on the per-sample mean-NLL
    scale, so its shrinkage does not vanish as the calibration fold grows.
    ``regularization`` must be at least ``1e-12``: a weaker penalty is below
    the solver's numerical resolution. With ``fit_intercept=False``, ``b_k`` is
    fixed at zero and ``a_k`` is a per-label inverse temperature.

    ``a_k`` is constrained to ``slope_bounds``, whose lower end is strictly
    positive, so each map is strictly increasing and the label's ROC-AUC is
    unchanged in exact arithmetic. In floating point, the transformed logits
    can tie nearly equal scores but never reverse their order; the sigmoid can
    additionally saturate distinct large logits to the same probability.

    The solver is a deterministic damped Newton method with Armijo
    backtracking, started at the identity. If the unconstrained minimizer
    violates a slope bound, the slope is fixed at that bound, which is exact
    for this convex problem, and the intercept is re-optimized. A label stops
    when half its squared Newton decrement is at most ``tolerance`` or after
    ``max_steps`` accepted Newton steps. It also stops, with ``converged``
    false, when no backtracking step satisfies the Armijo condition or when
    rounding makes the computed decrement smaller than half the lower bound
    that a positive-definite Hessian guarantees. A fit is kept only if it
    lowers the regularized objective by more than ``1e-12``, so an
    ``optimized`` label never has a higher calibration-fold NLL than before;
    otherwise the label keeps the identity with status ``identity_optimal``.
    Labels with one observed class keep the identity and are reported as
    excluded.

    This is a library capability only. No frozen pipeline calls it, its
    defaults are engineering defaults rather than preregistered values, and a
    fold-9 NLL decrease is not evidence of better calibration on held-out or
    external data.
    """

    labels = _validate_label_order(label_order)
    targets = _validate_targets(y_true, n_labels=len(labels))
    validated_logits = validate_logits(
        logits, label_order=labels, n_samples=targets.shape[0]
    )
    source_folds = _validate_calibration_fold_ids(
        calibration_fold_ids, n_samples=targets.shape[0]
    )
    penalty = _validate_sigmoid_scaling_regularization(regularization)
    if not isinstance(fit_intercept, bool):
        raise EvaluationValidationError("fit_intercept must be a boolean")
    lower_slope, upper_slope = _validate_slope_bounds(slope_bounds)
    newton_tolerance = _positive_real_number(tolerance, name="tolerance")
    step_limit = _validate_max_steps(max_steps)

    per_label: list[PerLabelSigmoidScaling] = []
    for index, label in enumerate(labels):
        label_targets = targets[:, index].astype(np.float64, copy=False)
        positives = int(targets[:, index].sum())
        negatives = int(targets.shape[0] - positives)
        reason = _degenerate_reason(positives, negatives)
        if reason is not None:
            per_label.append(
                PerLabelSigmoidScaling(
                    label=label,
                    slope=1.0,
                    intercept=0.0,
                    positives=positives,
                    negatives=negatives,
                    nll_before=None,
                    nll_after=None,
                    status=reason,
                    converged=False,
                    optimization_steps=0,
                    active_slope_bound=None,
                )
            )
            continue
        per_label.append(
            _fit_label_sigmoid_scaling(
                label=label,
                logits=validated_logits[:, index],
                targets=label_targets,
                positives=positives,
                negatives=negatives,
                regularization=penalty,
                fit_intercept=fit_intercept,
                slope_bounds=(lower_slope, upper_slope),
                tolerance=newton_tolerance,
                max_steps=step_limit,
            )
        )

    fitted_labels, excluded_labels, nll_before, nll_after, status, converged = (
        _summarize_sigmoid_scaling(per_label)
    )
    return ClasswiseSigmoidScalingResult(
        slopes=tuple(item.slope for item in per_label),
        intercepts=tuple(item.intercept for item in per_label),
        label_order=labels,
        n_samples=targets.shape[0],
        source_folds=source_folds,
        fitted_labels=fitted_labels,
        excluded_degenerate_labels=excluded_labels,
        nll_before=nll_before,
        nll_after=nll_after,
        status=status,
        converged=converged,
        regularization=penalty,
        fit_intercept=fit_intercept,
        slope_bounds=(lower_slope, upper_slope),
        tolerance=newton_tolerance,
        max_steps=step_limit,
        per_label=tuple(per_label),
    )


def compute_selective_predictions(
    y_true: ArrayLike,
    probabilities: ArrayLike,
    *,
    thresholds: Sequence[float] | ThresholdOptimizationResult,
    coverage_targets: Iterable[float] = (1.0, 0.9, 0.8, 0.7, 0.5),
    uncertainty: ArrayLike | None = None,
    label_order: Sequence[str] = LABEL_ORDER,
) -> SelectivePredictionResult:
    """Compute deterministic sample-level abstention results.

    Lower uncertainty is retained first.  With no supplied uncertainty, the
    score is mean normalized binary entropy across the five calibrated output
    probabilities. Hamming loss is the selective risk. ``ceil(target * N)``
    samples are retained so achieved coverage never falls below a nonzero
    requested target because of discrete sample counts.
    """

    labels = _validate_label_order(label_order)
    targets, scores = validate_multilabel_arrays(
        y_true, probabilities, label_order=labels
    )
    resolved_thresholds = _resolve_thresholds(thresholds, labels)
    coverages = _validate_coverage_targets(coverage_targets)
    uncertainty_values, uncertainty_method = _resolve_uncertainty(
        uncertainty, scores
    )

    predictions = scores >= np.asarray(resolved_thresholds, dtype=np.float64)[None, :]
    errors = predictions != targets.astype(np.bool_, copy=False)
    ranking = np.argsort(uncertainty_values, kind="stable")
    all_indices = np.arange(targets.shape[0], dtype=np.int64)
    points: list[SelectiveCoveragePoint] = []
    for target_coverage in coverages:
        selected_count = (
            0
            if target_coverage == 0.0
            else min(targets.shape[0], math.ceil(target_coverage * targets.shape[0]))
        )
        selected = ranking[:selected_count]
        selected_mask = np.zeros(targets.shape[0], dtype=np.bool_)
        selected_mask[selected] = True
        abstained = all_indices[~selected_mask]

        if selected_count:
            selected_errors = errors[selected]
            hamming_risk: float | None = float(selected_errors.mean())
            exact_match: float | None = float((~selected_errors.any(axis=1)).mean())
            per_label_error: tuple[float | None, ...] = tuple(
                float(value) for value in selected_errors.mean(axis=0)
            )
            mean_uncertainty: float | None = float(uncertainty_values[selected].mean())
        else:
            hamming_risk = None
            exact_match = None
            per_label_error = tuple(None for _ in labels)
            mean_uncertainty = None

        points.append(
            SelectiveCoveragePoint(
                target_coverage=target_coverage,
                achieved_coverage=float(selected_count / targets.shape[0]),
                selected_count=selected_count,
                abstained_count=targets.shape[0] - selected_count,
                hamming_risk=hamming_risk,
                exact_match_accuracy=exact_match,
                per_label_error_rate=per_label_error,
                mean_selected_uncertainty=mean_uncertainty,
                selected_indices=tuple(int(index) for index in selected),
                abstained_indices=tuple(int(index) for index in abstained),
            )
        )

    return SelectivePredictionResult(
        n_samples=targets.shape[0],
        label_order=labels,
        thresholds=resolved_thresholds,
        uncertainty_method=uncertainty_method,
        coverage_points=tuple(points),
    )


def stable_sigmoid(logits: ArrayLike) -> FloatArray:
    """Numerically stable elementwise logistic sigmoid."""

    values = _real_float_array(logits, name="logits")
    if not np.all(np.isfinite(values)):
        raise EvaluationValidationError("logits must contain only finite values")
    output = np.empty_like(values, dtype=np.float64)
    nonnegative = values >= 0.0
    output[nonnegative] = 1.0 / (1.0 + np.exp(-values[nonnegative]))
    exponent = np.exp(values[~nonnegative])
    output[~nonnegative] = exponent / (1.0 + exponent)
    return output


def _validate_label_order(label_order: Sequence[str]) -> tuple[str, ...]:
    labels = tuple(label_order)
    if labels != LABEL_ORDER:
        raise EvaluationValidationError(
            "label_order must be exactly "
            f"{LABEL_ORDER!r}; received {labels!r}"
        )
    return labels


def _validate_targets(y_true: ArrayLike, *, n_labels: int) -> IntArray:
    targets = np.asarray(y_true)
    if targets.ndim != 2 or targets.shape[1] != n_labels:
        raise EvaluationValidationError(
            f"y_true must have shape [n_samples, {n_labels}], received {targets.shape}"
        )
    if targets.shape[0] == 0:
        raise EvaluationValidationError("evaluation requires at least one sample")
    if not _is_binary_array(targets):
        raise EvaluationValidationError("y_true must contain only binary values 0 and 1")
    return targets.astype(np.int64, copy=False)


def _real_float_array(values: ArrayLike, *, name: str) -> FloatArray:
    try:
        raw = np.asarray(values)
        if np.iscomplexobj(raw) or (
            raw.dtype.kind == "O" and any(np.iscomplexobj(value) for value in raw.flat)
        ):
            raise EvaluationValidationError(f"{name} must contain real values")
        return np.asarray(raw, dtype=np.float64)
    except EvaluationValidationError:
        raise
    except (TypeError, ValueError, OverflowError) as exc:
        raise EvaluationValidationError(f"{name} must be numeric") from exc


def _validate_score_matrix(
    values: ArrayLike,
    *,
    name: str,
    n_labels: int,
    n_samples: int | None,
) -> FloatArray:
    scores = _real_float_array(values, name=name)
    if scores.ndim != 2 or scores.shape[1] != n_labels:
        raise EvaluationValidationError(
            f"{name} must have shape [n_samples, {n_labels}], received {scores.shape}"
        )
    if scores.shape[0] == 0:
        raise EvaluationValidationError(f"{name} requires at least one sample")
    if n_samples is not None and scores.shape[0] != n_samples:
        raise EvaluationValidationError(
            f"{name} has {scores.shape[0]} samples but expected {n_samples}"
        )
    if not np.all(np.isfinite(scores)):
        raise EvaluationValidationError(f"{name} must contain only finite values")
    return scores


def _is_binary_array(values: NDArray[np.generic]) -> bool:
    if np.iscomplexobj(values):
        return False
    try:
        finite = np.all(np.isfinite(values))
        binary = np.all((values == 0) | (values == 1))
    except TypeError:
        return False
    return bool(finite and binary)


def _validate_ece_bins(n_bins: int) -> int:
    if isinstance(n_bins, bool) or not isinstance(n_bins, int) or n_bins < 2:
        raise EvaluationValidationError("ece_bins must be an integer of at least 2")
    return n_bins


def _degenerate_reason(positives: int, negatives: int) -> str | None:
    if positives == 0:
        return "no_positive_examples"
    if negatives == 0:
        return "no_negative_examples"
    return None


def _binary_roc_auc(targets: IntArray, scores: FloatArray) -> float:
    """Rank-statistic ROC-AUC with average ranks for tied scores."""

    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=np.float64)
    start = 0
    while start < scores.size:
        end = start + 1
        while end < scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_rank = ((start + 1) + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end

    positives = int(targets.sum())
    negatives = int(targets.size - positives)
    positive_rank_sum = float(ranks[targets == 1].sum())
    return float(
        (positive_rank_sum - positives * (positives + 1) / 2.0)
        / (positives * negatives)
    )


def _binary_average_precision(targets: IntArray, scores: FloatArray) -> float:
    """Non-interpolated AP, grouping tied scores before recall increments."""

    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_targets = targets[order]
    true_positives = np.cumsum(sorted_targets, dtype=np.int64)
    false_positives = np.cumsum(1 - sorted_targets, dtype=np.int64)
    group_ends = np.flatnonzero(
        np.r_[sorted_scores[1:] != sorted_scores[:-1], True]
    )
    precision = true_positives[group_ends] / (
        true_positives[group_ends] + false_positives[group_ends]
    )
    recall = true_positives[group_ends] / int(targets.sum())
    recall_increments = np.diff(np.r_[0.0, recall])
    return float(np.sum(recall_increments * precision))


def _optional_mean(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _validate_calibration_fold_ids(
    fold_ids: ArrayLike,
    *,
    n_samples: int,
) -> tuple[int, ...]:
    folds = np.asarray(fold_ids)
    if folds.ndim != 1 or folds.shape[0] != n_samples:
        raise CalibrationLeakageError(
            "calibration_fold_ids must be one-dimensional with one value per sample"
        )
    if folds.dtype.kind not in {"i", "u"}:
        raise CalibrationLeakageError("calibration_fold_ids must contain integers")
    unique_folds = tuple(sorted(int(fold) for fold in np.unique(folds)))
    if unique_folds != CALIBRATION_FOLDS:
        raise CalibrationLeakageError(
            "fitted evaluation artifacts may use calibration fold 9 only; "
            f"received folds {unique_folds}"
        )
    return unique_folds


def _validate_threshold(value: float, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or np.iscomplexobj(value):
        raise EvaluationValidationError(f"{name} must be a real non-boolean number")
    try:
        threshold = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise EvaluationValidationError(f"{name} must be numeric") from error
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise EvaluationValidationError(f"{name} must be finite and in [0, 1]")
    return threshold


def _best_f1_threshold(
    targets: IntArray,
    probabilities: FloatArray,
    *,
    default: float,
) -> tuple[float, float]:
    candidates = np.unique(np.r_[probabilities, default, 0.0, 1.0])
    best_threshold = default
    best_score = -1.0
    for candidate_value in candidates:
        candidate = float(candidate_value)
        predictions = probabilities >= candidate
        positives = targets == 1
        true_positives = int(np.sum(predictions & positives))
        false_positives = int(np.sum(predictions & ~positives))
        false_negatives = int(np.sum(~predictions & positives))
        denominator = 2 * true_positives + false_positives + false_negatives
        score = 0.0 if denominator == 0 else 2.0 * true_positives / denominator
        score_better = score > best_score + 1e-12
        score_tied = abs(score - best_score) <= 1e-12
        tie_break_better = (
            abs(candidate - default),
            -candidate,
        ) < (
            abs(best_threshold - default),
            -best_threshold,
        )
        if score_better or (score_tied and tie_break_better):
            best_threshold = candidate
            best_score = score
    return best_threshold, float(best_score)


def _validate_temperature_bounds(bounds: tuple[float, float]) -> tuple[float, float]:
    if len(bounds) != 2:
        raise EvaluationValidationError("temperature_bounds must contain two values")
    lower, upper = (float(bounds[0]), float(bounds[1]))
    if (
        not math.isfinite(lower)
        or not math.isfinite(upper)
        or lower <= 0.0
        or lower >= upper
        or not lower <= 1.0 <= upper
    ):
        raise EvaluationValidationError(
            "temperature_bounds must be finite, positive, increasing, and include 1.0"
        )
    return lower, upper


def _binary_nll(logits: FloatArray, targets: FloatArray) -> float:
    losses = np.logaddexp(0.0, logits) - targets * logits
    return float(np.mean(losses))


def _golden_section_minimize(
    objective: object,
    *,
    lower: float,
    upper: float,
    tolerance: float,
    max_steps: int,
) -> tuple[float, float, int, bool]:
    if not callable(objective):
        raise TypeError("objective must be callable")
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    left = lower
    right = upper
    interior_right = left + ratio * (right - left)
    interior_left = right - ratio * (right - left)
    left_value = float(objective(interior_left))
    right_value = float(objective(interior_right))
    steps = 0
    while steps < max_steps and right - left > tolerance:
        if left_value <= right_value:
            right = interior_right
            interior_right = interior_left
            right_value = left_value
            interior_left = right - ratio * (right - left)
            left_value = float(objective(interior_left))
        else:
            left = interior_left
            interior_left = interior_right
            left_value = right_value
            interior_right = left + ratio * (right - left)
            right_value = float(objective(interior_right))
        steps += 1
    optimum = (left + right) / 2.0
    return optimum, float(objective(optimum)), steps, right - left <= tolerance


def _fit_label_sigmoid_scaling(
    *,
    label: str,
    logits: FloatArray,
    targets: FloatArray,
    positives: int,
    negatives: int,
    regularization: float,
    fit_intercept: bool,
    slope_bounds: tuple[float, float],
    tolerance: float,
    max_steps: int,
) -> PerLabelSigmoidScaling:
    lower_slope, upper_slope = slope_bounds
    label_logits = np.ascontiguousarray(logits, dtype=np.float64)
    label_targets = np.ascontiguousarray(targets, dtype=np.float64)
    before = _binary_nll(label_logits, label_targets)
    if not math.isfinite(before):
        raise EvaluationValidationError(f"label {label} calibration NLL must be finite")

    slope, intercept, steps, converged = _damped_newton_sigmoid_scaling(
        label_logits,
        label_targets,
        start=(1.0, 0.0),
        free=(0, 1) if fit_intercept else (0,),
        regularization=regularization,
        tolerance=tolerance,
        max_steps=max_steps,
    )
    active_bound: str | None = None
    if not lower_slope <= slope <= upper_slope:
        # The objective is convex, so when its unconstrained minimizer violates
        # the slope interval the constrained minimizer lies on that bound.
        active_bound = "lower" if slope < lower_slope else "upper"
        slope = lower_slope if active_bound == "lower" else upper_slope
        intercept = 0.0
        if fit_intercept:
            _, intercept, extra_steps, intercept_converged = _damped_newton_sigmoid_scaling(
                label_logits,
                label_targets,
                start=(slope, 0.0),
                free=(1,),
                regularization=regularization,
                tolerance=tolerance,
                max_steps=max_steps - steps,
            )
            steps += extra_steps
            converged = converged and intercept_converged

    objective_after = _sigmoid_scaling_objective(
        label_logits, label_targets, slope, intercept, regularization
    )
    # The identity map has zero penalty, so its objective is the raw NLL.
    if objective_after < before - 1e-12:
        after = _binary_nll(slope * label_logits + intercept, label_targets)
        status = "optimized"
    else:
        slope, intercept, after, status, active_bound = 1.0, 0.0, before, "identity_optimal", None
    return PerLabelSigmoidScaling(
        label=label,
        slope=slope,
        intercept=intercept,
        positives=positives,
        negatives=negatives,
        nll_before=before,
        nll_after=after,
        status=status,
        converged=converged,
        optimization_steps=steps,
        active_slope_bound=active_bound,
    )


def _summarize_sigmoid_scaling(
    per_label: Sequence[PerLabelSigmoidScaling],
) -> tuple[tuple[str, ...], tuple[str, ...], float | None, float | None, str, bool]:
    """Return fitted/excluded labels, mean NLL before/after, status, and convergence."""

    fitted = [item for item in per_label if item.status in _SIGMOID_SCALING_FITTED_STATUSES]
    if not fitted:
        status = "no_non_degenerate_labels"
    elif any(item.status == "optimized" for item in fitted):
        status = "optimized"
    else:
        status = "identity_optimal"
    return (
        tuple(item.label for item in fitted),
        tuple(
            item.label
            for item in per_label
            if item.status not in _SIGMOID_SCALING_FITTED_STATUSES
        ),
        _optional_mean([item.nll_before for item in fitted if item.nll_before is not None]),
        _optional_mean([item.nll_after for item in fitted if item.nll_after is not None]),
        status,
        bool(fitted) and all(item.converged for item in fitted),
    )


def _matches_optional_mean(reported: object, expected: float | None) -> bool:
    if reported is None or expected is None:
        return reported is None and expected is None
    value = _finite_real_number(reported, name="summary NLL")
    return math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-15)


def _damped_newton_sigmoid_scaling(
    logits: FloatArray,
    targets: FloatArray,
    *,
    start: tuple[float, float],
    free: tuple[int, ...],
    regularization: float,
    tolerance: float,
    max_steps: int,
) -> tuple[float, float, int, bool]:
    """Minimize the regularized objective over the ``free`` coordinates.

    Returns slope, intercept, accepted steps, and whether half the squared
    Newton decrement reached ``tolerance``. Accepted steps satisfy the Armijo
    condition, so the objective never increases from ``start``. A computed
    decrement below half of ``|g|^2 / trace(H)`` over the free coordinates,
    the least value a positive-definite Hessian allows, means rounding has
    corrupted the direction, so the solve stops without claiming convergence.
    """

    slope, intercept = start
    value = _sigmoid_scaling_objective(logits, targets, slope, intercept, regularization)
    steps = 0
    while True:
        gradient, hessian = _sigmoid_scaling_derivatives(
            logits, targets, slope, intercept, regularization
        )
        direction = _newton_direction(gradient, hessian, free)
        decrement = -(gradient[0] * direction[0] + gradient[1] * direction[1])
        if not math.isfinite(decrement):
            raise EvaluationValidationError("sigmoid-scaling Newton step must be finite")
        if decrement < 0.5 * _newton_decrement_lower_bound(gradient, hessian, free):
            return slope, intercept, steps, False
        if decrement / 2.0 <= tolerance:
            return slope, intercept, steps, True
        if steps >= max_steps:
            return slope, intercept, steps, False
        step_size = 1.0
        for _ in range(_MAX_BACKTRACKING_HALVINGS + 1):
            candidate_slope = slope + step_size * direction[0]
            candidate_intercept = intercept + step_size * direction[1]
            candidate_value = _sigmoid_scaling_objective(
                logits, targets, candidate_slope, candidate_intercept, regularization
            )
            if candidate_value <= value - _ARMIJO_FRACTION * step_size * decrement:
                break
            step_size /= 2.0
        else:
            return slope, intercept, steps, False
        slope, intercept, value = candidate_slope, candidate_intercept, candidate_value
        steps += 1


def _sigmoid_scaling_objective(
    logits: FloatArray,
    targets: FloatArray,
    slope: float,
    intercept: float,
    regularization: float,
) -> float:
    with np.errstate(over="ignore", invalid="ignore"):
        nll = _binary_nll(slope * logits + intercept, targets)
    slope_shift = slope - 1.0
    return nll + regularization * (slope_shift * slope_shift + intercept * intercept)


def _sigmoid_scaling_derivatives(
    logits: FloatArray,
    targets: FloatArray,
    slope: float,
    intercept: float,
    regularization: float,
) -> tuple[tuple[float, float], tuple[float, float, float, float]]:
    """Return the gradient and ``(h_aa, h_ab, h_bb, determinant)``.

    The determinant is assembled from non-negative terms (a weighted logit
    variance plus ridge terms) so it stays positive without cancellation.
    """

    scaled = slope * logits + intercept
    probabilities = stable_sigmoid(scaled)
    weights = probabilities * stable_sigmoid(-scaled)
    residuals = probabilities - targets
    ridge = 2.0 * regularization
    with np.errstate(over="ignore", invalid="ignore"):
        weight_mean = float(np.mean(weights))
        weighted_logit = float(np.mean(weights * logits))
        weighted_square = float(np.mean(weights * np.square(logits)))
        centre = weighted_logit / weight_mean if weight_mean > 0.0 else 0.0
        weighted_spread = float(np.mean(weights * np.square(logits - centre)))
        gradient = (
            float(np.mean(residuals * logits)) + ridge * (slope - 1.0),
            float(np.mean(residuals)) + ridge * intercept,
        )
    hessian = (
        weighted_square + ridge,
        weighted_logit,
        weight_mean + ridge,
        weight_mean * weighted_spread
        + ridge * (weight_mean + weighted_square)
        + ridge * ridge,
    )
    if not all(math.isfinite(value) for value in (*gradient, *hessian)) or hessian[3] <= 0.0:
        raise EvaluationValidationError(
            "sigmoid-scaling derivatives must be finite; logits are too large"
        )
    return gradient, hessian


def _newton_direction(
    gradient: tuple[float, float],
    hessian: tuple[float, float, float, float],
    free: tuple[int, ...],
) -> tuple[float, float]:
    slope_gradient, intercept_gradient = gradient
    h_aa, h_ab, h_bb, determinant = hessian
    if free == (0, 1):
        return (
            -(h_bb * slope_gradient - h_ab * intercept_gradient) / determinant,
            -(h_aa * intercept_gradient - h_ab * slope_gradient) / determinant,
        )
    if free == (0,):
        return -slope_gradient / h_aa, 0.0
    return 0.0, -intercept_gradient / h_bb


def _newton_decrement_lower_bound(
    gradient: tuple[float, float],
    hessian: tuple[float, float, float, float],
    free: tuple[int, ...],
) -> float:
    """Return ``|g|^2 / trace(H)`` over ``free``, a lower bound on ``g' H^-1 g``."""

    diagonal = (hessian[0], hessian[2])
    squared_norm = math.fsum(gradient[index] * gradient[index] for index in free)
    return squared_norm / math.fsum(diagonal[index] for index in free)


def _finite_real_number(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise EvaluationValidationError(f"{name} must be a real non-boolean number")
    number = float(value)
    if not math.isfinite(number):
        raise EvaluationValidationError(f"{name} must be finite")
    return number


def _positive_real_number(value: object, *, name: str) -> float:
    number = _finite_real_number(value, name=name)
    if number <= 0.0:
        raise EvaluationValidationError(f"{name} must be finite and positive")
    return number


def _nonnegative_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EvaluationValidationError(f"{name} must be a non-negative integer")
    return value


def _validate_max_steps(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise EvaluationValidationError("max_steps must be a positive integer")
    return value


def _field_tuple(value: object, *, name: str) -> tuple[object, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise EvaluationValidationError(f"{name} must be a sequence")
    return tuple(value)


def _validate_sigmoid_scaling_regularization(value: object) -> float:
    penalty = _positive_real_number(value, name="regularization")
    if penalty < _MIN_SIGMOID_SCALING_REGULARIZATION:
        raise EvaluationValidationError(
            f"regularization must be at least {_MIN_SIGMOID_SCALING_REGULARIZATION:g}; "
            "a weaker penalty is below the solver's numerical resolution"
        )
    return penalty


def _validate_slope_bounds(bounds: object) -> tuple[float, float]:
    values = _field_tuple(bounds, name="slope_bounds")
    if len(values) != 2:
        raise EvaluationValidationError("slope_bounds must contain two values")
    lower = _finite_real_number(values[0], name="slope_bounds[0]")
    upper = _finite_real_number(values[1], name="slope_bounds[1]")
    if lower <= 0.0 or lower >= upper or not lower <= 1.0 <= upper:
        raise EvaluationValidationError(
            "slope_bounds must be finite, positive, increasing, and include 1.0"
        )
    return lower, upper


def _resolve_thresholds(
    thresholds: Sequence[float] | ThresholdOptimizationResult,
    labels: tuple[str, ...],
) -> tuple[float, ...]:
    if isinstance(thresholds, ThresholdOptimizationResult):
        if thresholds.label_order != labels:
            raise EvaluationValidationError(
                "threshold artifact label order does not match evaluation label order"
            )
        values = thresholds.thresholds
    else:
        values = tuple(thresholds)
    if len(values) != len(labels):
        raise EvaluationValidationError(
            f"thresholds must contain {len(labels)} values, received {len(values)}"
        )
    return tuple(
        _validate_threshold(value, name=f"thresholds[{index}]")
        for index, value in enumerate(values)
    )


def _validate_coverage_targets(values: Iterable[float]) -> tuple[float, ...]:
    coverages = tuple(_validate_threshold(value, name="coverage target") for value in values)
    if not coverages:
        raise EvaluationValidationError("at least one coverage target is required")
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in coverages):
        raise EvaluationValidationError("coverage targets must be finite and in [0, 1]")
    if len(set(coverages)) != len(coverages):
        raise EvaluationValidationError("coverage targets must not contain duplicates")
    return coverages


def _resolve_uncertainty(
    uncertainty: ArrayLike | None,
    probabilities: FloatArray,
) -> tuple[FloatArray, str]:
    if uncertainty is not None:
        values = _real_float_array(uncertainty, name="uncertainty")
        if values.ndim != 1 or values.shape[0] != probabilities.shape[0]:
            raise EvaluationValidationError(
                "uncertainty must be one-dimensional with one value per sample"
            )
        if not np.all(np.isfinite(values)):
            raise EvaluationValidationError("uncertainty must contain only finite values")
        return values, "provided"

    epsilon = np.finfo(np.float64).eps
    clipped = np.clip(probabilities, epsilon, 1.0 - epsilon)
    entropy = -(
        clipped * np.log(clipped) + (1.0 - clipped) * np.log(1.0 - clipped)
    ) / math.log(2.0)
    return entropy.mean(axis=1), "mean_normalized_binary_entropy"
