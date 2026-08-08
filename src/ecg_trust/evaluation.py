"""Leakage-resistant evaluation utilities for PTB-XL multilabel models.

All functions in this module enforce the canonical five-label output order.
Operations that *fit* a decision rule (temperature scaling or thresholds) also
require row-level fold provenance and accept fold 9 only.  Fold 10 therefore
remains an evaluation set rather than an implicit source of fitted parameters.
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
    scores = np.asarray(probabilities, dtype=np.float64)
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

    values = np.asarray(logits, dtype=np.float64)
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


def _validate_score_matrix(
    values: ArrayLike,
    *,
    name: str,
    n_labels: int,
    n_samples: int | None,
) -> FloatArray:
    try:
        scores = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise EvaluationValidationError(f"{name} must be a numeric matrix") from exc
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
    threshold = float(value)
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
    coverages = tuple(float(value) for value in values)
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
        values = np.asarray(uncertainty, dtype=np.float64)
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
