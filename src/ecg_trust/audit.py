"""Patient-cluster uncertainty intervals and subgroup audits.

Bootstrap resampling is performed over patients, not ECG rows, so repeated
records from one patient remain together. Paired comparisons reuse each
patient draw for both models. Subgroup selective coverage applies a single
global abstention ranking to every subgroup; it does not manufacture equal
coverage by selecting a separate quota within each group.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ecg_trust.evaluation import (
    MacroMetrics,
    MultilabelMetrics,
    PerLabelMetrics,
    SelectivePredictionResult,
    ThresholdOptimizationResult,
    compute_multilabel_metrics,
    compute_selective_predictions,
    validate_multilabel_arrays,
)
from ecg_trust.protocol import LABEL_ORDER

IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]
PatientKey = tuple[str, int | str]
MetricKey = tuple[str, str]

_LABEL_METRICS = ("prevalence", "roc_auc", "average_precision", "brier_score", "ece")
_MODEL_LABEL_METRICS = ("roc_auc", "average_precision", "brier_score", "ece")
_MACRO_METRICS = ("roc_auc", "average_precision", "brier_score", "ece")
_HIGHER_IS_BETTER: dict[str, bool | None] = {
    "prevalence": None,
    "roc_auc": True,
    "average_precision": True,
    "brier_score": False,
    "ece": False,
}


class AuditValidationError(ValueError):
    """Raised when bootstrap or subgroup-audit inputs are invalid."""


@dataclass(frozen=True, slots=True)
class BootstrapMetricInterval:
    """Percentile interval with explicit invalid-replicate accounting."""

    metric: str
    estimate: float | None
    lower: float | None
    upper: float | None
    confidence_level: float
    valid_resamples: int
    invalid_resamples: int
    status: str
    higher_is_better: bool | None

    def to_dict(self) -> dict[str, object]:
        return {
            "metric": self.metric,
            "estimate": self.estimate,
            "lower": self.lower,
            "upper": self.upper,
            "confidence_level": self.confidence_level,
            "valid_resamples": self.valid_resamples,
            "invalid_resamples": self.invalid_resamples,
            "status": self.status,
            "higher_is_better": self.higher_is_better,
        }


@dataclass(frozen=True, slots=True)
class BootstrapLabelMetrics:
    """Bootstrap intervals for one label."""

    label: str
    prevalence: BootstrapMetricInterval
    roc_auc: BootstrapMetricInterval
    average_precision: BootstrapMetricInterval
    brier_score: BootstrapMetricInterval
    ece: BootstrapMetricInterval

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "prevalence": self.prevalence.to_dict(),
            "roc_auc": self.roc_auc.to_dict(),
            "average_precision": self.average_precision.to_dict(),
            "brier_score": self.brier_score.to_dict(),
            "ece": self.ece.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class BootstrapMacroMetrics:
    """Bootstrap intervals for macro metrics."""

    roc_auc: BootstrapMetricInterval
    average_precision: BootstrapMetricInterval
    brier_score: BootstrapMetricInterval
    ece: BootstrapMetricInterval

    def to_dict(self) -> dict[str, object]:
        return {
            "roc_auc": self.roc_auc.to_dict(),
            "average_precision": self.average_precision.to_dict(),
            "brier_score": self.brier_score.to_dict(),
            "ece": self.ece.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class PatientClusterBootstrapResult:
    """Serializable confidence intervals from patient-level resampling."""

    n_samples: int
    n_patients: int
    requested_resamples: int
    completed_resamples: int
    seed: int
    confidence_level: float
    minimum_valid_resamples: int
    ece_bins: int
    status: str
    label_order: tuple[str, ...]
    per_label: tuple[BootstrapLabelMetrics, ...]
    macro: BootstrapMacroMetrics

    def to_dict(self) -> dict[str, object]:
        return {
            "method": "patient_cluster_percentile_bootstrap",
            "n_samples": self.n_samples,
            "n_patients": self.n_patients,
            "requested_resamples": self.requested_resamples,
            "completed_resamples": self.completed_resamples,
            "seed": self.seed,
            "confidence_level": self.confidence_level,
            "minimum_valid_resamples": self.minimum_valid_resamples,
            "ece_bins": self.ece_bins,
            "status": self.status,
            "label_order": list(self.label_order),
            "per_label": [item.to_dict() for item in self.per_label],
            "macro": self.macro.to_dict(),
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        serialized = json.dumps(self.to_dict(), indent=indent, sort_keys=True)
        return f"{serialized}\n" if indent is not None else serialized


@dataclass(frozen=True, slots=True)
class PairedLabelDifferences:
    """Model-A-minus-model-B intervals for one label."""

    label: str
    roc_auc: BootstrapMetricInterval
    average_precision: BootstrapMetricInterval
    brier_score: BootstrapMetricInterval
    ece: BootstrapMetricInterval

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "roc_auc": self.roc_auc.to_dict(),
            "average_precision": self.average_precision.to_dict(),
            "brier_score": self.brier_score.to_dict(),
            "ece": self.ece.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class PairedModelDifferenceResult:
    """Paired patient-bootstrap differences between two prediction matrices."""

    model_a: str
    model_b: str
    n_samples: int
    n_patients: int
    requested_resamples: int
    completed_resamples: int
    seed: int
    confidence_level: float
    minimum_valid_resamples: int
    ece_bins: int
    status: str
    label_order: tuple[str, ...]
    per_label: tuple[PairedLabelDifferences, ...]
    macro: BootstrapMacroMetrics

    def to_dict(self) -> dict[str, object]:
        return {
            "method": "paired_patient_cluster_percentile_bootstrap",
            "difference_direction": "model_a_minus_model_b",
            "model_a": self.model_a,
            "model_b": self.model_b,
            "n_samples": self.n_samples,
            "n_patients": self.n_patients,
            "requested_resamples": self.requested_resamples,
            "completed_resamples": self.completed_resamples,
            "seed": self.seed,
            "confidence_level": self.confidence_level,
            "minimum_valid_resamples": self.minimum_valid_resamples,
            "ece_bins": self.ece_bins,
            "status": self.status,
            "label_order": list(self.label_order),
            "per_label": [item.to_dict() for item in self.per_label],
            "macro": self.macro.to_dict(),
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        serialized = json.dumps(self.to_dict(), indent=indent, sort_keys=True)
        return f"{serialized}\n" if indent is not None else serialized


@dataclass(frozen=True, slots=True)
class SubgroupSelectiveCoverage:
    """One subgroup's realized coverage under a global abstention gate."""

    target_global_coverage: float
    achieved_global_coverage: float
    subgroup_coverage: float
    selected_count: int
    abstained_count: int
    hamming_risk: float | None
    exact_match_accuracy: float | None
    status: str

    def to_dict(self) -> dict[str, object]:
        return {
            "target_global_coverage": self.target_global_coverage,
            "achieved_global_coverage": self.achieved_global_coverage,
            "subgroup_coverage": self.subgroup_coverage,
            "selected_count": self.selected_count,
            "abstained_count": self.abstained_count,
            "hamming_risk": self.hamming_risk,
            "exact_match_accuracy": self.exact_match_accuracy,
            "status": self.status,
        }


@dataclass(frozen=True, slots=True)
class SubgroupAuditEntry:
    """Descriptive metrics and selective coverage for one observed group."""

    attribute: str
    group_value: str
    group_value_type: str
    n_samples: int
    n_patients: int
    status: str
    metrics: MultilabelMetrics
    selective_coverage: tuple[SubgroupSelectiveCoverage, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "attribute": self.attribute,
            "group_value": self.group_value,
            "group_value_type": self.group_value_type,
            "n_samples": self.n_samples,
            "n_patients": self.n_patients,
            "status": self.status,
            "metrics": self.metrics.to_dict(),
            "selective_coverage": [point.to_dict() for point in self.selective_coverage],
        }


@dataclass(frozen=True, slots=True)
class SubgroupAuditResult:
    """Full-cohort reference metrics and deterministic subgroup slices."""

    n_samples: int
    n_patients: int
    label_order: tuple[str, ...]
    ece_bins: int
    minimum_group_samples: int
    minimum_group_patients: int
    overall_metrics: MultilabelMetrics
    global_selective_prediction: SelectivePredictionResult
    groups: tuple[SubgroupAuditEntry, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "n_samples": self.n_samples,
            "n_patients": self.n_patients,
            "label_order": list(self.label_order),
            "ece_bins": self.ece_bins,
            "minimum_group_samples": self.minimum_group_samples,
            "minimum_group_patients": self.minimum_group_patients,
            "overall_metrics": self.overall_metrics.to_dict(),
            "global_selective_prediction": self.global_selective_prediction.to_dict(),
            "groups": [group.to_dict() for group in self.groups],
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        serialized = json.dumps(self.to_dict(), indent=indent, sort_keys=True)
        return f"{serialized}\n" if indent is not None else serialized


def bootstrap_multilabel_metrics(
    y_true: ArrayLike,
    probabilities: ArrayLike,
    patient_ids: ArrayLike,
    *,
    n_resamples: int = 1_000,
    seed: int = 20_260_808,
    confidence_level: float = 0.95,
    minimum_valid_resamples: int | None = None,
    label_order: Sequence[str] = LABEL_ORDER,
    ece_bins: int = 15,
) -> PatientClusterBootstrapResult:
    """Estimate metric uncertainty by resampling complete patient clusters."""

    point = compute_multilabel_metrics(
        y_true, probabilities, label_order=label_order, ece_bins=ece_bins
    )
    targets, scores = validate_multilabel_arrays(
        y_true, probabilities, label_order=label_order
    )
    clusters = _build_patient_clusters(patient_ids, n_samples=targets.shape[0])
    settings = _validate_bootstrap_settings(
        n_resamples=n_resamples,
        seed=seed,
        confidence_level=confidence_level,
        minimum_valid_resamples=minimum_valid_resamples,
    )
    requested, resolved_seed, confidence, minimum_valid = settings
    point_values = _extract_metric_values(point, include_prevalence=True)
    distributions: dict[MetricKey, list[float]] = {
        key: [] for key in point_values
    }
    invalid_counts = {key: 0 for key in point_values}

    if len(clusters) < 2:
        completed = 0
        status = "insufficient_patient_clusters"
        invalid_counts = {key: requested for key in point_values}
    else:
        rng = np.random.default_rng(resolved_seed)
        for _ in range(requested):
            indices = _draw_cluster_indices(clusters, rng)
            replicate = compute_multilabel_metrics(
                targets[indices],
                scores[indices],
                label_order=label_order,
                ece_bins=ece_bins,
            )
            _record_values(
                _extract_metric_values(
                    replicate,
                    include_prevalence=True,
                    expected_macro_roc_labels=point.macro.roc_auc_labels,
                    expected_macro_ap_labels=point.macro.average_precision_labels,
                ),
                distributions,
                invalid_counts,
            )
        completed = requested
        status = "ok"

    return PatientClusterBootstrapResult(
        n_samples=targets.shape[0],
        n_patients=len(clusters),
        requested_resamples=requested,
        completed_resamples=completed,
        seed=resolved_seed,
        confidence_level=confidence,
        minimum_valid_resamples=minimum_valid,
        ece_bins=point.ece_bins,
        status=status,
        label_order=point.label_order,
        per_label=_build_label_intervals(
            point,
            distributions,
            invalid_counts,
            confidence_level=confidence,
            minimum_valid=minimum_valid,
            forced_status=status if status != "ok" else None,
        ),
        macro=_build_macro_intervals(
            point.macro,
            distributions,
            invalid_counts,
            confidence_level=confidence,
            minimum_valid=minimum_valid,
            forced_status=status if status != "ok" else None,
        ),
    )


def paired_model_difference_intervals(
    y_true: ArrayLike,
    probabilities_a: ArrayLike,
    probabilities_b: ArrayLike,
    patient_ids: ArrayLike,
    *,
    model_a: str,
    model_b: str,
    n_resamples: int = 1_000,
    seed: int = 20_260_808,
    confidence_level: float = 0.95,
    minimum_valid_resamples: int | None = None,
    label_order: Sequence[str] = LABEL_ORDER,
    ece_bins: int = 15,
) -> PairedModelDifferenceResult:
    """Estimate paired model-A-minus-model-B intervals on patient draws."""

    name_a = _validate_model_name(model_a, "model_a")
    name_b = _validate_model_name(model_b, "model_b")
    if name_a == name_b:
        raise AuditValidationError("model_a and model_b must have distinct names")
    targets, scores_a = validate_multilabel_arrays(
        y_true, probabilities_a, label_order=label_order
    )
    _, scores_b = validate_multilabel_arrays(
        y_true, probabilities_b, label_order=label_order
    )
    clusters = _build_patient_clusters(patient_ids, n_samples=targets.shape[0])
    settings = _validate_bootstrap_settings(
        n_resamples=n_resamples,
        seed=seed,
        confidence_level=confidence_level,
        minimum_valid_resamples=minimum_valid_resamples,
    )
    requested, resolved_seed, confidence, minimum_valid = settings

    point_a = compute_multilabel_metrics(
        targets, scores_a, label_order=label_order, ece_bins=ece_bins
    )
    point_b = compute_multilabel_metrics(
        targets, scores_b, label_order=label_order, ece_bins=ece_bins
    )
    point_values = _metric_differences(point_a, point_b)
    distributions: dict[MetricKey, list[float]] = {
        key: [] for key in point_values
    }
    invalid_counts = {key: 0 for key in point_values}

    if len(clusters) < 2:
        completed = 0
        status = "insufficient_patient_clusters"
        invalid_counts = {key: requested for key in point_values}
    else:
        rng = np.random.default_rng(resolved_seed)
        for _ in range(requested):
            indices = _draw_cluster_indices(clusters, rng)
            replicate_a = compute_multilabel_metrics(
                targets[indices],
                scores_a[indices],
                label_order=label_order,
                ece_bins=ece_bins,
            )
            replicate_b = compute_multilabel_metrics(
                targets[indices],
                scores_b[indices],
                label_order=label_order,
                ece_bins=ece_bins,
            )
            _record_values(
                _metric_differences(
                    replicate_a,
                    replicate_b,
                    expected_macro_roc_labels=point_a.macro.roc_auc_labels,
                    expected_macro_ap_labels=point_a.macro.average_precision_labels,
                ),
                distributions,
                invalid_counts,
            )
        completed = requested
        status = "ok"

    forced_status = status if status != "ok" else None
    return PairedModelDifferenceResult(
        model_a=name_a,
        model_b=name_b,
        n_samples=targets.shape[0],
        n_patients=len(clusters),
        requested_resamples=requested,
        completed_resamples=completed,
        seed=resolved_seed,
        confidence_level=confidence,
        minimum_valid_resamples=minimum_valid,
        ece_bins=point_a.ece_bins,
        status=status,
        label_order=point_a.label_order,
        per_label=_build_paired_label_intervals(
            point_values,
            distributions,
            invalid_counts,
            confidence_level=confidence,
            minimum_valid=minimum_valid,
            forced_status=forced_status,
        ),
        macro=_build_macro_intervals_from_values(
            point_values,
            distributions,
            invalid_counts,
            confidence_level=confidence,
            minimum_valid=minimum_valid,
            forced_status=forced_status,
        ),
    )


def audit_subgroups(
    y_true: ArrayLike,
    probabilities: ArrayLike,
    patient_ids: ArrayLike,
    subgroups: Mapping[str, ArrayLike],
    *,
    thresholds: Sequence[float] | ThresholdOptimizationResult = (0.5,) * 5,
    coverage_targets: Sequence[float] = (1.0, 0.9, 0.8, 0.5),
    uncertainty: ArrayLike | None = None,
    minimum_group_samples: int = 30,
    minimum_group_patients: int = 20,
    label_order: Sequence[str] = LABEL_ORDER,
    ece_bins: int = 15,
) -> SubgroupAuditResult:
    """Audit observed categorical groups under shared global abstention gates.

    Small groups are retained but marked ``small_group_descriptive_only``.
    Missing subgroup values are retained as ``<MISSING>`` instead of being
    silently dropped. Discrimination metrics that become single-class within a
    group are represented as ``None`` by :func:`compute_multilabel_metrics`.
    """

    targets, scores = validate_multilabel_arrays(
        y_true, probabilities, label_order=label_order
    )
    patient_keys, clusters = _validate_patient_ids(
        patient_ids, n_samples=targets.shape[0]
    )
    minimum_samples = _validate_positive_integer(
        minimum_group_samples, "minimum_group_samples"
    )
    minimum_patients = _validate_positive_integer(
        minimum_group_patients, "minimum_group_patients"
    )
    group_arrays = _validate_subgroups(subgroups, n_samples=targets.shape[0])

    overall_metrics = compute_multilabel_metrics(
        targets, scores, label_order=label_order, ece_bins=ece_bins
    )
    overall_selective = compute_selective_predictions(
        targets,
        scores,
        thresholds=thresholds,
        coverage_targets=coverage_targets,
        uncertainty=uncertainty,
        label_order=label_order,
    )
    resolved_thresholds = np.asarray(overall_selective.thresholds, dtype=np.float64)
    predictions = scores >= resolved_thresholds[None, :]
    errors = predictions != targets.astype(np.bool_, copy=False)

    entries: list[SubgroupAuditEntry] = []
    for attribute in sorted(group_arrays):
        group_keys = group_arrays[attribute]
        for group_type, group_value in sorted(set(group_keys)):
            member_indices = np.asarray(
                [index for index, key in enumerate(group_keys) if key == (group_type, group_value)],
                dtype=np.int64,
            )
            member_patient_count = len({patient_keys[index] for index in member_indices})
            group_status = (
                "small_group_descriptive_only"
                if member_indices.size < minimum_samples
                or member_patient_count < minimum_patients
                else "ok"
            )
            metrics = compute_multilabel_metrics(
                targets[member_indices],
                scores[member_indices],
                label_order=label_order,
                ece_bins=ece_bins,
            )
            selective_points: list[SubgroupSelectiveCoverage] = []
            member_mask = np.zeros(targets.shape[0], dtype=np.bool_)
            member_mask[member_indices] = True
            for global_point in overall_selective.coverage_points:
                selected_mask = np.zeros(targets.shape[0], dtype=np.bool_)
                selected_mask[list(global_point.selected_indices)] = True
                selected_members = np.flatnonzero(member_mask & selected_mask)
                selected_count = int(selected_members.size)
                if selected_count:
                    selected_errors = errors[selected_members]
                    hamming_risk: float | None = float(selected_errors.mean())
                    exact_match: float | None = float(
                        (~selected_errors.any(axis=1)).mean()
                    )
                    selective_status = "ok"
                else:
                    hamming_risk = None
                    exact_match = None
                    selective_status = "no_selected_samples"
                selective_points.append(
                    SubgroupSelectiveCoverage(
                        target_global_coverage=global_point.target_coverage,
                        achieved_global_coverage=global_point.achieved_coverage,
                        subgroup_coverage=float(selected_count / member_indices.size),
                        selected_count=selected_count,
                        abstained_count=int(member_indices.size - selected_count),
                        hamming_risk=hamming_risk,
                        exact_match_accuracy=exact_match,
                        status=selective_status,
                    )
                )
            entries.append(
                SubgroupAuditEntry(
                    attribute=attribute,
                    group_value=group_value,
                    group_value_type=group_type,
                    n_samples=int(member_indices.size),
                    n_patients=member_patient_count,
                    status=group_status,
                    metrics=metrics,
                    selective_coverage=tuple(selective_points),
                )
            )

    return SubgroupAuditResult(
        n_samples=targets.shape[0],
        n_patients=len(clusters),
        label_order=overall_metrics.label_order,
        ece_bins=overall_metrics.ece_bins,
        minimum_group_samples=minimum_samples,
        minimum_group_patients=minimum_patients,
        overall_metrics=overall_metrics,
        global_selective_prediction=overall_selective,
        groups=tuple(entries),
    )


def _build_label_intervals(
    point: MultilabelMetrics,
    distributions: dict[MetricKey, list[float]],
    invalid_counts: dict[MetricKey, int],
    *,
    confidence_level: float,
    minimum_valid: int,
    forced_status: str | None,
) -> tuple[BootstrapLabelMetrics, ...]:
    summaries: list[BootstrapLabelMetrics] = []
    for label_metrics in point.per_label:
        label = label_metrics.label
        intervals = {
            metric: _make_interval(
                metric=metric,
                estimate=_label_metric_value(label_metrics, metric),
                samples=distributions[(label, metric)],
                invalid_resamples=invalid_counts[(label, metric)],
                confidence_level=confidence_level,
                minimum_valid=minimum_valid,
                forced_status=forced_status,
            )
            for metric in _LABEL_METRICS
        }
        summaries.append(
            BootstrapLabelMetrics(
                label=label,
                prevalence=intervals["prevalence"],
                roc_auc=intervals["roc_auc"],
                average_precision=intervals["average_precision"],
                brier_score=intervals["brier_score"],
                ece=intervals["ece"],
            )
        )
    return tuple(summaries)


def _build_macro_intervals(
    point: MacroMetrics,
    distributions: dict[MetricKey, list[float]],
    invalid_counts: dict[MetricKey, int],
    *,
    confidence_level: float,
    minimum_valid: int,
    forced_status: str | None,
) -> BootstrapMacroMetrics:
    values = {metric: _macro_metric_value(point, metric) for metric in _MACRO_METRICS}
    return _build_macro_intervals_from_values(
        {("macro", metric): value for metric, value in values.items()},
        distributions,
        invalid_counts,
        confidence_level=confidence_level,
        minimum_valid=minimum_valid,
        forced_status=forced_status,
    )


def _build_macro_intervals_from_values(
    point_values: dict[MetricKey, float | None],
    distributions: dict[MetricKey, list[float]],
    invalid_counts: dict[MetricKey, int],
    *,
    confidence_level: float,
    minimum_valid: int,
    forced_status: str | None,
) -> BootstrapMacroMetrics:
    intervals = {
        metric: _make_interval(
            metric=metric,
            estimate=point_values[("macro", metric)],
            samples=distributions[("macro", metric)],
            invalid_resamples=invalid_counts[("macro", metric)],
            confidence_level=confidence_level,
            minimum_valid=minimum_valid,
            forced_status=forced_status,
        )
        for metric in _MACRO_METRICS
    }
    return BootstrapMacroMetrics(
        roc_auc=intervals["roc_auc"],
        average_precision=intervals["average_precision"],
        brier_score=intervals["brier_score"],
        ece=intervals["ece"],
    )


def _build_paired_label_intervals(
    point_values: dict[MetricKey, float | None],
    distributions: dict[MetricKey, list[float]],
    invalid_counts: dict[MetricKey, int],
    *,
    confidence_level: float,
    minimum_valid: int,
    forced_status: str | None,
) -> tuple[PairedLabelDifferences, ...]:
    summaries: list[PairedLabelDifferences] = []
    for label in LABEL_ORDER:
        intervals = {
            metric: _make_interval(
                metric=metric,
                estimate=point_values[(label, metric)],
                samples=distributions[(label, metric)],
                invalid_resamples=invalid_counts[(label, metric)],
                confidence_level=confidence_level,
                minimum_valid=minimum_valid,
                forced_status=forced_status,
            )
            for metric in _MODEL_LABEL_METRICS
        }
        summaries.append(
            PairedLabelDifferences(
                label=label,
                roc_auc=intervals["roc_auc"],
                average_precision=intervals["average_precision"],
                brier_score=intervals["brier_score"],
                ece=intervals["ece"],
            )
        )
    return tuple(summaries)


def _make_interval(
    *,
    metric: str,
    estimate: float | None,
    samples: list[float],
    invalid_resamples: int,
    confidence_level: float,
    minimum_valid: int,
    forced_status: str | None,
) -> BootstrapMetricInterval:
    valid = len(samples)
    if forced_status is not None:
        lower = None
        upper = None
        status = forced_status
    elif estimate is None:
        lower = None
        upper = None
        status = "undefined_point_estimate"
    elif valid < minimum_valid:
        lower = None
        upper = None
        status = "insufficient_valid_resamples"
    else:
        alpha = 1.0 - confidence_level
        lower, upper = (
            float(value)
            for value in np.quantile(samples, [alpha / 2.0, 1.0 - alpha / 2.0])
        )
        status = "ok_with_degenerate_replicates" if invalid_resamples else "ok"
    return BootstrapMetricInterval(
        metric=metric,
        estimate=estimate,
        lower=lower,
        upper=upper,
        confidence_level=confidence_level,
        valid_resamples=valid,
        invalid_resamples=invalid_resamples,
        status=status,
        higher_is_better=_HIGHER_IS_BETTER[metric],
    )


def _extract_metric_values(
    report: MultilabelMetrics,
    *,
    include_prevalence: bool,
    expected_macro_roc_labels: int | None = None,
    expected_macro_ap_labels: int | None = None,
) -> dict[MetricKey, float | None]:
    metrics = _LABEL_METRICS if include_prevalence else _MODEL_LABEL_METRICS
    values: dict[MetricKey, float | None] = {}
    for label_metrics in report.per_label:
        for metric in metrics:
            values[(label_metrics.label, metric)] = _label_metric_value(
                label_metrics, metric
            )
    for metric in _MACRO_METRICS:
        value = _macro_metric_value(report.macro, metric)
        if (
            metric == "roc_auc"
            and expected_macro_roc_labels is not None
            and report.macro.roc_auc_labels != expected_macro_roc_labels
        ):
            value = None
        if (
            metric == "average_precision"
            and expected_macro_ap_labels is not None
            and report.macro.average_precision_labels != expected_macro_ap_labels
        ):
            value = None
        values[("macro", metric)] = value
    return values


def _label_metric_value(label_metrics: PerLabelMetrics, metric: str) -> float | None:
    value = getattr(label_metrics, metric)
    return None if value is None else float(value)


def _macro_metric_value(macro: MacroMetrics, metric: str) -> float | None:
    value = getattr(macro, metric)
    return None if value is None else float(value)


def _metric_differences(
    report_a: MultilabelMetrics,
    report_b: MultilabelMetrics,
    *,
    expected_macro_roc_labels: int | None = None,
    expected_macro_ap_labels: int | None = None,
) -> dict[MetricKey, float | None]:
    values_a = _extract_metric_values(
        report_a,
        include_prevalence=False,
        expected_macro_roc_labels=expected_macro_roc_labels,
        expected_macro_ap_labels=expected_macro_ap_labels,
    )
    values_b = _extract_metric_values(
        report_b,
        include_prevalence=False,
        expected_macro_roc_labels=expected_macro_roc_labels,
        expected_macro_ap_labels=expected_macro_ap_labels,
    )
    return {
        key: _optional_difference(values_a[key], values_b[key])
        for key in values_a
    }


def _optional_difference(value_a: float | None, value_b: float | None) -> float | None:
    if value_a is None or value_b is None:
        return None
    return float(value_a - value_b)


def _record_values(
    values: dict[MetricKey, float | None],
    distributions: dict[MetricKey, list[float]],
    invalid_counts: dict[MetricKey, int],
) -> None:
    for key, value in values.items():
        if value is None or not math.isfinite(value):
            invalid_counts[key] += 1
        else:
            distributions[key].append(value)


def _validate_bootstrap_settings(
    *,
    n_resamples: int,
    seed: int,
    confidence_level: float,
    minimum_valid_resamples: int | None,
) -> tuple[int, int, float, int]:
    requested = _validate_positive_integer(n_resamples, "n_resamples")
    if requested < 2:
        raise AuditValidationError("n_resamples must be at least 2")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise AuditValidationError("seed must be a non-negative integer")
    confidence = float(confidence_level)
    if not math.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise AuditValidationError("confidence_level must be finite and strictly between 0 and 1")
    if minimum_valid_resamples is None:
        minimum_valid = max(2, math.ceil(requested * 0.5))
    else:
        minimum_valid = _validate_positive_integer(
            minimum_valid_resamples, "minimum_valid_resamples"
        )
        if minimum_valid > requested:
            raise AuditValidationError(
                "minimum_valid_resamples cannot exceed n_resamples"
            )
    return requested, seed, confidence, minimum_valid


def _validate_positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise AuditValidationError(f"{name} must be a positive integer")
    return value


def _build_patient_clusters(patient_ids: ArrayLike, *, n_samples: int) -> tuple[IntArray, ...]:
    _, clusters = _validate_patient_ids(patient_ids, n_samples=n_samples)
    return clusters


def _validate_patient_ids(
    patient_ids: ArrayLike,
    *,
    n_samples: int,
) -> tuple[tuple[PatientKey, ...], tuple[IntArray, ...]]:
    try:
        raw_ids = np.asarray(patient_ids, dtype=object)
    except (TypeError, ValueError) as exc:
        raise AuditValidationError("patient_ids must be a one-dimensional array") from exc
    if raw_ids.ndim != 1 or raw_ids.shape[0] != n_samples:
        raise AuditValidationError(
            "patient_ids must be one-dimensional with one value per sample"
        )
    keys = tuple(_normalize_patient_id(value) for value in raw_ids)
    grouped: dict[PatientKey, list[int]] = {}
    for index, key in enumerate(keys):
        grouped.setdefault(key, []).append(index)
    ordered_keys = sorted(grouped, key=lambda key: (key[0], str(key[1])))
    clusters = tuple(
        np.asarray(grouped[key], dtype=np.int64) for key in ordered_keys
    )
    return keys, clusters


def _normalize_patient_id(value: object) -> PatientKey:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool) or value is None:
        raise AuditValidationError("patient_ids cannot contain booleans or missing values")
    if isinstance(value, int):
        return ("integer", value)
    if isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            raise AuditValidationError(
                "numeric patient_ids must be finite integer values"
            )
        return ("integer", int(value))
    if isinstance(value, str) and value.strip():
        return ("string", value)
    raise AuditValidationError(
        "patient_ids must contain non-empty strings or integer-valued numbers"
    )


def _draw_cluster_indices(
    clusters: tuple[IntArray, ...],
    rng: np.random.Generator,
) -> IntArray:
    draws = rng.integers(0, len(clusters), size=len(clusters))
    return np.concatenate([clusters[int(draw)] for draw in draws]).astype(
        np.int64, copy=False
    )


def _validate_model_name(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AuditValidationError(f"{name} must be a non-empty string")
    return value.strip()


def _validate_subgroups(
    subgroups: Mapping[str, ArrayLike],
    *,
    n_samples: int,
) -> dict[str, tuple[tuple[str, str], ...]]:
    if not subgroups:
        raise AuditValidationError("at least one subgroup attribute is required")
    normalized: dict[str, tuple[tuple[str, str], ...]] = {}
    for attribute, values in subgroups.items():
        if not isinstance(attribute, str) or not attribute.strip():
            raise AuditValidationError("subgroup attribute names must be non-empty strings")
        try:
            array = np.asarray(values, dtype=object)
        except (TypeError, ValueError) as exc:
            raise AuditValidationError(
                f"subgroup {attribute!r} must be one-dimensional"
            ) from exc
        if array.ndim != 1 or array.shape[0] != n_samples:
            raise AuditValidationError(
                f"subgroup {attribute!r} must have one value per sample"
            )
        normalized_attribute = attribute.strip()
        if normalized_attribute in normalized:
            raise AuditValidationError(
                f"duplicate subgroup attribute after trimming: {normalized_attribute!r}"
            )
        normalized[normalized_attribute] = tuple(
            _normalize_group_value(value) for value in array
        )
    return normalized


def _normalize_group_value(value: object) -> tuple[str, str]:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ("missing", "<MISSING>")
    if isinstance(value, bool):
        return ("boolean", "true" if value else "false")
    if isinstance(value, int):
        return ("integer", str(value))
    if isinstance(value, float):
        if not math.isfinite(value):
            raise AuditValidationError("subgroup values cannot contain infinities")
        return ("number", repr(value))
    if isinstance(value, str):
        return ("string", value)
    raise AuditValidationError(
        "subgroup values must be strings, numbers, booleans, or missing"
    )
