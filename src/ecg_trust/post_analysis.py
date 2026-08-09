"""Read-only post-evaluation probability and selective-risk analyses.

The functions in this module never fit a calibrator, threshold, or abstention
gate.  They derive additional descriptive results from frozen logits,
probabilities, targets, and fold-9 decisions after the final batch is sealed.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ecg_trust.evaluation import compute_multilabel_metrics, validate_multilabel_arrays
from ecg_trust.protocol import LABEL_ORDER

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int64]


class PostAnalysisError(ValueError):
    """Raised when a post-evaluation input violates its frozen contract."""


@dataclass(frozen=True, slots=True)
class ReliabilityBin:
    """One equal-mass probability bin."""

    count: int
    probability_minimum: float
    probability_maximum: float
    mean_probability: float
    event_rate: float

    def to_dict(self) -> dict[str, object]:
        return {
            "count": self.count,
            "probability_minimum": self.probability_minimum,
            "probability_maximum": self.probability_maximum,
            "mean_probability": self.mean_probability,
            "event_rate": self.event_rate,
        }


@dataclass(frozen=True, slots=True)
class ReliabilityCurve:
    """Equal-mass reliability data for one label."""

    label: str
    requested_bins: int
    bins: tuple[ReliabilityBin, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "strategy": "equal_mass_stable_sort",
            "requested_bins": self.requested_bins,
            "realized_bins": len(self.bins),
            "bins": [item.to_dict() for item in self.bins],
        }


@dataclass(frozen=True, slots=True)
class DenseRiskCoverage:
    """Full entropy-ranked selective-risk curve and scalar summaries."""

    coverage: FloatArray
    uncertainty_cutoff: FloatArray
    hamming_risk: FloatArray
    exact_match_error: FloatArray
    brier_score: FloatArray
    log_loss: FloatArray
    aurc_hamming: float
    aurc_exact_match_error: float
    aurc_brier: float
    aurc_log_loss: float

    def to_dict(self) -> dict[str, object]:
        return {
            "ordering": "ascending_mean_normalized_binary_entropy_stable_index_tiebreak",
            "area_method": "mean_cumulative_risk_over_all_prefix_coverages",
            "coverage": self.coverage.tolist(),
            "uncertainty_cutoff": self.uncertainty_cutoff.tolist(),
            "hamming_risk": self.hamming_risk.tolist(),
            "exact_match_error": self.exact_match_error.tolist(),
            "brier_score": self.brier_score.tolist(),
            "log_loss": self.log_loss.tolist(),
            "aurc": {
                "hamming_risk": self.aurc_hamming,
                "exact_match_error": self.aurc_exact_match_error,
                "brier_score": self.aurc_brier,
                "log_loss": self.aurc_log_loss,
            },
        }


def mean_normalized_binary_entropy(probabilities: ArrayLike) -> FloatArray:
    """Match the frozen gate's mean normalized binary-entropy score."""

    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1:
        raise PostAnalysisError("probabilities must be a non-empty two-dimensional matrix")
    if not np.all(np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
        raise PostAnalysisError("probabilities must be finite and lie in [0, 1]")
    epsilon = np.finfo(np.float64).eps
    clipped = np.clip(values, epsilon, 1.0 - epsilon)
    entropy = -(
        clipped * np.log(clipped) + (1.0 - clipped) * np.log(1.0 - clipped)
    ) / math.log(2.0)
    return np.asarray(entropy.mean(axis=1), dtype=np.float64)


def multilabel_log_loss(y_true: ArrayLike, probabilities: ArrayLike) -> float:
    """Mean binary cross-entropy across records and labels."""

    targets, scores = validate_multilabel_arrays(y_true, probabilities)
    epsilon = np.finfo(np.float64).eps
    clipped = np.clip(scores, epsilon, 1.0 - epsilon)
    losses = -(
        targets.astype(np.float64) * np.log(clipped)
        + (1.0 - targets.astype(np.float64)) * np.log(1.0 - clipped)
    )
    return float(losses.mean())


def reliability_curves(
    y_true: ArrayLike,
    probabilities: ArrayLike,
    *,
    n_bins: int = 15,
    label_order: Sequence[str] = LABEL_ORDER,
) -> tuple[ReliabilityCurve, ...]:
    """Build deterministic equal-mass reliability curves for every label."""

    labels = tuple(label_order)
    if labels != LABEL_ORDER:
        raise PostAnalysisError("label_order must match the canonical superclass order")
    if isinstance(n_bins, bool) or not isinstance(n_bins, int) or n_bins < 2:
        raise PostAnalysisError("n_bins must be an integer of at least 2")
    targets, scores = validate_multilabel_arrays(
        y_true, probabilities, label_order=labels
    )
    realized = min(n_bins, targets.shape[0])
    curves: list[ReliabilityCurve] = []
    for label_index, label in enumerate(labels):
        order = np.argsort(scores[:, label_index], kind="stable")
        partitions = np.array_split(order, realized)
        bins: list[ReliabilityBin] = []
        for partition in partitions:
            if partition.size == 0:
                continue
            values = scores[partition, label_index]
            outcomes = targets[partition, label_index]
            bins.append(
                ReliabilityBin(
                    count=int(partition.size),
                    probability_minimum=float(values.min()),
                    probability_maximum=float(values.max()),
                    mean_probability=float(values.mean()),
                    event_rate=float(outcomes.mean()),
                )
            )
        curves.append(ReliabilityCurve(label=label, requested_bins=n_bins, bins=tuple(bins)))
    return tuple(curves)


def dense_risk_coverage(
    y_true: ArrayLike,
    probabilities: ArrayLike,
    *,
    thresholds: Sequence[float],
    uncertainty: ArrayLike | None = None,
) -> DenseRiskCoverage:
    """Evaluate every least-uncertain prefix without fitting a cutoff."""

    targets, scores = validate_multilabel_arrays(y_true, probabilities)
    threshold_values = _thresholds(thresholds, scores.shape[1])
    entropy = (
        mean_normalized_binary_entropy(scores)
        if uncertainty is None
        else _uncertainty(uncertainty, scores.shape[0])
    )
    predictions = scores >= threshold_values[None, :]
    target_bool = targets.astype(np.bool_, copy=False)
    per_sample_hamming = np.not_equal(predictions, target_bool).mean(axis=1)
    per_sample_exact_error = np.any(np.not_equal(predictions, target_bool), axis=1).astype(
        np.float64
    )
    per_sample_brier = np.square(scores - targets).mean(axis=1)
    epsilon = np.finfo(np.float64).eps
    clipped = np.clip(scores, epsilon, 1.0 - epsilon)
    per_sample_log_loss = -(
        targets * np.log(clipped) + (1 - targets) * np.log(1.0 - clipped)
    ).mean(axis=1)

    order = np.argsort(entropy, kind="stable")
    count = targets.shape[0]
    coverage = np.arange(1, count + 1, dtype=np.float64) / count
    denominators = np.arange(1, count + 1, dtype=np.float64)

    def cumulative(values: FloatArray) -> FloatArray:
        return np.asarray(np.cumsum(values[order], dtype=np.float64) / denominators)

    hamming = cumulative(np.asarray(per_sample_hamming, dtype=np.float64))
    exact_error = cumulative(per_sample_exact_error)
    brier = cumulative(np.asarray(per_sample_brier, dtype=np.float64))
    log_loss = cumulative(np.asarray(per_sample_log_loss, dtype=np.float64))
    return DenseRiskCoverage(
        coverage=coverage,
        uncertainty_cutoff=np.asarray(entropy[order], dtype=np.float64),
        hamming_risk=hamming,
        exact_match_error=exact_error,
        brier_score=brier,
        log_loss=log_loss,
        aurc_hamming=float(hamming.mean()),
        aurc_exact_match_error=float(exact_error.mean()),
        aurc_brier=float(brier.mean()),
        aurc_log_loss=float(log_loss.mean()),
    )


def error_detection_metrics(
    y_true: ArrayLike,
    probabilities: ArrayLike,
    *,
    thresholds: Sequence[float],
    uncertainty: ArrayLike | None = None,
) -> dict[str, object]:
    """Measure how well uncertainty detects any thresholded label error."""

    targets, scores = validate_multilabel_arrays(y_true, probabilities)
    threshold_values = _thresholds(thresholds, scores.shape[1])
    entropy = (
        mean_normalized_binary_entropy(scores)
        if uncertainty is None
        else _uncertainty(uncertainty, scores.shape[0])
    )
    errors = np.any(
        np.not_equal(scores >= threshold_values[None, :], targets.astype(np.bool_)),
        axis=1,
    ).astype(np.int64)
    positives = int(errors.sum())
    negatives = int(errors.size - positives)
    if positives == 0 or negatives == 0:
        return {
            "target": "any_thresholded_label_error",
            "uncertainty": "mean_normalized_binary_entropy",
            "positives": positives,
            "negatives": negatives,
            "roc_auc": None,
            "average_precision": None,
            "status": "degenerate_error_target",
        }
    return {
        "target": "any_thresholded_label_error",
        "uncertainty": "mean_normalized_binary_entropy",
        "positives": positives,
        "negatives": negatives,
        "roc_auc": _binary_roc_auc(errors, entropy),
        "average_precision": _binary_average_precision(errors, entropy),
        "status": "ok",
    }


def frozen_gate_audit(
    y_true: ArrayLike,
    probabilities: ArrayLike,
    *,
    thresholds: Sequence[float],
    gates: Sequence[Mapping[str, object]],
    subgroups: Mapping[str, ArrayLike] | None = None,
) -> tuple[dict[str, object], ...]:
    """Apply immutable entropy cutoffs and report proper scores/composition."""

    targets, scores = validate_multilabel_arrays(y_true, probabilities)
    threshold_values = _thresholds(thresholds, scores.shape[1])
    entropy = mean_normalized_binary_entropy(scores)
    predictions = scores >= threshold_values[None, :]
    target_bool = targets.astype(np.bool_)
    normalized_groups = _subgroups(subgroups, targets.shape[0])
    results: list[dict[str, object]] = []
    previous_target = math.inf
    for gate in gates:
        target_coverage = _finite_float(gate.get("target_coverage"), "target_coverage")
        maximum_entropy = _finite_float(gate.get("maximum_entropy"), "maximum_entropy")
        if not 0.0 < target_coverage <= 1.0 or not 0.0 <= maximum_entropy <= 1.0:
            raise PostAnalysisError("gate coverage and entropy must lie in their valid ranges")
        if target_coverage >= previous_target:
            raise PostAnalysisError("gates must be ordered by strictly decreasing coverage")
        previous_target = target_coverage
        selected = entropy <= maximum_entropy
        accepted = int(selected.sum())
        rejected = int(selected.size - accepted)
        if accepted == 0:
            raise PostAnalysisError("a frozen gate selected zero records")
        accepted_targets = targets[selected]
        accepted_scores = scores[selected]
        accepted_predictions = predictions[selected]
        accepted_truth = target_bool[selected]
        rejected_targets = targets[~selected]
        entry: dict[str, object] = {
            "target_coverage": target_coverage,
            "maximum_entropy": maximum_entropy,
            "achieved_coverage": float(accepted / selected.size),
            "selected_count": accepted,
            "abstained_count": rejected,
            "hamming_risk": float(np.not_equal(accepted_predictions, accepted_truth).mean()),
            "exact_match_accuracy": float(
                np.all(np.equal(accepted_predictions, accepted_truth), axis=1).mean()
            ),
            "brier_score": float(np.square(accepted_scores - accepted_targets).mean()),
            "log_loss": multilabel_log_loss(accepted_targets, accepted_scores),
            "accepted_prevalence": accepted_targets.mean(axis=0).tolist(),
            "rejected_prevalence": (
                rejected_targets.mean(axis=0).tolist() if rejected else [None] * scores.shape[1]
            ),
        }
        if normalized_groups:
            group_coverage: dict[str, dict[str, object]] = {}
            for attribute, values in normalized_groups.items():
                attribute_results: dict[str, object] = {}
                for value in sorted(set(values.tolist()), key=str):
                    members = values == value
                    attribute_results[str(value)] = {
                        "count": int(members.sum()),
                        "selected_count": int(np.logical_and(members, selected).sum()),
                        "coverage": float(selected[members].mean()),
                    }
                group_coverage[attribute] = attribute_results
            entry["subgroup_coverage"] = group_coverage
        results.append(entry)
    return tuple(results)


def derive_probability_audit(
    y_true: ArrayLike,
    raw_probabilities: ArrayLike,
    calibrated_probabilities: ArrayLike,
    *,
    thresholds: Sequence[float],
    gates: Sequence[Mapping[str, object]],
    subgroups: Mapping[str, ArrayLike] | None = None,
    reliability_bins: int = 15,
) -> dict[str, object]:
    """Produce the complete read-only probability audit for one frozen member."""

    targets, raw = validate_multilabel_arrays(y_true, raw_probabilities)
    calibrated_targets, calibrated = validate_multilabel_arrays(
        y_true, calibrated_probabilities
    )
    if not np.array_equal(targets, calibrated_targets):
        raise PostAnalysisError("raw and calibrated target rows differ")
    dense = dense_risk_coverage(targets, calibrated, thresholds=thresholds)
    return {
        "label_order": list(LABEL_ORDER),
        "n_samples": int(targets.shape[0]),
        "raw": {
            "metrics": compute_multilabel_metrics(targets, raw).to_dict(),
            "log_loss": multilabel_log_loss(targets, raw),
            "reliability": [
                curve.to_dict()
                for curve in reliability_curves(targets, raw, n_bins=reliability_bins)
            ],
        },
        "calibrated": {
            "metrics": compute_multilabel_metrics(targets, calibrated).to_dict(),
            "log_loss": multilabel_log_loss(targets, calibrated),
            "reliability": [
                curve.to_dict()
                for curve in reliability_curves(
                    targets, calibrated, n_bins=reliability_bins
                )
            ],
        },
        "dense_risk_coverage": dense.to_dict(),
        "error_detection": error_detection_metrics(
            targets, calibrated, thresholds=thresholds
        ),
        "frozen_gates": list(
            frozen_gate_audit(
                targets,
                calibrated,
                thresholds=thresholds,
                gates=gates,
                subgroups=subgroups,
            )
        ),
    }


def _thresholds(values: Sequence[float], n_labels: int) -> FloatArray:
    parsed = np.asarray(tuple(values), dtype=np.float64)
    if parsed.shape != (n_labels,) or not np.all(np.isfinite(parsed)):
        raise PostAnalysisError(f"thresholds must contain {n_labels} finite values")
    if np.any((parsed < 0.0) | (parsed > 1.0)):
        raise PostAnalysisError("thresholds must lie in [0, 1]")
    return parsed


def _uncertainty(values: ArrayLike, n_samples: int) -> FloatArray:
    parsed = np.asarray(values, dtype=np.float64)
    if parsed.shape != (n_samples,) or not np.all(np.isfinite(parsed)):
        raise PostAnalysisError("uncertainty must be one finite value per sample")
    return parsed


def _subgroups(
    values: Mapping[str, ArrayLike] | None, n_samples: int
) -> dict[str, NDArray[np.object_]]:
    if values is None:
        return {}
    result: dict[str, NDArray[np.object_]] = {}
    for name, raw in values.items():
        if not isinstance(name, str) or not name.strip():
            raise PostAnalysisError("subgroup attribute names must be non-empty strings")
        parsed = np.asarray(raw, dtype=object)
        if parsed.shape != (n_samples,):
            raise PostAnalysisError("subgroup arrays must align with prediction rows")
        if any(item is None or not str(item).strip() for item in parsed.tolist()):
            raise PostAnalysisError("subgroup values must be non-empty")
        result[name] = parsed
    return result


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PostAnalysisError(f"{name} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise PostAnalysisError(f"{name} must be finite")
    return parsed


def _binary_roc_auc(targets: IntArray, scores: FloatArray) -> float:
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=np.float64)
    start = 0
    while start < scores.size:
        end = start + 1
        while end < scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    positives = targets == 1
    positive_count = int(positives.sum())
    negative_count = int(targets.size - positive_count)
    rank_sum = float(ranks[positives].sum())
    return float(
        (rank_sum - positive_count * (positive_count + 1) / 2.0)
        / (positive_count * negative_count)
    )


def _binary_average_precision(targets: IntArray, scores: FloatArray) -> float:
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_targets = targets[order]
    positives = int(sorted_targets.sum())
    true_positives = 0
    false_positives = 0
    previous_recall = 0.0
    average_precision = 0.0
    start = 0
    while start < targets.size:
        end = start + 1
        while end < targets.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        group = sorted_targets[start:end]
        true_positives += int(group.sum())
        false_positives += int(group.size - group.sum())
        recall = true_positives / positives
        precision = true_positives / (true_positives + false_positives)
        average_precision += (recall - previous_recall) * precision
        previous_recall = recall
        start = end
    return float(average_precision)


__all__ = [
    "DenseRiskCoverage",
    "PostAnalysisError",
    "ReliabilityBin",
    "ReliabilityCurve",
    "dense_risk_coverage",
    "derive_probability_audit",
    "error_detection_metrics",
    "frozen_gate_audit",
    "mean_normalized_binary_entropy",
    "multilabel_log_loss",
    "reliability_curves",
]
