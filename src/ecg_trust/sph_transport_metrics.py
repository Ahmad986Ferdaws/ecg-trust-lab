"""Pure metrics for the frozen SPH external-transport stress test.

This module deliberately contains no fitting or optimization path.  It accepts
raw model logits plus decision parameters learned on PTB-XL fold 9, applies
those parameters unchanged, and returns identifier-free aggregate results.
Patient identifiers are used only inside patient-cluster bootstrap calls.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ecg_trust.audit import (
    bootstrap_multilabel_metrics,
    paired_model_difference_intervals,
)
from ecg_trust.evaluation import (
    compute_multilabel_metrics,
    stable_sigmoid,
    validate_logits,
    validate_multilabel_arrays,
)
from ecg_trust.post_analysis import mean_normalized_binary_entropy
from ecg_trust.protocol import LABEL_ORDER

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


class SPHTransportMetricsError(ValueError):
    """Raised when frozen SPH transport-metric inputs are invalid."""


def evaluate_sph_transport(
    y_true: ArrayLike,
    raw_logits: ArrayLike,
    patient_ids: ArrayLike,
    *,
    temperature: float,
    thresholds: Sequence[float],
    entropy_gates: Sequence[Mapping[str, object]],
    n_resamples: int = 1_000,
    seed: int = 20_260_816,
    confidence_level: float = 0.95,
    minimum_valid_resamples: int | None = None,
    label_order: Sequence[str] = LABEL_ORDER,
    ece_bins: int = 15,
) -> dict[str, object]:
    """Evaluate one frozen member without fitting anything on SPH.

    ``patient_ids`` are passed only to the patient-cluster bootstrap and are
    never copied into the returned mapping.  Discrimination metrics that are
    undefined for a single-class label remain ``None`` with the explicit
    ``degenerate_reason`` supplied by :func:`compute_multilabel_metrics`.
    """

    labels = tuple(label_order)
    targets, logits = _validated_targets_and_logits(y_true, raw_logits, labels)
    frozen_temperature = _validate_temperature(temperature)
    frozen_thresholds = _validate_thresholds(thresholds, len(labels))
    frozen_gates = _validate_entropy_gates(entropy_gates)

    raw_probabilities = stable_sigmoid(logits)
    calibrated_probabilities = stable_sigmoid(logits / frozen_temperature)
    raw_metrics = compute_multilabel_metrics(
        targets,
        raw_probabilities,
        label_order=labels,
        ece_bins=ece_bins,
    )
    calibrated_metrics = compute_multilabel_metrics(
        targets,
        calibrated_probabilities,
        label_order=labels,
        ece_bins=ece_bins,
    )
    raw_bootstrap = bootstrap_multilabel_metrics(
        targets,
        raw_probabilities,
        patient_ids,
        n_resamples=n_resamples,
        seed=seed,
        confidence_level=confidence_level,
        minimum_valid_resamples=minimum_valid_resamples,
        label_order=labels,
        ece_bins=ece_bins,
    )
    calibrated_bootstrap = bootstrap_multilabel_metrics(
        targets,
        calibrated_probabilities,
        patient_ids,
        n_resamples=n_resamples,
        seed=seed,
        confidence_level=confidence_level,
        minimum_valid_resamples=minimum_valid_resamples,
        label_order=labels,
        ece_bins=ece_bins,
    )

    return {
        "analysis_kind": "exploratory_external_transport_stress_test",
        "clinical_validation": False,
        "tuning_or_recalibration_on_sph": False,
        "n_samples": int(targets.shape[0]),
        "label_order": list(labels),
        "ece_bins": calibrated_metrics.ece_bins,
        "frozen_policy": {
            "temperature": frozen_temperature,
            "thresholds": frozen_thresholds.tolist(),
            "entropy_gates": [dict(gate) for gate in frozen_gates],
        },
        "probability_views": {
            "raw_sigmoid": {
                "metrics": raw_metrics.to_dict(),
                "patient_cluster_bootstrap": raw_bootstrap.to_dict(),
            },
            "frozen_temperature_calibrated": {
                "metrics": calibrated_metrics.to_dict(),
                "patient_cluster_bootstrap": calibrated_bootstrap.to_dict(),
            },
        },
        "frozen_threshold_decisions": _threshold_decisions(
            targets,
            calibrated_probabilities,
            frozen_thresholds,
        ),
        "frozen_entropy_gates": _frozen_gate_results(
            targets,
            calibrated_probabilities,
            frozen_thresholds,
            frozen_gates,
        ),
    }


def paired_sph_transport_differences(
    y_true: ArrayLike,
    resnet_logits: ArrayLike,
    transformer_logits: ArrayLike,
    patient_ids: ArrayLike,
    *,
    resnet_temperature: float,
    transformer_temperature: float,
    n_resamples: int = 1_000,
    seed: int = 20_260_816,
    confidence_level: float = 0.95,
    minimum_valid_resamples: int | None = None,
    label_order: Sequence[str] = LABEL_ORDER,
    ece_bins: int = 15,
) -> dict[str, object]:
    """Return paired Transformer-minus-ResNet intervals on shared patient draws."""

    labels = tuple(label_order)
    targets, resnet_values = _validated_targets_and_logits(
        y_true,
        resnet_logits,
        labels,
    )
    transformer_values = validate_logits(
        transformer_logits,
        label_order=labels,
        n_samples=targets.shape[0],
    )
    frozen_resnet_temperature = _validate_temperature(resnet_temperature)
    frozen_transformer_temperature = _validate_temperature(transformer_temperature)

    resnet_raw = stable_sigmoid(resnet_values)
    transformer_raw = stable_sigmoid(transformer_values)
    resnet_calibrated = stable_sigmoid(resnet_values / frozen_resnet_temperature)
    transformer_calibrated = stable_sigmoid(transformer_values / frozen_transformer_temperature)

    raw = paired_model_difference_intervals(
        targets,
        transformer_raw,
        resnet_raw,
        patient_ids,
        model_a="ecg_transformer",
        model_b="resnet1d",
        n_resamples=n_resamples,
        seed=seed,
        confidence_level=confidence_level,
        minimum_valid_resamples=minimum_valid_resamples,
        label_order=labels,
        ece_bins=ece_bins,
    )
    calibrated = paired_model_difference_intervals(
        targets,
        transformer_calibrated,
        resnet_calibrated,
        patient_ids,
        model_a="ecg_transformer",
        model_b="resnet1d",
        n_resamples=n_resamples,
        seed=seed,
        confidence_level=confidence_level,
        minimum_valid_resamples=minimum_valid_resamples,
        label_order=labels,
        ece_bins=ece_bins,
    )
    return {
        "analysis_kind": "exploratory_external_transport_stress_test",
        "clinical_validation": False,
        "tuning_or_recalibration_on_sph": False,
        "difference_direction": "ecg_transformer_minus_resnet1d",
        "n_samples": int(targets.shape[0]),
        "label_order": list(labels),
        "ece_bins": ece_bins,
        "frozen_temperatures": {
            "resnet1d": frozen_resnet_temperature,
            "ecg_transformer": frozen_transformer_temperature,
        },
        "probability_views": {
            "raw_sigmoid": raw.to_dict(),
            "frozen_temperature_calibrated": calibrated.to_dict(),
        },
    }


def _validated_targets_and_logits(
    y_true: ArrayLike,
    raw_logits: ArrayLike,
    labels: tuple[str, ...],
) -> tuple[IntArray, FloatArray]:
    logits = validate_logits(raw_logits, label_order=labels)
    # Validation of targets is intentionally routed through the canonical
    # probability contract so shape, binary values, and label order stay in
    # lockstep with every other evaluation path.
    targets, _ = validate_multilabel_arrays(
        y_true,
        stable_sigmoid(logits),
        label_order=labels,
    )
    return targets, logits


def _validate_temperature(value: float) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise SPHTransportMetricsError("temperature must be a finite positive number")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise SPHTransportMetricsError("temperature must be a finite positive number") from error
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise SPHTransportMetricsError("temperature must be a finite positive number")
    return parsed


def _validate_thresholds(values: Sequence[float], n_labels: int) -> FloatArray:
    raw_values = tuple(values)
    if any(isinstance(value, (bool, np.bool_)) for value in raw_values):
        raise SPHTransportMetricsError("thresholds must be a finite numeric sequence")
    try:
        thresholds = np.asarray(raw_values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise SPHTransportMetricsError("thresholds must be a finite numeric sequence") from error
    if thresholds.shape != (n_labels,) or not np.all(np.isfinite(thresholds)):
        raise SPHTransportMetricsError(f"thresholds must contain exactly {n_labels} finite values")
    if np.any((thresholds < 0.0) | (thresholds > 1.0)):
        raise SPHTransportMetricsError("thresholds must lie in [0, 1]")
    return thresholds


def _validate_entropy_gates(
    values: Sequence[Mapping[str, object]],
) -> tuple[dict[str, float], ...]:
    gates = tuple(values)
    if not gates:
        raise SPHTransportMetricsError("entropy_gates must contain at least one frozen gate")
    parsed: list[dict[str, float]] = []
    previous_target = math.inf
    for index, gate in enumerate(gates):
        if not isinstance(gate, Mapping):
            raise SPHTransportMetricsError(f"entropy_gates[{index}] must be a mapping")
        target = _finite_gate_value(gate.get("target_coverage"), "target_coverage", index)
        cutoff = _finite_gate_value(gate.get("maximum_entropy"), "maximum_entropy", index)
        if not 0.0 < target <= 1.0:
            raise SPHTransportMetricsError("gate target_coverage must lie in (0, 1]")
        if not 0.0 <= cutoff <= 1.0:
            raise SPHTransportMetricsError("gate maximum_entropy must lie in [0, 1]")
        if target >= previous_target:
            raise SPHTransportMetricsError(
                "entropy_gates must be ordered by strictly decreasing target_coverage"
            )
        previous_target = target
        parsed.append({"target_coverage": target, "maximum_entropy": cutoff})
    return tuple(parsed)


def _finite_gate_value(value: object, name: str, index: int) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise SPHTransportMetricsError(f"entropy_gates[{index}].{name} must be finite")
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise SPHTransportMetricsError(f"entropy_gates[{index}].{name} must be finite") from error
    if not math.isfinite(parsed):
        raise SPHTransportMetricsError(f"entropy_gates[{index}].{name} must be finite")
    return parsed


def _threshold_decisions(
    targets: IntArray,
    probabilities: FloatArray,
    thresholds: FloatArray,
) -> dict[str, object]:
    predictions = probabilities >= thresholds[None, :]
    errors = np.not_equal(predictions, targets.astype(np.bool_, copy=False))
    return {
        "hamming_risk": float(errors.mean()),
        "exact_match_accuracy": float((~errors.any(axis=1)).mean()),
    }


def _frozen_gate_results(
    targets: IntArray,
    probabilities: FloatArray,
    thresholds: FloatArray,
    gates: tuple[dict[str, float], ...],
) -> list[dict[str, object]]:
    entropy = mean_normalized_binary_entropy(probabilities)
    predictions = probabilities >= thresholds[None, :]
    truth = targets.astype(np.bool_, copy=False)
    results: list[dict[str, object]] = []
    for gate in gates:
        accepted_mask = entropy <= gate["maximum_entropy"]
        accepted_count = int(accepted_mask.sum())
        if accepted_count:
            errors = np.not_equal(predictions[accepted_mask], truth[accepted_mask])
            hamming_risk: float | None = float(errors.mean())
            exact_match_accuracy: float | None = float((~errors.any(axis=1)).mean())
            status = "ok"
        else:
            hamming_risk = None
            exact_match_accuracy = None
            status = "no_accepted_samples"
        results.append(
            {
                "target_coverage": gate["target_coverage"],
                "maximum_entropy": gate["maximum_entropy"],
                "observed_coverage": float(accepted_count / targets.shape[0]),
                "selected_count": accepted_count,
                "abstained_count": int(targets.shape[0] - accepted_count),
                "hamming_risk": hamming_risk,
                "exact_match_accuracy": exact_match_accuracy,
                "status": status,
            }
        )
    return results


__all__ = [
    "SPHTransportMetricsError",
    "evaluate_sph_transport",
    "paired_sph_transport_differences",
]
