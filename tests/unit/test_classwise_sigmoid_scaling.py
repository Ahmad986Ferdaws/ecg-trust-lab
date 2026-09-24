from __future__ import annotations

import copy
import json
import math
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

import ecg_trust.decisioning as decisioning
import ecg_trust.source_calibration.pipeline as source_calibration_pipeline
from ecg_trust.evaluation import (
    CalibrationLeakageError,
    ClasswiseSigmoidScalingResult,
    EvaluationValidationError,
    PerLabelSigmoidScaling,
    compute_multilabel_metrics,
    fit_classwise_sigmoid_scaling,
    fit_temperature_scaling,
    stable_sigmoid,
)
from ecg_trust.protocol import LABEL_ORDER

NORM, MI, STTC, CD, HYP = range(5)
_SMALL = 256


def _miscalibrated_fold(seed: int, n_samples: int = 4000) -> tuple[np.ndarray, np.ndarray]:
    """Draw targets from calibrated logits, then distort labels differently.

    NORM is overconfident (x3), MI is underconfident (/3), STTC is shifted by
    +1.5, and CD/HYP remain calibrated.
    """

    rng = np.random.default_rng(seed)
    true_logits = rng.normal(0.0, 1.5, size=(n_samples, 5))
    targets = (rng.random((n_samples, 5)) < stable_sigmoid(true_logits)).astype(np.int64)
    logits = true_logits.copy()
    logits[:, NORM] = 3.0 * true_logits[:, NORM]
    logits[:, MI] = true_logits[:, MI] / 3.0
    logits[:, STTC] = true_logits[:, STTC] + 1.5
    return logits, targets


def _per_label_nll(logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
    return np.mean(np.logaddexp(0.0, logits) - targets * logits, axis=0)


def _fold_9(targets: np.ndarray) -> np.ndarray:
    return np.full(targets.shape[0], 9)


def test_one_temperature_cannot_fix_opposite_miscalibration_but_classwise_scaling_does() -> None:
    logits, targets = _miscalibrated_fold(2026)
    raw_nll = _per_label_nll(logits, targets)

    global_fit = fit_temperature_scaling(
        logits=logits, y_true=targets, calibration_fold_ids=_fold_9(targets)
    )
    global_nll = _per_label_nll(global_fit.transform_logits(logits), targets)
    assert not (global_nll[NORM] < raw_nll[NORM] and global_nll[MI] < raw_nll[MI])
    for temperature in np.geomspace(0.05, 20.0, 240):
        swept_nll = _per_label_nll(logits / temperature, targets)
        assert not (swept_nll[NORM] < raw_nll[NORM] and swept_nll[MI] < raw_nll[MI])

    result = fit_classwise_sigmoid_scaling(
        logits=logits, y_true=targets, calibration_fold_ids=_fold_9(targets)
    )
    classwise_nll = _per_label_nll(result.transform_logits(logits), targets)
    assert result.status == "optimized"
    assert result.converged is True
    assert result.fitted_labels == LABEL_ORDER
    assert result.slopes[NORM] < 1.0 < result.slopes[MI]
    assert result.intercepts[STTC] < -1.0
    for index, item in enumerate(result.per_label):
        assert item.nll_after == pytest.approx(classwise_nll[index], rel=1e-12)
        assert item.nll_before == pytest.approx(raw_nll[index], rel=1e-12)
        assert item.nll_after <= item.nll_before
    distorted = [NORM, MI, STTC]
    assert np.all(classwise_nll[distorted] < raw_nll[distorted] - 0.05)
    assert np.all(classwise_nll[distorted] < global_nll[distorted])

    held_out_logits, held_out_targets = _miscalibrated_fold(2027)
    held_out_raw = _per_label_nll(held_out_logits, held_out_targets)
    held_out_classwise = _per_label_nll(
        result.transform_logits(held_out_logits), held_out_targets
    )
    assert np.all(held_out_classwise[distorted] < held_out_raw[distorted])
    raw_brier = compute_multilabel_metrics(
        held_out_targets, stable_sigmoid(held_out_logits)
    ).macro.brier_score
    classwise_brier = compute_multilabel_metrics(
        held_out_targets, result.predict_proba(held_out_logits)
    ).macro.brier_score
    assert classwise_brier < raw_brier


def test_positive_slopes_preserve_each_label_ranking_even_at_the_bounds() -> None:
    logits, targets = _miscalibrated_fold(2028)
    logits[:, CD] = -logits[:, CD]
    logits[:, HYP] = logits[:, HYP] / 50.0

    result = fit_classwise_sigmoid_scaling(
        logits=logits,
        y_true=targets,
        calibration_fold_ids=_fold_9(targets),
        regularization=1e-6,
    )

    assert result.per_label[CD].active_slope_bound == "lower"
    assert result.slopes[CD] == 0.05
    assert result.per_label[HYP].active_slope_bound == "upper"
    assert result.slopes[HYP] == 20.0
    assert all(slope > 0.0 for slope in result.slopes)
    transformed = result.transform_logits(logits)
    for index in range(len(LABEL_ORDER)):
        np.testing.assert_array_equal(
            np.argsort(transformed[:, index], kind="stable"),
            np.argsort(logits[:, index], kind="stable"),
        )
    raw = compute_multilabel_metrics(targets, stable_sigmoid(logits))
    calibrated = compute_multilabel_metrics(targets, result.predict_proba(logits))
    assert raw.per_label[CD].roc_auc is not None and raw.per_label[CD].roc_auc < 0.5
    for before, after in zip(raw.per_label, calibrated.per_label, strict=True):
        assert after.roc_auc == before.roc_auc
        assert after.average_precision == before.average_precision


def test_strong_regularization_shrinks_every_label_toward_the_identity_map() -> None:
    logits, targets = _miscalibrated_fold(2026)
    fits = [
        fit_classwise_sigmoid_scaling(
            logits=logits,
            y_true=targets,
            calibration_fold_ids=_fold_9(targets),
            regularization=regularization,
        )
        for regularization in (1e-4, 1e-2, 1.0, 1e2, 1e8)
    ]

    distances = [
        np.square(np.asarray(fit.slopes) - 1.0) + np.square(np.asarray(fit.intercepts))
        for fit in fits
    ]
    for weaker, stronger in zip(distances, distances[1:], strict=False):
        assert np.all(stronger <= weaker + 1e-9)
    strongest = fits[-1]
    assert strongest.regularization == 1e8
    assert np.max(np.abs(np.asarray(strongest.slopes) - 1.0)) < 1e-6
    assert np.max(np.abs(np.asarray(strongest.intercepts))) < 1e-6
    np.testing.assert_allclose(
        strongest.predict_proba(logits), stable_sigmoid(logits), rtol=0.0, atol=1e-5
    )

    frozen = fit_classwise_sigmoid_scaling(
        logits=logits,
        y_true=targets,
        calibration_fold_ids=_fold_9(targets),
        regularization=1e12,
    )
    assert frozen.status == "identity_optimal"
    assert frozen.converged is True
    assert frozen.slopes == (1.0,) * len(LABEL_ORDER)
    assert frozen.intercepts == (0.0,) * len(LABEL_ORDER)
    for item in frozen.per_label:
        assert item.status == "identity_optimal"
        assert item.nll_after == item.nll_before


def test_slope_only_mode_fits_per_label_temperatures_with_zero_intercepts() -> None:
    logits, targets = _miscalibrated_fold(2026)

    result = fit_classwise_sigmoid_scaling(
        logits=logits,
        y_true=targets,
        calibration_fold_ids=_fold_9(targets),
        fit_intercept=False,
    )

    assert result.fit_intercept is False
    assert result.intercepts == (0.0,) * len(LABEL_ORDER)
    assert all(item.intercept == 0.0 for item in result.per_label)
    assert result.slopes[NORM] < 1.0 < result.slopes[MI]
    assert all(item.nll_after is not None for item in result.per_label)

    shared_logits = np.tile(logits[:, [NORM]], (1, len(LABEL_ORDER)))
    shared_targets = np.tile(targets[:, [NORM]], (1, len(LABEL_ORDER)))
    global_fit = fit_temperature_scaling(
        logits=shared_logits,
        y_true=shared_targets,
        calibration_fold_ids=_fold_9(shared_targets),
    )
    per_label_temperature = fit_classwise_sigmoid_scaling(
        logits=shared_logits,
        y_true=shared_targets,
        calibration_fold_ids=_fold_9(shared_targets),
        regularization=1e-12,
        fit_intercept=False,
    )
    assert per_label_temperature.slopes == pytest.approx(
        (1.0 / global_fit.temperature,) * len(LABEL_ORDER), rel=1e-6
    )


def test_degenerate_calibration_labels_keep_the_identity_map_and_are_reported() -> None:
    logits, targets = _miscalibrated_fold(2026, n_samples=_SMALL)
    targets[:, MI] = 0
    targets[:, HYP] = 1

    result = fit_classwise_sigmoid_scaling(
        logits=logits, y_true=targets, calibration_fold_ids=_fold_9(targets)
    )

    assert result.fitted_labels == ("NORM", "STTC", "CD")
    assert result.excluded_degenerate_labels == ("MI", "HYP")
    assert result.per_label[MI].status == "no_positive_examples"
    assert result.per_label[HYP].status == "no_negative_examples"
    for index in (MI, HYP):
        item = result.per_label[index]
        assert (item.slope, item.intercept) == (1.0, 0.0)
        assert item.nll_before is None
        assert item.nll_after is None
        assert item.converged is False
        assert item.optimization_steps == 0
    np.testing.assert_array_equal(
        result.transform_logits(logits)[:, [MI, HYP]], logits[:, [MI, HYP]]
    )

    all_negative = fit_classwise_sigmoid_scaling(
        logits=logits,
        y_true=np.zeros_like(targets),
        calibration_fold_ids=_fold_9(targets),
    )
    assert all_negative.status == "no_non_degenerate_labels"
    assert all_negative.converged is False
    assert all_negative.fitted_labels == ()
    assert all_negative.excluded_degenerate_labels == LABEL_ORDER
    assert all_negative.nll_before is None
    assert all_negative.nll_after is None
    np.testing.assert_array_equal(all_negative.transform_logits(logits), logits)
    np.testing.assert_allclose(
        all_negative.predict_proba(logits), stable_sigmoid(logits), rtol=0.0, atol=1e-15
    )


def test_classwise_scaling_is_deterministic_and_does_not_mutate_inputs() -> None:
    logits, targets = _miscalibrated_fold(2026, n_samples=1000)
    folds = _fold_9(targets)
    snapshots = (logits.copy(), targets.copy(), folds.copy())

    first = fit_classwise_sigmoid_scaling(
        logits=logits, y_true=targets, calibration_fold_ids=folds
    )
    second = fit_classwise_sigmoid_scaling(
        logits=logits.copy(), y_true=targets.copy(), calibration_fold_ids=folds.copy()
    )

    assert first == second
    assert first.to_json() == second.to_json()
    for current, snapshot in zip((logits, targets, folds), snapshots, strict=True):
        np.testing.assert_array_equal(current, snapshot)


def test_exhausted_step_budget_is_reported_without_increasing_nll() -> None:
    logits, targets = _miscalibrated_fold(2026, n_samples=1000)

    result = fit_classwise_sigmoid_scaling(
        logits=logits,
        y_true=targets,
        calibration_fold_ids=_fold_9(targets),
        max_steps=1,
    )

    assert result.converged is False
    assert result.max_steps == 1
    for item in result.per_label:
        assert item.optimization_steps <= 1
        assert item.nll_after is not None and item.nll_before is not None
        assert item.nll_after <= item.nll_before
    assert result.per_label[NORM].converged is False


@pytest.mark.parametrize(
    ("overrides", "error", "match"),
    [
        ({"calibration_fold_ids": np.full(_SMALL, 10)}, CalibrationLeakageError, "fold 9 only"),
        (
            {"calibration_fold_ids": np.r_[np.full(_SMALL - 1, 9), 10]},
            CalibrationLeakageError,
            "fold 9 only",
        ),
        (
            {"label_order": ("MI", "NORM", "STTC", "CD", "HYP")},
            EvaluationValidationError,
            "label_order",
        ),
        ({"regularization": 0.0}, EvaluationValidationError, "regularization"),
        ({"regularization": -1.0}, EvaluationValidationError, "regularization"),
        ({"regularization": math.inf}, EvaluationValidationError, "regularization"),
        ({"regularization": math.nan}, EvaluationValidationError, "regularization"),
        ({"regularization": True}, EvaluationValidationError, "regularization"),
        ({"regularization": "0.1"}, EvaluationValidationError, "regularization"),
        ({"regularization": 1e-3 + 0j}, EvaluationValidationError, "regularization"),
        ({"fit_intercept": 1}, EvaluationValidationError, "fit_intercept"),
        ({"slope_bounds": (0.0, 20.0)}, EvaluationValidationError, "slope_bounds"),
        ({"slope_bounds": (1.5, 20.0)}, EvaluationValidationError, "include 1.0"),
        ({"slope_bounds": (0.05, 0.5)}, EvaluationValidationError, "include 1.0"),
        ({"slope_bounds": (2.0, 1.0)}, EvaluationValidationError, "slope_bounds"),
        ({"slope_bounds": (0.05, math.inf)}, EvaluationValidationError, "slope_bounds"),
        ({"slope_bounds": (0.05,)}, EvaluationValidationError, "two values"),
        ({"slope_bounds": "ab"}, EvaluationValidationError, "slope_bounds"),
        ({"tolerance": 0.0}, EvaluationValidationError, "tolerance"),
        ({"tolerance": math.nan}, EvaluationValidationError, "tolerance"),
        ({"max_steps": 0}, EvaluationValidationError, "max_steps"),
        ({"max_steps": True}, EvaluationValidationError, "max_steps"),
        ({"max_steps": 2.5}, EvaluationValidationError, "max_steps"),
    ],
)
def test_classwise_scaling_rejects_invalid_configuration(
    overrides: dict[str, Any],
    error: type[Exception],
    match: str,
) -> None:
    logits, targets = _miscalibrated_fold(2026, n_samples=_SMALL)
    arguments: dict[str, Any] = {
        "logits": logits,
        "y_true": targets,
        "calibration_fold_ids": _fold_9(targets),
        **overrides,
    }

    with pytest.raises(error, match=match):
        fit_classwise_sigmoid_scaling(**arguments)


def test_classwise_scaling_rejects_invalid_arrays_and_unrepresentable_logits() -> None:
    logits, targets = _miscalibrated_fold(2026, n_samples=_SMALL)
    folds = _fold_9(targets)
    non_finite = logits.copy()
    non_finite[0, 0] = np.nan

    with pytest.raises(EvaluationValidationError, match="finite"):
        fit_classwise_sigmoid_scaling(
            logits=non_finite, y_true=targets, calibration_fold_ids=folds
        )
    with pytest.raises(EvaluationValidationError, match="samples"):
        fit_classwise_sigmoid_scaling(
            logits=logits[:-1], y_true=targets, calibration_fold_ids=folds
        )
    with pytest.raises(EvaluationValidationError, match="real"):
        fit_classwise_sigmoid_scaling(
            logits=logits + 0j, y_true=targets, calibration_fold_ids=folds
        )
    with pytest.raises(EvaluationValidationError, match="binary"):
        fit_classwise_sigmoid_scaling(
            logits=logits, y_true=targets + 0.5, calibration_fold_ids=folds
        )
    with pytest.raises(EvaluationValidationError, match="too large"):
        fit_classwise_sigmoid_scaling(
            logits=np.where(targets == 1, 1e200, -1e200),
            y_true=targets,
            calibration_fold_ids=folds,
        )

    result = fit_classwise_sigmoid_scaling(
        logits=logits, y_true=targets, calibration_fold_ids=folds
    )
    with pytest.raises(EvaluationValidationError, match="finite"):
        result.transform_logits(np.full((2, 5), 1e308))
    with pytest.raises(EvaluationValidationError, match="label_order"):
        result.predict_proba(logits, label_order=("MI", "NORM", "STTC", "CD", "HYP"))


def _rebuild(payload: dict[str, Any]) -> ClasswiseSigmoidScalingResult:
    return ClasswiseSigmoidScalingResult(
        **{
            **payload,
            "per_label": [PerLabelSigmoidScaling(**item) for item in payload["per_label"]],
        }
    )


def _set_top_level(key: str, value: object) -> Callable[[dict[str, Any]], None]:
    def mutate(payload: dict[str, Any]) -> None:
        payload[key] = value

    return mutate


def _set_per_label(index: int, key: str, value: object) -> Callable[[dict[str, Any]], None]:
    def mutate(payload: dict[str, Any]) -> None:
        payload["per_label"][index][key] = value

    return mutate


@pytest.mark.parametrize(
    ("mutate", "error", "match"),
    [
        (_set_top_level("source_folds", [9, 10]), CalibrationLeakageError, "fold 9 only"),
        (_set_top_level("slopes", [2.0, 1.0, 1.0, 1.0, 1.0]), EvaluationValidationError, "match"),
        (_set_top_level("fit_intercept", False), EvaluationValidationError, "slope-only"),
        (_set_top_level("slope_bounds", [0.9, 1.1]), EvaluationValidationError, "slope_bounds"),
        (_set_top_level("regularization", 0.0), EvaluationValidationError, "regularization"),
        (_set_top_level("status", "identity_optimal"), EvaluationValidationError, "summarize"),
        (_set_top_level("converged", False), EvaluationValidationError, "summarize"),
        (_set_top_level("nll_after", 0.0), EvaluationValidationError, "summarize"),
        (
            _set_top_level("fitted_labels", list(LABEL_ORDER)),
            EvaluationValidationError,
            "summarize",
        ),
        (_set_top_level("max_steps", 1), EvaluationValidationError, "max_steps"),
        (_set_per_label(NORM, "slope", -0.5), EvaluationValidationError, "positive"),
        (_set_per_label(NORM, "intercept", math.inf), EvaluationValidationError, "finite"),
        (_set_per_label(NORM, "nll_after", 10.0), EvaluationValidationError, "increase"),
        (_set_per_label(NORM, "status", "fitted"), EvaluationValidationError, "status"),
        (_set_per_label(HYP, "slope", 2.0), EvaluationValidationError, "identity"),
        (_set_per_label(HYP, "status", "optimized"), EvaluationValidationError, "identity"),
        (_set_per_label(NORM, "active_slope_bound", "lower"), EvaluationValidationError, "bound"),
    ],
)
def test_classwise_scaling_artifact_revalidates_every_construction(
    mutate: Callable[[dict[str, Any]], None],
    error: type[Exception],
    match: str,
) -> None:
    logits, targets = _miscalibrated_fold(2026, n_samples=1000)
    targets[:, HYP] = 0
    result = fit_classwise_sigmoid_scaling(
        logits=logits, y_true=targets, calibration_fold_ids=_fold_9(targets)
    )
    payload = json.loads(result.to_json())
    json.dumps(result.to_dict(), allow_nan=False)
    assert payload["source_folds"] == [9]
    assert payload["per_label"][HYP]["status"] == "no_positive_examples"
    assert _rebuild(payload) == result

    tampered = copy.deepcopy(payload)
    mutate(tampered)
    with pytest.raises(error, match=match):
        _rebuild(tampered)


def test_classwise_scaling_is_opt_in_and_global_temperature_scaling_is_unchanged() -> None:
    for module in (decisioning, source_calibration_pipeline):
        assert module.fit_temperature_scaling is fit_temperature_scaling
        assert not hasattr(module, "fit_classwise_sigmoid_scaling")

    logits, targets = _miscalibrated_fold(2026)
    result = fit_temperature_scaling(
        logits=logits, y_true=targets, calibration_fold_ids=_fold_9(targets)
    )

    # Recorded from this call at 87857bf, before classwise scaling existed.
    assert result.temperature == pytest.approx(1.8965507304768174, rel=1e-9)
    assert result.nll_before == pytest.approx(0.6258067964519609, rel=1e-9)
    assert result.nll_after == pytest.approx(0.5865513221306994, rel=1e-9)
    assert (result.status, result.converged, result.optimization_steps) == (
        "optimized",
        True,
        55,
    )
    assert sorted(result.to_dict()) == [
        "converged",
        "excluded_degenerate_labels",
        "fitted_labels",
        "label_order",
        "n_samples",
        "nll_after",
        "nll_before",
        "optimization_steps",
        "source_folds",
        "status",
        "temperature",
        "temperature_bounds",
    ]
