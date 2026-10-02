from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest

from ecg_trust.conformal import (
    BinaryPredictionSets,
    ConformalMetrics,
    ConformalValidationError,
    evaluate_prediction_sets,
)


def _metrics() -> ConformalMetrics:
    sets = BinaryPredictionSets.from_masks(
        label_names=("a", "b"),
        include_not_supported=[[False, True], [True, True], [False, True]],
        include_supported=[[True, True], [False, True], [False, True]],
    )
    return evaluate_prediction_sets(sets, [[1, 0], [0, 1], [1, 1]])


@pytest.mark.parametrize(
    "updates",
    [
        {"marginal_coverage": 0.1},
        {"mean_set_size": 0.2},
        {"labelwise_mean_set_size": (0.0, 0.0)},
        {"joint_sample_coverage": 0.9},
        {"joint_sample_coverage": 0.1},
        {"singleton_fraction": 0.0, "empty_fraction": 0.0, "both_fraction": 1.0},
        {"singleton_fraction": 0.0, "empty_fraction": 1 / 3, "both_fraction": 2 / 3},
        {"n_samples": 0},
        {"n_labels": 3},
        {"labelwise_coverage": (float("nan"), 1.0)},
        {"mean_set_size": float("inf")},
        {"joint_sample_coverage": True},
        {"singleton_fraction": 0.9},
    ],
)
@pytest.mark.parametrize("entrypoint", ["constructor", "reader"])
def test_contradictory_metrics_are_rejected(updates: dict[str, object], entrypoint: str) -> None:
    metrics = _metrics()
    with pytest.raises(ConformalValidationError):
        if entrypoint == "reader":
            ConformalMetrics.from_dict({**metrics.to_dict(), **updates})
        else:
            replace(metrics, **updates)  # type: ignore[arg-type]  # Probe invalid runtime values.


def test_valid_v1_metric_bytes_are_unchanged() -> None:
    metrics = _metrics()
    payload = json.dumps(metrics.to_dict(), sort_keys=True, separators=(",", ":"))
    assert hashlib.sha256(payload.encode()).hexdigest() == (
        "a43e1a6de5d56565060c043b3426ebb1a8a9c941f4a9abc0475fdf8baae9e81b"
    )
    assert ConformalMetrics.from_dict(json.loads(payload)) == metrics


def test_small_roundoff_is_accepted_without_rewriting_values() -> None:
    metrics = _metrics()
    for field in ("marginal_coverage", "joint_sample_coverage", "mean_set_size"):
        value = getattr(metrics, field) + 5e-13
        changed = replace(metrics, **{field: value})
        assert getattr(changed, field) == value
        assert ConformalMetrics.from_dict(changed.to_dict()) == changed


@pytest.mark.parametrize("n_samples", [1, 3, 7, 1003])
@pytest.mark.parametrize("n_labels", [1, 2, 5, 17])
def test_generated_metrics_satisfy_relationships(n_samples: int, n_labels: int) -> None:
    rng = np.random.default_rng(n_samples * n_labels)
    for probability in (0.0, 0.3, 0.5, 0.9, 1.0):
        negative = rng.random((n_samples, n_labels)) < probability
        positive = rng.random((n_samples, n_labels)) < probability
        sets = BinaryPredictionSets.from_masks(
            label_names=tuple(f"label_{index}" for index in range(n_labels)),
            include_not_supported=negative,
            include_supported=positive,
        )
        targets = rng.integers(0, 2, (n_samples, n_labels))
        metrics = evaluate_prediction_sets(sets, targets)
        assert ConformalMetrics.from_dict(metrics.to_dict()) == metrics


def test_constructor_snapshots_labelwise_sequences() -> None:
    metrics = _metrics()
    coverage = list(metrics.labelwise_coverage)
    sizes = list(metrics.labelwise_mean_set_size)
    # Public runtime inputs may be mutable sequences despite the static tuple annotation.
    snapshot = replace(
        metrics,
        labelwise_coverage=coverage,  # type: ignore[arg-type]
        labelwise_mean_set_size=sizes,  # type: ignore[arg-type]
    )
    coverage[0] = 0.0
    sizes[0] = 0.0
    assert snapshot == metrics
