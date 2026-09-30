from __future__ import annotations

import numpy as np
import pytest

from ecg_trust.conformal import (
    ConformalValidationError,
    GroupConditionalConformal,
    LabelwiseBinaryConformal,
)

LABELS = ("NORM", "MI", "STTC", "CD", "HYP")


def _cohort(count: int, *, noise: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    targets = (rng.uniform(size=(count, 5)) < 0.3).astype(np.int64)
    logits = np.where(targets == 1, 1.5, -1.5) + rng.normal(0.0, noise, size=(count, 5))
    return 1.0 / (1.0 + np.exp(-logits)), targets


def _coverage(sets: object, targets: np.ndarray) -> float:
    negative = np.asarray(sets.include_not_supported)  # type: ignore[attr-defined]
    positive = np.asarray(sets.include_supported)  # type: ignore[attr-defined]
    covered = np.where(targets == 1, positive, negative)
    return float(covered.mean())


def _mixed(seed: int, easy: int, hard: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    easy_p, easy_y = _cohort(easy, noise=0.5, seed=seed)
    hard_p, hard_y = _cohort(hard, noise=3.0, seed=seed + 100)
    groups = ["under_80"] * easy + ["80_plus"] * hard
    return np.vstack((easy_p, hard_p)), np.vstack((easy_y, hard_y)), groups


def test_hard_minority_group_is_covered_only_by_group_conditional_sets() -> None:
    calibration_p, calibration_y, calibration_groups = _mixed(1, 4_000, 400)
    test_p, test_y, test_groups = _mixed(2, 4_000, 4_000)
    hard_rows = np.array([group == "80_plus" for group in test_groups])

    pooled = LabelwiseBinaryConformal.fit(
        calibration_p, calibration_y, label_names=LABELS, alpha=0.1
    )
    mondrian = GroupConditionalConformal.fit(
        calibration_p, calibration_y, calibration_groups, label_names=LABELS, alpha=0.1
    )

    pooled_hard = _coverage(pooled.predict(test_p[hard_rows]), test_y[hard_rows])
    mondrian_sets = mondrian.predict(test_p, test_groups)
    negative = np.asarray(mondrian_sets.include_not_supported)[hard_rows]
    positive = np.asarray(mondrian_sets.include_supported)[hard_rows]
    mondrian_hard = float(np.where(test_y[hard_rows] == 1, positive, negative).mean())

    assert pooled_hard < 0.85
    assert mondrian_hard >= 0.88
    assert mondrian.trivial_groups == ()
    assert set(mondrian.to_dict()["groups"]) == {"80_plus", "under_80"}  # type: ignore[arg-type]


def test_tiny_group_gets_trivial_sets_and_is_reported() -> None:
    probabilities, targets = _cohort(20, noise=1.0, seed=3)
    groups = ["large"] * 18 + ["tiny"] * 2

    model = GroupConditionalConformal.fit(
        probabilities, targets, groups, label_names=LABELS, alpha=0.1
    )
    sets = model.predict(probabilities[-2:], ["tiny", "tiny"])

    assert model.trivial_groups == ("tiny",)
    assert np.asarray(sets.include_not_supported).all()
    assert np.asarray(sets.include_supported).all()


def test_unknown_groups_and_malformed_inputs_are_refused() -> None:
    probabilities, targets = _cohort(50, noise=1.0, seed=4)
    model = GroupConditionalConformal.fit(
        probabilities, targets, ["a"] * 50, label_names=LABELS, alpha=0.1
    )

    with pytest.raises(ConformalValidationError, match="not calibrated"):
        model.predict(probabilities[:2], ["a", "b"])
    with pytest.raises(ConformalValidationError, match="one group name per row"):
        model.predict(probabilities[:2], ["a"])
    with pytest.raises(ConformalValidationError, match="sequence"):
        model.predict(probabilities[:1], "a")
    with pytest.raises(ConformalValidationError, match="non-empty"):
        GroupConditionalConformal.fit(probabilities, targets, [""] * 50, label_names=LABELS)
    with pytest.raises(ConformalValidationError, match="one row per group"):
        GroupConditionalConformal.fit(probabilities, targets[:10], ["a"] * 50, label_names=LABELS)


def test_models_are_validated_and_read_only() -> None:
    probabilities, targets = _cohort(50, noise=1.0, seed=5)
    first = LabelwiseBinaryConformal.fit(probabilities, targets, label_names=LABELS, alpha=0.1)
    other_alpha = LabelwiseBinaryConformal.fit(
        probabilities, targets, label_names=LABELS, alpha=0.2
    )
    models = {"a": first}

    frozen = GroupConditionalConformal(models=models)
    models["b"] = other_alpha

    assert tuple(frozen.models) == ("a",)
    with pytest.raises(TypeError):
        frozen.models["b"] = first  # type: ignore[index]
    with pytest.raises(ConformalValidationError, match="share label names and alpha"):
        GroupConditionalConformal(models={"a": first, "b": other_alpha})
    with pytest.raises(ConformalValidationError, match="at least one"):
        GroupConditionalConformal(models={})
    with pytest.raises(ConformalValidationError, match="LabelwiseBinaryConformal"):
        GroupConditionalConformal(models={"a": object()})  # type: ignore[dict-item]
