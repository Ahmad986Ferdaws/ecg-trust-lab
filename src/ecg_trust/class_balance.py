"""Derive BCE positive-class weights from training folds only.

``training.train_one_epoch`` already accepts a ``pos_weight`` tensor, but no
configuration sets one, so rare superclasses (HYP about 12% of PTB-XL) are
trained with unweighted binary cross-entropy. This module computes the
standard ``negatives / positives`` weight per label for a development
experiment, and refuses any row outside the protocol's training folds so the
weights can't leak model-selection, calibration, or final-test labels.

It changes nothing on its own: frozen configurations and sealed models keep
unweighted training. Whether weighting helps is an empirical question for a
separately scoped development run; it tends to shift calibration, so recheck
calibration after any weighted training.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import ArrayLike

from ecg_trust.protocol import LABEL_ORDER, TRAIN_FOLDS


class ClassBalanceError(ValueError):
    """Raised when class-weight inputs are malformed or leave the training folds."""


@dataclass(frozen=True, slots=True)
class LabelBalance:
    """Training-fold prevalence and positive-class weight for one label."""

    label: str
    positives: int
    negatives: int
    pos_weight: float

    @property
    def prevalence(self) -> float:
        return self.positives / (self.positives + self.negatives)


@dataclass(frozen=True, slots=True)
class ClassBalance:
    """Per-label weights in canonical label order."""

    folds: tuple[int, ...]
    records: int
    labels: tuple[LabelBalance, ...]

    def as_tensor(self) -> torch.Tensor:
        """Weights as a float32 vector for ``BCEWithLogitsLoss(pos_weight=...)``."""

        return torch.tensor([item.pos_weight for item in self.labels], dtype=torch.float32)

    def to_dict(self) -> dict[str, object]:
        return {
            "folds": list(self.folds),
            "records": self.records,
            "labels": {
                item.label: {
                    "positives": item.positives,
                    "negatives": item.negatives,
                    "prevalence": item.prevalence,
                    "pos_weight": item.pos_weight,
                }
                for item in self.labels
            },
        }


def _fold_ids(values: ArrayLike, count: int) -> tuple[int, ...]:
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.shape[0] != count:
        raise ClassBalanceError("fold_ids must be a vector with one entry per record")
    folds: list[int] = []
    for value in raw.tolist():
        if isinstance(value, bool) or not isinstance(value, int):
            raise ClassBalanceError("fold_ids must contain integers")
        folds.append(value)
    return tuple(folds)


def _targets(values: ArrayLike, labels: int) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 2 or raw.shape[1] != labels or raw.shape[0] == 0:
        raise ClassBalanceError(f"targets must have shape [records, {labels}]")
    if raw.dtype == np.object_ or np.issubdtype(raw.dtype, np.floating) or np.iscomplexobj(raw):
        raise ClassBalanceError("targets must be an integer or boolean 0/1 matrix")
    if not np.isin(raw, (0, 1)).all():
        raise ClassBalanceError("targets must contain only 0 and 1")
    return raw.astype(np.int64)


def training_class_balance(
    targets: ArrayLike,
    fold_ids: ArrayLike,
    *,
    training_folds: Sequence[int] = TRAIN_FOLDS,
    max_pos_weight: float | None = None,
    label_order: Sequence[str] = LABEL_ORDER,
) -> ClassBalance:
    """Compute ``negatives / positives`` per label from training-fold rows only.

    Every row must belong to ``training_folds``, which must be a subset of the
    protocol's training folds. ``max_pos_weight`` optionally caps extreme weights.
    """

    labels = tuple(label_order)
    if labels != LABEL_ORDER:
        raise ClassBalanceError("label_order must match the canonical superclass order")
    allowed = tuple(training_folds)
    if not allowed or any(
        isinstance(fold, bool) or not isinstance(fold, int) or fold not in TRAIN_FOLDS
        for fold in allowed
    ):
        raise ClassBalanceError(f"training_folds must be a non-empty subset of {TRAIN_FOLDS}")
    matrix = _targets(targets, len(labels))
    folds = _fold_ids(fold_ids, matrix.shape[0])
    outside = sorted({fold for fold in folds if fold not in allowed})
    if outside:
        raise ClassBalanceError(f"rows from folds {outside} are outside the training folds")
    cap: float | None = None
    if max_pos_weight is not None:
        if (
            isinstance(max_pos_weight, bool)
            or not isinstance(max_pos_weight, (int, float))
            or not math.isfinite(max_pos_weight)
            or max_pos_weight < 1.0
        ):
            raise ClassBalanceError("max_pos_weight must be finite and at least 1")
        cap = float(max_pos_weight)

    balances: list[LabelBalance] = []
    for index, label in enumerate(labels):
        positives = int(matrix[:, index].sum())
        negatives = int(matrix.shape[0] - positives)
        if positives == 0 or negatives == 0:
            raise ClassBalanceError(f"label {label!r} needs both positive and negative rows")
        weight = negatives / positives
        if cap is not None:
            weight = min(weight, cap)
        balances.append(
            LabelBalance(label=label, positives=positives, negatives=negatives, pos_weight=weight)
        )
    return ClassBalance(
        folds=tuple(sorted(set(folds))),
        records=int(matrix.shape[0]),
        labels=tuple(balances),
    )


__all__ = ["ClassBalance", "ClassBalanceError", "LabelBalance", "training_class_balance"]
