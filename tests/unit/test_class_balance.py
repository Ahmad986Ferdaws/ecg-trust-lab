from __future__ import annotations

import numpy as np
import pytest
import torch

from ecg_trust.class_balance import ClassBalanceError, training_class_balance
from ecg_trust.protocol import LABEL_ORDER


def _targets() -> np.ndarray:
    targets = np.zeros((10, 5), dtype=np.int64)
    targets[:5, 0] = 1
    targets[:1, 1] = 1
    targets[:2, 2] = 1
    targets[:8, 3] = 1
    targets[:4, 4] = 1
    return targets


def test_weights_are_negatives_over_positives_in_label_order() -> None:
    balance = training_class_balance(_targets(), [1, 2, 3, 4, 5, 6, 7, 1, 2, 3])

    assert [item.label for item in balance.labels] == list(LABEL_ORDER)
    assert [item.pos_weight for item in balance.labels] == pytest.approx([1.0, 9.0, 4.0, 0.25, 1.5])
    assert balance.labels[1].prevalence == pytest.approx(0.1)
    assert balance.folds == (1, 2, 3, 4, 5, 6, 7)
    assert balance.to_dict()["records"] == 10


def test_tensor_is_accepted_by_weighted_bce() -> None:
    balance = training_class_balance(_targets().astype(bool), np.full(10, 3))
    loss = torch.nn.BCEWithLogitsLoss(pos_weight=balance.as_tensor())

    value = loss(torch.zeros(10, 5), torch.from_numpy(_targets()).float())

    assert balance.as_tensor().dtype == torch.float32
    assert torch.isfinite(value)


def test_cap_limits_extreme_weights() -> None:
    balance = training_class_balance(_targets(), np.ones(10, dtype=np.int64), max_pos_weight=3.0)

    assert max(item.pos_weight for item in balance.labels) == pytest.approx(3.0)


@pytest.mark.parametrize("fold", [8, 9, 10])
def test_rows_outside_training_folds_are_refused(fold: int) -> None:
    folds = np.ones(10, dtype=np.int64)
    folds[-1] = fold

    with pytest.raises(ClassBalanceError, match="outside the training folds"):
        training_class_balance(_targets(), folds)


def test_training_fold_subset_is_enforced() -> None:
    with pytest.raises(ClassBalanceError, match="outside the training folds"):
        training_class_balance(_targets(), np.full(10, 2), training_folds=(1,))
    with pytest.raises(ClassBalanceError, match="subset"):
        training_class_balance(_targets(), np.full(10, 1), training_folds=(1, 9))
    with pytest.raises(ClassBalanceError, match="subset"):
        training_class_balance(_targets(), np.full(10, 1), training_folds=())


@pytest.mark.parametrize(
    ("targets", "folds", "kwargs", "message"),
    [
        (np.zeros((10, 5), dtype=np.int64), np.ones(10, dtype=np.int64), {}, "both positive"),
        (_targets().astype(float), np.ones(10, dtype=np.int64), {}, "integer or boolean"),
        (_targets() * 2, np.ones(10, dtype=np.int64), {}, "only 0 and 1"),
        (_targets()[:, :4], np.ones(10, dtype=np.int64), {}, "shape"),
        (_targets(), np.ones(9, dtype=np.int64), {}, "one entry per record"),
        (_targets(), np.ones(10), {}, "integers"),
        (_targets(), np.ones(10, dtype=bool), {}, "integers"),
        (_targets(), np.ones(10, dtype=np.int64), {"max_pos_weight": 0.5}, "at least 1"),
        (_targets(), np.ones(10, dtype=np.int64), {"max_pos_weight": float("inf")}, "finite"),
        (_targets(), np.ones(10, dtype=np.int64), {"label_order": LABEL_ORDER[::-1]}, "canonical"),
    ],
)
def test_malformed_inputs_are_rejected(
    targets: np.ndarray, folds: np.ndarray, kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ClassBalanceError, match=message):
        training_class_balance(targets, folds, **kwargs)  # type: ignore[arg-type]
