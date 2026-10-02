from __future__ import annotations

import warnings

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from ecg_trust.training import TrainingValidationError, evaluate, select_device, train_one_epoch


@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("field", ["targets", "pos_weight"])
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("imaginary", [0.0, 4.0])
def test_complex_supervision_is_rejected_before_conversion_or_model_execution(
    monkeypatch: pytest.MonkeyPatch,
    training: bool,
    field: str,
    dtype: torch.dtype,
    imaginary: float,
) -> None:
    model = nn.Linear(2, 2)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    monkeypatch.setattr(model, "forward", lambda _: pytest.fail("model executed on invalid data"))
    targets = torch.tensor([[0.0, 1.0]])
    pos_weight = torch.ones(2)
    if field == "targets":
        targets = targets.to(dtype) + imaginary * 1j
    else:
        pos_weight = pos_weight.to(dtype) + imaginary * 1j
    batch = [(torch.zeros(1, 2), targets)]
    runtime = select_device("cpu")

    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always")
        with pytest.raises(TrainingValidationError, match=rf"{field}.*real-valued"):
            if training:
                train_one_epoch(
                    model, batch, torch.optim.SGD(model.parameters(), lr=0.1),
                    runtime, pos_weight=pos_weight,
                )
            else:
                evaluate(model, batch, runtime, pos_weight=pos_weight)

    assert not emitted
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


@pytest.mark.parametrize(
    "dtype",
    [
        torch.bool, torch.uint8, torch.int64, torch.float16,
        torch.bfloat16, torch.float32, torch.float64,
    ],
)
def test_real_supervision_preserves_supported_dtypes(dtype: torch.dtype) -> None:
    logits = torch.tensor([[-1.0, 2.0], [1.0, -2.0]])
    targets = torch.tensor([[0, 1], [1, 0]], dtype=dtype)
    weights = torch.ones(2, dtype=dtype)
    expected = F.binary_cross_entropy_with_logits(
        logits, targets.float(), pos_weight=weights.float()
    )

    result = evaluate(
        nn.Identity(), [(logits, targets)], select_device("cpu"), pos_weight=weights
    )

    assert result.loss == pytest.approx(expected.item())
    torch.testing.assert_close(result.targets, targets.float())
