from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

from ecg_trust.training import (
    CheckpointValidationError,
    EarlyStopping,
    load_checkpoint,
    save_checkpoint,
)

_IDENTITY = {
    "expected_protocol_hash": "a" * 64,
    "expected_config": {},
    "expected_manifest_hash": "b" * 64,
}


def _model(fill: float) -> nn.Sequential:
    model = nn.Sequential(nn.Linear(2, 3), nn.BatchNorm1d(3), nn.Linear(3, 2))
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(fill)
        model[1].running_mean.fill_(fill)
    model.train()
    model[1].eval()
    return model


def _optimizer(model: nn.Module, learning_rate: float) -> torch.optim.Optimizer:
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    sum(parameter.square().sum() for parameter in model.parameters()).backward()
    optimizer.step()
    return optimizer


def _assert_state_equal(actual: Any, expected: Any) -> None:
    if isinstance(actual, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            _assert_state_equal(actual[key], expected[key])
    elif isinstance(actual, (tuple, list)):
        assert len(actual) == len(expected)
        for value, reference in zip(actual, expected, strict=True):
            _assert_state_equal(value, reference)
    else:
        assert actual == expected


def _setup(
    tmp_path: Path,
) -> tuple[Path, nn.Module, torch.optim.Optimizer, torch.amp.GradScaler, EarlyStopping]:
    source = _model(3.0)
    source_optimizer = _optimizer(source, 0.2)
    source_scaler = torch.amp.GradScaler("cpu", init_scale=256)
    source_stopper = EarlyStopping(patience=7)
    source_stopper.update(0.3, 3)
    path = tmp_path / "state.ckpt"
    save_checkpoint(
        path,
        model=source,
        optimizer=source_optimizer,
        scaler=source_scaler,
        epoch=3,
        protocol_hash="a" * 64,
        config={},
        manifest_hash="b" * 64,
        early_stopping=source_stopper,
    )
    target = _model(1.0)
    optimizer = _optimizer(target, 0.01)
    scaler = torch.amp.GradScaler("cpu", init_scale=8)
    scaler.scale(torch.tensor(1.0))  # Materialize the scale tensor as well as initial state.
    stopper = EarlyStopping(patience=2)
    stopper.update(0.7, 1)
    return path, target, optimizer, scaler, stopper


@pytest.mark.parametrize("failure", ["model", "optimizer", "scaler"])
def test_failed_checkpoint_load_restores_all_live_training_state(
    tmp_path: Path, failure: str
) -> None:
    path, model, optimizer, scaler, stopper = _setup(tmp_path)
    payload = torch.load(path, weights_only=True)
    if failure == "model":
        payload["model_state_dict"]["2.weight"] = torch.zeros(99, 99)
        # A load hook can change modes before a later parameter mismatch raises.
        model.register_load_state_dict_pre_hook(lambda module, *args: module.eval())
    elif failure == "optimizer":
        payload["optimizer_state_dict"]["param_groups"][0]["params"] = []
    else:
        # GradScaler writes its scale before looking up this key.
        del payload["scaler_state_dict"]["growth_factor"]
    torch.save(payload, path)
    components = (model, optimizer, scaler, stopper)
    before = [deepcopy(component.state_dict()) for component in components]
    modes = [module.training for module in model.modules()]
    parameter_ids = [id(parameter) for parameter in model.parameters()]

    with pytest.raises((RuntimeError, ValueError, KeyError)):
        load_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            early_stopping=stopper,
            **_IDENTITY,
        )

    for component, state in zip(components, before, strict=True):
        _assert_state_equal(component.state_dict(), state)
    assert [module.training for module in model.modules()] == modes
    assert [id(parameter) for parameter in model.parameters()] == parameter_ids


def test_successful_restore_keeps_parameter_bindings_and_mixed_modes(tmp_path: Path) -> None:
    path, model, optimizer, scaler, stopper = _setup(tmp_path)
    payload = torch.load(path, weights_only=True)
    modes = [module.training for module in model.modules()]
    parameter_ids = [id(parameter) for parameter in model.parameters()]

    load_checkpoint(
        path, model=model, optimizer=optimizer, scaler=scaler, early_stopping=stopper, **_IDENTITY
    )

    for component, key in zip(
        (model, optimizer, scaler, stopper),
        (
            "model_state_dict",
            "optimizer_state_dict",
            "scaler_state_dict",
            "early_stopping_state_dict",
        ),
        strict=True,
    ):
        _assert_state_equal(component.state_dict(), payload[key])
    assert [module.training for module in model.modules()] == modes
    assert [id(parameter) for parameter in model.parameters()] == parameter_ids
    assert [id(parameter) for parameter in optimizer.param_groups[0]["params"]] == parameter_ids


def test_broken_custom_rollback_reports_failure_and_restores_other_components(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, model, optimizer, scaler, stopper = _setup(tmp_path)
    before_model, before_optimizer = deepcopy(model.state_dict()), deepcopy(optimizer.state_dict())

    def broken_scaler_restore(state: dict[str, Any]) -> None:
        raise RuntimeError("custom scaler refuses checkpoint and rollback")

    monkeypatch.setattr(scaler, "load_state_dict", broken_scaler_restore)
    with pytest.raises(CheckpointValidationError, match="rollback also failed for scaler"):
        load_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            early_stopping=stopper,
            **_IDENTITY,
        )
    _assert_state_equal(model.state_dict(), before_model)
    _assert_state_equal(optimizer.state_dict(), before_optimizer)
