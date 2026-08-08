from __future__ import annotations

import random
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import SGD, AdamW
from torch.utils.data import DataLoader, TensorDataset

import ecg_trust.training as training_module
from ecg_trust.training import (
    CheckpointValidationError,
    EarlyStopping,
    TrainingRuntime,
    TrainingValidationError,
    evaluate,
    load_checkpoint,
    save_checkpoint,
    seed_everything,
    select_device,
    train_one_epoch,
)


class IdentityLogitModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0))

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs + self.anchor * 0.0


def _classification_loader(*, batch_size: int = 3) -> DataLoader[tuple[Tensor, Tensor]]:
    inputs = torch.tensor(
        [
            [0.2, -0.3, 0.4, 0.1],
            [-0.8, 0.7, 0.5, -0.2],
            [0.9, 0.3, -0.6, 0.4],
            [-0.1, -0.5, 0.8, 0.6],
            [0.4, 0.2, -0.9, -0.7],
            [0.3, -0.4, 0.1, 0.9],
            [-0.6, 0.8, 0.2, -0.3],
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor(
        [
            [1, 0, 0],
            [0, 1, 1],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 0],
            [1, 1, 1],
        ],
        dtype=torch.float32,
    )
    return DataLoader(TensorDataset(inputs, targets), batch_size=batch_size, shuffle=False)


def test_evaluation_uses_sample_weighted_bce_and_collects_aligned_outputs() -> None:
    logits = torch.tensor(
        [
            [-3.0, 2.0],
            [0.2, -0.1],
            [1.2, -2.1],
            [-0.4, 0.7],
            [8.0, -8.0],
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor(
        [[0, 1], [1, 0], [1, 0], [0, 1], [0, 1]], dtype=torch.float32
    )
    loader = DataLoader(TensorDataset(logits, targets), batch_size=4, shuffle=False)
    model = IdentityLogitModel()

    result = evaluate(model, loader, TrainingRuntime(torch.device("cpu"), False))
    expected_loss = F.binary_cross_entropy_with_logits(logits, targets).item()

    assert result.loss == pytest.approx(expected_loss, rel=1e-7)
    assert result.sample_count == 5
    torch.testing.assert_close(result.logits, logits)
    torch.testing.assert_close(result.probabilities, torch.sigmoid(logits))
    torch.testing.assert_close(result.targets, targets)
    assert result.max_gradient_norm is None
    assert not model.training
    assert model.anchor.grad is None


def test_training_updates_model_and_clips_each_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed_everything(14)
    model = nn.Linear(4, 3)
    original_weight = model.weight.detach().clone()
    optimizer = SGD(model.parameters(), lr=0.2)
    original_clip = training_module.clip_grad_norm_
    calls: list[float] = []

    def recording_clip(
        parameters: Iterable[Tensor], max_norm: float, **kwargs: object
    ) -> Tensor:
        calls.append(max_norm)
        return original_clip(parameters, max_norm, **kwargs)

    monkeypatch.setattr(training_module, "clip_grad_norm_", recording_clip)
    result = train_one_epoch(
        model,
        _classification_loader(),
        optimizer,
        TrainingRuntime(torch.device("cpu"), False),
        max_grad_norm=0.25,
    )

    assert model.training
    assert result.sample_count == 7
    assert result.logits.shape == (7, 3)
    assert result.targets.shape == (7, 3)
    assert result.max_gradient_norm is not None
    assert math_is_finite(result.max_gradient_norm)
    assert calls == [0.25, 0.25, 0.25]
    assert not torch.equal(model.weight.detach(), original_weight)


def math_is_finite(value: float) -> bool:
    return not np.isnan(value) and not np.isinf(value)


def test_seed_everything_repeats_all_random_sources() -> None:
    first_generator = seed_everything(12345)
    first = (
        random.random(),
        np.random.random(),
        torch.rand(4),
        torch.rand(4, generator=first_generator),
    )
    second_generator = seed_everything(12345)
    second = (
        random.random(),
        np.random.random(),
        torch.rand(4),
        torch.rand(4, generator=second_generator),
    )

    assert first[0] == second[0]
    assert first[1] == second[1]
    torch.testing.assert_close(first[2], second[2], rtol=0, atol=0)
    torch.testing.assert_close(first[3], second[3], rtol=0, atol=0)
    assert torch.are_deterministic_algorithms_enabled()


def test_device_selection_handles_cpu_cuda_and_bf16(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    runtime = select_device("auto")
    assert runtime.device == torch.device("cpu")
    assert not runtime.bf16_enabled
    with pytest.raises(TrainingValidationError, match="not available"):
        select_device("cuda")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    runtime = select_device("cuda:0")
    assert runtime.device == torch.device("cuda:0")
    assert runtime.bf16_enabled
    assert not select_device("cuda", enable_bf16=False).bf16_enabled
    with pytest.raises(TrainingValidationError, match="index 3"):
        select_device("cuda:3")


def test_rejects_empty_invalid_target_and_wrong_logit_shape() -> None:
    runtime = TrainingRuntime(torch.device("cpu"), False)
    with pytest.raises(TrainingValidationError, match="no samples"):
        evaluate(IdentityLogitModel(), [], runtime)

    bad_targets = TensorDataset(torch.zeros(2, 3), torch.full((2, 3), 0.5))
    with pytest.raises(TrainingValidationError, match="only 0 or 1"):
        evaluate(IdentityLogitModel(), DataLoader(bad_targets), runtime)

    wrong_model = nn.Linear(3, 2)
    binary_targets = TensorDataset(torch.zeros(2, 3), torch.zeros(2, 3))
    with pytest.raises(TrainingValidationError, match="identical"):
        evaluate(wrong_model, DataLoader(binary_targets), runtime)


def test_early_stopping_tracks_min_delta_stops_and_restores() -> None:
    stopper = EarlyStopping(patience=2, mode="min", min_delta=0.1)
    assert stopper.update(1.0, 0)
    assert not stopper.update(0.95, 1)
    assert stopper.bad_epochs == 1
    assert stopper.update(0.89, 2)
    assert stopper.best_epoch == 2
    assert not stopper.update(0.85, 3)
    assert not stopper.update(0.80, 4)
    assert stopper.stopped

    restored = EarlyStopping(patience=1)
    restored.load_state_dict(stopper.state_dict())
    assert restored.state_dict() == stopper.state_dict()
    assert not restored.update(0.0, 5)


def test_checkpoint_round_trip_verifies_identity_and_restores_all_state(
    tmp_path: Path,
) -> None:
    seed_everything(77)
    model = nn.Linear(4, 3)
    optimizer = AdamW(model.parameters(), lr=0.01)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    train_one_epoch(
        model,
        _classification_loader(batch_size=7),
        optimizer,
        TrainingRuntime(torch.device("cpu"), False),
        scaler=scaler,
    )
    expected_parameters = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    stopper = EarlyStopping(patience=3, mode="max", min_delta=0.001)
    stopper.update(0.71, 0)
    config: dict[str, object] = {
        "model": {"name": "tiny_linear", "outputs": 3},
        "folds": [1, 2, 3],
        "learning_rate": 0.01,
    }
    protocol_hash = "sha256:" + "a" * 64
    manifest_hash = "b" * 64
    checkpoint_path = tmp_path / "nested" / "last.ckpt"

    saved = save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        epoch=4,
        protocol_hash=protocol_hash,
        config=config,
        manifest_hash=manifest_hash,
        early_stopping=stopper,
    )
    assert checkpoint_path.is_file()
    assert not list(checkpoint_path.parent.glob(f".{checkpoint_path.name}.*.tmp"))
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert raw["model_state_dict"]
    assert raw["optimizer_state_dict"]["state"]
    assert raw["scaler_state_dict"] == {}
    assert raw["protocol_hash"] == protocol_hash
    assert raw["manifest_hash"] == manifest_hash
    assert raw["config"] == config
    assert raw["config_hash"] == saved.config_hash

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    restored_stopper = EarlyStopping(patience=1)
    loaded = load_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        expected_protocol_hash=protocol_hash,
        expected_config=config,
        expected_manifest_hash=manifest_hash,
        early_stopping=restored_stopper,
    )

    assert loaded == saved
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, expected_parameters[name])
    assert restored_stopper.state_dict() == stopper.state_dict()
    with pytest.raises(CheckpointValidationError, match="config hash"):
        load_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            expected_protocol_hash=protocol_hash,
            expected_config={**config, "learning_rate": 0.02},
            expected_manifest_hash=manifest_hash,
        )


def test_failed_checkpoint_write_preserves_destination(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    destination = tmp_path / "checkpoint.ckpt"
    destination.write_bytes(b"previous checkpoint")
    model = nn.Linear(2, 2)
    optimizer = SGD(model.parameters(), lr=0.1)

    def fail_save(*_: object, **__: object) -> None:
        raise RuntimeError("synthetic write failure")

    monkeypatch.setattr(training_module.torch, "save", fail_save)
    with pytest.raises(RuntimeError, match="synthetic write failure"):
        save_checkpoint(
            destination,
            model=model,
            optimizer=optimizer,
            scaler=None,
            epoch=0,
            protocol_hash="sha256:" + "c" * 64,
            config={"name": "tiny"},
            manifest_hash="d" * 64,
        )

    assert destination.read_bytes() == b"previous checkpoint"
    assert not list(tmp_path.glob(f".{destination.name}.*.tmp"))
