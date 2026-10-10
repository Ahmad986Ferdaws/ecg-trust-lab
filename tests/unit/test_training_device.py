from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import pytest
import torch

from ecg_trust.training import select_device


@pytest.mark.parametrize("current,requested", [(0, 1), (1, 0)])
def test_bf16_capability_uses_requested_device_and_restores_current_device(
    monkeypatch: pytest.MonkeyPatch, current: int, requested: int
) -> None:
    active_device = current
    queried: list[int] = []

    @contextmanager
    def device_context(device: torch.device) -> Iterator[None]:
        nonlocal active_device
        previous = active_device
        active_device = device.index if device.index is not None else previous
        try:
            yield
        finally:
            active_device = previous

    def bf16_supported() -> bool:
        queried.append(active_device)
        return active_device == 1

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "device", device_context)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", bf16_supported)

    result = select_device(f"cuda:{requested}")

    assert result.device == torch.device(f"cuda:{requested}")
    assert result.bf16_enabled is (requested == 1)
    assert queried == [requested]
    assert active_device == current


@pytest.mark.parametrize("requested", ["cpu", "cuda:1"])
def test_cpu_or_disabled_bf16_never_enters_cuda_context(
    monkeypatch: pytest.MonkeyPatch, requested: str
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(
        torch.cuda, "device", lambda _: pytest.fail("CUDA capability context entered")
    )

    result = select_device(requested, enable_bf16=requested == "cpu")

    assert not result.bf16_enabled
