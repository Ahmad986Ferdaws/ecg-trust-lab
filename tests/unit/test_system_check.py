from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from ecg_trust import system_check


def test_cuda_device_probe_failure_is_reported_as_json(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def failed_probe() -> dict[str, object]:
        raise RuntimeError("driver initialization failed")

    monkeypatch.setattr(system_check, "_device_report", failed_probe)
    with pytest.raises(SystemExit) as error:
        system_check.main([])
    assert error.value.code == 1
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "FAIL"
    assert "driver initialization failed" in report["error"]


def test_unavailable_cuda_still_fails_instead_of_using_cpu(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(system_check.torch.cuda, "is_available", lambda: False)
    with pytest.raises(SystemExit) as error:
        system_check.main([])
    assert error.value.code == 1
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "FAIL"
    assert "CUDA is unavailable" in report["error"]
    assert report["environment"]["cuda_available"] is False


@pytest.mark.parametrize("scaling", [False, True])
def test_finite_backward_step_matches_unscaled_float32_update(scaling: bool) -> None:
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = torch.amp.GradScaler("cpu", enabled=scaling)
    loss = model(torch.tensor([[2.0]])).square().mean()
    system_check._backward_step(loss, model, optimizer, scaler)
    assert model.weight.item() == pytest.approx(0.2, abs=1e-6)
    assert model.weight.grad is not None
    assert model.weight.grad.item() == pytest.approx(8.0)


def test_fp16_scaling_preserves_gradients_that_underflow_without_scaling() -> None:
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=1_000_000)
    # Reproduce the previous unscaled backward/step with a real half-precision
    # autograd path: the small derivative disappears before reaching FP32 weights.
    unscaled_loss = (model.weight.to(torch.float16) * 0.0001).square().float().sum()
    unscaled_loss.backward()
    optimizer.step()
    assert model.weight.item() == 1.0
    optimizer.zero_grad(set_to_none=True)
    scaler = torch.amp.GradScaler("cpu", enabled=True, init_scale=1024.0)
    loss = (model.weight.to(torch.float16) * 0.0001).square().float().sum()
    system_check._backward_step(loss, model, optimizer, scaler)
    assert model.weight.item() == pytest.approx(0.98, abs=0.001)


@pytest.mark.parametrize("invalid_loss", [False, True])
def test_nonfinite_loss_or_gradient_never_reaches_optimizer(invalid_loss: bool) -> None:
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = torch.amp.GradScaler("cpu", enabled=True)
    # sqrt(0) has a finite forward value and an infinite derivative.
    loss = model.weight.sum() * float("inf") if invalid_loss else model.weight.sqrt().sum()
    with pytest.raises(RuntimeError, match="Non-finite"):
        system_check._backward_step(loss, model, optimizer, scaler)
    assert model.weight.item() == 0.0


@pytest.mark.parametrize("bf16", [False, True])
def test_smoke_loop_selects_scaling_and_checks_each_step_on_cpu(
    monkeypatch: pytest.MonkeyPatch, bf16: bool
) -> None:
    # Only device selection/runtime queries are substituted. Autocast, autograd,
    # GradScaler and AdamW execute real CPU tensor operations in both modes.
    configured_scalers: list[tuple[str, bool]] = []
    checked_steps: list[float] = []
    original_step = system_check._backward_step

    def scaler(device: str, *, enabled: bool) -> torch.amp.GradScaler:
        configured_scalers.append((device, enabled))
        return torch.amp.GradScaler("cpu", enabled=enabled)

    def checked_step(*args: object) -> None:
        checked_steps.append(float(args[0].detach()))
        original_step(*args)

    simulated_torch = SimpleNamespace(
        device=lambda _: torch.device("cpu"),
        cuda=SimpleNamespace(
            is_available=lambda: True,
            is_bf16_supported=lambda: bf16,
            reset_peak_memory_stats=lambda: None,
            synchronize=lambda: None,
            max_memory_allocated=lambda: 0,
        ),
        amp=SimpleNamespace(GradScaler=scaler),
        optim=torch.optim,
        bfloat16=torch.bfloat16,
        float16=torch.float16,
        float32=torch.float32,
        randn=torch.randn,
        randint=torch.randint,
        isfinite=torch.isfinite,
        tensor=torch.tensor,
        autocast=lambda device_type, dtype: torch.autocast("cpu", dtype=dtype),
    )
    monkeypatch.setattr(system_check, "torch", simulated_torch)
    monkeypatch.setattr(system_check, "_backward_step", checked_step)
    monkeypatch.setattr(
        system_check,
        "CudaSmokeModel",
        lambda: nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(12, 5)),
    )
    report = system_check._training_smoke_test()
    assert configured_scalers == [("cuda", not bf16)]
    assert len(checked_steps) == report["steps"] == 3
    assert report["gradient_scaling"] is not bf16
    assert report["gradients_finite"] is True


@pytest.mark.parametrize("available", [False, True])
def test_environment_only_never_invokes_training_and_never_claims_pass(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], available: bool
) -> None:
    environment = {"cuda_available": available, "python": "3.12.test", "torch": "test"}
    monkeypatch.setattr(system_check, "_device_report", lambda: environment)

    def forbidden_training() -> None:
        pytest.fail("environment-only diagnostics must not run the training smoke test")

    monkeypatch.setattr(system_check, "_training_smoke_test", forbidden_training)
    system_check.main(["--environment-only"])
    report = json.loads(capsys.readouterr().out)
    assert report["environment"] == environment
    assert report["scope"] == "environment_only"
    assert report["status"] == "INFORMATIONAL"
    assert report["training_smoke_test_performed"] is False
    assert "training_smoke_test" not in report
    assert "does not satisfy" in report["notice"]


def test_default_command_preserves_training_success_report(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls = []

    def train() -> dict[str, object]:
        calls.append("training")
        return {"steps": 3, "gradients_finite": True}

    monkeypatch.setattr(system_check, "_device_report", lambda: {"cuda_available": True})
    monkeypatch.setattr(system_check, "_training_smoke_test", train)
    system_check.main([])
    assert calls == ["training"]
    assert json.loads(capsys.readouterr().out) == {
        "environment": {"cuda_available": True},
        "training_smoke_test": {"steps": 3, "gradients_finite": True},
        "status": "PASS",
    }


@pytest.mark.parametrize("args,exit_code", [(["--help"], 0), (["--unknown-option"], 2)])
def test_argument_handling_does_not_probe_or_train(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    args: list[str],
    exit_code: int,
) -> None:
    def forbidden_probe() -> None:
        pytest.fail("argument handling must happen before device probing")

    monkeypatch.setattr(system_check, "_device_report", forbidden_probe)
    monkeypatch.setattr(system_check, "_training_smoke_test", forbidden_probe)
    with pytest.raises(SystemExit) as error:
        system_check.main(args)
    assert error.value.code == exit_code
    captured = capsys.readouterr()
    if exit_code == 0:
        assert "--environment-only" in captured.out
        assert captured.err == ""
    else:
        assert captured.out == ""
        assert "unrecognized arguments" in captured.err


def test_environment_probe_failure_retains_diagnostic_scope(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def failed_probe() -> None:
        raise RuntimeError("synthetic probe failure")

    monkeypatch.setattr(system_check, "_device_report", failed_probe)
    with pytest.raises(SystemExit) as error:
        system_check.main(["--environment-only"])
    assert error.value.code == 1
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "FAIL"
    assert report["scope"] == "environment_only"
    assert report["training_smoke_test_performed"] is False
    assert "synthetic probe failure" in report["error"]


def test_main_reads_command_line_arguments_when_not_supplied(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(system_check.sys, "argv", ["ecg-verify", "--environment-only"])
    monkeypatch.setattr(system_check, "_device_report", lambda: {"cuda_available": False})
    system_check.main()
    assert json.loads(capsys.readouterr().out)["status"] == "INFORMATIONAL"
