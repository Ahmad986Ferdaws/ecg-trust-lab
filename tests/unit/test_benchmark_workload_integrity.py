from __future__ import annotations

import json
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from torch import nn

from ecg_trust import benchmark
from scripts import benchmark_models


class TinyModel(nn.Module):
    def __init__(self, hidden: int = 7) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(12, hidden), nn.ReLU(), nn.Linear(hidden, 5))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs.mean(dim=2))


def _install_tiny_specs(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[torch.Tensor]]:
    initial_weights: dict[str, list[torch.Tensor]] = {
        "default_resnet": [],
        "default_transformer": [],
    }

    def spec(name: str, hidden: int) -> SimpleNamespace:
        def build() -> nn.Module:
            model = TinyModel(hidden)
            initial_weights[name].append(
                torch.cat([p.detach().flatten() for p in model.parameters()])
            )
            return model

        return SimpleNamespace(name=name, build=build, metadata=lambda: {"name": name})

    specs = {
        "default_resnet": spec("default_resnet", 7),
        "default_transformer": spec("default_transformer", 11),
    }
    monkeypatch.setattr(
        benchmark_models, "selected_model_specs", lambda names: tuple(specs[n] for n in names)
    )
    return initial_weights


def _run_models(names: list[str]) -> dict[str, object]:
    args = benchmark_models.parse_args(["--cpu-smoke", "--seed", "2026", "--models", *names])
    return benchmark_models.run(args)


def test_seed_controls_factory_weights_and_losses_independent_of_model_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    weights = _install_tiny_specs(monkeypatch)
    losses: dict[str, list[float]] = {name: [] for name in weights}
    for ambient_seed, names in (
        (111, ["default_resnet", "default_transformer"]),
        (222, ["default_transformer", "default_resnet"]),
        (333, ["default_resnet"]),
    ):
        torch.manual_seed(ambient_seed)
        payload = _run_models(names)
        json.dumps(payload, allow_nan=False)
        for result in payload["results"]:
            losses[result["model_name"]].append(result["final_loss"])
    for name, runs in weights.items():
        assert all(torch.equal(runs[0], run) for run in runs[1:]), name
        assert len(set(losses[name])) == 1


def test_each_probe_factory_is_seeded_and_does_not_change_next_workload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    weights = _install_tiny_specs(monkeypatch)
    before = _run_models(["default_resnet"])["results"][0]["final_loss"]
    factory = benchmark_models.selected_model_specs(["default_resnet"])[0].build
    real_benchmark = benchmark.benchmark_train_steps

    def on_cpu(model: nn.Module, **kwargs: object) -> benchmark.TrainBenchmarkResult:
        # Exercise real changing-batch training/RNG behavior on CPU. This is a
        # probe coordination test, not a claim to have executed a CUDA workload.
        assert kwargs["device"] == torch.device("cuda")
        kwargs["device"] = torch.device("cpu")
        return real_benchmark(model, **kwargs)

    monkeypatch.setattr(benchmark, "benchmark_train_steps", on_cpu)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    result = benchmark.probe_safe_batch_size(
        factory,
        model_name="default_resnet",
        device=torch.device("cuda"),
        maximum_batch_size=4,
        precision="fp32",
        seed=2026,
    )
    assert result.maximum_successful_batch == 4
    assert [attempt.batch_size for attempt in result.attempts] == [1, 2, 4]
    after = _run_models(["default_resnet"])["results"][0]["final_loss"]
    assert before == after
    runs = weights["default_resnet"]
    assert len(runs) == 5
    assert all(torch.equal(runs[0], run) for run in runs[1:])


def test_unknown_probe_precision_fails_before_factory_or_rng_mutation() -> None:
    def must_not_build() -> nn.Module:
        raise AssertionError("invalid probe invoked the factory")

    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="precision must be"):
        benchmark.probe_safe_batch_size(
            must_not_build,
            model_name="tiny",
            device=torch.device("cuda"),
            maximum_batch_size=4,
            precision=cast(benchmark.Precision, "fp16"),
        )
    assert torch.equal(torch.get_rng_state(), rng)


@pytest.mark.parametrize("learning_rate", [float("inf"), float("-inf"), float("nan"), 0.0, -0.1])
def test_invalid_learning_rate_is_rejected_before_model_or_rng_mutation(
    learning_rate: float,
) -> None:
    model = TinyModel().eval()
    state = {name: value.clone() for name, value in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError):
        benchmark.benchmark_train_steps(
            model,
            model_name="tiny",
            device=torch.device("cpu"),
            batch_size=1,
            warmup_steps=0,
            measured_steps=1,
            precision="fp32",
            learning_rate=learning_rate,
        )
    assert not model.training
    assert torch.equal(torch.get_rng_state(), rng)
    assert all(torch.equal(state[name], value) for name, value in model.state_dict().items())


@pytest.mark.parametrize("precision", ["fp16", "FP32", ""])
def test_unknown_precision_fails_before_model_or_rng_mutation(precision: str) -> None:
    model = TinyModel().eval()
    state = {name: value.clone() for name, value in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="precision must be"):
        benchmark.benchmark_train_steps(
            model,
            model_name="tiny",
            device=torch.device("cpu"),
            batch_size=1,
            warmup_steps=0,
            measured_steps=1,
            precision=cast(benchmark.Precision, precision),
        )
    assert not model.training
    assert torch.equal(torch.get_rng_state(), rng)
    assert all(torch.equal(state[name], value) for name, value in model.state_dict().items())
