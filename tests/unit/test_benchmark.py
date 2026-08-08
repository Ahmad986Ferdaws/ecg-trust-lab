from __future__ import annotations

import json

import pytest
import torch
from torch import Tensor, nn

from ecg_trust.benchmark import (
    MODEL_SPECS,
    batch_probe_candidates,
    benchmark_train_steps,
    environment_metadata,
    percentile,
    selected_model_specs,
)
from ecg_trust.models import MATCHED_CAPACITY_PRESET, count_parameters


class TinyBenchmarkModel(nn.Module):
    def __init__(self, outputs: int = 5) -> None:
        super().__init__()
        self.features = nn.Conv1d(12, 4, kernel_size=5, padding=2)
        self.classifier = nn.Linear(4, outputs)

    def forward(self, inputs: Tensor) -> Tensor:
        features = self.features(inputs).mean(dim=2)
        return self.classifier(features)


def test_percentile_uses_linear_interpolation() -> None:
    assert percentile([1.0, 2.0, 3.0], 0.0) == 1.0
    assert percentile([1.0, 2.0, 3.0], 0.5) == 2.0
    assert percentile([1.0, 2.0, 3.0], 0.95) == pytest.approx(2.9)
    with pytest.raises(ValueError):
        percentile([], 0.5)


def test_cpu_training_benchmark_emits_json_safe_metrics() -> None:
    model = TinyBenchmarkModel()
    result = benchmark_train_steps(
        model,
        model_name="tiny",
        device=torch.device("cpu"),
        batch_size=2,
        warmup_steps=1,
        measured_steps=3,
        precision="fp32",
        seed=7,
    )
    payload = result.to_dict()

    assert result.batch_size == 2
    assert len(result.step_latencies_ms) == 3
    assert result.median_step_latency_ms > 0
    assert result.p95_step_latency_ms >= result.median_step_latency_ms
    assert result.throughput_samples_per_second > 0
    assert result.peak_allocated_vram_mib is None
    assert result.peak_reserved_vram_mib is None
    assert result.trainable_parameters == count_parameters(model)
    assert torch.isfinite(torch.tensor(result.final_loss))
    json.dumps(payload)


def test_benchmark_rejects_wrong_output_and_cpu_bf16() -> None:
    with pytest.raises(ValueError, match="raw logits"):
        benchmark_train_steps(
            TinyBenchmarkModel(outputs=4),
            model_name="bad",
            device=torch.device("cpu"),
            batch_size=1,
            warmup_steps=0,
            measured_steps=1,
            precision="fp32",
        )
    with pytest.raises(ValueError, match="only on CUDA"):
        benchmark_train_steps(
            TinyBenchmarkModel(),
            model_name="bad_precision",
            device=torch.device("cpu"),
            measured_steps=1,
            precision="bf16",
        )


def test_batch_probe_candidates_include_non_power_of_two_maximum() -> None:
    assert batch_probe_candidates(1) == (1,)
    assert batch_probe_candidates(8) == (1, 2, 4, 8)
    assert batch_probe_candidates(10) == (1, 2, 4, 8, 10)
    with pytest.raises(ValueError):
        batch_probe_candidates(0)


def test_model_specs_cover_default_and_matched_comparisons() -> None:
    assert set(MODEL_SPECS) == {
        "default_resnet",
        "default_transformer",
        "matched_resnet",
        "matched_transformer",
    }
    matched_resnet = MODEL_SPECS["matched_resnet"].build()
    matched_transformer = MODEL_SPECS["matched_transformer"].build()
    assert count_parameters(matched_resnet) == MATCHED_CAPACITY_PRESET.expected_resnet_parameters
    assert (
        count_parameters(matched_transformer)
        == MATCHED_CAPACITY_PRESET.expected_transformer_parameters
    )
    json.dumps(MODEL_SPECS["matched_transformer"].metadata())


def test_model_spec_selection_validates_names() -> None:
    selected = selected_model_specs(["default_transformer", "default_resnet"])
    assert [spec.name for spec in selected] == ["default_transformer", "default_resnet"]
    with pytest.raises(ValueError, match="at least one"):
        selected_model_specs([])
    with pytest.raises(ValueError, match="unknown"):
        selected_model_specs(["missing"])


def test_cpu_environment_metadata_is_json_safe() -> None:
    metadata = environment_metadata(torch.device("cpu"))
    assert metadata["device"] == "cpu"
    assert metadata["gpu"] is None
    json.dumps(metadata)
