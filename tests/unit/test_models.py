from __future__ import annotations

import json
from collections.abc import Callable

import pytest
import torch
from torch import Tensor, nn

from ecg_trust.constants import SUPERCLASSES
from ecg_trust.models import (
    MATCHED_CAPACITY_PRESET,
    ECGTransformer,
    ECGTransformerConfig,
    ResNet1D,
    ResNet1DConfig,
    TrainingPrevalencePredictor,
    build_matched_capacity_pair,
    comparison_metadata,
    count_parameters,
)


def _small_resnet() -> ResNet1D:
    return ResNet1D(
        ResNet1DConfig(
            stage_channels=(16, 32),
            blocks_per_stage=(1, 1),
            block_dropout=0.1,
            classifier_dropout=0.1,
        )
    )


def _small_transformer() -> ECGTransformer:
    return ECGTransformer(
        ECGTransformerConfig(
            patch_size=50,
            patch_stride=50,
            embedding_dim=64,
            depth=2,
            num_heads=4,
            mlp_ratio=2.0,
            dropout=0.1,
            attention_dropout=0.1,
        )
    )


MODEL_FACTORIES: tuple[Callable[[], nn.Module], ...] = (_small_resnet, _small_transformer)


@pytest.mark.parametrize("factory", MODEL_FACTORIES)
@pytest.mark.parametrize("batch_size", [1, 4])
def test_models_return_five_raw_logits_for_variable_batches(
    factory: Callable[[], nn.Module], batch_size: int
) -> None:
    model = factory().eval()
    inputs = torch.randn(batch_size, 12, 1000)
    with torch.no_grad():
        logits = model(inputs)

    assert logits.shape == (batch_size, len(SUPERCLASSES))
    assert torch.isfinite(logits).all()
    assert not any(isinstance(module, nn.Sigmoid) for module in model.modules())
    assert model.output_labels == SUPERCLASSES  # type: ignore[attr-defined]


@pytest.mark.parametrize("factory", MODEL_FACTORIES)
def test_models_have_finite_forward_and_backward(factory: Callable[[], nn.Module]) -> None:
    torch.manual_seed(7)
    model = factory().train()
    inputs = torch.randn(2, 12, 1000, requires_grad=True)
    targets = torch.randint(0, 2, (2, len(SUPERCLASSES))).float()

    logits = model(inputs)
    loss = nn.functional.binary_cross_entropy_with_logits(logits, targets)
    loss.backward()

    assert torch.isfinite(logits).all()
    assert torch.isfinite(loss)
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    assert gradients and all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients if gradient is not None)


@pytest.mark.parametrize("factory", MODEL_FACTORIES)
def test_eval_forward_is_deterministic(factory: Callable[[], nn.Module]) -> None:
    torch.manual_seed(11)
    model = factory().eval()
    inputs = torch.randn(3, 12, 1000)
    with torch.no_grad():
        first = model(inputs)
        second = model(inputs)
    assert torch.equal(first, second)


def test_seeded_initialization_is_reproducible() -> None:
    config = ECGTransformerConfig(embedding_dim=64, depth=1, num_heads=4)
    torch.manual_seed(23)
    first = ECGTransformer(config)
    torch.manual_seed(23)
    second = ECGTransformer(config)

    for first_parameter, second_parameter in zip(
        first.parameters(), second.parameters(), strict=True
    ):
        assert torch.equal(first_parameter, second_parameter)


def test_models_support_noncanonical_time_length() -> None:
    inputs = torch.randn(2, 12, 777)
    with torch.no_grad():
        assert _small_resnet().eval()(inputs).shape == (2, 5)
        assert _small_transformer().eval()(inputs).shape == (2, 5)


def test_default_parameter_counts_are_gpu_practical() -> None:
    resnet_parameters = sum(parameter.numel() for parameter in ResNet1D().parameters())
    transformer_parameters = sum(parameter.numel() for parameter in ECGTransformer().parameters())

    assert 2_000_000 <= resnet_parameters <= 15_000_000
    assert 2_000_000 <= transformer_parameters <= 15_000_000


@pytest.mark.parametrize("factory", MODEL_FACTORIES)
def test_models_reject_wrong_input_contract(factory: Callable[[], nn.Module]) -> None:
    model = factory()
    with pytest.raises(ValueError, match="12 leads"):
        model(torch.randn(2, 11, 1000))
    with pytest.raises(ValueError, match="shaped"):
        model(torch.randn(12, 1000))


@pytest.mark.parametrize(
    "constructor",
    [
        lambda: ResNet1DConfig(stage_channels=(16,), blocks_per_stage=(1, 1)),
        lambda: ResNet1DConfig(block_kernel_size=4),
        lambda: ECGTransformerConfig(embedding_dim=63, num_heads=8),
        lambda: ECGTransformerConfig(patch_size=1001),
        lambda: ECGTransformerConfig(lead_stem_kernel_size=4),
    ],
)
def test_invalid_model_configs_fail_fast(constructor: Callable[[], object]) -> None:
    with pytest.raises(ValueError):
        constructor()


def test_feature_interfaces_expose_attribution_tensors() -> None:
    inputs = torch.randn(2, 12, 1000)
    resnet_features: Tensor = _small_resnet().eval().forward_features(inputs)
    transformer = _small_transformer().eval()
    transformer_tokens: Tensor = transformer.forward_tokens(inputs)
    transformer_features: Tensor = transformer.forward_features(inputs)

    assert resnet_features.ndim == 3
    assert transformer_tokens.shape == (2, 21, 64)
    assert transformer_features.shape == (2, 64)


def test_training_prevalence_predictor_is_parameter_free() -> None:
    targets = torch.tensor(
        [
            [1, 0, 1, 0, 0],
            [1, 1, 0, 0, 0],
            [0, 0, 0, 1, 0],
        ],
        dtype=torch.float32,
    )
    baseline = TrainingPrevalencePredictor.from_targets(targets, smoothing=0.5)
    logits = baseline(torch.randn(4, 12, 1000))
    expected = (targets.sum(dim=0) + 0.5) / (targets.shape[0] + 1.0)

    assert logits.shape == (4, 5)
    assert torch.allclose(torch.sigmoid(logits[0]), expected)
    assert torch.equal(logits[0], logits[-1])
    assert sum(parameter.numel() for parameter in baseline.parameters()) == 0
    assert "constant_logits" in baseline.state_dict()


@pytest.mark.parametrize(
    "targets",
    [
        torch.empty(0, 5),
        torch.zeros(2, 4),
        torch.full((2, 5), 0.5),
        torch.full((2, 5), float("nan")),
    ],
)
def test_training_prevalence_predictor_rejects_invalid_targets(targets: Tensor) -> None:
    with pytest.raises(ValueError):
        TrainingPrevalencePredictor.from_targets(targets)


def test_matched_capacity_preset_is_within_fifteen_percent() -> None:
    resnet, transformer = build_matched_capacity_pair()
    resnet_count = count_parameters(resnet)
    transformer_count = count_parameters(transformer)
    relative_gap = abs(transformer_count / resnet_count - 1.0)

    assert resnet_count == MATCHED_CAPACITY_PRESET.expected_resnet_parameters
    assert transformer_count == MATCHED_CAPACITY_PRESET.expected_transformer_parameters
    assert relative_gap <= 0.15
    assert relative_gap < 0.002
    assert transformer.config.embedding_dim == 320
    assert transformer.config.depth == 7


def test_matched_capacity_metadata_is_json_serializable_and_observed() -> None:
    resnet, transformer = build_matched_capacity_pair()
    metadata = comparison_metadata(resnet, transformer)
    encoded = json.dumps(metadata, sort_keys=True)

    assert "ptbxl_100hz_matched_capacity_v1" in encoded
    assert metadata["observed_within_tolerance"] is True
    assert MATCHED_CAPACITY_PRESET.metadata()["within_tolerance"] is True


def test_parameter_count_can_include_frozen_parameters() -> None:
    model = nn.Sequential(nn.Linear(4, 3), nn.Linear(3, 2))
    model[0].requires_grad_(False)

    assert count_parameters(model) == 8
    assert count_parameters(model, trainable_only=False) == 23
