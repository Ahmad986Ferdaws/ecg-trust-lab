from __future__ import annotations

import json
from collections.abc import Callable

import pytest
import torch
from torch import Tensor, nn

from ecg_trust.explain import (
    attribution_stability_similarity,
    cross_method_temporal_similarity,
    deletion_faithfulness_curve,
    grad_cam_1d,
    integrated_gradients,
    integrated_gradients_with_delta,
    lead_ablation_faithfulness_curve,
    normalize_attributions,
    parameter_randomization_comparison,
    randomized_model_copy,
    temporal_faithfulness_curve,
    temporal_occlusion,
    validate_ecg_batch,
)
from ecg_trust.models import ECGTransformer, ECGTransformerConfig, ResNet1D, ResNet1DConfig


class LeadMeanModel(nn.Module):
    """Small deterministic five-logit model for perturbation tests."""

    def forward(self, inputs: Tensor) -> Tensor:
        means = inputs.mean(dim=2)
        return torch.stack(
            (
                means.sum(dim=1),
                means[:, 0],
                means[:, 1],
                means[:, 2],
                means[:, 3],
            ),
            dim=1,
        )


def _tiny_resnet() -> ResNet1D:
    return ResNet1D(
        ResNet1DConfig(
            stage_channels=(8,),
            blocks_per_stage=(1,),
            block_dropout=0.0,
            classifier_dropout=0.0,
        )
    )


def _tiny_transformer() -> ECGTransformer:
    return ECGTransformer(
        ECGTransformerConfig(
            patch_size=100,
            patch_stride=100,
            embedding_dim=32,
            depth=1,
            num_heads=4,
            mlp_ratio=2.0,
            dropout=0.0,
            attention_dropout=0.0,
        )
    )


def test_signed_normalization_preserves_sign_and_zero_maps() -> None:
    attributions = torch.zeros(2, 12, 1000)
    attributions[0, 0, 0] = 2.0
    attributions[0, 0, 1] = -1.0

    normalized = normalize_attributions(attributions)

    assert normalized[0, 0, 0].item() == 1.0
    assert normalized[0, 0, 1].item() == -0.5
    assert torch.count_nonzero(normalized[1]) == 0


def test_grad_cam_is_signed_temporal_and_restores_training_mode() -> None:
    torch.manual_seed(3)
    model = _tiny_resnet().train()
    inputs = torch.randn(2, 12, 1000)

    attributions = grad_cam_1d(model, inputs, torch.tensor([0, 1]))

    assert attributions.shape == (2, 1, 1000)
    assert torch.isfinite(attributions).all()
    assert attributions.abs().amax() <= 1.0
    assert model.training


@pytest.mark.parametrize("factory", [_tiny_resnet, _tiny_transformer])
def test_integrated_gradients_supports_both_architectures(
    factory: Callable[[], nn.Module],
) -> None:
    torch.manual_seed(5)
    model = factory().eval()
    inputs = torch.randn(1, 12, 1000)

    attributions = integrated_gradients(model, inputs, 2, n_steps=4)

    assert attributions.shape == inputs.shape
    assert torch.isfinite(attributions).all()
    assert attributions.abs().amax() <= 1.0


def test_integrated_gradients_returns_finite_completeness_delta() -> None:
    model = LeadMeanModel().eval()
    inputs = torch.randn(2, 12, 1000)

    attributions, delta = integrated_gradients_with_delta(
        model,
        inputs,
        torch.tensor([0, 1]),
        n_steps=8,
        internal_batch_size=4,
    )

    assert attributions.shape == inputs.shape
    assert delta.shape == (2,)
    assert torch.isfinite(delta).all()


def test_temporal_occlusion_returns_lead_time_tensor() -> None:
    model = LeadMeanModel()
    inputs = torch.randn(1, 12, 1000)

    attributions = temporal_occlusion(
        model,
        inputs,
        0,
        window_samples=250,
        stride_samples=250,
        perturbations_per_eval=2,
    )

    assert attributions.shape == inputs.shape
    assert torch.isfinite(attributions).all()
    assert attributions.abs().amax() <= 1.0


def test_temporal_deletion_curve_and_summary_are_well_formed() -> None:
    model = LeadMeanModel()
    inputs = torch.zeros(2, 12, 1000)
    inputs[:, 0] = 2.0
    inputs[:, 1] = 1.0
    attributions = torch.zeros(2, 1, 1000)
    attributions[:, :, :500] = 2.0
    attributions[:, :, 500:] = 1.0

    result = deletion_faithfulness_curve(
        model,
        inputs,
        attributions,
        targets=torch.tensor([0, 0]),
        fractions=(0.0, 0.5, 1.0),
    )
    summary = result.summary()

    assert result.target_probabilities.shape == (2, 3)
    assert result.target_logits.shape == (2, 3)
    assert torch.all(result.target_probabilities[:, :-1] >= result.target_probabilities[:, 1:])
    assert torch.allclose(
        result.target_probabilities[:, -1],
        torch.full_like(result.target_probabilities[:, -1], 0.5),
    )
    assert summary["method"] == "temporal_deletion"
    json.dumps(summary)


def test_insertion_least_and_random_controls_are_reproducible() -> None:
    model = LeadMeanModel()
    inputs = torch.zeros(1, 12, 1000)
    inputs[:, :, :500] = 2.0
    inputs[:, :, 500:] = 0.25
    attributions = torch.zeros(1, 1, 1000)
    attributions[:, :, :500] = 2.0
    attributions[:, :, 500:] = 0.1

    insertion = temporal_faithfulness_curve(
        model,
        inputs,
        attributions,
        0,
        operation="insertion",
        ranking="most_important",
        fractions=(0.0, 0.5, 1.0),
        temperature=2.0,
    )
    least = temporal_faithfulness_curve(
        model,
        inputs,
        attributions,
        0,
        operation="deletion",
        ranking="least_important",
        fractions=(0.0, 0.5, 1.0),
    )
    random_first = temporal_faithfulness_curve(
        model,
        inputs,
        attributions,
        0,
        operation="deletion",
        ranking="random",
        random_seed=41,
        fractions=(0.0, 0.5, 1.0),
    )
    random_second = temporal_faithfulness_curve(
        model,
        inputs,
        attributions,
        0,
        operation="deletion",
        ranking="random",
        random_seed=41,
        fractions=(0.0, 0.5, 1.0),
    )

    assert insertion.target_probabilities[0, 0].item() == pytest.approx(0.5)
    assert insertion.target_probabilities[0, -1] > insertion.target_probabilities[0, 0]
    assert least.target_probabilities[0, 1] > insertion.target_probabilities[0, 1]
    torch.testing.assert_close(
        random_first.target_probabilities, random_second.target_probabilities
    )
    torch.testing.assert_close(random_first.target_logits, random_second.target_logits)
    assert random_first.summary()["mean_target_logits"]

    with pytest.raises(ValueError, match="random_seed"):
        temporal_faithfulness_curve(
            model,
            inputs,
            attributions,
            0,
            operation="deletion",
            ranking="random",
        )


def test_faithfulness_can_score_positive_and_negative_target_status() -> None:
    model = LeadMeanModel()
    inputs = torch.ones(2, 12, 1000)
    attributions = torch.ones(2, 1, 1000)

    result = temporal_faithfulness_curve(
        model,
        inputs,
        attributions,
        torch.tensor([0, 0]),
        fractions=(0.0, 1.0),
        target_signs=torch.tensor([1, -1]),
    )

    torch.testing.assert_close(result.target_logits[0], -result.target_logits[1])
    torch.testing.assert_close(
        result.target_probabilities[0] + result.target_probabilities[1],
        torch.ones(2, dtype=torch.float64),
    )

    with pytest.raises(ValueError, match=r"-1 or \+1"):
        temporal_faithfulness_curve(
            model,
            inputs,
            attributions,
            0,
            fractions=(0.0, 1.0),
            target_signs=torch.tensor([1, 0]),
        )


def test_lead_ablation_curve_uses_lead_specific_attribution() -> None:
    model = LeadMeanModel()
    inputs = torch.zeros(1, 12, 1000)
    inputs[:, 0] = 2.0
    inputs[:, 1] = 1.0
    attributions = torch.zeros_like(inputs)
    attributions[:, 0] = 4.0
    attributions[:, 1] = 2.0

    result = lead_ablation_faithfulness_curve(model, inputs, attributions, 0)

    assert result.fractions.shape == (13,)
    assert result.target_probabilities.shape == (1, 13)
    assert result.target_probabilities[0, 0] > result.target_probabilities[0, 1]
    assert result.target_probabilities[0, -1].item() == pytest.approx(0.5)
    json.dumps(result.summary())

    with pytest.raises(ValueError, match="lead-specific"):
        lead_ablation_faithfulness_curve(model, inputs, attributions[:, :1], 0)

    random_first = lead_ablation_faithfulness_curve(
        model, inputs, attributions, 0, ranking="random", random_seed=17
    )
    random_second = lead_ablation_faithfulness_curve(
        model, inputs, attributions, 0, ranking="random", random_seed=17
    )
    torch.testing.assert_close(
        random_first.target_probabilities, random_second.target_probabilities
    )


def test_stability_similarity_preserves_direction_and_is_json_safe() -> None:
    reference = torch.randn(2, 12, 1000)
    identical = attribution_stability_similarity(reference, reference)
    opposite = attribution_stability_similarity(reference, -reference)
    zero = attribution_stability_similarity(
        torch.zeros_like(reference), torch.zeros_like(reference)
    )

    assert torch.allclose(identical.values, torch.ones_like(identical.values))
    assert torch.allclose(opposite.values, -torch.ones_like(opposite.values))
    assert torch.allclose(zero.values, torch.ones_like(zero.values))
    json.dumps(identical.summary())


def test_cross_method_similarity_aggregates_leads_and_reports_rank_agreement() -> None:
    temporal = torch.arange(1000, dtype=torch.float32).view(1, 1, 1000)
    lead_specific = temporal.expand(1, 12, 1000).clone()
    identical = cross_method_temporal_similarity(temporal, lead_specific)
    reversed_map = cross_method_temporal_similarity(temporal, lead_specific.flip(-1))

    assert identical.cosine.item() == pytest.approx(1.0)
    assert identical.spearman.item() == pytest.approx(1.0)
    assert reversed_map.spearman.item() == pytest.approx(-1.0)
    assert identical.cosine_valid.item() is True
    assert identical.spearman_valid.item() is True
    json.dumps(identical.summary())

    constant = cross_method_temporal_similarity(torch.ones(1, 1, 1000), torch.ones(1, 12, 1000))
    assert constant.cosine_valid.item() is True
    assert constant.spearman_valid.item() is False
    assert constant.spearman.item() == 0.0
    assert constant.summary()["mean_spearman"] is None


def test_parameter_randomization_is_reproducible_and_non_mutating() -> None:
    torch.manual_seed(17)
    original = _tiny_resnet().train()
    original_state = {name: value.detach().clone() for name, value in original.state_dict().items()}

    first = randomized_model_copy(original, seed=29)
    second = randomized_model_copy(original, seed=29)

    assert first.training and second.training
    for name, value in original.state_dict().items():
        assert torch.equal(value, original_state[name])
    for first_parameter, second_parameter in zip(
        first.parameters(), second.parameters(), strict=True
    ):
        assert torch.equal(first_parameter, second_parameter)
    assert any(
        not torch.equal(original_parameter, randomized_parameter)
        for original_parameter, randomized_parameter in zip(
            original.parameters(), first.parameters(), strict=True
        )
    )

    reference = torch.randn(2, 12, 1000)
    comparison = parameter_randomization_comparison(reference, -reference, seed=29)
    assert torch.allclose(comparison.values, -torch.ones_like(comparison.values))
    assert comparison.summary()["method"] == "parameter_randomization_seed_29"
    json.dumps(comparison.summary())


@pytest.mark.parametrize(
    "inputs, message",
    [
        (torch.zeros(12, 1000), "shaped"),
        (torch.zeros(1, 11, 1000), "shaped"),
        (torch.zeros(1, 12, 999), "shaped"),
        (torch.zeros(0, 12, 1000), "empty"),
        (torch.zeros(1, 12, 1000, dtype=torch.int64), "floating-point"),
    ],
)
def test_canonical_input_validation(inputs: Tensor, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        validate_ecg_batch(inputs)


def test_targets_and_faithfulness_fractions_fail_closed() -> None:
    model = LeadMeanModel()
    inputs = torch.zeros(1, 12, 1000)
    attributions = torch.zeros_like(inputs)

    with pytest.raises(ValueError, match="target indices"):
        integrated_gradients(model, inputs, 5, n_steps=2)
    with pytest.raises(ValueError, match="start at 0"):
        deletion_faithfulness_curve(model, inputs, attributions, 0, fractions=(0.1, 1.0))
    with pytest.raises(ValueError, match="strictly increasing"):
        deletion_faithfulness_curve(model, inputs, attributions, 0, fractions=(0.0, 0.5, 0.5, 1.0))


def test_attribution_methods_validate_five_logit_output() -> None:
    bad_model = nn.Sequential(nn.Flatten(), nn.Linear(12_000, 4))
    inputs = torch.zeros(1, 12, 1000)

    with pytest.raises(ValueError, match="raw logits"):
        integrated_gradients(bad_model, inputs, 0, n_steps=2)
    with pytest.raises(ValueError, match="raw logits"):
        temporal_occlusion(
            bad_model,
            inputs,
            0,
            window_samples=1000,
            stride_samples=1000,
            perturbations_per_eval=1,
        )
