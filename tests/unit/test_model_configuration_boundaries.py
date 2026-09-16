from __future__ import annotations

import pytest
import torch

from ecg_trust.models import ECGTransformerConfig, ResNet1DConfig, TrainingPrevalencePredictor


@pytest.mark.parametrize(
    "field", ["signal_length", "patch_size", "patch_stride", "embedding_dim", "depth",
              "num_heads", "lead_stem_kernel_size"],
)
@pytest.mark.parametrize("value", [True, 1.5])
def test_transformer_dimensions_require_integers(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        ECGTransformerConfig(**{field: value})


@pytest.mark.parametrize("ratio", [float("nan"), float("inf"), 1e308, 0.00001, True])
def test_transformer_mlp_requires_a_finite_nonempty_hidden_layer(ratio: float) -> None:
    with pytest.raises(ValueError):
        ECGTransformerConfig(mlp_ratio=ratio)


@pytest.mark.parametrize("field", ["stem_kernel_size", "block_kernel_size", "stem_stride",
                                    "stage_stride"])
@pytest.mark.parametrize("value", [True, 1.5])
def test_resnet_dimensions_require_integers(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        ResNet1DConfig(**{field: value})


def test_resnet_snapshots_stage_configuration() -> None:
    widths, depths = [8, 16], [1, 1]
    config = ResNet1DConfig(stage_channels=widths, blocks_per_stage=depths)
    widths[0], depths[0] = 0, 0
    assert config.stage_channels == (8, 16)
    assert config.blocks_per_stage == (1, 1)


@pytest.mark.parametrize("field", ["stage_channels", "blocks_per_stage"])
def test_resnet_stage_items_require_integers(field: str) -> None:
    with pytest.raises(ValueError):
        ResNet1DConfig(**{field: (True,) * 4})


def test_baseline_rejects_complex_prevalence_and_targets() -> None:
    with pytest.raises(ValueError, match="real"):
        TrainingPrevalencePredictor(torch.full((5,), 0.5 + 3j))
    with pytest.raises(ValueError, match="real"):
        TrainingPrevalencePredictor.from_targets(torch.ones((2, 5), dtype=torch.complex64))
