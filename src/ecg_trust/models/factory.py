"""Construct versioned ECG architectures without importing experiment execution.

Callers provide an already validated ModelConfig. Construction preserves the
preset, initialization order and matched-capacity drift check used in training.
"""

from __future__ import annotations

from torch import nn

from ecg_trust.experiment_config import ModelConfig
from ecg_trust.models import (
    MATCHED_CAPACITY_PRESET,
    ECGTransformer,
    ECGTransformerConfig,
    ResNet1D,
    ResNet1DConfig,
    count_parameters,
)


class ModelConstructionError(RuntimeError):
    """The constructed model differs from its versioned capacity contract."""


def build_model(config: ModelConfig) -> nn.Module:
    """Instantiate a smoke model or one side of the matched-capacity pair."""

    if config.preset == "matched_capacity":
        if config.architecture == "resnet1d":
            model: nn.Module = ResNet1D(MATCHED_CAPACITY_PRESET.resnet_config)
            expected_parameters = MATCHED_CAPACITY_PRESET.expected_resnet_parameters
        else:
            model = ECGTransformer(MATCHED_CAPACITY_PRESET.transformer_config)
            expected_parameters = MATCHED_CAPACITY_PRESET.expected_transformer_parameters
        observed_parameters = count_parameters(model)
        if observed_parameters != expected_parameters:
            raise ModelConstructionError(
                f"matched-capacity model drift: expected {expected_parameters}, "
                f"observed {observed_parameters}"
            )
        return model

    if config.architecture == "resnet1d":
        return ResNet1D(
            ResNet1DConfig(
                stage_channels=(16, 32),
                blocks_per_stage=(1, 1),
                block_dropout=0.0,
                classifier_dropout=0.0,
            )
        )
    return ECGTransformer(
        ECGTransformerConfig(
            patch_size=50,
            patch_stride=50,
            embedding_dim=64,
            depth=2,
            num_heads=4,
            mlp_ratio=2.0,
            dropout=0.0,
            attention_dropout=0.0,
        )
    )

