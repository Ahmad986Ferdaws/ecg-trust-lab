"""Auditable capacity-matched presets for architecture comparisons."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from torch import nn

from ecg_trust.models.resnet1d import ResNet1D, ResNet1DConfig
from ecg_trust.models.transformer import ECGTransformer, ECGTransformerConfig


def count_parameters(model: nn.Module, *, trainable_only: bool = True) -> int:
    """Count scalar model parameters, optionally including frozen parameters."""

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad or not trainable_only
    )


@dataclass(frozen=True, slots=True)
class MatchedCapacityPreset:
    """Versioned configuration and expected counts for a fair model pair."""

    name: str
    resnet_config: ResNet1DConfig
    transformer_config: ECGTransformerConfig
    expected_resnet_parameters: int
    expected_transformer_parameters: int
    tolerance_fraction: float = 0.15

    @property
    def transformer_to_resnet_ratio(self) -> float:
        return self.expected_transformer_parameters / self.expected_resnet_parameters

    @property
    def relative_gap_fraction(self) -> float:
        return abs(self.transformer_to_resnet_ratio - 1.0)

    def metadata(self) -> dict[str, object]:
        """Return JSON-serializable metadata suitable for an experiment record."""

        return {
            "name": self.name,
            "tolerance_fraction": self.tolerance_fraction,
            "expected_parameter_counts": {
                "resnet1d": self.expected_resnet_parameters,
                "ecg_transformer": self.expected_transformer_parameters,
            },
            "transformer_to_resnet_ratio": self.transformer_to_resnet_ratio,
            "relative_gap_fraction": self.relative_gap_fraction,
            "within_tolerance": self.relative_gap_fraction <= self.tolerance_fraction,
            "configs": {
                "resnet1d": asdict(self.resnet_config),
                "ecg_transformer": asdict(self.transformer_config),
            },
        }


MATCHED_CAPACITY_PRESET = MatchedCapacityPreset(
    name="ptbxl_100hz_matched_capacity_v1",
    resnet_config=ResNet1DConfig(
        stage_channels=(64, 128, 256, 512),
        blocks_per_stage=(2, 2, 2, 2),
        stem_kernel_size=15,
        block_kernel_size=7,
        stem_stride=2,
        stage_stride=2,
        block_dropout=0.10,
        classifier_dropout=0.20,
        zero_init_residual=True,
    ),
    transformer_config=ECGTransformerConfig(
        signal_length=1000,
        patch_size=20,
        patch_stride=20,
        embedding_dim=320,
        depth=7,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.10,
        attention_dropout=0.05,
        lead_stem_kernel_size=7,
    ),
    expected_resnet_parameters=8_739_973,
    expected_transformer_parameters=8_726_833,
)


def build_matched_capacity_pair(
    preset: MatchedCapacityPreset = MATCHED_CAPACITY_PRESET,
) -> tuple[ResNet1D, ECGTransformer]:
    """Instantiate a model pair and fail if implementation drift breaks matching."""

    resnet = ResNet1D(preset.resnet_config)
    transformer = ECGTransformer(preset.transformer_config)
    observed_resnet = count_parameters(resnet)
    observed_transformer = count_parameters(transformer)
    if observed_resnet != preset.expected_resnet_parameters:
        raise RuntimeError(
            "ResNet preset parameter count drifted: "
            f"expected {preset.expected_resnet_parameters}, observed {observed_resnet}"
        )
    if observed_transformer != preset.expected_transformer_parameters:
        raise RuntimeError(
            "transformer preset parameter count drifted: "
            f"expected {preset.expected_transformer_parameters}, observed {observed_transformer}"
        )
    if preset.relative_gap_fraction > preset.tolerance_fraction:
        raise RuntimeError(
            f"matched-capacity gap {preset.relative_gap_fraction:.3%} exceeds "
            f"tolerance {preset.tolerance_fraction:.3%}"
        )
    return resnet, transformer


def comparison_metadata(
    resnet: ResNet1D,
    transformer: ECGTransformer,
    *,
    preset: MatchedCapacityPreset = MATCHED_CAPACITY_PRESET,
) -> dict[str, object]:
    """Record observed counts alongside the immutable preset definition."""

    resnet_parameters = count_parameters(resnet)
    transformer_parameters = count_parameters(transformer)
    ratio = transformer_parameters / resnet_parameters
    relative_gap = abs(ratio - 1.0)
    return {
        "preset": preset.metadata(),
        "observed_parameter_counts": {
            "resnet1d": resnet_parameters,
            "ecg_transformer": transformer_parameters,
        },
        "observed_transformer_to_resnet_ratio": ratio,
        "observed_relative_gap_fraction": relative_gap,
        "observed_within_tolerance": relative_gap <= preset.tolerance_fraction,
    }
