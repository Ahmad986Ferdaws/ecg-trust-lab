"""Neural-network architectures for five-superclass PTB-XL classification."""

from ecg_trust.models.baseline import TrainingPrevalencePredictor
from ecg_trust.models.presets import (
    MATCHED_CAPACITY_PRESET,
    MatchedCapacityPreset,
    build_matched_capacity_pair,
    comparison_metadata,
    count_parameters,
)
from ecg_trust.models.resnet1d import ResNet1D, ResNet1DConfig
from ecg_trust.models.transformer import ECGTransformer, ECGTransformerConfig

__all__ = [
    "ECGTransformer",
    "ECGTransformerConfig",
    "MATCHED_CAPACITY_PRESET",
    "MatchedCapacityPreset",
    "ResNet1D",
    "ResNet1DConfig",
    "TrainingPrevalencePredictor",
    "build_matched_capacity_pair",
    "comparison_metadata",
    "count_parameters",
]
