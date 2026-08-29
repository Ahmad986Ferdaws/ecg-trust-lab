"""A strong residual 1D convolutional baseline for 12-lead ECGs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from torch import Tensor, nn

from ecg_trust.constants import LEADS, SUPERCLASSES


@dataclass(frozen=True, slots=True)
class ResNet1DConfig:
    """Configuration for :class:`ResNet1D`.

    The default is an approximately ResNet-18-sized temporal model.  It is
    intentionally compact enough for wide hyperparameter sweeps on a 12 GB
    GPU while retaining a large temporal receptive field.
    """

    stage_channels: tuple[int, ...] = (64, 128, 256, 512)
    blocks_per_stage: tuple[int, ...] = (2, 2, 2, 2)
    stem_kernel_size: int = 15
    block_kernel_size: int = 7
    stem_stride: int = 2
    stage_stride: int = 2
    block_dropout: float = 0.10
    classifier_dropout: float = 0.20
    zero_init_residual: bool = True

    def __post_init__(self) -> None:
        if not self.stage_channels:
            raise ValueError("stage_channels cannot be empty")
        if len(self.stage_channels) != len(self.blocks_per_stage):
            raise ValueError("stage_channels and blocks_per_stage must have equal length")
        if any(width <= 0 for width in self.stage_channels):
            raise ValueError("every stage width must be positive")
        if any(blocks <= 0 for blocks in self.blocks_per_stage):
            raise ValueError("every stage must contain at least one block")
        if self.stem_kernel_size <= 0 or self.stem_kernel_size % 2 == 0:
            raise ValueError("stem_kernel_size must be a positive odd integer")
        if self.block_kernel_size <= 0 or self.block_kernel_size % 2 == 0:
            raise ValueError("block_kernel_size must be a positive odd integer")
        if self.stem_stride <= 0 or self.stage_stride <= 0:
            raise ValueError("strides must be positive")
        for name, probability in (
            ("block_dropout", self.block_dropout),
            ("classifier_dropout", self.classifier_dropout),
        ):
            if not 0.0 <= probability < 1.0:
                raise ValueError(f"{name} must be in [0, 1)")


class ResidualBlock1D(nn.Module):
    """Two-convolution residual block with an optional projection shortcut."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        stride: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.activation = nn.SiLU(inplace=True)
        self.dropout = nn.Dropout1d(dropout) if dropout else nn.Identity()
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            padding=padding,
            bias=False,
        )
        self.norm2 = nn.BatchNorm1d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.shortcut: nn.Module = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, inputs: Tensor) -> Tensor:
        identity = self.shortcut(inputs)
        features = self.conv1(inputs)
        features = self.norm1(features)
        features = self.activation(features)
        features = self.dropout(features)
        features = self.conv2(features)
        features = self.norm2(features)
        return cast(Tensor, self.activation(features + identity))


class ResNet1D(nn.Module):
    """Residual temporal CNN producing five unnormalized diagnostic logits."""

    output_labels: tuple[str, ...] = SUPERCLASSES

    def __init__(self, config: ResNet1DConfig | None = None) -> None:
        super().__init__()
        self.config = config or ResNet1DConfig()
        first_width = self.config.stage_channels[0]
        self.stem = nn.Sequential(
            nn.Conv1d(
                len(LEADS),
                first_width,
                self.config.stem_kernel_size,
                stride=self.config.stem_stride,
                padding=self.config.stem_kernel_size // 2,
                bias=False,
            ),
            nn.BatchNorm1d(first_width),
            nn.SiLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        stages: list[nn.Module] = []
        in_channels = first_width
        for stage_index, (out_channels, block_count) in enumerate(
            zip(self.config.stage_channels, self.config.blocks_per_stage, strict=True)
        ):
            blocks: list[nn.Module] = []
            first_stride = 1 if stage_index == 0 else self.config.stage_stride
            blocks.append(
                ResidualBlock1D(
                    in_channels,
                    out_channels,
                    kernel_size=self.config.block_kernel_size,
                    stride=first_stride,
                    dropout=self.config.block_dropout,
                )
            )
            blocks.extend(
                ResidualBlock1D(
                    out_channels,
                    out_channels,
                    kernel_size=self.config.block_kernel_size,
                    stride=1,
                    dropout=self.config.block_dropout,
                )
                for _ in range(1, block_count)
            )
            stages.append(nn.Sequential(*blocks))
            in_channels = out_channels
        self.stages = nn.Sequential(*stages)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier_dropout = nn.Dropout(self.config.classifier_dropout)
        self.classifier = nn.Linear(in_channels, len(SUPERCLASSES))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Apply explicit, reproducible initialization to all learned layers."""

        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, (nn.BatchNorm1d, nn.GroupNorm)):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        if self.config.zero_init_residual:
            for module in self.modules():
                if isinstance(module, ResidualBlock1D) and module.norm2.weight is not None:
                    nn.init.zeros_(module.norm2.weight)

    @staticmethod
    def _validate_input(inputs: Tensor) -> None:
        if inputs.ndim != 3:
            raise ValueError(
                f"expected ECG input shaped [batch, {len(LEADS)}, time], got {tuple(inputs.shape)}"
            )
        if inputs.shape[1] != len(LEADS):
            raise ValueError(
                f"expected {len(LEADS)} leads in canonical order, got {inputs.shape[1]}"
            )
        if inputs.shape[2] < 16:
            raise ValueError("ECG input must contain at least 16 time samples")

    def forward_features(self, inputs: Tensor) -> Tensor:
        """Return the final temporal feature map for attribution or pooling."""

        self._validate_input(inputs)
        return cast(Tensor, self.stages(self.stem(inputs)))

    def forward_embedding(self, inputs: Tensor) -> Tensor:
        """Return the pooled pre-classifier representation for each ECG."""

        features = self.forward_features(inputs)
        return cast(Tensor, self.global_pool(features).squeeze(-1))

    def forward(self, inputs: Tensor) -> Tensor:
        embedding = self.forward_embedding(inputs)
        return cast(Tensor, self.classifier(self.classifier_dropout(embedding)))
