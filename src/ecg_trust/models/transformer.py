"""Patch-token transformer for multi-label 12-lead ECG classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ecg_trust.constants import LEADS, SUPERCLASSES


@dataclass(frozen=True, slots=True)
class ECGTransformerConfig:
    """Configuration for :class:`ECGTransformer`.

    At 100 Hz, the default 20-sample patch represents 200 ms and yields 50
    temporal tokens for a ten-second PTB-XL signal.
    """

    signal_length: int = 1000
    patch_size: int = 20
    patch_stride: int = 20
    embedding_dim: int = 256
    depth: int = 6
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.10
    attention_dropout: float = 0.05
    lead_stem_kernel_size: int | None = 7

    def __post_init__(self) -> None:
        if self.signal_length <= 0:
            raise ValueError("signal_length must be positive")
        if self.patch_size <= 0 or self.patch_size > self.signal_length:
            raise ValueError("patch_size must be positive and no larger than signal_length")
        if self.patch_stride <= 0:
            raise ValueError("patch_stride must be positive")
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if self.depth <= 0:
            raise ValueError("depth must be positive")
        if self.num_heads <= 0 or self.embedding_dim % self.num_heads != 0:
            raise ValueError("num_heads must evenly divide embedding_dim")
        if self.mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be positive")
        for name, probability in (
            ("dropout", self.dropout),
            ("attention_dropout", self.attention_dropout),
        ):
            if not 0.0 <= probability < 1.0:
                raise ValueError(f"{name} must be in [0, 1)")
        if self.lead_stem_kernel_size is not None and (
            self.lead_stem_kernel_size <= 0 or self.lead_stem_kernel_size % 2 == 0
        ):
            raise ValueError("lead_stem_kernel_size must be None or a positive odd integer")


class TransformerBlock(nn.Module):
    """Pre-normalized self-attention block with independent dropout controls."""

    def __init__(self, config: ECGTransformerConfig) -> None:
        super().__init__()
        hidden_dim = round(config.embedding_dim * config.mlp_ratio)
        self.attention_norm = nn.LayerNorm(config.embedding_dim)
        self.attention = nn.MultiheadAttention(
            config.embedding_dim,
            config.num_heads,
            dropout=config.attention_dropout,
            batch_first=True,
        )
        self.attention_output_dropout = nn.Dropout(config.dropout)
        self.mlp_norm = nn.LayerNorm(config.embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(config.embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_dim, config.embedding_dim),
            nn.Dropout(config.dropout),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        normalized = self.attention_norm(inputs)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            need_weights=False,
        )
        features = inputs + self.attention_output_dropout(attended)
        mlp_features = cast(Tensor, self.mlp(self.mlp_norm(features)))
        return cast(Tensor, features + mlp_features)


class ECGTransformer(nn.Module):
    """Temporal patch transformer producing five unnormalized diagnostic logits."""

    output_labels: tuple[str, ...] = SUPERCLASSES

    def __init__(self, config: ECGTransformerConfig | None = None) -> None:
        super().__init__()
        self.config = config or ECGTransformerConfig()
        stem_kernel = self.config.lead_stem_kernel_size
        if stem_kernel is None:
            self.lead_stem: nn.Module = nn.Identity()
        else:
            self.lead_stem = nn.Sequential(
                nn.Conv1d(
                    len(LEADS),
                    len(LEADS),
                    stem_kernel,
                    padding=stem_kernel // 2,
                    groups=len(LEADS),
                    bias=False,
                ),
                nn.BatchNorm1d(len(LEADS)),
                nn.GELU(),
            )
        self.patch_projection = nn.Conv1d(
            len(LEADS),
            self.config.embedding_dim,
            kernel_size=self.config.patch_size,
            stride=self.config.patch_stride,
        )
        maximum_patches = self._patch_count(self.config.signal_length)
        self.class_token = nn.Parameter(torch.empty(1, 1, self.config.embedding_dim))
        self.position_embedding = nn.Parameter(
            torch.empty(1, maximum_patches + 1, self.config.embedding_dim)
        )
        self.embedding_dropout = nn.Dropout(self.config.dropout)
        self.blocks = nn.ModuleList(
            TransformerBlock(self.config) for _ in range(self.config.depth)
        )
        self.final_norm = nn.LayerNorm(self.config.embedding_dim)
        self.classifier = nn.Linear(self.config.embedding_dim, len(SUPERCLASSES))
        self.reset_parameters()

    def _patch_count(self, signal_length: int) -> int:
        if signal_length < self.config.patch_size:
            raise ValueError(
                f"signal length {signal_length} is shorter than patch size {self.config.patch_size}"
            )
        return 1 + (signal_length - self.config.patch_size) // self.config.patch_stride

    def reset_parameters(self) -> None:
        """Apply explicit initialization to patch, attention, and output layers."""

        nn.init.trunc_normal_(self.class_token, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        for module in self.modules():
            if isinstance(module, nn.MultiheadAttention):
                nn.init.xavier_uniform_(module.in_proj_weight)
                if module.in_proj_bias is not None:
                    nn.init.zeros_(module.in_proj_bias)
                if module.bias_k is not None:
                    nn.init.zeros_(module.bias_k)
                if module.bias_v is not None:
                    nn.init.zeros_(module.bias_v)

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

    def _position_encoding(self, patch_count: int) -> Tensor:
        if patch_count == self.position_embedding.shape[1] - 1:
            return self.position_embedding
        class_position = self.position_embedding[:, :1]
        patch_positions = self.position_embedding[:, 1:].transpose(1, 2)
        resized = F.interpolate(
            patch_positions,
            size=patch_count,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)
        return torch.cat((class_position, resized), dim=1)

    def forward_tokens(self, inputs: Tensor) -> Tensor:
        """Return normalized class and temporal tokens for analysis."""

        self._validate_input(inputs)
        self._patch_count(inputs.shape[2])
        features = self.lead_stem(inputs)
        patch_tokens = self.patch_projection(features).transpose(1, 2)
        class_token = self.class_token.expand(inputs.shape[0], -1, -1)
        tokens = torch.cat((class_token, patch_tokens), dim=1)
        tokens = self.embedding_dropout(tokens + self._position_encoding(patch_tokens.shape[1]))
        for block in self.blocks:
            tokens = block(tokens)
        return cast(Tensor, self.final_norm(tokens))

    def forward_features(self, inputs: Tensor) -> Tensor:
        """Return the final class-token embedding before the output head."""

        return self.forward_tokens(inputs)[:, 0]

    def forward(self, inputs: Tensor) -> Tensor:
        return cast(Tensor, self.classifier(self.forward_features(inputs)))
