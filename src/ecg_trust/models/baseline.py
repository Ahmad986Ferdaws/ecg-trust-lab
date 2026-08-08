"""Non-neural reference predictors for experiment sanity checks."""

from __future__ import annotations

from typing import Self

import torch
from torch import Tensor, nn

from ecg_trust.constants import LEADS, SUPERCLASSES


class TrainingPrevalencePredictor(nn.Module):
    """Parameter-free predictor fitted only to training-label prevalence.

    The five constant logits are stored as a buffer, so they travel with a
    checkpoint but are never optimized.  Additive smoothing prevents infinite
    logits when a development subset happens to contain an all-zero or all-one
    label column.
    """

    output_labels: tuple[str, ...] = SUPERCLASSES
    constant_logits: Tensor

    def __init__(self, prevalence: Tensor) -> None:
        super().__init__()
        if prevalence.ndim != 1 or prevalence.shape[0] != len(SUPERCLASSES):
            raise ValueError(f"prevalence must have shape [{len(SUPERCLASSES)}]")
        prevalence = prevalence.detach().to(dtype=torch.float32, device="cpu")
        if not torch.isfinite(prevalence).all():
            raise ValueError("prevalence must be finite")
        if not torch.logical_and(prevalence > 0, prevalence < 1).all():
            raise ValueError("smoothed prevalence values must lie strictly between 0 and 1")
        self.register_buffer("constant_logits", torch.logit(prevalence))

    @classmethod
    def from_targets(cls, targets: Tensor, *, smoothing: float = 0.5) -> Self:
        """Fit smoothed class prevalence from a training-only binary target matrix."""

        if targets.ndim != 2 or targets.shape[1] != len(SUPERCLASSES):
            raise ValueError(f"targets must have shape [records, {len(SUPERCLASSES)}]")
        if targets.shape[0] == 0:
            raise ValueError("targets cannot be empty")
        if smoothing <= 0:
            raise ValueError("smoothing must be positive")
        numeric_targets = targets.detach().to(dtype=torch.float64, device="cpu")
        if not torch.isfinite(numeric_targets).all():
            raise ValueError("targets must be finite")
        if not torch.logical_or(numeric_targets == 0, numeric_targets == 1).all():
            raise ValueError("targets must be binary")
        positives = numeric_targets.sum(dim=0)
        prevalence = (positives + smoothing) / (targets.shape[0] + 2 * smoothing)
        return cls(prevalence.to(dtype=torch.float32))

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 3:
            raise ValueError(
                f"expected ECG input shaped [batch, {len(LEADS)}, time], got {tuple(inputs.shape)}"
            )
        if inputs.shape[1] != len(LEADS):
            raise ValueError(
                f"expected {len(LEADS)} leads in canonical order, got {inputs.shape[1]}"
            )
        logits = self.constant_logits.to(device=inputs.device)
        expanded: Tensor = logits.unsqueeze(0).expand(inputs.shape[0], -1)
        return expanded
