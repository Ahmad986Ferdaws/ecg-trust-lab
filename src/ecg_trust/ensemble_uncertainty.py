"""Decompose ensemble uncertainty for multilabel binary predictions.

The frozen abstention gate scores one probability vector with mean binary
entropy. When several frozen members are available (for example the three
seeds of each architecture), their disagreement is a second, cheap signal.
For each record and label, with member probabilities ``p_m``:

* total uncertainty ``H(mean_m p_m)``: entropy of the ensemble mean;
* aleatoric (expected) uncertainty ``mean_m H(p_m)``;
* epistemic uncertainty (mutual information) ``total - aleatoric``, which is
  non-negative because binary entropy is concave.

This is the standard information-theoretic split (Depeweg et al., ICML 2018;
Lakshminarayanan et al., NeurIPS 2017). Entropies are in bits with the same
epsilon clipping as ``post_analysis.mean_normalized_binary_entropy``.

Caveat: members that share training data and architecture make correlated
errors, so low disagreement doesn't prove a prediction is right. Disagreement
across architectures is the more independent signal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


class EnsembleUncertaintyError(ValueError):
    """Raised when member probabilities are malformed."""


@dataclass(frozen=True, slots=True)
class EnsembleUncertainty:
    """Per-record, per-label uncertainty components in bits."""

    members: int
    total: FloatArray
    aleatoric: FloatArray
    epistemic: FloatArray
    member_variance: FloatArray

    def per_record(self, component: str, *, reduction: str = "mean") -> FloatArray:
        """Reduce one component across labels with ``mean`` or ``max``."""

        values = {
            "total": self.total,
            "aleatoric": self.aleatoric,
            "epistemic": self.epistemic,
            "member_variance": self.member_variance,
        }.get(component)
        if values is None:
            raise EnsembleUncertaintyError(f"unknown uncertainty component {component!r}")
        if reduction == "mean":
            return np.asarray(values.mean(axis=1), dtype=np.float64)
        if reduction == "max":
            return np.asarray(values.max(axis=1), dtype=np.float64)
        raise EnsembleUncertaintyError("reduction must be 'mean' or 'max'")


def _binary_entropy_bits(values: FloatArray) -> FloatArray:
    epsilon = np.finfo(np.float64).eps
    clipped = np.clip(values, epsilon, 1.0 - epsilon)
    entropy = -(clipped * np.log(clipped) + (1.0 - clipped) * np.log(1.0 - clipped))
    return np.asarray(entropy / math.log(2.0), dtype=np.float64)


def _member_probabilities(values: ArrayLike) -> FloatArray:
    try:
        raw = np.asarray(values)
    except (TypeError, ValueError, OverflowError) as error:
        raise EnsembleUncertaintyError("member probabilities must be numeric") from error
    if raw.dtype == np.object_ or np.issubdtype(raw.dtype, np.bool_):
        raise EnsembleUncertaintyError("member probabilities must use a real numeric dtype")
    if np.iscomplexobj(raw) or not np.issubdtype(raw.dtype, np.number):
        raise EnsembleUncertaintyError("member probabilities must be real-valued")
    if raw.ndim != 3 or min(raw.shape) < 1:
        raise EnsembleUncertaintyError(
            "member probabilities must have shape [members, records, labels]"
        )
    if raw.shape[0] < 2:
        raise EnsembleUncertaintyError("an ensemble needs at least two members")
    probabilities = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(probabilities).all() or np.any(
        (probabilities < 0.0) | (probabilities > 1.0)
    ):
        raise EnsembleUncertaintyError("member probabilities must be finite and lie in [0, 1]")
    return probabilities


def decompose_ensemble_uncertainty(member_probabilities: ArrayLike) -> EnsembleUncertainty:
    """Split ensemble uncertainty into total, aleatoric, and epistemic parts."""

    probabilities = _member_probabilities(member_probabilities)
    mean = probabilities.mean(axis=0)
    total = _binary_entropy_bits(mean)
    aleatoric = _binary_entropy_bits(probabilities).mean(axis=0)
    epistemic = np.maximum(total - aleatoric, 0.0)
    return EnsembleUncertainty(
        members=int(probabilities.shape[0]),
        total=np.asarray(total, dtype=np.float64),
        aleatoric=np.asarray(aleatoric, dtype=np.float64),
        epistemic=np.asarray(epistemic, dtype=np.float64),
        member_variance=np.asarray(probabilities.var(axis=0), dtype=np.float64),
    )


__all__ = [
    "EnsembleUncertainty",
    "EnsembleUncertaintyError",
    "decompose_ensemble_uncertainty",
]
