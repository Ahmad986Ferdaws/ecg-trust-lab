"""Architecture-aware attribution and faithfulness utilities.

These helpers quantify model behavior only.  They do not convert an attribution
map into a clinical explanation or establish that a highlighted region is
causal in the physiological sense.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal, cast

import torch
from captum.attr import IntegratedGradients, Occlusion  # type: ignore[import-untyped]
from torch import Tensor, nn
from torch.nn import functional as F

from ecg_trust.constants import LEADS, SUPERCLASSES
from ecg_trust.models import ResNet1D

CANONICAL_SAMPLES = 1000
TargetSpec = int | Tensor
SignSpec = int | Tensor


def validate_ecg_batch(inputs: Tensor) -> None:
    """Validate the canonical PTB-XL model-input contract."""

    expected = f"[batch, {len(LEADS)}, {CANONICAL_SAMPLES}]"
    if inputs.ndim != 3 or tuple(inputs.shape[1:]) != (len(LEADS), CANONICAL_SAMPLES):
        raise ValueError(f"expected ECG input shaped {expected}, got {tuple(inputs.shape)}")
    if inputs.shape[0] < 1:
        raise ValueError("ECG batch cannot be empty")
    if not torch.is_floating_point(inputs):
        raise ValueError("ECG input must use a floating-point dtype")
    if not torch.isfinite(inputs).all():
        raise ValueError("ECG input must contain only finite values")


def _target_indices(targets: TargetSpec, batch_size: int, device: torch.device) -> Tensor:
    if isinstance(targets, bool):
        raise ValueError("target index cannot be boolean")
    if isinstance(targets, int):
        indices = torch.full((batch_size,), targets, dtype=torch.long, device=device)
    elif isinstance(targets, Tensor):
        raw = targets.detach().to(device=device)
        if raw.ndim == 0:
            raw = raw.expand(batch_size)
        if raw.ndim != 1 or raw.shape[0] != batch_size:
            raise ValueError(f"target tensor must have shape [{batch_size}]")
        if torch.is_floating_point(raw) and not torch.equal(raw, raw.round()):
            raise ValueError("target indices must be integers")
        indices = raw.to(dtype=torch.long)
    else:
        raise TypeError("targets must be an integer or tensor")
    if not torch.logical_and(indices >= 0, indices < len(SUPERCLASSES)).all():
        raise ValueError(f"target indices must lie in [0, {len(SUPERCLASSES) - 1}]")
    return indices


def _target_signs(signs: SignSpec | None, batch_size: int, device: torch.device) -> Tensor:
    if signs is None:
        return torch.ones(batch_size, dtype=torch.float32, device=device)
    if isinstance(signs, bool):
        raise ValueError("target sign cannot be boolean")
    if isinstance(signs, int):
        result = torch.full((batch_size,), signs, dtype=torch.float32, device=device)
    elif isinstance(signs, Tensor):
        result = signs.detach().to(device=device, dtype=torch.float32)
        if result.ndim == 0:
            result = result.expand(batch_size)
        if result.shape != (batch_size,):
            raise ValueError(f"target signs must have shape [{batch_size}]")
    else:
        raise TypeError("target signs must be an integer, tensor, or None")
    if not torch.all((result == -1.0) | (result == 1.0)):
        raise ValueError("target signs must contain only -1 or +1")
    return result


def _validate_logits(logits: object, batch_size: int) -> Tensor:
    if not isinstance(logits, Tensor):
        raise ValueError("model must return a tensor")
    if logits.shape != (batch_size, len(SUPERCLASSES)):
        raise ValueError(
            f"model must return [{batch_size}, {len(SUPERCLASSES)}] raw logits, "
            f"got {tuple(logits.shape)}"
        )
    if not torch.isfinite(logits).all():
        raise ValueError("model returned non-finite logits")
    return logits


def _validate_attributions(attributions: Tensor, batch_size: int) -> Tensor:
    if attributions.ndim == 2:
        attributions = attributions.unsqueeze(1)
    expected_tail = (CANONICAL_SAMPLES,)
    if (
        attributions.ndim != 3
        or attributions.shape[0] != batch_size
        or attributions.shape[1] not in {1, len(LEADS)}
        or tuple(attributions.shape[2:]) != expected_tail
    ):
        raise ValueError(
            "attributions must have shape [batch, 1000], [batch, 1, 1000], or [batch, 12, 1000]"
        )
    if batch_size < 1:
        raise ValueError("attribution batch cannot be empty")
    if not torch.is_floating_point(attributions) or not torch.isfinite(attributions).all():
        raise ValueError("attributions must be finite floating-point values")
    return attributions


def _baseline_like(inputs: Tensor, baseline: Tensor | None) -> Tensor:
    if baseline is None:
        return torch.zeros_like(inputs)
    if baseline.shape != inputs.shape:
        raise ValueError(f"baseline must have shape {tuple(inputs.shape)}")
    baseline = baseline.to(device=inputs.device, dtype=inputs.dtype)
    if not torch.isfinite(baseline).all():
        raise ValueError("baseline must contain only finite values")
    return baseline


@contextmanager
def _evaluating(model: nn.Module) -> Iterator[None]:
    was_training = model.training
    model.eval()
    try:
        yield
    finally:
        model.train(was_training)


def normalize_attributions(attributions: Tensor, *, epsilon: float = 1e-12) -> Tensor:
    """Scale each sample to unit maximum magnitude while preserving every sign."""

    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    validated = _validate_attributions(attributions, attributions.shape[0])
    reduction_dimensions = tuple(range(1, validated.ndim))
    scale = validated.abs().amax(dim=reduction_dimensions, keepdim=True)
    safe_scale = torch.where(scale > epsilon, scale, torch.ones_like(scale))
    normalized = validated / safe_scale
    return torch.where(scale > epsilon, normalized, torch.zeros_like(normalized))


def grad_cam_1d(
    model: ResNet1D,
    inputs: Tensor,
    targets: TargetSpec,
    *,
    normalize: bool = True,
) -> Tensor:
    """Compute signed Grad-CAM over the ResNet's final temporal feature map.

    The result is ``[batch, 1, 1000]``.  Unlike display-oriented Grad-CAM
    implementations, no ReLU is applied, so evidence opposing a target is not
    silently discarded.
    """

    validate_ecg_batch(inputs)
    target_indices = _target_indices(targets, inputs.shape[0], inputs.device)
    working_inputs = inputs.detach().clone().requires_grad_(True)
    with _evaluating(model), torch.enable_grad():
        features = model.forward_features(working_inputs)
        pooled = model.global_pool(features).squeeze(-1)
        logits = _validate_logits(
            model.classifier(model.classifier_dropout(pooled)), inputs.shape[0]
        )
        selected = logits.gather(1, target_indices.unsqueeze(1)).sum()
        gradients = torch.autograd.grad(selected, features, retain_graph=False)[0]
        channel_weights = gradients.mean(dim=-1, keepdim=True)
        temporal_map = (channel_weights * features).sum(dim=1, keepdim=True)
        temporal_map = F.interpolate(
            temporal_map,
            size=CANONICAL_SAMPLES,
            mode="linear",
            align_corners=False,
        )
    result = temporal_map.detach()
    return normalize_attributions(result) if normalize else result


def integrated_gradients(
    model: nn.Module,
    inputs: Tensor,
    targets: TargetSpec,
    *,
    baseline: Tensor | None = None,
    n_steps: int = 32,
    internal_batch_size: int | None = None,
    normalize: bool = True,
) -> Tensor:
    """Compute signed Integrated Gradients for either primary architecture."""

    attributions, _ = integrated_gradients_with_delta(
        model,
        inputs,
        targets,
        baseline=baseline,
        n_steps=n_steps,
        internal_batch_size=internal_batch_size,
        normalize=normalize,
    )
    return attributions


def integrated_gradients_with_delta(
    model: nn.Module,
    inputs: Tensor,
    targets: TargetSpec,
    *,
    baseline: Tensor | None = None,
    n_steps: int = 32,
    internal_batch_size: int | None = None,
    normalize: bool = True,
) -> tuple[Tensor, Tensor]:
    """Compute Integrated Gradients and its per-example completeness delta."""

    validate_ecg_batch(inputs)
    if n_steps < 2:
        raise ValueError("n_steps must be at least 2")
    if internal_batch_size is not None and internal_batch_size < 1:
        raise ValueError("internal_batch_size must be positive")
    target_indices = _target_indices(targets, inputs.shape[0], inputs.device)
    baselines = _baseline_like(inputs, baseline)
    working_inputs = inputs.detach().clone().requires_grad_(True)
    with _evaluating(model), torch.enable_grad():
        with torch.no_grad():
            _validate_logits(model(inputs), inputs.shape[0])
        attribution_method = IntegratedGradients(model, multiply_by_inputs=True)
        raw_result = attribution_method.attribute(
            working_inputs,
            baselines=baselines,
            target=target_indices.tolist(),
            n_steps=n_steps,
            method="gausslegendre",
            internal_batch_size=internal_batch_size,
            return_convergence_delta=True,
        )
    if not isinstance(raw_result, tuple) or len(raw_result) != 2:
        raise ValueError("Integrated Gradients did not return attributions and delta")
    result = cast(Tensor, raw_result[0]).detach()
    delta = cast(Tensor, raw_result[1]).detach()
    _validate_attributions(result, inputs.shape[0])
    if delta.shape != (inputs.shape[0],) or not torch.isfinite(delta).all():
        raise ValueError("Integrated Gradients convergence delta is invalid")
    attributed = normalize_attributions(result) if normalize else result
    return attributed, delta.cpu().to(torch.float64)


def temporal_occlusion(
    model: nn.Module,
    inputs: Tensor,
    targets: TargetSpec,
    *,
    baseline: Tensor | None = None,
    window_samples: int = 50,
    stride_samples: int = 25,
    perturbations_per_eval: int = 8,
    normalize: bool = True,
) -> Tensor:
    """Occlude temporal windows across all 12 leads and measure target change."""

    validate_ecg_batch(inputs)
    if not 1 <= window_samples <= CANONICAL_SAMPLES:
        raise ValueError(f"window_samples must lie in [1, {CANONICAL_SAMPLES}]")
    if stride_samples < 1:
        raise ValueError("stride_samples must be positive")
    if perturbations_per_eval < 1:
        raise ValueError("perturbations_per_eval must be positive")
    target_indices = _target_indices(targets, inputs.shape[0], inputs.device)
    baselines = _baseline_like(inputs, baseline)
    with _evaluating(model), torch.no_grad():
        _validate_logits(model(inputs), inputs.shape[0])
        attribution_method = Occlusion(model)
        attributed = attribution_method.attribute(
            inputs,
            sliding_window_shapes=(len(LEADS), window_samples),
            strides=(len(LEADS), stride_samples),
            baselines=baselines,
            target=target_indices.tolist(),
            perturbations_per_eval=perturbations_per_eval,
        )
    result = cast(Tensor, attributed).detach()
    _validate_attributions(result, inputs.shape[0])
    return normalize_attributions(result) if normalize else result


def _validated_fractions(fractions: Iterable[float]) -> Tensor:
    values = tuple(float(value) for value in fractions)
    if len(values) < 2:
        raise ValueError("at least two fractions are required")
    if values[0] != 0.0 or values[-1] != 1.0:
        raise ValueError("fractions must start at 0 and end at 1")
    if any(not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("fractions must lie in [0, 1]")
    if any(right <= left for left, right in zip(values, values[1:], strict=False)):
        raise ValueError("fractions must be strictly increasing")
    return torch.tensor(values, dtype=torch.float64)


def _temperature(value: float) -> float:
    parsed = float(value)
    if not 0.0 < parsed < float("inf"):
        raise ValueError("temperature must be finite and positive")
    return parsed


def _target_scores(
    model: nn.Module,
    inputs: Tensor,
    targets: Tensor,
    *,
    temperature: float,
    target_signs: Tensor,
) -> tuple[Tensor, Tensor]:
    logits = _validate_logits(model(inputs), inputs.shape[0])
    selected = logits.gather(1, targets.unsqueeze(1)).squeeze(1) * target_signs
    return selected, (selected / _temperature(temperature)).sigmoid()


def _float_list(values: Tensor) -> list[float]:
    return [float(value) for value in values.detach().cpu().tolist()]


@dataclass(frozen=True, slots=True)
class FaithfulnessCurve:
    """Tensor-valued perturbation curve with a JSON-safe aggregate summary."""

    method: str
    fractions: Tensor
    target_logits: Tensor
    target_probabilities: Tensor
    target_indices: Tensor

    def summary(self) -> dict[str, object]:
        mean_curve = self.target_probabilities.mean(dim=0)
        mean_drop = (self.target_probabilities[:, :1] - self.target_probabilities).mean(dim=0)
        mean_logits = self.target_logits.mean(dim=0)
        mean_logit_drop = (self.target_logits[:, :1] - self.target_logits).mean(dim=0)
        span = float(self.fractions[-1] - self.fractions[0])
        area = float(torch.trapezoid(mean_curve, self.fractions) / span)
        return {
            "method": self.method,
            "examples": int(self.target_probabilities.shape[0]),
            "target_indices": [int(value) for value in self.target_indices.tolist()],
            "fractions": _float_list(self.fractions),
            "mean_target_probabilities": _float_list(mean_curve),
            "mean_probability_drop": _float_list(mean_drop),
            "mean_target_logits": _float_list(mean_logits),
            "mean_logit_drop": _float_list(mean_logit_drop),
            "mean_area_under_curve": area,
            "mean_area_over_curve": float(mean_curve[0] - area),
            "mean_drop_after_full_ablation": float(mean_drop[-1]),
        }


def deletion_faithfulness_curve(
    model: nn.Module,
    inputs: Tensor,
    attributions: Tensor,
    targets: TargetSpec,
    *,
    baseline: Tensor | None = None,
    fractions: Iterable[float] = (0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0),
    temperature: float = 1.0,
    target_signs: SignSpec | None = None,
) -> FaithfulnessCurve:
    """Delete highest-magnitude time points across all leads and rescore."""

    result = temporal_faithfulness_curve(
        model,
        inputs,
        attributions,
        targets,
        baseline=baseline,
        fractions=fractions,
        operation="deletion",
        ranking="most_important",
        temperature=temperature,
        target_signs=target_signs,
    )
    return FaithfulnessCurve(
        method="temporal_deletion",
        fractions=result.fractions,
        target_logits=result.target_logits,
        target_probabilities=result.target_probabilities,
        target_indices=result.target_indices,
    )


def temporal_faithfulness_curve(
    model: nn.Module,
    inputs: Tensor,
    attributions: Tensor,
    targets: TargetSpec,
    *,
    baseline: Tensor | None = None,
    fractions: Iterable[float] = (0.0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0),
    operation: Literal["deletion", "insertion"] = "deletion",
    ranking: Literal["most_important", "least_important", "random"] = ("most_important"),
    random_seed: int | None = None,
    temperature: float = 1.0,
    target_signs: SignSpec | None = None,
) -> FaithfulnessCurve:
    """Delete or insert time points under a fixed importance/random ranking."""

    validate_ecg_batch(inputs)
    validated_attributions = _validate_attributions(attributions, inputs.shape[0]).to(
        device=inputs.device
    )
    target_indices = _target_indices(targets, inputs.shape[0], inputs.device)
    score_signs = _target_signs(target_signs, inputs.shape[0], inputs.device)
    baselines = _baseline_like(inputs, baseline)
    fraction_tensor = _validated_fractions(fractions)
    temporal_importance = validated_attributions.abs().mean(dim=1)
    if ranking == "most_important":
        order = torch.argsort(temporal_importance, dim=1, descending=True, stable=True)
    elif ranking == "least_important":
        order = torch.argsort(temporal_importance, dim=1, descending=False, stable=True)
    elif ranking == "random":
        if isinstance(random_seed, bool) or not isinstance(random_seed, int):
            raise ValueError("random ranking requires an integer random_seed")
        generator = torch.Generator(device=inputs.device)
        generator.manual_seed(random_seed)
        random_scores = torch.rand(
            temporal_importance.shape,
            device=inputs.device,
            dtype=inputs.dtype,
            generator=generator,
        )
        order = torch.argsort(random_scores, dim=1, descending=True, stable=True)
    else:  # pragma: no cover - Literal protects typed callers
        raise ValueError("unsupported temporal ranking")
    if operation not in {"deletion", "insertion"}:
        raise ValueError("operation must be deletion or insertion")
    logits: list[Tensor] = []
    probabilities: list[Tensor] = []
    with _evaluating(model), torch.inference_mode():
        for fraction in fraction_tensor:
            change_count = round(float(fraction) * CANONICAL_SAMPLES)
            mask = torch.zeros(
                (inputs.shape[0], CANONICAL_SAMPLES), dtype=torch.bool, device=inputs.device
            )
            if change_count:
                mask.scatter_(1, order[:, :change_count], True)
            perturbed = (
                torch.where(mask.unsqueeze(1), baselines, inputs)
                if operation == "deletion"
                else torch.where(mask.unsqueeze(1), inputs, baselines)
            )
            target_logits, target_probabilities = _target_scores(
                model,
                perturbed,
                target_indices,
                temperature=temperature,
                target_signs=score_signs,
            )
            logits.append(target_logits)
            probabilities.append(target_probabilities)
    return FaithfulnessCurve(
        method=f"temporal_{operation}_{ranking}",
        fractions=fraction_tensor,
        target_logits=torch.stack(logits, dim=1).detach().cpu().to(torch.float64),
        target_probabilities=(torch.stack(probabilities, dim=1).detach().cpu().to(torch.float64)),
        target_indices=target_indices.detach().cpu(),
    )


def lead_ablation_faithfulness_curve(
    model: nn.Module,
    inputs: Tensor,
    attributions: Tensor,
    targets: TargetSpec,
    *,
    baseline: Tensor | None = None,
    ranking: Literal["most_important", "least_important", "random"] = ("most_important"),
    random_seed: int | None = None,
    temperature: float = 1.0,
    target_signs: SignSpec | None = None,
) -> FaithfulnessCurve:
    """Ablate leads from most to least attributed and rescore each prefix."""

    validate_ecg_batch(inputs)
    validated_attributions = _validate_attributions(attributions, inputs.shape[0])
    if validated_attributions.shape[1] != len(LEADS):
        raise ValueError("lead ablation requires lead-specific [batch, 12, 1000] attributions")
    validated_attributions = validated_attributions.to(device=inputs.device)
    target_indices = _target_indices(targets, inputs.shape[0], inputs.device)
    score_signs = _target_signs(target_signs, inputs.shape[0], inputs.device)
    baselines = _baseline_like(inputs, baseline)
    lead_importance = validated_attributions.abs().mean(dim=2)
    if ranking == "most_important":
        ablation_order = torch.argsort(lead_importance, dim=1, descending=True, stable=True)
    elif ranking == "least_important":
        ablation_order = torch.argsort(lead_importance, dim=1, descending=False, stable=True)
    elif ranking == "random":
        if isinstance(random_seed, bool) or not isinstance(random_seed, int):
            raise ValueError("random ranking requires an integer random_seed")
        generator = torch.Generator(device=inputs.device)
        generator.manual_seed(random_seed)
        random_scores = torch.rand(
            lead_importance.shape,
            device=inputs.device,
            dtype=inputs.dtype,
            generator=generator,
        )
        ablation_order = torch.argsort(random_scores, dim=1, descending=True, stable=True)
    else:  # pragma: no cover - Literal protects typed callers
        raise ValueError("unsupported lead ranking")
    fractions = torch.arange(len(LEADS) + 1, dtype=torch.float64) / len(LEADS)
    logits: list[Tensor] = []
    probabilities: list[Tensor] = []
    with _evaluating(model), torch.inference_mode():
        for lead_count in range(len(LEADS) + 1):
            mask = torch.zeros(
                (inputs.shape[0], len(LEADS)), dtype=torch.bool, device=inputs.device
            )
            if lead_count:
                mask.scatter_(1, ablation_order[:, :lead_count], True)
            perturbed = torch.where(mask.unsqueeze(2), baselines, inputs)
            target_logits, target_probabilities = _target_scores(
                model,
                perturbed,
                target_indices,
                temperature=temperature,
                target_signs=score_signs,
            )
            logits.append(target_logits)
            probabilities.append(target_probabilities)
    return FaithfulnessCurve(
        method=("lead_ablation" if ranking == "most_important" else f"lead_ablation_{ranking}"),
        fractions=fractions,
        target_logits=torch.stack(logits, dim=1).detach().cpu().to(torch.float64),
        target_probabilities=(torch.stack(probabilities, dim=1).detach().cpu().to(torch.float64)),
        target_indices=target_indices.detach().cpu(),
    )


def _cosine_similarity(reference: Tensor, candidate: Tensor, *, epsilon: float) -> Tensor:
    if reference.shape != candidate.shape:
        raise ValueError("attribution tensors must have identical shapes")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    reference = _validate_attributions(reference, reference.shape[0]).to(torch.float64)
    candidate = _validate_attributions(candidate, candidate.shape[0]).to(
        device=reference.device, dtype=torch.float64
    )
    reference_flat = reference.flatten(start_dim=1)
    candidate_flat = candidate.flatten(start_dim=1)
    numerator = (reference_flat * candidate_flat).sum(dim=1)
    reference_norm = torch.linalg.vector_norm(reference_flat, dim=1)
    candidate_norm = torch.linalg.vector_norm(candidate_flat, dim=1)
    denominator = reference_norm * candidate_norm
    similarity = numerator / denominator.clamp_min(epsilon)
    both_zero = torch.logical_and(reference_norm <= epsilon, candidate_norm <= epsilon)
    return torch.where(both_zero, torch.ones_like(similarity), similarity).clamp(-1.0, 1.0)


@dataclass(frozen=True, slots=True)
class SimilarityResult:
    """Per-example signed cosine similarities and JSON-safe statistics."""

    method: str
    values: Tensor

    def summary(self) -> dict[str, object]:
        values = self.values.detach().cpu().to(torch.float64)
        return {
            "method": self.method,
            "examples": int(values.numel()),
            "values": _float_list(values),
            "mean": float(values.mean()),
            "standard_deviation": float(values.std(unbiased=False)),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
        }


def attribution_stability_similarity(
    reference: Tensor, candidate: Tensor, *, epsilon: float = 1e-12
) -> SimilarityResult:
    """Compare repeated/noisy attributions using signed cosine similarity."""

    values = _cosine_similarity(reference, candidate, epsilon=epsilon).detach().cpu()
    return SimilarityResult(method="signed_cosine_stability", values=values)


@dataclass(frozen=True, slots=True)
class CrossMethodSimilarity:
    """Temporal absolute-map agreement for two attribution methods."""

    cosine: Tensor
    spearman: Tensor
    cosine_valid: Tensor
    spearman_valid: Tensor

    def summary(self) -> dict[str, object]:
        cosine = self.cosine.detach().cpu().to(torch.float64)
        spearman = self.spearman.detach().cpu().to(torch.float64)
        cosine_valid = self.cosine_valid.detach().cpu().to(torch.bool)
        spearman_valid = self.spearman_valid.detach().cpu().to(torch.bool)
        return {
            "method": "absolute_temporal_cross_method_agreement",
            "examples": int(cosine.numel()),
            "cosine_values": _float_list(cosine),
            "spearman_values": _float_list(spearman),
            "cosine_valid": [bool(value) for value in cosine_valid.tolist()],
            "spearman_valid": [bool(value) for value in spearman_valid.tolist()],
            "mean_cosine": (float(cosine[cosine_valid].mean()) if cosine_valid.any() else None),
            "mean_spearman": (
                float(spearman[spearman_valid].mean()) if spearman_valid.any() else None
            ),
        }


def _temporal_absolute_map(attributions: Tensor) -> Tensor:
    validated = _validate_attributions(attributions, attributions.shape[0]).to(torch.float64)
    return validated.abs().mean(dim=1)


def _average_ranks(values: Tensor) -> Tensor:
    ranks = torch.empty_like(values, dtype=torch.float64)
    for row_index in range(values.shape[0]):
        row = values[row_index]
        order = torch.argsort(row, stable=True)
        sorted_values = row[order]
        start = 0
        while start < row.numel():
            end = start + 1
            while end < row.numel() and bool(sorted_values[end] == sorted_values[start]):
                end += 1
            ranks[row_index, order[start:end]] = (start + 1 + end) / 2.0
            start = end
    return ranks


def cross_method_temporal_similarity(
    reference: Tensor,
    candidate: Tensor,
    *,
    epsilon: float = 1e-12,
) -> CrossMethodSimilarity:
    """Compare attribution methods after absolute lead-to-time aggregation."""

    reference_map = _temporal_absolute_map(reference)
    candidate_map = _temporal_absolute_map(candidate).to(reference_map.device)
    if reference_map.shape != candidate_map.shape:
        raise ValueError("attribution methods must align by batch and time")
    centered_reference = reference_map - reference_map.mean(dim=1, keepdim=True)
    centered_candidate = candidate_map - candidate_map.mean(dim=1, keepdim=True)
    cosine_valid = (torch.linalg.vector_norm(reference_map, dim=1) > epsilon) & (
        torch.linalg.vector_norm(candidate_map, dim=1) > epsilon
    )
    cosine = torch.where(
        cosine_valid,
        _cosine_rows(reference_map, candidate_map, epsilon=epsilon),
        torch.zeros(reference_map.shape[0], dtype=torch.float64, device=reference_map.device),
    )
    reference_ranks = _average_ranks(centered_reference)
    candidate_ranks = _average_ranks(centered_candidate)
    centered_reference_ranks = reference_ranks - reference_ranks.mean(dim=1, keepdim=True)
    centered_candidate_ranks = candidate_ranks - candidate_ranks.mean(dim=1, keepdim=True)
    spearman_valid = (torch.linalg.vector_norm(centered_reference_ranks, dim=1) > epsilon) & (
        torch.linalg.vector_norm(centered_candidate_ranks, dim=1) > epsilon
    )
    spearman = torch.where(
        spearman_valid,
        _cosine_rows(
            centered_reference_ranks,
            centered_candidate_ranks,
            epsilon=epsilon,
        ),
        torch.zeros(reference_map.shape[0], dtype=torch.float64, device=reference_map.device),
    )
    return CrossMethodSimilarity(
        cosine=cosine.detach().cpu(),
        spearman=spearman.detach().cpu(),
        cosine_valid=cosine_valid.detach().cpu(),
        spearman_valid=spearman_valid.detach().cpu(),
    )


def _cosine_rows(reference: Tensor, candidate: Tensor, *, epsilon: float) -> Tensor:
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    numerator = (reference * candidate).sum(dim=1)
    reference_norm = torch.linalg.vector_norm(reference, dim=1)
    candidate_norm = torch.linalg.vector_norm(candidate, dim=1)
    denominator = reference_norm * candidate_norm
    similarity = numerator / denominator.clamp_min(epsilon)
    both_zero = torch.logical_and(reference_norm <= epsilon, candidate_norm <= epsilon)
    return torch.where(both_zero, torch.ones_like(similarity), similarity).clamp(-1.0, 1.0)


def randomized_model_copy[ModelT: nn.Module](model: ModelT, *, seed: int) -> ModelT:
    """Deep-copy a model and deterministically reinitialize its parameters."""

    randomized = copy.deepcopy(model)
    parameter_devices = {
        parameter.device.index
        for parameter in randomized.parameters()
        if parameter.device.type == "cuda" and parameter.device.index is not None
    }
    trainable_parameters = [
        parameter for parameter in randomized.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise ValueError("parameter randomization requires a model with trainable parameters")

    with torch.random.fork_rng(devices=sorted(parameter_devices)):
        torch.manual_seed(seed)
        for module in randomized.modules():
            reset_running_stats = getattr(module, "reset_running_stats", None)
            if callable(reset_running_stats):
                reset_running_stats()
        top_level_reset = getattr(randomized, "reset_parameters", None)
        if callable(top_level_reset):
            top_level_reset()
        else:
            for module in randomized.modules():
                if module is randomized or any(module.children()):
                    continue
                reset_parameters = getattr(module, "reset_parameters", None)
                if callable(reset_parameters):
                    reset_parameters()
    randomized.train(model.training)
    return randomized


def parameter_randomization_comparison(
    reference_attributions: Tensor,
    randomized_attributions: Tensor,
    *,
    seed: int,
    epsilon: float = 1e-12,
) -> SimilarityResult:
    """Compare original and randomized-model attributions by signed similarity."""

    values = (
        _cosine_similarity(reference_attributions, randomized_attributions, epsilon=epsilon)
        .detach()
        .cpu()
    )
    return SimilarityResult(method=f"parameter_randomization_seed_{seed}", values=values)


__all__ = [
    "CANONICAL_SAMPLES",
    "FaithfulnessCurve",
    "CrossMethodSimilarity",
    "SimilarityResult",
    "attribution_stability_similarity",
    "deletion_faithfulness_curve",
    "cross_method_temporal_similarity",
    "grad_cam_1d",
    "integrated_gradients",
    "integrated_gradients_with_delta",
    "lead_ablation_faithfulness_curve",
    "normalize_attributions",
    "parameter_randomization_comparison",
    "randomized_model_copy",
    "temporal_occlusion",
    "temporal_faithfulness_curve",
    "validate_ecg_batch",
]
