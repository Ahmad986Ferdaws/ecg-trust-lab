"""Fail-closed PyTorch adapter for external ECG representation encoders."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, cast, runtime_checkable

import numpy as np
import torch
from torch import Tensor, nn

from ecg_trust.constants import LEADS
from ecg_trust.foundation.contracts import (
    CANONICAL_DTYPE,
    CANONICAL_INPUT_SAMPLES,
    CANONICAL_INPUT_UNIT,
    EXTERNAL_ONLY_LIMIT,
    RESEARCH_USE_LIMIT,
    EvaluationMode,
    FoundationAdapterError,
    FoundationModelSpec,
    TrainabilityError,
    TrainabilityPolicy,
    canonical_sha256,
)

EMBEDDING_ARTIFACT_SCHEMA_VERSION = 1
EMBEDDING_ARTIFACT_TYPE = "ecg_trust.private_foundation_embeddings"


@runtime_checkable
class FoundationEncoderProtocol(Protocol):
    """Minimal representation-forward protocol used by the adapter."""

    def __call__(self, waveforms: Tensor) -> Tensor:
        """Return one fixed-size embedding per ECG."""


@dataclass(frozen=True, slots=True, init=False)
class PrivateEmbeddingArtifact:
    """Private embeddings plus identifier-free, content-bound metadata."""

    _embeddings: Tensor
    model_spec_sha256: str
    evaluation_mode: EvaluationMode
    input_batch_sha256: str
    embedding_tensor_sha256: str
    batch_size: int
    embedding_dimension: int
    encoder_state_sha256: str
    artifact_sha256: str

    @classmethod
    def _create(
        cls,
        *,
        embeddings: Tensor,
        model_spec_sha256: str,
        evaluation_mode: EvaluationMode,
        input_batch_sha256: str,
        encoder_state_sha256: str,
    ) -> PrivateEmbeddingArtifact:
        private_embeddings = embeddings.detach().cpu().to(dtype=torch.float32).contiguous().clone()
        embedding_hash = float32_tensor_sha256(private_embeddings, domain="embedding")
        instance = object.__new__(cls)
        object.__setattr__(instance, "_embeddings", private_embeddings)
        object.__setattr__(instance, "model_spec_sha256", model_spec_sha256)
        object.__setattr__(instance, "evaluation_mode", evaluation_mode)
        object.__setattr__(instance, "input_batch_sha256", input_batch_sha256)
        object.__setattr__(instance, "embedding_tensor_sha256", embedding_hash)
        object.__setattr__(instance, "batch_size", private_embeddings.shape[0])
        object.__setattr__(instance, "embedding_dimension", private_embeddings.shape[1])
        object.__setattr__(instance, "encoder_state_sha256", encoder_state_sha256)
        object.__setattr__(instance, "artifact_sha256", canonical_sha256(instance._body()))
        return instance

    @property
    def embeddings(self) -> Tensor:
        """Return a defensive copy of the private derived tensor."""

        return self._embeddings.clone()

    def _body(self) -> dict[str, object]:
        return {
            "schema_version": EMBEDDING_ARTIFACT_SCHEMA_VERSION,
            "artifact_type": EMBEDDING_ARTIFACT_TYPE,
            "model_spec_sha256": self.model_spec_sha256,
            "evaluation_mode": self.evaluation_mode.value,
            "input_batch_sha256": self.input_batch_sha256,
            "embedding_tensor_sha256": self.embedding_tensor_sha256,
            "encoder_state_sha256": self.encoder_state_sha256,
            "input_shape": [self.batch_size, len(LEADS), CANONICAL_INPUT_SAMPLES],
            "embedding_shape": [self.batch_size, self.embedding_dimension],
            "dtype": CANONICAL_DTYPE,
            "input_unit": CANONICAL_INPUT_UNIT,
            "privacy_classification": "private_derived_tensor_identifier_free_metadata",
            "scope_limit": EXTERNAL_ONLY_LIMIT,
            "research_use_limit": RESEARCH_USE_LIMIT,
        }

    def to_private_metadata(self) -> dict[str, object]:
        """Return content lineage without rows, identifiers, or filesystem paths."""

        return {**self._body(), "artifact_sha256": self.artifact_sha256}


def verify_trainable_parameters(model: nn.Module, policy: TrainabilityPolicy) -> None:
    """Require the actual trainable set to exactly equal the declared allow-list."""

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if not isinstance(policy, TrainabilityPolicy):
        raise TypeError("policy must be a TrainabilityPolicy")
    named = dict(model.named_parameters())
    allowed = set(policy.allowed_trainable_parameter_names)
    missing = sorted(allowed - set(named))
    actual = {name for name, parameter in named.items() if parameter.requires_grad}
    unexpected = sorted(actual - allowed)
    declared_but_frozen = sorted(allowed - actual)
    if missing or unexpected or declared_but_frozen:
        raise TrainabilityError(
            "trainable parameter set does not exactly match policy: "
            f"missing={missing}, unexpected={unexpected}, "
            f"declared_but_frozen={declared_but_frozen}"
        )


class FoundationEncoderAdapter:
    """Run a disclosed external encoder under deterministic inference controls."""

    def __init__(
        self,
        *,
        spec: FoundationModelSpec,
        encoder: nn.Module,
        trainability_policy: TrainabilityPolicy,
    ) -> None:
        if not isinstance(spec, FoundationModelSpec):
            raise TypeError("spec must be a FoundationModelSpec")
        if not isinstance(encoder, nn.Module):
            raise TypeError("encoder must be a torch.nn.Module")
        if not isinstance(trainability_policy, TrainabilityPolicy):
            raise TypeError("trainability_policy must be a TrainabilityPolicy")
        self._spec = spec
        self._encoder = encoder
        self._policy = trainability_policy

    def extract(self, batch: Tensor) -> PrivateEmbeddingArtifact:
        """Extract canonical finite embeddings without exposing row identifiers."""

        _validate_batch(batch)
        verify_trainable_parameters(self._encoder, self._policy)
        input_hash = float32_tensor_sha256(batch, domain="canonical_ecg_batch")
        adapter_input = batch.detach().clone()
        adapter_input_hash = float32_tensor_sha256(
            adapter_input,
            domain="canonical_ecg_batch",
        )
        training_modes = tuple((module, module.training) for module in self._encoder.modules())
        frozen = self._policy.mode is EvaluationMode.FROZEN_ENCODER
        state_snapshot = _clone_state_dict(self._encoder) if frozen else None
        state_before = model_state_sha256(self._encoder)
        forward_error: Exception | None = None
        verification_error: Exception | None = None
        state_after: str | None = None
        input_after: str | None = None
        output: object = None
        try:
            self._encoder.eval()
            with torch.inference_mode():
                output = self._encoder(adapter_input)
        except Exception as exc:  # Model errors are normalized after cleanup.
            forward_error = exc
        finally:
            try:
                state_after = model_state_sha256(self._encoder)
                input_after = float32_tensor_sha256(
                    adapter_input,
                    domain="canonical_ecg_batch",
                )
            except Exception as exc:
                verification_error = exc
            finally:
                for module, was_training in training_modes:
                    module.training = was_training

        if verification_error is not None:
            if frozen and state_snapshot is not None:
                try:
                    self._encoder.load_state_dict(state_snapshot, strict=True)
                except Exception as exc:
                    raise FoundationAdapterError(
                        "frozen encoder state became unverifiable and restoration failed"
                    ) from exc
            raise FoundationAdapterError("encoder state or input became unverifiable") from (
                verification_error
            )
        if state_after is None or input_after is None:  # pragma: no cover - defensive
            raise FoundationAdapterError("encoder verification did not complete")

        if input_after != adapter_input_hash:
            raise FoundationAdapterError("encoder mutated its canonical input batch")
        if frozen and state_after != state_before:
            if state_snapshot is None:  # pragma: no cover - defensive type narrowing
                raise FoundationAdapterError("frozen encoder state snapshot is unavailable")
            try:
                self._encoder.load_state_dict(state_snapshot, strict=True)
            except Exception as exc:
                raise FoundationAdapterError(
                    "frozen encoder mutated state and restoration failed"
                ) from exc
            raise FoundationAdapterError("frozen encoder mutated parameters or buffers")
        if forward_error is not None:
            raise FoundationAdapterError("external encoder forward pass failed") from forward_error
        if not isinstance(output, Tensor):
            raise FoundationAdapterError("encoder must return a torch.Tensor embedding")
        _validate_embeddings(
            output,
            batch_size=batch.shape[0],
            embedding_dimension=self._spec.embedding_dimension,
            expected_device=batch.device,
        )
        if float32_tensor_sha256(batch, domain="canonical_ecg_batch") != input_hash:
            raise FoundationAdapterError("caller input changed during representation extraction")
        return PrivateEmbeddingArtifact._create(
            embeddings=output,
            model_spec_sha256=self._spec.spec_sha256,
            evaluation_mode=self._policy.mode,
            input_batch_sha256=input_hash,
            encoder_state_sha256=state_before,
        )


def _validate_batch(batch: Tensor) -> None:
    if not isinstance(batch, Tensor):
        raise TypeError("batch must be a torch.Tensor")
    if batch.dtype is not torch.float32:
        raise FoundationAdapterError("batch must use canonical float32 physical-mV values")
    if batch.ndim != 3 or batch.shape[1:] != (len(LEADS), CANONICAL_INPUT_SAMPLES):
        raise FoundationAdapterError(
            f"batch must have shape [batch, {len(LEADS)}, {CANONICAL_INPUT_SAMPLES}]"
        )
    if not 1 <= batch.shape[0] <= 1_000_000:
        raise FoundationAdapterError("batch size must be in [1, 1000000]")
    if not batch.is_contiguous():
        raise FoundationAdapterError("batch must be contiguous")
    if batch.requires_grad:
        raise FoundationAdapterError("batch must be a detached representation input")
    if not torch.isfinite(batch).all().item():
        raise FoundationAdapterError("batch must contain finite physical-mV values")


def _validate_embeddings(
    embeddings: Tensor,
    *,
    batch_size: int,
    embedding_dimension: int,
    expected_device: torch.device,
) -> None:
    if embeddings.shape != (batch_size, embedding_dimension):
        raise FoundationAdapterError(
            f"embedding must have shape [{batch_size}, {embedding_dimension}]"
        )
    if embeddings.dtype is not torch.float32:
        raise FoundationAdapterError("embedding dtype must be float32")
    if embeddings.device != expected_device:
        raise FoundationAdapterError("embedding must remain on the input device")
    if embeddings.requires_grad:
        raise FoundationAdapterError("embedding must be produced under inference mode")
    if not torch.isfinite(embeddings).all().item():
        raise FoundationAdapterError("embedding contains non-finite values")


def float32_tensor_sha256(tensor: Tensor, *, domain: str) -> str:
    """Canonical content hash over a finite float32 tensor and its exact shape."""

    if not isinstance(tensor, Tensor) or tensor.dtype is not torch.float32:
        raise FoundationAdapterError("tensor hash input must be float32")
    if not torch.isfinite(tensor).all().item():
        raise FoundationAdapterError("tensor hash input must be finite")
    array = tensor.detach().cpu().contiguous().numpy().astype(np.dtype("<f4"), copy=False)
    shape = ",".join(str(value) for value in tensor.shape)
    header = f"ecg_trust.{domain}.v1|float32|shape={shape}|little-endian|".encode()
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(array.tobytes(order="C"))
    return "sha256:" + digest.hexdigest()


def model_state_sha256(model: nn.Module) -> str:
    """Hash all named state tensors without serializing files or paths."""

    state = cast(Mapping[str, Tensor], model.state_dict())
    digest = hashlib.sha256(b"ecg_trust.foundation_encoder_state.v1|")
    for name in sorted(state):
        tensor = state[name]
        if not isinstance(tensor, Tensor) or tensor.layout is not torch.strided:
            raise FoundationAdapterError("encoder state must contain dense tensors")
        detached = tensor.detach().cpu().contiguous()
        header = json.dumps(
            {
                "name": name,
                "shape": list(detached.shape),
                "dtype": str(detached.dtype),
            },
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        raw = detached.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return "sha256:" + digest.hexdigest()


def _clone_state_dict(model: nn.Module) -> dict[str, Tensor]:
    return {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
