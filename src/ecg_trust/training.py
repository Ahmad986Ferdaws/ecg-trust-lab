"""Model-agnostic, reproducible multi-label training primitives.

The functions in this module deliberately keep architecture concerns out of the
training loop.  A model need only accept a tensor batch and return a two-
dimensional tensor of logits with the same shape as its binary targets.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer

CHECKPOINT_SCHEMA_VERSION = 1
_SHA256_PREFIX = "sha256:"
_HEX_DIGITS = frozenset("0123456789abcdef")


class TrainingValidationError(ValueError):
    """Raised when a training input violates the multi-label contract."""


class CheckpointValidationError(ValueError):
    """Raised when checkpoint content or provenance is invalid."""


@dataclass(frozen=True, slots=True)
class TrainingRuntime:
    """Resolved execution device and mixed-precision policy."""

    device: torch.device
    bf16_enabled: bool

    def __post_init__(self) -> None:
        if self.device.type not in {"cpu", "cuda"}:
            raise TrainingValidationError("only CPU and CUDA runtimes are supported")
        if self.bf16_enabled and self.device.type != "cuda":
            raise TrainingValidationError("bf16 autocast may only be enabled for CUDA")


def select_device(
    requested: str | torch.device = "auto",
    *,
    enable_bf16: bool = True,
) -> TrainingRuntime:
    """Resolve ``auto``/CPU/CUDA and enable bf16 only on capable CUDA devices."""

    if isinstance(requested, str) and requested.casefold() == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        try:
            device = torch.device(requested)
        except (TypeError, RuntimeError) as error:
            raise TrainingValidationError(f"invalid device request {requested!r}") from error
    if device.type not in {"cpu", "cuda"}:
        raise TrainingValidationError("device must be 'auto', 'cpu', or 'cuda'")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise TrainingValidationError("CUDA was requested but is not available")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise TrainingValidationError(f"CUDA device index {device.index} is not available")
    bf16_enabled = bool(
        enable_bf16 and device.type == "cuda" and torch.cuda.is_bf16_supported()
    )
    return TrainingRuntime(device=device, bf16_enabled=bf16_enabled)


def seed_everything(seed: int, *, deterministic: bool = True) -> torch.Generator:
    """Seed Python, NumPy, PyTorch, CUDA, and a returned DataLoader generator."""

    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise TrainingValidationError("seed must be an integer in [0, 2**32)")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = False
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def seed_dataloader_worker(worker_id: int) -> None:
    """Deterministically seed Python and NumPy inside a PyTorch worker."""

    if isinstance(worker_id, bool) or not isinstance(worker_id, int) or worker_id < 0:
        raise TrainingValidationError("worker_id must be a non-negative integer")
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


@dataclass(frozen=True, slots=True)
class EpochResult:
    """Sample-weighted epoch loss and aligned CPU prediction artifacts."""

    loss: float
    sample_count: int
    logits: Tensor
    probabilities: Tensor
    targets: Tensor
    max_gradient_norm: float | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.loss):
            raise TrainingValidationError("epoch loss must be finite")
        if self.sample_count < 1:
            raise TrainingValidationError("sample_count must be positive")
        expected_shape = self.targets.shape
        if self.logits.shape != expected_shape or self.probabilities.shape != expected_shape:
            raise TrainingValidationError("epoch logits, probabilities, and targets must align")
        if self.targets.ndim != 2 or self.targets.shape[0] != self.sample_count:
            raise TrainingValidationError("epoch artifacts must have shape [samples, labels]")
        for name, tensor in (
            ("logits", self.logits),
            ("probabilities", self.probabilities),
            ("targets", self.targets),
        ):
            if tensor.device.type != "cpu":
                raise TrainingValidationError(f"epoch {name} must be stored on CPU")
            if tensor.dtype != torch.float32:
                raise TrainingValidationError(f"epoch {name} must use float32")
            if not torch.isfinite(tensor).all().item():
                raise TrainingValidationError(f"epoch {name} must be finite")


def _unpack_batch(batch: object) -> tuple[Tensor, Tensor]:
    if not isinstance(batch, (tuple, list)) or len(batch) != 2:
        raise TrainingValidationError("each batch must be a (inputs, targets) pair")
    inputs, targets = batch
    if not isinstance(inputs, Tensor) or not isinstance(targets, Tensor):
        raise TrainingValidationError("batch inputs and targets must be tensors")
    if inputs.ndim < 1 or targets.ndim != 2:
        raise TrainingValidationError(
            "inputs need a batch axis and targets must be [batch, labels]"
        )
    if inputs.shape[0] != targets.shape[0] or targets.shape[0] < 1:
        raise TrainingValidationError("input and target batch sizes must match and be non-empty")
    return inputs, targets


def _prepare_pos_weight(pos_weight: Tensor | None, device: torch.device) -> Tensor | None:
    if pos_weight is None:
        return None
    if pos_weight.ndim != 1 or pos_weight.numel() < 1:
        raise TrainingValidationError("pos_weight must be a non-empty one-dimensional tensor")
    converted = pos_weight.detach().to(device=device, dtype=torch.float32)
    if not torch.isfinite(converted).all().item() or (converted < 0).any().item():
        raise TrainingValidationError("pos_weight must contain finite non-negative values")
    return converted


def _run_epoch(
    model: nn.Module,
    data: Iterable[object],
    runtime: TrainingRuntime,
    *,
    optimizer: Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    pos_weight: Tensor | None,
    max_grad_norm: float | None,
) -> EpochResult:
    training = optimizer is not None
    if max_grad_norm is not None and (
        not math.isfinite(max_grad_norm) or max_grad_norm <= 0.0
    ):
        raise TrainingValidationError("max_grad_norm must be finite and positive")
    if not training and scaler is not None:
        raise TrainingValidationError("a gradient scaler is only valid during training")
    if not training and max_grad_norm is not None:
        raise TrainingValidationError("gradient clipping is only valid during training")

    model.train(training)
    resolved_pos_weight = _prepare_pos_weight(pos_weight, runtime.device)
    total_loss = 0.0
    sample_count = 0
    collected_logits: list[Tensor] = []
    collected_targets: list[Tensor] = []
    maximum_gradient_norm: float | None = None
    scaler_enabled = scaler is not None and scaler.is_enabled()

    for batch in data:
        inputs, targets = _unpack_batch(batch)
        inputs = inputs.to(device=runtime.device, non_blocking=runtime.device.type == "cuda")
        targets = targets.to(
            device=runtime.device,
            dtype=torch.float32,
            non_blocking=runtime.device.type == "cuda",
        )
        if not torch.isfinite(targets).all().item():
            raise TrainingValidationError("targets must be finite")
        if not torch.all((targets == 0.0) | (targets == 1.0)).item():
            raise TrainingValidationError("targets must contain only 0 or 1")

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), torch.autocast(
            device_type=runtime.device.type,
            dtype=torch.bfloat16,
            enabled=runtime.bf16_enabled,
        ):
            raw_logits = model(inputs)
            if not isinstance(raw_logits, Tensor):
                raise TrainingValidationError("model must return a tensor of logits")
            logits = raw_logits
            if logits.ndim != 2 or logits.shape != targets.shape:
                raise TrainingValidationError(
                    "model logits and targets must have identical [batch, labels] shape"
                )
            if (
                resolved_pos_weight is not None
                and resolved_pos_weight.numel() != logits.shape[1]
            ):
                raise TrainingValidationError("pos_weight length must match the label count")
            loss = F.binary_cross_entropy_with_logits(
                logits,
                targets,
                pos_weight=resolved_pos_weight,
                reduction="mean",
            )
        if not torch.isfinite(loss).item():
            raise TrainingValidationError("BCEWithLogits loss became non-finite")

        if optimizer is not None:
            if scaler_enabled:
                if scaler is None:  # pragma: no cover - narrowed by scaler_enabled
                    raise AssertionError("enabled scaler unexpectedly missing")
                scaler.scale(loss).backward()  # type: ignore[no-untyped-call]
                scaler.unscale_(optimizer)
            else:
                loss.backward()  # type: ignore[no-untyped-call]
            if max_grad_norm is not None:
                gradient_norm = clip_grad_norm_(
                    model.parameters(),
                    max_norm=max_grad_norm,
                    error_if_nonfinite=True,
                )
                gradient_norm_value = float(gradient_norm.detach().cpu().item())
                maximum_gradient_norm = (
                    gradient_norm_value
                    if maximum_gradient_norm is None
                    else max(maximum_gradient_norm, gradient_norm_value)
                )
            if scaler_enabled:
                if scaler is None:  # pragma: no cover - narrowed by scaler_enabled
                    raise AssertionError("enabled scaler unexpectedly missing")
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

        batch_size = targets.shape[0]
        total_loss += float(loss.detach().cpu().item()) * batch_size
        sample_count += batch_size
        collected_logits.append(logits.detach().to(device="cpu", dtype=torch.float32))
        collected_targets.append(targets.detach().to(device="cpu", dtype=torch.float32))

    if sample_count == 0:
        raise TrainingValidationError("data loader produced no samples")
    all_logits = torch.cat(collected_logits, dim=0).contiguous()
    all_targets = torch.cat(collected_targets, dim=0).contiguous()
    probabilities = torch.sigmoid(all_logits).contiguous()
    return EpochResult(
        loss=total_loss / sample_count,
        sample_count=sample_count,
        logits=all_logits,
        probabilities=probabilities,
        targets=all_targets,
        max_gradient_norm=maximum_gradient_norm,
    )


def train_one_epoch(
    model: nn.Module,
    data: Iterable[object],
    optimizer: Optimizer,
    runtime: TrainingRuntime,
    *,
    scaler: torch.amp.GradScaler | None = None,
    pos_weight: Tensor | None = None,
    max_grad_norm: float | None = None,
) -> EpochResult:
    """Run one BCEWithLogits training epoch and collect aligned predictions."""

    return _run_epoch(
        model,
        data,
        runtime,
        optimizer=optimizer,
        scaler=scaler,
        pos_weight=pos_weight,
        max_grad_norm=max_grad_norm,
    )


def evaluate(
    model: nn.Module,
    data: Iterable[object],
    runtime: TrainingRuntime,
    *,
    pos_weight: Tensor | None = None,
) -> EpochResult:
    """Evaluate with BCEWithLogits and no gradient construction."""

    return _run_epoch(
        model,
        data,
        runtime,
        optimizer=None,
        scaler=None,
        pos_weight=pos_weight,
        max_grad_norm=None,
    )


@dataclass(slots=True)
class EarlyStopping:
    """Serializable early-stopping state for a scalar validation metric."""

    patience: int
    mode: Literal["min", "max"] = "min"
    min_delta: float = 0.0
    best_score: float | None = None
    best_epoch: int | None = None
    bad_epochs: int = 0
    stopped: bool = False

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if isinstance(self.patience, bool) or not isinstance(self.patience, int):
            raise TrainingValidationError("patience must be a positive integer")
        if self.patience < 1:
            raise TrainingValidationError("patience must be a positive integer")
        if self.mode not in {"min", "max"}:
            raise TrainingValidationError("early-stopping mode must be 'min' or 'max'")
        if not math.isfinite(self.min_delta) or self.min_delta < 0.0:
            raise TrainingValidationError("min_delta must be finite and non-negative")
        if self.best_score is not None and not math.isfinite(self.best_score):
            raise TrainingValidationError("best_score must be finite when present")
        if self.best_epoch is not None and self.best_epoch < 0:
            raise TrainingValidationError("best_epoch must be non-negative when present")
        if self.bad_epochs < 0:
            raise TrainingValidationError("bad_epochs must be non-negative")
        if (self.best_score is None) != (self.best_epoch is None):
            raise TrainingValidationError("best_score and best_epoch must be present together")
        if self.best_score is None and (self.bad_epochs != 0 or self.stopped):
            raise TrainingValidationError("uninitialized early stopping cannot have bad epochs")
        if self.stopped != (self.bad_epochs >= self.patience):
            raise TrainingValidationError("stopped must agree with bad_epochs and patience")

    def update(self, score: float, epoch: int) -> bool:
        """Record a metric and return whether it established a new best score."""

        if not math.isfinite(score):
            raise TrainingValidationError("early-stopping score must be finite")
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise TrainingValidationError("epoch must be a non-negative integer")
        if self.stopped:
            return False
        improved = self.best_score is None
        if self.best_score is not None:
            if self.mode == "min":
                improved = score < self.best_score - self.min_delta
            else:
                improved = score > self.best_score + self.min_delta
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
            self.stopped = self.bad_epochs >= self.patience
        return improved

    def state_dict(self) -> dict[str, object]:
        """Return a checkpoint-safe representation."""

        return {
            "patience": self.patience,
            "mode": self.mode,
            "min_delta": self.min_delta,
            "best_score": self.best_score,
            "best_epoch": self.best_epoch,
            "bad_epochs": self.bad_epochs,
            "stopped": self.stopped,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        """Validate and restore state, including the stopping policy."""

        required = {
            "patience",
            "mode",
            "min_delta",
            "best_score",
            "best_epoch",
            "bad_epochs",
            "stopped",
        }
        if set(state) != required:
            raise TrainingValidationError("invalid early-stopping state keys")
        patience = state["patience"]
        mode = state["mode"]
        min_delta = state["min_delta"]
        best_score = state["best_score"]
        best_epoch = state["best_epoch"]
        bad_epochs = state["bad_epochs"]
        stopped = state["stopped"]
        if isinstance(patience, bool) or not isinstance(patience, int):
            raise TrainingValidationError("early-stopping patience state must be an integer")
        if mode not in {"min", "max"}:
            raise TrainingValidationError("invalid early-stopping mode state")
        if isinstance(min_delta, bool) or not isinstance(min_delta, (int, float)):
            raise TrainingValidationError("invalid early-stopping min_delta state")
        if best_score is not None and (
            isinstance(best_score, bool) or not isinstance(best_score, (int, float))
        ):
            raise TrainingValidationError("invalid early-stopping best_score state")
        if best_epoch is not None and (
            isinstance(best_epoch, bool) or not isinstance(best_epoch, int)
        ):
            raise TrainingValidationError("invalid early-stopping best_epoch state")
        if isinstance(bad_epochs, bool) or not isinstance(bad_epochs, int):
            raise TrainingValidationError("invalid early-stopping bad_epochs state")
        if not isinstance(stopped, bool):
            raise TrainingValidationError("invalid early-stopping stopped state")
        validated = EarlyStopping(
            patience=patience,
            mode=mode,
            min_delta=float(min_delta),
            best_score=None if best_score is None else float(best_score),
            best_epoch=best_epoch,
            bad_epochs=bad_epochs,
            stopped=stopped,
        )
        self.patience = validated.patience
        self.mode = validated.mode
        self.min_delta = validated.min_delta
        self.best_score = validated.best_score
        self.best_epoch = validated.best_epoch
        self.bad_epochs = validated.bad_epochs
        self.stopped = validated.stopped


@dataclass(frozen=True, slots=True)
class CheckpointMetadata:
    """Validated non-tensor checkpoint metadata returned after restoration."""

    epoch: int
    protocol_hash: str
    manifest_hash: str
    config_hash: str
    config: dict[str, object]


def _validate_sha256(value: str, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise CheckpointValidationError(f"{field} must be a SHA-256 string")
    digest = value.removeprefix(_SHA256_PREFIX)
    if len(digest) != 64 or any(character not in _HEX_DIGITS for character in digest):
        raise CheckpointValidationError(
            f"{field} must contain 64 lowercase hexadecimal SHA-256 characters"
        )
    return value


def _canonical_config(config: Mapping[str, object]) -> tuple[dict[str, object], str]:
    if not all(isinstance(key, str) for key in config):
        raise CheckpointValidationError("config keys must be strings")
    try:
        serialized = json.dumps(
            config,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        decoded: object = json.loads(serialized)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise CheckpointValidationError("config must be JSON-serializable and finite") from error
    if not isinstance(decoded, dict):
        raise CheckpointValidationError("config must be a mapping")
    canonical = cast(dict[str, object], decoded)
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return canonical, f"{_SHA256_PREFIX}{digest}"


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    scaler: torch.amp.GradScaler | None,
    epoch: int,
    protocol_hash: str,
    config: Mapping[str, object],
    manifest_hash: str,
    early_stopping: EarlyStopping | None = None,
) -> CheckpointMetadata:
    """Atomically save complete training state and scientific provenance."""

    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise CheckpointValidationError("epoch must be a non-negative integer")
    protocol_hash = _validate_sha256(protocol_hash, "protocol_hash")
    manifest_hash = _validate_sha256(manifest_hash, "manifest_hash")
    canonical_config, config_hash = _canonical_config(config)
    payload: dict[str, object] = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "epoch": epoch,
        "protocol_hash": protocol_hash,
        "manifest_hash": manifest_hash,
        "config": canonical_config,
        "config_hash": config_hash,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": None if scaler is None else scaler.state_dict(),
        "early_stopping_state_dict": (
            None if early_stopping is None else early_stopping.state_dict()
        ),
    }

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return CheckpointMetadata(
        epoch=epoch,
        protocol_hash=protocol_hash,
        manifest_hash=manifest_hash,
        config_hash=config_hash,
        config=canonical_config,
    )


def _checkpoint_mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise CheckpointValidationError(f"checkpoint {field} must be a string-keyed mapping")
    return cast(Mapping[str, Any], value)


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    scaler: torch.amp.GradScaler | None,
    expected_protocol_hash: str,
    expected_config: Mapping[str, object],
    expected_manifest_hash: str,
    early_stopping: EarlyStopping | None = None,
    map_location: str | torch.device = "cpu",
    strict_model: bool = True,
) -> CheckpointMetadata:
    """Verify provenance before restoring model, optimizer, scaler, and stopper."""

    expected_protocol_hash = _validate_sha256(expected_protocol_hash, "expected_protocol_hash")
    expected_manifest_hash = _validate_sha256(expected_manifest_hash, "expected_manifest_hash")
    expected_canonical_config, expected_config_hash = _canonical_config(expected_config)
    try:
        loaded: object = torch.load(path, map_location=map_location, weights_only=True)
    except Exception as error:
        raise CheckpointValidationError(f"could not load checkpoint {path!s}: {error}") from error
    checkpoint = _checkpoint_mapping(loaded, "root")
    required = {
        "schema_version",
        "epoch",
        "protocol_hash",
        "manifest_hash",
        "config",
        "config_hash",
        "model_state_dict",
        "optimizer_state_dict",
        "scaler_state_dict",
        "early_stopping_state_dict",
    }
    if set(checkpoint) != required:
        raise CheckpointValidationError("checkpoint keys do not match the supported schema")
    schema_version = checkpoint["schema_version"]
    if schema_version != CHECKPOINT_SCHEMA_VERSION:
        raise CheckpointValidationError(
            f"unsupported checkpoint schema_version {schema_version!r}"
        )
    epoch = checkpoint["epoch"]
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise CheckpointValidationError("checkpoint epoch must be a non-negative integer")
    stored_protocol_hash = checkpoint["protocol_hash"]
    stored_manifest_hash = checkpoint["manifest_hash"]
    stored_config_hash = checkpoint["config_hash"]
    if stored_protocol_hash != expected_protocol_hash:
        raise CheckpointValidationError("checkpoint protocol hash does not match")
    if stored_manifest_hash != expected_manifest_hash:
        raise CheckpointValidationError("checkpoint manifest hash does not match")
    if stored_config_hash != expected_config_hash:
        raise CheckpointValidationError("checkpoint config hash does not match")
    stored_config = _checkpoint_mapping(checkpoint["config"], "config")
    stored_canonical_config, recomputed_hash = _canonical_config(
        cast(Mapping[str, object], stored_config)
    )
    if (
        recomputed_hash != stored_config_hash
        or stored_canonical_config != expected_canonical_config
    ):
        raise CheckpointValidationError("checkpoint config content failed verification")

    model_state = _checkpoint_mapping(checkpoint["model_state_dict"], "model_state_dict")
    optimizer_state = _checkpoint_mapping(
        checkpoint["optimizer_state_dict"], "optimizer_state_dict"
    )
    scaler_state = checkpoint["scaler_state_dict"]
    stopper_state = checkpoint["early_stopping_state_dict"]
    validated_scaler_state: dict[str, Any] | None = None
    if scaler_state is not None:
        validated_scaler_state = dict(
            _checkpoint_mapping(scaler_state, "scaler_state_dict")
        )
        if scaler is None and validated_scaler_state:
            raise CheckpointValidationError(
                "checkpoint contains scaler state but no scaler was supplied"
            )
    elif scaler is not None and scaler.is_enabled():
        raise CheckpointValidationError("checkpoint has no state for the enabled scaler")

    validated_stopper: EarlyStopping | None = None
    if stopper_state is not None:
        if early_stopping is None:
            raise CheckpointValidationError(
                "checkpoint contains early-stopping state but no stopper was supplied"
            )
        validated_stopper = EarlyStopping(patience=1)
        validated_stopper.load_state_dict(
            cast(Mapping[str, object], _checkpoint_mapping(stopper_state, "early_stopping"))
        )

    model.load_state_dict(cast(Any, model_state), strict=strict_model)
    optimizer.load_state_dict(cast(dict[str, Any], optimizer_state))
    if scaler is not None and validated_scaler_state is not None:
        scaler.load_state_dict(validated_scaler_state)
    if early_stopping is not None and validated_stopper is not None:
        early_stopping.load_state_dict(validated_stopper.state_dict())

    return CheckpointMetadata(
        epoch=epoch,
        protocol_hash=expected_protocol_hash,
        manifest_hash=expected_manifest_hash,
        config_hash=expected_config_hash,
        config=stored_canonical_config,
    )
