"""UI-independent, provenance-checked inference backend for the research demo."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Literal, Self, cast

import numpy as np
import numpy.typing as npt
import torch
import wfdb  # type: ignore[import-untyped]
from torch import Tensor, nn

from ecg_trust.constants import LEADS, PTBXL_VERSION, SUPERCLASSES, TARGET_COLUMNS
from ecg_trust.data.dataset import (
    NormalizationStats,
    NormalizationValidationError,
)
from ecg_trust.data.manifest import sha256_file
from ecg_trust.experiment_config import ExperimentConfigError, ModelConfig
from ecg_trust.experiment_runner import DevelopmentRunnerError, build_experiment_model
from ecg_trust.models import ECGTransformer, ResNet1D, count_parameters
from ecg_trust.protocol import CALIBRATION_FOLDS, TRAIN_FOLDS, ExperimentProtocol
from ecg_trust.training import CHECKPOINT_SCHEMA_VERSION

DEMO_POLICY_SCHEMA_VERSION = 1
EXPECTED_FREQUENCY_HZ = 100.0
EXPECTED_SAMPLES = 1000
PHYSICAL_UNITS = "mV"
RESEARCH_ONLY_NOTICE = (
    "Research-only prototype. It is not a medical device and must not be used "
    "for diagnosis, treatment, triage, or emergency decisions."
)
LIMITATIONS: tuple[str, ...] = (
    "Performance is limited by the PTB-XL population, labels, and 100 Hz preprocessing.",
    "A confident or accepted output can still be wrong; abstention is not a safety guarantee.",
    "Attributions show model sensitivity, not physiological causality or clinician reasoning.",
    "Only complete ten-second, 100 Hz, 12-lead signals in physical millivolts are supported.",
)

AttributionMethod = Literal["integrated_gradients", "grad_cam"]
Decision = Literal["accept", "abstain"]
SignalArray = Tensor | npt.NDArray[np.generic]
_HEX_DIGITS = frozenset("0123456789abcdef")


class DemoArtifactError(ValueError):
    """Raised when frozen inference artifacts are malformed or mismatched."""


class DemoInputError(ValueError):
    """Raised when an ECG does not satisfy the demo input contract."""


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise DemoArtifactError(f"{context} must be a string-keyed object")
    return cast(Mapping[str, object], value)


def _exact_keys(value: Mapping[str, object], required: set[str], context: str) -> None:
    missing = sorted(required.difference(value))
    unexpected = sorted(set(value).difference(required))
    if missing or unexpected:
        raise DemoArtifactError(
            f"{context} keys do not match the supported schema; "
            f"missing={missing}, unexpected={unexpected}"
        )


def _string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DemoArtifactError(f"{field} must be a non-empty string")
    return value


def _number(value: object, field: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DemoArtifactError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise DemoArtifactError(f"{field} must be finite and >= {minimum}")
    return result


def _sequence(value: object, field: str) -> list[object]:
    if not isinstance(value, list):
        raise DemoArtifactError(f"{field} must be an array")
    return cast(list[object], value)


def _sha256(value: object, field: str, *, require_prefix: bool) -> str:
    text = _string(value, field)
    has_prefix = text.startswith("sha256:")
    if require_prefix and not has_prefix:
        raise DemoArtifactError(f"{field} must use the sha256: prefix")
    if not require_prefix and has_prefix:
        raise DemoArtifactError(f"{field} must be an unprefixed SHA-256 digest")
    digest = text.removeprefix("sha256:")
    if len(digest) != 64 or any(character not in _HEX_DIGITS for character in digest):
        raise DemoArtifactError(f"{field} must contain a lowercase SHA-256 digest")
    return text


def _canonical_config(config: Mapping[str, object]) -> tuple[dict[str, object], str]:
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
        raise DemoArtifactError("resolved config must be finite JSON") from error
    if not isinstance(decoded, dict):
        raise DemoArtifactError("resolved config must be a JSON object")
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return cast(dict[str, object], decoded), f"sha256:{digest}"


def _read_json(path: Path, context: str) -> Mapping[str, object]:
    try:
        decoded: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DemoArtifactError(f"could not load {context} from {path!s}: {error}") from error
    return _mapping(decoded, context)


@dataclass(frozen=True, slots=True)
class DecisionProvenance:
    """Exact source identities for calibration and the abstention gate."""

    dataset_version: str
    protocol_hash: str
    manifest_hash: str
    checkpoint_config_hash: str
    checkpoint_sha256: str
    resolved_config_sha256: str
    normalization_sha256: str
    calibration_folds: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.dataset_version != PTBXL_VERSION:
            raise DemoArtifactError(f"dataset_version must be {PTBXL_VERSION}")
        _sha256(self.protocol_hash, "protocol_hash", require_prefix=True)
        _sha256(self.manifest_hash, "manifest_hash", require_prefix=False)
        _sha256(self.checkpoint_config_hash, "checkpoint_config_hash", require_prefix=True)
        _sha256(self.checkpoint_sha256, "checkpoint_sha256", require_prefix=False)
        _sha256(self.resolved_config_sha256, "resolved_config_sha256", require_prefix=False)
        _sha256(self.normalization_sha256, "normalization_sha256", require_prefix=False)
        if self.calibration_folds != CALIBRATION_FOLDS:
            raise DemoArtifactError(
                f"calibration and gating must be frozen from fold {CALIBRATION_FOLDS} only"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset_version": self.dataset_version,
            "protocol_hash": self.protocol_hash,
            "manifest_hash": self.manifest_hash,
            "checkpoint_config_hash": self.checkpoint_config_hash,
            "checkpoint_sha256": self.checkpoint_sha256,
            "resolved_config_sha256": self.resolved_config_sha256,
            "normalization_sha256": self.normalization_sha256,
            "calibration_folds": list(self.calibration_folds),
        }


@dataclass(frozen=True, slots=True)
class FrozenDecisionPolicy:
    """Temperature calibration, thresholds, and one frozen entropy gate."""

    temperature: float
    classification_thresholds: tuple[float, ...]
    uncertainty_threshold: float
    provenance: DecisionProvenance
    label_order: tuple[str, ...] = SUPERCLASSES
    calibration_method: str = "temperature_scaling"
    gate_method: str = "mean_normalized_binary_entropy"

    def __post_init__(self) -> None:
        if self.label_order != SUPERCLASSES:
            raise DemoArtifactError(f"label_order must be exactly {SUPERCLASSES!r}")
        if self.calibration_method != "temperature_scaling":
            raise DemoArtifactError("only temperature_scaling calibration is supported")
        if self.gate_method != "mean_normalized_binary_entropy":
            raise DemoArtifactError("only mean_normalized_binary_entropy gating is supported")
        if not math.isfinite(self.temperature) or self.temperature <= 0.0:
            raise DemoArtifactError("temperature must be finite and positive")
        if len(self.classification_thresholds) != len(SUPERCLASSES):
            raise DemoArtifactError(
                f"classification_thresholds must contain {len(SUPERCLASSES)} values"
            )
        if any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0
            for value in self.classification_thresholds
        ):
            raise DemoArtifactError("classification thresholds must be finite and in [0, 1]")
        if not math.isfinite(self.uncertainty_threshold) or not (
            0.0 <= self.uncertainty_threshold <= 1.0
        ):
            raise DemoArtifactError("uncertainty_threshold must be finite and in [0, 1]")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": DEMO_POLICY_SCHEMA_VERSION,
            "label_order": list(self.label_order),
            "calibration": {
                "method": self.calibration_method,
                "temperature": self.temperature,
            },
            "classification_thresholds": list(self.classification_thresholds),
            "gate": {
                "method": self.gate_method,
                "uncertainty_threshold": self.uncertainty_threshold,
            },
            "provenance": self.provenance.to_dict(),
        }

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        os.replace(temporary, destination)

    @classmethod
    def load(cls, path: str | Path) -> Self:
        payload = _read_json(Path(path), "decision policy")
        _exact_keys(
            payload,
            {
                "schema_version",
                "label_order",
                "calibration",
                "classification_thresholds",
                "gate",
                "provenance",
            },
            "decision policy",
        )
        if payload["schema_version"] != DEMO_POLICY_SCHEMA_VERSION:
            raise DemoArtifactError("unsupported decision policy schema_version")
        labels = tuple(
            _string(value, "label_order item")
            for value in _sequence(payload["label_order"], "label_order")
        )
        calibration = _mapping(payload["calibration"], "calibration")
        _exact_keys(calibration, {"method", "temperature"}, "calibration")
        thresholds = tuple(
            _number(value, "classification_thresholds item", minimum=0.0)
            for value in _sequence(
                payload["classification_thresholds"], "classification_thresholds"
            )
        )
        gate = _mapping(payload["gate"], "gate")
        _exact_keys(gate, {"method", "uncertainty_threshold"}, "gate")
        provenance_payload = _mapping(payload["provenance"], "provenance")
        _exact_keys(
            provenance_payload,
            {
                "dataset_version",
                "protocol_hash",
                "manifest_hash",
                "checkpoint_config_hash",
                "checkpoint_sha256",
                "resolved_config_sha256",
                "normalization_sha256",
                "calibration_folds",
            },
            "provenance",
        )
        raw_folds = _sequence(provenance_payload["calibration_folds"], "calibration_folds")
        folds = tuple(
            int(value)
            for value in raw_folds
            if isinstance(value, int) and not isinstance(value, bool)
        )
        if len(folds) != len(raw_folds):
            raise DemoArtifactError("calibration_folds must contain integers")
        provenance = DecisionProvenance(
            dataset_version=_string(provenance_payload["dataset_version"], "dataset_version"),
            protocol_hash=_string(provenance_payload["protocol_hash"], "protocol_hash"),
            manifest_hash=_string(provenance_payload["manifest_hash"], "manifest_hash"),
            checkpoint_config_hash=_string(
                provenance_payload["checkpoint_config_hash"], "checkpoint_config_hash"
            ),
            checkpoint_sha256=_string(provenance_payload["checkpoint_sha256"], "checkpoint_sha256"),
            resolved_config_sha256=_string(
                provenance_payload["resolved_config_sha256"], "resolved_config_sha256"
            ),
            normalization_sha256=_string(
                provenance_payload["normalization_sha256"], "normalization_sha256"
            ),
            calibration_folds=folds,
        )
        return cls(
            temperature=_number(calibration["temperature"], "temperature", minimum=0.0),
            classification_thresholds=thresholds,
            uncertainty_threshold=_number(
                gate["uncertainty_threshold"], "uncertainty_threshold", minimum=0.0
            ),
            provenance=provenance,
            label_order=labels,
            calibration_method=_string(calibration["method"], "calibration.method"),
            gate_method=_string(gate["method"], "gate.method"),
        )


@dataclass(frozen=True, slots=True)
class AttributionPayload:
    """Signed, normalized attribution values ready for a waveform plot."""

    method: AttributionMethod
    target_label: str
    values: Tensor
    coordinate_space: str

    def __post_init__(self) -> None:
        if self.target_label not in SUPERCLASSES:
            raise DemoInputError("attribution target label is not canonical")
        if self.values.ndim != 2 or self.values.shape[0] not in {1, len(LEADS)}:
            raise DemoInputError("attribution values must have shape [1, 1000] or [12, 1000]")
        if self.values.shape[1] != EXPECTED_SAMPLES or not torch.isfinite(self.values).all():
            raise DemoInputError("attribution values must be finite with 1000 samples")

    def to_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "target_label": self.target_label,
            "shape": list(self.values.shape),
            "values": self.values.tolist(),
            "signed": True,
            "normalized_to_unit_max_magnitude": True,
            "coordinate_space": self.coordinate_space,
        }


@dataclass(frozen=True, slots=True)
class DemoPrediction:
    """One immutable inference response with tensor and JSON-safe views."""

    label_order: tuple[str, ...]
    raw_logits: Tensor
    raw_probabilities: Tensor
    calibrated_probabilities: Tensor
    threshold_predictions: tuple[bool, ...]
    uncertainty: float
    gate_threshold: float
    decision: Decision
    decision_reason: str
    source: str
    attribution: AttributionPayload | None
    artifact_provenance: Mapping[str, object]

    def to_dict(self) -> dict[str, object]:
        predictions_exposed = self.decision == "accept"
        payload: dict[str, object] = {
            "label_order": list(self.label_order),
            "decision": {
                "status": ("LEGACY_BASELINE_DISPLAY_ALLOWED" if predictions_exposed else "ABSTAIN"),
                "reason": self.decision_reason,
                "predictions_exposed": predictions_exposed,
            },
            "predictions_exposed": predictions_exposed,
            "system_scope": "legacy_entropy_baseline_not_trust_sentinel",
            "source": self.source,
            "artifact_provenance": dict(self.artifact_provenance),
            "safety": {
                "notice": RESEARCH_ONLY_NOTICE,
                "limitations": list(LIMITATIONS),
            },
        }
        if predictions_exposed:
            raw = [float(value) for value in self.raw_probabilities.tolist()]
            calibrated = [float(value) for value in self.calibrated_probabilities.tolist()]
            payload.update(
                {
                    "raw_logits": [float(value) for value in self.raw_logits.tolist()],
                    "raw_probabilities": dict(zip(self.label_order, raw, strict=True)),
                    "calibrated_probabilities": dict(
                        zip(self.label_order, calibrated, strict=True)
                    ),
                    "threshold_predictions": dict(
                        zip(self.label_order, self.threshold_predictions, strict=True)
                    ),
                    "positive_labels": [
                        label
                        for label, predicted in zip(
                            self.label_order, self.threshold_predictions, strict=True
                        )
                        if predicted
                    ],
                    "uncertainty": self.uncertainty,
                    "gate_threshold": self.gate_threshold,
                    "attribution": (
                        None if self.attribution is None else self.attribution.to_dict()
                    ),
                }
            )
        return payload


def _validate_normalization(stats: NormalizationStats) -> None:
    provenance = stats.provenance
    if provenance.dataset_version != PTBXL_VERSION:
        raise DemoArtifactError("normalization dataset version is incompatible")
    if provenance.training_folds != TRAIN_FOLDS:
        raise DemoArtifactError("normalization must be fitted on training folds 1-7 only")
    if provenance.target_columns != TARGET_COLUMNS:
        raise DemoArtifactError("normalization target order is incompatible")
    if provenance.samples_per_record != EXPECTED_SAMPLES or not math.isclose(
        provenance.sampling_frequency_hz,
        EXPECTED_FREQUENCY_HZ,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise DemoArtifactError("normalization is not for canonical 100 Hz ECG inputs")


def _load_resolved_config(path: Path) -> tuple[dict[str, object], str]:
    payload = _read_json(path, "resolved config")
    _exact_keys(payload, {"config_hash", "config"}, "resolved config")
    stored_hash = _sha256(payload["config_hash"], "config_hash", require_prefix=True)
    config_mapping = _mapping(payload["config"], "resolved config.config")
    canonical, observed_hash = _canonical_config(config_mapping)
    if observed_hash != stored_hash:
        raise DemoArtifactError("resolved config hash does not match its content")
    return canonical, stored_hash


def _model_selection(config: Mapping[str, object]) -> ModelConfig:
    model_payload = _mapping(config.get("model"), "resolved config.model")
    try:
        return ModelConfig.from_mapping(
            {
                "architecture": model_payload.get("architecture"),
                "preset": model_payload.get("preset"),
            }
        )
    except ExperimentConfigError as error:
        raise DemoArtifactError(f"unsupported model configuration: {error}") from error


def _validate_model_metadata(
    model: nn.Module, config: Mapping[str, object], selection: ModelConfig
) -> None:
    model_payload = _mapping(config.get("model"), "resolved config.model")
    expected_class = model_payload.get("class")
    observed_class = f"{type(model).__module__}.{type(model).__qualname__}"
    if expected_class is not None and expected_class != observed_class:
        raise DemoArtifactError("resolved model class does not match selected architecture")
    expected_parameters = model_payload.get("trainable_parameters")
    if expected_parameters is not None and expected_parameters != count_parameters(model):
        raise DemoArtifactError("resolved model parameter count does not match architecture")
    expected_architecture_config = model_payload.get("resolved_architecture_config")
    raw_config = getattr(model, "config", None)
    if expected_architecture_config is not None:
        if raw_config is None or not is_dataclass(raw_config) or isinstance(raw_config, type):
            raise DemoArtifactError("model has no dataclass architecture configuration")
        canonical_architecture, _ = _canonical_config({"config": asdict(raw_config)})
        if expected_architecture_config != canonical_architecture["config"]:
            raise DemoArtifactError("resolved architecture configuration does not match preset")
    if selection.architecture == "resnet1d" and not isinstance(model, ResNet1D):
        raise DemoArtifactError("selected ResNet architecture constructed the wrong model class")


def _load_checkpoint_model(
    checkpoint_path: Path,
    *,
    config: Mapping[str, object],
    config_hash: str,
    policy: FrozenDecisionPolicy,
) -> nn.Module:
    try:
        decoded: object = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise DemoArtifactError(
            f"could not load checkpoint {checkpoint_path!s}: {error}"
        ) from error
    checkpoint = _mapping(decoded, "checkpoint")
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
    _exact_keys(checkpoint, required, "checkpoint")
    if checkpoint["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise DemoArtifactError("unsupported checkpoint schema_version")
    checkpoint_config = _mapping(checkpoint["config"], "checkpoint.config")
    canonical_checkpoint_config, recomputed_hash = _canonical_config(checkpoint_config)
    if recomputed_hash != checkpoint["config_hash"]:
        raise DemoArtifactError("checkpoint config hash does not match its content")
    if canonical_checkpoint_config != dict(config) or checkpoint["config_hash"] != config_hash:
        raise DemoArtifactError("checkpoint and resolved config do not match")
    provenance = policy.provenance
    if checkpoint["protocol_hash"] != provenance.protocol_hash:
        raise DemoArtifactError("checkpoint protocol hash does not match decision policy")
    if checkpoint["manifest_hash"] != provenance.manifest_hash:
        raise DemoArtifactError("checkpoint manifest hash does not match decision policy")
    if checkpoint["config_hash"] != provenance.checkpoint_config_hash:
        raise DemoArtifactError("checkpoint config hash does not match decision policy")
    canonical_protocol = ExperimentProtocol.canonical()
    if checkpoint["protocol_hash"] != canonical_protocol.protocol_hash:
        raise DemoArtifactError("checkpoint does not use the canonical experiment protocol")

    selection = _model_selection(config)
    try:
        model = build_experiment_model(selection)
    except DevelopmentRunnerError as error:
        raise DemoArtifactError(f"could not construct supported model: {error}") from error
    _validate_model_metadata(model, config, selection)
    state_dict = _mapping(checkpoint["model_state_dict"], "checkpoint.model_state_dict")
    try:
        model.load_state_dict(cast(Any, state_dict), strict=True)
    except (RuntimeError, ValueError) as error:
        raise DemoArtifactError(f"checkpoint model weights are incompatible: {error}") from error
    model.requires_grad_(False)
    return model.cpu().eval()


def _canonical_lead_positions(lead_names: Sequence[object], context: str) -> list[int]:
    if isinstance(lead_names, (str, bytes)) or len(lead_names) != len(LEADS):
        raise DemoInputError(f"{context} must contain exactly the canonical 12 leads")
    canonical_by_key = {lead.casefold(): lead for lead in LEADS}
    positions: dict[str, int] = {}
    for index, raw_name in enumerate(lead_names):
        if not isinstance(raw_name, str):
            raise DemoInputError(f"{context} contains a non-string lead name")
        canonical = canonical_by_key.get(raw_name.strip().casefold())
        if canonical is None or canonical in positions:
            raise DemoInputError(f"{context} contains an unexpected or duplicate lead")
        positions[canonical] = index
    if set(positions) != set(LEADS):
        raise DemoInputError(f"{context} is missing one or more canonical leads")
    return [positions[lead] for lead in LEADS]


def validate_physical_signal(
    signal: SignalArray,
    *,
    sampling_frequency_hz: float = EXPECTED_FREQUENCY_HZ,
    lead_names: Sequence[str] = LEADS,
    units: str = PHYSICAL_UNITS,
) -> Tensor:
    """Validate an in-memory physical ECG and return contiguous float32 data."""

    try:
        frequency = float(sampling_frequency_hz)
    except (TypeError, ValueError) as error:
        raise DemoInputError("sampling frequency must be numeric") from error
    if not math.isfinite(frequency) or not math.isclose(
        frequency, EXPECTED_FREQUENCY_HZ, rel_tol=0.0, abs_tol=1e-9
    ):
        raise DemoInputError(f"sampling frequency must be exactly {EXPECTED_FREQUENCY_HZ} Hz")
    if not isinstance(units, str) or units.strip().casefold() != PHYSICAL_UNITS.casefold():
        raise DemoInputError(f"physical signal units must be {PHYSICAL_UNITS}")
    positions = _canonical_lead_positions(lead_names, "lead_names")
    if positions != list(range(len(LEADS))):
        raise DemoInputError("in-memory lead_names must already use canonical order")
    try:
        tensor = torch.as_tensor(signal, dtype=torch.float32)
    except (TypeError, ValueError, RuntimeError) as error:
        raise DemoInputError(f"signal must be numeric: {error}") from error
    if tensor.shape != (len(LEADS), EXPECTED_SAMPLES):
        raise DemoInputError(
            f"signal must have shape [{len(LEADS)}, {EXPECTED_SAMPLES}], got {tuple(tensor.shape)}"
        )
    if not torch.isfinite(tensor).all():
        raise DemoInputError("signal must contain only finite physical values")
    return tensor.detach().cpu().contiguous()


def load_wfdb_physical_signal(record_path: str | Path) -> Tensor:
    """Read, validate, and canonically reorder one physical WFDB record."""

    path = Path(record_path)
    if path.suffix.casefold() in {".hea", ".dat"}:
        path = path.with_suffix("")
    if not path.with_suffix(".hea").is_file() or not path.with_suffix(".dat").is_file():
        raise DemoInputError("WFDB record requires matching .hea and .dat files")
    try:
        record = wfdb.rdrecord(str(path))
    except Exception as error:
        raise DemoInputError(f"could not read WFDB record {path!s}: {error}") from error
    try:
        frequency = float(cast(float | int | str, getattr(record, "fs", None)))
    except (TypeError, ValueError) as error:
        raise DemoInputError("WFDB record has an invalid sampling frequency") from error
    if not math.isfinite(frequency) or not math.isclose(
        frequency, EXPECTED_FREQUENCY_HZ, rel_tol=0.0, abs_tol=1e-9
    ):
        raise DemoInputError(f"WFDB record must use {EXPECTED_FREQUENCY_HZ} Hz")

    physical = getattr(record, "p_signal", None)
    if physical is None:
        raise DemoInputError("WFDB record has no physical signal")
    array = np.asarray(physical, dtype=np.float32)
    if array.shape != (EXPECTED_SAMPLES, len(LEADS)):
        raise DemoInputError(
            f"WFDB physical signal must have shape [{EXPECTED_SAMPLES}, {len(LEADS)}]"
        )
    raw_names = getattr(record, "sig_name", None)
    if not isinstance(raw_names, Sequence):
        raise DemoInputError("WFDB record has no valid lead names")
    positions = _canonical_lead_positions(cast(Sequence[object], raw_names), "WFDB leads")
    raw_units = getattr(record, "units", None)
    if not isinstance(raw_units, Sequence) or isinstance(raw_units, (str, bytes)):
        raise DemoInputError("WFDB record has no physical units")
    if len(raw_units) != len(LEADS) or any(
        not isinstance(unit, str) or unit.strip().casefold() != PHYSICAL_UNITS.casefold()
        for unit in raw_units
    ):
        raise DemoInputError(f"all WFDB leads must use physical units {PHYSICAL_UNITS}")
    canonical = np.ascontiguousarray(array[:, positions].T, dtype=np.float32)
    return validate_physical_signal(canonical)


class DemoInferenceBackend:
    """Frozen CPU inference pipeline suitable for a future UI adapter."""

    def __init__(
        self,
        *,
        model: nn.Module,
        normalization: NormalizationStats,
        policy: FrozenDecisionPolicy,
        artifact_provenance: Mapping[str, object],
    ) -> None:
        self.model = model.cpu().eval()
        self.normalization = normalization
        self.policy = policy
        self.artifact_provenance = dict(artifact_provenance)
        self._mean = torch.tensor(normalization.mean, dtype=torch.float32).unsqueeze(1)
        self._std = torch.tensor(normalization.std, dtype=torch.float32).unsqueeze(1)

    @classmethod
    def load(
        cls,
        *,
        checkpoint_path: str | Path,
        resolved_config_path: str | Path,
        normalization_path: str | Path,
        decision_policy_path: str | Path,
    ) -> Self:
        checkpoint = Path(checkpoint_path)
        resolved_config = Path(resolved_config_path)
        normalization_file = Path(normalization_path)
        decision_file = Path(decision_policy_path)
        policy = FrozenDecisionPolicy.load(decision_file)
        provenance = policy.provenance
        actual_hashes = {
            "checkpoint_sha256": sha256_file(checkpoint),
            "resolved_config_sha256": sha256_file(resolved_config),
            "normalization_sha256": sha256_file(normalization_file),
            "decision_policy_sha256": sha256_file(decision_file),
        }
        for field in (
            "checkpoint_sha256",
            "resolved_config_sha256",
            "normalization_sha256",
        ):
            if actual_hashes[field] != getattr(provenance, field):
                raise DemoArtifactError(f"{field} does not match decision-policy provenance")
        try:
            normalization = NormalizationStats.load(normalization_file)
        except NormalizationValidationError as error:
            raise DemoArtifactError(f"invalid normalization artifact: {error}") from error
        _validate_normalization(normalization)
        config, config_hash = _load_resolved_config(resolved_config)
        if config_hash != provenance.checkpoint_config_hash:
            raise DemoArtifactError("resolved config hash does not match decision policy")
        model = _load_checkpoint_model(
            checkpoint,
            config=config,
            config_hash=config_hash,
            policy=policy,
        )
        return cls(
            model=model,
            normalization=normalization,
            policy=policy,
            artifact_provenance={
                **actual_hashes,
                "protocol_hash": provenance.protocol_hash,
                "manifest_hash": provenance.manifest_hash,
                "checkpoint_config_hash": provenance.checkpoint_config_hash,
                "dataset_version": provenance.dataset_version,
                "calibration_folds": list(provenance.calibration_folds),
            },
        )

    def _predict(
        self,
        physical_signal: Tensor,
        *,
        source: str,
        attribution_method: AttributionMethod | None,
        attribution_target: str | int | None,
        integrated_gradients_steps: int,
    ) -> DemoPrediction:
        normalized = ((physical_signal - self._mean) / self._std).contiguous()
        if not torch.isfinite(normalized).all():
            raise DemoInputError("signal became non-finite after frozen normalization")
        inputs = normalized.unsqueeze(0)
        with torch.inference_mode():
            logits = self.model(inputs)
        if not isinstance(logits, Tensor) or logits.shape != (1, len(SUPERCLASSES)):
            raise DemoArtifactError("loaded model violated the five-logit output contract")
        logits = logits[0].detach().cpu().to(torch.float64)
        if not torch.isfinite(logits).all():
            raise DemoArtifactError("loaded model produced non-finite logits")
        raw_probabilities = logits.sigmoid()
        calibrated_probabilities = (logits / self.policy.temperature).sigmoid()
        epsilon = torch.finfo(calibrated_probabilities.dtype).eps
        clipped = calibrated_probabilities.clamp(epsilon, 1.0 - epsilon)
        entropy = -(clipped * clipped.log() + (1.0 - clipped) * (1.0 - clipped).log()) / math.log(
            2.0
        )
        uncertainty = float(entropy.mean())
        accepted = uncertainty <= self.policy.uncertainty_threshold
        decision: Decision = "accept" if accepted else "abstain"
        reason = (
            "uncertainty_within_frozen_fold9_gate"
            if accepted
            else "uncertainty_exceeds_frozen_fold9_gate"
        )
        threshold_predictions = tuple(
            bool(probability >= threshold)
            for probability, threshold in zip(
                calibrated_probabilities.tolist(),
                self.policy.classification_thresholds,
                strict=True,
            )
        )

        attribution: AttributionPayload | None = None
        if attribution_method is not None:
            target_index = self._attribution_target(attribution_target, calibrated_probabilities)
            if attribution_method == "grad_cam":
                if not isinstance(self.model, ResNet1D):
                    raise DemoInputError("grad_cam attribution is supported only for ResNet1D")
                from ecg_trust.explain import grad_cam_1d

                values = grad_cam_1d(self.model, inputs, target_index)[0]
            elif attribution_method == "integrated_gradients":
                from ecg_trust.explain import integrated_gradients

                values = integrated_gradients(
                    self.model,
                    inputs,
                    target_index,
                    n_steps=integrated_gradients_steps,
                )[0]
            else:
                raise DemoInputError(f"unsupported attribution method {attribution_method!r}")
            attribution = AttributionPayload(
                method=attribution_method,
                target_label=SUPERCLASSES[target_index],
                values=values.detach().cpu().to(torch.float32).contiguous(),
                coordinate_space="frozen_normalized_model_input",
            )

        return DemoPrediction(
            label_order=SUPERCLASSES,
            raw_logits=logits,
            raw_probabilities=raw_probabilities,
            calibrated_probabilities=calibrated_probabilities,
            threshold_predictions=threshold_predictions,
            uncertainty=uncertainty,
            gate_threshold=self.policy.uncertainty_threshold,
            decision=decision,
            decision_reason=reason,
            source=source,
            attribution=attribution,
            artifact_provenance=self.artifact_provenance,
        )

    def extract_embedding_signal(
        self,
        signal: SignalArray,
        *,
        sampling_frequency_hz: float = EXPECTED_FREQUENCY_HZ,
        lead_names: Sequence[str] = LEADS,
        units: str = PHYSICAL_UNITS,
    ) -> Tensor:
        """Return the frozen pre-classifier embedding for an OOD detector.

        The method preserves the existing prediction path and runs a separate
        inference-only feature pass.  ResNet temporal features are pooled with
        the model's own global-pooling layer; the transformer already returns
        its final class-token embedding.
        """

        physical = validate_physical_signal(
            signal,
            sampling_frequency_hz=sampling_frequency_hz,
            lead_names=lead_names,
            units=units,
        )
        normalized = ((physical - self._mean) / self._std).contiguous()
        if not torch.isfinite(normalized).all():
            raise DemoInputError("signal became non-finite after frozen normalization")
        inputs = normalized.unsqueeze(0)
        with torch.inference_mode():
            if isinstance(self.model, ResNet1D):
                embedding = self.model.forward_embedding(inputs)
            elif isinstance(self.model, ECGTransformer):
                embedding = self.model.forward_features(inputs)
            else:  # pragma: no cover - loading already restricts architectures.
                raise DemoArtifactError("loaded model has no supported embedding contract")
        if (
            not isinstance(embedding, Tensor)
            or embedding.ndim != 2
            or embedding.shape[0] != 1
            or embedding.shape[1] < 1
            or not torch.isfinite(embedding).all()
        ):
            raise DemoArtifactError("loaded model violated the finite embedding contract")
        return embedding[0].detach().cpu().to(torch.float32).contiguous()

    @staticmethod
    def _attribution_target(target: str | int | None, probabilities: Tensor) -> int:
        if target is None:
            return int(probabilities.argmax())
        if isinstance(target, bool):
            raise DemoInputError("attribution target cannot be boolean")
        if isinstance(target, int):
            if not 0 <= target < len(SUPERCLASSES):
                raise DemoInputError("attribution target index is out of range")
            return target
        if isinstance(target, str) and target in SUPERCLASSES:
            return SUPERCLASSES.index(target)
        raise DemoInputError(f"attribution target must be one of {SUPERCLASSES!r}")

    def predict_signal(
        self,
        signal: SignalArray,
        *,
        sampling_frequency_hz: float = EXPECTED_FREQUENCY_HZ,
        lead_names: Sequence[str] = LEADS,
        units: str = PHYSICAL_UNITS,
        attribution_method: AttributionMethod | None = None,
        attribution_target: str | int | None = None,
        integrated_gradients_steps: int = 32,
    ) -> DemoPrediction:
        physical = validate_physical_signal(
            signal,
            sampling_frequency_hz=sampling_frequency_hz,
            lead_names=lead_names,
            units=units,
        )
        return self._predict(
            physical,
            source="in_memory",
            attribution_method=attribution_method,
            attribution_target=attribution_target,
            integrated_gradients_steps=integrated_gradients_steps,
        )

    def predict_record(
        self,
        record_path: str | Path,
        *,
        attribution_method: AttributionMethod | None = None,
        attribution_target: str | int | None = None,
        integrated_gradients_steps: int = 32,
    ) -> DemoPrediction:
        physical = load_wfdb_physical_signal(record_path)
        return self._predict(
            physical,
            source=str(Path(record_path)),
            attribution_method=attribution_method,
            attribution_target=attribution_target,
            integrated_gradients_steps=integrated_gradients_steps,
        )


__all__ = [
    "DEMO_POLICY_SCHEMA_VERSION",
    "LIMITATIONS",
    "RESEARCH_ONLY_NOTICE",
    "AttributionPayload",
    "DecisionProvenance",
    "DemoArtifactError",
    "DemoInferenceBackend",
    "DemoInputError",
    "DemoPrediction",
    "FrozenDecisionPolicy",
    "load_wfdb_physical_signal",
    "validate_physical_signal",
]
