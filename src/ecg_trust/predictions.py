"""Immutable, integrity-checked prediction artifacts.

Prediction arrays are stored in a non-pickled NPZ archive and provenance is
stored in a canonical JSON sidecar. The sidecar binds itself to the archive
with SHA-256, while an artifact hash covers both the archive digest and all
metadata. Files are committed with same-directory atomic replacements and an
existing artifact pair is never overwritten.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Self, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ecg_trust import __version__
from ecg_trust.protocol import (
    LABEL_ORDER,
    ExperimentProtocol,
    FinalTestAccessToken,
    FoldRole,
)

PREDICTION_SCHEMA_VERSION = 1
PREDICTION_ARTIFACT_TYPE = "ecg_trust.multilabel_predictions"
_SHA256_PATTERN = re.compile(r"(?:sha256:)?([0-9a-fA-F]{64})\Z")
_TIMESTAMP_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z\Z")
_ARRAY_NAMES = ("ecg_id", "patient_id", "strat_fold", "targets", "raw_logits")
_CALIBRATED_NAME = "calibrated_probabilities"

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
SmallIntArray = NDArray[np.int8]
StringArray = NDArray[np.str_]
IdentifierArray = IntArray | StringArray
MetadataScalar = str | int | float | bool | None


class PredictionArtifactError(ValueError):
    """Raised when a prediction artifact violates its data contract."""


class PredictionIntegrityError(PredictionArtifactError):
    """Raised when stored prediction data fails integrity verification."""


class PredictionAlignmentError(PredictionArtifactError):
    """Raised when two model artifacts do not describe identical rows."""


@dataclass(frozen=True, slots=True)
class PredictionArtifactFiles:
    """Paths and integrity hashes committed by a save operation."""

    npz_path: Path
    json_path: Path
    npz_sha256: str
    artifact_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "npz_path": str(self.npz_path),
            "json_path": str(self.json_path),
            "npz_sha256": self.npz_sha256,
            "artifact_sha256": self.artifact_sha256,
        }


@dataclass(frozen=True, slots=True, init=False)
class PredictionArtifact:
    """Validated in-memory representation with read-only aligned arrays."""

    ecg_id: IdentifierArray
    patient_id: IdentifierArray
    strat_fold: SmallIntArray
    targets: SmallIntArray
    raw_logits: FloatArray
    calibrated_probabilities: FloatArray | None
    label_order: tuple[str, ...]
    model_name: str
    model_seed: int
    protocol_hash: str
    config_hash: str
    manifest_hash: str
    fold_role: FoldRole
    created_at_utc: str
    producer: str
    software_versions: Mapping[str, str]
    extra_metadata: Mapping[str, MetadataScalar]
    integrity_sha256: str | None

    @classmethod
    def _create(
        cls,
        *,
        ecg_id: ArrayLike,
        patient_id: ArrayLike,
        strat_fold: ArrayLike,
        targets: ArrayLike,
        raw_logits: ArrayLike,
        calibrated_probabilities: ArrayLike | None,
        label_order: Sequence[str],
        model_name: str,
        model_seed: int,
        protocol: ExperimentProtocol,
        protocol_hash: str,
        config_hash: str,
        manifest_hash: str,
        fold_role: FoldRole | str,
        created_at_utc: str,
        producer: str,
        software_versions: Mapping[str, str],
        extra_metadata: Mapping[str, MetadataScalar],
        integrity_sha256: str | None,
        test_access: FinalTestAccessToken | None,
    ) -> Self:
        labels = _validate_label_order(label_order)
        ecg_ids = _validate_identifier_array(ecg_id, name="ecg_id")
        patient_ids = _validate_identifier_array(patient_id, name="patient_id")
        n_samples = int(ecg_ids.shape[0])
        if n_samples == 0:
            raise PredictionArtifactError("prediction artifacts require at least one row")
        if patient_ids.shape[0] != n_samples:
            raise PredictionArtifactError("patient_id must align one-to-one with ecg_id")
        if len(set(_identifier_values(ecg_ids))) != n_samples:
            raise PredictionArtifactError("ecg_id values must be unique")

        folds = _validate_fold_array(strat_fold, n_samples=n_samples)
        binary_targets = _validate_targets(
            targets, n_samples=n_samples, n_labels=len(labels)
        )
        logits = _validate_float_matrix(
            raw_logits,
            name="raw_logits",
            n_samples=n_samples,
            n_labels=len(labels),
            probabilities=False,
        )
        calibrated = (
            None
            if calibrated_probabilities is None
            else _validate_float_matrix(
                calibrated_probabilities,
                name="calibrated_probabilities",
                n_samples=n_samples,
                n_labels=len(labels),
                probabilities=True,
            )
        )
        role = _validate_fold_role(fold_role)
        normalized_protocol_hash = _normalize_sha256(protocol_hash, "protocol_hash")
        if normalized_protocol_hash != protocol.protocol_hash:
            raise PredictionArtifactError(
                "artifact protocol_hash does not match the supplied protocol"
            )
        normalized_config_hash = _normalize_sha256(config_hash, "config_hash")
        normalized_manifest_hash = _normalize_sha256(manifest_hash, "manifest_hash")
        expected_folds = protocol.folds_for(role, test_access=test_access)
        observed_folds = tuple(sorted(int(value) for value in np.unique(folds)))
        if not set(observed_folds).issubset(expected_folds):
            raise PredictionArtifactError(
                f"fold role {role.value!r} permits {expected_folds}, "
                f"but artifact contains {observed_folds}"
            )
        _validate_patient_fold_consistency(patient_ids, folds)

        normalized_model_name = _nonempty_string(model_name, "model_name")
        normalized_seed = _validate_seed(model_seed)
        timestamp = _validate_timestamp(created_at_utc)
        normalized_producer = _nonempty_string(producer, "producer")
        versions = _validate_string_mapping(software_versions, "software_versions")
        metadata = _validate_extra_metadata(extra_metadata)
        integrity = (
            None
            if integrity_sha256 is None
            else _normalize_sha256(integrity_sha256, "integrity_sha256")
        )

        order = np.argsort(ecg_ids, kind="stable")
        ecg_ids = _readonly_copy(ecg_ids[order])
        patient_ids = _readonly_copy(patient_ids[order])
        folds = _readonly_copy(folds[order])
        binary_targets = _readonly_copy(binary_targets[order])
        logits = _readonly_copy(logits[order])
        if calibrated is not None:
            calibrated = _readonly_copy(calibrated[order])

        instance = object.__new__(cls)
        object.__setattr__(instance, "ecg_id", ecg_ids)
        object.__setattr__(instance, "patient_id", patient_ids)
        object.__setattr__(instance, "strat_fold", folds)
        object.__setattr__(instance, "targets", binary_targets)
        object.__setattr__(instance, "raw_logits", logits)
        object.__setattr__(instance, "calibrated_probabilities", calibrated)
        object.__setattr__(instance, "label_order", labels)
        object.__setattr__(instance, "model_name", normalized_model_name)
        object.__setattr__(instance, "model_seed", normalized_seed)
        object.__setattr__(instance, "protocol_hash", normalized_protocol_hash)
        object.__setattr__(instance, "config_hash", normalized_config_hash)
        object.__setattr__(instance, "manifest_hash", normalized_manifest_hash)
        object.__setattr__(instance, "fold_role", role)
        object.__setattr__(instance, "created_at_utc", timestamp)
        object.__setattr__(instance, "producer", normalized_producer)
        object.__setattr__(
            instance, "software_versions", MappingProxyType(dict(versions))
        )
        object.__setattr__(
            instance, "extra_metadata", MappingProxyType(dict(metadata))
        )
        object.__setattr__(instance, "integrity_sha256", integrity)
        return instance

    @property
    def n_samples(self) -> int:
        return int(self.ecg_id.shape[0])

    @property
    def folds(self) -> tuple[int, ...]:
        return tuple(sorted(int(value) for value in np.unique(self.strat_fold)))

    @property
    def alignment_sha256(self) -> str:
        """Hash identities, folds, targets, and label order for paired use."""

        payload: dict[str, object] = {
            "label_order": list(self.label_order),
            "ecg_id": _identifier_payload(self.ecg_id),
            "patient_id": _identifier_payload(self.patient_id),
            "strat_fold": self.strat_fold.astype(int).tolist(),
            "targets": self.targets.astype(int).tolist(),
        }
        return _hash_canonical_json(payload)

    def probabilities(self, *, require_calibrated: bool = False) -> FloatArray:
        """Return calibrated probabilities, or sigmoid logits when permitted."""

        if self.calibrated_probabilities is not None:
            return self.calibrated_probabilities
        if require_calibrated:
            raise PredictionArtifactError(
                "artifact has no calibrated probabilities; raw-logit fallback was disabled"
            )
        result = _stable_sigmoid(self.raw_logits)
        return _readonly_copy(result)

    def to_summary_dict(self) -> dict[str, object]:
        """Return JSON-safe provenance without embedding prediction arrays."""

        return {
            "schema_version": PREDICTION_SCHEMA_VERSION,
            "artifact_type": PREDICTION_ARTIFACT_TYPE,
            "model": {"name": self.model_name, "seed": self.model_seed},
            "protocol_hash": self.protocol_hash,
            "config_hash": self.config_hash,
            "manifest_hash": self.manifest_hash,
            "fold_role": self.fold_role.value,
            "folds": list(self.folds),
            "label_order": list(self.label_order),
            "record_count": self.n_samples,
            "has_calibrated_probabilities": self.calibrated_probabilities is not None,
            "alignment_sha256": self.alignment_sha256,
            "created": {
                "timestamp_utc": self.created_at_utc,
                "producer": self.producer,
                "software_versions": dict(self.software_versions),
                "extra": dict(self.extra_metadata),
            },
            "integrity_sha256": self.integrity_sha256,
        }

    def _array_dict(self) -> dict[str, NDArray[np.generic]]:
        arrays: dict[str, NDArray[np.generic]] = {
            "ecg_id": self.ecg_id,
            "patient_id": self.patient_id,
            "strat_fold": self.strat_fold,
            "targets": self.targets,
            "raw_logits": self.raw_logits,
        }
        if self.calibrated_probabilities is not None:
            arrays[_CALIBRATED_NAME] = self.calibrated_probabilities
        return arrays


def create_prediction_artifact(
    *,
    ecg_id: ArrayLike,
    patient_id: ArrayLike,
    strat_fold: ArrayLike,
    targets: ArrayLike,
    raw_logits: ArrayLike,
    model_name: str,
    model_seed: int,
    protocol: ExperimentProtocol,
    config_hash: str,
    manifest_hash: str,
    fold_role: FoldRole | str,
    calibrated_probabilities: ArrayLike | None = None,
    label_order: Sequence[str] = LABEL_ORDER,
    created_at_utc: str | None = None,
    producer: str = "ecg_trust.predictions",
    extra_metadata: Mapping[str, MetadataScalar] | None = None,
    test_access: FinalTestAccessToken | None = None,
) -> PredictionArtifact:
    """Create a canonical, sorted, read-only prediction artifact."""

    timestamp = created_at_utc or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    software_versions = {
        "ecg_trust": __version__,
        "numpy": np.__version__,
        "python": platform.python_version(),
    }
    return PredictionArtifact._create(
        ecg_id=ecg_id,
        patient_id=patient_id,
        strat_fold=strat_fold,
        targets=targets,
        raw_logits=raw_logits,
        calibrated_probabilities=calibrated_probabilities,
        label_order=label_order,
        model_name=model_name,
        model_seed=model_seed,
        protocol=protocol,
        protocol_hash=protocol.protocol_hash,
        config_hash=config_hash,
        manifest_hash=manifest_hash,
        fold_role=fold_role,
        created_at_utc=timestamp,
        producer=producer,
        software_versions=software_versions,
        extra_metadata=extra_metadata or {},
        integrity_sha256=None,
        test_access=test_access,
    )


def save_prediction_artifact(
    artifact: PredictionArtifact,
    path: str | Path,
    *,
    protocol: ExperimentProtocol,
    test_access: FinalTestAccessToken | None = None,
) -> PredictionArtifactFiles:
    """Atomically save a new NPZ/JSON pair without overwriting existing files."""

    _validate_artifact_for_protocol(artifact, protocol, test_access=test_access)
    npz_path, json_path = _resolve_artifact_paths(path, for_save=True)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    existing = [candidate for candidate in (npz_path, json_path) if candidate.exists()]
    if existing:
        names = ", ".join(str(candidate) for candidate in existing)
        raise FileExistsError(f"prediction artifact is immutable; refusing to overwrite {names}")

    npz_temp = _temporary_path(npz_path)
    json_temp = _temporary_path(json_path)
    npz_committed = False
    try:
        _write_npz(npz_temp, artifact._array_dict())
        npz_digest = _sha256_file(npz_temp)
        npz_size = npz_temp.stat().st_size
        metadata = _storage_metadata(
            artifact,
            npz_filename=npz_path.name,
            npz_size=npz_size,
            npz_sha256=npz_digest,
        )
        artifact_digest = _artifact_digest(metadata)
        metadata["artifact_sha256"] = artifact_digest
        _write_json(json_temp, metadata)

        os.replace(npz_temp, npz_path)
        npz_committed = True
        os.replace(json_temp, json_path)
    except Exception:
        _unlink_if_present(npz_temp)
        _unlink_if_present(json_temp)
        if npz_committed and not json_path.exists():
            _unlink_if_present(npz_path)
        raise

    return PredictionArtifactFiles(
        npz_path=npz_path,
        json_path=json_path,
        npz_sha256=npz_digest,
        artifact_sha256=artifact_digest,
    )


def load_prediction_artifact(
    path: str | Path,
    *,
    protocol: ExperimentProtocol,
    test_access: FinalTestAccessToken | None = None,
    expected_config_hash: str | None = None,
    expected_manifest_hash: str | None = None,
) -> PredictionArtifact:
    """Load an artifact after metadata, role gate, and all hashes verify."""

    npz_path, json_path = _resolve_artifact_paths(path, for_save=False)
    root = _load_metadata(json_path)
    metadata_protocol_hash = _normalize_sha256(
        root["protocol_hash"], "protocol_hash"
    )
    if metadata_protocol_hash != protocol.protocol_hash:
        raise PredictionArtifactError(
            "stored protocol_hash does not match the supplied protocol"
        )
    role = _validate_fold_role(root["fold_role"])
    # Gate declared final-test artifacts before opening their prediction archive.
    protocol.folds_for(role, test_access=test_access)

    stored_artifact_hash = _normalize_sha256(
        root["artifact_sha256"], "artifact_sha256"
    )
    integrity_payload = dict(root)
    del integrity_payload["artifact_sha256"]
    observed_artifact_hash = _artifact_digest(integrity_payload)
    if stored_artifact_hash != observed_artifact_hash:
        raise PredictionIntegrityError(
            "JSON sidecar integrity mismatch: artifact_sha256 is invalid"
        )

    if _expect_string(root["npz_file"], "npz_file") != npz_path.name:
        raise PredictionIntegrityError("sidecar npz_file does not match the archive path")
    if not npz_path.is_file():
        raise PredictionIntegrityError(f"prediction archive is missing: {npz_path}")
    expected_size = _expect_integer(root["npz_size_bytes"], "npz_size_bytes", minimum=1)
    if npz_path.stat().st_size != expected_size:
        raise PredictionIntegrityError("prediction archive size does not match sidecar")
    expected_npz_hash = _normalize_sha256(root["npz_sha256"], "npz_sha256")
    observed_npz_hash = _sha256_file(npz_path)
    if expected_npz_hash != observed_npz_hash:
        raise PredictionIntegrityError("prediction archive SHA-256 mismatch")

    arrays = _load_npz_arrays(npz_path, root["arrays"])
    created = _expect_mapping(root["created"], "created")
    _expect_keys(
        created,
        required={"timestamp_utc", "producer", "software_versions", "extra"},
        context="created",
    )
    model = _expect_mapping(root["model"], "model")
    _expect_keys(model, required={"name", "seed"}, context="model")
    artifact = PredictionArtifact._create(
        ecg_id=arrays["ecg_id"],
        patient_id=arrays["patient_id"],
        strat_fold=arrays["strat_fold"],
        targets=arrays["targets"],
        raw_logits=arrays["raw_logits"],
        calibrated_probabilities=arrays.get(_CALIBRATED_NAME),
        label_order=_expect_string_sequence(root["label_order"], "label_order"),
        model_name=_expect_string(model["name"], "model.name"),
        model_seed=_expect_integer(model["seed"], "model.seed", minimum=0),
        protocol=protocol,
        protocol_hash=metadata_protocol_hash,
        config_hash=_expect_string(root["config_hash"], "config_hash"),
        manifest_hash=_expect_string(root["manifest_hash"], "manifest_hash"),
        fold_role=role,
        created_at_utc=_expect_string(created["timestamp_utc"], "created.timestamp_utc"),
        producer=_expect_string(created["producer"], "created.producer"),
        software_versions=_expect_string_mapping(
            created["software_versions"], "created.software_versions"
        ),
        extra_metadata=_expect_metadata_mapping(created["extra"], "created.extra"),
        integrity_sha256=stored_artifact_hash,
        test_access=test_access,
    )
    _verify_loaded_metadata(artifact, root)

    if expected_config_hash is not None and artifact.config_hash != _normalize_sha256(
        expected_config_hash, "expected_config_hash"
    ):
        raise PredictionIntegrityError("stored config_hash does not match expectation")
    if expected_manifest_hash is not None and artifact.manifest_hash != _normalize_sha256(
        expected_manifest_hash, "expected_manifest_hash"
    ):
        raise PredictionIntegrityError("stored manifest_hash does not match expectation")
    return artifact


def assert_prediction_artifacts_aligned(
    first: PredictionArtifact,
    second: PredictionArtifact,
) -> None:
    """Prove two model artifacts can enter a paired patient bootstrap."""

    scalar_mismatches: list[str] = []
    if first.protocol_hash != second.protocol_hash:
        scalar_mismatches.append("protocol_hash")
    if first.manifest_hash != second.manifest_hash:
        scalar_mismatches.append("manifest_hash")
    if first.label_order != second.label_order:
        scalar_mismatches.append("label_order")
    if first.fold_role != second.fold_role:
        scalar_mismatches.append("fold_role")
    if first.alignment_sha256 != second.alignment_sha256:
        scalar_mismatches.append("alignment_sha256")
    if scalar_mismatches:
        raise PredictionAlignmentError(
            "prediction artifacts are not aligned: " + ", ".join(scalar_mismatches)
        )
    for name in ("ecg_id", "patient_id", "strat_fold", "targets"):
        if not np.array_equal(getattr(first, name), getattr(second, name)):
            raise PredictionAlignmentError(f"prediction artifacts differ in {name}")


def _validate_artifact_for_protocol(
    artifact: PredictionArtifact,
    protocol: ExperimentProtocol,
    *,
    test_access: FinalTestAccessToken | None,
) -> None:
    if not isinstance(artifact, PredictionArtifact):
        raise TypeError("artifact must be a PredictionArtifact")
    if artifact.protocol_hash != protocol.protocol_hash:
        raise PredictionArtifactError(
            "artifact protocol_hash does not match the supplied protocol"
        )
    expected = protocol.folds_for(artifact.fold_role, test_access=test_access)
    if not set(artifact.folds).issubset(expected):
        raise PredictionArtifactError(
            f"fold role {artifact.fold_role.value!r} rejects folds {artifact.folds}"
        )
    _validate_patient_fold_consistency(artifact.patient_id, artifact.strat_fold)


def _storage_metadata(
    artifact: PredictionArtifact,
    *,
    npz_filename: str,
    npz_size: int,
    npz_sha256: str,
) -> dict[str, object]:
    return {
        "schema_version": PREDICTION_SCHEMA_VERSION,
        "artifact_type": PREDICTION_ARTIFACT_TYPE,
        "model": {"name": artifact.model_name, "seed": artifact.model_seed},
        "protocol_hash": artifact.protocol_hash,
        "config_hash": artifact.config_hash,
        "manifest_hash": artifact.manifest_hash,
        "fold_role": artifact.fold_role.value,
        "folds": list(artifact.folds),
        "label_order": list(artifact.label_order),
        "record_count": artifact.n_samples,
        "alignment_sha256": artifact.alignment_sha256,
        "created": {
            "timestamp_utc": artifact.created_at_utc,
            "producer": artifact.producer,
            "software_versions": dict(artifact.software_versions),
            "extra": dict(artifact.extra_metadata),
        },
        "arrays": _array_descriptors(artifact._array_dict()),
        "npz_file": npz_filename,
        "npz_size_bytes": npz_size,
        "npz_sha256": npz_sha256,
    }


def _verify_loaded_metadata(
    artifact: PredictionArtifact,
    root: Mapping[str, object],
) -> None:
    if _expect_integer(root["record_count"], "record_count", minimum=1) != artifact.n_samples:
        raise PredictionIntegrityError("record_count does not match prediction arrays")
    stored_folds = _expect_integer_tuple(root["folds"], "folds")
    if stored_folds != artifact.folds:
        raise PredictionIntegrityError("stored folds do not match prediction arrays")
    alignment = _normalize_sha256(root["alignment_sha256"], "alignment_sha256")
    if alignment != artifact.alignment_sha256:
        raise PredictionIntegrityError("row alignment SHA-256 mismatch")
    stored_descriptors = _parse_array_descriptors(root["arrays"])
    actual_descriptors = _array_descriptors(artifact._array_dict())
    if stored_descriptors != actual_descriptors:
        raise PredictionIntegrityError(
            "stored array descriptors do not match normalized prediction arrays"
        )


def _array_descriptors(
    arrays: Mapping[str, NDArray[np.generic]],
) -> dict[str, object]:
    return {
        name: {"dtype": str(array.dtype), "shape": list(array.shape)}
        for name, array in arrays.items()
    }


def _load_metadata(path: Path) -> Mapping[str, object]:
    if not path.is_file():
        raise PredictionIntegrityError(f"prediction sidecar is missing: {path}")
    if path.stat().st_size > 1_000_000:
        raise PredictionIntegrityError("prediction sidecar is unreasonably large")
    try:
        decoded: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PredictionIntegrityError(f"could not decode prediction sidecar: {exc}") from exc
    root = _expect_mapping(decoded, "prediction sidecar")
    _expect_keys(
        root,
        required={
            "schema_version",
            "artifact_type",
            "model",
            "protocol_hash",
            "config_hash",
            "manifest_hash",
            "fold_role",
            "folds",
            "label_order",
            "record_count",
            "alignment_sha256",
            "created",
            "arrays",
            "npz_file",
            "npz_size_bytes",
            "npz_sha256",
            "artifact_sha256",
        },
        context="prediction sidecar",
    )
    if _expect_integer(root["schema_version"], "schema_version", minimum=1) != 1:
        raise PredictionArtifactError(
            f"unsupported prediction schema_version: {root['schema_version']!r}"
        )
    if _expect_string(root["artifact_type"], "artifact_type") != PREDICTION_ARTIFACT_TYPE:
        raise PredictionArtifactError("unexpected prediction artifact_type")
    return root


def _load_npz_arrays(
    path: Path,
    descriptor_value: object,
) -> dict[str, NDArray[np.generic]]:
    descriptors = _parse_array_descriptors(descriptor_value)
    expected_names = set(_ARRAY_NAMES)
    if _CALIBRATED_NAME in descriptors:
        expected_names.add(_CALIBRATED_NAME)
    if set(descriptors) != expected_names:
        raise PredictionIntegrityError(
            f"invalid array inventory: expected {sorted(expected_names)}, "
            f"received {sorted(descriptors)}"
        )
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != expected_names:
                raise PredictionIntegrityError(
                    "NPZ array inventory does not match the JSON sidecar"
                )
            arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}
    except PredictionIntegrityError:
        raise
    except (OSError, ValueError) as exc:
        raise PredictionIntegrityError(f"could not decode prediction archive: {exc}") from exc
    for name, array in arrays.items():
        descriptor = descriptors[name]
        if descriptor["dtype"] != str(array.dtype) or descriptor["shape"] != list(array.shape):
            raise PredictionIntegrityError(f"array descriptor mismatch for {name}")
    return arrays


def _parse_array_descriptors(value: object) -> dict[str, dict[str, object]]:
    mapping = _expect_mapping(value, "arrays")
    result: dict[str, dict[str, object]] = {}
    for name, descriptor_value in mapping.items():
        descriptor = _expect_mapping(descriptor_value, f"arrays.{name}")
        _expect_keys(
            descriptor,
            required={"dtype", "shape"},
            context=f"arrays.{name}",
        )
        dtype = _expect_string(descriptor["dtype"], f"arrays.{name}.dtype")
        shape = list(_expect_integer_tuple(descriptor["shape"], f"arrays.{name}.shape"))
        result[name] = {"dtype": dtype, "shape": shape}
    return result


def _artifact_digest(metadata_without_artifact_hash: Mapping[str, object]) -> str:
    canonical = json.dumps(
        metadata_without_artifact_hash,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _hash_canonical_json(payload: Mapping[str, object]) -> str:
    serialized = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return "sha256:" + hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _write_npz(path: Path, arrays: Mapping[str, NDArray[np.generic]]) -> None:
    with path.open("wb") as handle:
        if _CALIBRATED_NAME in arrays:
            np.savez_compressed(
                handle,
                ecg_id=arrays["ecg_id"],
                patient_id=arrays["patient_id"],
                strat_fold=arrays["strat_fold"],
                targets=arrays["targets"],
                raw_logits=arrays["raw_logits"],
                calibrated_probabilities=arrays[_CALIBRATED_NAME],
            )
        else:
            np.savez_compressed(
                handle,
                ecg_id=arrays["ecg_id"],
                patient_id=arrays["patient_id"],
                strat_fold=arrays["strat_fold"],
                targets=arrays["targets"],
                raw_logits=arrays["raw_logits"],
            )
        handle.flush()
        os.fsync(handle.fileno())


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    serialized = json.dumps(
        payload, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True
    )
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(serialized)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _temporary_path(destination: Path) -> Path:
    descriptor, raw_path = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    return Path(raw_path)


def _unlink_if_present(path: Path) -> None:
    with suppress(OSError):
        path.unlink(missing_ok=True)


def _resolve_artifact_paths(path: str | Path, *, for_save: bool) -> tuple[Path, Path]:
    candidate = Path(path)
    suffix = candidate.suffix.casefold()
    if for_save and suffix != ".npz":
        raise PredictionArtifactError("save path must end in .npz")
    if suffix == ".npz":
        return candidate, candidate.with_suffix(".json")
    if not for_save and suffix == ".json":
        return candidate.with_suffix(".npz"), candidate
    raise PredictionArtifactError("prediction artifact path must end in .npz or .json")


def _validate_label_order(label_order: Sequence[str]) -> tuple[str, ...]:
    labels = tuple(label_order)
    if labels != LABEL_ORDER:
        raise PredictionArtifactError(
            f"label_order must be exactly {LABEL_ORDER!r}; received {labels!r}"
        )
    return labels


def _validate_identifier_array(value: ArrayLike, *, name: str) -> IdentifierArray:
    try:
        raw = np.asarray(value, dtype=object)
    except (TypeError, ValueError) as exc:
        raise PredictionArtifactError(f"{name} must be a one-dimensional array") from exc
    if raw.ndim != 1:
        raise PredictionArtifactError(f"{name} must be one-dimensional")
    normalized: list[int | str] = []
    kinds: set[str] = set()
    for item in raw:
        if isinstance(item, np.generic):
            item = item.item()
        if isinstance(item, bool) or item is None:
            raise PredictionArtifactError(
                f"{name} cannot contain booleans or missing values"
            )
        if isinstance(item, int):
            normalized.append(item)
            kinds.add("integer")
        elif isinstance(item, float) and math.isfinite(item) and item.is_integer():
            normalized.append(int(item))
            kinds.add("integer")
        elif isinstance(item, str) and item.strip():
            normalized.append(item)
            kinds.add("string")
        else:
            raise PredictionArtifactError(
                f"{name} must contain non-empty strings or integer-valued numbers"
            )
    if len(kinds) > 1:
        raise PredictionArtifactError(f"{name} cannot mix string and integer identifiers")
    if not kinds:
        return np.asarray([], dtype=np.int64)
    if kinds == {"integer"}:
        try:
            return np.asarray(normalized, dtype=np.int64)
        except OverflowError as exc:
            raise PredictionArtifactError(f"{name} integer is outside int64 range") from exc
    return np.asarray(normalized, dtype=np.str_)


def _validate_fold_array(value: ArrayLike, *, n_samples: int) -> SmallIntArray:
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.shape[0] != n_samples:
        raise PredictionArtifactError("strat_fold must have one value per prediction row")
    if raw.dtype.kind not in {"i", "u"}:
        raise PredictionArtifactError("strat_fold must contain integers")
    if np.any((raw < 1) | (raw > 10)):
        raise PredictionArtifactError("strat_fold values must be between 1 and 10")
    return raw.astype(np.int8, copy=True)


def _validate_targets(
    value: ArrayLike,
    *,
    n_samples: int,
    n_labels: int,
) -> SmallIntArray:
    raw = np.asarray(value)
    if raw.shape != (n_samples, n_labels):
        raise PredictionArtifactError(
            f"targets must have shape {(n_samples, n_labels)}, received {raw.shape}"
        )
    try:
        finite = bool(np.all(np.isfinite(raw)))
        binary = bool(np.all((raw == 0) | (raw == 1)))
    except TypeError as exc:
        raise PredictionArtifactError("targets must be numeric binary values") from exc
    if not finite or not binary:
        raise PredictionArtifactError("targets must contain only finite binary values 0 and 1")
    return raw.astype(np.int8, copy=True)


def _validate_float_matrix(
    value: ArrayLike,
    *,
    name: str,
    n_samples: int,
    n_labels: int,
    probabilities: bool,
) -> FloatArray:
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise PredictionArtifactError(f"{name} must be a numeric matrix") from exc
    if matrix.shape != (n_samples, n_labels):
        raise PredictionArtifactError(
            f"{name} must have shape {(n_samples, n_labels)}, received {matrix.shape}"
        )
    if not np.all(np.isfinite(matrix)):
        raise PredictionArtifactError(f"{name} must contain only finite values")
    if probabilities and np.any((matrix < 0.0) | (matrix > 1.0)):
        raise PredictionArtifactError(f"{name} must lie in the closed interval [0, 1]")
    return matrix.copy()


def _validate_patient_fold_consistency(
    patient_ids: IdentifierArray,
    folds: SmallIntArray,
) -> None:
    assignments: dict[int | str, int] = {}
    for patient, fold_value in zip(_identifier_values(patient_ids), folds, strict=True):
        fold = int(fold_value)
        previous = assignments.setdefault(patient, fold)
        if previous != fold:
            raise PredictionArtifactError(
                f"patient_id {patient!r} occurs in multiple folds: {previous} and {fold}"
            )


def _identifier_values(values: IdentifierArray) -> tuple[int | str, ...]:
    if values.dtype.kind in {"i", "u"}:
        return tuple(int(value) for value in values)
    return tuple(str(value) for value in values)


def _identifier_payload(values: IdentifierArray) -> dict[str, object]:
    kind = "integer" if values.dtype.kind in {"i", "u"} else "string"
    return {"kind": kind, "values": list(_identifier_values(values))}


def _validate_fold_role(value: object) -> FoldRole:
    if isinstance(value, FoldRole):
        return value
    if not isinstance(value, str):
        raise PredictionArtifactError("fold_role must be a valid FoldRole string")
    try:
        return FoldRole(value)
    except ValueError as exc:
        raise PredictionArtifactError(f"unknown fold_role: {value!r}") from exc


def _normalize_sha256(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise PredictionArtifactError(f"{name} must be a SHA-256 string")
    match = _SHA256_PATTERN.fullmatch(value.strip())
    if match is None:
        raise PredictionArtifactError(
            f"{name} must be 64 hexadecimal characters, optionally prefixed by 'sha256:'"
        )
    return "sha256:" + match.group(1).lower()


def _validate_seed(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 2**32:
        raise PredictionArtifactError("model_seed must be an integer in [0, 2**32)")
    return value


def _validate_timestamp(value: object) -> str:
    if not isinstance(value, str) or _TIMESTAMP_PATTERN.fullmatch(value) is None:
        raise PredictionArtifactError(
            "created_at_utc must use canonical UTC format YYYY-MM-DDTHH:MM:SSZ"
        )
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    except ValueError as exc:
        raise PredictionArtifactError("created_at_utc is not a valid timestamp") from exc
    return value


def _nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PredictionArtifactError(f"{name} must be a non-empty string")
    return value.strip()


def _validate_string_mapping(
    value: Mapping[str, str],
    name: str,
) -> dict[str, str]:
    result: dict[str, str] = {}
    for key, item in value.items():
        normalized_key = _nonempty_string(key, f"{name} key")
        result[normalized_key] = _nonempty_string(item, f"{name}.{normalized_key}")
    if not result:
        raise PredictionArtifactError(f"{name} must not be empty")
    return dict(sorted(result.items()))


def _validate_extra_metadata(
    value: Mapping[str, MetadataScalar],
) -> dict[str, MetadataScalar]:
    result: dict[str, MetadataScalar] = {}
    for key, item in value.items():
        normalized_key = _nonempty_string(key, "extra_metadata key")
        if not isinstance(item, (str, int, float, bool)) and item is not None:
            raise PredictionArtifactError(
                f"extra_metadata.{normalized_key} must be a JSON scalar"
            )
        if isinstance(item, float) and not math.isfinite(item):
            raise PredictionArtifactError(
                f"extra_metadata.{normalized_key} must be finite"
            )
        result[normalized_key] = item
    return dict(sorted(result.items()))


def _stable_sigmoid(logits: FloatArray) -> FloatArray:
    output = np.empty_like(logits, dtype=np.float64)
    nonnegative = logits >= 0.0
    output[nonnegative] = 1.0 / (1.0 + np.exp(-logits[nonnegative]))
    exponent = np.exp(logits[~nonnegative])
    output[~nonnegative] = exponent / (1.0 + exponent)
    return output


def _readonly_copy[ArrayDType: np.generic](
    array: NDArray[ArrayDType],
) -> NDArray[ArrayDType]:
    result = np.asarray(array).copy()
    result.flags.writeable = False
    return result


def _expect_mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise PredictionArtifactError(f"{context} must be a string-keyed mapping")
    return cast(Mapping[str, object], value)


def _expect_keys(
    mapping: Mapping[str, object],
    *,
    required: set[str],
    context: str,
) -> None:
    missing = sorted(required.difference(mapping))
    unexpected = sorted(set(mapping).difference(required))
    if missing or unexpected:
        raise PredictionArtifactError(
            f"{context} has invalid keys; missing={missing}, unexpected={unexpected}"
        )


def _expect_string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PredictionArtifactError(f"{context} must be a non-empty string")
    return value


def _expect_integer(value: object, context: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise PredictionArtifactError(f"{context} must be an integer >= {minimum}")
    return value


def _expect_string_sequence(value: object, context: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise PredictionArtifactError(f"{context} must be a list of strings")
    return tuple(cast(list[str], value))


def _expect_integer_tuple(value: object, context: str) -> tuple[int, ...]:
    if not isinstance(value, list) or any(
        isinstance(item, bool) or not isinstance(item, int) for item in value
    ):
        raise PredictionArtifactError(f"{context} must be a list of integers")
    return tuple(cast(list[int], value))


def _expect_string_mapping(value: object, context: str) -> dict[str, str]:
    mapping = _expect_mapping(value, context)
    if not all(isinstance(item, str) for item in mapping.values()):
        raise PredictionArtifactError(f"{context} values must be strings")
    return cast(dict[str, str], dict(mapping))


def _expect_metadata_mapping(
    value: object,
    context: str,
) -> dict[str, MetadataScalar]:
    mapping = _expect_mapping(value, context)
    result: dict[str, MetadataScalar] = {}
    for key, item in mapping.items():
        if not isinstance(item, (str, int, float, bool)) and item is not None:
            raise PredictionArtifactError(f"{context}.{key} must be a JSON scalar")
        result[key] = item
    return result
