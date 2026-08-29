"""Immutable private embedding artifacts for OOD completion.

The archive deliberately contains row-level ECG and patient identifiers and is
therefore private research evidence.  Its canonical sidecar contains only
aggregate counts, provenance hashes, and logical array hashes; it never stores
filesystem paths or row-level values.
"""

from __future__ import annotations

import errno
import hashlib
import io
import json
import os
import re
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Self, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

EMBEDDING_ARTIFACT_SCHEMA_VERSION = 1
EMBEDDING_ARTIFACT_TYPE = "ecg_trust.ood_embedding_artifact"
EMBEDDING_DIMENSION = 512

_ARRAY_NAMES = ("ecg_id", "patient_id", "strat_fold", "embedding")
_HASH_FIELDS = (
    "checkpoint_sha256",
    "config_sha256",
    "normalization_sha256",
    "manifest_sha256",
    "protocol_sha256",
    "runtime_sha256",
)
_ROOT_KEYS = {
    "schema_version",
    "artifact_type",
    "visibility",
    "contains_row_level_identifiers",
    "role",
    "expected_folds",
    "record_count",
    "patient_count",
    "embedding_dimension",
    "alignment_sha256",
    "embedding_tensor_sha256",
    *_HASH_FIELDS,
    "arrays",
    "npz_file",
    "npz_size_bytes",
    "npz_sha256",
    "artifact_sha256",
}
_SHA256_RE = re.compile(r"(?:sha256:)?([0-9a-fA-F]{64})\Z")
_MAX_RECORDS = 100_000
_MAX_SIDECAR_BYTES = 64 * 1024
_MAX_NPZ_BYTES = 512 * 1024 * 1024
_MAX_NPY_HEADER_BYTES = 64 * 1024
_UNSUPPORTED_DIRECTORY_FSYNC_ERRNOS = frozenset(
    code
    for code in (
        errno.EBADF,
        errno.EINVAL,
        getattr(errno, "ENOTSUP", None),
        getattr(errno, "EOPNOTSUPP", None),
    )
    if code is not None
)

Int64Array = NDArray[np.int64]
Int8Array = NDArray[np.int8]
Float32Array = NDArray[np.float32]


class EmbeddingRole(StrEnum):
    """Private cohort role for one embedding archive."""

    REFERENCE = "REFERENCE"
    THRESHOLD_FIT = "THRESHOLD_FIT"
    SOURCE_VALIDATION = "SOURCE_VALIDATION"


class EmbeddingArtifactError(ValueError):
    """Raised when an embedding artifact violates its data contract."""


class EmbeddingArtifactIntegrityError(EmbeddingArtifactError):
    """Raised when persisted embedding evidence fails integrity checks."""


@dataclass(frozen=True, slots=True, init=False)
class EmbeddingArtifact:
    """Validated, sorted, read-only private embeddings and provenance."""

    ecg_id: Int64Array
    patient_id: Int64Array
    strat_fold: Int8Array
    embedding: Float32Array
    role: EmbeddingRole
    expected_folds: tuple[int, ...]
    checkpoint_sha256: str
    config_sha256: str
    normalization_sha256: str
    manifest_sha256: str
    protocol_sha256: str
    runtime_sha256: str
    alignment_sha256: str
    embedding_tensor_sha256: str
    artifact_sha256: str | None
    npz_file_sha256: str | None
    sidecar_file_sha256: str | None
    npz_size_bytes: int | None
    sidecar_size_bytes: int | None

    @classmethod
    def _create(
        cls,
        *,
        ecg_id: ArrayLike,
        patient_id: ArrayLike,
        strat_fold: ArrayLike,
        embedding: ArrayLike,
        role: EmbeddingRole | str,
        expected_folds: Sequence[int],
        checkpoint_sha256: str,
        config_sha256: str,
        normalization_sha256: str,
        manifest_sha256: str,
        protocol_sha256: str,
        runtime_sha256: str,
        canonicalize_order: bool,
        artifact_sha256: str | None,
        npz_file_sha256: str | None,
        sidecar_file_sha256: str | None,
        npz_size_bytes: int | None,
        sidecar_size_bytes: int | None,
    ) -> Self:
        normalized_role = _embedding_role(role)
        folds_expected = _expected_folds(expected_folds)
        ecg_ids, patient_ids, folds, embeddings = _validated_arrays(
            ecg_id=ecg_id,
            patient_id=patient_id,
            strat_fold=strat_fold,
            embedding=embedding,
            canonicalize_order=canonicalize_order,
        )
        observed_folds = tuple(int(value) for value in np.unique(folds))
        if observed_folds != folds_expected:
            raise EmbeddingArtifactError(
                f"observed folds {observed_folds!r} do not equal expected_folds "
                f"{folds_expected!r}"
            )
        hashes = {
            "checkpoint_sha256": _normalize_sha256(
                checkpoint_sha256, "checkpoint_sha256"
            ),
            "config_sha256": _normalize_sha256(config_sha256, "config_sha256"),
            "normalization_sha256": _normalize_sha256(
                normalization_sha256, "normalization_sha256"
            ),
            "manifest_sha256": _normalize_sha256(manifest_sha256, "manifest_sha256"),
            "protocol_sha256": _normalize_sha256(protocol_sha256, "protocol_sha256"),
            "runtime_sha256": _normalize_sha256(runtime_sha256, "runtime_sha256"),
        }
        instance = object.__new__(cls)
        object.__setattr__(instance, "ecg_id", ecg_ids)
        object.__setattr__(instance, "patient_id", patient_ids)
        object.__setattr__(instance, "strat_fold", folds)
        object.__setattr__(instance, "embedding", embeddings)
        object.__setattr__(instance, "role", normalized_role)
        object.__setattr__(instance, "expected_folds", folds_expected)
        for name, value in hashes.items():
            object.__setattr__(instance, name, value)
        object.__setattr__(
            instance,
            "alignment_sha256",
            _identity_alignment_sha256(ecg_ids, patient_ids, folds),
        )
        object.__setattr__(
            instance,
            "embedding_tensor_sha256",
            _embedding_tensor_sha256(embeddings),
        )
        object.__setattr__(
            instance,
            "artifact_sha256",
            None
            if artifact_sha256 is None
            else _normalize_sha256(artifact_sha256, "artifact_sha256"),
        )
        object.__setattr__(
            instance,
            "npz_file_sha256",
            None
            if npz_file_sha256 is None
            else _normalize_sha256(npz_file_sha256, "npz_file_sha256"),
        )
        object.__setattr__(
            instance,
            "sidecar_file_sha256",
            None
            if sidecar_file_sha256 is None
            else _normalize_sha256(sidecar_file_sha256, "sidecar_file_sha256"),
        )
        for name, size_value in (
            ("npz_size_bytes", npz_size_bytes),
            ("sidecar_size_bytes", sidecar_size_bytes),
        ):
            if size_value is not None and (
                isinstance(size_value, bool)
                or not isinstance(size_value, int)
                or size_value <= 0
            ):
                raise EmbeddingArtifactError(f"{name} must be a positive integer when present")
            object.__setattr__(instance, name, size_value)
        return instance

    @property
    def record_count(self) -> int:
        return int(self.ecg_id.shape[0])

    @property
    def patient_count(self) -> int:
        return int(np.unique(self.patient_id).shape[0])

    def to_summary_dict(self) -> dict[str, object]:
        """Return identifier-free metadata suitable for downstream binding."""

        return {
            "schema_version": EMBEDDING_ARTIFACT_SCHEMA_VERSION,
            "artifact_type": EMBEDDING_ARTIFACT_TYPE,
            "visibility": "PRIVATE",
            "contains_row_level_identifiers": True,
            "role": self.role.value,
            "expected_folds": list(self.expected_folds),
            "record_count": self.record_count,
            "patient_count": self.patient_count,
            "embedding_dimension": EMBEDDING_DIMENSION,
            "alignment_sha256": self.alignment_sha256,
            "embedding_tensor_sha256": self.embedding_tensor_sha256,
            **{name: getattr(self, name) for name in _HASH_FIELDS},
            "artifact_sha256": self.artifact_sha256,
            "npz_file_sha256": self.npz_file_sha256,
            "sidecar_file_sha256": self.sidecar_file_sha256,
        }

    def _arrays(self) -> dict[str, NDArray[np.generic]]:
        return {
            "ecg_id": self.ecg_id,
            "patient_id": self.patient_id,
            "strat_fold": self.strat_fold,
            "embedding": self.embedding,
        }


@dataclass(frozen=True, slots=True)
class EmbeddingArtifactFiles:
    """Paths and physical/logical identities committed by a save."""

    npz_path: Path
    json_path: Path
    artifact_sha256: str
    npz_file_sha256: str
    sidecar_file_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "npz_path": str(self.npz_path),
            "json_path": str(self.json_path),
            "artifact_sha256": self.artifact_sha256,
            "npz_file_sha256": self.npz_file_sha256,
            "sidecar_file_sha256": self.sidecar_file_sha256,
        }


def create_embedding_artifact(
    *,
    ecg_id: ArrayLike,
    patient_id: ArrayLike,
    strat_fold: ArrayLike,
    embedding: ArrayLike,
    role: EmbeddingRole | str,
    expected_folds: Sequence[int],
    checkpoint_sha256: str,
    config_sha256: str,
    normalization_sha256: str,
    manifest_sha256: str,
    protocol_sha256: str,
    runtime_sha256: str,
) -> EmbeddingArtifact:
    """Create a canonical in-memory private embedding artifact."""

    return EmbeddingArtifact._create(
        ecg_id=ecg_id,
        patient_id=patient_id,
        strat_fold=strat_fold,
        embedding=embedding,
        role=role,
        expected_folds=expected_folds,
        checkpoint_sha256=checkpoint_sha256,
        config_sha256=config_sha256,
        normalization_sha256=normalization_sha256,
        manifest_sha256=manifest_sha256,
        protocol_sha256=protocol_sha256,
        runtime_sha256=runtime_sha256,
        canonicalize_order=True,
        artifact_sha256=None,
        npz_file_sha256=None,
        sidecar_file_sha256=None,
        npz_size_bytes=None,
        sidecar_size_bytes=None,
    )


def save_embedding_artifact(
    artifact: EmbeddingArtifact,
    path: str | Path,
) -> EmbeddingArtifactFiles:
    """Atomically commit a new NPZ/canonical-JSON pair without overwriting."""

    if not isinstance(artifact, EmbeddingArtifact):
        raise TypeError("artifact must be an EmbeddingArtifact")
    normalized = EmbeddingArtifact._create(
        ecg_id=artifact.ecg_id,
        patient_id=artifact.patient_id,
        strat_fold=artifact.strat_fold,
        embedding=artifact.embedding,
        role=artifact.role,
        expected_folds=artifact.expected_folds,
        checkpoint_sha256=artifact.checkpoint_sha256,
        config_sha256=artifact.config_sha256,
        normalization_sha256=artifact.normalization_sha256,
        manifest_sha256=artifact.manifest_sha256,
        protocol_sha256=artifact.protocol_sha256,
        runtime_sha256=artifact.runtime_sha256,
        canonicalize_order=False,
        artifact_sha256=None,
        npz_file_sha256=None,
        sidecar_file_sha256=None,
        npz_size_bytes=None,
        sidecar_size_bytes=None,
    )
    npz_path, json_path = _artifact_paths(path, for_save=True)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    _assert_destinations_absent(npz_path, json_path)

    npz_temp = _temporary_path(npz_path)
    json_temp = _temporary_path(json_path)
    npz_committed = False
    json_committed = False
    try:
        _write_npz(npz_temp, normalized._arrays())
        npz_size = npz_temp.stat().st_size
        if not 0 < npz_size <= _MAX_NPZ_BYTES:
            raise EmbeddingArtifactError("embedding NPZ size is outside the supported bound")
        npz_sha256 = _sha256_file(npz_temp)
        body = _storage_body(
            normalized,
            npz_file=npz_path.name,
            npz_size_bytes=npz_size,
            npz_sha256=npz_sha256,
        )
        artifact_sha256 = _canonical_sha256(body)
        sidecar = {**body, "artifact_sha256": artifact_sha256}
        _write_canonical_json(json_temp, sidecar)

        _link_new(npz_temp, npz_path)
        npz_committed = True
        _fsync_directory(npz_path.parent)
        _link_new(json_temp, json_path)
        json_committed = True
        _fsync_directory(json_path.parent)
        loaded = load_embedding_artifact(
            npz_path,
            expected_artifact_sha256=artifact_sha256,
            expected_npz_file_sha256=npz_sha256,
        )
        if loaded.alignment_sha256 != normalized.alignment_sha256:
            raise EmbeddingArtifactIntegrityError("saved embedding alignment changed")
        sidecar_sha256 = _sha256_file(json_path)
    except Exception:
        if json_committed:
            _unlink_if_same_file(json_path, json_temp)
        if npz_committed:
            _unlink_if_same_file(npz_path, npz_temp)
        _unlink_best_effort(npz_temp)
        _unlink_best_effort(json_temp)
        raise
    else:
        _unlink_strict(npz_temp)
        _unlink_strict(json_temp)

    return EmbeddingArtifactFiles(
        npz_path=npz_path,
        json_path=json_path,
        artifact_sha256=artifact_sha256,
        npz_file_sha256=npz_sha256,
        sidecar_file_sha256=sidecar_sha256,
    )


def load_embedding_artifact(
    path: str | Path,
    *,
    expected_artifact_sha256: str | None = None,
    expected_npz_file_sha256: str | None = None,
    expected_role: EmbeddingRole | str | None = None,
) -> EmbeddingArtifact:
    """Load private embeddings only after strict schema and integrity verification."""

    npz_path, json_path = _artifact_paths(path, for_save=False)
    root, sidecar_bytes = _load_canonical_sidecar(json_path)
    if set(root) != _ROOT_KEYS:
        raise EmbeddingArtifactIntegrityError("embedding sidecar keys are not canonical")
    if root["schema_version"] != EMBEDDING_ARTIFACT_SCHEMA_VERSION:
        raise EmbeddingArtifactIntegrityError("unsupported embedding artifact schema version")
    if root["artifact_type"] != EMBEDDING_ARTIFACT_TYPE:
        raise EmbeddingArtifactIntegrityError("unexpected embedding artifact type")
    if root["visibility"] != "PRIVATE" or root["contains_row_level_identifiers"] is not True:
        raise EmbeddingArtifactIntegrityError("embedding artifact privacy declaration is invalid")

    stored_artifact_sha256 = _normalize_sha256(
        root["artifact_sha256"], "artifact_sha256", integrity=True
    )
    body = dict(root)
    del body["artifact_sha256"]
    if _canonical_sha256(body) != stored_artifact_sha256:
        raise EmbeddingArtifactIntegrityError("embedding sidecar self-hash mismatch")
    if expected_artifact_sha256 is not None and stored_artifact_sha256 != _normalize_sha256(
        expected_artifact_sha256, "expected_artifact_sha256"
    ):
        raise EmbeddingArtifactIntegrityError("embedding artifact hash differs from expectation")

    role = _embedding_role(root["role"], integrity=True)
    if expected_role is not None and role is not _embedding_role(expected_role):
        raise EmbeddingArtifactIntegrityError("embedding role differs from expectation")
    folds_expected = _expected_folds(root["expected_folds"], integrity=True)
    record_count = _positive_integer(
        root["record_count"], "record_count", maximum=_MAX_RECORDS, integrity=True
    )
    patient_count = _positive_integer(
        root["patient_count"], "patient_count", maximum=record_count, integrity=True
    )
    if root["embedding_dimension"] != EMBEDDING_DIMENSION:
        raise EmbeddingArtifactIntegrityError("embedding_dimension must be exactly 512")
    descriptors = _array_descriptors(record_count)
    if root["arrays"] != descriptors:
        raise EmbeddingArtifactIntegrityError("embedding array descriptors are not canonical")
    stored_alignment = _normalize_sha256(
        root["alignment_sha256"], "alignment_sha256", integrity=True
    )
    stored_tensor_hash = _normalize_sha256(
        root["embedding_tensor_sha256"], "embedding_tensor_sha256", integrity=True
    )
    bound_hashes = {
        name: _normalize_sha256(root[name], name, integrity=True) for name in _HASH_FIELDS
    }

    npz_name = _basename(root["npz_file"], "npz_file")
    if npz_name != npz_path.name:
        raise EmbeddingArtifactIntegrityError("sidecar NPZ filename does not match archive")
    if npz_path.is_symlink() or not npz_path.is_file():
        raise EmbeddingArtifactIntegrityError("embedding NPZ is missing or symbolic")
    npz_size = _positive_integer(
        root["npz_size_bytes"],
        "npz_size_bytes",
        maximum=_MAX_NPZ_BYTES,
        integrity=True,
    )
    npz_snapshot = _read_bounded_snapshot(
        npz_path,
        maximum_bytes=_MAX_NPZ_BYTES,
        context="embedding NPZ",
    )
    if len(npz_snapshot) != npz_size:
        raise EmbeddingArtifactIntegrityError("embedding NPZ size mismatch")
    npz_sha256 = _normalize_sha256(root["npz_sha256"], "npz_sha256", integrity=True)
    if expected_npz_file_sha256 is not None and npz_sha256 != _normalize_sha256(
        expected_npz_file_sha256, "expected_npz_file_sha256"
    ):
        raise EmbeddingArtifactIntegrityError("embedding NPZ hash differs from expectation")
    if _sha256_bytes(npz_snapshot) != npz_sha256:
        raise EmbeddingArtifactIntegrityError("embedding NPZ SHA-256 mismatch")

    _preflight_npz(npz_snapshot, record_count=record_count)
    arrays = _load_npz(npz_snapshot)
    try:
        artifact = EmbeddingArtifact._create(
            ecg_id=arrays["ecg_id"],
            patient_id=arrays["patient_id"],
            strat_fold=arrays["strat_fold"],
            embedding=arrays["embedding"],
            role=role,
            expected_folds=folds_expected,
            checkpoint_sha256=bound_hashes["checkpoint_sha256"],
            config_sha256=bound_hashes["config_sha256"],
            normalization_sha256=bound_hashes["normalization_sha256"],
            manifest_sha256=bound_hashes["manifest_sha256"],
            protocol_sha256=bound_hashes["protocol_sha256"],
            runtime_sha256=bound_hashes["runtime_sha256"],
            canonicalize_order=False,
            artifact_sha256=stored_artifact_sha256,
            npz_file_sha256=npz_sha256,
            sidecar_file_sha256=_sha256_bytes(sidecar_bytes),
            npz_size_bytes=len(npz_snapshot),
            sidecar_size_bytes=len(sidecar_bytes),
        )
    except EmbeddingArtifactIntegrityError:
        raise
    except EmbeddingArtifactError as error:
        raise EmbeddingArtifactIntegrityError(
            f"stored embedding arrays violate their contract: {error}"
        ) from error
    if artifact.record_count != record_count or artifact.patient_count != patient_count:
        raise EmbeddingArtifactIntegrityError("embedding aggregate counts do not match arrays")
    if artifact.alignment_sha256 != stored_alignment:
        raise EmbeddingArtifactIntegrityError("embedding identity alignment hash mismatch")
    if artifact.embedding_tensor_sha256 != stored_tensor_hash:
        raise EmbeddingArtifactIntegrityError("embedding tensor hash mismatch")
    return artifact


def _validated_arrays(
    *,
    ecg_id: ArrayLike,
    patient_id: ArrayLike,
    strat_fold: ArrayLike,
    embedding: ArrayLike,
    canonicalize_order: bool,
) -> tuple[Int64Array, Int64Array, Int8Array, Float32Array]:
    ecg_ids = _integer_array(ecg_id, "ecg_id", np.int64)
    if not 0 < ecg_ids.shape[0] <= _MAX_RECORDS:
        raise EmbeddingArtifactError(
            f"embedding artifacts require 1..{_MAX_RECORDS} records"
        )
    patient_ids = _integer_array(patient_id, "patient_id", np.int64)
    if patient_ids.shape != ecg_ids.shape:
        raise EmbeddingArtifactError("patient_id must align one-to-one with ecg_id")
    if np.unique(ecg_ids).shape[0] != ecg_ids.shape[0]:
        raise EmbeddingArtifactError("ecg_id values must be unique")
    folds = _fold_array(strat_fold, record_count=int(ecg_ids.shape[0]))
    embeddings = _embedding_array(embedding, record_count=int(ecg_ids.shape[0]))
    _validate_patient_fold_consistency(patient_ids, folds)

    if canonicalize_order:
        order = np.argsort(ecg_ids, kind="stable")
        ecg_ids = ecg_ids[order]
        patient_ids = patient_ids[order]
        folds = folds[order]
        embeddings = embeddings[order]
    elif not bool(np.all(ecg_ids[1:] > ecg_ids[:-1])):
        raise EmbeddingArtifactIntegrityError("stored ecg_id values must be strictly sorted")
    return (
        cast(Int64Array, _readonly_copy(ecg_ids, np.int64)),
        cast(Int64Array, _readonly_copy(patient_ids, np.int64)),
        cast(Int8Array, _readonly_copy(folds, np.int8)),
        cast(Float32Array, _readonly_copy(embeddings, np.float32)),
    )


def _integer_array(value: ArrayLike, name: str, dtype: type[np.signedinteger[Any]]) -> Int64Array:
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.dtype.kind not in {"i", "u"}:
        raise EmbeddingArtifactError(f"{name} must be a one-dimensional integer array")
    if raw.dtype.kind == "u" and raw.size and int(raw.max()) > np.iinfo(np.int64).max:
        raise EmbeddingArtifactError(f"{name} contains a value outside int64 range")
    try:
        return cast(Int64Array, raw.astype(dtype, copy=True))
    except (OverflowError, TypeError, ValueError) as error:
        raise EmbeddingArtifactError(f"{name} could not be represented as int64") from error


def _fold_array(value: ArrayLike, *, record_count: int) -> Int8Array:
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.shape[0] != record_count or raw.dtype.kind not in {"i", "u"}:
        raise EmbeddingArtifactError("strat_fold must be one integer per record")
    if np.any((raw < 1) | (raw > 10)):
        raise EmbeddingArtifactError("strat_fold values must be between 1 and 10")
    return cast(Int8Array, raw.astype(np.int8, copy=True))


def _embedding_array(value: ArrayLike, *, record_count: int) -> Float32Array:
    raw = np.asarray(value)
    if raw.shape != (record_count, EMBEDDING_DIMENSION):
        raise EmbeddingArtifactError(
            f"embedding must have shape {(record_count, EMBEDDING_DIMENSION)!r}"
        )
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            matrix = raw.astype(np.float32, copy=True)
    except (TypeError, ValueError) as error:
        raise EmbeddingArtifactError("embedding must be a numeric float32 matrix") from error
    if not np.all(np.isfinite(matrix)):
        raise EmbeddingArtifactError("embedding must contain only finite values")
    return matrix


def _validate_patient_fold_consistency(patient_id: Int64Array, strat_fold: Int8Array) -> None:
    by_patient: dict[int, int] = {}
    for patient, fold in zip(patient_id.tolist(), strat_fold.tolist(), strict=True):
        prior = by_patient.setdefault(patient, fold)
        if prior != fold:
            raise EmbeddingArtifactError("a patient occurs in multiple embedding folds")


def _readonly_copy(
    value: NDArray[np.generic], dtype: type[np.generic]
) -> NDArray[np.generic]:
    normalized_dtype = np.dtype(dtype)
    contiguous = np.ascontiguousarray(value, dtype=normalized_dtype)
    payload = contiguous.tobytes(order="C")
    return np.frombuffer(payload, dtype=normalized_dtype).reshape(contiguous.shape)


def _expected_folds(
    value: object, *, integrity: bool = False
) -> tuple[int, ...]:
    error_type = EmbeddingArtifactIntegrityError if integrity else EmbeddingArtifactError
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise error_type("expected_folds must be a non-empty integer sequence")
    folds: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, np.integer)):
            raise error_type("expected_folds must contain integers")
        fold = int(item)
        if not 1 <= fold <= 10:
            raise error_type("expected_folds values must be between 1 and 10")
        folds.append(fold)
    result = tuple(folds)
    if not result or result != tuple(sorted(set(result))):
        raise error_type("expected_folds must be non-empty, unique, and sorted")
    return result


def _embedding_role(value: object, *, integrity: bool = False) -> EmbeddingRole:
    error_type = EmbeddingArtifactIntegrityError if integrity else EmbeddingArtifactError
    if not isinstance(value, str):
        raise error_type("unsupported embedding role")
    try:
        return EmbeddingRole(value)
    except (TypeError, ValueError) as error:
        raise error_type("unsupported embedding role") from error


def _identity_alignment_sha256(
    ecg_id: Int64Array, patient_id: Int64Array, strat_fold: Int8Array
) -> str:
    return _canonical_sha256(
        {
            "schema_version": 1,
            "ecg_id": ecg_id.tolist(),
            "patient_id": patient_id.tolist(),
            "strat_fold": strat_fold.astype(int).tolist(),
        }
    )


def _embedding_tensor_sha256(embedding: Float32Array) -> str:
    descriptor = _canonical_json(
        {"dtype": "float32", "shape": list(embedding.shape)}
    ).encode("utf-8")
    little_endian = np.ascontiguousarray(embedding, dtype=np.dtype("<f4"))
    digest = hashlib.sha256()
    digest.update(descriptor)
    digest.update(b"\n")
    digest.update(little_endian.tobytes(order="C"))
    return "sha256:" + digest.hexdigest()


def _storage_body(
    artifact: EmbeddingArtifact,
    *,
    npz_file: str,
    npz_size_bytes: int,
    npz_sha256: str,
) -> dict[str, object]:
    return {
        "schema_version": EMBEDDING_ARTIFACT_SCHEMA_VERSION,
        "artifact_type": EMBEDDING_ARTIFACT_TYPE,
        "visibility": "PRIVATE",
        "contains_row_level_identifiers": True,
        "role": artifact.role.value,
        "expected_folds": list(artifact.expected_folds),
        "record_count": artifact.record_count,
        "patient_count": artifact.patient_count,
        "embedding_dimension": EMBEDDING_DIMENSION,
        "alignment_sha256": artifact.alignment_sha256,
        "embedding_tensor_sha256": artifact.embedding_tensor_sha256,
        **{name: getattr(artifact, name) for name in _HASH_FIELDS},
        "arrays": _array_descriptors(artifact.record_count),
        "npz_file": _basename(npz_file, "npz_file"),
        "npz_size_bytes": npz_size_bytes,
        "npz_sha256": _normalize_sha256(npz_sha256, "npz_sha256"),
    }


def _array_descriptors(record_count: int) -> dict[str, object]:
    return {
        "ecg_id": {"dtype": "int64", "shape": [record_count]},
        "patient_id": {"dtype": "int64", "shape": [record_count]},
        "strat_fold": {"dtype": "int8", "shape": [record_count]},
        "embedding": {
            "dtype": "float32",
            "shape": [record_count, EMBEDDING_DIMENSION],
        },
    }


def _artifact_paths(path: str | Path, *, for_save: bool) -> tuple[Path, Path]:
    candidate = Path(path)
    suffix = candidate.suffix.casefold()
    if for_save and suffix != ".npz":
        raise EmbeddingArtifactError("embedding artifact save path must end in .npz")
    if suffix == ".npz":
        return candidate, candidate.with_suffix(".json")
    if not for_save and suffix == ".json":
        return candidate.with_suffix(".npz"), candidate
    raise EmbeddingArtifactError("embedding artifact path must end in .npz or .json")


def _assert_destinations_absent(npz_path: Path, json_path: Path) -> None:
    existing = [
        path for path in (npz_path, json_path) if path.exists() or path.is_symlink()
    ]
    if existing:
        raise FileExistsError(
            "embedding artifact is immutable; refusing to overwrite "
            + ", ".join(str(path) for path in existing)
        )


def _temporary_path(destination: Path) -> Path:
    descriptor, raw = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    return Path(raw)


def _write_npz(path: Path, arrays: Mapping[str, NDArray[np.generic]]) -> None:
    with path.open("wb") as handle:
        np.savez_compressed(
            handle,
            ecg_id=arrays["ecg_id"],
            patient_id=arrays["patient_id"],
            strat_fold=arrays["strat_fold"],
            embedding=arrays["embedding"],
        )
        handle.flush()
        os.fsync(handle.fileno())


def _write_canonical_json(path: Path, value: Mapping[str, object]) -> None:
    payload = (_canonical_json(value) + "\n").encode("utf-8")
    if len(payload) > _MAX_SIDECAR_BYTES:
        raise EmbeddingArtifactError("embedding sidecar exceeds its size bound")
    with path.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _link_new(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except FileExistsError as error:
        raise FileExistsError(
            f"embedding artifact is immutable; refusing to overwrite {destination}"
        ) from error
    except OSError as error:
        raise EmbeddingArtifactError(
            f"could not atomically commit embedding artifact {destination}"
        ) from error


def _fsync_directory(path: Path) -> None:
    """Persist directory-entry changes where the host supports directory fsync."""

    if os.name == "nt":
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        if error.errno in _UNSUPPORTED_DIRECTORY_FSYNC_ERRNOS:
            return
        raise EmbeddingArtifactError("could not open embedding directory for sync") from error
    try:
        os.fsync(descriptor)
    except OSError as error:
        if error.errno not in _UNSUPPORTED_DIRECTORY_FSYNC_ERRNOS:
            raise EmbeddingArtifactError("could not sync embedding directory") from error
    finally:
        os.close(descriptor)


def _unlink_if_same_file(path: Path, temporary: Path) -> None:
    removed = False
    try:
        if path.samefile(temporary):
            path.unlink()
            removed = True
    except OSError:
        pass
    if removed:
        with suppress(EmbeddingArtifactError):
            _fsync_directory(path.parent)


def _unlink_strict(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return
    except OSError as error:
        raise EmbeddingArtifactError("could not remove embedding temporary file") from error
    _fsync_directory(path.parent)


def _unlink_best_effort(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return
    except OSError:
        return
    else:
        with suppress(EmbeddingArtifactError):
            _fsync_directory(path.parent)


def _load_canonical_sidecar(path: Path) -> tuple[Mapping[str, object], bytes]:
    if path.is_symlink() or not path.is_file():
        raise EmbeddingArtifactIntegrityError("embedding sidecar is missing or symbolic")
    try:
        raw = _read_bounded_snapshot(
            path,
            maximum_bytes=_MAX_SIDECAR_BYTES,
            context="embedding sidecar",
        )
        decoded: object = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise EmbeddingArtifactIntegrityError(
            f"could not decode embedding sidecar: {error}"
        ) from error
    root = _mapping(decoded, "embedding sidecar")
    try:
        canonical = (_canonical_json(root) + "\n").encode("utf-8")
    except EmbeddingArtifactError as error:
        raise EmbeddingArtifactIntegrityError("embedding sidecar is not finite JSON") from error
    if raw != canonical:
        raise EmbeddingArtifactIntegrityError("embedding sidecar bytes are not canonical JSON")
    return root, raw


def _preflight_npz(snapshot: bytes, *, record_count: int) -> None:
    expected_bytes = {
        "ecg_id.npy": record_count * np.dtype(np.int64).itemsize,
        "patient_id.npy": record_count * np.dtype(np.int64).itemsize,
        "strat_fold.npy": record_count * np.dtype(np.int8).itemsize,
        "embedding.npy": record_count * EMBEDDING_DIMENSION * np.dtype(np.float32).itemsize,
    }
    try:
        with zipfile.ZipFile(io.BytesIO(snapshot), mode="r") as archive:
            members = archive.infolist()
            if archive.comment or len(members) != len(expected_bytes):
                raise EmbeddingArtifactIntegrityError("embedding NPZ inventory is invalid")
            if {member.filename for member in members} != set(expected_bytes):
                raise EmbeddingArtifactIntegrityError("embedding NPZ member names are invalid")
            for member in members:
                payload_bytes = expected_bytes[member.filename]
                if member.flag_bits & 0x1:
                    raise EmbeddingArtifactIntegrityError("embedding NPZ must not be encrypted")
                if not payload_bytes < member.file_size <= payload_bytes + _MAX_NPY_HEADER_BYTES:
                    raise EmbeddingArtifactIntegrityError(
                        "embedding NPZ member size exceeds its bounded array contract"
                    )
    except EmbeddingArtifactIntegrityError:
        raise
    except (OSError, zipfile.BadZipFile) as error:
        raise EmbeddingArtifactIntegrityError(
            f"could not inspect embedding NPZ: {error}"
        ) from error


def _load_npz(snapshot: bytes) -> dict[str, NDArray[np.generic]]:
    try:
        with np.load(io.BytesIO(snapshot), allow_pickle=False) as archive:
            if tuple(archive.files) != _ARRAY_NAMES:
                raise EmbeddingArtifactIntegrityError(
                    "embedding NPZ array names or order are invalid"
                )
            arrays = {name: np.asarray(archive[name]).copy() for name in _ARRAY_NAMES}
    except EmbeddingArtifactIntegrityError:
        raise
    except (OSError, EOFError, ValueError, KeyError, zipfile.BadZipFile) as error:
        raise EmbeddingArtifactIntegrityError(f"could not load embedding NPZ: {error}") from error
    expected_dtypes = {
        "ecg_id": np.dtype(np.int64),
        "patient_id": np.dtype(np.int64),
        "strat_fold": np.dtype(np.int8),
        "embedding": np.dtype(np.float32),
    }
    for name, dtype in expected_dtypes.items():
        if arrays[name].dtype != dtype:
            raise EmbeddingArtifactIntegrityError(f"embedding NPZ {name} dtype is invalid")
    return arrays


def _read_bounded_snapshot(path: Path, *, maximum_bytes: int, context: str) -> bytes:
    """Read one bounded immutable snapshot for all later verification and decoding."""

    try:
        with path.open("rb") as handle:
            snapshot = handle.read(maximum_bytes + 1)
    except OSError as error:
        raise EmbeddingArtifactIntegrityError(f"could not read {context}") from error
    if not snapshot or len(snapshot) > maximum_bytes:
        raise EmbeddingArtifactIntegrityError(f"{context} size is invalid")
    return snapshot


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise EmbeddingArtifactIntegrityError(f"{context} must be a string-keyed object")
    return cast(Mapping[str, object], value)


def _positive_integer(
    value: object,
    context: str,
    *,
    maximum: int,
    integrity: bool,
) -> int:
    error_type = EmbeddingArtifactIntegrityError if integrity else EmbeddingArtifactError
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise error_type(f"{context} must be an integer in [1, {maximum}]")
    return value


def _basename(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or ":" in value
        or "\x00" in value
        or Path(value).name != value
    ):
        raise EmbeddingArtifactIntegrityError(f"{context} must be a path-free filename")
    return value


def _normalize_sha256(value: object, context: str, *, integrity: bool = False) -> str:
    error_type = EmbeddingArtifactIntegrityError if integrity else EmbeddingArtifactError
    if not isinstance(value, str):
        raise error_type(f"{context} must be a SHA-256 digest")
    match = _SHA256_RE.fullmatch(value)
    if match is None:
        raise error_type(f"{context} must be a SHA-256 digest")
    return "sha256:" + match.group(1).lower()


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise EmbeddingArtifactError("artifact content must be finite canonical JSON") from error


def _canonical_sha256(value: object) -> str:
    return _sha256_bytes(_canonical_json(value).encode("utf-8"))


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


__all__ = [
    "EMBEDDING_ARTIFACT_SCHEMA_VERSION",
    "EMBEDDING_ARTIFACT_TYPE",
    "EMBEDDING_DIMENSION",
    "EmbeddingArtifact",
    "EmbeddingArtifactError",
    "EmbeddingArtifactFiles",
    "EmbeddingArtifactIntegrityError",
    "EmbeddingRole",
    "create_embedding_artifact",
    "load_embedding_artifact",
    "save_embedding_artifact",
]
