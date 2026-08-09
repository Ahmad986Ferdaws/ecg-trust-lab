"""Immutable, integrity-bound NPZ/JSON artifacts for post-evaluation audits."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

AUDIT_ARRAY_SCHEMA_VERSION = 1
_ROOT_KEYS = {
    "schema_version",
    "artifact_type",
    "npz_file",
    "npz_size_bytes",
    "npz_sha256",
    "arrays",
    "metadata",
    "artifact_sha256",
}


class AuditArtifactError(ValueError):
    """Raised when an audit artifact violates its storage contract."""


class AuditArtifactIntegrityError(AuditArtifactError):
    """Raised when a stored audit artifact fails integrity verification."""


@dataclass(frozen=True, slots=True)
class AuditArrayArtifact:
    """Verified immutable arrays and metadata."""

    npz_path: Path
    json_path: Path
    artifact_type: str
    artifact_sha256: str
    npz_sha256: str
    arrays: Mapping[str, NDArray[np.generic]]
    metadata: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class AuditArrayFiles:
    """Committed file identities returned by a save operation."""

    npz_path: Path
    json_path: Path
    artifact_sha256: str
    npz_sha256: str
    json_file_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "npz_path": str(self.npz_path),
            "json_path": str(self.json_path),
            "artifact_sha256": self.artifact_sha256,
            "npz_sha256": self.npz_sha256,
            "json_file_sha256": self.json_file_sha256,
        }


def save_audit_array_artifact(
    path: str | Path,
    *,
    artifact_type: str,
    arrays: Mapping[str, NDArray[np.generic]],
    metadata: Mapping[str, object],
) -> AuditArrayFiles:
    """Atomically save a new self-hashed NPZ/JSON pair without overwriting."""

    npz_path, json_path = _paths(path)
    kind = _artifact_type(artifact_type)
    validated_arrays = _validated_arrays(arrays)
    normalized_metadata = _finite_json_mapping(metadata, "metadata")
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    existing = [candidate for candidate in (npz_path, json_path) if candidate.exists()]
    if existing:
        raise FileExistsError(
            "audit artifact is immutable; refusing to overwrite "
            + ", ".join(str(candidate) for candidate in existing)
        )

    npz_temp = _temporary_path(npz_path)
    json_temp = _temporary_path(json_path)
    npz_committed = False
    try:
        _write_npz(npz_temp, validated_arrays)
        npz_sha256 = _sha256_file(npz_temp)
        body: dict[str, object] = {
            "schema_version": AUDIT_ARRAY_SCHEMA_VERSION,
            "artifact_type": kind,
            "npz_file": npz_path.name,
            "npz_size_bytes": npz_temp.stat().st_size,
            "npz_sha256": npz_sha256,
            "arrays": {
                name: {"shape": list(array.shape), "dtype": array.dtype.str}
                for name, array in validated_arrays.items()
            },
            "metadata": normalized_metadata,
        }
        artifact_sha256 = _canonical_sha256(body)
        sidecar = {**body, "artifact_sha256": artifact_sha256}
        _write_json(json_temp, sidecar)
        os.replace(npz_temp, npz_path)
        npz_committed = True
        os.replace(json_temp, json_path)
    except Exception:
        _unlink(npz_temp)
        _unlink(json_temp)
        if npz_committed and not json_path.exists():
            _unlink(npz_path)
        raise
    return AuditArrayFiles(
        npz_path=npz_path,
        json_path=json_path,
        artifact_sha256=artifact_sha256,
        npz_sha256=npz_sha256,
        json_file_sha256=_sha256_file(json_path),
    )


def load_audit_array_artifact(
    path: str | Path,
    *,
    expected_artifact_type: str | None = None,
    expected_metadata: Mapping[str, object] | None = None,
) -> AuditArrayArtifact:
    """Load a pair only after verifying every stored identity and array."""

    npz_path, json_path = _paths(path)
    try:
        decoded: object = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AuditArtifactIntegrityError(f"could not decode audit sidecar: {error}") from error
    root = _mapping(decoded, "audit sidecar")
    if set(root) != _ROOT_KEYS:
        raise AuditArtifactIntegrityError("audit sidecar keys are not canonical")
    if root["schema_version"] != AUDIT_ARRAY_SCHEMA_VERSION:
        raise AuditArtifactIntegrityError("unsupported audit array schema version")
    artifact_type = _artifact_type(root["artifact_type"])
    if expected_artifact_type is not None and artifact_type != _artifact_type(
        expected_artifact_type
    ):
        raise AuditArtifactIntegrityError("audit artifact type differs from expectation")
    stored_artifact_sha256 = _hash(root["artifact_sha256"], "artifact_sha256")
    body = dict(root)
    del body["artifact_sha256"]
    if _canonical_sha256(body) != stored_artifact_sha256:
        raise AuditArtifactIntegrityError("audit sidecar self-hash mismatch")
    if root["npz_file"] != npz_path.name:
        raise AuditArtifactIntegrityError("audit sidecar NPZ filename mismatch")
    if not npz_path.is_file():
        raise AuditArtifactIntegrityError("audit NPZ is missing")
    size = root["npz_size_bytes"]
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise AuditArtifactIntegrityError("audit NPZ size is invalid")
    if npz_path.stat().st_size != size:
        raise AuditArtifactIntegrityError("audit NPZ size mismatch")
    npz_sha256 = _hash(root["npz_sha256"], "npz_sha256")
    if _sha256_file(npz_path) != npz_sha256:
        raise AuditArtifactIntegrityError("audit NPZ SHA-256 mismatch")
    metadata = _finite_json_mapping(_mapping(root["metadata"], "metadata"), "metadata")
    if expected_metadata is not None and metadata != _finite_json_mapping(
        expected_metadata, "expected_metadata"
    ):
        raise AuditArtifactIntegrityError("audit metadata differs from expectation")
    signatures = _mapping(root["arrays"], "arrays")
    arrays = _load_npz(npz_path, signatures)
    return AuditArrayArtifact(
        npz_path=npz_path,
        json_path=json_path,
        artifact_type=artifact_type,
        artifact_sha256=stored_artifact_sha256,
        npz_sha256=npz_sha256,
        arrays=MappingProxyType(arrays),
        metadata=MappingProxyType(metadata),
    )


def _paths(path: str | Path) -> tuple[Path, Path]:
    source = Path(path)
    npz_path = source if source.suffix.casefold() == ".npz" else source.with_suffix(".npz")
    return npz_path.resolve(), npz_path.with_suffix(".json").resolve()


def _artifact_type(value: object) -> str:
    if not isinstance(value, str) or not value.strip() or not value.startswith("ecg_trust."):
        raise AuditArtifactError("artifact_type must be a non-empty ecg_trust.* string")
    return value


def _validated_arrays(
    values: Mapping[str, NDArray[np.generic]],
) -> dict[str, NDArray[np.generic]]:
    if not isinstance(values, Mapping) or not values:
        raise AuditArtifactError("arrays must be a non-empty mapping")
    result: dict[str, NDArray[np.generic]] = {}
    for name in sorted(values):
        if not isinstance(name, str) or not name or not name.isidentifier():
            raise AuditArtifactError("array names must be non-empty identifiers")
        array = np.asarray(values[name])
        if array.dtype.hasobject or array.dtype.kind not in "biufSU":
            raise AuditArtifactError(f"array {name!r} has an unsupported dtype")
        if array.ndim < 1 or array.size < 1:
            raise AuditArtifactError(f"array {name!r} must be non-empty")
        if array.dtype.kind == "f" and not np.all(np.isfinite(array)):
            raise AuditArtifactError(f"array {name!r} must contain only finite values")
        immutable = np.array(array, copy=True)
        immutable.setflags(write=False)
        result[name] = cast(NDArray[np.generic], immutable)
    return result


def _finite_json_mapping(value: Mapping[str, object], context: str) -> dict[str, object]:
    try:
        serialized = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        decoded: object = json.loads(serialized)
    except (TypeError, ValueError) as error:
        raise AuditArtifactError(f"{context} must contain finite JSON") from error
    return dict(_mapping(decoded, context))


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise AuditArtifactError(f"{context} must be a string-keyed mapping")
    return cast(Mapping[str, object], value)


def _canonical_sha256(value: Mapping[str, object]) -> str:
    serialized = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return "sha256:" + hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _hash(value: object, context: str) -> str:
    if not isinstance(value, str):
        raise AuditArtifactIntegrityError(f"{context} must be a prefixed SHA-256")
    digest = value.removeprefix("sha256:")
    if (
        not value.startswith("sha256:")
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise AuditArtifactIntegrityError(f"{context} must be a prefixed SHA-256")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _temporary_path(destination: Path) -> Path:
    descriptor, raw = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    return Path(raw)


def _write_npz(path: Path, arrays: Mapping[str, NDArray[np.generic]]) -> None:
    with path.open("wb") as handle:
        np.savez_compressed(handle, **cast(dict[str, Any], dict(arrays)))


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    serialized = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
    )
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(serialized)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _load_npz(
    path: Path, signatures: Mapping[str, object]
) -> dict[str, NDArray[np.generic]]:
    result: dict[str, NDArray[np.generic]] = {}
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(signatures):
                raise AuditArtifactIntegrityError("audit NPZ member set mismatch")
            for name in sorted(signatures):
                signature = _mapping(signatures[name], f"arrays.{name}")
                if set(signature) != {"shape", "dtype"}:
                    raise AuditArtifactIntegrityError("audit array signature is not canonical")
                shape = signature["shape"]
                if not isinstance(shape, list) or not all(
                    isinstance(item, int) and not isinstance(item, bool) and item >= 0
                    for item in shape
                ):
                    raise AuditArtifactIntegrityError("audit array shape is invalid")
                dtype = signature["dtype"]
                if not isinstance(dtype, str):
                    raise AuditArtifactIntegrityError("audit array dtype is invalid")
                array = np.asarray(archive[name])
                if list(array.shape) != shape or array.dtype.str != dtype:
                    raise AuditArtifactIntegrityError("audit array signature mismatch")
                if array.dtype.hasobject or (
                    array.dtype.kind == "f" and not np.all(np.isfinite(array))
                ):
                    raise AuditArtifactIntegrityError("audit array content is invalid")
                immutable = np.array(array, copy=True)
                immutable.setflags(write=False)
                result[name] = cast(NDArray[np.generic], immutable)
    except AuditArtifactIntegrityError:
        raise
    except (OSError, ValueError, KeyError) as error:
        raise AuditArtifactIntegrityError(f"could not load audit NPZ: {error}") from error
    return result


def _unlink(path: Path) -> None:
    with suppress(OSError):
        path.unlink(missing_ok=True)


__all__ = [
    "AuditArrayArtifact",
    "AuditArrayFiles",
    "AuditArtifactError",
    "AuditArtifactIntegrityError",
    "load_audit_array_artifact",
    "save_audit_array_artifact",
]
