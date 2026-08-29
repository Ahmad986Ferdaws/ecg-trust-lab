"""Immutable TrustBundle manifests and content verification."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Annotated, Literal, Self

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)

from ecg_trust.constants import LEADS, PTBXL_VERSION, SUPERCLASSES
from ecg_trust.contracts import Sha256Digest

TRUST_BUNDLE_SCHEMA_VERSION: Literal["ecg_trust.trust_bundle.v1"] = "ecg_trust.trust_bundle.v1"
TRUST_BUNDLE_BODY_SCHEMA_VERSION: Literal["ecg_trust.trust_bundle.v1"] = "ecg_trust.trust_bundle.v1"
TRUST_BUNDLE_PARENT_SCHEMA_VERSION: Literal["ecg_trust.trust_bundle_parent.v1"] = (
    "ecg_trust.trust_bundle_parent.v1"
)
TRUST_BUNDLE_COMPATIBILITY_SCHEMA_VERSION: Literal["ecg_trust.trust_bundle_compatibility.v1"] = (
    "ecg_trust.trust_bundle_compatibility.v1"
)
MAX_MANIFEST_BYTES = 1024 * 1024

RegistryIdentifier = Annotated[
    str,
    StringConstraints(
        strict=True,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
        min_length=1,
        max_length=160,
    ),
]
RelativeArtifactPath = Annotated[
    str,
    StringConstraints(strict=True, min_length=1, max_length=512),
]
GitCommit = Annotated[
    str,
    StringConstraints(strict=True, pattern=r"^[0-9a-f]{40}$", min_length=40, max_length=40),
]
RegistryMediaType = Annotated[
    str,
    StringConstraints(
        strict=True,
        pattern=r"^[a-z0-9][a-z0-9.+-]*/[a-z0-9][a-z0-9.+-]*$",
        min_length=3,
        max_length=128,
    ),
]
PositiveInt = Annotated[int, Field(strict=True, ge=1)]
FinitePositiveFloat = Annotated[
    float,
    Field(strict=True, gt=0.0, allow_inf_nan=False),
]
LeadName = Literal["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
SuperclassLabel = Literal["NORM", "MI", "STTC", "CD", "HYP"]


class ReleaseRegistryError(ValueError):
    """Base class for registry validation failures."""


class TrustBundleFormatError(ReleaseRegistryError):
    """Raised when a manifest is not canonical or schema-valid."""


class TrustBundleIntegrityError(ReleaseRegistryError):
    """Raised when a manifest or parent file identity does not verify."""


class TrustBundleCompatibilityError(ReleaseRegistryError):
    """Raised when a release cannot run under the expected scientific contract."""


class StrictRegistryModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_default=True,
        use_enum_values=False,
    )


class ArtifactRole(StrEnum):
    CHECKPOINT = "CHECKPOINT"
    RESOLVED_CONFIG = "RESOLVED_CONFIG"
    NORMALIZATION = "NORMALIZATION"
    DECISION_POLICY = "DECISION_POLICY"
    QUALITY_POLICY = "QUALITY_POLICY"
    DISTRIBUTION_POLICY = "DISTRIBUTION_POLICY"
    CONFORMAL_POLICY = "CONFORMAL_POLICY"
    LABEL_ONTOLOGY = "LABEL_ONTOLOGY"
    SAFETY_DOCUMENT = "SAFETY_DOCUMENT"
    PROTOCOL = "PROTOCOL"
    DATASET_MANIFEST = "DATASET_MANIFEST"
    ENVIRONMENT_LOCK = "ENVIRONMENT_LOCK"
    EVIDENCE_BUNDLE = "EVIDENCE_BUNDLE"


_SINGLETON_ROLES = frozenset(ArtifactRole) - {ArtifactRole.CHECKPOINT}


class TrustBundleCompatibility(StrictRegistryModel):
    """Complete runtime/scientific compatibility identity for one release."""

    schema_version: Literal["ecg_trust.trust_bundle_compatibility.v1"]
    task: Literal["FIVE_SUPERCLASS_MULTILABEL_ECG_CLASSIFICATION"]
    source_dataset: Literal["PTB-XL"]
    source_dataset_version: Literal["1.0.3"]
    label_order: tuple[SuperclassLabel, ...]
    lead_order: tuple[LeadName, ...]
    sampling_frequency_hz: FinitePositiveFloat
    samples_per_lead: PositiveInt
    duration_seconds: FinitePositiveFloat
    physical_units: Literal["mV"]
    input_dtype: Literal["float32"]
    model_output_count: PositiveInt
    model_member_count: PositiveInt
    calibration_method: Literal["temperature_scaling"]
    gate_method: Literal["mean_normalized_binary_entropy"]
    calibration_folds: tuple[PositiveInt, ...]

    @model_validator(mode="after")
    def _canonical_v1_contract(self) -> Self:
        if self.source_dataset_version != PTBXL_VERSION:
            raise ValueError("source dataset version is not the canonical PTB-XL version")
        if self.label_order != SUPERCLASSES:
            raise ValueError("label_order must exactly match the five canonical superclasses")
        if self.lead_order != LEADS:
            raise ValueError("lead_order must exactly match the canonical 12-lead order")
        if self.sampling_frequency_hz != 100.0:
            raise ValueError("sampling_frequency_hz must be exactly 100 Hz")
        if self.samples_per_lead != 1000 or self.duration_seconds != 10.0:
            raise ValueError("input duration must be exactly 1000 samples / 10 seconds")
        if self.model_output_count != len(SUPERCLASSES):
            raise ValueError("model_output_count must equal the canonical label count")
        if self.calibration_folds != (9,):
            raise ValueError("calibration_folds must be exactly fold 9")
        return self

    @classmethod
    def canonical(cls, *, model_member_count: int = 1) -> Self:
        """Construct the sole v1 scientific compatibility profile."""

        return cls(
            schema_version=TRUST_BUNDLE_COMPATIBILITY_SCHEMA_VERSION,
            task="FIVE_SUPERCLASS_MULTILABEL_ECG_CLASSIFICATION",
            source_dataset="PTB-XL",
            source_dataset_version="1.0.3",
            label_order=("NORM", "MI", "STTC", "CD", "HYP"),
            lead_order=(
                "I",
                "II",
                "III",
                "aVR",
                "aVL",
                "aVF",
                "V1",
                "V2",
                "V3",
                "V4",
                "V5",
                "V6",
            ),
            sampling_frequency_hz=100.0,
            samples_per_lead=1000,
            duration_seconds=10.0,
            physical_units="mV",
            input_dtype="float32",
            model_output_count=len(SUPERCLASSES),
            model_member_count=model_member_count,
            calibration_method="temperature_scaling",
            gate_method="mean_normalized_binary_entropy",
            calibration_folds=(9,),
        )


class TrustBundleParent(StrictRegistryModel):
    """One file identity bound into a TrustBundle."""

    schema_version: Literal["ecg_trust.trust_bundle_parent.v1"]
    artifact_id: RegistryIdentifier
    role: ArtifactRole
    relative_path: RelativeArtifactPath
    size_bytes: PositiveInt
    file_sha256: Sha256Digest
    media_type: RegistryMediaType

    @field_validator("relative_path")
    @classmethod
    def _safe_canonical_relative_path(cls, value: str) -> str:
        if "\\" in value or "\x00" in value or ":" in value:
            raise ValueError("relative_path must be a portable POSIX path")
        path = PurePosixPath(value)
        if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
            raise ValueError("relative_path must remain beneath the artifact root")
        if path.as_posix() != value:
            raise ValueError("relative_path must be canonical POSIX text")
        return value


class TrustBundleBody(StrictRegistryModel):
    """Canonical self-hashed manifest body."""

    schema_version: Literal["ecg_trust.trust_bundle.v1"]
    release_id: RegistryIdentifier
    created_at: AwareDatetime
    code_commit: GitCommit
    protocol_sha256: Sha256Digest
    dataset_manifest_sha256: Sha256Digest
    environment_lock_sha256: Sha256Digest
    compatibility: TrustBundleCompatibility
    intended_use: Literal["RESEARCH_ONLY"]
    parents: tuple[TrustBundleParent, ...] = Field(min_length=1)

    @field_validator("parents")
    @classmethod
    def _parents_are_canonical(
        cls, value: tuple[TrustBundleParent, ...]
    ) -> tuple[TrustBundleParent, ...]:
        identities = [parent.artifact_id for parent in value]
        paths = [parent.relative_path for parent in value]
        if len(set(identities)) != len(identities):
            raise ValueError("parent artifact_id values must be unique")
        if len(set(paths)) != len(paths):
            raise ValueError("parent relative_path values must be unique")
        canonical = sorted(value, key=lambda parent: (parent.role.value, parent.artifact_id))
        if list(value) != canonical:
            raise ValueError("parents must be sorted by role then artifact_id")
        return value

    @model_validator(mode="after")
    def _complete_release_graph(self) -> Self:
        by_role: dict[ArtifactRole, list[TrustBundleParent]] = {role: [] for role in ArtifactRole}
        for parent in self.parents:
            by_role[parent.role].append(parent)
        for role in _SINGLETON_ROLES:
            if len(by_role[role]) != 1:
                raise ValueError(f"TrustBundle requires exactly one {role.value} parent")
        if len(by_role[ArtifactRole.CHECKPOINT]) != self.compatibility.model_member_count:
            raise ValueError("CHECKPOINT parent count must equal model_member_count")
        if by_role[ArtifactRole.PROTOCOL][0].file_sha256 != self.protocol_sha256:
            raise ValueError("protocol_sha256 does not match the PROTOCOL parent")
        if by_role[ArtifactRole.DATASET_MANIFEST][0].file_sha256 != self.dataset_manifest_sha256:
            raise ValueError("dataset_manifest_sha256 does not match the DATASET_MANIFEST parent")
        if by_role[ArtifactRole.ENVIRONMENT_LOCK][0].file_sha256 != self.environment_lock_sha256:
            raise ValueError("environment_lock_sha256 does not match the ENVIRONMENT_LOCK parent")
        return self


class TrustBundle(TrustBundleBody):
    """Immutable release manifest, including its canonical body digest."""

    manifest_sha256: Sha256Digest

    @model_validator(mode="after")
    def _self_hash_verifies(self) -> Self:
        observed = canonical_sha256(self.model_dump(mode="json", exclude={"manifest_sha256"}))
        if observed != self.manifest_sha256:
            raise ValueError("TrustBundle manifest_sha256 does not match its canonical body")
        return self


@dataclass(frozen=True, slots=True)
class VerifiedTrustBundle:
    """A TrustBundle whose complete parent graph has been verified on disk."""

    manifest: TrustBundle
    artifact_root: Path
    files_by_artifact_id: Mapping[str, Path]


def canonical_json_bytes(value: Mapping[str, object]) -> bytes:
    """Serialize finite JSON using the repository's canonical representation."""

    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise TrustBundleFormatError("value is not finite canonical JSON") from error


def canonical_sha256(value: Mapping[str, object]) -> str:
    """Return a prefixed SHA-256 over canonical JSON."""

    return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Return a prefixed SHA-256 for one regular file."""

    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def bind_parent_file(
    artifact_root: str | Path,
    *,
    artifact_id: str,
    role: ArtifactRole,
    relative_path: str,
    media_type: str,
) -> TrustBundleParent:
    """Create a parent identity only after safely resolving and hashing its file."""

    root = _verified_root(artifact_root)
    path = _resolve_parent_file(root, relative_path)
    return TrustBundleParent(
        schema_version=TRUST_BUNDLE_PARENT_SCHEMA_VERSION,
        artifact_id=artifact_id,
        role=role,
        relative_path=relative_path,
        size_bytes=path.stat().st_size,
        file_sha256=sha256_file(path),
        media_type=media_type,
    )


def seal_trust_bundle(
    *,
    release_id: str,
    created_at: datetime,
    code_commit: str,
    protocol_sha256: str,
    dataset_manifest_sha256: str,
    environment_lock_sha256: str,
    compatibility: TrustBundleCompatibility,
    parents: Sequence[TrustBundleParent],
) -> TrustBundle:
    """Validate a complete release graph and seal its canonical self-hash."""

    body = TrustBundleBody(
        schema_version=TRUST_BUNDLE_BODY_SCHEMA_VERSION,
        release_id=release_id,
        created_at=created_at,
        code_commit=code_commit,
        protocol_sha256=protocol_sha256,
        dataset_manifest_sha256=dataset_manifest_sha256,
        environment_lock_sha256=environment_lock_sha256,
        compatibility=compatibility,
        intended_use="RESEARCH_ONLY",
        parents=tuple(parents),
    )
    payload = body.model_dump(mode="json")
    payload["schema_version"] = TRUST_BUNDLE_SCHEMA_VERSION
    return TrustBundle.model_validate({**payload, "manifest_sha256": canonical_sha256(payload)})


def save_trust_bundle_manifest(path: str | Path, bundle: TrustBundle) -> Path:
    """Create a canonical manifest once; existing destinations are never replaced."""

    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    serialized = canonical_json_bytes(bundle.model_dump(mode="json")) + b"\n"
    descriptor: int | None = None
    created = False
    try:
        descriptor = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
            0o600,
        )
        created = True
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        raise FileExistsError(
            f"TrustBundle manifests are immutable; refusing to overwrite {destination!s}"
        ) from None
    except Exception:
        if descriptor is not None:
            os.close(descriptor)
        if created:
            with suppress(FileNotFoundError):
                destination.unlink()
        raise
    return destination


def load_trust_bundle_manifest(path: str | Path) -> TrustBundle:
    """Load only a schema-valid, self-hashed, canonically serialized manifest."""

    source = Path(path)
    try:
        size = source.stat().st_size
        if size < 2 or size > MAX_MANIFEST_BYTES:
            raise TrustBundleFormatError("TrustBundle manifest size is invalid")
        raw = source.read_bytes()
        bundle = TrustBundle.model_validate_json(raw)
    except TrustBundleFormatError:
        raise
    except (OSError, ValidationError, UnicodeError) as error:
        raise TrustBundleFormatError(f"could not decode TrustBundle manifest: {error}") from error
    expected = canonical_json_bytes(bundle.model_dump(mode="json")) + b"\n"
    if raw != expected:
        raise TrustBundleFormatError("TrustBundle manifest is not canonical JSON")
    return bundle


def verify_trust_bundle(
    bundle: TrustBundle,
    artifact_root: str | Path,
    *,
    expected_compatibility: TrustBundleCompatibility,
    expected_release_id: str | None = None,
) -> VerifiedTrustBundle:
    """Fail closed on release identity, compatibility, path, size, or digest mismatch."""

    if bundle.compatibility != expected_compatibility:
        raise TrustBundleCompatibilityError(
            "TrustBundle compatibility differs from the runtime expectation"
        )
    if expected_release_id is not None and bundle.release_id != expected_release_id:
        raise TrustBundleCompatibilityError("TrustBundle release_id differs from expectation")
    root = _verified_root(artifact_root)
    verified: dict[str, Path] = {}
    for parent in bundle.parents:
        path = _resolve_parent_file(root, parent.relative_path)
        if path.stat().st_size != parent.size_bytes:
            raise TrustBundleIntegrityError(
                f"parent size mismatch for artifact {parent.artifact_id}"
            )
        if sha256_file(path) != parent.file_sha256:
            raise TrustBundleIntegrityError(
                f"parent SHA-256 mismatch for artifact {parent.artifact_id}"
            )
        verified[parent.artifact_id] = path
    return VerifiedTrustBundle(
        manifest=bundle,
        artifact_root=root,
        files_by_artifact_id=MappingProxyType(verified),
    )


def load_and_verify_trust_bundle(
    manifest_path: str | Path,
    *,
    artifact_root: str | Path,
    expected_compatibility: TrustBundleCompatibility,
    expected_release_id: str | None = None,
) -> VerifiedTrustBundle:
    """Load a manifest and verify every compatibility and parent-file boundary."""

    bundle = load_trust_bundle_manifest(manifest_path)
    return verify_trust_bundle(
        bundle,
        artifact_root,
        expected_compatibility=expected_compatibility,
        expected_release_id=expected_release_id,
    )


def _verified_root(artifact_root: str | Path) -> Path:
    raw = Path(artifact_root)
    if raw.is_symlink():
        raise TrustBundleIntegrityError("artifact root must not be a symbolic link")
    try:
        root = raw.resolve(strict=True)
    except OSError as error:
        raise TrustBundleIntegrityError(f"artifact root is unavailable: {error}") from error
    if not root.is_dir():
        raise TrustBundleIntegrityError("artifact root must be a directory")
    return root


def _resolve_parent_file(root: Path, relative_path: str) -> Path:
    try:
        canonical_path = TrustBundleParent._safe_canonical_relative_path(relative_path)
    except (TypeError, ValueError) as error:
        raise TrustBundleIntegrityError(f"unsafe parent path {relative_path!r}") from error
    candidate = root.joinpath(*PurePosixPath(canonical_path).parts)
    if candidate.is_symlink():
        raise TrustBundleIntegrityError("parent artifacts must not be symbolic links")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise TrustBundleIntegrityError(
            f"parent artifact is unavailable: {relative_path}"
        ) from error
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise TrustBundleIntegrityError("parent artifact escaped its root or is not a file")
    return resolved


__all__ = [
    "MAX_MANIFEST_BYTES",
    "TRUST_BUNDLE_BODY_SCHEMA_VERSION",
    "TRUST_BUNDLE_COMPATIBILITY_SCHEMA_VERSION",
    "TRUST_BUNDLE_PARENT_SCHEMA_VERSION",
    "TRUST_BUNDLE_SCHEMA_VERSION",
    "ArtifactRole",
    "ReleaseRegistryError",
    "TrustBundle",
    "TrustBundleBody",
    "TrustBundleCompatibility",
    "TrustBundleCompatibilityError",
    "TrustBundleFormatError",
    "TrustBundleIntegrityError",
    "TrustBundleParent",
    "VerifiedTrustBundle",
    "bind_parent_file",
    "canonical_json_bytes",
    "canonical_sha256",
    "load_and_verify_trust_bundle",
    "load_trust_bundle_manifest",
    "save_trust_bundle_manifest",
    "seal_trust_bundle",
    "sha256_file",
    "verify_trust_bundle",
]
