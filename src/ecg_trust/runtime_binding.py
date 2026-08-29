"""Cryptographic binding between a verified TrustBundle and loaded runtime objects.

The registry proves that a release graph was intact at one instant.  This module
keeps that proof attached to every component loaded from the graph and repeats
the complete parent verification at readiness and inference boundaries.  Local
paths are intentionally private and every public exception is path-free.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Final

from ecg_trust.registry import ArtifactRole, VerifiedTrustBundle, sha256_file

_PREFIXED_SHA256_RE: Final = re.compile(r"^sha256:([0-9a-f]{64})$")
_INTEGRITY_ERROR: Final = "runtime release integrity verification failed"
_LOAD_ERROR: Final = "verified runtime component loading failed"


class RuntimeBindingError(RuntimeError):
    """A verified release graph no longer matches its sealed identities."""


class RuntimeComponentLoadError(RuntimeBindingError):
    """A component could not be loaded from its verified parent artifacts."""


@dataclass(frozen=True, slots=True)
class RuntimeArtifactIdentity:
    """Path-free identity of one TrustBundle parent."""

    artifact_id: str
    role: ArtifactRole
    file_sha256: str
    size_bytes: int
    media_type: str

    def __post_init__(self) -> None:
        if not self.artifact_id:
            raise RuntimeBindingError(_INTEGRITY_ERROR)
        if not isinstance(self.role, ArtifactRole):
            raise RuntimeBindingError(_INTEGRITY_ERROR)
        if _PREFIXED_SHA256_RE.fullmatch(self.file_sha256) is None:
            raise RuntimeBindingError(_INTEGRITY_ERROR)
        if isinstance(self.size_bytes, bool) or self.size_bytes <= 0:
            raise RuntimeBindingError(_INTEGRITY_ERROR)
        if not self.media_type:
            raise RuntimeBindingError(_INTEGRITY_ERROR)

    @property
    def unprefixed_sha256(self) -> str:
        """Return the digest form used by the identifier-only service contract."""

        return self.file_sha256.removeprefix("sha256:")


@dataclass(frozen=True, slots=True)
class RuntimeArtifact:
    """Verified loader input; its local path is never included in repr/JSON."""

    identity: RuntimeArtifactIdentity
    _path: Path = field(repr=False, compare=False)

    @property
    def private_path(self) -> Path:
        """Return the verified local path for a trusted in-process loader only."""

        return self._path


@dataclass(frozen=True, slots=True)
class BoundRuntimeComponent[T]:
    """A loaded value plus the exact verified parents supplied to its loader."""

    value: T
    manifest_sha256: str
    artifact_identities: tuple[RuntimeArtifactIdentity, ...]
    _binding_token: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if _PREFIXED_SHA256_RE.fullmatch(self.manifest_sha256) is None:
            raise RuntimeBindingError(_INTEGRITY_ERROR)
        if not self.artifact_identities:
            raise RuntimeBindingError(_INTEGRITY_ERROR)
        artifact_ids = tuple(item.artifact_id for item in self.artifact_identities)
        if len(set(artifact_ids)) != len(artifact_ids):
            raise RuntimeBindingError(_INTEGRITY_ERROR)


@dataclass(frozen=True, slots=True, init=False)
class RuntimeTrustBinding:
    """Immutable release identity with fresh, thread-safe parent verification."""

    release_id: str
    manifest_sha256: str
    _artifact_root: Path = field(repr=False)
    _artifacts: tuple[RuntimeArtifact, ...] = field(repr=False)
    _token: object = field(repr=False, compare=False)
    _lock: RLock = field(repr=False, compare=False)

    def __init__(self, verified_bundle: VerifiedTrustBundle) -> None:
        if not isinstance(verified_bundle, VerifiedTrustBundle):
            raise TypeError("verified_bundle must be a VerifiedTrustBundle")
        try:
            root = verified_bundle.artifact_root.resolve(strict=True)
            if verified_bundle.artifact_root.is_symlink() or not root.is_dir():
                raise RuntimeBindingError
            manifest = verified_bundle.manifest
            if _PREFIXED_SHA256_RE.fullmatch(manifest.manifest_sha256) is None:
                raise RuntimeBindingError
            expected_ids = {parent.artifact_id for parent in manifest.parents}
            if set(verified_bundle.files_by_artifact_id) != expected_ids:
                raise RuntimeBindingError
            artifacts = tuple(
                RuntimeArtifact(
                    identity=RuntimeArtifactIdentity(
                        artifact_id=parent.artifact_id,
                        role=parent.role,
                        file_sha256=parent.file_sha256,
                        size_bytes=parent.size_bytes,
                        media_type=parent.media_type,
                    ),
                    _path=verified_bundle.files_by_artifact_id[parent.artifact_id],
                )
                for parent in manifest.parents
            )
        except Exception:
            raise RuntimeBindingError(_INTEGRITY_ERROR) from None

        object.__setattr__(self, "release_id", manifest.release_id)
        object.__setattr__(self, "manifest_sha256", manifest.manifest_sha256)
        object.__setattr__(self, "_artifact_root", root)
        object.__setattr__(self, "_artifacts", artifacts)
        object.__setattr__(self, "_token", object())
        object.__setattr__(self, "_lock", RLock())
        self.verify_intact()

    @property
    def service_manifest_sha256(self) -> str:
        """Return the unprefixed manifest digest carried by ``VerifiedRelease``."""

        return self.manifest_sha256.removeprefix("sha256:")

    def artifacts_for_role(self, role: ArtifactRole) -> tuple[RuntimeArtifact, ...]:
        """Return canonical verified loader inputs for one TrustBundle role."""

        if not isinstance(role, ArtifactRole):
            raise TypeError("role must be an ArtifactRole")
        return tuple(artifact for artifact in self._artifacts if artifact.identity.role is role)

    def require_single(self, role: ArtifactRole) -> RuntimeArtifact:
        """Return the sole artifact for a singleton role or fail closed."""

        artifacts = self.artifacts_for_role(role)
        if len(artifacts) != 1:
            raise RuntimeBindingError(_INTEGRITY_ERROR)
        return artifacts[0]

    def require_checkpoints(self) -> tuple[RuntimeArtifact, ...]:
        """Return all checkpoint parents in canonical manifest order."""

        checkpoints = self.artifacts_for_role(ArtifactRole.CHECKPOINT)
        if not checkpoints:
            raise RuntimeBindingError(_INTEGRITY_ERROR)
        return checkpoints

    def identities_for_role(self, role: ArtifactRole) -> tuple[RuntimeArtifactIdentity, ...]:
        """Return path-free identities for validation and audit assertions."""

        return tuple(artifact.identity for artifact in self.artifacts_for_role(role))

    def verify_intact(self) -> None:
        """Rehash the complete parent graph and fail without leaking local paths."""

        with self._lock:
            try:
                raw_root = self._artifact_root
                if raw_root.is_symlink():
                    raise RuntimeBindingError
                root = raw_root.resolve(strict=True)
                if root != self._artifact_root or not root.is_dir():
                    raise RuntimeBindingError
                for artifact in self._artifacts:
                    path = artifact.private_path
                    if path.is_symlink():
                        raise RuntimeBindingError
                    resolved = path.resolve(strict=True)
                    if (
                        resolved != path
                        or not resolved.is_relative_to(root)
                        or not resolved.is_file()
                    ):
                        raise RuntimeBindingError
                    stat = resolved.stat()
                    if stat.st_size != artifact.identity.size_bytes:
                        raise RuntimeBindingError
                    if sha256_file(resolved) != artifact.identity.file_sha256:
                        raise RuntimeBindingError
            except Exception:
                raise RuntimeBindingError(_INTEGRITY_ERROR) from None

    def is_intact(self) -> bool:
        """Return a path-free readiness signal after a fresh complete verification."""

        try:
            self.verify_intact()
        except RuntimeBindingError:
            return False
        return True

    def load_component[T](
        self,
        artifacts: Iterable[RuntimeArtifact],
        loader: Callable[[tuple[RuntimeArtifact, ...]], T],
    ) -> BoundRuntimeComponent[T]:
        """Load from exact verified parents and reverify before and after the call."""

        selected = tuple(artifacts)
        if not selected:
            raise RuntimeBindingError(_INTEGRITY_ERROR)
        canonical_by_id = {artifact.identity.artifact_id: artifact for artifact in self._artifacts}
        if any(
            canonical_by_id.get(artifact.identity.artifact_id) is not artifact
            for artifact in selected
        ):
            raise RuntimeBindingError(_INTEGRITY_ERROR)
        self.verify_intact()
        try:
            value = loader(selected)
        except Exception:
            raise RuntimeComponentLoadError(_LOAD_ERROR) from None
        self.verify_intact()
        return BoundRuntimeComponent(
            value=value,
            manifest_sha256=self.manifest_sha256,
            artifact_identities=tuple(artifact.identity for artifact in selected),
            _binding_token=self._token,
        )

    def assert_component_roles(
        self,
        component: BoundRuntimeComponent[object],
        expected_roles: tuple[ArtifactRole, ...],
    ) -> None:
        """Prove a loaded object came from the expected parents of this manifest."""

        if (
            component._binding_token is not self._token
            or component.manifest_sha256 != self.manifest_sha256
        ):
            raise RuntimeBindingError(_INTEGRITY_ERROR)
        actual_roles = tuple(identity.role for identity in component.artifact_identities)
        if actual_roles != expected_roles:
            raise RuntimeBindingError(_INTEGRITY_ERROR)
        for identity in component.artifact_identities:
            canonical = tuple(
                artifact.identity
                for artifact in self._artifacts
                if artifact.identity.artifact_id == identity.artifact_id
            )
            if canonical != (identity,):
                raise RuntimeBindingError(_INTEGRITY_ERROR)


__all__ = [
    "BoundRuntimeComponent",
    "RuntimeArtifact",
    "RuntimeArtifactIdentity",
    "RuntimeBindingError",
    "RuntimeComponentLoadError",
    "RuntimeTrustBinding",
]
