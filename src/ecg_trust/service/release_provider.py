"""Fail-closed TrustBundle adapter for the Sentinel release-provider protocol."""

from __future__ import annotations

import re
from _thread import RLock
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from ecg_trust.registry import (
    TrustBundleCompatibility,
    VerifiedTrustBundle,
    load_and_verify_trust_bundle,
)
from ecg_trust.service.sentinel_service import (
    DependencyUnavailableError,
    ResourceNotFoundError,
    VerifiedRelease,
)

_SERVICE_RELEASE_ID_RE: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_PREFIXED_SHA256_RE: Final = re.compile(r"^sha256:([a-f0-9]{64})$")
_UNAVAILABLE_MESSAGE: Final = "verified release is unavailable"
_NOT_FOUND_MESSAGE: Final = "release not found"


class _VerificationFailure(RuntimeError):
    """Private path-free signal collapsed at the public provider boundary."""


@dataclass(frozen=True, slots=True, init=False)
class TrustBundleReleaseProvider:
    """Reverify one immutable TrustBundle graph at every public boundary.

    No verified state or resolved parent paths are cached. The lock only serializes
    local reads so callers observe one complete verification attempt at a time.
    """

    _manifest_path: Path = field(repr=False)
    _artifact_root: Path = field(repr=False)
    _expected_compatibility: TrustBundleCompatibility = field(repr=False)
    _expected_release_id: str
    _lock: RLock = field(repr=False, compare=False)

    def __init__(
        self,
        *,
        manifest_path: str | Path,
        artifact_root: str | Path,
        expected_compatibility: TrustBundleCompatibility,
        expected_release_id: str,
    ) -> None:
        if not isinstance(expected_compatibility, TrustBundleCompatibility):
            raise TypeError("expected_compatibility must be a TrustBundleCompatibility")
        if (
            not isinstance(expected_release_id, str)
            or _SERVICE_RELEASE_ID_RE.fullmatch(expected_release_id) is None
        ):
            raise ValueError("expected_release_id must be a bounded opaque service identifier")
        object.__setattr__(self, "_manifest_path", Path(manifest_path))
        object.__setattr__(self, "_artifact_root", Path(artifact_root))
        object.__setattr__(
            self,
            "_expected_compatibility",
            expected_compatibility.model_copy(deep=True),
        )
        object.__setattr__(self, "_expected_release_id", expected_release_id)
        object.__setattr__(self, "_lock", RLock())

    def _load_verified_bundle(self) -> VerifiedTrustBundle:
        with self._lock:
            try:
                if self._manifest_path.is_symlink():
                    raise _VerificationFailure
                verified_bundle = load_and_verify_trust_bundle(
                    self._manifest_path,
                    artifact_root=self._artifact_root,
                    expected_compatibility=self._expected_compatibility,
                    expected_release_id=self._expected_release_id,
                )
                if _PREFIXED_SHA256_RE.fullmatch(verified_bundle.manifest.manifest_sha256) is None:
                    raise _VerificationFailure
                return verified_bundle
            except Exception:
                # Registry errors can contain local paths. Never propagate or chain them.
                raise _VerificationFailure(_UNAVAILABLE_MESSAGE) from None

    def is_ready(self) -> bool:
        """Return readiness only after a fresh complete graph verification."""

        try:
            self._load_verified_bundle()
        except _VerificationFailure:
            return False
        return True

    def get_release(self, release_id: str) -> VerifiedRelease:
        """Reverify first, then return the sole configured safe release projection."""

        try:
            verified_bundle = self._load_verified_bundle()
        except _VerificationFailure:
            raise DependencyUnavailableError(_UNAVAILABLE_MESSAGE) from None
        if not isinstance(release_id, str) or release_id != self._expected_release_id:
            raise ResourceNotFoundError(_NOT_FOUND_MESSAGE) from None
        match = _PREFIXED_SHA256_RE.fullmatch(verified_bundle.manifest.manifest_sha256)
        if match is None:  # defensive: _load_verified_bundle already checked this.
            raise DependencyUnavailableError(_UNAVAILABLE_MESSAGE) from None
        return VerifiedRelease(
            release_id=self._expected_release_id,
            artifact_sha256=match.group(1),
            verified=True,
            locked=True,
        )

    def get_active_release(self) -> VerifiedRelease:
        """Freshly verify and return the sole release configured for readiness."""

        return self.get_release(self._expected_release_id)

    def get_verified_bundle(self, release_id: str) -> VerifiedTrustBundle:
        """Return a fresh private runtime graph; this object must never be serialized."""

        try:
            verified_bundle = self._load_verified_bundle()
        except _VerificationFailure:
            raise DependencyUnavailableError(_UNAVAILABLE_MESSAGE) from None
        if not isinstance(release_id, str) or release_id != self._expected_release_id:
            raise ResourceNotFoundError(_NOT_FOUND_MESSAGE) from None
        return verified_bundle
