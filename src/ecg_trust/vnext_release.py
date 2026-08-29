"""Canonical fail-closed Trust Sentinel vNext release gate.

The lower-level registry intentionally remains useful for non-Sentinel bundles.
Calling its helpers does not promote a Sentinel release.  This module is the
single boundary for checking whether frozen source-calibration v1 may enter a
release assembler.  Version 1 is PENDING-only, so this gate deliberately has no
success path until a distinct versioned OOD-completion protocol exists.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import NoReturn

from ecg_trust.registry import (
    ArtifactRole,
    TrustBundleIntegrityError,
    TrustBundleParent,
    bind_parent_file,
)
from ecg_trust.source_calibration import (
    SourceCalibrationError,
    SourceCalibrationResult,
    assert_complete_release_ready,
    load_source_calibration_result_bytes,
)


class VNextReleaseAssemblyError(ValueError):
    """Raised when evidence cannot be promoted through the Sentinel boundary."""


def guard_vnext_release_assembly_v1(
    *,
    source_calibration: SourceCalibrationResult,
    source_calibration_relative_path: str,
    artifact_root: str | Path,
    manifest_relative_path: str,
    parents: Sequence[TrustBundleParent],
) -> NoReturn:
    """Validate frozen v1 evidence and then deny release assembly.

    The supplied source-calibration object is not trusted on its own.  Its
    canonical bytes are reloaded from the sole ``EVIDENCE_BUNDLE`` parent and
    compared before readiness is evaluated. Every declared parent is rebound
    to its current file bytes, and the unused destination is checked without
    creating it. The current v1 source contract has no ready state, so this
    function always raises and cannot construct or register a TrustBundle.
    """

    if not isinstance(source_calibration, SourceCalibrationResult):
        raise TypeError("source_calibration must be a loaded SourceCalibrationResult")

    root = _verified_artifact_root(artifact_root)
    _safe_new_manifest_path(root, manifest_relative_path)
    rebound = _reverify_parent_declarations(root, parents)
    by_role = _parents_by_role(rebound)

    evidence_parent = _require_single(by_role, ArtifactRole.EVIDENCE_BUNDLE)
    source_path = _canonical_relative_path(
        source_calibration_relative_path,
        context="source_calibration_relative_path",
    )
    if evidence_parent.relative_path != source_path:
        raise VNextReleaseAssemblyError(
            "the source calibration must be the sole EVIDENCE_BUNDLE parent"
        )

    try:
        loaded = load_source_calibration_result_bytes(
            _resolved_parent_path(root, evidence_parent).read_bytes()
        )
    except (OSError, SourceCalibrationError) as error:
        raise VNextReleaseAssemblyError(
            "source-calibration evidence failed canonical integrity verification"
        ) from error
    if loaded != source_calibration:
        raise VNextReleaseAssemblyError(
            "supplied source calibration differs from the registered evidence bytes"
        )

    try:
        assert_complete_release_ready(loaded)
    except SourceCalibrationError as error:
        raise VNextReleaseAssemblyError(
            "source calibration is not release-ready while OOD evidence is pending"
        ) from error

    raise VNextReleaseAssemblyError(
        "frozen source-calibration v1 has no promotable state; "
        "a new versioned OOD-completion evidence protocol is required"
    )


def _verified_artifact_root(value: str | Path) -> Path:
    raw = Path(value)
    if raw.is_symlink():
        raise VNextReleaseAssemblyError("artifact_root must not be a symbolic link")
    try:
        root = raw.resolve(strict=True)
    except OSError as error:
        raise VNextReleaseAssemblyError("artifact_root is unavailable") from error
    if not root.is_dir():
        raise VNextReleaseAssemblyError("artifact_root must be a directory")
    return root


def _canonical_relative_path(value: str, *, context: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value or ":" in value:
        raise VNextReleaseAssemblyError(f"{context} must be a canonical relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise VNextReleaseAssemblyError(f"{context} must remain beneath artifact_root")
    if path.as_posix() != value:
        raise VNextReleaseAssemblyError(f"{context} must be a canonical relative POSIX path")
    return value


def _safe_new_manifest_path(root: Path, relative_path: str) -> Path:
    canonical = _canonical_relative_path(relative_path, context="manifest_relative_path")
    path = PurePosixPath(canonical)
    if path.suffix != ".json":
        raise VNextReleaseAssemblyError("manifest_relative_path must end in .json")
    parent = root.joinpath(*path.parts[:-1])
    if parent.is_symlink():
        raise VNextReleaseAssemblyError("manifest parent must not be a symbolic link")
    try:
        resolved_parent = parent.resolve(strict=True)
    except OSError as error:
        raise VNextReleaseAssemblyError("manifest parent directory is unavailable") from error
    if not resolved_parent.is_dir() or not resolved_parent.is_relative_to(root):
        raise VNextReleaseAssemblyError("manifest path escaped artifact_root")
    destination = resolved_parent / path.name
    if destination.exists() or destination.is_symlink():
        raise VNextReleaseAssemblyError("release manifest destination already exists")
    return destination


def _reverify_parent_declarations(
    root: Path,
    parents: Sequence[TrustBundleParent],
) -> tuple[TrustBundleParent, ...]:
    if isinstance(parents, (str, bytes)):
        raise TypeError("parents must be a sequence of TrustBundleParent values")
    declared = tuple(parents)
    if not declared or any(not isinstance(parent, TrustBundleParent) for parent in declared):
        raise TypeError("parents must contain TrustBundleParent values")
    artifact_ids = tuple(parent.artifact_id for parent in declared)
    relative_paths = tuple(parent.relative_path for parent in declared)
    if len(set(artifact_ids)) != len(artifact_ids):
        raise VNextReleaseAssemblyError("release parent artifact identities must be unique")
    if len(set(relative_paths)) != len(relative_paths):
        raise VNextReleaseAssemblyError("release parent paths must be unique")
    rebound: list[TrustBundleParent] = []
    for parent in declared:
        try:
            observed = bind_parent_file(
                root,
                artifact_id=parent.artifact_id,
                role=parent.role,
                relative_path=parent.relative_path,
                media_type=parent.media_type,
            )
        except (OSError, TrustBundleIntegrityError, ValueError) as error:
            raise VNextReleaseAssemblyError(
                "release parent failed path or file verification"
            ) from error
        if observed != parent:
            raise VNextReleaseAssemblyError(
                f"release parent changed after binding: {parent.artifact_id}"
            )
        rebound.append(observed)
    return tuple(rebound)


def _parents_by_role(
    parents: Sequence[TrustBundleParent],
) -> dict[ArtifactRole, tuple[TrustBundleParent, ...]]:
    return {
        role: tuple(parent for parent in parents if parent.role is role) for role in ArtifactRole
    }


def _require_single(
    by_role: dict[ArtifactRole, tuple[TrustBundleParent, ...]],
    role: ArtifactRole,
) -> TrustBundleParent:
    matches = by_role[role]
    if len(matches) != 1:
        raise VNextReleaseAssemblyError(f"release requires exactly one {role.value} parent")
    return matches[0]


def _resolved_parent_path(root: Path, parent: TrustBundleParent) -> Path:
    candidate = root.joinpath(*PurePosixPath(parent.relative_path).parts)
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:  # pragma: no cover - rebound immediately before this call
        raise VNextReleaseAssemblyError("source-calibration evidence is unavailable") from error
    if not resolved.is_file() or not resolved.is_relative_to(root):
        raise VNextReleaseAssemblyError("source-calibration evidence escaped artifact_root")
    return resolved


__all__ = [
    "VNextReleaseAssemblyError",
    "guard_vnext_release_assembly_v1",
]
