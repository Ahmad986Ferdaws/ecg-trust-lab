from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from ecg_trust.registry import (
    ArtifactRole,
    TrustBundleCompatibility,
    TrustBundleParent,
    bind_parent_file,
    seal_trust_bundle,
    verify_trust_bundle,
)
from ecg_trust.runtime_binding import (
    RuntimeArtifact,
    RuntimeBindingError,
    RuntimeTrustBinding,
)

NOW = datetime(2026, 8, 24, 12, 0, tzinfo=UTC)


def _verified_bundle(root: Path, *, quality_payload: bytes = b"quality-policy-v1\n"):
    declarations: list[tuple[str, ArtifactRole, str]] = []
    for role in ArtifactRole:
        artifact_id = "checkpoint-0" if role is ArtifactRole.CHECKPOINT else role.value.lower()
        filename = f"{artifact_id}.bin"
        payload = (
            quality_payload
            if role is ArtifactRole.QUALITY_POLICY
            else f"{role.value}:{artifact_id}\n".encode()
        )
        (root / filename).write_bytes(payload)
        declarations.append((artifact_id, role, filename))
    parents = tuple(
        sorted(
            (
                bind_parent_file(
                    root,
                    artifact_id=artifact_id,
                    role=role,
                    relative_path=filename,
                    media_type="application/octet-stream",
                )
                for artifact_id, role, filename in declarations
            ),
            key=lambda parent: (parent.role.value, parent.artifact_id),
        )
    )
    by_role: dict[ArtifactRole, TrustBundleParent] = {
        parent.role: parent for parent in parents if parent.role is not ArtifactRole.CHECKPOINT
    }
    compatibility = TrustBundleCompatibility.canonical()
    bundle = seal_trust_bundle(
        release_id="release-vnext",
        created_at=NOW,
        code_commit="a" * 40,
        protocol_sha256=by_role[ArtifactRole.PROTOCOL].file_sha256,
        dataset_manifest_sha256=by_role[ArtifactRole.DATASET_MANIFEST].file_sha256,
        environment_lock_sha256=by_role[ArtifactRole.ENVIRONMENT_LOCK].file_sha256,
        compatibility=compatibility,
        parents=parents,
    )
    return verify_trust_bundle(bundle, root, expected_compatibility=compatibility)


def test_binding_carries_exact_manifest_roles_and_path_free_loaded_identity(tmp_path: Path) -> None:
    verified = _verified_bundle(tmp_path)
    binding = RuntimeTrustBinding(verified)
    quality = binding.require_single(ArtifactRole.QUALITY_POLICY)

    loaded = binding.load_component(
        (quality,),
        lambda inputs: inputs[0].private_path.read_bytes(),
    )

    assert binding.release_id == "release-vnext"
    assert binding.service_manifest_sha256 == (
        verified.manifest.manifest_sha256.removeprefix("sha256:")
    )
    assert loaded.value == b"quality-policy-v1\n"
    assert loaded.artifact_identities == (quality.identity,)
    assert str(tmp_path) not in repr(binding)
    assert str(tmp_path) not in repr(loaded)


def test_same_size_parent_mutation_fails_closed_without_path_details(tmp_path: Path) -> None:
    binding = RuntimeTrustBinding(_verified_bundle(tmp_path))
    quality = binding.require_single(ArtifactRole.QUALITY_POLICY)
    original = quality.private_path.read_bytes()
    quality.private_path.write_bytes(b"X" * len(original))

    assert not binding.is_intact()
    with pytest.raises(RuntimeBindingError) as captured:
        binding.verify_intact()
    assert str(captured.value) == "runtime release integrity verification failed"
    assert str(tmp_path) not in str(captured.value)


def test_cross_bundle_same_release_and_role_cannot_substitute_component(tmp_path: Path) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    first = RuntimeTrustBinding(_verified_bundle(first_root, quality_payload=b"policy-a\n"))
    second = RuntimeTrustBinding(_verified_bundle(second_root, quality_payload=b"policy-b\n"))
    first_quality = first.require_single(ArtifactRole.QUALITY_POLICY)
    loaded_from_first = first.load_component((first_quality,), lambda _: "same-version-v1")

    assert first.release_id == second.release_id
    assert first.manifest_sha256 != second.manifest_sha256
    with pytest.raises(RuntimeBindingError, match="integrity verification failed"):
        second.assert_component_roles(
            loaded_from_first,
            (ArtifactRole.QUALITY_POLICY,),
        )


def test_forged_loader_path_and_during_load_tampering_are_rejected(tmp_path: Path) -> None:
    binding = RuntimeTrustBinding(_verified_bundle(tmp_path))
    quality = binding.require_single(ArtifactRole.QUALITY_POLICY)
    forged_path = tmp_path / "forged.bin"
    forged_path.write_bytes(quality.private_path.read_bytes())
    forged = RuntimeArtifact(identity=quality.identity, _path=forged_path)

    with pytest.raises(RuntimeBindingError, match="integrity verification failed"):
        binding.load_component((forged,), lambda _: object())

    def tampering_loader(inputs: tuple[RuntimeArtifact, ...]) -> object:
        source = inputs[0].private_path
        source.write_bytes(b"Z" * source.stat().st_size)
        return object()

    with pytest.raises(RuntimeBindingError, match="integrity verification failed"):
        binding.load_component((quality,), tampering_loader)
