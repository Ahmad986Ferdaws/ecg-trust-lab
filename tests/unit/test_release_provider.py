from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ecg_trust.registry import (
    ArtifactRole,
    TrustBundle,
    TrustBundleCompatibility,
    TrustBundleParent,
    bind_parent_file,
    save_trust_bundle_manifest,
    seal_trust_bundle,
)
from ecg_trust.service.release_provider import TrustBundleReleaseProvider
from ecg_trust.service.sentinel_service import (
    DependencyUnavailableError,
    ResourceNotFoundError,
    VerifiedReleaseProvider,
)

RELEASE_ID = "release-001"
CREATED_AT = datetime(2026, 8, 24, 12, 0, tzinfo=UTC)


@dataclass(frozen=True, slots=True)
class BundleFixture:
    root: Path
    manifest_path: Path
    bundle: TrustBundle
    compatibility: TrustBundleCompatibility
    parent_paths: dict[ArtifactRole, Path]


def _create_bundle(root: Path, *, release_id: str = RELEASE_ID) -> BundleFixture:
    parent_paths: dict[ArtifactRole, Path] = {}
    declarations: list[tuple[str, ArtifactRole, str]] = []
    for role in ArtifactRole:
        if role is ArtifactRole.CHECKPOINT:
            artifact_id = "checkpoint-0"
            filename = "checkpoint-0.bin"
        else:
            artifact_id = role.value.lower().replace("_", "-")
            filename = role.value.lower().replace("_", "-") + ".bin"
        path = root / filename
        path.write_bytes(f"{role.value}:{artifact_id}\n".encode("ascii"))
        parent_paths[role] = path
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
    compatibility = TrustBundleCompatibility.canonical(model_member_count=1)
    bundle = seal_trust_bundle(
        release_id=release_id,
        created_at=CREATED_AT,
        code_commit="a" * 40,
        protocol_sha256=by_role[ArtifactRole.PROTOCOL].file_sha256,
        dataset_manifest_sha256=by_role[ArtifactRole.DATASET_MANIFEST].file_sha256,
        environment_lock_sha256=by_role[ArtifactRole.ENVIRONMENT_LOCK].file_sha256,
        compatibility=compatibility,
        parents=parents,
    )
    manifest_path = root / "trust-bundle.json"
    save_trust_bundle_manifest(manifest_path, bundle)
    return BundleFixture(
        root=root,
        manifest_path=manifest_path,
        bundle=bundle,
        compatibility=compatibility,
        parent_paths=parent_paths,
    )


def _provider(
    fixture: BundleFixture,
    *,
    expected_compatibility: TrustBundleCompatibility | None = None,
    expected_release_id: str = RELEASE_ID,
) -> TrustBundleReleaseProvider:
    return TrustBundleReleaseProvider(
        manifest_path=fixture.manifest_path,
        artifact_root=fixture.root,
        expected_compatibility=expected_compatibility or fixture.compatibility,
        expected_release_id=expected_release_id,
    )


def _assert_unavailable_without_path(
    provider: TrustBundleReleaseProvider,
    tmp_path: Path,
    *,
    release_id: str = RELEASE_ID,
) -> None:
    assert provider.is_ready() is False
    with pytest.raises(DependencyUnavailableError) as captured:
        provider.get_release(release_id)
    assert str(captured.value) == "verified release is unavailable"
    assert str(tmp_path) not in str(captured.value)
    assert "trust-bundle.json" not in str(captured.value)


def test_good_bundle_satisfies_protocol_and_projects_only_safe_digest(tmp_path: Path) -> None:
    fixture = _create_bundle(tmp_path)
    concrete = _provider(fixture)
    provider: VerifiedReleaseProvider = concrete

    assert provider.is_ready() is True
    release = provider.get_release(RELEASE_ID)
    assert release.release_id == RELEASE_ID
    assert release.verified is True
    assert release.locked is True
    assert release.artifact_sha256 == fixture.bundle.manifest_sha256.removeprefix("sha256:")
    assert len(release.artifact_sha256) == 64
    assert ":" not in release.artifact_sha256
    runtime_bundle = concrete.get_verified_bundle(RELEASE_ID)
    assert runtime_bundle.manifest.manifest_sha256 == fixture.bundle.manifest_sha256
    assert set(runtime_bundle.files_by_artifact_id) == {
        parent.artifact_id for parent in fixture.bundle.parents
    }
    assert str(tmp_path) not in repr(concrete)


def test_unknown_requested_release_is_generic_and_path_free(tmp_path: Path) -> None:
    fixture = _create_bundle(tmp_path)
    provider = _provider(fixture)

    with pytest.raises(ResourceNotFoundError) as captured:
        provider.get_release("release-unknown")

    assert str(captured.value) == "release not found"
    assert str(tmp_path) not in str(captured.value)


def test_wrong_configured_release_identity_fails_closed(tmp_path: Path) -> None:
    fixture = _create_bundle(tmp_path)
    provider = _provider(fixture, expected_release_id="release-002")

    _assert_unavailable_without_path(provider, tmp_path, release_id="release-002")


def test_wrong_expected_compatibility_fails_closed(tmp_path: Path) -> None:
    fixture = _create_bundle(tmp_path)
    incompatible = TrustBundleCompatibility.canonical(model_member_count=2)
    provider = _provider(fixture, expected_compatibility=incompatible)

    _assert_unavailable_without_path(provider, tmp_path)


def test_parent_mutation_after_success_is_reverified_and_fails_closed(tmp_path: Path) -> None:
    fixture = _create_bundle(tmp_path)
    provider = _provider(fixture)
    checkpoint = fixture.parent_paths[ArtifactRole.CHECKPOINT]
    original = checkpoint.read_bytes()

    assert provider.is_ready() is True
    assert provider.get_release(RELEASE_ID).verified is True
    checkpoint.write_bytes(b"X" * len(original))

    _assert_unavailable_without_path(provider, tmp_path)
    with pytest.raises(DependencyUnavailableError, match="verified release is unavailable"):
        provider.get_verified_bundle(RELEASE_ID)

    checkpoint.write_bytes(original)
    assert provider.is_ready() is True
    assert provider.get_release(RELEASE_ID).artifact_sha256 == (
        fixture.bundle.manifest_sha256.removeprefix("sha256:")
    )


def test_manifest_mutation_after_success_is_reverified_and_fails_closed(tmp_path: Path) -> None:
    fixture = _create_bundle(tmp_path)
    provider = _provider(fixture)
    original = fixture.manifest_path.read_bytes()

    assert provider.is_ready() is True
    mutated = original.replace(b'"code_commit":"aaaa', b'"code_commit":"baaa', 1)
    assert mutated != original
    fixture.manifest_path.write_bytes(mutated)

    _assert_unavailable_without_path(provider, tmp_path)


def test_missing_manifest_is_a_readiness_failure_not_constructor_failure(tmp_path: Path) -> None:
    provider = TrustBundleReleaseProvider(
        manifest_path=tmp_path / "missing-manifest.json",
        artifact_root=tmp_path,
        expected_compatibility=TrustBundleCompatibility.canonical(),
        expected_release_id=RELEASE_ID,
    )

    _assert_unavailable_without_path(provider, tmp_path)


def test_missing_artifact_root_fails_closed(tmp_path: Path) -> None:
    provider = TrustBundleReleaseProvider(
        manifest_path=tmp_path / "missing" / "trust-bundle.json",
        artifact_root=tmp_path / "missing",
        expected_compatibility=TrustBundleCompatibility.canonical(),
        expected_release_id=RELEASE_ID,
    )

    _assert_unavailable_without_path(provider, tmp_path)


def test_malformed_manifest_fails_closed_without_decoder_details(tmp_path: Path) -> None:
    malformed = tmp_path / "trust-bundle.json"
    malformed.write_bytes(b'{"local_path":"C:\\\\private\\\\model.pt"')
    provider = TrustBundleReleaseProvider(
        manifest_path=malformed,
        artifact_root=tmp_path,
        expected_compatibility=TrustBundleCompatibility.canonical(),
        expected_release_id=RELEASE_ID,
    )

    _assert_unavailable_without_path(provider, tmp_path)
    with pytest.raises(DependencyUnavailableError) as captured:
        provider.get_release(RELEASE_ID)
    assert "private" not in str(captured.value).lower()
    assert "local_path" not in str(captured.value)


def test_missing_parent_after_initial_success_fails_closed(tmp_path: Path) -> None:
    fixture = _create_bundle(tmp_path)
    provider = _provider(fixture)
    assert provider.is_ready() is True

    fixture.parent_paths[ArtifactRole.QUALITY_POLICY].unlink()

    _assert_unavailable_without_path(provider, tmp_path)


def test_provider_is_frozen_and_configuration_rejects_nonservice_release_id(
    tmp_path: Path,
) -> None:
    fixture = _create_bundle(tmp_path)
    provider = _provider(fixture)
    attribute_name = "_expected_release_id"
    with pytest.raises(FrozenInstanceError):
        setattr(provider, attribute_name, "release-mutated")

    with pytest.raises(ValueError, match="bounded opaque"):
        TrustBundleReleaseProvider(
            manifest_path=fixture.manifest_path,
            artifact_root=fixture.root,
            expected_compatibility=fixture.compatibility,
            expected_release_id="../release",
        )


def test_concurrent_readiness_and_release_reads_are_consistent(tmp_path: Path) -> None:
    fixture = _create_bundle(tmp_path)
    provider = _provider(fixture)
    expected_digest = fixture.bundle.manifest_sha256.removeprefix("sha256:")

    def read_boundary(index: int) -> str:
        if index % 2 == 0:
            return "ready" if provider.is_ready() else "not-ready"
        return provider.get_release(RELEASE_ID).artifact_sha256

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = tuple(executor.map(read_boundary, range(40)))

    assert results.count("ready") == 20
    assert results.count(expected_digest) == 20
    assert "not-ready" not in results
