from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from ecg_trust.registry import (
    TRUST_BUNDLE_PARENT_SCHEMA_VERSION,
    ArtifactRole,
    TrustBundle,
    TrustBundleCompatibility,
    TrustBundleCompatibilityError,
    TrustBundleFormatError,
    TrustBundleIntegrityError,
    TrustBundleParent,
    bind_parent_file,
    canonical_json_bytes,
    load_and_verify_trust_bundle,
    load_trust_bundle_manifest,
    save_trust_bundle_manifest,
    seal_trust_bundle,
    verify_trust_bundle,
)

NOW = datetime(2026, 8, 24, 6, 30, tzinfo=UTC)


def _write_parent_files(root: Path, *, checkpoint_count: int = 1) -> None:
    files = {
        "resolved-config.json": b'{"architecture":"resnet1d"}\n',
        "normalization.json": b'{"mean":[0.0],"std":[1.0]}\n',
        "decision-policy.json": b'{"gate":"fold9"}\n',
        "quality-policy.json": b'{"quality_gate":"frozen"}\n',
        "distribution-policy.json": b'{"score_direction":"higher_is_ood"}\n',
        "conformal-policy.json": b'{"coverage_scope":"labelwise_marginal"}\n',
        "label-ontology.json": b'{"labels":["NORM","MI","STTC","CD","HYP"]}\n',
        "safety-document.md": b"Research use only. Not for clinical decisions.\n",
        "protocol.yaml": b"protocol: canonical\n",
        "dataset-manifest.json": b'{"dataset":"PTB-XL 1.0.3"}\n',
        "uv.lock": b"version = 1\n",
        "evidence.json": b'{"aggregate_only":true}\n',
    }
    for index in range(checkpoint_count):
        files[f"model-{index}.ckpt"] = f"checkpoint-{index}".encode()
    for name, content in files.items():
        (root / name).write_bytes(content)


def _parents(root: Path, *, checkpoint_count: int = 1) -> tuple[TrustBundleParent, ...]:
    declarations: list[tuple[str, ArtifactRole, str, str]] = [
        (
            "resolved-config",
            ArtifactRole.RESOLVED_CONFIG,
            "resolved-config.json",
            "application/json",
        ),
        ("normalization", ArtifactRole.NORMALIZATION, "normalization.json", "application/json"),
        (
            "decision-policy",
            ArtifactRole.DECISION_POLICY,
            "decision-policy.json",
            "application/json",
        ),
        (
            "quality-policy",
            ArtifactRole.QUALITY_POLICY,
            "quality-policy.json",
            "application/json",
        ),
        (
            "distribution-policy",
            ArtifactRole.DISTRIBUTION_POLICY,
            "distribution-policy.json",
            "application/json",
        ),
        (
            "conformal-policy",
            ArtifactRole.CONFORMAL_POLICY,
            "conformal-policy.json",
            "application/json",
        ),
        (
            "label-ontology",
            ArtifactRole.LABEL_ONTOLOGY,
            "label-ontology.json",
            "application/json",
        ),
        (
            "safety-document",
            ArtifactRole.SAFETY_DOCUMENT,
            "safety-document.md",
            "text/markdown",
        ),
        ("protocol", ArtifactRole.PROTOCOL, "protocol.yaml", "application/yaml"),
        (
            "dataset-manifest",
            ArtifactRole.DATASET_MANIFEST,
            "dataset-manifest.json",
            "application/json",
        ),
        ("environment-lock", ArtifactRole.ENVIRONMENT_LOCK, "uv.lock", "text/plain"),
        ("evidence", ArtifactRole.EVIDENCE_BUNDLE, "evidence.json", "application/json"),
    ]
    declarations.extend(
        (
            f"checkpoint-{index}",
            ArtifactRole.CHECKPOINT,
            f"model-{index}.ckpt",
            "application/octet-stream",
        )
        for index in range(checkpoint_count)
    )
    bound = [
        bind_parent_file(
            root,
            artifact_id=artifact_id,
            role=role,
            relative_path=relative_path,
            media_type=media_type,
        )
        for artifact_id, role, relative_path, media_type in declarations
    ]
    return tuple(sorted(bound, key=lambda parent: (parent.role.value, parent.artifact_id)))


def _bundle(
    root: Path,
    *,
    release_id: str = "release-001",
    checkpoint_count: int = 1,
) -> tuple[TrustBundle, TrustBundleCompatibility]:
    _write_parent_files(root, checkpoint_count=checkpoint_count)
    parents = _parents(root, checkpoint_count=checkpoint_count)
    by_role = {
        parent.role: parent for parent in parents if parent.role is not ArtifactRole.CHECKPOINT
    }
    compatibility = TrustBundleCompatibility.canonical(model_member_count=checkpoint_count)
    bundle = seal_trust_bundle(
        release_id=release_id,
        created_at=NOW,
        code_commit="a" * 40,
        protocol_sha256=by_role[ArtifactRole.PROTOCOL].file_sha256,
        dataset_manifest_sha256=by_role[ArtifactRole.DATASET_MANIFEST].file_sha256,
        environment_lock_sha256=by_role[ArtifactRole.ENVIRONMENT_LOCK].file_sha256,
        compatibility=compatibility,
        parents=parents,
    )
    return bundle, compatibility


def test_manifest_round_trip_is_canonical_immutable_and_fully_verified(tmp_path: Path) -> None:
    bundle, compatibility = _bundle(tmp_path)
    manifest = tmp_path / "trust-bundle.json"
    save_trust_bundle_manifest(manifest, bundle)

    assert manifest.read_bytes() == canonical_json_bytes(bundle.model_dump(mode="json")) + b"\n"
    loaded = load_trust_bundle_manifest(manifest)
    assert loaded == bundle
    verified = load_and_verify_trust_bundle(
        manifest,
        artifact_root=tmp_path,
        expected_compatibility=compatibility,
        expected_release_id="release-001",
    )
    assert verified.manifest.manifest_sha256 == bundle.manifest_sha256
    assert set(verified.files_by_artifact_id) == {parent.artifact_id for parent in bundle.parents}
    with pytest.raises(FileExistsError, match="immutable"):
        save_trust_bundle_manifest(manifest, bundle)


def test_manifest_rejects_unknown_fields_wrong_self_hash_and_noncanonical_json(
    tmp_path: Path,
) -> None:
    bundle, _ = _bundle(tmp_path)
    manifest = tmp_path / "trust-bundle.json"

    payload = bundle.model_dump(mode="json")
    payload["unknown"] = "field"
    manifest.write_bytes(canonical_json_bytes(payload) + b"\n")
    with pytest.raises(TrustBundleFormatError, match="could not decode"):
        load_trust_bundle_manifest(manifest)

    payload = bundle.model_dump(mode="json")
    payload["manifest_sha256"] = "sha256:" + "0" * 64
    manifest.write_bytes(canonical_json_bytes(payload) + b"\n")
    with pytest.raises(TrustBundleFormatError, match="could not decode"):
        load_trust_bundle_manifest(manifest)

    manifest.write_text(
        json.dumps(bundle.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8"
    )
    with pytest.raises(TrustBundleFormatError, match="not canonical"):
        load_trust_bundle_manifest(manifest)


def test_parent_tampering_size_change_and_missing_file_fail_closed(tmp_path: Path) -> None:
    bundle, compatibility = _bundle(tmp_path)
    checkpoint = tmp_path / "model-0.ckpt"

    checkpoint.write_bytes(b"checkpoint-X")
    with pytest.raises(TrustBundleIntegrityError, match="SHA-256"):
        verify_trust_bundle(
            bundle,
            tmp_path,
            expected_compatibility=compatibility,
        )

    checkpoint.write_bytes(b"different-size")
    with pytest.raises(TrustBundleIntegrityError, match="size mismatch"):
        verify_trust_bundle(
            bundle,
            tmp_path,
            expected_compatibility=compatibility,
        )

    checkpoint.unlink()
    with pytest.raises(TrustBundleIntegrityError, match="unavailable"):
        verify_trust_bundle(
            bundle,
            tmp_path,
            expected_compatibility=compatibility,
        )


def test_runtime_compatibility_and_release_identity_are_explicit(tmp_path: Path) -> None:
    bundle, compatibility = _bundle(tmp_path)
    with pytest.raises(TrustBundleCompatibilityError, match="compatibility"):
        verify_trust_bundle(
            bundle,
            tmp_path,
            expected_compatibility=TrustBundleCompatibility.canonical(model_member_count=2),
        )
    with pytest.raises(TrustBundleCompatibilityError, match="release_id"):
        verify_trust_bundle(
            bundle,
            tmp_path,
            expected_compatibility=compatibility,
            expected_release_id="release-002",
        )


@pytest.mark.parametrize(
    "relative_path",
    ["../checkpoint.ckpt", "/checkpoint.ckpt", "folder\\checkpoint.ckpt", "C:/x.ckpt"],
)
def test_parent_paths_cannot_escape_or_use_platform_specific_text(relative_path: str) -> None:
    with pytest.raises(ValidationError, match="relative_path"):
        TrustBundleParent(
            schema_version=TRUST_BUNDLE_PARENT_SCHEMA_VERSION,
            artifact_id="checkpoint",
            role=ArtifactRole.CHECKPOINT,
            relative_path=relative_path,
            size_bytes=1,
            file_sha256="sha256:" + "a" * 64,
            media_type="application/octet-stream",
        )


def test_release_graph_requires_every_parent_and_exact_checkpoint_count(tmp_path: Path) -> None:
    _write_parent_files(tmp_path, checkpoint_count=2)
    parents = _parents(tmp_path, checkpoint_count=2)
    singleton = {
        parent.role: parent for parent in parents if parent.role is not ArtifactRole.CHECKPOINT
    }
    compatibility = TrustBundleCompatibility.canonical(model_member_count=2)

    with pytest.raises(ValidationError, match="CHECKPOINT parent count"):
        seal_trust_bundle(
            release_id="release-incomplete",
            created_at=NOW,
            code_commit="b" * 40,
            protocol_sha256=singleton[ArtifactRole.PROTOCOL].file_sha256,
            dataset_manifest_sha256=singleton[ArtifactRole.DATASET_MANIFEST].file_sha256,
            environment_lock_sha256=singleton[ArtifactRole.ENVIRONMENT_LOCK].file_sha256,
            compatibility=compatibility,
            parents=tuple(parent for parent in parents if parent.artifact_id != "checkpoint-1"),
        )

    with pytest.raises(ValidationError, match="exactly one EVIDENCE_BUNDLE"):
        seal_trust_bundle(
            release_id="release-incomplete",
            created_at=NOW,
            code_commit="b" * 40,
            protocol_sha256=singleton[ArtifactRole.PROTOCOL].file_sha256,
            dataset_manifest_sha256=singleton[ArtifactRole.DATASET_MANIFEST].file_sha256,
            environment_lock_sha256=singleton[ArtifactRole.ENVIRONMENT_LOCK].file_sha256,
            compatibility=compatibility,
            parents=tuple(
                parent for parent in parents if parent.role is not ArtifactRole.EVIDENCE_BUNDLE
            ),
        )


@pytest.mark.parametrize(
    "required_role",
    [
        ArtifactRole.QUALITY_POLICY,
        ArtifactRole.DISTRIBUTION_POLICY,
        ArtifactRole.CONFORMAL_POLICY,
        ArtifactRole.LABEL_ONTOLOGY,
        ArtifactRole.SAFETY_DOCUMENT,
    ],
)
def test_vnext_policy_ontology_and_safety_parents_are_independently_required(
    tmp_path: Path,
    required_role: ArtifactRole,
) -> None:
    _write_parent_files(tmp_path)
    parents = _parents(tmp_path)
    singleton = {
        parent.role: parent for parent in parents if parent.role is not ArtifactRole.CHECKPOINT
    }
    compatibility = TrustBundleCompatibility.canonical()

    with pytest.raises(ValidationError, match=f"exactly one {required_role.value}"):
        seal_trust_bundle(
            release_id="release-missing-vnext-parent",
            created_at=NOW,
            code_commit="c" * 40,
            protocol_sha256=singleton[ArtifactRole.PROTOCOL].file_sha256,
            dataset_manifest_sha256=singleton[ArtifactRole.DATASET_MANIFEST].file_sha256,
            environment_lock_sha256=singleton[ArtifactRole.ENVIRONMENT_LOCK].file_sha256,
            compatibility=compatibility,
            parents=tuple(parent for parent in parents if parent.role is not required_role),
        )
