from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ecg_trust.audit_artifacts import (
    AuditArtifactError,
    AuditArtifactIntegrityError,
    load_audit_array_artifact,
    save_audit_array_artifact,
)


def _arrays() -> dict[str, np.ndarray]:
    return {
        "ecg_id": np.asarray([11, 12, 13], dtype=np.int64),
        "scores": np.asarray([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float32),
    }


def test_round_trip_is_immutable_and_integrity_bound(tmp_path: Path) -> None:
    path = tmp_path / "audit.npz"
    files = save_audit_array_artifact(
        path,
        artifact_type="ecg_trust.test_audit_arrays",
        arrays=_arrays(),
        metadata={"spec_sha256": "sha256:" + "a" * 64, "case": "clean"},
    )
    artifact = load_audit_array_artifact(
        path,
        expected_artifact_type="ecg_trust.test_audit_arrays",
        expected_metadata={"spec_sha256": "sha256:" + "a" * 64, "case": "clean"},
    )
    assert files.artifact_sha256 == artifact.artifact_sha256
    assert files.npz_sha256 == artifact.npz_sha256
    np.testing.assert_array_equal(artifact.arrays["ecg_id"], [11, 12, 13])
    assert not artifact.arrays["scores"].flags.writeable
    with pytest.raises(FileExistsError):
        save_audit_array_artifact(
            path,
            artifact_type="ecg_trust.test_audit_arrays",
            arrays=_arrays(),
            metadata={"case": "clean"},
        )


def test_sidecar_and_npz_mutation_are_detected(tmp_path: Path) -> None:
    path = tmp_path / "audit.npz"
    save_audit_array_artifact(
        path,
        artifact_type="ecg_trust.test_audit_arrays",
        arrays=_arrays(),
        metadata={"case": "clean"},
    )
    sidecar_path = path.with_suffix(".json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["metadata"]["case"] = "changed"
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    with pytest.raises(AuditArtifactIntegrityError, match="self-hash"):
        load_audit_array_artifact(path)

    sidecar["metadata"]["case"] = "clean"
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(AuditArtifactIntegrityError, match="size"):
        load_audit_array_artifact(path)


def test_rejects_nonfinite_object_and_invalid_metadata(tmp_path: Path) -> None:
    with pytest.raises(AuditArtifactError, match="finite"):
        save_audit_array_artifact(
            tmp_path / "nan.npz",
            artifact_type="ecg_trust.test_audit_arrays",
            arrays={"values": np.asarray([np.nan])},
            metadata={},
        )
    with pytest.raises(AuditArtifactError, match="unsupported"):
        save_audit_array_artifact(
            tmp_path / "object.npz",
            artifact_type="ecg_trust.test_audit_arrays",
            arrays={"values": np.asarray([object()], dtype=object)},
            metadata={},
        )
    with pytest.raises(AuditArtifactError, match="finite JSON"):
        save_audit_array_artifact(
            tmp_path / "metadata.npz",
            artifact_type="ecg_trust.test_audit_arrays",
            arrays=_arrays(),
            metadata={"bad": float("inf")},
        )
