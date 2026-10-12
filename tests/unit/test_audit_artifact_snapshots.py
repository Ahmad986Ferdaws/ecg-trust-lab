from __future__ import annotations

import json

import numpy as np
import pytest

from ecg_trust.audit_artifacts import load_audit_array_artifact, save_audit_array_artifact


def _saved(tmp_path, *, array=None):
    path = tmp_path / "snapshot.npz"
    metadata = {"summary": {"thresholds": [0.1, 0.2], "passed": True}}
    save_audit_array_artifact(
        path,
        artifact_type="ecg_trust.snapshot_test",
        arrays={"scores": np.array([0.5]) if array is None else array},
        metadata=metadata,
    )
    return path, metadata, load_audit_array_artifact(path)


@pytest.mark.parametrize("dtype", [np.float32, np.int64, np.bool_, "U4", "S4"])
def test_loaded_arrays_cannot_reenable_writes_through_their_backing_storage(tmp_path, dtype):
    source = np.array([[1, 0], [0, 1]], dtype=dtype)
    _, _, artifact = _saved(tmp_path, array=source)
    values = artifact.arrays["scores"]
    current = values
    while isinstance(current, np.ndarray):
        with pytest.raises(ValueError):
            current.setflags(write=True)
        current = current.base
    np.testing.assert_array_equal(values, source)
    assert values.dtype == source.dtype
    assert values.shape == source.shape


@pytest.mark.parametrize("access", ["direct", "items", "dict"])
def test_nested_metadata_edits_do_not_change_the_verified_snapshot(tmp_path, access):
    path, expected, artifact = _saved(tmp_path)
    original_files = (path.read_bytes(), path.with_suffix(".json").read_bytes())
    original_identity = artifact.artifact_sha256
    view = (
        artifact.metadata
        if access == "direct"
        else dict(artifact.metadata.items())
        if access == "items"
        else dict(artifact.metadata)
    )
    view["summary"]["thresholds"].append(0.9)
    view["summary"]["passed"] = False
    assert artifact.metadata == expected
    assert json.loads(json.dumps(dict(artifact.metadata))) == expected
    assert artifact.artifact_sha256 == original_identity
    assert (path.read_bytes(), path.with_suffix(".json").read_bytes()) == original_files
    assert load_audit_array_artifact(path, expected_metadata=expected).metadata == expected
