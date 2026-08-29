from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import ecg_trust.ood_completion.embedding_artifact as embedding_module
from ecg_trust.ood_completion.embedding_artifact import (
    EMBEDDING_ARTIFACT_TYPE,
    EMBEDDING_DIMENSION,
    EmbeddingArtifact,
    EmbeddingArtifactError,
    EmbeddingArtifactIntegrityError,
    EmbeddingRole,
    create_embedding_artifact,
    load_embedding_artifact,
    save_embedding_artifact,
)


def _hash(character: str) -> str:
    return "sha256:" + character * 64


def _inputs() -> dict[str, Any]:
    embedding = np.arange(4 * EMBEDDING_DIMENSION, dtype=np.float32).reshape(
        4, EMBEDDING_DIMENSION
    )
    return {
        "ecg_id": np.asarray([30, 10, 20, 40], dtype=np.int64),
        "patient_id": np.asarray([3, 1, 2, 3], dtype=np.int64),
        "strat_fold": np.full(4, 9, dtype=np.int8),
        "embedding": embedding,
        "role": EmbeddingRole.THRESHOLD_FIT,
        "expected_folds": (9,),
        "checkpoint_sha256": _hash("1"),
        "config_sha256": _hash("2"),
        "normalization_sha256": _hash("3"),
        "manifest_sha256": _hash("4"),
        "protocol_sha256": _hash("5"),
        "runtime_sha256": _hash("6"),
    }


def _artifact(**changes: Any) -> EmbeddingArtifact:
    values = _inputs()
    values.update(changes)
    return create_embedding_artifact(**values)


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _file_hash(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _reseal_sidecar(path: Path, payload: dict[str, object]) -> None:
    body = dict(payload)
    body.pop("artifact_sha256", None)
    payload = {
        **body,
        "artifact_sha256": "sha256:"
        + hashlib.sha256(_canonical(body).encode("utf-8")).hexdigest(),
    }
    path.write_text(_canonical(payload) + "\n", encoding="utf-8", newline="\n")


def test_create_sorts_rows_and_freezes_exact_private_array_contract() -> None:
    source = _inputs()
    artifact = create_embedding_artifact(**source)

    assert artifact.role is EmbeddingRole.THRESHOLD_FIT
    assert artifact.expected_folds == (9,)
    assert artifact.record_count == 4
    assert artifact.patient_count == 3
    assert artifact.ecg_id.tolist() == [10, 20, 30, 40]
    assert artifact.patient_id.tolist() == [1, 2, 3, 3]
    assert artifact.ecg_id.dtype == np.dtype(np.int64)
    assert artifact.patient_id.dtype == np.dtype(np.int64)
    assert artifact.strat_fold.dtype == np.dtype(np.int8)
    assert artifact.embedding.dtype == np.dtype(np.float32)
    assert artifact.embedding.shape == (4, EMBEDDING_DIMENSION)
    assert np.array_equal(artifact.embedding[0], source["embedding"][1])
    assert all(
        not array.flags.writeable
        for array in (
            artifact.ecg_id,
            artifact.patient_id,
            artifact.strat_fold,
            artifact.embedding,
        )
    )
    assert all(
        array.flags.c_contiguous
        for array in (
            artifact.ecg_id,
            artifact.patient_id,
            artifact.strat_fold,
            artifact.embedding,
        )
    )
    for array in (
        artifact.ecg_id,
        artifact.patient_id,
        artifact.strat_fold,
        artifact.embedding,
    ):
        with pytest.raises(ValueError):
            array.setflags(write=True)
    assert artifact.alignment_sha256.startswith("sha256:")
    assert artifact.embedding_tensor_sha256.startswith("sha256:")
    assert artifact.artifact_sha256 is None
    with pytest.raises(ValueError):
        artifact.embedding[0, 0] = 0.0


def test_logical_hashes_are_order_invariant_and_separate_identity_from_tensor() -> None:
    first = _artifact()
    values = _inputs()
    order = np.asarray([1, 2, 0, 3])
    reordered = _artifact(
        ecg_id=values["ecg_id"][order],
        patient_id=values["patient_id"][order],
        strat_fold=values["strat_fold"][order],
        embedding=values["embedding"][order],
    )
    changed_embedding = np.asarray(values["embedding"]).copy()
    changed_embedding[0, 0] += 1.0
    changed = _artifact(embedding=changed_embedding)

    assert first.alignment_sha256 == reordered.alignment_sha256
    assert first.embedding_tensor_sha256 == reordered.embedding_tensor_sha256
    assert first.alignment_sha256 == changed.alignment_sha256
    assert first.embedding_tensor_sha256 != changed.embedding_tensor_sha256


def test_canonical_atomic_round_trip_binds_arrays_counts_hashes_and_no_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = _artifact()
    destination = tmp_path / "private" / "threshold-fit-embeddings.npz"
    files = save_embedding_artifact(artifact, destination)
    observed_allow_pickle: list[object] = []
    original_load = np.load

    def tracked_load(*args: object, **kwargs: object) -> object:
        observed_allow_pickle.append(kwargs.get("allow_pickle"))
        return original_load(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        embedding_module.np,  # type: ignore[attr-defined]
        "load",
        tracked_load,
    )
    loaded = load_embedding_artifact(
        files.json_path,
        expected_artifact_sha256=files.artifact_sha256,
        expected_npz_file_sha256=files.npz_file_sha256,
        expected_role=EmbeddingRole.THRESHOLD_FIT,
    )

    assert observed_allow_pickle == [False]
    assert np.array_equal(loaded.ecg_id, artifact.ecg_id)
    assert np.array_equal(loaded.patient_id, artifact.patient_id)
    assert np.array_equal(loaded.strat_fold, artifact.strat_fold)
    assert np.array_equal(loaded.embedding, artifact.embedding)
    assert loaded.artifact_sha256 == files.artifact_sha256
    assert loaded.npz_file_sha256 == files.npz_file_sha256
    assert loaded.sidecar_file_sha256 == files.sidecar_file_sha256
    assert files.npz_file_sha256 == _file_hash(files.npz_path)
    assert files.sidecar_file_sha256 == _file_hash(files.json_path)
    for array in (loaded.ecg_id, loaded.patient_id, loaded.strat_fold, loaded.embedding):
        with pytest.raises(ValueError):
            array.setflags(write=True)

    raw = files.json_path.read_bytes()
    sidecar = json.loads(raw)
    assert raw == (_canonical(sidecar) + "\n").encode("utf-8")
    assert sidecar["artifact_type"] == EMBEDDING_ARTIFACT_TYPE
    assert sidecar["visibility"] == "PRIVATE"
    assert sidecar["contains_row_level_identifiers"] is True
    assert sidecar["record_count"] == 4
    assert sidecar["patient_count"] == 3
    assert sidecar["npz_file"] == destination.name
    assert sidecar["arrays"] == {
        "ecg_id": {"dtype": "int64", "shape": [4]},
        "patient_id": {"dtype": "int64", "shape": [4]},
        "strat_fold": {"dtype": "int8", "shape": [4]},
        "embedding": {"dtype": "float32", "shape": [4, EMBEDDING_DIMENSION]},
    }
    assert str(tmp_path) not in raw.decode("utf-8")
    assert "\\" not in str(sidecar["npz_file"])
    assert set(loaded.to_summary_dict()).isdisjoint({"ecg_id", "patient_id", "embedding"})

    with np.load(files.npz_path, allow_pickle=False) as archive:
        assert tuple(archive.files) == ("ecg_id", "patient_id", "strat_fold", "embedding")
        assert archive["ecg_id"].dtype == np.dtype(np.int64)
        assert archive["patient_id"].dtype == np.dtype(np.int64)
        assert archive["strat_fold"].dtype == np.dtype(np.int8)
        assert archive["embedding"].dtype == np.dtype(np.float32)


def test_existing_member_of_pair_is_never_overwritten(tmp_path: Path) -> None:
    destination = tmp_path / "embeddings.npz"
    files = save_embedding_artifact(_artifact(), destination)
    original_npz = files.npz_path.read_bytes()
    original_json = files.json_path.read_bytes()

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        save_embedding_artifact(_artifact(), destination)

    assert files.npz_path.read_bytes() == original_npz
    assert files.json_path.read_bytes() == original_json


def test_pair_commit_rolls_back_its_npz_when_sidecar_link_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_link = os.link
    calls = 0

    def fail_second_link(source: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise FileExistsError
        original_link(source, destination)

    monkeypatch.setattr(
        embedding_module.os,  # type: ignore[attr-defined]
        "link",
        fail_second_link,
    )
    destination = tmp_path / "embeddings.npz"

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        save_embedding_artifact(_artifact(), destination)

    assert not destination.exists()
    assert not destination.with_suffix(".json").exists()
    assert not list(tmp_path.glob(".*.tmp"))


def test_pair_commit_syncs_parent_after_each_published_member(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: list[Path] = []
    monkeypatch.setattr(
        embedding_module,
        "_fsync_directory",
        lambda path: observed.append(path),
    )
    destination = tmp_path / "private" / "embeddings.npz"

    save_embedding_artifact(_artifact(), destination)

    assert observed == [destination.parent] * 4


def test_directory_sync_failure_rolls_back_published_member(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0

    def fail_first_sync(_path: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise EmbeddingArtifactError("simulated directory sync failure")

    monkeypatch.setattr(embedding_module, "_fsync_directory", fail_first_sync)
    destination = tmp_path / "embeddings.npz"

    with pytest.raises(EmbeddingArtifactError, match="simulated directory sync failure"):
        save_embedding_artifact(_artifact(), destination)

    assert not destination.exists()
    assert not destination.with_suffix(".json").exists()
    assert not list(tmp_path.glob(".*.tmp"))


def test_success_cleanup_sync_failure_is_not_suppressed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0

    def fail_first_cleanup_sync(_path: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise EmbeddingArtifactError("simulated cleanup sync failure")

    monkeypatch.setattr(embedding_module, "_fsync_directory", fail_first_cleanup_sync)
    destination = tmp_path / "embeddings.npz"

    with pytest.raises(EmbeddingArtifactError, match="simulated cleanup sync failure"):
        save_embedding_artifact(_artifact(), destination)

    assert destination.is_file()
    assert destination.with_suffix(".json").is_file()
    assert len(list(tmp_path.glob(".*.tmp"))) == 1


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda values: values.update(ecg_id=np.asarray([30, 10, 10, 40])),
            "unique",
        ),
        (
            lambda values: values.update(ecg_id=np.asarray([1.0, 2.0, 3.0, 4.0])),
            "integer array",
        ),
        (
            lambda values: values.update(embedding=np.zeros((4, 511))),
            "shape",
        ),
        (
            lambda values: values.update(
                embedding=np.full((4, EMBEDDING_DIMENSION), np.nan)
            ),
            "finite",
        ),
        (
            lambda values: values.update(expected_folds=(8, 9)),
            "do not equal expected_folds",
        ),
        (
            lambda values: values.update(expected_folds=(9, 8)),
            "unique, and sorted",
        ),
        (
            lambda values: values.update(role="NOT_A_ROLE"),
            "unsupported embedding role",
        ),
    ],
)
def test_create_rejects_malformed_contracts(
    mutation: Callable[[dict[str, Any]], None], message: str
) -> None:
    values = _inputs()
    mutation(values)

    with pytest.raises(EmbeddingArtifactError, match=message):
        create_embedding_artifact(**values)


def test_patient_fold_consistency_is_enforced() -> None:
    with pytest.raises(EmbeddingArtifactError, match="patient occurs in multiple"):
        _artifact(
            patient_id=np.asarray([7, 7, 8, 9]),
            strat_fold=np.asarray([8, 9, 8, 9]),
            expected_folds=(8, 9),
            role=EmbeddingRole.REFERENCE,
        )


def test_npz_and_sidecar_tampering_are_detected(tmp_path: Path) -> None:
    first = save_embedding_artifact(_artifact(), tmp_path / "npz-tamper.npz")
    content = bytearray(first.npz_path.read_bytes())
    content[len(content) // 2] ^= 0x01
    first.npz_path.write_bytes(content)
    with pytest.raises(EmbeddingArtifactIntegrityError, match="SHA-256 mismatch"):
        load_embedding_artifact(first.npz_path)

    second = save_embedding_artifact(_artifact(), tmp_path / "json-tamper.npz")
    sidecar = json.loads(second.json_path.read_text(encoding="utf-8"))
    sidecar["record_count"] = 3
    second.json_path.write_text(
        _canonical(sidecar) + "\n", encoding="utf-8", newline="\n"
    )
    with pytest.raises(EmbeddingArtifactIntegrityError, match="self-hash mismatch"):
        load_embedding_artifact(second.json_path)

    third = save_embedding_artifact(_artifact(), tmp_path / "format-tamper.npz")
    sidecar = json.loads(third.json_path.read_text(encoding="utf-8"))
    third.json_path.write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(EmbeddingArtifactIntegrityError, match="not canonical JSON"):
        load_embedding_artifact(third.json_path)


def test_npz_hash_preflight_and_decode_share_one_immutable_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = save_embedding_artifact(_artifact(), tmp_path / "snapshot.npz")
    original_preflight = embedding_module._preflight_npz
    observed_snapshots: list[bytes] = []

    def replace_path_after_hash(snapshot: bytes, *, record_count: int) -> None:
        observed_snapshots.append(snapshot)
        files.npz_path.write_bytes(b"changed-after-snapshot")
        original_preflight(snapshot, record_count=record_count)

    monkeypatch.setattr(embedding_module, "_preflight_npz", replace_path_after_hash)

    loaded = load_embedding_artifact(
        files.npz_path,
        expected_artifact_sha256=files.artifact_sha256,
        expected_npz_file_sha256=files.npz_file_sha256,
    )

    assert len(observed_snapshots) == 1
    assert loaded.embedding_tensor_sha256 == _artifact().embedding_tensor_sha256
    assert files.npz_path.read_bytes() == b"changed-after-snapshot"


def test_resealed_path_and_dtype_tampering_are_still_rejected(tmp_path: Path) -> None:
    path_case = save_embedding_artifact(_artifact(), tmp_path / "path-case.npz")
    sidecar = json.loads(path_case.json_path.read_text(encoding="utf-8"))
    sidecar["npz_file"] = str((tmp_path / "path-case.npz").resolve())
    _reseal_sidecar(path_case.json_path, sidecar)
    with pytest.raises(EmbeddingArtifactIntegrityError, match="path-free filename"):
        load_embedding_artifact(path_case.json_path)

    dtype_case = save_embedding_artifact(_artifact(), tmp_path / "dtype-case.npz")
    with np.load(dtype_case.npz_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}
    with dtype_case.npz_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            ecg_id=arrays["ecg_id"],
            patient_id=arrays["patient_id"],
            strat_fold=arrays["strat_fold"],
            embedding=arrays["embedding"].view(np.int32),
        )
    sidecar = json.loads(dtype_case.json_path.read_text(encoding="utf-8"))
    sidecar["npz_size_bytes"] = dtype_case.npz_path.stat().st_size
    sidecar["npz_sha256"] = _file_hash(dtype_case.npz_path)
    _reseal_sidecar(dtype_case.json_path, sidecar)
    with pytest.raises(EmbeddingArtifactIntegrityError, match="embedding.*dtype"):
        load_embedding_artifact(dtype_case.npz_path)


def test_expected_external_identities_fail_closed(tmp_path: Path) -> None:
    files = save_embedding_artifact(_artifact(), tmp_path / "expected.npz")

    with pytest.raises(EmbeddingArtifactIntegrityError, match="artifact hash differs"):
        load_embedding_artifact(files.npz_path, expected_artifact_sha256=_hash("a"))
    with pytest.raises(EmbeddingArtifactIntegrityError, match="NPZ hash differs"):
        load_embedding_artifact(files.npz_path, expected_npz_file_sha256=_hash("b"))
    with pytest.raises(EmbeddingArtifactIntegrityError, match="role differs"):
        load_embedding_artifact(files.npz_path, expected_role=EmbeddingRole.REFERENCE)
