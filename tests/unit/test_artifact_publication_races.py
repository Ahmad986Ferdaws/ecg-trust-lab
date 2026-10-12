from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import numpy as np
import pytest

from ecg_trust import _artifact_storage, audit_artifacts, predictions
from ecg_trust.protocol import ExperimentProtocol, FoldRole


@pytest.fixture(params=["audit", "prediction"])
def writer(request):
    protocol = ExperimentProtocol.canonical()
    if request.param == "audit":
        module = audit_artifacts

        def save(path, value=0.5):
            return module.save_audit_array_artifact(
                path,
                artifact_type="ecg_trust.publication_test",
                arrays={"scores": np.array([value])},
                metadata={"value": value},
            )

        def load(path):
            return module.load_audit_array_artifact(path).artifact_sha256
    else:
        module = predictions

        def save(path, value=0.5):
            artifact = module.create_prediction_artifact(
                ecg_id=[1],
                patient_id=[1],
                strat_fold=[8],
                targets=[[1, 0, 0, 0, 0]],
                raw_logits=[[value] * 5],
                model_name="synthetic",
                model_seed=2026,
                protocol=protocol,
                config_hash="a" * 64,
                manifest_hash="b" * 64,
                fold_role=FoldRole.MODEL_SELECTION,
                created_at_utc="2026-10-11T00:00:00Z",
            )
            return module.save_prediction_artifact(artifact, path, protocol=protocol)

        def load(path):
            return module.load_prediction_artifact(path, protocol=protocol).integrity_sha256

    return module, save, load


@pytest.mark.parametrize("suffix", [".npz", ".json"])
def test_racing_destination_is_never_overwritten(tmp_path, monkeypatch, writer, suffix):
    module, save, _ = writer
    destination = tmp_path / "artifact.npz"
    collision = destination.with_suffix(suffix)
    write_json = module._write_json

    def create_collision(path, payload):
        write_json(path, payload)
        collision.write_bytes(b"other producer's evidence")

    monkeypatch.setattr(module, "_write_json", create_collision)
    with pytest.raises(FileExistsError):
        save(destination)
    assert collision.read_bytes() == b"other producer's evidence"
    assert set(tmp_path.iterdir()) == {collision}


@pytest.mark.parametrize("suffix", [".npz", ".json"])
def test_dangling_destination_symlink_is_preserved(tmp_path, writer, suffix):
    _, save, _ = writer
    destination = tmp_path / "artifact.npz"
    collision = destination.with_suffix(suffix)
    outside = tmp_path / "unrequested-target"
    collision.symlink_to(outside)
    with pytest.raises(FileExistsError):
        save(destination)
    assert collision.is_symlink()
    assert not outside.exists()
    assert set(tmp_path.iterdir()) == {collision}


def test_concurrent_producers_leave_exactly_one_verifiable_pair(tmp_path, monkeypatch, writer):
    module, save, load = writer
    destination = tmp_path / "artifact.npz"
    barrier = Barrier(2)
    write_json = module._write_json

    def stage_together(path, payload):
        write_json(path, payload)
        barrier.wait(timeout=10)

    monkeypatch.setattr(module, "_write_json", stage_together)

    def attempt(value):
        try:
            return save(destination, value)
        except FileExistsError:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(attempt, [0.25, 0.75]))
    winners = [result for result in results if result is not None]
    assert len(winners) == 1
    assert load(destination) == winners[0].artifact_sha256
    assert set(tmp_path.iterdir()) == {destination, destination.with_suffix(".json")}


def test_second_temporary_allocation_failure_removes_the_first(tmp_path, monkeypatch, writer):
    module, save, _ = writer
    temporary_path = module._temporary_path
    calls = 0

    def fail_second(path):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("synthetic staging failure")
        return temporary_path(path)

    monkeypatch.setattr(module, "_temporary_path", fail_second)
    with pytest.raises(OSError, match="staging failure"):
        save(tmp_path / "artifact.npz")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("fail_on", [1, 2])
def test_publication_failure_cleans_up_only_this_producers_files(
    tmp_path,
    monkeypatch,
    writer,
    fail_on,
):
    _, save, _ = writer
    real_link = _artifact_storage.os.link
    calls = 0

    def fail_link(source, destination):
        nonlocal calls
        calls += 1
        if calls == fail_on:
            raise OSError("synthetic publication failure")
        real_link(source, destination)

    monkeypatch.setattr(_artifact_storage.os, "link", fail_link)
    with pytest.raises(OSError, match="publication failure"):
        save(tmp_path / "artifact.npz")
    assert list(tmp_path.iterdir()) == []
