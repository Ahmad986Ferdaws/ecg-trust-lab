from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from ecg_trust.predictions import (
    PredictionAlignmentError,
    PredictionArtifact,
    PredictionArtifactError,
    PredictionIntegrityError,
    assert_prediction_artifacts_aligned,
    create_prediction_artifact,
    load_prediction_artifact,
    save_prediction_artifact,
)
from ecg_trust.protocol import (
    FINAL_TEST_CONFIRMATION,
    ExperimentProtocol,
    FinalTestAccessError,
    FinalTestAccessToken,
    FoldRole,
    authorize_final_test_access,
)


def _hash(value: str, *, prefix: bool = True) -> str:
    digest = hashlib.sha256(value.encode()).hexdigest()
    return f"sha256:{digest}" if prefix else digest


ArrayInputs = dict[str, NDArray[np.generic]]


def _arrays(fold: int = 8) -> ArrayInputs:
    ecg_id = np.asarray([103, 101, 104, 102])
    targets = np.asarray(
        [
            [1, 0, 0, 1, 0],
            [0, 1, 0, 0, 1],
            [1, 1, 0, 0, 0],
            [0, 0, 1, 1, 0],
        ]
    )
    logits = (targets * 4.0) - 2.0
    return {
        "ecg_id": ecg_id,
        "patient_id": np.asarray([13, 11, 14, 12]),
        "strat_fold": np.full(4, fold, dtype=np.int64),
        "targets": targets,
        "raw_logits": logits,
        "calibrated_probabilities": 1.0 / (1.0 + np.exp(-logits / 1.5)),
    }


def _create_from_arrays(
    arrays: ArrayInputs,
    *,
    model_name: str = "resnet1d",
    model_seed: int = 1,
    protocol: ExperimentProtocol | None = None,
    config_hash: str | None = None,
    manifest_hash: str | None = None,
    fold_role: FoldRole = FoldRole.MODEL_SELECTION,
    extra_metadata: dict[str, str | int | float | bool | None] | None = None,
    test_access: FinalTestAccessToken | None = None,
) -> PredictionArtifact:
    resolved_protocol = protocol or ExperimentProtocol.canonical()
    return create_prediction_artifact(
        ecg_id=arrays["ecg_id"],
        patient_id=arrays["patient_id"],
        strat_fold=arrays["strat_fold"],
        targets=arrays["targets"],
        raw_logits=arrays["raw_logits"],
        calibrated_probabilities=arrays.get("calibrated_probabilities"),
        model_name=model_name,
        model_seed=model_seed,
        protocol=resolved_protocol,
        config_hash=config_hash or _hash(f"config-{model_name}"),
        manifest_hash=manifest_hash or _hash("manifest", prefix=False),
        fold_role=fold_role,
        created_at_utc="2026-08-08T12:00:00Z",
        extra_metadata=extra_metadata,
        test_access=test_access,
    )


def _artifact(
    *,
    fold: int = 8,
    role: FoldRole = FoldRole.MODEL_SELECTION,
    model_name: str = "resnet1d",
    test_access: FinalTestAccessToken | None = None,
) -> PredictionArtifact:
    return _create_from_arrays(
        _arrays(fold),
        model_name=model_name,
        model_seed=7,
        fold_role=role,
        extra_metadata={"checkpoint": "best.pt", "epoch": 12},
        test_access=test_access,
    )


def test_create_sorts_rows_freezes_arrays_and_normalizes_hashes() -> None:
    artifact = _artifact()

    assert artifact.ecg_id.tolist() == [101, 102, 103, 104]
    assert artifact.patient_id.tolist() == [11, 12, 13, 14]
    assert artifact.targets[0].tolist() == [0, 1, 0, 0, 1]
    assert artifact.fold_role is FoldRole.MODEL_SELECTION
    assert artifact.folds == (8,)
    assert artifact.label_order == ("NORM", "MI", "STTC", "CD", "HYP")
    assert artifact.manifest_hash.startswith("sha256:")
    assert artifact.extra_metadata["epoch"] == 12
    assert artifact.raw_logits.flags.writeable is False
    with pytest.raises(ValueError):
        artifact.raw_logits[0, 0] = 99.0
    with pytest.raises(TypeError):
        artifact.extra_metadata["new"] = True  # type: ignore[index]


def test_atomic_npz_json_round_trip_and_expected_hashes(tmp_path: Path) -> None:
    artifact = _artifact()
    protocol = ExperimentProtocol.canonical()
    destination = tmp_path / "resnet-fold8.npz"

    stored = save_prediction_artifact(artifact, destination, protocol=protocol)
    loaded = load_prediction_artifact(
        stored.json_path,
        protocol=protocol,
        expected_config_hash=artifact.config_hash,
        expected_manifest_hash=artifact.manifest_hash,
    )

    assert stored.npz_path == destination
    assert stored.json_path == destination.with_suffix(".json")
    assert stored.npz_sha256.startswith("sha256:")
    assert loaded.integrity_sha256 == stored.artifact_sha256
    assert np.array_equal(loaded.ecg_id, artifact.ecg_id)
    assert np.array_equal(loaded.patient_id, artifact.patient_id)
    assert np.array_equal(loaded.strat_fold, artifact.strat_fold)
    assert np.array_equal(loaded.targets, artifact.targets)
    assert np.array_equal(loaded.raw_logits, artifact.raw_logits)
    assert loaded.calibrated_probabilities is not None
    assert artifact.calibrated_probabilities is not None
    assert np.array_equal(
        loaded.calibrated_probabilities,
        artifact.calibrated_probabilities,
    )
    metadata = json.loads(stored.json_path.read_text(encoding="utf-8"))
    assert metadata["arrays"]["targets"] == {"dtype": "int8", "shape": [4, 5]}
    assert metadata["alignment_sha256"] == loaded.alignment_sha256
    assert not list(tmp_path.glob("*.tmp"))


def test_existing_artifact_pair_is_never_overwritten(tmp_path: Path) -> None:
    artifact = _artifact()
    protocol = ExperimentProtocol.canonical()
    destination = tmp_path / "predictions.npz"
    first = save_prediction_artifact(artifact, destination, protocol=protocol)
    original_npz = first.npz_path.read_bytes()
    original_json = first.json_path.read_bytes()

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        save_prediction_artifact(artifact, destination, protocol=protocol)

    assert first.npz_path.read_bytes() == original_npz
    assert first.json_path.read_bytes() == original_json


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda arrays: arrays.update(ecg_id=np.asarray([1, 1, 2, 3])), "unique"),
        (lambda arrays: arrays.update(raw_logits=np.full((4, 5), np.nan)), "finite"),
        (
            lambda arrays: arrays.update(calibrated_probabilities=np.full((4, 5), 1.1)),
            r"\[0, 1\]",
        ),
        (lambda arrays: arrays.update(targets=np.zeros((4, 4))), "shape"),
    ],
)
def test_prediction_array_contract_rejects_invalid_data(
    mutation: Callable[[ArrayInputs], None],
    message: str,
) -> None:
    arrays = _arrays()
    mutation(arrays)

    with pytest.raises(PredictionArtifactError, match=message):
        _create_from_arrays(arrays)


def test_patient_fold_consistency_and_role_membership_are_enforced() -> None:
    arrays = _arrays(fold=1)
    arrays["patient_id"] = np.asarray([11, 11, 12, 13])
    arrays["strat_fold"] = np.asarray([1, 2, 1, 1])

    with pytest.raises(PredictionArtifactError, match="multiple folds"):
        _create_from_arrays(arrays, fold_role=FoldRole.TRAIN)

    with pytest.raises(PredictionArtifactError, match="permits"):
        _create_from_arrays(_arrays(fold=10), fold_role=FoldRole.TRAIN)


def test_final_test_creation_save_and_load_each_require_protocol_token(tmp_path: Path) -> None:
    protocol = ExperimentProtocol.canonical()
    with pytest.raises(FinalTestAccessError, match="fold 10 is sealed"):
        _create_from_arrays(
            _arrays(fold=10), protocol=protocol, fold_role=FoldRole.FINAL_TEST
        )

    token = authorize_final_test_access(
        protocol,
        purpose="write locked final predictions",
        confirmation=FINAL_TEST_CONFIRMATION,
    )
    artifact = _artifact(
        fold=10,
        role=FoldRole.FINAL_TEST,
        test_access=token,
    )
    destination = tmp_path / "final.npz"
    with pytest.raises(FinalTestAccessError, match="fold 10 is sealed"):
        save_prediction_artifact(artifact, destination, protocol=protocol)

    stored = save_prediction_artifact(
        artifact, destination, protocol=protocol, test_access=token
    )
    with pytest.raises(FinalTestAccessError, match="fold 10 is sealed"):
        load_prediction_artifact(stored.npz_path, protocol=protocol)
    loaded = load_prediction_artifact(
        stored.npz_path, protocol=protocol, test_access=token
    )
    assert loaded.fold_role is FoldRole.FINAL_TEST
    assert loaded.folds == (10,)


def test_npz_and_json_tampering_are_detected(tmp_path: Path) -> None:
    protocol = ExperimentProtocol.canonical()
    first = save_prediction_artifact(
        _artifact(), tmp_path / "npz-tamper.npz", protocol=protocol
    )
    content = bytearray(first.npz_path.read_bytes())
    content[len(content) // 2] ^= 0x01
    first.npz_path.write_bytes(content)
    with pytest.raises(PredictionIntegrityError, match="SHA-256 mismatch"):
        load_prediction_artifact(first.npz_path, protocol=protocol)

    second = save_prediction_artifact(
        _artifact(), tmp_path / "json-tamper.npz", protocol=protocol
    )
    metadata = json.loads(second.json_path.read_text(encoding="utf-8"))
    metadata["model"]["name"] = "tampered"
    second.json_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(PredictionIntegrityError, match="sidecar integrity"):
        load_prediction_artifact(second.json_path, protocol=protocol)


def test_alignment_guard_supports_paired_bootstrap_and_detects_target_drift() -> None:
    first = _artifact(model_name="resnet1d")
    second = _artifact(model_name="transformer")
    assert_prediction_artifacts_aligned(first, second)

    changed = _arrays()
    changed["targets"] = changed["targets"].copy()
    changed["targets"][0, 0] = 1 - changed["targets"][0, 0]
    drifted = _create_from_arrays(
        changed,
        model_name="transformer",
        model_seed=7,
        config_hash=_hash("transformer"),
        fold_role=FoldRole.MODEL_SELECTION,
    )
    with pytest.raises(PredictionAlignmentError, match="alignment_sha256"):
        assert_prediction_artifacts_aligned(first, drifted)


def test_probability_fallback_and_calibrated_requirement() -> None:
    arrays = _arrays()
    arrays.pop("calibrated_probabilities")
    artifact = _create_from_arrays(arrays, model_seed=7)

    expected = 1.0 / (1.0 + np.exp(-artifact.raw_logits))
    assert np.allclose(artifact.probabilities(), expected)
    assert artifact.probabilities().flags.writeable is False
    with pytest.raises(PredictionArtifactError, match="no calibrated probabilities"):
        artifact.probabilities(require_calibrated=True)


def test_invalid_save_suffix_and_expected_provenance_mismatch(tmp_path: Path) -> None:
    artifact = _artifact()
    protocol = ExperimentProtocol.canonical()
    with pytest.raises(PredictionArtifactError, match="must end in .npz"):
        save_prediction_artifact(artifact, tmp_path / "predictions.bin", protocol=protocol)

    stored = save_prediction_artifact(
        artifact, tmp_path / "predictions.npz", protocol=protocol
    )
    with pytest.raises(PredictionIntegrityError, match="manifest_hash"):
        load_prediction_artifact(
            stored.npz_path,
            protocol=protocol,
            expected_manifest_hash=_hash("different-manifest"),
        )
