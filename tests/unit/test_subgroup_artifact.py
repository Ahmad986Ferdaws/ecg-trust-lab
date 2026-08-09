from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd  # type: ignore[import-untyped]
import pytest

import ecg_trust.final_batch as final_batch_module
from ecg_trust.final_batch import FinalBatchSettings, run_final_batch
from ecg_trust.protocol import FINAL_TEST_CONFIRMATION, ExperimentProtocol
from ecg_trust.release_gates import ReleaseIntegrityError
from ecg_trust.subgroup_artifact import (
    SubgroupArtifactError,
    SubgroupIntegrityError,
    build_subgroup_artifact,
    canonical_sha256,
    load_subgroup_artifact,
    save_subgroup_artifact,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ecg_id": [3, 1, 2, 4],
            "patient_id": [30, 10, 20, 40],
            "strat_fold": [10, 1, 10, 10],
            "age": [300.0, 60.0, 20.0, 55.0],
            "sex": [1, 0, 0, 1],
        }
    )


def _manifest(tmp_path: Path, frame: pd.DataFrame | None = None) -> Path:
    path = tmp_path / "manifest.csv"
    (frame if frame is not None else _frame()).to_csv(path, index=False)
    return path


def test_subgroup_artifact_round_trip_is_sorted_label_free_and_immutable(
    tmp_path: Path,
) -> None:
    protocol = ExperimentProtocol.canonical()
    manifest = _manifest(tmp_path)
    artifact = build_subgroup_artifact(manifest, protocol=protocol)

    assert artifact.ecg_id == (2, 3, 4)
    assert artifact.patient_id == (20, 30, 40)
    assert artifact.sex == ("male", "female", "female")
    assert artifact.age_band == ("<40", "80+", "40-59")
    assert artifact.record_count == 3
    assert artifact.patient_count == 3
    assert artifact.artifact_sha256 is not None
    source = artifact.to_payload()["source_manifest"]
    assert isinstance(source, dict)
    assert source["columns_read"] == [
        "ecg_id",
        "patient_id",
        "strat_fold",
        "age",
        "sex",
    ]
    assert source["diagnostic_target_columns_read"] is False

    output, digest = save_subgroup_artifact(artifact, tmp_path / "subgroups.json")
    loaded = load_subgroup_artifact(output, protocol=protocol)
    assert loaded.to_payload() == artifact.to_payload()
    assert digest == artifact.artifact_sha256
    with pytest.raises(FileExistsError, match="immutable subgroup"):
        save_subgroup_artifact(artifact, output)


def test_builder_requests_only_demographic_columns_from_parquet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol = ExperimentProtocol.canonical()
    path = tmp_path / "manifest.parquet"
    path.write_bytes(b"synthetic parquet identity")
    observed: list[list[str] | None] = []

    def fake_read_parquet(
        source: Path, *, columns: list[str] | None = None
    ) -> pd.DataFrame:
        assert Path(source) == path
        observed.append(columns)
        return _frame()

    monkeypatch.setattr(pd, "read_parquet", fake_read_parquet)
    build_subgroup_artifact(path, protocol=protocol)
    assert observed == [
        ["ecg_id", "patient_id", "strat_fold", "age", "sex"]
    ]


def test_tampered_artifact_and_tampered_source_are_rejected(tmp_path: Path) -> None:
    protocol = ExperimentProtocol.canonical()
    manifest = _manifest(tmp_path)
    artifact = build_subgroup_artifact(manifest, protocol=protocol)
    output, _ = save_subgroup_artifact(artifact, tmp_path / "subgroups.json")

    payload = json.loads(output.read_text(encoding="utf-8"))
    payload["attributes"]["sex"][0] = "female"
    output.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SubgroupIntegrityError, match="self-hash"):
        load_subgroup_artifact(output, protocol=protocol)

    clean_output, _ = save_subgroup_artifact(
        artifact, tmp_path / "subgroups-clean.json"
    )
    frame = _frame()
    frame.loc[frame["ecg_id"].eq(2), "age"] = 21.0
    frame.to_csv(manifest, index=False)
    with pytest.raises(SubgroupIntegrityError, match="source manifest changed"):
        load_subgroup_artifact(clean_output, protocol=protocol)


def test_semantically_changed_definitions_fail_even_with_recomputed_hash(
    tmp_path: Path,
) -> None:
    protocol = ExperimentProtocol.canonical()
    artifact = build_subgroup_artifact(_manifest(tmp_path), protocol=protocol)
    output, _ = save_subgroup_artifact(artifact, tmp_path / "subgroups.json")
    payload = json.loads(output.read_text(encoding="utf-8"))
    payload["definitions"]["sex"]["mapping"]["0"] = "female"
    unhashed = dict(payload)
    del unhashed["artifact_sha256"]
    payload["artifact_sha256"] = canonical_sha256(unhashed)
    output.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SubgroupIntegrityError, match="definitions"):
        load_subgroup_artifact(output, protocol=protocol, verify_source=False)


def test_builder_rejects_noncanonical_demographics_and_patient_fold_leakage(
    tmp_path: Path,
) -> None:
    protocol = ExperimentProtocol.canonical()
    invalid_sex = _frame()
    invalid_sex.loc[invalid_sex["ecg_id"].eq(2), "sex"] = 2
    with pytest.raises(SubgroupArtifactError, match="sex must be 0, 1"):
        build_subgroup_artifact(
            _manifest(tmp_path, invalid_sex), protocol=protocol
        )

    leaked = _frame()
    leaked.loc[leaked["ecg_id"].eq(2), "patient_id"] = 10
    with pytest.raises(SubgroupArtifactError, match="patient occurs"):
        build_subgroup_artifact(_manifest(tmp_path, leaked), protocol=protocol)


@pytest.mark.parametrize("invalid_age", [121.0, 299.0, 301.0])
def test_builder_accepts_only_the_explicit_age_300_sentinel(
    tmp_path: Path, invalid_age: float
) -> None:
    protocol = ExperimentProtocol.canonical()
    invalid = _frame()
    invalid.loc[invalid["ecg_id"].eq(3), "age"] = invalid_age
    with pytest.raises(SubgroupArtifactError, match="censored sentinel 300"):
        build_subgroup_artifact(_manifest(tmp_path, invalid), protocol=protocol)


def test_expected_release_manifest_hash_is_enforced(tmp_path: Path) -> None:
    protocol = ExperimentProtocol.canonical()
    manifest = _manifest(tmp_path)
    with pytest.raises(SubgroupIntegrityError, match="release bundle"):
        build_subgroup_artifact(
            manifest,
            protocol=protocol,
            expected_manifest_sha256="sha256:" + "0" * 64,
        )


def test_final_batch_rejects_bad_subgroups_before_plan_or_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol = ExperimentProtocol.canonical()
    malformed = tmp_path / "malformed-subgroups.json"
    malformed.write_text(
        json.dumps({"ecg_id": [1], "attributes": {"sex": ["female"]}}),
        encoding="utf-8",
    )
    settings = FinalBatchSettings.create(
        output_directory=tmp_path / "final",
        subgroup_path=malformed,
        bootstrap_resamples=10,
    )
    manifest_hash = "sha256:" + "1" * 64
    monkeypatch.setattr(
        final_batch_module,
        "load_refit_bundle",
        lambda *args, **kwargs: SimpleNamespace(manifest_sha256=manifest_hash),
    )
    monkeypatch.setattr(
        final_batch_module,
        "load_calibration_bundle",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    plan_called = False

    def forbidden_plan(*args: object, **kwargs: object) -> object:
        nonlocal plan_called
        plan_called = True
        raise AssertionError("final plan must not be built for malformed subgroups")

    monkeypatch.setattr(final_batch_module, "build_final_batch_plan", forbidden_plan)
    ledger = tmp_path / "opening-ledger.json"
    with pytest.raises(ReleaseIntegrityError, match="invalid frozen subgroup"):
        run_final_batch(
            refit_bundle_path=tmp_path / "refits.json",
            calibration_bundle_path=tmp_path / "calibration.json",
            settings=settings,
            ledger_path=ledger,
            protocol=protocol,
            purpose="one-time preregistered final evaluation",
            operator="test",
            confirmation=FINAL_TEST_CONFIRMATION,
        )
    assert plan_called is False
    assert not ledger.exists()
