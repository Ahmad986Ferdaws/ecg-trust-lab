from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from ecg_trust.constants import EXPECTED_PATIENTS, EXPECTED_RECORDS, PTBXL_VERSION, SUPERCLASSES
from ecg_trust.data.manifest import (
    EXPECTED_SUPERCLASS_COUNTS,
    LABEL_COLUMNS,
    ManifestError,
    build_manifest,
    parse_scp_codes,
    parse_sha256sums,
    sha256_file,
    verify_sha256sums,
    write_manifest_artifacts,
)


def _write_dataset(root: Path) -> None:
    statements = pd.DataFrame(
        {
            "diagnostic": [1, 1, 1, 1, 1, 0, 1],
            "diagnostic_class": ["NORM", "MI", "STTC", "CD", "HYP", "", "UNKNOWN"],
        },
        index=["NORM_CODE", "MI_CODE", "ST_CODE", "CD_CODE", "HYP_CODE", "RHYTHM", "X"],
    )
    statements.to_csv(root / "scp_statements.csv")
    metadata = pd.DataFrame(
        {
            "ecg_id": [5, 2, 4, 1, 3, 6],
            "patient_id": [14, 11, 13, 10, 12, 10],
            "strat_fold": [10, 8, 9, 1, 1, 1],
            "filename_lr": [
                "records100/00000/00005_lr",
                "records100/00000/00002_lr",
                "records100/00000/00004_lr",
                "records100/00000/00001_lr",
                "records100/00000/00003_lr",
                "records100/00000/00006_lr",
            ],
            "scp_codes": [
                "{'CD_CODE': 100.0, 'HYP_CODE': 50.0}",
                "{'MI_CODE': 80.0}",
                "{'ST_CODE': 0.0}",
                "{'NORM_CODE': 100.0}",
                "{'RHYTHM': 100.0}",
                "{'NORM_CODE': 50.0, 'MI_CODE': 50.0}",
            ],
            "age": [80, 60, 70, 30, 50, 31],
            "sex": [1, 0, 1, 0, 1, 0],
        }
    )
    metadata.to_csv(root / "ptbxl_database.csv", index=False)
    for stem in metadata["filename_lr"]:
        path = root / stem
        path.parent.mkdir(parents=True, exist_ok=True)
        path.with_suffix(".hea").write_text("synthetic header\n", encoding="ascii")
        path.with_suffix(".dat").write_bytes(b"\x00\x01")


def test_official_contract_constants() -> None:
    assert PTBXL_VERSION == "1.0.3"
    assert EXPECTED_RECORDS == 21_799
    assert EXPECTED_PATIENTS == 18_869
    assert EXPECTED_SUPERCLASS_COUNTS == {
        "NORM": 9_514,
        "MI": 5_469,
        "STTC": 5_235,
        "CD": 4_898,
        "HYP": 2_649,
    }


def test_parse_scp_codes_is_safe_and_canonical(tmp_path: Path) -> None:
    assert parse_scp_codes("{'B': 0, 'A': 100.0}") == {"A": 100.0, "B": 0.0}
    marker = tmp_path / "must_not_exist"
    payload = f"__import__('pathlib').Path({str(marker)!r}).touch()"
    with pytest.raises(ManifestError, match="invalid scp_codes"):
        parse_scp_codes(payload)
    assert not marker.exists()


@pytest.mark.parametrize(
    "value",
    ["[]", "{'A': True}", "{1: 5}", "{'A': float('nan')}", None],
)
def test_parse_scp_codes_rejects_malformed_values(value: object) -> None:
    with pytest.raises(ManifestError):
        parse_scp_codes(value)


def test_build_manifest_maps_labels_sorts_and_excludes_unlabeled(tmp_path: Path) -> None:
    _write_dataset(tmp_path)
    manifest, summary = build_manifest(tmp_path, strict_official_counts=False)

    assert manifest["ecg_id"].tolist() == [1, 2, 4, 5, 6]
    assert manifest["split"].tolist() == [
        "development_train",
        "model_selection",
        "calibration",
        "test",
        "development_train",
    ]
    assert manifest.loc[manifest["ecg_id"] == 5, "labels"].item() == "CD|HYP"
    assert manifest.loc[manifest["ecg_id"] == 6, "labels"].item() == "NORM|MI"
    assert manifest.loc[:, list(LABEL_COLUMNS)].dtypes.astype(str).eq("int8").all()
    assert summary["dataset_version"] == "1.0.3"
    assert summary["source_records"] == 6
    assert summary["manifest_records"] == 5
    assert summary["unlabeled_records_excluded"] == 1
    assert summary["superclasses"] == list(SUPERCLASSES)


def test_build_manifest_detects_patient_leakage(tmp_path: Path) -> None:
    _write_dataset(tmp_path)
    metadata_path = tmp_path / "ptbxl_database.csv"
    metadata = pd.read_csv(metadata_path)
    metadata.loc[metadata["ecg_id"] == 6, "strat_fold"] = 2
    metadata.to_csv(metadata_path, index=False)

    with pytest.raises(ManifestError, match="multiple folds"):
        build_manifest(tmp_path, strict_official_counts=False)


def test_build_manifest_rejects_unsafe_and_missing_paths(tmp_path: Path) -> None:
    _write_dataset(tmp_path)
    metadata_path = tmp_path / "ptbxl_database.csv"
    metadata = pd.read_csv(metadata_path)
    metadata.loc[0, "filename_lr"] = "../outside"
    metadata.to_csv(metadata_path, index=False)
    with pytest.raises(ManifestError, match="unsafe relative path"):
        build_manifest(tmp_path, strict_official_counts=False)

    _write_dataset(tmp_path)
    missing = tmp_path / "records100" / "00000" / "00005_lr.dat"
    missing.unlink()
    with pytest.raises(ManifestError, match="missing waveform files"):
        build_manifest(tmp_path, strict_official_counts=False)


def test_manifest_artifacts_are_byte_deterministic(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    _write_dataset(dataset_root)
    manifest, summary = build_manifest(dataset_root, strict_official_counts=False)

    first = write_manifest_artifacts(manifest, summary, tmp_path / "first")
    second = write_manifest_artifacts(manifest, summary, tmp_path / "second")

    assert first.csv_sha256 == second.csv_sha256
    assert first.parquet_sha256 == second.parquet_sha256
    assert first.summary_sha256 == second.summary_sha256
    assert first.checksums_path.read_bytes() == second.checksums_path.read_bytes()
    payload = json.loads(first.summary_path.read_text(encoding="utf-8"))
    assert payload["manifest"]["csv_sha256"] == sha256_file(first.csv_path)
    assert payload["manifest"]["parquet_sha256"] == sha256_file(first.parquet_path)


def test_sha256_inventory_parsing_and_verification(tmp_path: Path) -> None:
    target = tmp_path / "records100" / "00000" / "00001_lr.dat"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"PTB-XL")
    digest = sha256_file(target)
    inventory = parse_sha256sums(f"{digest}  ./records100/00000/00001_lr.dat\n")

    verified = verify_sha256sums(tmp_path, inventory)
    assert verified == {"records100/00000/00001_lr.dat": digest}

    target.write_bytes(b"corrupt")
    with pytest.raises(ManifestError, match="SHA-256 mismatch"):
        verify_sha256sums(tmp_path, inventory)


def test_sha256_inventory_rejects_traversal() -> None:
    with pytest.raises(ManifestError, match="unsafe relative path"):
        parse_sha256sums(f"{'0' * 64}  ../outside\n")
