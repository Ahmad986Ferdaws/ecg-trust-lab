from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ecg_trust.ood_completion.cohorts import CohortRecord
from ecg_trust.ood_completion.waveform_inventory import (
    CHECKSUM_SUBSET_DOMAIN,
    OfficialWaveformSubset,
    OODWaveformIntegrityError,
    build_official_waveform_subset,
    official_checksum_subset_sha256,
    verify_official_waveform_subset,
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def test_subset_encoding_matches_the_exact_protocol_vector() -> None:
    pairs = (
        ("records/b.hea", "b" * 64),
        ("records/a.dat", "a" * 64),
    )
    payload = {
        "algorithm": "official_checksum_subset_v1",
        "files": [
            {"relative_path": "records/a.dat", "sha256": "a" * 64},
            {"relative_path": "records/b.hea", "sha256": "b" * 64},
        ],
        "schema_version": 1,
    }
    canonical = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    expected = "sha256:" + hashlib.sha256(CHECKSUM_SUBSET_DOMAIN + canonical).hexdigest()

    assert official_checksum_subset_sha256(pairs) == expected
    assert official_checksum_subset_sha256(tuple(reversed(pairs))) == expected


def test_build_and_verify_selected_dat_hea_pairs(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    record_dir = dataset_root / "records"
    record_dir.mkdir(parents=True)
    contents = {
        "records/one.dat": b"one-data",
        "records/one.hea": b"one-header",
        "records/two.dat": b"two-data",
        "records/two.hea": b"two-header",
    }
    for relative_path, payload in contents.items():
        (dataset_root / relative_path).write_bytes(payload)
    inventory = {path: _sha256(payload) for path, payload in contents.items()}
    records = (
        CohortRecord(2, 20, 9, "records/two"),
        CohortRecord(1, 10, 8, "records/one"),
    )

    subset = build_official_waveform_subset(records, official_checksums=inventory)

    assert subset.record_count == 2
    assert subset.file_count == 4
    assert subset.relative_paths == tuple(sorted(contents))
    assert verify_official_waveform_subset(dataset_root, subset) == subset.subset_sha256
    with pytest.raises(TypeError):
        subset.official_sha256_by_path["records/one.dat"] = "0" * 64  # type: ignore[index]


def test_direct_subset_construction_defensively_freezes_its_mapping() -> None:
    original = {"records/one.dat": "a" * 64}
    subset = OfficialWaveformSubset(
        record_count=1,
        relative_paths=("records/one.dat",),
        official_sha256_by_path=original,
        subset_sha256="sha256:" + "b" * 64,
    )

    original.clear()

    assert subset.official_sha256_by_path == {"records/one.dat": "a" * 64}
    with pytest.raises(TypeError):
        subset.official_sha256_by_path["records/one.dat"] = "c" * 64  # type: ignore[index]


def test_tampered_or_incomplete_waveform_subset_fails_closed(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    record_dir = dataset_root / "records"
    record_dir.mkdir(parents=True)
    data_path = record_dir / "one.dat"
    header_path = record_dir / "one.hea"
    data_path.write_bytes(b"data")
    header_path.write_bytes(b"header")
    checksums = {
        "records/one.dat": _sha256(b"data"),
        "records/one.hea": _sha256(b"header"),
    }
    records = (CohortRecord(1, 10, 8, "records/one"),)
    subset = build_official_waveform_subset(records, official_checksums=checksums)
    data_path.write_bytes(b"tampered")

    with pytest.raises(OODWaveformIntegrityError, match="do not match"):
        verify_official_waveform_subset(dataset_root, subset)
    with pytest.raises(OODWaveformIntegrityError, match="absent"):
        build_official_waveform_subset(
            records,
            official_checksums={"records/one.dat": _sha256(b"data")},
        )


def test_duplicate_records_and_bad_digests_are_rejected() -> None:
    record = CohortRecord(1, 10, 8, "records/one")
    inventory = {
        "records/one.dat": "a" * 64,
        "records/one.hea": "b" * 64,
    }

    with pytest.raises(OODWaveformIntegrityError, match="duplicated"):
        build_official_waveform_subset((record, record), official_checksums=inventory)
    with pytest.raises(OODWaveformIntegrityError, match="lowercase"):
        official_checksum_subset_sha256((("records/one.dat", "A" * 64),))
