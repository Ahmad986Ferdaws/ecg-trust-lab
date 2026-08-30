from __future__ import annotations

import binascii
import json
import shutil
import sys
import tarfile
import zipfile
from collections.abc import Callable
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest

from ecg_trust.constants import LEADS
from ecg_trust.ood_v2 import inventory as inventory_module
from ecg_trust.ood_v2.inventory import (
    CHALLENGE_2011_DATASET,
    CHALLENGE_2011_VERSION,
    CONFIRMATION_LOCKBOX_ROLE,
    ZZU_PEDIATRIC_DATASET,
    ZZU_PEDIATRIC_VERSION,
    ArchiveExtractionClosure,
    ExternalInventoryError,
    ExternalInventoryRecord,
    ExternalWaveformInventory,
    ZZUPediatricCandidate,
    build_challenge_tar_extraction_closure,
    build_external_inventory,
    build_zzu_split_zip_extraction_closure,
    external_inventory_public_projection,
    inventory_challenge_2011_record,
    inventory_zzu_pediatric_record,
    load_external_inventory,
    parse_challenge_2011_quality_lists,
    parse_seven_zip_slt_listing,
    parse_zzu_pediatric_attributes_csv,
    resolve_inventory_record_base,
    resolve_seven_zip_tool_binding,
    save_external_inventory,
    select_zzu_pediatric_inventory_records,
    validate_challenge_2011_set_a_inventory,
    verify_challenge_tar_extraction_closure,
    verify_external_inventory,
    verify_seven_zip_tool_binding,
    verify_wfdb_candidate_file_set,
    verify_zzu_split_zip_extraction_closure,
)


def _write_record(root: Path, record_ref: str, marker: bytes) -> Path:
    base = root / record_ref
    base.parent.mkdir(parents=True, exist_ok=True)
    Path(f"{base}.hea").write_bytes(b"header-" + marker)
    Path(f"{base}.dat").write_bytes(b"data-" + marker)
    return base


@pytest.fixture
def header_only_reader(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    observed: list[str] = []

    def fake_rdheader(record_base: str) -> SimpleNamespace:
        observed.append(record_base)
        sample_count = 5_000 if "challenge" in record_base else 5_500
        data_file_name = f"{Path(record_base).name}.dat"
        return SimpleNamespace(
            fs=500.0,
            sig_len=sample_count,
            n_sig=12,
            sig_name=list(LEADS),
            units=["mV"] * 12,
            file_name=[data_file_name] * 12,
        )

    monkeypatch.setattr(inventory_module.wfdb, "rdheader", fake_rdheader)
    monkeypatch.setattr(
        inventory_module.wfdb,
        "rdrecord",
        lambda *args, **kwargs: pytest.fail("inventory must never read waveform amplitudes"),
        raising=False,
    )
    return observed


def _inventory(
    tmp_path: Path,
    header_only_reader: list[str],
) -> tuple[ExternalWaveformInventory, ExternalInventoryRecord, ExternalInventoryRecord]:
    _write_record(tmp_path, "challenge/a01", b"challenge")
    _write_record(tmp_path, "zzu/p001", b"pediatric")
    challenge = inventory_challenge_2011_record(
        tmp_path,
        dataset_version=CHALLENGE_2011_VERSION,
        site="private-challenge-site-name",
        site_alias="challenge-set-a",
        record_ref="challenge/a01",
        quality_label="acceptable",
    )
    pediatric = inventory_zzu_pediatric_record(
        tmp_path,
        dataset_version=ZZU_PEDIATRIC_VERSION,
        site="private-pediatric-site-name",
        site_alias="pediatric-external",
        patient_key="zzu-patient-001",
        record_ref="zzu/p001",
        pediatric_12_lead=True,
    )
    assert len(header_only_reader) == 2
    return build_external_inventory((pediatric, challenge)), challenge, pediatric


def test_inventory_is_canonical_self_hashed_and_round_trips_exactly(
    tmp_path: Path,
    header_only_reader: list[str],
) -> None:
    built, challenge, pediatric = _inventory(tmp_path, header_only_reader)
    inventory = build_external_inventory((pediatric, challenge))
    output = tmp_path / "inventory.json"

    save_external_inventory(inventory, output)
    loaded = load_external_inventory(output)

    assert loaded == inventory
    assert loaded.record_count == 2
    assert loaded.records[0].dataset == CHALLENGE_2011_DATASET
    assert loaded.records[1].dataset == ZZU_PEDIATRIC_DATASET
    assert loaded.inventory_sha256.startswith("sha256:")
    assert output.read_bytes() == inventory.to_canonical_json_bytes()
    assert output.read_bytes().endswith(b"\n")
    assert verify_external_inventory(tmp_path, loaded) == loaded.inventory_sha256
    assert resolve_inventory_record_base(tmp_path, challenge) == tmp_path / "challenge/a01"
    assert built == inventory


def test_record_metadata_is_dataset_specific_and_complete(
    tmp_path: Path,
    header_only_reader: list[str],
) -> None:
    _, challenge, pediatric = _inventory(tmp_path, header_only_reader)

    assert challenge.patient_key is None
    assert challenge.challenge_quality_label == "acceptable"
    assert challenge.pediatric_12_lead is None
    assert challenge.source_role == CONFIRMATION_LOCKBOX_ROLE
    assert pediatric.patient_key == "zzu-patient-001"
    assert pediatric.challenge_quality_label is None
    assert pediatric.pediatric_12_lead is True
    assert challenge.duration_seconds == 10.0
    assert pediatric.duration_seconds == 11.0
    for record in (challenge, pediatric):
        assert record.sampling_frequency_hz == 500.0
        assert record.source_sample_count in {5_000, 5_500}
        assert record.lead_count == 12
        assert record.raw_ordered_leads == LEADS
        assert record.canonical_ordered_leads == LEADS
        assert record.raw_data_file_names == (f"{Path(record.record_ref).name}.dat",) * 12
        assert record.raw_physical_units == ("mV",) * 12
        assert record.raw_header_size_bytes > 0
        assert record.raw_data_size_bytes > 0
        assert len(record.raw_header_sha256) == 64
        assert len(record.raw_data_sha256) == 64


def test_public_projection_contains_only_aggregate_metadata(
    tmp_path: Path,
    header_only_reader: list[str],
) -> None:
    inventory, _, _ = _inventory(tmp_path, header_only_reader)
    projection = external_inventory_public_projection(inventory)
    encoded = json.dumps(projection, sort_keys=True)

    assert projection["record_count"] == 2
    assert projection["group_count"] == 2
    assert str(projection["projection_sha256"]).startswith("sha256:")
    assert "challenge/a01" not in encoded
    assert "zzu/p001" not in encoded
    assert "zzu-patient-001" not in encoded
    assert "private-challenge-site-name" not in encoded
    assert "private-pediatric-site-name" not in encoded
    assert "challenge-set-a" in encoded
    assert "pediatric-external" in encoded
    assert inventory.records[0].raw_data_sha256 not in encoded
    assert inventory.records[1].raw_header_sha256 not in encoded


def test_tampered_json_and_raw_files_are_rejected(
    tmp_path: Path,
    header_only_reader: list[str],
) -> None:
    inventory, _, _ = _inventory(tmp_path, header_only_reader)
    output = tmp_path / "inventory.json"
    save_external_inventory(inventory, output)
    output.write_bytes(output.read_bytes().replace(b'"lead_count":12', b'"lead_count":11', 1))

    with pytest.raises(ExternalInventoryError):
        load_external_inventory(output)

    save_external_inventory(inventory, output)
    output.write_bytes(output.read_bytes()[:-1] + b" \n")
    with pytest.raises(ExternalInventoryError, match="canonical"):
        load_external_inventory(output)

    Path(f"{tmp_path / 'challenge/a01'}.dat").write_bytes(b"tampered-data")
    with pytest.raises(ExternalInventoryError, match="no longer matches"):
        verify_external_inventory(tmp_path, inventory)


def test_traversal_symlinks_and_duplicate_bindings_fail_closed(
    tmp_path: Path,
    header_only_reader: list[str],
) -> None:
    _, challenge, _ = _inventory(tmp_path, header_only_reader)
    with pytest.raises(ExternalInventoryError, match="traversal"):
        inventory_challenge_2011_record(
            tmp_path,
            dataset_version=CHALLENGE_2011_VERSION,
            site="site",
            site_alias="site-a",
            record_ref="../a01",
            quality_label="indeterminate",
        )
    with pytest.raises(ExternalInventoryError, match="duplicate"):
        build_external_inventory((challenge, challenge))

    target = Path(f"{tmp_path / 'challenge/a01'}.dat")
    link_base = tmp_path / "challenge/link"
    Path(f"{link_base}.hea").write_bytes(b"header-link")
    try:
        Path(f"{link_base}.dat").symlink_to(target)
    except OSError:
        pytest.skip("symlink creation is not available on this Windows configuration")
    with pytest.raises(ExternalInventoryError, match="symlink"):
        inventory_challenge_2011_record(
            tmp_path,
            dataset_version=CHALLENGE_2011_VERSION,
            site="site",
            site_alias="site-a",
            record_ref="challenge/link",
            quality_label="indeterminate",
        )


def test_inventory_rejects_a_junction_in_record_ancestry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record_base = _write_record(tmp_path, "challenge/a01", b"challenge")
    junction = record_base.parent
    original = Path.is_junction
    monkeypatch.setattr(
        Path,
        "is_junction",
        lambda self: self == junction or original(self),
    )
    with pytest.raises(ExternalInventoryError, match="indirect"):
        inventory_challenge_2011_record(
            tmp_path,
            dataset_version=CHALLENGE_2011_VERSION,
            site="site",
            site_alias="site-a",
            record_ref="challenge/a01",
            quality_label="acceptable",
        )


def test_challenge_quality_lists_assign_indeterminate_and_reject_overlap() -> None:
    labels = parse_challenge_2011_quality_lists(
        "a01\na02\na03\n",
        "a01\n",
        "a02\n",
        expected_record_count=3,
    )

    assert dict(labels) == {
        "a01": "acceptable",
        "a02": "unacceptable",
        "a03": "indeterminate",
    }
    with pytest.raises(TypeError):
        labels["a01"] = "indeterminate"  # type: ignore[index]
    with pytest.raises(ExternalInventoryError, match="overlap"):
        parse_challenge_2011_quality_lists(
            "a01\n", "a01\n", "a01\n", expected_record_count=1
        )
    with pytest.raises(ExternalInventoryError, match="unknown"):
        parse_challenge_2011_quality_lists(
            "a01\n", "a02\n", "a01\n", expected_record_count=1
        )


def test_patient_key_is_nullable_only_for_challenge() -> None:
    with pytest.raises(ExternalInventoryError, match="patient_key"):
        ExternalInventoryRecord(
            dataset=ZZU_PEDIATRIC_DATASET,
            dataset_version=ZZU_PEDIATRIC_VERSION,
            site="site",
            site_alias="site-z",
            patient_key=None,
            record_ref="zzu/p001",
            source_role=CONFIRMATION_LOCKBOX_ROLE,
            raw_header_sha256="a" * 64,
            raw_header_size_bytes=1,
            raw_data_sha256="b" * 64,
            raw_data_size_bytes=1,
            sampling_frequency_hz=500.0,
            source_sample_count=5_000,
            duration_seconds=10.0,
            lead_count=12,
            raw_ordered_leads=LEADS,
            canonical_ordered_leads=LEADS,
            raw_data_file_names=("p001.dat",) * 12,
            raw_physical_units=("mV",) * 12,
            challenge_quality_label=None,
            pediatric_12_lead=True,
        )


def test_challenge_inventory_requires_the_frozen_complete_labeled_set(
    tmp_path: Path,
    header_only_reader: list[str],
) -> None:
    _, challenge, _ = _inventory(tmp_path, header_only_reader)
    challenge_only = build_external_inventory((challenge,))

    assert (
        validate_challenge_2011_set_a_inventory(
            challenge_only,
            expected_record_count=1,
            expected_quality_by_record={"challenge/a01": "acceptable"},
        )
        == challenge_only.inventory_sha256
    )
    with pytest.raises(ExternalInventoryError, match="exactly 1000"):
        validate_challenge_2011_set_a_inventory(challenge_only)
    with pytest.raises(ExternalInventoryError, match="quality labels"):
        validate_challenge_2011_set_a_inventory(
            challenge_only,
            expected_record_count=1,
            expected_quality_by_record={"challenge/a01": "unacceptable"},
        )


def test_zzu_header_only_selection_counts_every_frozen_exclusion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    zzu_leads = tuple(
        {"aVR": "AVR", "aVL": "AVL", "aVF": "AVF"}.get(lead, lead) for lead in LEADS
    )
    names_by_ref: dict[str, tuple[float, int, tuple[str, ...]]] = {
        "P00001_E01": (500.0, 5_000, zzu_leads),
        "P00002_E01": (250.0, 5_000, zzu_leads),
        "P00003_E01": (500.0, 4_500, zzu_leads),
        "P00004_E01": (500.0, 5_000, LEADS[:9]),
        "P00005_E01": (500.0, 5_000, (*LEADS[:-1], "X")),
    }
    candidates: list[ZZUPediatricCandidate] = []
    for index, (name, (_, sample_count, leads)) in enumerate(names_by_ref.items(), start=1):
        patient_id = f"P{index:05d}"
        record_ref = f"{patient_id[:3]}/{patient_id}/{name}"
        _write_record(tmp_path, record_ref, name.encode("ascii"))
        candidates.append(
            ZZUPediatricCandidate(
                dataset_version=ZZU_PEDIATRIC_VERSION,
                site="private-zzu-site",
                site_alias="pediatric-external",
                patient_key=f"{ZZU_PEDIATRIC_DATASET}:{patient_id}",
                record_ref=record_ref,
                ecg_id=name,
                declared_lead_count=len(leads),
                pediatric_12_lead=len(leads) == 12,
                declared_sample_count=sample_count,
            )
        )

    def fake_rdheader(record_base: str) -> SimpleNamespace:
        name = Path(record_base).name
        frequency, sample_count, leads = names_by_ref[name]
        return SimpleNamespace(
            fs=frequency,
            sig_len=sample_count,
            n_sig=len(leads),
            sig_name=list(leads),
            units=["mV"] * len(leads),
            file_name=[f"{Path(record_base).name}.dat"] * len(leads),
        )

    monkeypatch.setattr(inventory_module.wfdb, "rdheader", fake_rdheader)
    monkeypatch.setattr(
        inventory_module.wfdb,
        "rdrecord",
        lambda *args, **kwargs: pytest.fail("candidate selection must be header-only"),
        raising=False,
    )

    selected, summary = select_zzu_pediatric_inventory_records(tmp_path, candidates)

    assert [record.record_ref for record in selected] == ["P00/P00001/P00001_E01"]
    assert selected[0].raw_ordered_leads == zzu_leads
    assert selected[0].canonical_ordered_leads == LEADS
    assert selected[0].raw_data_file_names == ("P00001_E01.dat",) * 12
    assert summary.candidate_record_count == 5
    assert summary.selected_record_count == 1
    assert summary.excluded_record_count == 4
    assert summary.exclusion_counts == {
        "pediatric_12_lead_flag_false": 1,
        "sampling_frequency_not_500_hz": 1,
        "duration_under_10_seconds": 1,
        "lead_count_not_12": 0,
        "noncanonical_lead_set": 1,
    }
    assert summary.summary_sha256.startswith("sha256:")
    with pytest.raises(TypeError):
        summary.exclusion_counts["duration_under_10_seconds"] = 2  # type: ignore[index]

    mismatched = replace(candidates[0], declared_sample_count=4_999)
    with pytest.raises(ExternalInventoryError, match="sample count differs"):
        select_zzu_pediatric_inventory_records(tmp_path, (mismatched, *candidates[1:]))


@pytest.mark.parametrize("unsupported", ["avr", "AvR", " aVR", "aVR ", "AVr"])
def test_inventory_rejects_casefold_and_whitespace_lead_aliases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    unsupported: str,
) -> None:
    _write_record(tmp_path, "zzu/bad-alias", b"bad-alias")
    raw_leads = tuple(unsupported if lead == "aVR" else lead for lead in LEADS)
    monkeypatch.setattr(
        inventory_module.wfdb,
        "rdheader",
        lambda _: SimpleNamespace(
            fs=500.0,
            sig_len=5_000,
                n_sig=12,
                sig_name=list(raw_leads),
                units=["mV"] * 12,
                file_name=["bad-alias.dat"] * 12,
        ),
    )

    with pytest.raises(ExternalInventoryError, match="unsupported|canonical"):
        inventory_zzu_pediatric_record(
            tmp_path,
            dataset_version=ZZU_PEDIATRIC_VERSION,
            site="private-zzu-site",
            site_alias="pediatric-external",
            patient_key="patient-bad-alias",
            record_ref="zzu/bad-alias",
            pediatric_12_lead=True,
        )


@pytest.mark.parametrize("bad_name", ["other.dat", "../bound.dat", "sub/bound.dat"])
def test_inventory_rejects_alternate_or_traversing_wfdb_data_bindings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bad_name: str,
) -> None:
    _write_record(tmp_path, "zzu/bound", b"bound")
    monkeypatch.setattr(
        inventory_module.wfdb,
        "rdheader",
        lambda _: SimpleNamespace(
            fs=500.0,
            sig_len=5_000,
            n_sig=12,
            sig_name=list(LEADS),
            units=["mV"] * 12,
            file_name=[bad_name] * 12,
        ),
    )

    with pytest.raises(ExternalInventoryError, match="bind every lead"):
        inventory_zzu_pediatric_record(
            tmp_path,
            dataset_version=ZZU_PEDIATRIC_VERSION,
            site="private-zzu-site",
            site_alias="pediatric-external",
            patient_key="patient-bound",
            record_ref="zzu/bound",
            pediatric_12_lead=True,
        )


def test_runtime_verification_compares_raw_and_canonical_lead_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_record(tmp_path, "zzu/raw-names", b"raw-names")
    aliases = tuple(
        {"aVR": "AVR", "aVL": "AVL", "aVF": "AVF"}.get(lead, lead) for lead in LEADS
    )
    observed_names = aliases

    def fake_rdheader(_: str) -> SimpleNamespace:
        return SimpleNamespace(
            fs=500.0,
            sig_len=5_000,
            n_sig=12,
            sig_name=list(observed_names),
            units=["mV"] * 12,
            file_name=["raw-names.dat"] * 12,
        )

    monkeypatch.setattr(inventory_module.wfdb, "rdheader", fake_rdheader)
    record = inventory_zzu_pediatric_record(
        tmp_path,
        dataset_version=ZZU_PEDIATRIC_VERSION,
        site="private-zzu-site",
        site_alias="pediatric-external",
        patient_key="patient-raw-names",
        record_ref="zzu/raw-names",
        pediatric_12_lead=True,
    )
    inventory = build_external_inventory((record,))

    assert record.raw_ordered_leads == aliases
    assert record.canonical_ordered_leads == LEADS
    observed_names = LEADS
    with pytest.raises(ExternalInventoryError, match="no longer matches"):
        verify_external_inventory(tmp_path, inventory)


def test_runtime_verification_rejects_changed_header_data_file_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_record(tmp_path, "zzu/file-binding", b"file-binding")
    observed_file_name = "file-binding.dat"

    def fake_rdheader(_: str) -> SimpleNamespace:
        return SimpleNamespace(
            fs=500.0,
            sig_len=5_000,
            n_sig=12,
            sig_name=list(LEADS),
            units=["mV"] * 12,
            file_name=[observed_file_name] * 12,
        )

    monkeypatch.setattr(inventory_module.wfdb, "rdheader", fake_rdheader)
    record = inventory_zzu_pediatric_record(
        tmp_path,
        dataset_version=ZZU_PEDIATRIC_VERSION,
        site="private-zzu-site",
        site_alias="pediatric-external",
        patient_key="patient-file-binding",
        record_ref="zzu/file-binding",
        pediatric_12_lead=True,
    )
    inventory = build_external_inventory((record,))

    observed_file_name = "other.dat"
    with pytest.raises(ExternalInventoryError, match="bind every lead"):
        verify_external_inventory(tmp_path, inventory)


def test_challenge_patient_key_and_uppercase_zzu_aliases_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_record(tmp_path, "challenge/a01", b"challenge")
    uppercase = tuple(
        {"aVR": "AVR", "aVL": "AVL", "aVF": "AVF"}.get(lead, lead) for lead in LEADS
    )
    monkeypatch.setattr(
        inventory_module.wfdb,
        "rdheader",
        lambda _: SimpleNamespace(
            fs=500.0,
            sig_len=5_000,
            n_sig=12,
            sig_name=list(uppercase),
            units=["mV"] * 12,
            file_name=["a01.dat"] * 12,
        ),
    )
    with pytest.raises(ExternalInventoryError, match="patient_key"):
        inventory_challenge_2011_record(
            tmp_path,
            dataset_version=CHALLENGE_2011_VERSION,
            site="site",
            site_alias="alias",
            patient_key="forbidden",
            record_ref="challenge/a01",
            quality_label="acceptable",
        )
    with pytest.raises(ExternalInventoryError, match="unsupported"):
        inventory_challenge_2011_record(
            tmp_path,
            dataset_version=CHALLENGE_2011_VERSION,
            site="site",
            site_alias="alias",
            record_ref="challenge/a01",
            quality_label="acceptable",
        )


@pytest.mark.parametrize("raw_unit", ["mv", "MV", "mV ", " mV"])
def test_inventory_rejects_nonexact_raw_units_before_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    raw_unit: str,
) -> None:
    _write_record(tmp_path, "zzu/unit", b"unit")
    monkeypatch.setattr(
        inventory_module.wfdb,
        "rdheader",
        lambda _: SimpleNamespace(
            fs=500.0,
            sig_len=5_000,
            n_sig=12,
            sig_name=list(LEADS),
            units=[raw_unit] * 12,
            file_name=["unit.dat"] * 12,
        ),
    )
    with pytest.raises(ExternalInventoryError, match="mV|canonical"):
        inventory_zzu_pediatric_record(
            tmp_path,
            dataset_version=ZZU_PEDIATRIC_VERSION,
            site="site",
            site_alias="alias",
            patient_key="patient",
            record_ref="zzu/unit",
            pediatric_12_lead=True,
        )


def test_record_sample_count_and_raw_unit_tamper_fail_closed(
    tmp_path: Path,
    header_only_reader: list[str],
) -> None:
    _, challenge, _ = _inventory(tmp_path, header_only_reader)
    sample_tamper = challenge.to_dict()
    sample_tamper["source_sample_count"] = 4_999
    with pytest.raises(ExternalInventoryError, match="duration_seconds"):
        ExternalInventoryRecord.from_dict(sample_tamper)
    unit_tamper = challenge.to_dict()
    unit_tamper["raw_physical_units"] = ["MV"] * 12
    with pytest.raises(ExternalInventoryError, match="mV"):
        ExternalInventoryRecord.from_dict(unit_tamper)


@pytest.mark.parametrize(
    "row",
    [
        "P99/P00001/P00001_E01,P00001_E01,P00001,5000,12",
        "P00/P99999/P00001_E01,P00001_E01,P00001,5000,12",
        "P00/P00001/P00001_E02,P00001_E01,P00001,5000,12",
        "P00/P00001/P00001_E01,P00002_E01,P00001,5000,12",
        "../P00001/P00001_E01,P00001_E01,P00001,5000,12",
    ],
)
def test_zzu_metadata_full_identity_relationship_fails_closed(row: str) -> None:
    payload = "Filename,ECG_ID,Patient_ID,Sampling_point,Lead\n" + row + "\n"
    with pytest.raises(ExternalInventoryError):
        parse_zzu_pediatric_attributes_csv(
            payload,
            site="site",
            site_alias="alias",
            expected_record_count=1,
            expected_patient_count=1,
        )


def test_exact_wfdb_candidate_set_rejects_extra_missing_and_unpaired(tmp_path: Path) -> None:
    _write_record(tmp_path, "P00/P00001/P00001_E01", b"one")
    expected = ("P00/P00001/P00001_E01",)
    assert verify_wfdb_candidate_file_set(tmp_path, expected) == expected
    _write_record(tmp_path, "P00/P00002/P00002_E01", b"two")
    with pytest.raises(ExternalInventoryError, match="extra or missing"):
        verify_wfdb_candidate_file_set(tmp_path, expected)
    Path(f"{tmp_path / 'P00/P00002/P00002_E01'}.dat").unlink()
    with pytest.raises(ExternalInventoryError, match="unpaired"):
        verify_wfdb_candidate_file_set(tmp_path, expected)


def _write_tar_file(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    member = tarfile.TarInfo(name)
    member.size = len(payload)
    archive.addfile(member, BytesIO(payload))


def _challenge_closure_fixture(tmp_path: Path) -> tuple[Path, Path, tuple[str, ...]]:
    extraction_root = tmp_path / "extracted" / "set-a"
    payloads = {
        "c001.hea": b"header",
        "c001.dat": b"data",
        "c001.txt": b"ignored diagnosis text",
        "HEADER.shtml": b"ignored release page",
        "RECORDS": b"c001\n",
        "RECORDS-acceptable": b"c001\n",
        "RECORDS-unacceptable": b"# empty\n",
    }
    for relative, payload in payloads.items():
        destination = extraction_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
    archive_path = tmp_path / "set-a.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        directory = tarfile.TarInfo("set-a")
        directory.type = tarfile.DIRTYPE
        archive.addfile(directory)
        for relative, payload in payloads.items():
            _write_tar_file(archive, f"set-a/{relative}", payload)
    required = (
        "c001.hea",
        "c001.dat",
        "RECORDS",
        "RECORDS-acceptable",
        "RECORDS-unacceptable",
    )
    return archive_path, extraction_root, required


def test_challenge_archive_closure_binds_all_release_files_and_round_trips(
    tmp_path: Path,
) -> None:
    archive_path, extraction_root, required = _challenge_closure_fixture(tmp_path)
    closure = build_challenge_tar_extraction_closure(
        archive_path,
        extraction_root,
        expected_required_relative_paths=required,
    )

    assert isinstance(closure, ArchiveExtractionClosure)
    assert closure.member_count == 7
    assert {member.role for member in closure.members} == {
        "wfdb_header",
        "wfdb_data",
        "quality_reference",
        "ignored_release_file",
    }
    assert sum(member.role == "ignored_release_file" for member in closure.members) == 2
    assert verify_challenge_tar_extraction_closure(
        archive_path, extraction_root, closure
    ) == closure.closure_sha256


def test_challenge_archive_closure_detects_midpass_extraction_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_path, extraction_root, required = _challenge_closure_fixture(tmp_path)
    original = inventory_module._snapshot_extraction_tree
    snapshot_count = 0

    def changing_snapshot(root: Path) -> object:
        nonlocal snapshot_count
        result = original(root)
        snapshot_count += 1
        if snapshot_count == 1:
            (extraction_root / "c001.dat").write_bytes(b"changed-midpass")
        return result

    monkeypatch.setattr(inventory_module, "_snapshot_extraction_tree", changing_snapshot)
    with pytest.raises(ExternalInventoryError, match="changed during"):
        build_challenge_tar_extraction_closure(
            archive_path,
            extraction_root,
            expected_required_relative_paths=required,
        )


@pytest.mark.parametrize("failure", ["tamper", "extra", "missing"])
def test_challenge_archive_closure_rejects_tree_byte_and_member_drift(
    tmp_path: Path,
    failure: str,
) -> None:
    archive_path, extraction_root, required = _challenge_closure_fixture(tmp_path)
    if failure == "tamper":
        (extraction_root / "c001.dat").write_bytes(b"changed")
    elif failure == "extra":
        (extraction_root / "extra.txt").write_bytes(b"extra")
    else:
        (extraction_root / "c001.hea").unlink()
    with pytest.raises(ExternalInventoryError, match="differ|extra or missing"):
        build_challenge_tar_extraction_closure(
            archive_path,
            extraction_root,
            expected_required_relative_paths=required,
        )


@pytest.mark.parametrize("failure", ["traversal", "link", "duplicate", "unsupported"])
def test_challenge_archive_closure_rejects_unsafe_or_unknown_members(
    tmp_path: Path,
    failure: str,
) -> None:
    archive_path, extraction_root, required = _challenge_closure_fixture(tmp_path)
    with tarfile.open(archive_path, "w:gz") as archive:
        if failure == "traversal":
            _write_tar_file(archive, "../evil.dat", b"evil")
        elif failure == "link":
            member = tarfile.TarInfo("set-a/link.dat")
            member.type = tarfile.SYMTYPE
            member.linkname = "c001.dat"
            archive.addfile(member)
        elif failure == "duplicate":
            _write_tar_file(archive, "set-a/c001.dat", b"data")
            _write_tar_file(archive, "set-a/c001.dat", b"data")
        else:
            _write_tar_file(archive, "set-a/unknown.bin", b"unknown")
    with pytest.raises(ExternalInventoryError):
        build_challenge_tar_extraction_closure(
            archive_path,
            extraction_root,
            expected_required_relative_paths=required,
        )


def _zzu_closure_fixture(
    tmp_path: Path,
    *,
    archive_payloads: dict[str, bytes] | None = None,
) -> tuple[
    Path,
    Path,
    Path,
    Path,
    tuple[str, ...],
    Callable[[Path, tuple[str, ...]], str],
]:
    relative_header = "P00/P00001/P00001_E01.hea"
    relative_data = "P00/P00001/P00001_E01.dat"
    evaluated_root = tmp_path / "evaluated" / "Child_ecg"
    for relative, payload in {
        relative_header: b"zzu-header",
        relative_data: b"zzu-data",
    }.items():
        destination = evaluated_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
    z01 = tmp_path / "Child_ecg.z01"
    zip_path = tmp_path / "Child_ecg.zip"
    z01.write_bytes(b"split-part-one")
    zip_path.write_bytes(b"split-part-two")
    executable = tmp_path / "7z.exe"
    executable.write_bytes(b"seven-zip-binary")
    (tmp_path / "7z.dll").write_bytes(b"seven-zip-library")
    payloads = archive_payloads or {
        f"Child_ecg/{relative_header}": b"zzu-header",
        f"Child_ecg/{relative_data}": b"zzu-data",
    }

    def runner(_executable: Path, arguments: tuple[str, ...]) -> str:
        command = arguments[0]
        if command == "i":
            return "7-Zip 26.02 (x64)\n"
        if command == "l":
            blocks = [
                "Path = C:/private/Child_ecg.zip\n"
                "Type = zip\n"
                "Physical Size = 1\n"
                "Multivolume = +\n"
                "Volumes = 2"
            ]
            for path, payload in payloads.items():
                crc = f"{binascii.crc32(payload) & 0xFFFFFFFF:08X}"
                blocks.append(
                    f"Path = {path}\n"
                    f"Size = {len(payload)}\n"
                    "Folder = -\n"
                    "Attributes = A\n"
                    f"CRC = {crc}\n"
                    "Encrypted = -"
                )
            return "\n\n".join(blocks) + "\n"
        if command == "t":
            return "Everything is Ok\n"
        if command == "x":
            output_argument = next(value for value in arguments if value.startswith("-o"))
            output_root = Path(output_argument[2:])
            for path, payload in payloads.items():
                destination = output_root.joinpath(*path.split("/"))
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(payload)
            return "Everything is Ok\n"
        raise AssertionError(f"unexpected 7-Zip command {command!r}")

    return (
        z01,
        zip_path,
        evaluated_root,
        executable,
        (relative_header, relative_data),
        runner,
    )


def test_zzu_split_zip_closure_binds_tool_crc_sha_and_round_trips(tmp_path: Path) -> None:
    z01, zip_path, root, executable, required, raw_runner = _zzu_closure_fixture(tmp_path)
    runner = raw_runner
    assert callable(runner)
    closure = build_zzu_split_zip_extraction_closure(
        z01,
        zip_path,
        root,
        executable,
        expected_required_relative_paths=required,
        runner=runner,
    )

    assert closure.member_count == 2
    assert {member.role for member in closure.members} == {"wfdb_header", "wfdb_data"}
    assert all(member.archive_crc32 is not None for member in closure.members)
    assert closure.tool_binding is not None
    assert closure.tool_binding.version == "26.02"
    assert closure.tool_binding.executable_name == "7z.exe"
    assert closure.tool_binding.library_name == "7z.dll"
    assert verify_seven_zip_tool_binding(
        executable,
        closure.tool_binding,
        runner=runner,
    ) == closure.tool_binding.tool_sha256
    assert verify_zzu_split_zip_extraction_closure(
        z01,
        zip_path,
        root,
        executable,
        closure,
        runner=runner,
    ) == closure.closure_sha256
    (executable.parent / "7z.dll").write_bytes(b"changed-library")
    with pytest.raises(ExternalInventoryError, match="changed"):
        verify_seven_zip_tool_binding(executable, closure.tool_binding, runner=runner)


def test_zzu_split_zip_closure_stage_callback_is_exact_and_result_neutral(
    tmp_path: Path,
) -> None:
    z01, zip_path, root, executable, required, runner = _zzu_closure_fixture(tmp_path)
    baseline = build_zzu_split_zip_extraction_closure(
        z01,
        zip_path,
        root,
        executable,
        expected_required_relative_paths=required,
        runner=runner,
    )
    observed: list[str] = []

    staged = build_zzu_split_zip_extraction_closure(
        z01,
        zip_path,
        root,
        executable,
        expected_required_relative_paths=required,
        runner=runner,
        stage_callback=observed.append,
    )

    assert staged == baseline
    assert observed == [
        "zzu_tool_resolution",
        "zzu_archive_listing",
        "zzu_archive_test",
        "zzu_evaluated_tree_snapshot",
        "zzu_isolated_extraction",
        "zzu_archive_comparison",
    ]
    verification_observed: list[str] = []
    assert verify_zzu_split_zip_extraction_closure(
        z01,
        zip_path,
        root,
        executable,
        baseline,
        runner=runner,
        stage_callback=verification_observed.append,
    ) == baseline.closure_sha256
    assert verification_observed == observed


def test_zzu_split_zip_closure_stage_callback_failure_is_not_swallowed(
    tmp_path: Path,
) -> None:
    z01, zip_path, root, executable, required, raw_runner = _zzu_closure_fixture(tmp_path)
    runner_commands: list[str] = []

    def runner(executable_path: Path, arguments: tuple[str, ...]) -> str:
        runner_commands.append(arguments[0])
        return raw_runner(executable_path, arguments)

    failure = RuntimeError("synthetic stage callback failure")
    observed: list[str] = []

    def stage_callback(stage: str) -> None:
        observed.append(stage)
        if stage == "zzu_archive_test":
            raise failure

    with pytest.raises(RuntimeError) as raised:
        build_zzu_split_zip_extraction_closure(
            z01,
            zip_path,
            root,
            executable,
            expected_required_relative_paths=required,
            runner=runner,
            stage_callback=stage_callback,
        )

    assert raised.value is failure
    assert observed == [
        "zzu_tool_resolution",
        "zzu_archive_listing",
        "zzu_archive_test",
    ]
    assert runner_commands == ["i", "l"]


def test_zzu_split_zip_closure_rejects_noncallable_stage_callback(
    tmp_path: Path,
) -> None:
    z01, zip_path, root, executable, required, runner = _zzu_closure_fixture(tmp_path)

    with pytest.raises(TypeError, match="stage_callback must be callable or None"):
        build_zzu_split_zip_extraction_closure(
            z01,
            zip_path,
            root,
            executable,
            expected_required_relative_paths=required,
            runner=runner,
            stage_callback=object(),  # type: ignore[arg-type]
        )


def test_zzu_split_zip_closure_attributes_early_precheck_failure(
    tmp_path: Path,
) -> None:
    _, zip_path, root, executable, required, runner = _zzu_closure_fixture(tmp_path)
    invalid_z01 = tmp_path / "Child_ecg.part"
    invalid_z01.write_bytes(b"split-part-one")
    observed: list[str] = []

    with pytest.raises(ExternalInventoryError, match="matching .z01/.zip"):
        build_zzu_split_zip_extraction_closure(
            invalid_z01,
            zip_path,
            root,
            executable,
            expected_required_relative_paths=required,
            runner=runner,
            stage_callback=observed.append,
        )

    assert observed == ["zzu_tool_resolution"]


def test_scoop_shim_resolves_and_binds_only_real_tool_identity(tmp_path: Path) -> None:
    real_root = tmp_path / "apps/7zip/current"
    real_root.mkdir(parents=True)
    real_executable = real_root / "7z.exe"
    real_executable.write_bytes(b"real-seven-zip")
    (real_root / "7z.dll").write_bytes(b"real-library")
    shim_root = tmp_path / "shims"
    shim_root.mkdir()
    shim_executable = shim_root / "7z.exe"
    shim_executable.write_bytes(b"shim")
    (shim_root / "7z.shim").write_text(
        f'path = "{real_executable!s}"\n', encoding="utf-8"
    )

    def runner(executable: Path, arguments: tuple[str, ...]) -> str:
        assert executable == real_executable.resolve()
        assert arguments[0] == "i"
        return "7-Zip 26.02 (x64)\n"

    binding = resolve_seven_zip_tool_binding(shim_executable, runner=runner)
    encoded = json.dumps(binding.to_dict(), sort_keys=True)

    assert binding.executable_name == "7z.exe"
    assert binding.library_name == "7z.dll"
    assert str(real_root) not in encoded
    assert str(shim_root) not in encoded


def test_zzu_split_zip_closure_detects_midpass_evaluated_tree_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    z01, zip_path, root, executable, required, runner = _zzu_closure_fixture(tmp_path)
    original = inventory_module._snapshot_extraction_tree
    evaluated_snapshot_count = 0

    def changing_snapshot(snapshot_root: Path) -> object:
        nonlocal evaluated_snapshot_count
        result = original(snapshot_root)
        if snapshot_root.resolve() == root.resolve():
            evaluated_snapshot_count += 1
            if evaluated_snapshot_count == 1:
                (root / "P00/P00001/P00001_E01.dat").write_bytes(b"changed-midpass")
        return result

    monkeypatch.setattr(inventory_module, "_snapshot_extraction_tree", changing_snapshot)
    with pytest.raises(ExternalInventoryError, match="changed during"):
        build_zzu_split_zip_extraction_closure(
            z01,
            zip_path,
            root,
            executable,
            expected_required_relative_paths=required,
            runner=runner,
        )


def test_private_inventory_and_public_aggregate_bind_both_archive_closures(
    tmp_path: Path,
    header_only_reader: list[str],
) -> None:
    challenge_archive, challenge_root, challenge_required = _challenge_closure_fixture(
        tmp_path / "challenge-closure"
    )
    challenge_closure = build_challenge_tar_extraction_closure(
        challenge_archive,
        challenge_root,
        expected_required_relative_paths=challenge_required,
    )
    z01, zip_path, zzu_root, executable, zzu_required, runner = _zzu_closure_fixture(
        tmp_path / "zzu-closure"
    )
    zzu_closure = build_zzu_split_zip_extraction_closure(
        z01,
        zip_path,
        zzu_root,
        executable,
        expected_required_relative_paths=zzu_required,
        runner=runner,
    )
    record_inventory, _, _ = _inventory(tmp_path / "records", header_only_reader)
    inventory = build_external_inventory(
        record_inventory.records,
        archive_closures=(zzu_closure, challenge_closure),
    )
    output = tmp_path / "private-inventory.json"
    save_external_inventory(inventory, output)

    loaded = load_external_inventory(output)
    projection = external_inventory_public_projection(loaded)
    encoded = json.dumps(projection, sort_keys=True)

    assert loaded.archive_closures == (challenge_closure, zzu_closure)
    assert [item["member_count"] for item in projection["archive_closures"]] == [7, 2]  # type: ignore[index]
    assert challenge_closure.closure_sha256 in encoded
    assert zzu_closure.closure_sha256 in encoded
    assert "P00001_E01" not in encoded
    assert "c001.dat" not in encoded
    assert "set-a.tar.gz" not in encoded

    tampered = json.loads(output.read_bytes())
    tampered["archive_closures"][0]["members"][0]["size_bytes"] += 1
    output.write_text(
        json.dumps(tampered, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ExternalInventoryError):
        load_external_inventory(output)


def test_inventory_loader_rejects_oversize_input_before_parsing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inventory_path = tmp_path / "oversize.json"
    inventory_path.write_bytes(b"x" * 65)
    monkeypatch.setattr(inventory_module, "MAX_INVENTORY_BYTES", 64)

    with pytest.raises(ExternalInventoryError, match="bounded size limit"):
        load_external_inventory(inventory_path)


def test_seven_zip_runner_enforces_output_and_wall_clock_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = Path(sys.executable)
    assert inventory_module._run_seven_zip(
        executable, ("-c", "print('bounded')")
    ).splitlines() == ["bounded"]

    monkeypatch.setattr(inventory_module, "SEVEN_ZIP_STDOUT_LIMIT_BYTES", 32)
    with pytest.raises(ExternalInventoryError, match="output exceeded"):
        inventory_module._run_seven_zip(
            executable,
            ("-c", "import sys; sys.stdout.write('x' * 4096)"),
        )

    monkeypatch.setattr(inventory_module, "SEVEN_ZIP_STDOUT_LIMIT_BYTES", 64 * 1024)
    monkeypatch.setattr(inventory_module, "SEVEN_ZIP_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(inventory_module, "SEVEN_ZIP_POLL_SECONDS", 0.001)
    with pytest.raises(ExternalInventoryError, match="timed out"):
        inventory_module._run_seven_zip(
            executable,
            ("-c", "import time; time.sleep(1)"),
        )


@pytest.mark.parametrize("failure", ["tamper", "extra", "missing", "traversal"])
def test_zzu_split_zip_closure_rejects_tamper_and_member_set_drift(
    tmp_path: Path,
    failure: str,
) -> None:
    base_payloads = {
        "Child_ecg/P00/P00001/P00001_E01.hea": b"zzu-header",
        "Child_ecg/P00/P00001/P00001_E01.dat": b"zzu-data",
    }
    if failure == "extra":
        base_payloads["Child_ecg/P00/P00002/P00002_E01.dat"] = b"extra"
    elif failure == "missing":
        del base_payloads["Child_ecg/P00/P00001/P00001_E01.hea"]
    elif failure == "traversal":
        base_payloads["Child_ecg/../evil.dat"] = b"evil"
    z01, zip_path, root, executable, required, runner = _zzu_closure_fixture(
        tmp_path,
        archive_payloads=base_payloads,
    )
    if failure == "tamper":
        (root / "P00/P00001/P00001_E01.dat").write_bytes(b"changed")
    with pytest.raises(ExternalInventoryError):
        build_zzu_split_zip_extraction_closure(
            z01,
            zip_path,
            root,
            executable,
            expected_required_relative_paths=required,
            runner=runner,
        )


@pytest.mark.parametrize(
    "listing",
    [
        "Path = ../evil.dat\nSize = 1\nFolder = -\nCRC = 00000000\n",
        "Path = Child_ecg/a.dat\nSize = 1\nFolder = -\nCRC = 00000000\n"
        "\nPath = Child_ecg/a.dat\nSize = 1\nFolder = -\nCRC = 00000000\n",
        "Path = Child_ecg/a.dat\nSize = 1\nFolder = -\nCRC = 00000000\n"
        "Symbolic Link = target\n",
    ],
)
def test_seven_zip_listing_parser_rejects_traversal_duplicates_and_links(
    listing: str,
) -> None:
    with pytest.raises(ExternalInventoryError):
        parse_seven_zip_slt_listing(listing)


def _seven_zip_slt_file_block(path: str) -> str:
    return (
        f"Path = {path}\n"
        "Size = 1\n"
        "Folder = -\n"
        "Attributes = A\n"
        "CRC = 00000000\n"
        "Encrypted = -\n"
    )


def test_seven_zip_listing_parser_normalizes_windows_presentation_paths() -> None:
    listing = (
        "Path = Child_ecg\\P00\n"
        "Size = 0\n"
        "Folder = +\n"
        "Attributes = D\n"
        "CRC = \n"
        "Encrypted = -\n\n"
        + _seven_zip_slt_file_block(
            r"Child_ecg\P00\P00001\P00001_E01.hea"
        )
    )

    members = parse_seven_zip_slt_listing(listing)

    assert [(member.path, member.is_directory) for member in members] == [
        ("Child_ecg/P00", True),
        ("Child_ecg/P00/P00001/P00001_E01.hea", False),
    ]


@pytest.mark.parametrize(
    "path",
    [
        r"Child_ecg\P00/file.dat",
        r"\\server\share\file.dat",
        r"\Child_ecg\file.dat",
        r"C:\Child_ecg\file.dat",
        r"Child_ecg\..\evil.dat",
        r"Child_ecg\.\file.dat",
        r"Child_ecg\\file.dat",
        "Child_ecg\\folder\\",
        r"Child_ecg\folder.\file.dat",
        r"Child_ecg\folder \file.dat",
        r"Child_ecg\AUX.txt",
        "Child_ecg\\bad\x01\\file.dat",
    ],
)
def test_seven_zip_listing_parser_rejects_unsafe_windows_paths(path: str) -> None:
    with pytest.raises(ExternalInventoryError):
        parse_seven_zip_slt_listing(_seven_zip_slt_file_block(path))


@pytest.mark.parametrize(
    ("first", "second"),
    [
        (r"Child_ecg\P00\record.dat", "Child_ecg/P00/record.dat"),
        (r"Child_ecg\P00\RECORD.dat", r"child_ecg\p00\record.dat"),
    ],
)
def test_seven_zip_listing_parser_rejects_normalized_path_collisions(
    first: str,
    second: str,
) -> None:
    listing = _seven_zip_slt_file_block(first) + "\n" + _seven_zip_slt_file_block(second)

    with pytest.raises(ExternalInventoryError, match="duplicate member paths"):
        parse_seven_zip_slt_listing(listing)


def _exact_bound_seven_zip_26_02_or_skip() -> Path:
    candidates: list[Path] = [
        Path(r"C:\Program Files\7-Zip\7z.exe"),
        Path.home() / "scoop" / "shims" / "7z.exe",
    ]
    for command in ("7z", "7z.exe"):
        located = shutil.which(command)
        if located is not None:
            candidates.append(Path(located))
    expected_identity = (
        "26.02",
        "7z.exe",
        576_000,
        "83967f1b02b43c4efeda302795722c809e0e81b8307de73558d10484d5676a7d",
        "7z.dll",
        1_906_688,
        "69fd4df057985c40e510e2fac182881c7f85e90aa13ec703f763a8fdb2ce61f8",
    )
    for requested in dict.fromkeys(candidates):
        try:
            executable, _ = inventory_module._resolve_seven_zip_executable(requested)
            binding = resolve_seven_zip_tool_binding(requested)
        except (OSError, ExternalInventoryError):
            continue
        observed_identity = (
            binding.version,
            binding.executable_name,
            binding.executable_size_bytes,
            binding.executable_sha256,
            binding.library_name,
            binding.library_size_bytes,
            binding.library_sha256,
        )
        if observed_identity == expected_identity:
            return executable
    pytest.skip("the exact frozen 7-Zip 26.02 executable is unavailable")


@pytest.mark.skipif(sys.platform != "win32", reason="exact Windows 7-Zip presentation")
def test_exact_windows_seven_zip_nested_path_smoke(tmp_path: Path) -> None:
    executable = _exact_bound_seven_zip_26_02_or_skip()
    archive = tmp_path / "synthetic-nested.zip"
    nested_path = "Child_ecg/P00/P00001/P00001_E01.hea"
    with zipfile.ZipFile(archive, mode="w", compression=zipfile.ZIP_STORED) as handle:
        handle.writestr(nested_path, b"synthetic-header")

    listing = inventory_module._run_seven_zip(
        executable,
        ("l", "-slt", "-sccUTF-8", str(archive)),
    )
    windows_presentation = nested_path.replace("/", "\\")

    assert f"Path = {windows_presentation}" in listing
    members = parse_seven_zip_slt_listing(listing)
    assert [(member.path, member.is_directory) for member in members] == [
        (nested_path, False)
    ]
