from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

import ecg_trust.ood_completion.cohorts as cohort_module
from ecg_trust.ood_completion.cohorts import (
    COHORT_IDENTITY_ALGORITHM,
    COHORT_IDENTITY_COLUMNS,
    COHORT_IDENTITY_DOMAIN,
    CohortCounts,
    CohortRecord,
    OODCohortError,
    OODCohortIntegrityError,
    OODExpectedCohortCounts,
    OrderedCohort,
    load_ood_cohorts,
    normalize_record_path,
    ordered_role_input_identity_sha256,
)
from ecg_trust.ood_completion.embedding_artifact import EmbeddingRole
from ecg_trust.source_calibration import SourceRole, patient_split_role


def _patient_for(role: SourceRole, *, start: int) -> int:
    return next(
        patient_id
        for patient_id in range(start, start + 100_000)
        if patient_split_role(patient_id=patient_id, salt="trust-sentinel-v1") is role
    )


def _source_frame() -> tuple[pd.DataFrame, dict[str, int]]:
    patients = {
        "decision": _patient_for(SourceRole.DECISION_FIT, start=100),
        "threshold": _patient_for(
            SourceRole.CONFORMAL_AND_OOD_THRESHOLD_FIT, start=100_100
        ),
        "validation": _patient_for(SourceRole.SOURCE_VALIDATION, start=200_100),
    }
    rows = [
        {
            "ecg_id": 50,
            "patient_id": patients["threshold"],
            "strat_fold": 9,
            "record_path": "records100/00000/00050_lr",
            "label_NORM": 1,
            "waveform": "must-never-be-requested",
        },
        {
            "ecg_id": 10,
            "patient_id": 900_001,
            "strat_fold": 1,
            "record_path": "records100/00000/00010_lr",
            "label_NORM": 0,
            "waveform": "must-never-be-requested",
        },
        {
            "ecg_id": 70,
            "patient_id": patients["validation"],
            "strat_fold": 9,
            "record_path": "records100/00000/00070_lr",
            "label_NORM": 1,
            "waveform": "must-never-be-requested",
        },
        {
            "ecg_id": 40,
            "patient_id": patients["decision"],
            "strat_fold": 9,
            "record_path": "records100/00000/00040_lr",
            "label_NORM": 0,
            "waveform": "must-never-be-requested",
        },
        {
            "ecg_id": 20,
            "patient_id": 900_002,
            "strat_fold": 8,
            "record_path": "records100/00000/00020_lr",
            "label_NORM": 1,
            "waveform": "must-never-be-requested",
        },
        {
            "ecg_id": 60,
            "patient_id": patients["threshold"],
            "strat_fold": 9,
            "record_path": "records100/00000/00060_lr",
            "label_NORM": 0,
            "waveform": "must-never-be-requested",
        },
        {
            "ecg_id": 41,
            "patient_id": patients["decision"],
            "strat_fold": 9,
            "record_path": "records100/00000/00041_lr",
            "label_NORM": 1,
            "waveform": "must-never-be-requested",
        },
        {
            "ecg_id": 999,
            "patient_id": 999_999,
            "strat_fold": 10,
            "record_path": "records100/00000/00999_lr",
            "label_NORM": 1,
            "waveform": "sealed-fold10",
        },
    ]
    return pd.DataFrame(rows), patients


def _expected_counts() -> OODExpectedCohortCounts:
    return OODExpectedCohortCounts(
        reference=CohortCounts(records=2, patients=2),
        decision_fit=CohortCounts(records=2, patients=1),
        threshold_fit=CohortCounts(records=2, patients=1),
        source_validation=CohortCounts(records=1, patients=1),
        full_fold9=CohortCounts(records=5, patients=3),
    )


def _install_parquet_reader(
    monkeypatch: pytest.MonkeyPatch,
    frame: pd.DataFrame,
    *,
    honor_filter: bool = True,
) -> list[dict[str, object]]:
    calls: list[dict[str, object]] = []

    def fake_read_parquet(
        path: Path,
        *,
        columns: list[str],
        filters: list[tuple[str, str, int]],
    ) -> pd.DataFrame:
        calls.append({"path": path, "columns": columns, "filters": filters})
        selected = frame.loc[frame["strat_fold"] <= 9] if honor_filter else frame
        return selected.loc[:, columns].copy()

    monkeypatch.setattr(cohort_module.pd, "read_parquet", fake_read_parquet)
    return calls


def test_parquet_loader_pushes_fold10_filter_and_requests_no_labels_or_waveforms(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame, patients = _source_frame()
    calls = _install_parquet_reader(monkeypatch, frame)
    manifest = tmp_path / "manifest.parquet"
    manifest.write_bytes(b"synthetic parquet placeholder")

    cohorts = load_ood_cohorts(
        manifest,
        patient_split_salt="trust-sentinel-v1",
        expected_counts=_expected_counts(),
    )

    assert calls == [
        {
            "path": manifest,
            "columns": list(COHORT_IDENTITY_COLUMNS),
            "filters": [("strat_fold", "<=", 9)],
        }
    ]
    requested = set(calls[0]["columns"])
    assert requested == {"ecg_id", "patient_id", "strat_fold", "record_path"}
    assert "label_NORM" not in requested
    assert "waveform" not in requested
    assert cohorts.reference.role is EmbeddingRole.REFERENCE
    assert cohorts.threshold_fit.role is EmbeddingRole.THRESHOLD_FIT
    assert cohorts.source_validation.role is EmbeddingRole.SOURCE_VALIDATION
    assert [record.ecg_id for record in cohorts.reference.records] == [10, 20]
    assert [record.ecg_id for record in cohorts.threshold_fit.records] == [50, 60]
    assert [record.ecg_id for record in cohorts.source_validation.records] == [70]
    assert {record.patient_id for record in cohorts.threshold_fit.records} == {
        patients["threshold"]
    }
    assert {record.patient_id for record in cohorts.source_validation.records} == {
        patients["validation"]
    }
    assert cohorts.decision_fit_counts == CohortCounts(records=2, patients=1)
    assert cohorts.full_fold9_counts == CohortCounts(records=5, patients=3)
    assert cohorts.reference_sha256 == cohorts.reference.identity_sha256
    assert all(
        record.strat_fold <= 9
        for cohort in (
            cohorts.reference,
            cohorts.threshold_fit,
            cohorts.source_validation,
        )
        for record in cohort.records
    )
    fold9_records = tuple(
        CohortRecord(
            ecg_id=int(row.ecg_id),
            patient_id=int(row.patient_id),
            strat_fold=int(row.strat_fold),
            record_path=str(row.record_path),
        )
        for row in frame.loc[frame["strat_fold"] == 9].itertuples(index=False)
    )
    assert cohorts.full_fold9_sha256 == ordered_role_input_identity_sha256(fold9_records)


def test_loader_rejects_fold10_even_if_parquet_engine_violates_pushdown_filter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame, _ = _source_frame()
    _install_parquet_reader(monkeypatch, frame, honor_filter=False)
    manifest = tmp_path / "manifest.parquet"
    manifest.write_bytes(b"synthetic parquet placeholder")

    with pytest.raises(OODCohortError, match="authorized folds 1-9"):
        load_ood_cohorts(manifest, patient_split_salt="trust-sentinel-v1")


def test_domain_separated_identity_is_exact_sorted_and_has_no_trailing_newline() -> None:
    records = (
        CohortRecord(2, 20, 9, "records/00002_lr"),
        CohortRecord(1, 10, 9, "records/00001_lr"),
    )
    payload = {
        "algorithm": COHORT_IDENTITY_ALGORITHM,
        "records": [
            {
                "ecg_id": 1,
                "patient_id": 10,
                "record_path": "records/00001_lr",
                "strat_fold": 9,
            },
            {
                "ecg_id": 2,
                "patient_id": 20,
                "record_path": "records/00002_lr",
                "strat_fold": 9,
            },
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
    expected = "sha256:" + hashlib.sha256(COHORT_IDENTITY_DOMAIN + canonical).hexdigest()
    literal_escape = "sha256:" + hashlib.sha256(
        b"ecg_trust.ordered_role_input_identity.v1\\x00" + canonical
    ).hexdigest()

    assert COHORT_IDENTITY_DOMAIN.endswith(b"\x00")
    assert ordered_role_input_identity_sha256(records) == expected
    assert ordered_role_input_identity_sha256(tuple(reversed(records))) == expected
    assert expected != literal_escape
    assert OrderedCohort.create(EmbeddingRole.THRESHOLD_FIT, records).records == tuple(
        sorted(records, key=lambda record: record.ecg_id)
    )


@pytest.mark.parametrize(
    "path",
    [
        "",
        "/absolute/record",
        "C:/absolute/record",
        "C:relative/record",
        "../escape",
        "records/../escape",
        "records/./record",
        "records//record",
        "records\\record",
        "records/record\x00hidden",
    ],
)
def test_record_path_must_be_normalized_safe_relative_posix(path: str) -> None:
    with pytest.raises(OODCohortError, match="record_path"):
        normalize_record_path(path)

    with pytest.raises(OODCohortError, match="record_path"):
        CohortRecord(1, 2, 9, path)


def test_valid_record_path_is_preserved_exactly() -> None:
    path = "records100/00000/00001_lr"

    assert normalize_record_path(path) == path
    assert CohortRecord(1, 2, 9, path).record_path == path


def test_duplicate_ecg_and_cross_fold_patient_leakage_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base, _ = _source_frame()
    duplicate = base.loc[base["strat_fold"] <= 9].copy()
    duplicate.loc[duplicate.index[-1], "ecg_id"] = duplicate.iloc[0]["ecg_id"]
    _install_parquet_reader(monkeypatch, duplicate)
    manifest = tmp_path / "duplicate.parquet"
    manifest.write_bytes(b"synthetic parquet placeholder")
    with pytest.raises(OODCohortIntegrityError, match="ecg_id values must be unique"):
        load_ood_cohorts(manifest, patient_split_salt="trust-sentinel-v1")

    leaking, patients = _source_frame()
    leaking.loc[leaking["strat_fold"] == 1, "patient_id"] = patients["threshold"]
    _install_parquet_reader(monkeypatch, leaking)
    leak_manifest = tmp_path / "leak.parquet"
    leak_manifest.write_bytes(b"synthetic parquet placeholder")
    with pytest.raises(OODCohortIntegrityError, match="multiple manifest folds"):
        load_ood_cohorts(leak_manifest, patient_split_salt="trust-sentinel-v1")


def test_unexpected_counts_and_incomplete_expected_partition_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame, _ = _source_frame()
    _install_parquet_reader(monkeypatch, frame)
    manifest = tmp_path / "manifest.parquet"
    manifest.write_bytes(b"synthetic parquet placeholder")
    mismatched = OODExpectedCohortCounts(
        reference=CohortCounts(records=3, patients=3),
        decision_fit=CohortCounts(records=2, patients=1),
        threshold_fit=CohortCounts(records=2, patients=1),
        source_validation=CohortCounts(records=1, patients=1),
        full_fold9=CohortCounts(records=5, patients=3),
    )

    with pytest.raises(OODCohortIntegrityError, match="reference counts differ"):
        load_ood_cohorts(
            manifest,
            patient_split_salt="trust-sentinel-v1",
            expected_counts=mismatched,
        )

    with pytest.raises(OODCohortError, match="exhaust full_fold9"):
        OODExpectedCohortCounts(
            reference=CohortCounts(records=2, patients=2),
            decision_fit=CohortCounts(records=1, patients=1),
            threshold_fit=CohortCounts(records=2, patients=1),
            source_validation=CohortCounts(records=1, patients=1),
            full_fold9=CohortCounts(records=5, patients=3),
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda frame: frame.__setitem__("ecg_id", 1.5), "ecg_id must contain integers"),
        (
            lambda frame: frame.__setitem__("record_path", "../escape"),
            "record_path",
        ),
        (lambda frame: frame.drop(columns=["record_path"], inplace=True), "four label-free"),
    ],
)
def test_manifest_contract_rejects_malformed_label_free_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: Callable[[pd.DataFrame], None],
    message: str,
) -> None:
    frame, _ = _source_frame()
    frame = frame.loc[frame["strat_fold"] <= 9, list(COHORT_IDENTITY_COLUMNS)].copy()
    mutation(frame)

    def fake_read_parquet(*args: object, **kwargs: object) -> pd.DataFrame:
        return frame.copy()

    monkeypatch.setattr(cohort_module.pd, "read_parquet", fake_read_parquet)
    manifest = tmp_path / "manifest.parquet"
    manifest.write_bytes(b"synthetic parquet placeholder")
    with pytest.raises(OODCohortError, match=message):
        load_ood_cohorts(manifest, patient_split_salt="trust-sentinel-v1")


def test_loader_accepts_only_existing_nonsymlink_parquet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        cohort_module.pd,
        "read_parquet",
        lambda *args, **kwargs: pytest.fail("reader must not be called"),
    )
    with pytest.raises(OODCohortError, match="parquet"):
        load_ood_cohorts(tmp_path / "manifest.csv", patient_split_salt="salt")
    with pytest.raises(OODCohortError, match="missing or symbolic"):
        load_ood_cohorts(tmp_path / "missing.parquet", patient_split_salt="salt")
