from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import torch
from scipy.signal import resample_poly

from ecg_trust.constants import LEADS, TARGET_COLUMNS
from ecg_trust.data.sph import (
    SPH_AMBIGUOUS_PRIMARY_CODES,
    SPH_DIRECT_PRIMARY_CODES,
    SPHExternalTransportDataset,
    SPHMetadataValidationError,
    SPHRecordValidationError,
    build_sph_transport_manifest,
    load_sph_signal,
    map_sph_superclasses,
    parse_sph_aha_codes,
    read_sph_code_dictionary,
    select_sph_exact_10s_records,
    select_sph_transport_records,
)


def _category(code: int) -> str:
    if code == 1:
        return "A"
    if code in {160, 161, 165, 166}:
        return "M"
    if code in {145, 146, 147, 148, 152, 153, 155}:
        return "L"
    if code in {80, 81, 82, 83, 84, 85, 86, 87, 88}:
        return "H"
    if code in {101, 102, 104, 105, 106, 108}:
        return "I"
    if code in {140, 142, 143}:
        return "K"
    if 200 <= code < 500:
        return "Modifier"
    return "C"


def _write_code_dictionary(path: Path) -> None:
    codes = sorted(SPH_DIRECT_PRIMARY_CODES | SPH_AMBIGUOUS_PRIMARY_CODES | {22, 332, 363})
    pd.DataFrame(
        [
            {
                "Category": _category(code),
                "Code": code,
                "Description": f"Synthetic definition {code}",
            }
            for code in codes
        ]
    ).to_csv(path, index=False)


def _write_metadata(path: Path) -> None:
    pd.DataFrame(
        [
            {
                "ECG_ID": "A00005",
                "AHA_Code": "332+160;80",
                "Patient_ID": "S00005",
                "Age": 55,
                "Sex": "M",
                "N": 5000,
                "Date": "2020-03-04",
            },
            {
                "ECG_ID": "A00002",
                "AHA_Code": "22",
                "Patient_ID": "S00002",
                "Age": 32,
                "Sex": "F",
                "N": 5000,
                "Date": "2019-09-03",
            },
            {
                "ECG_ID": "A00003",
                "AHA_Code": "145",
                "Patient_ID": "S00003",
                "Age": 63,
                "Sex": "M",
                "N": 6000,
                "Date": "2020-07-16",
            },
            {
                "ECG_ID": "A00004",
                "AHA_Code": "101;142;363+152",
                "Patient_ID": "S00004",
                "Age": 31,
                "Sex": "F",
                "N": 5000,
                "Date": "2020-07-14",
            },
            {
                "ECG_ID": "A00001",
                "AHA_Code": "1;1",
                "Patient_ID": "S00001",
                "Age": 47,
                "Sex": "M",
                "N": 5000,
                "Date": "2020-01-07",
            },
            {
                "ECG_ID": "A00006",
                "AHA_Code": "1;160",
                "Patient_ID": "S00006",
                "Age": 68,
                "Sex": "F",
                "N": 5000,
                "Date": "2020-02-02",
            },
        ]
    ).to_csv(path, index=False)


def _manifest(tmp_path: Path) -> pd.DataFrame:
    metadata_path = tmp_path / "metadata.csv"
    code_path = tmp_path / "code.csv"
    _write_metadata(metadata_path)
    _write_code_dictionary(code_path)
    return build_sph_transport_manifest(metadata_path, code_path)


def _native_signal() -> np.ndarray:
    time = np.arange(5_000, dtype=np.float64) / 500.0
    return np.stack(
        [
            0.2 * lead_index
            + np.sin(2.0 * np.pi * (1.0 + lead_index / 10.0) * time)
            + 0.05 * np.sin(2.0 * np.pi * 120.0 * time)
            for lead_index in range(len(LEADS))
        ]
    )


def _write_hdf5(path: Path, signal: np.ndarray) -> None:
    with h5py.File(path, "w") as handle:
        handle.create_dataset("ecg", data=signal)


def test_parses_composite_aha_codes_by_exact_integer_token() -> None:
    statements = parse_sph_aha_codes("332+160;145+363;80")

    assert [statement.primary_code for statement in statements] == [160, 145, 80]
    assert [statement.modifier_codes for statement in statements] == [(332,), (363,), ()]
    assert map_sph_superclasses(statements) == (0, 1, 1, 0, 0)
    assert {statement.primary_code for statement in statements} & SPH_AMBIGUOUS_PRIMARY_CODES == {
        80
    }


@pytest.mark.parametrize(
    "encoded",
    ["", "1;;160", "160+", "abc", "332", "160+145", "160+332+332"],
)
def test_rejects_malformed_aha_encodings(encoded: str) -> None:
    with pytest.raises(SPHMetadataValidationError):
        parse_sph_aha_codes(encoded)


def test_builds_manifest_and_applies_frozen_primary_cohort(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)

    assert manifest["ecg_id"].tolist() == [
        "A00001",
        "A00002",
        "A00003",
        "A00004",
        "A00005",
        "A00006",
    ]
    normal = manifest.loc[manifest["ecg_id"] == "A00001"].iloc[0]
    assert normal["primary_codes"] == (1, 1)
    assert normal[list(TARGET_COLUMNS)].tolist() == [1, 0, 0, 0, 0]

    ambiguous = manifest.loc[manifest["ecg_id"] == "A00004"].iloc[0]
    assert ambiguous["primary_codes"] == (101, 142, 152)
    assert ambiguous["modifier_codes"] == (363,)
    assert ambiguous["ambiguous_primary_codes"] == (152,)
    assert ambiguous[list(TARGET_COLUMNS)].tolist() == [0, 0, 0, 1, 1]
    assert ambiguous["mapping_status"] == "mapped_with_ambiguous"

    unmapped = manifest.loc[manifest["ecg_id"] == "A00002"].iloc[0]
    assert unmapped[list(TARGET_COLUMNS)].tolist() == [0, 0, 0, 0, 0]
    assert unmapped["mapping_status"] == "unmapped"

    primary = select_sph_transport_records(manifest)
    assert primary["ecg_id"].tolist() == ["A00001", "A00004", "A00005", "A00006"]
    assert primary["mapped_target_count"].tolist() == [1, 2, 1, 2]
    assert primary["norm_abnormal_conflict"].tolist() == [False, False, False, True]

    no_ambiguous = select_sph_transport_records(manifest, exclude_ambiguous=True)
    assert no_ambiguous["ecg_id"].tolist() == ["A00001", "A00006"]
    no_conflicts = select_sph_transport_records(manifest, exclude_norm_conflicts=True)
    assert no_conflicts["ecg_id"].tolist() == ["A00001", "A00004", "A00005"]

    exact_10s = select_sph_exact_10s_records(manifest)
    assert exact_10s["ecg_id"].tolist() == [
        "A00001",
        "A00002",
        "A00004",
        "A00005",
        "A00006",
    ]


def test_dictionary_and_metadata_validation_fail_closed(tmp_path: Path) -> None:
    code_path = tmp_path / "code.csv"
    _write_code_dictionary(code_path)
    definitions = read_sph_code_dictionary(code_path)
    assert definitions[1].description == "Synthetic definition 1"

    with pytest.raises(SPHMetadataValidationError, match="unknown codes"):
        parse_sph_aha_codes("999", known_codes=definitions)

    metadata_path = tmp_path / "metadata.csv"
    _write_metadata(metadata_path)
    invalid = pd.read_csv(metadata_path, dtype=str)
    invalid.loc[1, "N"] = "5000.5"
    invalid.to_csv(metadata_path, index=False)
    with pytest.raises(SPHMetadataValidationError, match="positive decimal integer"):
        build_sph_transport_manifest(metadata_path, code_path)


def test_loads_finite_millivolts_and_resamples_with_polyphase_filter(tmp_path: Path) -> None:
    native = _native_signal()
    record_path = tmp_path / "A00001.h5"
    _write_hdf5(record_path, native)

    signal = load_sph_signal(record_path)
    expected = resample_poly(native, up=1, down=5, axis=1).astype(np.float32)

    assert signal.shape == (12, 1_000)
    assert signal.dtype == np.float32
    assert signal.flags.c_contiguous
    assert np.isfinite(signal).all()
    np.testing.assert_allclose(signal, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("case", ["missing", "group", "shape", "non_numeric", "non_finite"])
def test_rejects_malformed_hdf5_records(tmp_path: Path, case: str) -> None:
    record_path = tmp_path / f"{case}.h5"
    with h5py.File(record_path, "w") as handle:
        if case == "group":
            handle.create_group("ecg")
        elif case == "shape":
            handle.create_dataset("ecg", data=np.zeros((5_000, 12), dtype=np.float32))
        elif case == "non_numeric":
            handle.create_dataset("ecg", data=np.full((12, 5_000), b"x", dtype="S1"))
        elif case == "non_finite":
            values = np.zeros((12, 5_000), dtype=np.float32)
            values[2, 20] = np.nan
            handle.create_dataset("ecg", data=values)

    with pytest.raises(SPHRecordValidationError):
        load_sph_signal(record_path)


def test_torch_dataset_uses_selected_manifest_without_normalizing(tmp_path: Path) -> None:
    selected = select_sph_transport_records(_manifest(tmp_path))
    selected = selected.loc[selected["ecg_id"] == "A00001"].reset_index(drop=True)
    native = _native_signal()
    _write_hdf5(tmp_path / "A00001.h5", native)

    dataset = SPHExternalTransportDataset(selected, tmp_path)
    signal, target = dataset[0]

    assert len(dataset) == 1
    assert dataset.record_path(0) == tmp_path / "A00001.h5"
    assert signal.shape == (12, 1_000)
    assert signal.dtype == torch.float32
    torch.testing.assert_close(
        signal,
        torch.from_numpy(resample_poly(native, up=1, down=5, axis=1).astype(np.float32)),
    )
    torch.testing.assert_close(target, torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0]))


def test_dataset_rejects_unmapped_and_unsafe_paths(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    unmapped = manifest.loc[manifest["ecg_id"] == "A00002"].reset_index(drop=True)
    with pytest.raises(SPHMetadataValidationError, match="direct target"):
        SPHExternalTransportDataset(unmapped, tmp_path)
    broad_dataset = SPHExternalTransportDataset(unmapped, tmp_path, allow_all_zero=True)
    assert len(broad_dataset) == 1

    selected = select_sph_transport_records(manifest).iloc[[0]].copy()
    selected.loc[0, "record_path"] = "../escape.h5"
    with pytest.raises(SPHMetadataValidationError, match="traversal"):
        SPHExternalTransportDataset(selected, tmp_path)
