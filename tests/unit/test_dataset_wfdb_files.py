from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import wfdb

from ecg_trust.constants import LEADS, TARGET_COLUMNS
from ecg_trust.data.dataset import PTBXLDataset, RecordValidationError


def _dataset(root: Path, record_path: str = "record") -> PTBXLDataset:
    return PTBXLDataset(
        pd.DataFrame(
            [{"record_path": record_path, "strat_fold": 1, **dict.fromkeys(TARGET_COLUMNS, 1)}]
        ),
        root,
    )


def _write_record(directory: Path) -> np.ndarray:
    directory.mkdir(exist_ok=True)
    values = np.tile(np.sin(np.arange(1000) / 20.0)[:, None], (1, 12))
    wfdb.wrsamp(
        "source", fs=100, units=["mV"] * 12, sig_name=list(LEADS),
        p_signal=values, fmt=["16"] * 12, write_dir=str(directory),
    )
    return values


def _copy_header(source: Path, destination: Path) -> None:
    header = source.read_text(encoding="ascii")
    destination.write_text(header.replace("source 12 ", "record 12 ", 1), encoding="ascii")


def test_real_wfdb_rejects_external_signal_file_named_by_header(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outside = tmp_path / "outside"
    _write_record(outside)
    root = tmp_path / "dataset"
    root.mkdir()
    _copy_header(outside / "source.hea", root / "record.hea")
    (root / "source.dat").symlink_to(outside / "source.dat")
    # The actual header points to source.dat; there is no record.dat to check.
    assert wfdb.rdheader(str(root / "record")).file_name == ["source.dat"] * 12
    monkeypatch.setattr(wfdb, "rdrecord", lambda _: pytest.fail("signal decoder was called"))

    with pytest.raises(RecordValidationError, match="path escapes dataset root"):
        _dataset(root).load_signal(0)


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("signal_symlink", [False, True])
def test_real_wfdb_accepts_declared_signal_files_inside_root(
    tmp_path: Path, nested: bool, signal_symlink: bool
) -> None:
    directory = tmp_path / "records100" if nested else tmp_path
    values = _write_record(directory)
    _copy_header(directory / "source.hea", directory / "record.hea")
    if signal_symlink:
        (directory / "source.dat").rename(directory / "physical.dat")
        (directory / "source.dat").symlink_to(directory / "physical.dat")
    signal = _dataset(tmp_path, "records100/record" if nested else "record").load_signal(0)

    np.testing.assert_allclose(signal.numpy(), values.T, atol=2e-5, rtol=0)


def test_real_wfdb_rejects_segments_without_opening_segment_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "record.hea").write_text(
        "record/1 12 100 1000\nmissing_segment 1000\n", encoding="ascii"
    )
    monkeypatch.setattr(wfdb, "rdrecord", lambda _: pytest.fail("signal decoder was called"))

    with pytest.raises(RecordValidationError, match="single-segment"):
        _dataset(tmp_path).load_signal(0)


@pytest.mark.parametrize("filename", ["../escape.dat", "/escape.dat", "C:\\escape.dat", None])
def test_rejects_unsafe_header_signal_paths_before_decoding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, filename: object
) -> None:
    monkeypatch.setattr(
        wfdb, "rdheader",
        lambda *args, **kwargs: SimpleNamespace(n_sig=12, file_name=[filename] * 12),
    )
    monkeypatch.setattr(wfdb, "rdrecord", lambda _: pytest.fail("signal decoder was called"))

    with pytest.raises(RecordValidationError, match="invalid WFDB signal file"):
        _dataset(tmp_path).load_signal(0)


@pytest.mark.parametrize("filenames", [None, "record.dat", [], ["record.dat"] * 11])
def test_rejects_missing_or_misaligned_header_signal_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, filenames: object
) -> None:
    monkeypatch.setattr(
        wfdb, "rdheader",
        lambda *args, **kwargs: SimpleNamespace(n_sig=12, file_name=filenames),
    )
    monkeypatch.setattr(wfdb, "rdrecord", lambda _: pytest.fail("signal decoder was called"))

    with pytest.raises(RecordValidationError, match="one signal filename"):
        _dataset(tmp_path).load_signal(0)
