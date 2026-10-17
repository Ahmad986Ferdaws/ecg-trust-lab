from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import ecg_trust.data.dataset as dataset_module
from ecg_trust.constants import TARGET_COLUMNS
from ecg_trust.data.dataset import PTBXLDataset, RecordValidationError


def _dataset(root: Path, record_path: str) -> PTBXLDataset:
    manifest = pd.DataFrame(
        [{"record_path": record_path, "strat_fold": 1, **dict.fromkeys(TARGET_COLUMNS, 1)}]
    )
    return PTBXLDataset(manifest, root)


def test_rejects_directory_symlink_escape_before_wfdb_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "records100").symlink_to(outside, target_is_directory=True)
    dataset = _dataset(root, "records100/example")
    monkeypatch.setattr(
        dataset_module.wfdb, "rdrecord", lambda _: pytest.fail("escaped record was opened")
    )

    with pytest.raises(RecordValidationError, match="path escapes dataset root"):
        dataset.load_signal(0)


@pytest.mark.parametrize("suffix", [".hea", ".dat"])
@pytest.mark.parametrize("reference_suffix", ["", ".hea", ".dat"])
def test_rejects_canonical_record_file_symlink_escape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, suffix: str, reference_suffix: str
) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    outside = tmp_path / f"external{suffix}"
    outside.write_text("external waveform file", encoding="utf-8")
    (root / f"record{suffix}").symlink_to(outside)
    dataset = _dataset(root, f"record{reference_suffix}")
    monkeypatch.setattr(
        dataset_module.wfdb,
        "rdheader",
        lambda *args, **kwargs: SimpleNamespace(n_sig=12, file_name=["record.dat"] * 12),
    )

    with pytest.raises(RecordValidationError, match="path escapes dataset root"):
        dataset.load_signal(0)


def test_allows_symlinks_with_targets_inside_dataset_root(tmp_path: Path) -> None:
    target = tmp_path / "canonical"
    target.mkdir()
    (tmp_path / "records100").symlink_to(target, target_is_directory=True)
    for suffix in (".hea", ".dat"):
        original = target / f"original{suffix}"
        original.write_text("waveform fixture", encoding="utf-8")
        (target / f"record{suffix}").symlink_to(original)

    assert _dataset(tmp_path, "records100/record.hea").record_path(0) == target / "record"


def test_rechecks_symlink_targets_at_each_record_access(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    directory = root / "records100"
    directory.mkdir()
    dataset = _dataset(root, "records100/record")
    assert dataset.record_path(0) == directory / "record"
    directory.rmdir()
    directory.symlink_to(tmp_path, target_is_directory=True)

    with pytest.raises(RecordValidationError, match="path escapes dataset root"):
        dataset.record_path(0)
