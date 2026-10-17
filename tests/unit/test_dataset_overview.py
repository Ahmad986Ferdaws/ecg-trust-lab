from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import wfdb
from matplotlib.figure import Figure

from ecg_trust.constants import LEADS, TARGET_COLUMNS
from scripts import plot_dataset_overview as overview


def _manifest() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ecg_id": 1,
                "patient_id": 1,
                "strat_fold": 1,
                "record_path": "train",
                **dict.fromkeys(TARGET_COLUMNS, 1),
            },
            {
                "ecg_id": 2,
                "patient_id": 2,
                "strat_fold": 10,
                "record_path": "heldout",
                **dict.fromkeys(TARGET_COLUMNS, 1),
            },
        ]
    )


def _write_waveform(root: Path) -> None:
    waveform = np.tile(np.sin(np.arange(1000) / 10)[:, None], (1, 12))
    wfdb.wrsamp(
        "train",
        fs=100,
        units=["mV"] * 12,
        sig_name=list(LEADS),
        p_signal=waveform,
        fmt=["16"] * 12,
        write_dir=str(root),
    )


def test_overview_validation_preserves_counts_and_exact_integer_identifiers() -> None:
    manifest = _manifest()
    identifiers = [2**53 + 1, 2**63 - 1]
    manifest["ecg_id"] = identifiers
    validated = overview._validated_manifest(manifest)
    assert validated["ecg_id"].tolist() == identifiers
    assert validated.loc[:, list(TARGET_COLUMNS)].sum().tolist() == [2] * 5
    assert validated.groupby("strat_fold").size().to_dict() == {1: 1, 10: 1}


def _arguments(root: Path, manifest: pd.DataFrame) -> list[str]:
    path = root / "manifest.csv"
    manifest.to_csv(path, index=False)
    return [
        "--manifest",
        str(path),
        "--dataset-root",
        str(root),
        "--output-dir",
        str(root / "figures"),
    ]


@pytest.mark.parametrize("invalid", ["label", "fold", "missing", "duplicate", "leakage"])
def test_invalid_aggregate_manifest_never_publishes_charts(
    tmp_path: Path, invalid: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest()
    if invalid == "label":
        manifest.loc[1, TARGET_COLUMNS[0]] = 2  # Invalid held-out count, not selected waveform.
    elif invalid == "fold":
        manifest.loc[1, "strat_fold"] = 11
    elif invalid == "missing":
        manifest = manifest.drop(columns=[TARGET_COLUMNS[0]])
    elif invalid == "duplicate":
        manifest.loc[1, "ecg_id"] = 1
    else:
        manifest.loc[1, "patient_id"] = 1
    monkeypatch.setattr(wfdb, "rdrecord", lambda *args, **kwargs: pytest.fail("waveform read"))
    assert overview.main(_arguments(tmp_path, manifest)) == 1
    assert not list((tmp_path / "figures").glob("*.png"))


def test_missing_training_waveform_preserves_both_previous_figures(tmp_path: Path) -> None:
    output = tmp_path / "figures"
    output.mkdir()
    for name in ("ptbxl_label_and_fold_counts.png", "ptbxl_representative_ecg.png"):
        (output / name).write_bytes(b"previous figure")
    assert overview.main(_arguments(tmp_path, _manifest())) == 1
    assert all(path.read_bytes() == b"previous figure" for path in output.iterdir())


@pytest.mark.parametrize("failed_save", [1, 2])
def test_render_failure_cleans_staging_and_closes_figures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failed_save: int
) -> None:
    _write_waveform(tmp_path)
    original_save = Figure.savefig
    saves = 0

    def fail_save(self: Figure, path: Path, **kwargs: object) -> None:
        nonlocal saves
        saves += 1
        if saves == failed_save:
            Path(path).write_bytes(b"incomplete")
            raise OSError("disk full")
        original_save(self, path, **kwargs)

    figures_before = plt.get_fignums()
    monkeypatch.setattr(Figure, "savefig", fail_save)
    assert overview.main(_arguments(tmp_path, _manifest())) == 1
    assert plt.get_fignums() == figures_before
    assert not list((tmp_path / "figures").iterdir())


def test_valid_overview_publishes_both_figures_without_reading_heldout_waveforms(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_waveform(tmp_path)
    original_read = wfdb.rdrecord
    opened: list[str] = []

    def read_record(path: str, **kwargs: object) -> object:
        opened.append(Path(path).name)
        return original_read(path, **kwargs)

    monkeypatch.setattr(wfdb, "rdrecord", read_record)
    figures_before = plt.get_fignums()
    assert overview.main(_arguments(tmp_path, _manifest())) == 0
    assert opened == ["train"]
    output = tmp_path / "figures"
    assert {path.name for path in output.iterdir()} == {
        "ptbxl_label_and_fold_counts.png",
        "ptbxl_representative_ecg.png",
    }
    assert all(path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n") for path in output.iterdir())
    assert plt.get_fignums() == figures_before
