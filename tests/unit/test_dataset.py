from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch

import ecg_trust.data.dataset as dataset_module
from ecg_trust.constants import LEADS, SUPERCLASSES
from ecg_trust.data.dataset import (
    ManifestValidationError,
    NormalizationStats,
    NormalizationValidationError,
    PTBXLDataset,
    RecordValidationError,
    compute_normalization_stats,
)
from ecg_trust.protocol import (
    FINAL_TEST_CONFIRMATION,
    ExperimentProtocol,
    FinalTestAccessError,
    authorize_final_test_access,
)


def _targets(*positive: str) -> dict[str, int]:
    return {f"label_{label}": int(label in positive) for label in SUPERCLASSES}


def _manifest(*rows: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _signal(*, offset: float = 0.0, samples: int = 1_000) -> np.ndarray:
    time = np.arange(samples, dtype=np.float64) / 100.0
    return np.column_stack([time + offset + 10.0 * lead_index for lead_index in range(len(LEADS))])


def _record(
    signal: np.ndarray,
    *,
    lead_order: tuple[str, ...] = LEADS,
    frequency_hz: float = 100.0,
) -> SimpleNamespace:
    canonical_position = {lead.casefold(): index for index, lead in enumerate(LEADS)}
    columns = [canonical_position[lead.casefold()] for lead in lead_order]
    return SimpleNamespace(
        fs=frequency_hz,
        p_signal=signal[:, columns],
        sig_name=list(lead_order),
    )


def test_loads_reorders_and_filters_manifest_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    shuffled_leads = (
        "v6",
        "AVL",
        "II",
        "v1",
        "I",
        "AVR",
        "III",
        "aVF",
        "V5",
        "V2",
        "V4",
        "V3",
    )
    expected = _signal(offset=2.5)
    calls: list[str] = []

    def fake_read(record_name: str) -> SimpleNamespace:
        calls.append(record_name)
        return _record(expected, lead_order=shuffled_leads)

    monkeypatch.setattr(dataset_module.wfdb, "rdrecord", fake_read)
    manifest = _manifest(
        {
            "record_path": "selected.hea",
            "split": "development",
            "strat_fold": 1,
            **_targets("NORM", "CD"),
        },
        {
            "record_path": "wrong_fold",
            "split": "development",
            "strat_fold": 2,
            **_targets("MI"),
        },
        {
            "record_path": "wrong_split",
            "split": "test",
            "strat_fold": 1,
            **_targets("STTC"),
        },
    )

    dataset = PTBXLDataset(
        manifest,
        tmp_path,
        split="development",
        folds=1,
    )
    signal, target = dataset[0]

    assert len(dataset) == 1
    assert dataset.folds == (1,)
    assert dataset.manifest["record_path"].tolist() == ["selected.hea"]
    assert dataset.record_path(0) == tmp_path / "selected"
    assert calls == [str(tmp_path / "selected")]
    assert signal.shape == (12, 1_000)
    assert signal.dtype == torch.float32
    assert signal.is_contiguous()
    assert torch.isfinite(signal).all()
    torch.testing.assert_close(signal, torch.from_numpy(expected.T.astype(np.float32)))
    assert target.dtype == torch.float32
    torch.testing.assert_close(target, torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0]))


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("frequency", "sampling frequency"),
        ("length", "has 999 samples"),
        ("duplicate_lead", "duplicate lead"),
        ("unexpected_lead", "unexpected lead"),
        ("non_finite", "non-finite signal"),
        ("missing_signal", "no physical signal"),
    ],
)
def test_rejects_malformed_wfdb_records(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    case: str,
    message: str,
) -> None:
    signal = _signal()
    lead_order = list(LEADS)
    frequency_hz = 100.0
    physical_signal: np.ndarray | None = signal
    if case == "frequency":
        frequency_hz = 500.0
    elif case == "length":
        physical_signal = signal[:-1]
    elif case == "duplicate_lead":
        lead_order[-1] = "V5"
        physical_signal = signal.copy()
    elif case == "unexpected_lead":
        lead_order[-1] = "XYZ"
        physical_signal = signal.copy()
    elif case == "non_finite":
        physical_signal = signal.copy()
        physical_signal[20, 4] = np.nan
    elif case == "missing_signal":
        physical_signal = None

    if case in {"duplicate_lead", "unexpected_lead"}:
        record = SimpleNamespace(
            fs=frequency_hz,
            p_signal=physical_signal,
            sig_name=lead_order,
        )
    else:
        record = SimpleNamespace(
            fs=frequency_hz,
            p_signal=physical_signal,
            sig_name=list(LEADS),
        )
    monkeypatch.setattr(dataset_module.wfdb, "rdrecord", lambda _: record)
    dataset = PTBXLDataset(
        _manifest({"record_path": "record", "strat_fold": 1, **_targets("NORM")}),
        tmp_path,
    )

    with pytest.raises(RecordValidationError, match=message):
        dataset.load_signal(0)


def test_rejects_malformed_manifest_and_empty_selection(tmp_path: Path) -> None:
    valid_row = {"record_path": "record", "strat_fold": 1, **_targets("NORM")}

    missing_target = dict(valid_row)
    del missing_target["label_HYP"]
    with pytest.raises(ManifestValidationError, match="missing required columns"):
        PTBXLDataset(_manifest(missing_target), tmp_path)

    non_binary = dict(valid_row)
    non_binary["label_MI"] = 0.5
    with pytest.raises(ManifestValidationError, match="only 0 or 1"):
        PTBXLDataset(_manifest(non_binary), tmp_path)

    all_negative = {"record_path": "record", "strat_fold": 1, **_targets()}
    with pytest.raises(ManifestValidationError, match="at least one positive"):
        PTBXLDataset(_manifest(all_negative), tmp_path)

    traversal = {"record_path": "../escape", "strat_fold": 1, **_targets("NORM")}
    with pytest.raises(ManifestValidationError, match="invalid record path"):
        PTBXLDataset(_manifest(traversal), tmp_path)

    fractional_fold = dict(valid_row)
    fractional_fold["strat_fold"] = 1.5
    with pytest.raises(ManifestValidationError, match="finite integers"):
        PTBXLDataset(_manifest(fractional_fold), tmp_path, folds=1)

    with pytest.raises(ManifestValidationError, match="no manifest rows match"):
        PTBXLDataset(_manifest(valid_row), tmp_path, folds=2)


def test_fold_10_is_sealed_for_direct_and_split_access(tmp_path: Path) -> None:
    protocol = ExperimentProtocol.canonical()
    manifest = _manifest(
        {
            "record_path": "final_test_record",
            "split": "test",
            "strat_fold": 10,
            **_targets("MI"),
        }
    )

    with pytest.raises(FinalTestAccessError, match="fold 10 is sealed"):
        PTBXLDataset(manifest, tmp_path, folds=10)
    with pytest.raises(FinalTestAccessError, match="fold 10 is sealed"):
        PTBXLDataset(manifest, tmp_path, split="test")

    token = authorize_final_test_access(
        protocol,
        purpose="one-time locked evaluation in the unit test",
        confirmation=FINAL_TEST_CONFIRMATION,
    )
    dataset = PTBXLDataset(
        manifest,
        tmp_path,
        split="test",
        protocol=protocol,
        test_access=token,
    )
    assert len(dataset) == 1
    assert dataset.manifest["record_path"].tolist() == ["final_test_record"]


def test_computes_training_fold_only_streaming_stats_and_round_trips(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    signals = {
        "train_a": _signal(offset=-3.0),
        "train_b": _signal(offset=7.0),
        "held_out": _signal(offset=10_000.0),
    }

    def fake_read(record_name: str) -> SimpleNamespace:
        return _record(signals[Path(record_name).name])

    monkeypatch.setattr(dataset_module.wfdb, "rdrecord", fake_read)
    manifest = _manifest(
        {"record_path": "train_a", "strat_fold": 1, **_targets("NORM")},
        {"record_path": "train_b", "strat_fold": 1, **_targets("MI")},
        {"record_path": "held_out", "strat_fold": 2, **_targets("STTC")},
    )

    stats = compute_normalization_stats(manifest, tmp_path, training_folds=(1,))
    expected = np.concatenate([signals["train_a"], signals["train_b"]], axis=0)

    np.testing.assert_allclose(stats.mean, expected.mean(axis=0), rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(stats.std, expected.std(axis=0), rtol=1e-7, atol=1e-7)
    assert stats.leads == LEADS
    assert stats.provenance.training_folds == (1,)
    assert stats.provenance.record_count == 2
    assert stats.provenance.sample_count == 2_000
    assert stats.provenance.dataset_version == "1.0.3"
    assert len(stats.provenance.manifest_sha256) == 64

    reordered_stats = compute_normalization_stats(
        manifest.iloc[::-1].reset_index(drop=True), tmp_path, training_folds=1
    )
    assert reordered_stats.provenance.manifest_sha256 == stats.provenance.manifest_sha256
    np.testing.assert_allclose(reordered_stats.mean, stats.mean, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(reordered_stats.std, stats.std, rtol=0.0, atol=1e-12)

    stats_path = tmp_path / "artifacts" / "normalization.json"
    stats.save(stats_path)
    loaded = NormalizationStats.load(stats_path)
    assert loaded == stats
    on_disk = json.loads(stats_path.read_text(encoding="utf-8"))
    assert on_disk["schema_version"] == 1
    assert on_disk["provenance"]["training_folds"] == [1]

    normalized_dataset = PTBXLDataset(
        manifest,
        tmp_path,
        folds=1,
        normalization=loaded,
    )
    normalized = torch.stack(
        [normalized_dataset.load_signal(index) for index in range(len(normalized_dataset))]
    )
    flattened = normalized.permute(1, 0, 2).reshape(len(LEADS), -1)
    torch.testing.assert_close(flattened.mean(dim=1), torch.zeros(len(LEADS)), atol=2e-6, rtol=0)
    torch.testing.assert_close(flattened.std(dim=1, correction=0), torch.ones(len(LEADS)))


def test_rejects_zero_variance_stats_and_malformed_stats_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    constant = np.column_stack(
        [np.full(1_000, lead_index, dtype=np.float64) for lead_index in range(len(LEADS))]
    )
    monkeypatch.setattr(
        dataset_module.wfdb,
        "rdrecord",
        lambda _: _record(constant),
    )
    manifest = _manifest({"record_path": "constant", "strat_fold": 1, **_targets("NORM")})
    with pytest.raises(NormalizationValidationError, match="zero-variance leads"):
        compute_normalization_stats(manifest, tmp_path, training_folds=1)

    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text('{"schema_version": 999}', encoding="utf-8")
    with pytest.raises(NormalizationValidationError, match="unsupported"):
        NormalizationStats.load(invalid_path)
