from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

import ecg_trust.data.dataset as dataset_module
from ecg_trust.constants import LEADS, TARGET_COLUMNS
from ecg_trust.data.dataset import (
    NormalizationProvenance,
    NormalizationStats,
    NormalizationValidationError,
    compute_normalization_stats,
)


def _provenance() -> NormalizationProvenance:
    return NormalizationProvenance(
        dataset_version="1.0.3",
        manifest_sha256="0" * 64,
        training_folds=(1,),
        record_count=1,
        sample_count=1000,
        sampling_frequency_hz=100.0,
        samples_per_record=1000,
        path_column="record_path",
        fold_column="strat_fold",
        target_columns=TARGET_COLUMNS,
    )


@pytest.mark.parametrize("folds", [(8,), (9,), (10,), (11,), (1, 8), (1, 9)])
def test_normalization_rejects_nontraining_folds_before_reading_waveforms(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, folds: tuple[int, ...]
) -> None:
    manifest = pd.DataFrame(
        [{"record_path": "record", "strat_fold": 1, **dict.fromkeys(TARGET_COLUMNS, 1)}]
    )
    monkeypatch.setattr(
        dataset_module.wfdb, "rdrecord", lambda _: pytest.fail("waveform should not be read")
    )

    with pytest.raises(NormalizationValidationError, match="training folds 1-7"):
        compute_normalization_stats(manifest, tmp_path, training_folds=folds)


@pytest.mark.parametrize("folds", [(8,), (9,), (10,), (11,), (1, 8), (1, 9)])
def test_normalization_rejects_nontraining_provenance_on_creation_and_load(
    tmp_path: Path, folds: tuple[int, ...]
) -> None:
    provenance = _provenance()
    with pytest.raises(NormalizationValidationError, match="training folds 1-7"):
        replace(provenance, training_folds=folds)

    stats = NormalizationStats(
        mean=(0.0,) * 12, std=(1.0,) * 12, leads=LEADS, provenance=provenance
    )
    contaminated = stats.to_dict()
    contaminated["provenance"]["training_folds"] = list(folds)
    path = tmp_path / "normalization.json"
    path.write_text(json.dumps(contaminated), encoding="utf-8")
    with pytest.raises(NormalizationValidationError, match="training folds 1-7"):
        NormalizationStats.load(path)


@pytest.mark.parametrize("folds", [(1,), (7,), (1, 3, 7), tuple(range(1, 8))])
def test_training_fold_subsets_remain_valid_provenance(folds: tuple[int, ...]) -> None:
    assert replace(_provenance(), training_folds=folds).training_folds == folds
