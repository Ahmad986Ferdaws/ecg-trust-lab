from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from ecg_trust.constants import LEADS, SUPERCLASSES
from ecg_trust.data.dataset import (
    NormalizationProvenance,
    NormalizationStats,
    NormalizationValidationError,
)


def _provenance() -> NormalizationProvenance:
    return NormalizationProvenance(
        dataset_version="synthetic", manifest_sha256="0" * 64,
        training_folds=(1,), record_count=1, sample_count=1000,
        sampling_frequency_hz=100.0, samples_per_record=1000,
        path_column="path", fold_column="fold", target_columns=SUPERCLASSES,
    )


def test_statistics_and_provenance_are_independent_snapshots(tmp_path: Path) -> None:
    means, deviations, leads = [0.0] * 12, [1.0] * 12, list(LEADS)
    folds, targets = [1], list(SUPERCLASSES)
    provenance = replace(_provenance(), training_folds=folds, target_columns=targets)
    stats = NormalizationStats(mean=means, std=deviations, leads=leads, provenance=provenance)
    original = stats.to_dict()
    means[0], deviations[0], leads[0] = float("nan"), 0, "invalid"
    folds.append(10)
    targets[0] = "invalid"
    assert stats.to_dict() == original
    path = tmp_path / "stats.json"
    stats.save(path)
    assert NormalizationStats.load(path) == stats


@pytest.mark.parametrize("field", ["mean", "std", "leads", "provenance"])
def test_malformed_statistics_fields_use_domain_error(field: str) -> None:
    fields = dict(mean=(0.,) * 12, std=(1.,) * 12, leads=LEADS, provenance=_provenance())
    fields[field] = None
    with pytest.raises(NormalizationValidationError):
        NormalizationStats(**fields)
