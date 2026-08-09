from __future__ import annotations

from collections.abc import Callable
from types import MappingProxyType
from typing import cast

import numpy as np
import pandas as pd
import pytest
import torch
from torch import Tensor, nn
from torch.utils.data import Dataset

from ecg_trust.audit_runtime import (
    AuditInferenceSettings,
    AuditMemberRuntime,
    AuditRuntimeIntegrityError,
    CleanLogitMismatchError,
)
from ecg_trust.constants import LEADS, TARGET_COLUMNS
from ecg_trust.data.dataset import NormalizationProvenance, NormalizationStats
from ecg_trust.decisioning import CalibrationDecisionArtifact
from ecg_trust.predictions import PredictionArtifact, create_prediction_artifact
from ecg_trust.protocol import (
    FINAL_TEST_CONFIRMATION,
    TRAIN_FOLDS,
    ExperimentProtocol,
    FoldRole,
    authorize_final_test_access,
)
from ecg_trust.release_gates import CalibrationMember, RefitMember
from ecg_trust.training import TrainingRuntime


class _PhysicalDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(self, values: tuple[float, ...], targets: np.ndarray) -> None:
        self.values = values
        self.targets = targets

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        signal = torch.full((12, 1_000), self.values[index], dtype=torch.float32)
        target = torch.tensor(self.targets[index], dtype=torch.float32)
        return signal, target


class _LeadMeanModel(nn.Module):
    def forward(self, signals: Tensor) -> Tensor:
        base = signals[:, 0, :].mean(dim=1, keepdim=True)
        offsets = torch.arange(5, device=signals.device, dtype=torch.float32)
        return base + offsets.unsqueeze(0)


def _normalization() -> NormalizationStats:
    return NormalizationStats(
        mean=tuple(1.0 for _ in LEADS),
        std=tuple(2.0 for _ in LEADS),
        leads=LEADS,
        provenance=NormalizationProvenance(
            dataset_version="1.0.3",
            manifest_sha256="a" * 64,
            training_folds=TRAIN_FOLDS,
            record_count=1,
            sample_count=1_000,
            sampling_frequency_hz=100.0,
            samples_per_record=1_000,
            path_column="record_path",
            fold_column="strat_fold",
            target_columns=TARGET_COLUMNS,
        ),
    )


def _manifest() -> pd.DataFrame:
    targets = np.array(
        [
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
        ],
        dtype=np.int8,
    )
    rows: list[dict[str, object]] = []
    for ecg_id, patient_id, values in zip(
        (20, 10), (200, 100), targets, strict=True
    ):
        row: dict[str, object] = {
            "ecg_id": ecg_id,
            "patient_id": patient_id,
            "strat_fold": 10,
        }
        row.update(
            {
                target: int(value)
                for target, value in zip(TARGET_COLUMNS, values, strict=True)
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _prediction(*, mismatch: bool = False) -> PredictionArtifact:
    protocol = ExperimentProtocol.canonical()
    token = authorize_final_test_access(
        protocol,
        purpose="focused audit runtime test",
        confirmation=FINAL_TEST_CONFIRMATION,
    )
    manifest = _manifest()
    # Manifest order is ECG 20 then 10; the artifact canonicalizes it to 10 then 20.
    logits = np.array(
        [
            [2.0, 3.0, 4.0, 5.0, 6.0],
            [1.0, 2.0, 3.0, 4.0, 5.0],
        ],
        dtype=np.float64,
    )
    if mismatch:
        logits[0, 0] += 0.25
    artifact = create_prediction_artifact(
        ecg_id=manifest["ecg_id"].to_numpy(),
        patient_id=manifest["patient_id"].to_numpy(),
        strat_fold=manifest["strat_fold"].to_numpy(),
        targets=manifest.loc[:, list(TARGET_COLUMNS)].to_numpy(),
        raw_logits=logits,
        model_name="toy_refit",
        model_seed=2026,
        protocol=protocol,
        config_hash="sha256:" + "b" * 64,
        manifest_hash="sha256:" + "c" * 64,
        fold_role=FoldRole.FINAL_TEST,
        test_access=token,
    )
    object.__setattr__(artifact, "integrity_sha256", "sha256:" + "e" * 64)
    return artifact


def _runtime(*, mismatch: bool = False) -> AuditMemberRuntime:
    manifest = _manifest()
    targets = manifest.loc[:, list(TARGET_COLUMNS)].to_numpy(dtype=np.int8)
    return AuditMemberRuntime(
        member_id="resnet1d-seed2026",
        architecture="resnet1d",
        seed=2026,
        refit=cast(RefitMember, object()),
        calibration=cast(CalibrationMember, object()),
        decisions=cast(CalibrationDecisionArtifact, object()),
        sealed_prediction=_prediction(mismatch=mismatch),
        model=_LeadMeanModel(),
        physical_dataset=_PhysicalDataset((5.0, 3.0), targets),
        selected_manifest=manifest,
        normalization=_normalization(),
        resolved_config=MappingProxyType({"run_name": "toy_refit"}),
        checkpoint_sha256="sha256:" + "d" * 64,
        checkpoint_epoch=2,
        settings=AuditInferenceSettings(
            batch_size=2,
            num_workers=0,
            device="cpu",
            bf16=False,
            seed=2026,
        ),
        runtime=TrainingRuntime(device=torch.device("cpu"), bf16_enabled=False),
    )


def test_clean_inference_is_exact_aligned_and_deterministic() -> None:
    member = _runtime()

    first = member.infer_logits()
    second = member.infer_logits()
    evidence = member.assert_clean_logit_equivalence()

    assert first.ecg_id.tolist() == [10, 20]
    assert first.patient_id.tolist() == [100, 200]
    assert first.raw_logits.dtype == np.float64
    assert np.array_equal(first.raw_logits, member.sealed_prediction.raw_logits)
    assert np.array_equal(first.raw_logits, second.raw_logits)
    assert not first.raw_logits.flags.writeable
    assert evidence.exact is True
    assert evidence.maximum_absolute_error == 0.0
    assert evidence.logit_count == 10


def test_transform_runs_in_physical_space_before_frozen_normalization() -> None:
    member = _runtime()
    observed_ids: list[list[int]] = []
    observed_physical_means: list[float] = []

    def add_two_mv(signals_mv: Tensor, ecg_id: np.ndarray) -> Tensor:
        observed_ids.append(ecg_id.astype(int).tolist())
        observed_physical_means.append(float(signals_mv.mean().item()))
        return signals_mv + 2.0

    clean = member.infer_logits()
    shifted = member.infer_logits(cast(Callable[..., Tensor], add_two_mv))

    # (x + 2 - mean) / std is exactly one normalized unit above (x - mean) / std.
    assert np.array_equal(shifted.raw_logits, clean.raw_logits + 1.0)
    assert observed_ids == [[20, 10]]
    assert observed_physical_means == [4.0]


def test_clean_logit_gate_has_zero_tolerance() -> None:
    member = _runtime(mismatch=True)

    with pytest.raises(CleanLogitMismatchError) as captured:
        member.assert_clean_logit_equivalence()

    assert captured.value.member_id == "resnet1d-seed2026"
    assert captured.value.mismatch_count == 1
    assert captured.value.maximum_absolute_error == pytest.approx(0.25)


def test_transform_cannot_change_physical_batch_shape() -> None:
    member = _runtime()

    def invalid_transform(signals_mv: Tensor, ecg_id: np.ndarray) -> Tensor:
        del ecg_id
        return signals_mv[:, :, :-1]

    with pytest.raises(
        AuditRuntimeIntegrityError,
        match="transform changed the ECG shape",
    ):
        member.infer_logits(cast(Callable[..., Tensor], invalid_transform))


def test_normalization_rejects_nonphysical_shape_and_nonfinite_values() -> None:
    member = _runtime()

    with pytest.raises(AuditRuntimeIntegrityError, match="batch shape"):
        member.normalize_physical_batch(torch.zeros((1, 12, 999)))
    invalid = torch.zeros((1, 12, 1_000))
    invalid[0, 0, 0] = float("nan")
    with pytest.raises(AuditRuntimeIntegrityError, match="non-finite"):
        member.normalize_physical_batch(invalid)
