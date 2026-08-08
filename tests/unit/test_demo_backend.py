from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
import torch
import wfdb  # type: ignore[import-untyped]
from torch.optim import AdamW

from ecg_trust.constants import LEADS, PTBXL_VERSION, SUPERCLASSES, TARGET_COLUMNS
from ecg_trust.data.dataset import NormalizationProvenance, NormalizationStats
from ecg_trust.data.manifest import sha256_file
from ecg_trust.demo_backend import (
    LIMITATIONS,
    RESEARCH_ONLY_NOTICE,
    DecisionProvenance,
    DemoArtifactError,
    DemoInferenceBackend,
    DemoInputError,
    FrozenDecisionPolicy,
    load_wfdb_physical_signal,
    validate_physical_signal,
)
from ecg_trust.experiment_config import ModelConfig
from ecg_trust.experiment_runner import build_experiment_model
from ecg_trust.models import count_parameters
from ecg_trust.protocol import TRAIN_FOLDS, ExperimentProtocol
from ecg_trust.training import save_checkpoint


@dataclass(frozen=True)
class DemoFiles:
    checkpoint: Path
    resolved_config: Path
    normalization: Path
    policy: Path


def _write_demo_files(
    root: Path,
    *,
    architecture: Literal["resnet1d", "ecg_transformer"] = "resnet1d",
    gate_threshold: float = 1.0,
) -> DemoFiles:
    selection = ModelConfig(architecture=architecture, preset="smoke")
    model = build_experiment_model(selection)
    raw_model_config = model.config  # type: ignore[attr-defined]
    config: dict[str, object] = {
        "model": {
            "architecture": selection.architecture,
            "preset": selection.preset,
            "class": f"{type(model).__module__}.{type(model).__qualname__}",
            "trainable_parameters": count_parameters(model),
            "resolved_architecture_config": asdict(raw_model_config),
        }
    }
    protocol = ExperimentProtocol.canonical()
    manifest_hash = "a" * 64
    checkpoint_path = root / "best.ckpt"
    metadata = save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=AdamW(model.parameters(), lr=1e-3),
        scaler=None,
        epoch=2,
        protocol_hash=protocol.protocol_hash,
        config=config,
        manifest_hash=manifest_hash,
    )
    resolved_path = root / "resolved_config.json"
    resolved_path.write_text(
        json.dumps(
            {"config_hash": metadata.config_hash, "config": metadata.config},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    normalization = NormalizationStats(
        mean=(0.0,) * len(LEADS),
        std=(1.0,) * len(LEADS),
        leads=LEADS,
        provenance=NormalizationProvenance(
            dataset_version=PTBXL_VERSION,
            manifest_sha256="b" * 64,
            training_folds=TRAIN_FOLDS,
            record_count=1,
            sample_count=1000,
            sampling_frequency_hz=100.0,
            samples_per_record=1000,
            path_column="record_path",
            fold_column="strat_fold",
            target_columns=TARGET_COLUMNS,
        ),
    )
    normalization_path = root / "normalization.json"
    normalization.save(normalization_path)
    policy = FrozenDecisionPolicy(
        temperature=1.5,
        classification_thresholds=(0.5,) * len(SUPERCLASSES),
        uncertainty_threshold=gate_threshold,
        provenance=DecisionProvenance(
            dataset_version=PTBXL_VERSION,
            protocol_hash=protocol.protocol_hash,
            manifest_hash=manifest_hash,
            checkpoint_config_hash=metadata.config_hash,
            checkpoint_sha256=sha256_file(checkpoint_path),
            resolved_config_sha256=sha256_file(resolved_path),
            normalization_sha256=sha256_file(normalization_path),
            calibration_folds=(9,),
        ),
    )
    policy_path = root / "decision_policy.json"
    policy.save(policy_path)
    return DemoFiles(checkpoint_path, resolved_path, normalization_path, policy_path)


def _load_backend(files: DemoFiles) -> DemoInferenceBackend:
    return DemoInferenceBackend.load(
        checkpoint_path=files.checkpoint,
        resolved_config_path=files.resolved_config,
        normalization_path=files.normalization,
        decision_policy_path=files.policy,
    )


@pytest.mark.parametrize("architecture", ["resnet1d", "ecg_transformer"])
def test_backend_loads_supported_checkpoint_and_returns_fixed_order_probabilities(
    tmp_path: Path,
    architecture: Literal["resnet1d", "ecg_transformer"],
) -> None:
    files = _write_demo_files(tmp_path, architecture=architecture)
    backend = _load_backend(files)

    result = backend.predict_signal(torch.zeros(12, 1000))
    payload = result.to_dict()

    assert result.label_order == SUPERCLASSES
    assert result.raw_logits.shape == (5,)
    assert result.raw_probabilities.shape == (5,)
    assert result.calibrated_probabilities.shape == (5,)
    assert torch.all((result.raw_probabilities >= 0) & (result.raw_probabilities <= 1))
    assert result.decision == "accept"
    assert result.decision_reason == "uncertainty_within_frozen_fold9_gate"
    assert list(payload["raw_probabilities"]) == list(SUPERCLASSES)
    assert list(payload["calibrated_probabilities"]) == list(SUPERCLASSES)
    assert payload["safety"]["notice"] == RESEARCH_ONLY_NOTICE
    assert payload["safety"]["limitations"] == list(LIMITATIONS)
    assert payload["artifact_provenance"]["calibration_folds"] == [9]
    json.dumps(payload)


def test_backend_abstains_with_frozen_gate_reason(tmp_path: Path) -> None:
    backend = _load_backend(_write_demo_files(tmp_path, gate_threshold=0.0))

    result = backend.predict_signal(np.zeros((12, 1000), dtype=np.float32))

    assert result.decision == "abstain"
    assert result.decision_reason == "uncertainty_exceeds_frozen_fold9_gate"
    assert result.uncertainty > result.gate_threshold


def test_signed_gradcam_payload_is_plot_ready(tmp_path: Path) -> None:
    backend = _load_backend(_write_demo_files(tmp_path))

    result = backend.predict_signal(
        torch.randn(12, 1000),
        attribution_method="grad_cam",
        attribution_target="MI",
    )

    assert result.attribution is not None
    assert result.attribution.target_label == "MI"
    assert result.attribution.values.shape == (1, 1000)
    assert torch.isfinite(result.attribution.values).all()
    attribution_payload = result.attribution.to_dict()
    assert attribution_payload["signed"] is True
    assert attribution_payload["shape"] == [1, 1000]
    json.dumps(attribution_payload)


def test_integrated_gradients_works_for_transformer(tmp_path: Path) -> None:
    backend = _load_backend(_write_demo_files(tmp_path, architecture="ecg_transformer"))

    result = backend.predict_signal(
        torch.randn(12, 1000),
        attribution_method="integrated_gradients",
        attribution_target=0,
        integrated_gradients_steps=2,
    )

    assert result.attribution is not None
    assert result.attribution.values.shape == (12, 1000)
    with pytest.raises(DemoInputError, match="only for ResNet1D"):
        backend.predict_signal(
            torch.zeros(12, 1000), attribution_method="grad_cam"
        )


@pytest.mark.parametrize(
    "signal, frequency, leads, units, message",
    [
        (torch.zeros(12, 999), 100.0, LEADS, "mV", "shape"),
        (torch.zeros(12, 1000), 500.0, LEADS, "mV", "100.0 Hz"),
        (torch.zeros(12, 1000), 100.0, tuple(reversed(LEADS)), "mV", "canonical order"),
        (torch.zeros(12, 1000), 100.0, LEADS, "V", "millivolts|mV"),
    ],
)
def test_in_memory_signal_contract_rejects_malformed_inputs(
    signal: torch.Tensor,
    frequency: float,
    leads: tuple[str, ...],
    units: str,
    message: str,
) -> None:
    with pytest.raises(DemoInputError, match=message):
        validate_physical_signal(
            signal,
            sampling_frequency_hz=frequency,
            lead_names=leads,
            units=units,
        )


def test_in_memory_signal_rejects_nonfinite_values() -> None:
    signal = torch.zeros(12, 1000)
    signal[0, 0] = float("nan")
    with pytest.raises(DemoInputError, match="finite"):
        validate_physical_signal(signal)


def test_wfdb_record_loading_and_prediction(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    backend = _load_backend(_write_demo_files(artifact_dir))
    time = np.linspace(0.0, 10.0, 1000, endpoint=False)
    physical = np.stack(
        [0.1 * np.sin(2 * np.pi * (index + 1) * time) for index in range(12)], axis=1
    )
    wfdb.wrsamp(
        "demo_record",
        fs=100,
        units=["mV"] * 12,
        sig_name=list(LEADS),
        p_signal=physical,
        fmt=["16"] * 12,
        write_dir=str(tmp_path),
    )

    loaded = load_wfdb_physical_signal(tmp_path / "demo_record.hea")
    result = backend.predict_record(tmp_path / "demo_record")

    assert loaded.shape == (12, 1000)
    assert torch.isfinite(loaded).all()
    assert result.source.endswith("demo_record")


@pytest.mark.parametrize("artifact_name", ["checkpoint", "resolved_config", "normalization"])
def test_backend_rejects_tampered_bound_artifacts(
    tmp_path: Path, artifact_name: str
) -> None:
    files = _write_demo_files(tmp_path)
    path = getattr(files, artifact_name)
    with path.open("ab") as handle:
        handle.write(b"\n")

    with pytest.raises(DemoArtifactError, match="does not match"):
        _load_backend(files)


def test_policy_rejects_non_fold9_provenance(tmp_path: Path) -> None:
    files = _write_demo_files(tmp_path)
    payload = json.loads(files.policy.read_text(encoding="utf-8"))
    payload["provenance"]["calibration_folds"] = [8]
    files.policy.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DemoArtifactError, match="fold"):
        FrozenDecisionPolicy.load(files.policy)
