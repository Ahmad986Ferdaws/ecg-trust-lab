from __future__ import annotations

import _thread
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import psutil  # type: ignore[import-untyped]
import pytest
import torch
import yaml  # type: ignore[import-untyped]

from ecg_trust.constants import LEADS
from ecg_trust.data.dataset import NormalizationStats
from ecg_trust.experiment_config import ModelConfig
from ecg_trust.experiment_runner import build_experiment_model
from ecg_trust.ood_completion.runtime import prepare_resnet_for_embedding
from ecg_trust.ood_v2 import bundle as bundle_module
from ecg_trust.ood_v2 import pipeline
from ecg_trust.ood_v2.adapters import (
    ADAPTER_VERSION,
    PHYSICAL_UNITS,
    RESAMPLE_PADTYPE,
    RESAMPLE_WINDOW,
    TARGET_FREQUENCY_HZ,
    TARGET_SAMPLES,
    WINDOW_SECONDS,
    AdapterProvenance,
    CanonicalExternalSignal,
    ExternalECGAdapterError,
)
from ecg_trust.ood_v2.bundle import (
    ACCESS_MARKER_FILENAME,
    FAILURE_RECEIPT_FILENAME,
    SUCCESS_MANIFEST_FILENAME,
    sha256_bytes,
    sha256_file,
    verify_external_v2_bundle,
)
from ecg_trust.ood_v2.inventory import (
    CHALLENGE_2011_DATASET,
    CHALLENGE_2011_VERSION,
    CONFIRMATION_LOCKBOX_ROLE,
    ExternalInventoryRecord,
)
from ecg_trust.ood_v2.models import AggregateRouteCounts, ResamplingUnit
from ecg_trust.quality.signal_quality import (
    DEFAULT_SIGNAL_QUALITY_CONFIG,
    SignalMetadata,
    assess_signal_quality,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARENT_PATH = PROJECT_ROOT / "configs" / "trust_sentinel_ood_external_v2.yaml"
SUCCESSOR_PARENT_PATH = (
    PROJECT_ROOT / "configs" / "trust_sentinel_ood_external_v2_1.yaml"
)


def _inventory_record() -> ExternalInventoryRecord:
    return ExternalInventoryRecord(
        dataset=CHALLENGE_2011_DATASET,
        dataset_version=CHALLENGE_2011_VERSION,
        site="PhysioNet Challenge 2011 Set A",
        site_alias="challenge-2011-set-a",
        patient_key=None,
        record_ref="A0001",
        source_role=CONFIRMATION_LOCKBOX_ROLE,
        raw_header_sha256="a" * 64,
        raw_header_size_bytes=128,
        raw_data_sha256="b" * 64,
        raw_data_size_bytes=24_000,
        sampling_frequency_hz=500.0,
        source_sample_count=5_000,
        duration_seconds=10.0,
        lead_count=len(LEADS),
        raw_ordered_leads=LEADS,
        canonical_ordered_leads=LEADS,
        raw_data_file_names=("A0001.dat",) * len(LEADS),
        raw_physical_units=(PHYSICAL_UNITS,) * len(LEADS),
        challenge_quality_label="acceptable",
        pediatric_12_lead=None,
    )


def _adapted_signal(*, source_sample_count: int = 5_000) -> CanonicalExternalSignal:
    provenance = AdapterProvenance(
        adapter_version=ADAPTER_VERSION,
        raw_header_sha256="a" * 64,
        raw_header_size_bytes=128,
        raw_data_sha256="b" * 64,
        raw_data_size_bytes=24_000,
        source_frequency_hz=500.0,
        source_sample_count=source_sample_count,
        source_duration_seconds=10.0,
        source_lead_names=LEADS,
        canonical_leads=LEADS,
        output_leads=LEADS,
        source_data_file_names=("A0001.dat",) * len(LEADS),
        raw_physical_units=(PHYSICAL_UNITS,) * len(LEADS),
        physical_units=PHYSICAL_UNITS,
        window_start_sample=0,
        window_source_samples=5_000,
        window_seconds=WINDOW_SECONDS,
        resample_up=1,
        resample_down=5,
        resample_window=RESAMPLE_WINDOW,
        resample_padtype=RESAMPLE_PADTYPE,
        target_frequency_hz=TARGET_FREQUENCY_HZ,
        target_samples=TARGET_SAMPLES,
    )
    return CanonicalExternalSignal(
        np.zeros((len(LEADS), TARGET_SAMPLES), dtype=np.float32),
        provenance,
    )


def _private_row(
    *,
    adapted: CanonicalExternalSignal | None = None,
    quality_status: str = "pass",
    route: str = "UNSUPPORTED_INPUT",
) -> Any:
    signal = _adapted_signal() if adapted is None else adapted
    return pipeline._PrivateRecordEvidence(  # noqa: SLF001
        dataset=CHALLENGE_2011_DATASET,
        record_ref="A0001",
        patient_key=None,
        challenge_quality_label="acceptable",
        adapter_provenance_sha256=signal.provenance_sha256,
        adapter_source_sample_count=signal.provenance.source_sample_count,
        adapter_raw_physical_units=signal.provenance.raw_physical_units,
        canonical_signal_sha256=pipeline._tensor_sha256(signal.signal_mv),  # noqa: SLF001
        quality_report_sha256="sha256:" + "c" * 64,
        quality_report=None,
        quality_status=quality_status,
        quality_reason_codes=() if quality_status == "pass" else ("nonfinite_signal",),
        route=route,
        distribution_score=300.0 if quality_status == "pass" else None,
        entropy=0.1 if quality_status == "pass" else None,
        entropy_accepted=True if quality_status == "pass" else None,
        conformal_decisions=("positive",) * 5 if quality_status == "pass" else None,
        all_conformal_decisions_singleton=True if quality_status == "pass" else None,
    )


def test_original_v2_parent_is_metadata_visible_but_never_executable(
    tmp_path: Path,
) -> None:
    parent = pipeline.load_parent_config(PARENT_PATH)
    assert parent.status == "frozen_parent_preregistration_pre_download"
    with pytest.raises(
        pipeline.OODExternalV2ExecutionError,
        match="^PRE_INFERENCE_PROTOCOL_INFEASIBLE:",
    ):
        pipeline.prepare_ood_external_v2(
            parent_path=PARENT_PATH,
            child_path=tmp_path / "must-not-be-read.json",
            project_root=PROJECT_ROOT,
            code_revision="0" * 40,
        )
    assert not (tmp_path / "must-not-be-read.json").exists()


def test_original_v2_freeze_refuses_before_output_or_source_access(
    tmp_path: Path,
) -> None:
    output = tmp_path / "child.json"
    with pytest.raises(
        pipeline.OODExternalV2ExecutionError,
        match="^PRE_INFERENCE_PROTOCOL_INFEASIBLE:",
    ):
        pipeline.freeze_external_v2_child_contract(
            parent_path=PARENT_PATH,
            project_root=PROJECT_ROOT,
            inventory_path=tmp_path / "missing-private-inventory.json",
            public_projection_path=tmp_path / "missing-public-projection.json",
            implementation_revision="0" * 40,
            frozen_at_utc="2026-08-29T00:00:00Z",
            challenge_root=tmp_path / "missing-challenge",
            zzu_root=tmp_path / "missing-zzu",
            challenge_records=1_000,
            zzu_records=12_328,
            zzu_patients=10_350,
            selected_records_total=13_328,
            output_path=output,
        )
    assert not output.exists()


def _isolated_successor_preflight_project(tmp_path: Path) -> Path:
    for relative in (
        "configs/trust_sentinel_ood_external_v2.yaml",
        "configs/trust_sentinel_ood_external_v2_1.yaml",
        "configs/trust_sentinel_ood_external_v2_termination.yaml",
        "docs/TRUST_SENTINEL_OOD_EXTERNAL_V2_INFEASIBILITY.md",
        "artifacts/trust_sentinel/ood_external_v2_preflight/private/"
        "external-waveform-inventory.json",
        "artifacts/trust_sentinel/ood_external_v2_preflight/public/"
        "external-inventory-summary.json",
    ):
        source = PROJECT_ROOT / relative
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    subprocess.run(
        ["git", "init"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    for arguments in (
        ("config", "user.name", "Protocol Test"),
        ("config", "user.email", "protocol-test@example.invalid"),
        ("remote", "add", "origin", pipeline.EXPECTED_GIT_REMOTE_URL),
        ("config", "branch.main.remote", "origin"),
        ("config", "branch.main.merge", "refs/heads/main"),
    ):
        subprocess.run(
            ["git", *arguments],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )
    return tmp_path


def test_successor_preflight_binds_predecessor_and_exact_parent_path(
    tmp_path: Path,
) -> None:
    isolated = _isolated_successor_preflight_project(tmp_path)
    parent_path = isolated / "configs" / SUCCESSOR_PARENT_PATH.name
    verified = pipeline.verify_successor_parent_preflight(
        parent_path,
        project_root=isolated,
    )
    assert verified.path == parent_path
    assert len(verified.raw_source_bindings) == 10
    assert verified.seven_zip_tool_binding.version == "26.02"
    assert verified.seven_zip_tool_binding.executable_name == "7z.exe"
    assert verified.seven_zip_tool_binding.library_name == "7z.dll"


def test_successor_preflight_rejects_hash_identical_parent_copy(
    tmp_path: Path,
) -> None:
    isolated = _isolated_successor_preflight_project(tmp_path / "project")
    copied = isolated / "alternate" / SUCCESSOR_PARENT_PATH.name
    copied.parent.mkdir()
    shutil.copyfile(isolated / "configs" / SUCCESSOR_PARENT_PATH.name, copied)
    with pytest.raises(
        pipeline.OODExternalV2ConfigError,
        match="exact canonical project path",
    ):
        pipeline.verify_successor_parent_preflight(
            copied,
            project_root=isolated,
        )


@pytest.mark.parametrize(
    ("keys", "replacement", "message"),
    (
        (
            (
                "revision_boundary",
                "remote",
                "allowed_static_remote_ref",
                "revision",
            ),
            "0" * 40,
            "successor remote declaration differs",
        ),
        (
            (
                "design_history",
                "pre_inventory_remote_preflight",
                "first_frozen_implementation_revision",
            ),
            "0" * 40,
            "successor pre-inventory amendment declaration differs",
        ),
        (
            (
                "design_history",
                "private_remote_authentication_preflight",
                "predecessor_amended_implementation_revision",
            ),
            "0" * 40,
            "successor private-remote authentication amendment declaration differs",
        ),
        (
            (
                "revision_boundary",
                "remote",
                "authentication",
                "gcm_environment_exact",
                "GCM_TRACE_SECRETS",
            ),
            "1",
            "successor remote declaration differs",
        ),
        (
            (
                "revision_boundary",
                "remote",
                "authentication",
                "visibility_proof",
                "anonymous_return_code",
            ),
            0,
            "successor remote declaration differs",
        ),
        (
            (
                "revision_boundary",
                "remote",
                "authentication",
                "process_boundary",
                "stream_capture",
                "git_remote_standard_output_limit_bytes",
            ),
            8_192,
            "successor remote declaration differs",
        ),
    ),
)
def test_successor_preflight_enforces_remote_and_amendment_semantics(
    tmp_path: Path,
    keys: tuple[str, ...],
    replacement: object,
    message: str,
) -> None:
    isolated = _isolated_successor_preflight_project(tmp_path)
    parent_path = isolated / "configs" / SUCCESSOR_PARENT_PATH.name
    payload = cast(dict[str, Any], yaml.safe_load(parent_path.read_text(encoding="utf-8")))
    target: dict[str, Any] = payload
    for key in keys[:-1]:
        target = cast(dict[str, Any], target[key])
    target[keys[-1]] = replacement
    parent_path.write_text(
        yaml.safe_dump(payload, allow_unicode=False, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(pipeline.OODExternalV2ConfigError, match=message):
        pipeline.verify_successor_parent_preflight(
            parent_path,
            project_root=isolated,
        )


def test_record_bootstrap_uses_pinned_ordered_index_draws() -> None:
    observed = pipeline._bootstrap_rates(  # noqa: SLF001
        np.asarray([True, False, True], dtype=np.bool_),
        resampling_unit=ResamplingUnit.RECORD,
        seed=7,
        replicates=8,
    )
    expected = np.asarray(
        [
            2 / 3,
            2 / 3,
            1.0,
            1.0,
            2 / 3,
            1.0,
            2 / 3,
            2 / 3,
        ],
        dtype=np.float64,
    )
    assert np.array_equal(observed, expected)


def test_adapter_inventory_crosslink_accepts_exact_source_metadata() -> None:
    pipeline._verify_adapter_against_inventory(  # noqa: SLF001
        _adapted_signal(),
        _inventory_record(),
    )


def test_adapter_inventory_crosslink_rejects_source_sample_mismatch() -> None:
    with pytest.raises(
        ExternalECGAdapterError,
        match="provenance differs",
    ):
        pipeline._verify_adapter_against_inventory(  # noqa: SLF001
            _adapted_signal(source_sample_count=5_001),
            _inventory_record(),
        )


def test_adapter_failure_is_terminal_but_quality_invalid_retains_provenance() -> None:
    invalid = _private_row(quality_status="invalid", route="INVALID_INPUT")
    pipeline._verify_private_route_semantics(  # noqa: SLF001
        invalid,
        threshold=270.9668613705653,
    )
    pipeline._verify_private_adapter_semantics(  # noqa: SLF001
        invalid,
        _inventory_record(),
    )

    adapter_failure = replace(
        invalid,
        adapter_provenance_sha256=None,
        adapter_source_sample_count=None,
        adapter_raw_physical_units=None,
        canonical_signal_sha256=None,
        quality_report_sha256=None,
        quality_reason_codes=("adapter_contract_failure",),
    )
    with pytest.raises(
        pipeline.OODExternalV2IntegrityError,
        match="cannot encode an adapter-contract failure",
    ):
        pipeline._verify_private_adapter_semantics(  # noqa: SLF001
            adapter_failure,
            _inventory_record(),
        )


def test_natural_quality_invalid_does_not_break_adapter_integrity_gate() -> None:
    invalid = _private_row(quality_status="invalid", route="INVALID_INPUT")
    assert pipeline._raw_adapter_evidence_complete(  # noqa: SLF001
        (invalid,),
        skipped=0,
    )
    assert not pipeline._raw_adapter_evidence_complete(  # noqa: SLF001
        (replace(invalid, adapter_provenance_sha256=None),),
        skipped=0,
    )
    assert not pipeline._raw_adapter_evidence_complete(  # noqa: SLF001
        (invalid,),
        skipped=1,
    )


def test_record_evidence_index_omits_full_quality_report_body() -> None:
    report = {"large": ["private"] * 1_000}
    row = replace(_private_row(), quality_report=report)
    serialized = pipeline._private_record_index_dict(row)  # noqa: SLF001
    assert serialized["quality_report"] is None
    assert serialized["quality_report_sha256"] == row.quality_report_sha256


def test_raw_adapter_replay_is_array_exact_and_adapter_failure_is_terminal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    record = _inventory_record()
    adapted = _adapted_signal()
    row = _private_row(adapted=adapted)
    (tmp_path / "A0001.hea").write_bytes(b"header")
    (tmp_path / "A0001.dat").write_bytes(b"data")
    inputs = cast(
        Any,
        SimpleNamespace(
            inventory=SimpleNamespace(records=(record,)),
            dataset_roots={CHALLENGE_2011_DATASET: tmp_path},
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "_verify_canonical_signal_sidecar",
        lambda *_args, **_kwargs: ({},),
    )
    stored = adapted.signal_mv.copy()
    monkeypatch.setattr(
        pipeline,
        "_load_canonical_signal_shard",
        lambda *_args, **_kwargs: {0: stored},
    )
    monkeypatch.setattr(pipeline, "_adapter_for_record", lambda *_args: adapted)
    monkeypatch.setattr(pipeline, "_verify_raw_source_files_unchanged", lambda *_args: None)
    pipeline._verify_raw_to_canonical_replay(  # noqa: SLF001
        tmp_path,
        inputs=inputs,
        expected_records=(row,),
    )

    stored[0, 0] = np.float32(1.0)
    with pytest.raises(
        pipeline.OODExternalV2IntegrityError,
        match="adapter replay differs",
    ):
        pipeline._verify_raw_to_canonical_replay(  # noqa: SLF001
            tmp_path,
            inputs=inputs,
            expected_records=(row,),
        )

    def fail_adapter(*_args: object) -> CanonicalExternalSignal:
        raise ExternalECGAdapterError("injected adapter contract failure")

    stored[:] = adapted.signal_mv
    monkeypatch.setattr(pipeline, "_adapter_for_record", fail_adapter)
    with pytest.raises(ExternalECGAdapterError, match="injected adapter"):
        pipeline._verify_raw_to_canonical_replay(  # noqa: SLF001
            tmp_path,
            inputs=inputs,
            expected_records=(row,),
        )


def test_full_backbone_replay_matches_both_stored_passes_and_handles_zero_pass(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    row = _private_row()
    signal = _adapted_signal().signal_mv
    inputs = cast(Any, SimpleNamespace(inventory=SimpleNamespace(records=(_inventory_record(),))))
    monkeypatch.setattr(
        pipeline,
        "_verify_canonical_signal_sidecar",
        lambda *_args, **_kwargs: ({},),
    )
    monkeypatch.setattr(
        pipeline,
        "_load_canonical_signal_shard",
        lambda *_args, **_kwargs: {0: signal},
    )
    monkeypatch.setattr(pipeline, "model_state_sha256", lambda _model: "stable")
    expected = np.arange(512, dtype=np.float32).reshape(1, 512)
    monkeypatch.setattr(
        pipeline,
        "extract_embeddings_twice",
        lambda *_args, **_kwargs: SimpleNamespace(
            first=expected.copy(),
            repeated=expected.copy(),
        ),
    )
    normalization = SimpleNamespace(mean=(0.0,) * 12, std=(1.0,) * 12)
    first, repeated = pipeline._replay_quality_pass_embeddings(  # noqa: SLF001
        tmp_path,
        inputs=inputs,
        expected_records=(row,),
        model=cast(Any, object()),
        normalization=cast(Any, normalization),
        runtime=cast(Any, object()),
    )
    assert np.array_equal(first, expected)
    assert np.array_equal(repeated, expected)

    monkeypatch.setattr(
        pipeline,
        "extract_embeddings_twice",
        lambda *_args, **_kwargs: SimpleNamespace(
            first=expected.copy(),
            repeated=expected + np.float32(1.0),
        ),
    )
    with pytest.raises(
        pipeline.OODExternalV2IntegrityError,
        match="full-model CUDA embedding replay",
    ):
        pipeline._replay_quality_pass_embeddings(  # noqa: SLF001
            tmp_path,
            inputs=inputs,
            expected_records=(row,),
            model=cast(Any, object()),
            normalization=cast(Any, normalization),
            runtime=cast(Any, object()),
        )

    monkeypatch.setattr(
        pipeline,
        "extract_embeddings_twice",
        lambda *_args, **_kwargs: pytest.fail("zero-pass replay must not run the model"),
    )
    empty_first, empty_repeated = pipeline._replay_quality_pass_embeddings(  # noqa: SLF001
        tmp_path,
        inputs=inputs,
        expected_records=(replace(row, quality_status="reacquire", route="REACQUIRE"),),
        model=cast(Any, object()),
        normalization=cast(Any, normalization),
        runtime=cast(Any, object()),
    )
    assert empty_first.shape == (0, 512)
    assert np.array_equal(empty_first, empty_repeated)


def test_private_normalization_copy_is_hash_bound(tmp_path: Path) -> None:
    source = (
        PROJECT_ROOT
        / "artifacts"
        / "preprocessing"
        / "ptbxl_v1.0.3_train_folds_1-7_normalization.json"
    )
    copied = tmp_path / "frozen-normalization.json"
    shutil.copyfile(source, copied)
    expected = sha256_file(copied)
    pipeline._load_private_normalization(  # noqa: SLF001
        copied,
        expected_file_sha256=expected,
    )
    copied.write_bytes(copied.read_bytes() + b" ")
    with pytest.raises(
        pipeline.OODExternalV2IntegrityError,
        match="normalization hash differs",
    ):
        pipeline._load_private_normalization(  # noqa: SLF001
            copied,
            expected_file_sha256=expected,
        )


def test_normalization_is_bit_exact_torch_float32_path() -> None:
    normalization = NormalizationStats.load(
        PROJECT_ROOT
        / "artifacts"
        / "preprocessing"
        / "ptbxl_v1.0.3_train_folds_1-7_normalization.json"
    )
    base = np.linspace(-0.5, 0.5, 1_000, dtype=np.float32)
    base[1] = np.nextafter(base[1], np.float32(np.inf), dtype=np.float32)
    signals = np.ascontiguousarray(
        np.stack([np.stack([base + index / 100 for index in range(12)])]),
        dtype=np.float32,
    )
    dataset = pipeline._NormalizedSignalDataset(signals, normalization)  # noqa: SLF001
    observed, _ = dataset[0]
    mean = torch.tensor(normalization.mean, dtype=torch.float32).unsqueeze(1)
    std = torch.tensor(normalization.std, dtype=torch.float32).unsqueeze(1)
    expected = (torch.from_numpy(signals[0]) - mean) / std
    assert torch.equal(observed, expected.contiguous())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="frozen CUDA host unavailable")
def test_head_only_cuda_replay_is_bit_exact_on_frozen_host() -> None:
    runtime = pipeline._configure_frozen_external_v2_cuda()  # noqa: SLF001
    model = build_experiment_model(
        ModelConfig(architecture="resnet1d", preset="matched_capacity")
    )
    prepared = prepare_resnet_for_embedding(model, runtime=runtime)
    embeddings = np.linspace(
        -1.0,
        1.0,
        257 * 512,
        dtype=np.float32,
    ).reshape(257, 512)
    first = pipeline._classify_embeddings(  # noqa: SLF001
        prepared,
        embeddings,
        runtime=runtime,
    )
    repeated = pipeline._classify_embeddings(  # noqa: SLF001
        prepared,
        embeddings,
        runtime=runtime,
    )
    assert first.dtype == np.dtype(np.float64)
    assert np.array_equal(first, repeated)


def test_claim_publication_witness_fires_only_after_directory_fsync(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    monkeypatch.setattr(
        pipeline,
        "_fsync_directory",
        lambda _path: events.append("directory_fsync"),
    )
    pipeline._atomic_write_new(  # noqa: SLF001
        tmp_path / "claim.json",
        b"{}\n",
        publication_witness=lambda: events.append("published"),
    )
    assert events == ["directory_fsync", "published"]


def test_claim_visibility_witness_fires_before_directory_fsync(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    monkeypatch.setattr(
        pipeline,
        "_fsync_directory",
        lambda _path: events.append("directory_fsync"),
    )
    pipeline._atomic_write_new(  # noqa: SLF001
        tmp_path / "claim.json",
        b"{}\n",
        visibility_witness=lambda: events.append("visible"),
        publication_witness=lambda: events.append("durable"),
    )
    assert events == ["visible", "directory_fsync", "durable"]


def test_visible_claim_consumes_one_shot_when_directory_fsync_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []

    def fail_fsync(_path: Path) -> None:
        events.append("directory_fsync_failed")
        raise OSError("injected durability failure")

    monkeypatch.setattr(pipeline, "_fsync_directory", fail_fsync)
    with pytest.raises(
        pipeline.OODExternalV2ExecutionError,
        match="atomic artifact commit failed",
    ):
        pipeline._atomic_write_new(  # noqa: SLF001
            tmp_path / "claim.json",
            b"{}\n",
            visibility_witness=lambda: events.append("visible"),
        )
    assert events == ["visible", "directory_fsync_failed"]
    assert (tmp_path / "claim.json").read_bytes() == b"{}\n"


def test_staging_parent_is_flushed_before_staging_is_returned(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "evidence"
    observed: list[Path] = []
    monkeypatch.setattr(
        pipeline,
        "_fsync_directory",
        lambda path: observed.append(path),
    )
    staging = pipeline._create_durable_staging_directory(  # noqa: SLF001
        output_root,
        expected_parent_identity=pipeline._owned_directory_identity(tmp_path),  # noqa: SLF001
    )
    assert observed == [tmp_path]
    assert staging.parent == tmp_path
    assert staging.is_dir()


def test_staging_parent_flush_failure_removes_empty_staging(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fail_fsync(_path: Path) -> None:
        raise OSError("injected staging parent durability failure")

    monkeypatch.setattr(pipeline, "_fsync_directory", fail_fsync)
    with pytest.raises(
        pipeline.OODExternalV2ExecutionError,
        match="staging directory parent entry is not durable",
    ):
        pipeline._create_durable_staging_directory(  # noqa: SLF001
            tmp_path / "evidence",
            expected_parent_identity=pipeline._owned_directory_identity(  # noqa: SLF001
                tmp_path
            ),
        )
    assert tuple(tmp_path.iterdir()) == ()


def test_output_root_visibility_is_owned_before_parent_flush_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    staging = tmp_path / ".evidence.staging-owned"
    staging.mkdir()
    (staging / "member").write_bytes(b"x")
    marker_bytes = b"marker"
    (staging / ACCESS_MARKER_FILENAME).write_bytes(marker_bytes)
    output = tmp_path / "evidence"
    identity = pipeline._owned_directory_identity(staging)  # noqa: SLF001
    owned = pipeline._OutputRootOwnershipState(identity)  # noqa: SLF001

    def fail_fsync(_path: Path) -> None:
        raise OSError("injected output parent durability failure")

    monkeypatch.setattr(pipeline, "_fsync_directory", fail_fsync)
    with pytest.raises(
        pipeline._ExternalV2OutputCommitError,  # noqa: SLF001
    ) as captured:
        pipeline._commit_staged_directory(  # noqa: SLF001
            staging,
            output,
            visibility_witness=owned.mark_visible,
            expected_directory_identity=identity,
            expected_marker_bytes=marker_bytes,
            expected_parent_identity=pipeline._owned_directory_identity(  # noqa: SLF001
                tmp_path
            ),
        )
    assert captured.value.output_root_committed is True
    assert owned.visible is True
    assert output.is_dir()
    assert not staging.exists()


def _fake_verified_inputs() -> SimpleNamespace:
    return SimpleNamespace(
        child=SimpleNamespace(file_sha256="sha256:" + "1" * 64),
        inventory=SimpleNamespace(inventory_sha256="sha256:" + "2" * 64),
        parent=SimpleNamespace(file_sha256="sha256:" + "3" * 64),
        project_root=PROJECT_ROOT,
    )


def test_postclaim_failure_never_writes_into_foreign_output_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    staging = tmp_path / ".evidence.staging-owned"
    staging.mkdir()
    (staging / ACCESS_MARKER_FILENAME).write_bytes(b"marker")
    identity = pipeline._owned_directory_identity(staging)  # noqa: SLF001
    output = tmp_path / "evidence"
    output.mkdir()
    (output / "foreign").write_bytes(b"foreign")
    monkeypatch.setattr(
        pipeline,
        "_verify_runtime_scratch_empty",
        lambda _root: (_ for _ in ()).throw(AssertionError("must not run")),
    )
    pipeline._retain_postclaim_failure(  # noqa: SLF001
        staging=staging,
        output_root=output,
        inputs=cast(Any, _fake_verified_inputs()),
        code_revision="4" * 40,
        error=RuntimeError("injected"),
        output_root_owned=False,
        terminal_manifest_visible=False,
        external_claim_file_sha256="sha256:" + "5" * 64,
        owner_nonce="6" * 64,
        expected_directory_identity=identity,
        expected_marker_bytes=b"marker",
        expected_parent_identity=pipeline._owned_directory_identity(tmp_path),  # noqa: SLF001
    )
    assert (output / "foreign").read_bytes() == b"foreign"
    assert not (output / FAILURE_RECEIPT_FILENAME).exists()
    assert (staging / FAILURE_RECEIPT_FILENAME).is_file()


def test_terminal_link_visible_then_flush_failure_is_ambiguous_not_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / "evidence"
    output.mkdir()
    marker_bytes = b"marker"
    (output / ACCESS_MARKER_FILENAME).write_bytes(marker_bytes)
    identity = pipeline._owned_directory_identity(output)  # noqa: SLF001

    def fail_fsync(_path: Path) -> None:
        raise OSError("injected terminal directory durability failure")

    monkeypatch.setattr(pipeline, "_fsync_directory", fail_fsync)
    with pytest.raises(
        pipeline.OODExternalV2ExecutionError,
        match="terminal success-manifest commit failed",
    ):
        terminal = pipeline._TerminalManifestState()  # noqa: SLF001
        pipeline._atomic_write_terminal_success(  # noqa: SLF001
            output,
            b"{}\n",
            visibility_witness=terminal.mark_visible,
            ownership_verifier=lambda: pipeline._verify_owned_evidence_directory(  # noqa: SLF001
                output,
                expected_identity=identity,
                expected_marker_bytes=marker_bytes,
            ),
        )
    assert (output / SUCCESS_MANIFEST_FILENAME).is_file()
    assert terminal.visible is True

    monkeypatch.undo()
    monkeypatch.setattr(
        pipeline,
        "_verify_runtime_scratch_empty",
        lambda _root: (_ for _ in ()).throw(AssertionError("must not run")),
    )
    pipeline._retain_postclaim_failure(  # noqa: SLF001
        staging=tmp_path / ".unused-staging",
        output_root=output,
        inputs=cast(Any, _fake_verified_inputs()),
        code_revision="4" * 40,
        error=RuntimeError("post-publication reread failed"),
        output_root_owned=True,
        terminal_manifest_visible=True,
        external_claim_file_sha256="sha256:" + "5" * 64,
        owner_nonce="6" * 64,
        expected_directory_identity=identity,
        expected_marker_bytes=marker_bytes,
        expected_parent_identity=pipeline._owned_directory_identity(tmp_path),  # noqa: SLF001
    )
    receipt = json.loads(
        (output / FAILURE_RECEIPT_FILENAME).read_text(encoding="ascii")
    )
    assert receipt["terminal_state"] == "AMBIGUOUS_TERMINAL_COMMIT"
    assert receipt["external_claim_file_sha256"] == "sha256:" + "5" * 64
    assert receipt["owner_nonce"] == "6" * 64
    with pytest.raises(ValueError):
        verify_external_v2_bundle(output)


def test_public_five_state_route_counts_are_rederived_from_private_rows() -> None:
    rows = tuple(
        SimpleNamespace(route="UNSUPPORTED_INPUT") for _ in range(13_328)
    )
    good = SimpleNamespace(
        final_route_counts=AggregateRouteCounts(
            INVALID_INPUT=0,
            REACQUIRE=0,
            UNSUPPORTED_INPUT=13_328,
            ABSTAIN=0,
            PREDICTION_ALLOWED=0,
            total_records=13_328,
        )
    )
    pipeline._verify_result_route_counts(  # noqa: SLF001
        cast(Any, good),
        cast(Any, rows),
    )
    tampered = SimpleNamespace(
        final_route_counts=AggregateRouteCounts(
            INVALID_INPUT=0,
            REACQUIRE=0,
            UNSUPPORTED_INPUT=13_327,
            ABSTAIN=1,
            PREDICTION_ALLOWED=0,
            total_records=13_328,
        )
    )
    with pytest.raises(
        pipeline.OODExternalV2IntegrityError,
        match="public five-state route counts differ",
    ):
        pipeline._verify_result_route_counts(  # noqa: SLF001
            cast(Any, tampered),
            cast(Any, rows),
        )


def test_live_git_remote_state_requires_exact_main_and_backup_tag_lines(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    revision = "a" * 40

    def exact_git(_root: Path, *arguments: str, **_kwargs: object) -> Any:
        if arguments == ("remote",):
            stdout = "origin\n"
        elif arguments[:3] == (
            "remote",
            "get-url",
            "--all",
        ) or arguments[:4] == ("remote", "get-url", "--push", "--all"):
            stdout = pipeline.EXPECTED_GIT_REMOTE_URL + "\n"
        elif arguments[:2] == ("rev-parse", "--verify"):
            stdout = revision + "\n"
        elif arguments[:2] == ("cat-file", "-t"):
            stdout = "commit\n"
        elif arguments[:2] == ("merge-base", "--is-ancestor"):
            stdout = ""
        else:
            raise AssertionError(arguments)
        return SimpleNamespace(stdout=stdout, stderr="", returncode=0)

    exact_remote = (
        "ref: refs/heads/main\tHEAD\n"
        f"{revision}\tHEAD\n"
        f"{revision}\trefs/heads/main\n"
        f"{pipeline.EXPECTED_GIT_REMOTE_BACKUP_TAG_REVISION}\t"
        f"{pipeline.EXPECTED_GIT_REMOTE_BACKUP_TAG_REF}\n"
    )
    monkeypatch.setattr(pipeline, "_run_git", exact_git)
    monkeypatch.setattr(
        pipeline,
        "_verify_private_remote_anonymous_denial",
        lambda _root: None,
    )
    monkeypatch.setattr(
        pipeline,
        "_run_exact_private_live_remote",
        lambda _root: exact_remote,
    )
    pipeline._verify_git_remote_state(  # noqa: SLF001
        tmp_path,
        expected_revision=revision,
    )

    monkeypatch.setattr(
        pipeline,
        "_run_exact_private_live_remote",
        lambda _root: (
            "ref: refs/heads/main\tHEAD\n"
            f"{'b' * 40}\tHEAD\n"
            f"{'b' * 40}\trefs/heads/main\n"
            f"{pipeline.EXPECTED_GIT_REMOTE_BACKUP_TAG_REVISION}\t"
            f"{pipeline.EXPECTED_GIT_REMOTE_BACKUP_TAG_REF}\n"
        ),
    )
    with pytest.raises(
        pipeline.OODExternalV2IntegrityError,
        match="not the exact pushed frozen revision",
    ):
        pipeline._verify_git_remote_state(  # noqa: SLF001
            tmp_path,
            expected_revision=revision,
        )

    def non_ancestor_tag(_root: Path, *arguments: str, **kwargs: object) -> Any:
        result = exact_git(_root, *arguments, **kwargs)
        if arguments[:2] == ("merge-base", "--is-ancestor"):
            result.returncode = 1
        return result

    monkeypatch.setattr(pipeline, "_run_git", non_ancestor_tag)
    monkeypatch.setattr(
        pipeline,
        "_run_exact_private_live_remote",
        lambda _root: exact_remote,
    )
    with pytest.raises(
        pipeline.OODExternalV2IntegrityError,
        match="not the exact pushed frozen revision",
    ):
        pipeline._verify_git_remote_state(  # noqa: SLF001
            tmp_path,
            expected_revision=revision,
        )


@pytest.mark.parametrize(
    "malformed_suffix",
    (
        "\n",
        f"{'3' * 40}\trefs/heads/extra\n",
        "github_pat_synthetic_secret_must_not_surface\n",
    ),
)
def test_live_git_remote_state_rejects_malformed_authenticated_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    malformed_suffix: str,
) -> None:
    revision = "a" * 40
    exact_remote = (
        "ref: refs/heads/main\tHEAD\n"
        f"{revision}\tHEAD\n"
        f"{revision}\trefs/heads/main\n"
        f"{pipeline.EXPECTED_GIT_REMOTE_BACKUP_TAG_REVISION}\t"
        f"{pipeline.EXPECTED_GIT_REMOTE_BACKUP_TAG_REF}\n"
    )

    def local_git(_root: Path, *arguments: str, **_kwargs: object) -> Any:
        if arguments == ("remote",):
            stdout = "origin\n"
        elif arguments[:3] == ("remote", "get-url", "--all") or arguments[:4] == (
            "remote",
            "get-url",
            "--push",
            "--all",
        ):
            stdout = pipeline.EXPECTED_GIT_REMOTE_URL + "\n"
        elif arguments[:2] == ("rev-parse", "--verify"):
            stdout = revision + "\n"
        elif arguments[:2] == ("cat-file", "-t"):
            stdout = "commit\n"
        elif arguments[:2] == ("merge-base", "--is-ancestor"):
            stdout = ""
        else:
            raise AssertionError(arguments)
        return SimpleNamespace(stdout=stdout, stderr="", returncode=0)

    monkeypatch.setattr(pipeline, "_run_git", local_git)
    monkeypatch.setattr(
        pipeline,
        "_verify_private_remote_anonymous_denial",
        lambda _root: None,
    )
    monkeypatch.setattr(
        pipeline,
        "_run_exact_private_live_remote",
        lambda _root: exact_remote + malformed_suffix,
    )
    with pytest.raises(pipeline.OODExternalV2IntegrityError) as captured:
        pipeline._verify_git_remote_state(  # noqa: SLF001
            tmp_path,
            expected_revision=revision,
        )

    assert "github_pat_synthetic" not in str(captured.value)
    assert "exact pushed frozen revision" in str(captured.value)


def test_private_remote_security_contract_has_literal_exact_values() -> None:
    assert pipeline.PRIVATE_REMOTE_ANONYMOUS_GIT_CONFIG == (
        "credential.helper=",
        "credential.interactive=false",
        "credential.guiPrompt=false",
        "credential.allowUnsafeRemotes=false",
        "http.followRedirects=false",
        "http.sslVerify=true",
    )
    assert pipeline.PRIVATE_REMOTE_GIT_CONFIG == (
        "credential.helper=",
        "credential.helper=manager",
        "credential.interactive=false",
        "credential.guiPrompt=false",
        "credential.allowUnsafeRemotes=false",
        "credential.credentialStore=wincredman",
        "credential.namespace=git",
        "credential.useHttpPath=false",
        "credential.username=Ahmad986Ferdaws",
        "credential.https://github.com.provider=github",
        "credential.trace=false",
        "credential.traceSecrets=false",
        "credential.traceMsAuth=false",
        "credential.debug=false",
        "http.followRedirects=false",
        "http.sslVerify=true",
    )
    assert tuple(pipeline.PRIVATE_REMOTE_GCM_ENVIRONMENT.items()) == (
        ("GCM_ALLOW_UNSAFE_REMOTES", "0"),
        ("GCM_CREDENTIAL_STORE", "wincredman"),
        ("GCM_DEBUG", "0"),
        ("GCM_GUI_PROMPT", "0"),
        ("GCM_INTERACTIVE", "0"),
        ("GCM_NAMESPACE", "git"),
        ("GCM_PROVIDER", "github"),
        ("GCM_TRACE", "0"),
        ("GCM_TRACE_MSAUTH", "0"),
        ("GCM_TRACE_SECRETS", "0"),
    )
    assert pipeline.EXPECTED_GIT_CREDENTIAL_MANAGER_VERSION_STDOUT == (
        b"2.7.3+5fa7116896c82164996a609accd1c5ad90fe730a\r\n"
    )
    assert pipeline.EXPECTED_PRIVATE_REMOTE_ANONYMOUS_STDERR == (
        b"fatal: unable to get password from user\n"
    )


def _process_ids_with_command_marker(marker: str) -> set[int]:
    observed: set[int] = set()
    for process in psutil.process_iter(("pid", "cmdline")):
        try:
            command_line = process.cmdline()
        except (OSError, psutil.Error):
            continue
        if marker in command_line:
            observed.add(process.pid)
    return observed


def _observe_process_marker(
    marker: str,
    stop: threading.Event,
    observed: set[int],
) -> None:
    while not stop.wait(0.01):
        observed.update(_process_ids_with_command_marker(marker))


def _tree_probe_command(marker: str, *, overflow_stream: int | None) -> list[str]:
    overflow = ""
    if overflow_stream is not None:
        overflow = f"os.write({overflow_stream},b'x'*65536);"
    child = (
        "import os,subprocess,sys,time;"
        "subprocess.Popen([sys.executable,'-c',"
        "'import time;time.sleep(60)',sys.argv[1]]);"
        "print(sys.argv[1],flush=True);"
        "time.sleep(0.3);"
        f"{overflow}"
        "time.sleep(60)"
    )
    return [sys.executable, "-c", child, marker]


def test_bounded_windows_process_contract_has_exact_atomic_job_values() -> None:
    assert pipeline.WINDOWS_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE == 0x00002000
    assert pipeline.WINDOWS_PROCESS_ATTRIBUTE_HANDLE_LIST == 0x00020002
    assert pipeline.WINDOWS_PROCESS_ATTRIBUTE_JOB_LIST == 0x0002000D
    assert pipeline.WINDOWS_EXTENDED_STARTUPINFO_PRESENT == 0x00080000
    assert pipeline.WINDOWS_CREATE_UNICODE_ENVIRONMENT == 0x00000400
    assert pipeline.WINDOWS_CREATE_NO_WINDOW == 0x08000000
    assert pipeline.PRIVATE_REMOTE_STDOUT_LIMIT_BYTES == 4_096
    assert pipeline.PRIVATE_REMOTE_STDERR_LIMIT_BYTES == 4_096
    assert pipeline.GCM_VERSION_STDOUT_LIMIT_BYTES == 256
    assert pipeline.GCM_VERSION_STDERR_LIMIT_BYTES == 256
    assert pipeline.PRIVATE_REMOTE_TIMEOUT_SECONDS == 60.0
    assert pipeline.GCM_VERSION_TIMEOUT_SECONDS == 30.0
    assert pipeline.WINDOWS_PRIVATE_PROCESS_CLEANUP_TIMEOUT_SECONDS == 10.0


@pytest.mark.skipif(os.name != "nt", reason="exact frozen Windows process contract")
def test_bounded_windows_process_returns_exact_streams_status_env_cwd_and_devnull(
    tmp_path: Path,
) -> None:
    environment = {
        "BOUND_PROCESS_EXACT": "synthetic-value",
        "PATH": os.fspath(Path(sys.executable).parent),
        "SYSTEMROOT": os.environ["SYSTEMROOT"],
    }
    child = (
        "import os,sys;"
        "stdin_bytes=sys.stdin.buffer.read();"
        "sys.stdout.write(os.getcwd()+'|'+os.environ['BOUND_PROCESS_EXACT']+'|'"
        "+str(len(stdin_bytes)));"
        "sys.stderr.write('synthetic-stderr');"
        "raise SystemExit(7)"
    )

    completed = pipeline._run_bounded_windows_process(  # noqa: SLF001
        [sys.executable, "-c", child],
        cwd=tmp_path,
        environment=environment,
        timeout_seconds=5,
        stdout_limit_bytes=1_024,
        stderr_limit_bytes=1_024,
        failure_message="synthetic bounded process failed",
    )

    assert completed.returncode == 7
    assert completed.stdout.decode() == f"{tmp_path}|synthetic-value|0"
    assert completed.stderr == b"synthetic-stderr"
    assert list(tmp_path.iterdir()) == []


@pytest.mark.skipif(os.name != "nt", reason="exact frozen Windows process contract")
def test_bounded_windows_process_timeout_kills_child_and_grandchild(
    tmp_path: Path,
) -> None:
    marker = f"ecg-bounded-timeout-{os.getpid()}-{time.time_ns()}"
    observed_pids: set[int] = set()
    stop_observer = threading.Event()
    observer = threading.Thread(
        target=_observe_process_marker,
        args=(marker, stop_observer, observed_pids),
        daemon=True,
    )
    observer.start()
    started = time.monotonic()
    try:
        with pytest.raises(pipeline.OODExternalV2IntegrityError) as captured:
            pipeline._run_bounded_windows_process(  # noqa: SLF001
                _tree_probe_command(marker, overflow_stream=None),
                cwd=tmp_path,
                environment=dict(os.environ),
                timeout_seconds=1.5,
                stdout_limit_bytes=4_096,
                stderr_limit_bytes=4_096,
                failure_message="synthetic bounded process failed",
            )
    finally:
        stop_observer.set()
        observer.join(2)

    assert str(captured.value) == "synthetic bounded process failed"
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    assert len(observed_pids) >= 2
    assert _process_ids_with_command_marker(marker) == set()
    assert time.monotonic() - started < 12
    assert not observer.is_alive()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.skipif(os.name != "nt", reason="exact frozen Windows process contract")
@pytest.mark.parametrize("overflow_stream", (1, 2))
def test_bounded_windows_process_overflow_kills_child_and_grandchild(
    tmp_path: Path,
    overflow_stream: int,
) -> None:
    marker = (
        f"ecg-bounded-overflow-{overflow_stream}-{os.getpid()}-{time.time_ns()}"
    )
    observed_pids: set[int] = set()
    stop_observer = threading.Event()
    observer = threading.Thread(
        target=_observe_process_marker,
        args=(marker, stop_observer, observed_pids),
        daemon=True,
    )
    observer.start()
    try:
        with pytest.raises(pipeline.OODExternalV2IntegrityError) as captured:
            pipeline._run_bounded_windows_process(  # noqa: SLF001
                _tree_probe_command(marker, overflow_stream=overflow_stream),
                cwd=tmp_path,
                environment=dict(os.environ),
                timeout_seconds=5,
                stdout_limit_bytes=1_024,
                stderr_limit_bytes=1_024,
                failure_message="synthetic bounded process failed",
            )
    finally:
        stop_observer.set()
        observer.join(2)

    assert str(captured.value) == "synthetic bounded process failed"
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    assert len(observed_pids) >= 2
    assert _process_ids_with_command_marker(marker) == set()
    assert not observer.is_alive()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.skipif(os.name != "nt", reason="exact frozen Windows process contract")
def test_bounded_windows_process_interrupt_wipes_buffers_and_kills_tree(
    tmp_path: Path,
) -> None:
    marker = f"SYNTHETIC_INTERRUPT_CAPTURE_SECRET_{os.getpid()}_{time.time_ns()}"
    child = (
        "import subprocess,sys,time;"
        "subprocess.Popen([sys.executable,'-c',"
        "'import time;time.sleep(60)',sys.argv[1]]);"
        "print(sys.argv[1],flush=True);"
        "time.sleep(60)"
    )
    command = [sys.executable, "-c", child, marker]
    observed_pids: set[int] = set()
    stop_observer = threading.Event()
    observer = threading.Thread(
        target=_observe_process_marker,
        args=(marker, stop_observer, observed_pids),
        daemon=True,
    )
    observer.start()
    interrupt_requested = threading.Event()
    interrupt_abort = threading.Event()

    def request_interrupt_after_tree_observed() -> None:
        deadline = time.monotonic() + 3
        while len(observed_pids) < 2 and time.monotonic() < deadline:
            if interrupt_abort.wait(0.01):
                return
        interrupt_requested.set()
        _thread.interrupt_main()

    interrupt = threading.Thread(
        target=request_interrupt_after_tree_observed,
        daemon=True,
    )
    interrupt.start()
    try:
        with pytest.raises(pipeline.OODExternalV2IntegrityError) as captured:
            pipeline._run_bounded_windows_process(  # noqa: SLF001
                command,
                cwd=tmp_path,
                environment=dict(os.environ),
                timeout_seconds=4,
                stdout_limit_bytes=4_096,
                stderr_limit_bytes=4_096,
                failure_message="synthetic bounded process failed",
            )
    finally:
        interrupt_abort.set()
        interrupt.join(4)
        stop_observer.set()
        observer.join(2)

    assert str(captured.value) == "synthetic bounded process failed"
    assert marker not in repr(captured.value)
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    traceback = captured.value.__traceback__
    while traceback is not None:
        for value in traceback.tb_frame.f_locals.values():
            if isinstance(value, bytearray):
                assert marker.encode() not in bytes(value)
        traceback = traceback.tb_next
    assert len(observed_pids) >= 2
    assert interrupt_requested.is_set()
    assert _process_ids_with_command_marker(marker) == set()
    assert not interrupt.is_alive()
    assert not observer.is_alive()
    assert list(tmp_path.iterdir()) == []


def test_private_live_remote_environment_overrides_gcm_and_discards_secrets(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    executable = tmp_path / "mingw64" / "bin" / "git.exe"
    inherited = (
        "GH_TOKEN",
        "GITHUB_TOKEN",
        "GIT_ASKPASS",
        "SSH_ASKPASS",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "GCM_AUTHORITY",
        "GCM_CREDENTIAL_STORE",
        "GCM_DEBUG",
        "GCM_INTERACTIVE",
        "GCM_PROVIDER",
        "GCM_TRACE",
        "GCM_TRACE_SECRETS",
    )
    for name in inherited:
        monkeypatch.setenv(name, "synthetic-secret-or-control")

    base = pipeline._sanitized_git_environment(executable)  # noqa: SLF001
    observed = pipeline._private_live_remote_environment(executable)  # noqa: SLF001

    expected = dict(base)
    expected.update(pipeline.PRIVATE_REMOTE_GCM_ENVIRONMENT)
    assert observed == expected
    assert not {
        "GH_TOKEN",
        "GITHUB_TOKEN",
        "GIT_ASKPASS",
        "SSH_ASKPASS",
    }.intersection(observed)
    assert observed["GCM_CREDENTIAL_STORE"] == "wincredman"
    assert observed["GCM_INTERACTIVE"] == "0"
    assert observed["GCM_TRACE_SECRETS"] == "0"


def test_private_remote_anonymous_denial_is_exact_and_credentialless(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    executable = tmp_path / "mingw64" / "bin" / "git.exe"
    base_environment = {"PATH": os.fspath(executable.parent)}
    observed: dict[str, object] = {}
    monkeypatch.setattr(
        pipeline,
        "_git_executable_paths",
        lambda: (tmp_path / "cmd" / "git.exe", executable, tmp_path),
    )
    monkeypatch.setattr(
        pipeline,
        "_verify_git_runtime_tree_before_provenance",
        lambda: None,
    )
    monkeypatch.setattr(
        pipeline,
        "_verify_git_repository_controls",
        lambda root: observed.setdefault("controls_root", root),
    )
    monkeypatch.setattr(
        pipeline,
        "_sanitized_git_environment",
        lambda value: dict(base_environment) if value == executable else {},
    )

    def denied_run(
        command: list[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[bytes]:
        observed.update(command=command, **kwargs)
        return subprocess.CompletedProcess(
            command,
            128,
            b"",
            pipeline.EXPECTED_PRIVATE_REMOTE_ANONYMOUS_STDERR,
        )

    monkeypatch.setattr(pipeline, "_run_bounded_windows_process", denied_run)
    pipeline._verify_private_remote_anonymous_denial(tmp_path)  # noqa: SLF001

    anonymous_arguments = [
        item
        for config_entry in pipeline.PRIVATE_REMOTE_ANONYMOUS_GIT_CONFIG
        for item in ("-c", config_entry)
    ]
    assert observed["command"] == pipeline._bound_git_command(  # noqa: SLF001
        tmp_path,
        executable,
        *anonymous_arguments,
        "ls-remote",
        "--symref",
        pipeline.EXPECTED_GIT_REMOTE_URL,
    )
    assert "credential.helper=manager" not in cast(list[str], observed["command"])
    assert observed["controls_root"] == tmp_path
    assert observed["cwd"] == executable.parent
    assert observed["environment"] == base_environment
    assert observed["timeout_seconds"] == 60
    assert observed["stdout_limit_bytes"] == pipeline.PRIVATE_REMOTE_STDOUT_LIMIT_BYTES
    assert observed["stderr_limit_bytes"] == pipeline.PRIVATE_REMOTE_STDERR_LIMIT_BYTES
    assert observed["failure_message"] == (
        "private Git remote anonymous-access probe failed"
    )


@pytest.mark.parametrize(
    ("returncode", "stdout", "stderr"),
    (
        (0, b"public refs\n", b""),
        (128, b"partial refs\n", b"denied\n"),
        (128, b"", b""),
        (128, b"", b"network failure\n"),
        (1, b"", b"denied\n"),
    ),
)
def test_private_remote_anonymous_denial_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    returncode: int,
    stdout: bytes,
    stderr: bytes,
) -> None:
    executable = tmp_path / "mingw64" / "bin" / "git.exe"
    monkeypatch.setattr(
        pipeline,
        "_git_executable_paths",
        lambda: (tmp_path / "cmd" / "git.exe", executable, tmp_path),
    )
    monkeypatch.setattr(
        pipeline,
        "_verify_git_runtime_tree_before_provenance",
        lambda: None,
    )
    monkeypatch.setattr(pipeline, "_verify_git_repository_controls", lambda _root: None)
    monkeypatch.setattr(
        pipeline,
        "_sanitized_git_environment",
        lambda _value: {"PATH": os.fspath(executable.parent)},
    )
    monkeypatch.setattr(
        pipeline,
        "_run_bounded_windows_process",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command,
            returncode,
            stdout,
            stderr,
        ),
    )

    with pytest.raises(
        pipeline.OODExternalV2IntegrityError,
        match="anonymous-access denial was not proven",
    ):
        pipeline._verify_private_remote_anonymous_denial(tmp_path)  # noqa: SLF001


def test_private_remote_command_rejects_userinfo_and_ambient_secret_carriers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    executable = tmp_path / "mingw64" / "bin" / "git.exe"
    monkeypatch.setattr(
        pipeline,
        "EXPECTED_GIT_REMOTE_URL",
        "https://token@github.com/Ahmad986Ferdaws/ecg-trust-lab.git",
    )
    with pytest.raises(
        pipeline.OODExternalV2IntegrityError,
        match="URL boundary differs",
    ):
        pipeline._private_remote_command(  # noqa: SLF001
            tmp_path,
            executable,
            authenticated=True,
        )

    monkeypatch.setattr(
        pipeline,
        "EXPECTED_GIT_REMOTE_URL",
        "https://github.com/Ahmad986Ferdaws/ecg-trust-lab.git",
    )
    monkeypatch.setattr(
        pipeline,
        "_sanitized_git_environment",
        lambda _value: {
            "PATH": os.fspath(executable.parent),
            "GITHUB_TOKEN": "github_pat_synthetic",
        },
    )
    with pytest.raises(pipeline.OODExternalV2IntegrityError) as captured:
        pipeline._private_remote_command(  # noqa: SLF001
            tmp_path,
            executable,
            authenticated=False,
        )
    assert "github_pat_synthetic" not in str(captured.value)


def test_git_credential_manager_rejects_missing_or_wrong_bytes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    executable = tmp_path / "mingw64" / "bin" / "git.exe"
    executable.parent.mkdir(parents=True)
    monkeypatch.setattr(
        pipeline,
        "_assert_direct_ancestry",
        lambda path, *, context: path,
    )

    with pytest.raises(
        pipeline.OODExternalV2IntegrityError,
        match="credential manager is unavailable",
    ):
        pipeline._git_credential_manager_path(executable)  # noqa: SLF001

    helper = executable.parent / pipeline.EXPECTED_GIT_CREDENTIAL_MANAGER_NAME
    helper.write_bytes(b"wrong helper bytes")
    with pytest.raises(
        pipeline.OODExternalV2IntegrityError,
        match="credential manager differs",
    ):
        pipeline._git_credential_manager_path(executable)  # noqa: SLF001


def test_git_credential_manager_version_probe_is_exact_and_noninteractive(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    executable = tmp_path / "mingw64" / "bin" / "git.exe"
    helper = executable.parent / pipeline.EXPECTED_GIT_CREDENTIAL_MANAGER_NAME
    observed: dict[str, object] = {}
    environment = {"GCM_INTERACTIVE": "0", "PATH": os.fspath(executable.parent)}

    def helper_path(value: Path) -> Path:
        assert value == executable
        return helper

    monkeypatch.setattr(pipeline, "_git_credential_manager_path", helper_path)
    monkeypatch.setattr(
        pipeline,
        "_private_live_remote_environment",
        lambda value: environment if value == executable else {},
    )

    def fake_run(
        command: list[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[bytes]:
        observed.update(command=command, **kwargs)
        return subprocess.CompletedProcess(
            command,
            0,
            pipeline.EXPECTED_GIT_CREDENTIAL_MANAGER_VERSION_STDOUT,
            b"",
        )

    monkeypatch.setattr(pipeline, "_run_bounded_windows_process", fake_run)
    pipeline._verify_git_credential_manager(executable)  # noqa: SLF001

    assert observed == {
        "command": [os.fspath(helper), "--version"],
        "cwd": helper.parent,
        "environment": environment,
        "failure_message": "frozen Git credential manager probe failed",
        "stderr_limit_bytes": pipeline.GCM_VERSION_STDERR_LIMIT_BYTES,
        "stdout_limit_bytes": pipeline.GCM_VERSION_STDOUT_LIMIT_BYTES,
        "timeout_seconds": 30,
    }

    for invalid_stdout in (
        b" " + pipeline.EXPECTED_GIT_CREDENTIAL_MANAGER_VERSION_STDOUT,
        pipeline.EXPECTED_GIT_CREDENTIAL_MANAGER_VERSION_STDOUT + b"\r\n",
        pipeline.EXPECTED_GIT_CREDENTIAL_MANAGER_VERSION_STDOUT.replace(
            b"\r\n", b"\n"
        ),
        b"2.7.3-forged\r\n",
    ):
        monkeypatch.setattr(
            pipeline,
            "_run_bounded_windows_process",
            lambda *_args, _stdout=invalid_stdout, **_kwargs: (
                subprocess.CompletedProcess([], 0, _stdout, b"")
            ),
        )
        with pytest.raises(
            pipeline.OODExternalV2IntegrityError,
            match="credential manager differs",
        ):
            pipeline._verify_git_credential_manager(executable)  # noqa: SLF001


def test_private_live_remote_uses_exact_argv_env_and_devnull(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    launcher = tmp_path / "cmd" / "git.exe"
    executable = tmp_path / "mingw64" / "bin" / "git.exe"
    install_root = tmp_path
    base_environment = {"PATH": os.fspath(executable.parent)}
    environment = dict(base_environment)
    environment.update(pipeline.PRIVATE_REMOTE_GCM_ENVIRONMENT)
    observed: dict[str, object] = {}
    monkeypatch.setattr(
        pipeline,
        "_git_executable_paths",
        lambda: (launcher, executable, install_root),
    )
    monkeypatch.setattr(
        pipeline,
        "_verify_git_runtime_tree_before_provenance",
        lambda: None,
    )
    monkeypatch.setattr(
        pipeline,
        "_verify_git_repository_controls",
        lambda root: observed.setdefault("controls_root", root),
    )
    monkeypatch.setattr(
        pipeline,
        "_verify_git_credential_manager",
        lambda value: observed.setdefault("gcm_executable", value),
    )
    monkeypatch.setattr(
        pipeline,
        "_sanitized_git_environment",
        lambda value: dict(base_environment) if value == executable else {},
    )

    def fake_run(
        command: list[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[bytes]:
        observed.update(command=command, **kwargs)
        return subprocess.CompletedProcess(command, 0, b"synthetic refs\n", b"")

    monkeypatch.setattr(pipeline, "_run_bounded_windows_process", fake_run)
    assert pipeline._run_exact_private_live_remote(tmp_path) == "synthetic refs\n"

    credential_arguments = [
        item
        for config_entry in pipeline.PRIVATE_REMOTE_GIT_CONFIG
        for item in ("-c", config_entry)
    ]
    assert observed["command"] == [
        os.fspath(executable),
        "--no-pager",
        "--no-replace-objects",
        f"--git-dir={tmp_path / '.git'}",
        f"--work-tree={tmp_path}",
        "-c",
        "core.hooksPath=NUL",
        "-c",
        "protocol.file.allow=never",
        "-c",
        "core.fsmonitor=false",
        "-c",
        "core.untrackedCache=false",
        "-c",
        "core.ignoreStat=false",
        "-c",
        "core.preloadIndex=false",
        "-c",
        "core.checkStat=default",
        "-c",
        "core.trustctime=true",
        "-c",
        "core.sparseCheckout=false",
        "-c",
        "core.sparseCheckoutCone=false",
        "-c",
        "extensions.worktreeConfig=false",
        *credential_arguments,
        "ls-remote",
        "--symref",
        pipeline.EXPECTED_GIT_REMOTE_URL,
    ]
    assert observed["controls_root"] == tmp_path
    assert observed["gcm_executable"] == executable
    assert observed["cwd"] == executable.parent
    assert observed["environment"] == environment
    assert observed["timeout_seconds"] == 60
    assert observed["stdout_limit_bytes"] == pipeline.PRIVATE_REMOTE_STDOUT_LIMIT_BYTES
    assert observed["stderr_limit_bytes"] == pipeline.PRIVATE_REMOTE_STDERR_LIMIT_BYTES
    assert observed["failure_message"] == (
        "private Git remote preflight could not be executed"
    )


def test_private_live_remote_failure_is_single_attempt_and_never_surfaces_secret(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    executable = tmp_path / "mingw64" / "bin" / "git.exe"
    calls: list[list[str]] = []
    monkeypatch.setattr(
        pipeline,
        "_git_executable_paths",
        lambda: (tmp_path / "cmd" / "git.exe", executable, tmp_path),
    )
    monkeypatch.setattr(
        pipeline,
        "_verify_git_runtime_tree_before_provenance",
        lambda: None,
    )
    monkeypatch.setattr(pipeline, "_verify_git_repository_controls", lambda _root: None)
    monkeypatch.setattr(pipeline, "_verify_git_credential_manager", lambda _value: None)
    base_environment = {"PATH": os.fspath(executable.parent)}
    monkeypatch.setattr(
        pipeline,
        "_sanitized_git_environment",
        lambda _value: dict(base_environment),
    )
    secret = "github_pat_synthetic_secret_must_not_surface"

    def failed_run(
        command: list[str],
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[bytes]:
        calls.append(command)
        return subprocess.CompletedProcess(
            command,
            128,
            f"username=owner password={secret}\n".encode(),
            f"credential failure {secret}\n".encode(),
        )

    monkeypatch.setattr(pipeline, "_run_bounded_windows_process", failed_run)
    with pytest.raises(pipeline.OODExternalV2IntegrityError) as captured:
        pipeline._run_exact_private_live_remote(tmp_path)

    assert len(calls) == 1
    assert secret not in str(captured.value)
    assert secret not in repr(captured.value)
    assert captured.value.__cause__ is None
    assert pipeline.EXPECTED_GIT_REMOTE_URL in calls[0]


def test_private_live_remote_timeout_discards_partial_secret_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    executable = tmp_path / "mingw64" / "bin" / "git.exe"
    command = [os.fspath(executable), "ls-remote", pipeline.EXPECTED_GIT_REMOTE_URL]
    monkeypatch.setattr(
        pipeline,
        "_git_executable_paths",
        lambda: (tmp_path / "cmd" / "git.exe", executable, tmp_path),
    )
    monkeypatch.setattr(
        pipeline,
        "_verify_git_runtime_tree_before_provenance",
        lambda: None,
    )
    monkeypatch.setattr(pipeline, "_verify_git_repository_controls", lambda _root: None)
    monkeypatch.setattr(pipeline, "_verify_git_credential_manager", lambda _value: None)
    monkeypatch.setattr(
        pipeline,
        "_private_remote_command",
        lambda _root, _executable, *, authenticated: (command, {})
        if authenticated
        else ([], {}),
    )
    secret = b"github_pat_synthetic_timeout_secret"

    def timeout_run(*_args: object, **_kwargs: object) -> None:
        raise pipeline.OODExternalV2IntegrityError(
            "private Git remote preflight could not be executed"
        ) from None

    monkeypatch.setattr(pipeline, "_run_bounded_windows_process", timeout_run)
    with pytest.raises(pipeline.OODExternalV2IntegrityError) as captured:
        pipeline._run_exact_private_live_remote(tmp_path)

    assert secret.decode() not in str(captured.value)
    assert secret.decode() not in repr(captured.value)
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None


def test_private_live_remote_invalid_utf8_has_no_retained_exception_context(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    executable = tmp_path / "mingw64" / "bin" / "git.exe"
    command = [os.fspath(executable), "ls-remote", pipeline.EXPECTED_GIT_REMOTE_URL]
    monkeypatch.setattr(
        pipeline,
        "_git_executable_paths",
        lambda: (tmp_path / "cmd" / "git.exe", executable, tmp_path),
    )
    monkeypatch.setattr(
        pipeline,
        "_verify_git_runtime_tree_before_provenance",
        lambda: None,
    )
    monkeypatch.setattr(pipeline, "_verify_git_repository_controls", lambda _root: None)
    monkeypatch.setattr(pipeline, "_verify_git_credential_manager", lambda _value: None)
    monkeypatch.setattr(
        pipeline,
        "_private_remote_command",
        lambda _root, _executable, *, authenticated: (command, {})
        if authenticated
        else ([], {}),
    )
    secret = b"github_pat_synthetic_invalid_utf8_\xff"
    monkeypatch.setattr(
        pipeline,
        "_run_bounded_windows_process",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            command,
            0,
            secret,
            b"",
        ),
    )

    with pytest.raises(pipeline.OODExternalV2IntegrityError) as captured:
        pipeline._run_exact_private_live_remote(tmp_path)

    assert "github_pat_synthetic" not in str(captured.value)
    assert "github_pat_synthetic" not in repr(captured.value)
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None


def _amendment_git_runner(
    implementation_revision: str,
    *,
    first_revision_line: str | None = None,
    first_commit_count: str = "1",
    first_diff_stdout: str | None = None,
    revision_line: str | None = None,
    commit_count: str = "1",
    diff_stdout: str | None = None,
) -> Any:
    resolved_diff = diff_stdout
    if resolved_diff is None:
        resolved_diff = "".join(
            f"M\t{path}\n"
            for path in pipeline.SUCCESSOR_PRIVATE_REMOTE_AMENDMENT_MODIFIED_PATHS
        )
    first_diff = first_diff_stdout
    if first_diff is None:
        first_diff = "".join(
            f"M\t{path}\n" for path in pipeline.SUCCESSOR_AMENDMENT_MODIFIED_PATHS
        )
    responses = {
        (
            "rev-list",
            "--parents",
            "-n",
            "1",
            pipeline.SECOND_FROZEN_SUCCESSOR_IMPLEMENTATION_REVISION,
        ): first_revision_line
        or (
            f"{pipeline.SECOND_FROZEN_SUCCESSOR_IMPLEMENTATION_REVISION} "
            f"{pipeline.FIRST_FROZEN_SUCCESSOR_IMPLEMENTATION_REVISION}\n"
        ),
        (
            "rev-list",
            "--count",
            f"{pipeline.FIRST_FROZEN_SUCCESSOR_IMPLEMENTATION_REVISION}.."
            f"{pipeline.SECOND_FROZEN_SUCCESSOR_IMPLEMENTATION_REVISION}",
        ): f"{first_commit_count}\n",
        (
            "diff",
            "--name-status",
            "--no-renames",
            f"{pipeline.FIRST_FROZEN_SUCCESSOR_IMPLEMENTATION_REVISION}.."
            f"{pipeline.SECOND_FROZEN_SUCCESSOR_IMPLEMENTATION_REVISION}",
            "--",
        ): first_diff,
        (
            "rev-list",
            "--parents",
            "-n",
            "1",
            implementation_revision,
        ): revision_line
        or (
            f"{implementation_revision} "
            f"{pipeline.SECOND_FROZEN_SUCCESSOR_IMPLEMENTATION_REVISION}\n"
        ),
        (
            "rev-list",
            "--count",
            f"{pipeline.SECOND_FROZEN_SUCCESSOR_IMPLEMENTATION_REVISION}.."
            f"{implementation_revision}",
        ): f"{commit_count}\n",
        (
            "diff",
            "--name-status",
            "--no-renames",
            f"{pipeline.SECOND_FROZEN_SUCCESSOR_IMPLEMENTATION_REVISION}.."
            f"{implementation_revision}",
            "--",
        ): resolved_diff,
    }

    def runner(_root: Path, *arguments: str, **_kwargs: object) -> Any:
        if arguments not in responses:
            raise AssertionError(arguments)
        return SimpleNamespace(
            stdout=responses[arguments],
            stderr="",
            returncode=0,
        )

    return runner


def test_successor_amendment_revision_binds_parent_blob_and_exact_diff(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    revision = "d" * 40
    observed: list[dict[str, object]] = []
    monkeypatch.setattr(pipeline, "_run_git", _amendment_git_runner(revision))

    def verify_old_parent(
        _root: Path,
        **kwargs: object,
    ) -> None:
        observed.append(dict(kwargs))

    monkeypatch.setattr(pipeline, "_verify_historical_revision_blob", verify_old_parent)
    pipeline._verify_successor_amendment_revision(  # noqa: SLF001
        tmp_path,
        implementation_revision=revision,
    )

    assert observed == [
        {
            "context": "first frozen successor parent",
            "expected_file_sha256": (
                pipeline.FIRST_FROZEN_SUCCESSOR_PARENT_CONFIG_SHA256
            ),
            "relative_path": pipeline.SUCCESSOR_PARENT_CONFIG_PATH,
            "revision": pipeline.FIRST_FROZEN_SUCCESSOR_IMPLEMENTATION_REVISION,
        },
        {
            "context": "second frozen successor parent",
            "expected_file_sha256": (
                pipeline.SECOND_FROZEN_SUCCESSOR_PARENT_CONFIG_SHA256
            ),
            "relative_path": pipeline.SUCCESSOR_PARENT_CONFIG_PATH,
            "revision": pipeline.SECOND_FROZEN_SUCCESSOR_IMPLEMENTATION_REVISION,
        },
    ]


@pytest.mark.parametrize(
    "runner_kwargs",
    (
        {
            "first_revision_line": (
                f"{pipeline.SECOND_FROZEN_SUCCESSOR_IMPLEMENTATION_REVISION} "
                f"{'e' * 40}\n"
            )
        },
        {"first_commit_count": "2"},
        {"first_diff_stdout": "A\tunexpected.txt\n"},
    ),
)
def test_successor_amendment_revision_rejects_x1_to_x2_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    runner_kwargs: dict[str, str],
) -> None:
    revision = "d" * 40
    monkeypatch.setattr(
        pipeline,
        "_run_git",
        _amendment_git_runner(revision, **runner_kwargs),
    )
    monkeypatch.setattr(
        pipeline,
        "_verify_historical_revision_blob",
        lambda *_args, **_kwargs: None,
    )

    with pytest.raises(
        pipeline.OODExternalV2IntegrityError,
        match="first successor amendment",
    ):
        pipeline._verify_successor_amendment_revision(  # noqa: SLF001
            tmp_path,
            implementation_revision=revision,
        )


@pytest.mark.parametrize(
    ("runner_kwargs", "message"),
    (
        (
            {"revision_line": f"{'d' * 40} {'e' * 40}\n"},
            "sole direct child",
        ),
        ({"commit_count": "2"}, "sole direct child"),
        ({"diff_stdout": "A\tunexpected.txt\n"}, "non-modification"),
        (
            {
                "diff_stdout": "".join(
                    f"M\t{path}\n"
                    for path in (
                        pipeline.SUCCESSOR_PRIVATE_REMOTE_AMENDMENT_MODIFIED_PATHS[:-1]
                    )
                )
            },
            "paths differ",
        ),
        (
            {
                "diff_stdout": "".join(
                    f"M\t{path}\n"
                    for path in pipeline.SUCCESSOR_PRIVATE_REMOTE_AMENDMENT_MODIFIED_PATHS
                )
                + "M\textra.txt\n"
            },
            "paths differ",
        ),
        ({"diff_stdout": "R100\told.txt\tnew.txt\n"}, "non-modification"),
    ),
)
def test_successor_amendment_revision_rejects_lineage_or_diff_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    runner_kwargs: dict[str, str],
    message: str,
) -> None:
    revision = "d" * 40
    monkeypatch.setattr(
        pipeline,
        "_run_git",
        _amendment_git_runner(revision, **runner_kwargs),
    )
    monkeypatch.setattr(
        pipeline,
        "_verify_historical_revision_blob",
        lambda *_args, **_kwargs: None,
    )

    with pytest.raises(pipeline.OODExternalV2IntegrityError, match=message):
        pipeline._verify_successor_amendment_revision(  # noqa: SLF001
            tmp_path,
            implementation_revision=revision,
        )


def test_successor_amendment_revision_rejects_old_parent_blob_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    revision = "d" * 40
    monkeypatch.setattr(pipeline, "_run_git", _amendment_git_runner(revision))

    def reject_old_parent(*_args: object, **_kwargs: object) -> None:
        raise pipeline.OODExternalV2IntegrityError("old parent blob differs")

    monkeypatch.setattr(pipeline, "_verify_historical_revision_blob", reject_old_parent)
    with pytest.raises(
        pipeline.OODExternalV2IntegrityError,
        match="old parent blob differs",
    ):
        pipeline._verify_successor_amendment_revision(  # noqa: SLF001
            tmp_path,
            implementation_revision=revision,
        )


def test_successor_amendment_revision_rejects_x2_parent_blob_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    revision = "d" * 40
    monkeypatch.setattr(pipeline, "_run_git", _amendment_git_runner(revision))

    def reject_second_parent(
        _root: Path,
        **kwargs: object,
    ) -> None:
        if kwargs.get("context") == "second frozen successor parent":
            raise pipeline.OODExternalV2IntegrityError("X2 parent blob differs")

    monkeypatch.setattr(
        pipeline,
        "_verify_historical_revision_blob",
        reject_second_parent,
    )
    with pytest.raises(
        pipeline.OODExternalV2IntegrityError,
        match="X2 parent blob differs",
    ):
        pipeline._verify_successor_amendment_revision(  # noqa: SLF001
            tmp_path,
            implementation_revision=revision,
        )


def test_historical_blob_verification_allows_current_amended_bytes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    revision = pipeline.FIRST_FROZEN_SUCCESSOR_IMPLEMENTATION_REVISION
    relative_path = pipeline.SUCCESSOR_PARENT_CONFIG_PATH
    current = tmp_path.joinpath(*relative_path.split("/"))
    current.parent.mkdir(parents=True)
    current.write_bytes(b"current amended parent")
    historical = b"first frozen parent"

    def git_result(_root: Path, *arguments: str, **_kwargs: object) -> Any:
        assert arguments == ("cat-file", "-t", f"{revision}:{relative_path}")
        return SimpleNamespace(stdout="blob\n", stderr="", returncode=0)

    def git_bytes(_root: Path, *arguments: str) -> bytes:
        assert arguments == ("show", f"{revision}:{relative_path}")
        return historical

    monkeypatch.setattr(pipeline, "_run_git", git_result)
    monkeypatch.setattr(pipeline, "_run_git_bytes", git_bytes)
    pipeline._verify_historical_revision_blob(  # noqa: SLF001
        tmp_path,
        revision=revision,
        relative_path=relative_path,
        expected_file_sha256=sha256_bytes(historical),
        context="first frozen successor parent",
    )

    assert current.read_bytes() != historical


def test_historical_blob_verification_rejects_non_blob_object_type(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    revision = pipeline.FIRST_FROZEN_SUCCESSOR_IMPLEMENTATION_REVISION
    relative_path = pipeline.SUCCESSOR_PARENT_CONFIG_PATH
    historical = b"first frozen parent"
    monkeypatch.setattr(
        pipeline,
        "_run_git",
        lambda *_args, **_kwargs: SimpleNamespace(
            stdout="tree\n",
            stderr="",
            returncode=0,
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "_run_git_bytes",
        lambda *_args, **_kwargs: historical,
    )

    with pytest.raises(
        pipeline.OODExternalV2IntegrityError,
        match="differs from its exact historical Git blob",
    ):
        pipeline._verify_historical_revision_blob(  # noqa: SLF001
            tmp_path,
            revision=revision,
            relative_path=relative_path,
            expected_file_sha256=sha256_bytes(historical),
            context="first frozen successor parent",
        )


def test_historical_blob_verification_rejects_wrong_blob_sha256(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    revision = pipeline.FIRST_FROZEN_SUCCESSOR_IMPLEMENTATION_REVISION
    relative_path = pipeline.SUCCESSOR_PARENT_CONFIG_PATH
    expected = b"first frozen parent"
    monkeypatch.setattr(
        pipeline,
        "_run_git",
        lambda *_args, **_kwargs: SimpleNamespace(
            stdout="blob\n",
            stderr="",
            returncode=0,
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "_run_git_bytes",
        lambda *_args, **_kwargs: b"tampered historical parent",
    )

    with pytest.raises(
        pipeline.OODExternalV2IntegrityError,
        match="differs from its exact historical Git blob",
    ):
        pipeline._verify_historical_revision_blob(  # noqa: SLF001
            tmp_path,
            revision=revision,
            relative_path=relative_path,
            expected_file_sha256=sha256_bytes(expected),
            context="first frozen successor parent",
        )


def test_revision_boundary_rejects_merge_child_freeze_commit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    implementation = "a" * 40
    execution = "b" * 40
    second_parent = "c" * 40
    monkeypatch.setattr(
        pipeline,
        "_verify_clean_git_revision",
        lambda _root: execution,
    )
    monkeypatch.setattr(
        pipeline,
        "_verify_successor_amendment_revision",
        lambda *_args, **_kwargs: None,
    )

    def git_result(_root: Path, *arguments: str, **_kwargs: object) -> Any:
        if arguments[:2] == ("merge-base", "--is-ancestor"):
            return SimpleNamespace(stdout="", stderr="", returncode=0)
        if arguments[:3] == ("rev-list", "--count", f"{implementation}..{execution}"):
            return SimpleNamespace(stdout="1\n", stderr="", returncode=0)
        if arguments[:4] == ("rev-list", "--parents", "-n", "1"):
            return SimpleNamespace(
                stdout=f"{execution} {implementation} {second_parent}\n",
                stderr="",
                returncode=0,
            )
        raise AssertionError(arguments)

    monkeypatch.setattr(pipeline, "_run_git", git_result)
    with pytest.raises(
        pipeline.OODExternalV2IntegrityError,
        match="exactly one child-freeze commit",
    ):
        pipeline._verify_revision_boundary(  # noqa: SLF001
            tmp_path,
            child=cast(
                Any,
                SimpleNamespace(implementation_revision=implementation),
            ),
            execution_revision=execution,
        )


@pytest.mark.parametrize(
    ("option", "tagged_line"),
    (("-t", "S tracked.py\n"), ("-v", "h tracked.py\n")),
)
def test_clean_revision_rejects_special_git_index_bits(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    option: str,
    tagged_line: str,
) -> None:
    def git_result(_root: Path, *arguments: str, **_kwargs: object) -> Any:
        if arguments == ("ls-files", option):
            return SimpleNamespace(stdout=tagged_line, stderr="", returncode=0)
        if arguments[:1] == ("ls-files",):
            return SimpleNamespace(stdout="H tracked.py\n", stderr="", returncode=0)
        raise AssertionError(arguments)

    monkeypatch.setattr(pipeline, "_run_git", git_result)
    with pytest.raises(
        pipeline.OODExternalV2IntegrityError,
        match="skip-worktree or assume-unchanged",
    ):
        pipeline._verify_clean_git_revision(tmp_path)  # noqa: SLF001


def test_private_git_history_must_be_exactly_empty(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed: tuple[str, ...] = ()

    def empty_history(_root: Path, *arguments: str, **_kwargs: object) -> Any:
        nonlocal observed
        observed = arguments
        return SimpleNamespace(stdout="", stderr="", returncode=0)

    monkeypatch.setattr(pipeline, "_run_git", empty_history)
    pipeline._verify_private_history_absent(tmp_path)  # noqa: SLF001
    assert observed == (
        "log",
        "--full-history",
        "--all",
        "--reflog",
        "--format=%H",
        "--",
        *pipeline.FORBIDDEN_GIT_HISTORY_PATHS,
    )

    monkeypatch.setattr(
        pipeline,
        "_run_git",
        lambda *_args, **_kwargs: SimpleNamespace(
            stdout="f" * 40 + "\n",
            stderr="",
            returncode=0,
        ),
    )
    with pytest.raises(
        pipeline.OODExternalV2IntegrityError,
        match="appear in Git history",
    ):
        pipeline._verify_private_history_absent(tmp_path)  # noqa: SLF001


def test_private_history_full_history_finds_deleted_merged_side_parent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repository = tmp_path / "synthetic-merge-history"
    repository.mkdir()
    git_executable = shutil.which("git")
    assert git_executable is not None

    def git(*arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [git_executable, *arguments],
            cwd=repository,
            check=True,
            capture_output=True,
            encoding="utf-8",
            errors="strict",
            text=True,
        )

    git("init", "--initial-branch=main")
    git("config", "user.name", "Synthetic History Test")
    git("config", "user.email", "synthetic-history@example.invalid")
    (repository / "README.md").write_text("base\n", encoding="utf-8")
    git("add", "README.md")
    git("commit", "-m", "base")

    git("switch", "-c", "side-parent")
    protected_relative = "data/raw/external-ood/deleted-before-merge.txt"
    protected = repository / protected_relative
    protected.parent.mkdir(parents=True)
    protected.write_text("synthetic protected evidence\n", encoding="utf-8")
    git("add", "--", protected_relative)
    git("commit", "-m", "add protected path on side parent")
    side_revision = git("rev-parse", "HEAD").stdout.strip()

    git("switch", "main")
    (repository / "main-only.txt").write_text("main\n", encoding="utf-8")
    git("add", "main-only.txt")
    git("commit", "-m", "advance main")
    git("merge", "--no-ff", "--no-commit", "side-parent")
    protected.unlink()
    git("add", "-u", "--", protected_relative)
    git("commit", "-m", "merge while deleting protected path")
    git("branch", "-D", "side-parent")
    git("reflog", "expire", "--expire=now", "--all")

    simplified = git(
        "log",
        "--all",
        "--reflog",
        "--format=%H",
        "--",
        *pipeline.FORBIDDEN_GIT_HISTORY_PATHS,
    ).stdout
    full_history = git(
        "log",
        "--full-history",
        "--all",
        "--reflog",
        "--format=%H",
        "--",
        *pipeline.FORBIDDEN_GIT_HISTORY_PATHS,
    ).stdout
    assert simplified == ""
    assert side_revision in full_history.splitlines()
    assert not protected.exists()

    observed: tuple[str, ...] = ()

    def synthetic_git(
        root: Path,
        *arguments: str,
        allow_empty: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        nonlocal observed
        assert root == repository
        assert allow_empty is False
        observed = arguments
        return git(*arguments)

    monkeypatch.setattr(pipeline, "_run_git", synthetic_git)
    with pytest.raises(
        pipeline.OODExternalV2IntegrityError,
        match="appear in Git history",
    ):
        pipeline._verify_private_history_absent(repository)  # noqa: SLF001
    assert observed[:2] == ("log", "--full-history")


def _runtime_scratch_layout(root: Path) -> None:
    (root / "pycache").mkdir(parents=True)
    (root / "temp").mkdir()
    (root / "home" / "AppData" / "Roaming").mkdir(parents=True)
    (root / "home" / "AppData" / "Local").mkdir()


def _runtime_scratch_environment(root: Path) -> dict[str, str]:
    return {
        "APPDATA": str(root / "home" / "AppData" / "Roaming"),
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "CUDA_CACHE_DISABLE": "1",
        "LOCALAPPDATA": str(root / "home" / "AppData" / "Local"),
        "PATH": r"C:\Windows\System32",
        "PROGRAMFILES": r"C:\Program Files",
        "PROGRAMW6432": r"C:\Program Files",
        "TEMP": str(root / "temp"),
        "TMP": str(root / "temp"),
        "TORCHINDUCTOR_CACHE_DIR": str(root / "temp"),
        "USERPROFILE": str(root / "home"),
    }


def test_runtime_environment_hash_material_canonicalizes_fresh_owned_roots(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    parent = project / "artifacts" / "trust_sentinel"
    parent.mkdir(parents=True)
    roots = tuple(
        parent / f".ood_external_v2_1.runtime-{'a' * 64}"
        for _ in range(1)
    ) + (parent / f".ood_external_v2_1.runtime-{'b' * 64}",)
    observed: list[tuple[tuple[str, str], ...]] = []
    for root in roots:
        _runtime_scratch_layout(root)
        with monkeypatch.context() as context:
            context.setattr(os, "environ", _runtime_scratch_environment(root))
            observed.append(
                pipeline._frozen_runtime_environment_material(  # noqa: SLF001
                    root,
                    project_root=project,
                )
            )
    assert observed[0] == observed[1]
    assert all(str(root) not in repr(observed) for root in roots)


def test_native_module_audit_requires_base_python_not_uv_redirector(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    redirector = tmp_path / "venv" / "Scripts" / "python.exe"
    base = tmp_path / "python-base"
    base_python = base / "python.exe"
    site = tmp_path / "venv" / "Lib" / "site-packages"
    system_root = tmp_path / "Windows"
    system32 = system_root / "System32"
    winsxs = system_root / "WinSxS"
    for directory in (redirector.parent, base, site, system32, winsxs):
        directory.mkdir(parents=True, exist_ok=True)
    for path in (
        redirector,
        base_python,
        system32 / "nvidia-smi.exe",
        system32 / "nvml.dll",
        system32 / "nvcuda.dll",
    ):
        path.write_bytes(b"bound")
    monkeypatch.setenv("SYSTEMROOT", str(system_root))
    monkeypatch.setattr(
        pipeline,
        "_nvidia_driver_tool_paths",
        lambda: (
            system32 / "nvidia-smi.exe",
            system32 / "nvml.dll",
            system32 / "nvcuda.dll",
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "_loaded_windows_native_module_paths",
        lambda: (base_python,),
    )
    pipeline._verify_loaded_native_module_origins(  # noqa: SLF001
        python_executable=redirector,
        python_base_target=base,
        site_packages=site,
    )

    monkeypatch.setattr(
        pipeline,
        "_loaded_windows_native_module_paths",
        lambda: (redirector,),
    )
    with pytest.raises(
        pipeline.OODExternalV2IntegrityError,
        match="exact CPython base executable",
    ):
        pipeline._verify_loaded_native_module_origins(  # noqa: SLF001
            python_executable=redirector,
            python_base_target=base,
            site_packages=site,
        )


def test_complete_bundle_tree_snapshot_detects_same_size_member_replacement(
    tmp_path: Path,
) -> None:
    root = tmp_path / "bundle"
    for relative in bundle_module.BUNDLE_MEMBER_PATHS:
        path = root / Path(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    before = bundle_module._exact_tree_snapshot(  # noqa: SLF001
        root,
        include_success_manifest=False,
    )
    target = root / Path(bundle_module.BUNDLE_MEMBER_PATHS[-1])
    target.write_bytes(b"y")
    after = bundle_module._exact_tree_snapshot(  # noqa: SLF001
        root,
        include_success_manifest=False,
    )
    assert after != before


def test_bundle_snapshot_rejects_an_indirect_parent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "bundle"
    for relative in bundle_module.BUNDLE_MEMBER_PATHS:
        path = root / Path(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    indirect_parent = root.parent
    original = Path.is_junction
    monkeypatch.setattr(
        Path,
        "is_junction",
        lambda self: self == indirect_parent or original(self),
    )
    with pytest.raises(
        bundle_module.ExternalV2BundleError,
        match="indirect filesystem component",
    ):
        bundle_module._exact_tree_snapshot(  # noqa: SLF001
            root,
            include_success_manifest=False,
        )


def test_git_repository_controls_reject_worktree_config_and_fsmonitor(
    tmp_path: Path,
) -> None:
    root = _isolated_successor_preflight_project(tmp_path)
    config = root / ".git" / "config"
    config.write_text(
        config.read_text(encoding="utf-8") + "[core]\n\tfsmonitor = evil.exe\n",
        encoding="utf-8",
    )
    with pytest.raises(
        pipeline.OODExternalV2IntegrityError,
        match="unapproved section",
    ):
        pipeline._verify_git_repository_controls(root)  # noqa: SLF001

    config.write_text(
        config.read_text(encoding="utf-8").rsplit("[core]", maxsplit=1)[0],
        encoding="utf-8",
    )
    (root / ".git" / "config.worktree").write_text("[core]\n", encoding="utf-8")
    with pytest.raises(
        pipeline.OODExternalV2IntegrityError,
        match="alternates, grafts",
    ):
        pipeline._verify_git_repository_controls(root)  # noqa: SLF001


@pytest.mark.parametrize("relative_path", ("shallow", "shallow.lock"))
def test_git_repository_controls_reject_shallow_state(
    tmp_path: Path,
    relative_path: str,
) -> None:
    root = _isolated_successor_preflight_project(tmp_path)
    (root / ".git" / relative_path).write_text("0" * 40 + "\n", encoding="ascii")

    with pytest.raises(
        pipeline.OODExternalV2IntegrityError,
        match="shallow state",
    ):
        pipeline._verify_git_repository_controls(root)  # noqa: SLF001


@pytest.mark.parametrize("relative_path", ("shallow", "shallow.lock"))
def test_git_repository_controls_reject_indirect_shallow_state(
    tmp_path: Path,
    relative_path: str,
) -> None:
    root = _isolated_successor_preflight_project(tmp_path)
    target = root / ".git" / f"actual-{relative_path}"
    target.write_text("0" * 40 + "\n", encoding="ascii")
    linked = root / ".git" / relative_path
    try:
        linked.symlink_to(target)
    except OSError:
        pytest.skip("file symlink creation is unavailable")

    with pytest.raises(
        pipeline.OODExternalV2IntegrityError,
        match="shallow state",
    ):
        pipeline._verify_git_repository_controls(root)  # noqa: SLF001


def test_inventory_builder_preflight_binds_frozen_runtime_and_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "project"
    parent_path = root / pipeline.SUCCESSOR_PARENT_CONFIG_PATH
    parent_path.parent.mkdir(parents=True)
    parent_path.write_bytes(b"frozen")
    (root / "artifacts" / "trust_sentinel").mkdir(parents=True)
    revision = "a" * 40
    parent_hash = "sha256:" + "b" * 64
    source_tree = pipeline.ProjectSourceTreeBinding(  # noqa: SLF001
        files=(),
        file_count=0,
        total_bytes=0,
        tree_sha256="sha256:" + "c" * 64,
    )
    runtime = SimpleNamespace(
        python_environment_sha256="sha256:" + "d" * 64,
        git_tool=SimpleNamespace(
            runtime_tree=SimpleNamespace(tree_sha256="sha256:" + "e" * 64)
        ),
    )
    parent = SimpleNamespace(
        path=parent_path,
        status="frozen_parent_preregistration_pre_waveform",
        file_sha256=parent_hash,
        output_root="artifacts/trust_sentinel/ood_external_v2_1",
        claim_path="artifacts/trust_sentinel/.ood_external_v2_1.one-shot-claim.json",
    )
    monkeypatch.setattr(
        pipeline,
        "EXPECTED_SUCCESSOR_PARENT_CONFIG_SHA256",
        parent_hash,
    )
    monkeypatch.setattr(pipeline, "_strict_project_root", lambda _r: root)
    monkeypatch.setattr(pipeline, "_load_parent_for_operation", lambda *_a, **_k: parent)
    monkeypatch.setattr(pipeline, "_current_runtime_environment", lambda: runtime)
    monkeypatch.setattr(pipeline, "_build_project_source_tree", lambda _r: source_tree)
    monkeypatch.setattr(pipeline, "_verify_clean_git_revision", lambda _r: revision)
    monkeypatch.setattr(pipeline, "_verify_git_remote_state", lambda *_a, **_k: None)
    monkeypatch.setattr(pipeline, "_verify_private_history_absent", lambda _r: None)
    monkeypatch.setattr(
        pipeline,
        "_verify_successor_amendment_revision",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(pipeline, "_verify_tracked_head_blob", lambda *_a, **_k: None)
    monkeypatch.setattr(
        pipeline,
        "_verify_project_source_tree_at_revisions",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        pipeline,
        "_verify_imported_project_module_origins",
        lambda *_a, **_k: None,
    )

    verified = pipeline.verify_inventory_builder_preflight(
        parent_path,
        root,
        revision,
    )
    assert verified.status == "INVENTORY_BUILDER_PREFLIGHT_VERIFIED"
    assert verified.parent_config_file_sha256 == parent_hash
    assert verified.project_source_tree_sha256 == source_tree.tree_sha256
    assert verified.python_environment_sha256 == runtime.python_environment_sha256

    (root / parent.output_root).mkdir()
    with pytest.raises(
        pipeline.OODExternalV2IntegrityError,
        match="output root must be absent",
    ):
        pipeline.verify_inventory_builder_preflight(parent_path, root, revision)


def test_claim_publication_witness_does_not_fire_when_directory_fsync_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    published = False

    def fail_fsync(_path: Path) -> None:
        raise OSError("injected durability failure")

    def witness() -> None:
        nonlocal published
        published = True

    monkeypatch.setattr(pipeline, "_fsync_directory", fail_fsync)
    with pytest.raises(
        pipeline.OODExternalV2ExecutionError,
        match="atomic artifact commit failed",
    ):
        pipeline._atomic_write_new(  # noqa: SLF001
            tmp_path / "claim.json",
            b"{}\n",
            publication_witness=witness,
        )
    assert published is False


def test_private_quality_report_tamper_is_rejected() -> None:
    signal = np.zeros((len(LEADS), TARGET_SAMPLES), dtype=np.float32)
    report = assess_signal_quality(
        signal,
        SignalMetadata.canonical(DEFAULT_SIGNAL_QUALITY_CONFIG),
    )
    report_body = pipeline._quality_report_dict(report)  # noqa: SLF001
    evidence = pipeline._PrivateRecordEvidence(  # noqa: SLF001
        dataset=CHALLENGE_2011_DATASET,
        record_ref="A0001",
        patient_key=None,
        challenge_quality_label="acceptable",
        adapter_provenance_sha256="sha256:" + "c" * 64,
        adapter_source_sample_count=5_000,
        adapter_raw_physical_units=(PHYSICAL_UNITS,) * len(LEADS),
        canonical_signal_sha256="sha256:" + "d" * 64,
        quality_report_sha256=pipeline._quality_report_sha256(report_body),  # noqa: SLF001
        quality_report=report_body,
        quality_status=report.status.value,
        quality_reason_codes=tuple(reason.value for reason in report.reason_codes),
        route="REACQUIRE",
        distribution_score=None,
        entropy=None,
        entropy_accepted=None,
        conformal_decisions=None,
        all_conformal_decisions_singleton=None,
    )
    pipeline._verify_private_quality_report_semantics(evidence)  # noqa: SLF001

    tampered = deepcopy(report_body)
    tampered["status"] = "pass"
    with pytest.raises(
        pipeline.OODExternalV2IntegrityError,
        match="quality status or reason codes differ",
    ):
        pipeline._verify_private_quality_report_semantics(  # noqa: SLF001
            replace(evidence, quality_report=tampered, quality_status="pass")
        )
