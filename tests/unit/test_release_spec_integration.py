from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import cast

import numpy as np
import pytest

import ecg_trust.final_evaluation_spec as spec_module
import ecg_trust.release_gates as release_gates
from ecg_trust.final_evaluation_spec import (
    create_final_evaluation_spec,
    load_final_evaluation_spec,
    save_final_evaluation_spec,
)
from ecg_trust.prediction_export import PredictionExportRequest, PredictionExportResult
from ecg_trust.predictions import create_prediction_artifact, save_prediction_artifact
from ecg_trust.protocol import (
    CALIBRATION_FOLDS,
    LABEL_ORDER,
    ExperimentProtocol,
    FoldRole,
    load_protocol,
)
from ecg_trust.release_gates import RefitBundle, RefitMember, export_fold9_predictions
from ecg_trust.subgroup_artifact import build_subgroup_artifact, save_subgroup_artifact


def _prefixed_file_hash(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _hash(tag: str) -> str:
    return "sha256:" + hashlib.sha256(tag.encode("utf-8")).hexdigest()


def _write(path: Path, value: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    return path.resolve()


def _runtime_envelope(project_root: Path) -> dict[str, object]:
    lock_path = _write(project_root / "uv.lock", "version = 1\n")
    return {
        "project_root": str(project_root.resolve()),
        "git": {"revision": "a" * 40, "dirty": False},
        "dependency_lock": {
            "path": str(lock_path),
            "sha256": _prefixed_file_hash(lock_path),
        },
        "software": {
            "python": "3.12.13",
            "ecg_trust": "0.1.0",
            "torch": "2.13.0+cu130",
            "cuda_runtime": "13.0",
            "cudnn": 91100,
            "nvidia_driver": "596.49",
            "installed_environment": {
                "distribution_count": 6,
                "distributions_sha256": _hash("installed-environment"),
                "core_packages": {
                    "numpy": "2.0.0",
                    "scipy": "1.14.0",
                    "scikit-learn": "1.5.0",
                    "pandas": "2.2.0",
                    "wfdb": "4.2.0",
                    "pyarrow": "17.0.0",
                },
            },
        },
        "hardware": {
            "requested_device": "cuda:0",
            "resolved_device": "cuda:0",
            "device_name": "Synthetic CUDA GPU",
            "device_uuid": "GPU-aba61180-7244-4ad9-bdf5-60c764cb1d59",
            "pci_domain_id": 0,
            "pci_bus_id": 1,
            "pci_device_id": 0,
            "device_capability": [12, 0],
            "total_memory_bytes": 12 * 1024**3,
            "bf16_supported": True,
        },
        "policy": {
            "allow_device_auto": False,
            "require_cuda": True,
            "bf16_requested": True,
            "bf16_required": True,
            "autocast_dtype": "torch.bfloat16",
            "deterministic_algorithms_enabled": True,
            "deterministic_algorithms_warn_only": False,
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
            "float32_matmul_precision": "highest",
            "cublas_workspace_config": ":4096:8",
            "cuda_visible_devices": None,
        },
    }


def _synthetic_manifest(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    targets = np.asarray(
        [
            [1, 0, 0, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 0, 1, 0, 1],
            [1, 1, 0, 0, 0],
        ],
        dtype=np.int8,
    )
    lines = [
        "ecg_id,patient_id,strat_fold,age,sex,"
        "label_NORM,label_MI,label_STTC,label_CD,label_HYP"
    ]
    for index, target in enumerate(targets, start=1):
        lines.append(
            f"{index},{100 + index},9,{30 + index},{index % 2},"
            + ",".join(str(int(value)) for value in target)
        )
    # These are synthetic sentinels, not protected labels. The subgroup/spec
    # path reads only identity, age, and sex, while the release check predicate-
    # selects fold 9 before materializing target columns.
    lines.extend(
        [
            "10,210,10,44,0,SEALED,SEALED,SEALED,SEALED,SEALED",
            "11,211,10,71,1,SEALED,SEALED,SEALED,SEALED,SEALED",
        ]
    )
    _write(path, "\n".join(lines) + "\n")
    return (
        np.arange(1, 5, dtype=np.int64),
        np.arange(101, 105, dtype=np.int64),
        targets,
    )


def _member(
    root: Path,
    *,
    architecture: str,
    seed: int,
    protocol: ExperimentProtocol,
    manifest_path: Path,
    manifest_sha256: str,
) -> RefitMember:
    member_id = f"{architecture}-seed{seed}"
    member_root = root / member_id
    checkpoint = _write(member_root / "final.ckpt", f"checkpoint:{member_id}\n")
    resolved_config = _write(
        member_root / "resolved-config.json",
        json.dumps(
            {"config": {"loader": {"batch_size": 2, "num_workers": 0}}},
            sort_keys=True,
        )
        + "\n",
    )
    metadata = _write(member_root / "metadata.json", "{}\n")
    protocol_file = _write(member_root / "protocol.json", "{}\n")
    history = _write(member_root / "history.json", "{}\n")
    normalization = _write(member_root / "normalization.json", "{}\n")
    source_checkpoint = _write(member_root / "source.ckpt", "source\n")
    completion = _write(member_root / "refit-completion.json", "{}\n")
    freeze = _write(member_root / "freeze.json", "{}\n")
    source_completion = _write(member_root / "source-completion.json", "{}\n")
    config_hash = _hash(f"config:{member_id}")
    selection = {"member_id": member_id, "frozen": True}
    return RefitMember(
        member_id=member_id,
        comparison_id="synthetic-exact-six",
        architecture=architecture,
        seed=seed,
        run_name=member_id,
        run_dir=member_root.resolve(),
        completion_path=completion,
        completion_sha256=_prefixed_file_hash(completion),
        freeze_artifact_path=freeze,
        freeze_artifact_sha256=_prefixed_file_hash(freeze),
        recipe_sha256=_hash(f"recipe:{member_id}"),
        source_member_completion_path=source_completion,
        source_member_completion_sha256=_prefixed_file_hash(source_completion),
        final_checkpoint_path=checkpoint,
        final_checkpoint_sha256=_prefixed_file_hash(checkpoint),
        resolved_config_path=resolved_config,
        resolved_config_file_sha256=_prefixed_file_hash(resolved_config),
        resolved_config_hash=config_hash,
        metadata_path=metadata,
        metadata_sha256=_prefixed_file_hash(metadata),
        protocol_path=protocol_file,
        protocol_file_sha256=_prefixed_file_hash(protocol_file),
        history_path=history,
        history_sha256=_prefixed_file_hash(history),
        protocol_hash=protocol.protocol_hash,
        manifest_path=manifest_path.resolve(),
        manifest_sha256=manifest_sha256,
        normalization_path=normalization,
        normalization_sha256=_prefixed_file_hash(normalization),
        source_checkpoint_path=source_checkpoint,
        source_checkpoint_sha256=_prefixed_file_hash(source_checkpoint),
        frozen_epochs=3,
        selection_provenance=selection,
        selection_lineage_sha256=release_gates.canonical_sha256(selection),
        lineage_sha256=_hash(f"lineage:{member_id}"),
    )


def test_exact_six_fold9_export_uses_real_final_spec_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol_path = Path("configs/protocol.yaml").resolve()
    protocol = load_protocol(protocol_path)
    manifest_path = tmp_path / "synthetic-manifest.csv"
    ecg_id, patient_id, targets = _synthetic_manifest(manifest_path)
    manifest_sha256 = _prefixed_file_hash(manifest_path)

    subgroup = build_subgroup_artifact(
        manifest_path,
        protocol=protocol,
        expected_manifest_sha256=manifest_sha256,
    )
    subgroup_path, _ = save_subgroup_artifact(
        subgroup, tmp_path / "fold10-subgroups.json"
    )
    deviations_path = _write(
        tmp_path / "PROTOCOL_DEVIATIONS.md", "# Synthetic test deviations\nNone.\n"
    )
    refit_path = _write(tmp_path / "refit-bundle.json", "{}\n")
    members = tuple(
        _member(
            tmp_path / "refits",
            architecture=architecture,
            seed=seed,
            protocol=protocol,
            manifest_path=manifest_path,
            manifest_sha256=manifest_sha256,
        )
        for architecture in release_gates.EXPECTED_ARCHITECTURES
        for seed in release_gates.EXPECTED_SEEDS
    )
    unsigned = RefitBundle(
        protocol_hash=protocol.protocol_hash,
        manifest_sha256=manifest_sha256,
        normalization_sha256=members[0].normalization_sha256,
        label_order=tuple(LABEL_ORDER),
        members=members,
        created_at_utc="2026-08-08T12:00:00+00:00",
        artifact_sha256=None,
    )
    bundle = replace(
        unsigned,
        artifact_sha256=release_gates.canonical_sha256(
            unsigned.to_payload(include_integrity=False)
        ),
    )

    def load_synthetic_refit(
        path: str | Path,
        *,
        protocol: ExperimentProtocol,
        verify_sources: bool = True,
    ) -> RefitBundle:
        del verify_sources
        assert Path(path).resolve() == refit_path
        assert protocol.protocol_hash == bundle.protocol_hash
        return bundle

    monkeypatch.setattr(release_gates, "load_refit_bundle", load_synthetic_refit)
    runtime = _runtime_envelope(tmp_path / "runtime-project")

    def capture_runtime(project_root: Path, requested_device: str) -> dict[str, object]:
        assert project_root == Path(cast(str, runtime["project_root"]))
        assert requested_device == "cuda:0"
        return copy.deepcopy(runtime)

    monkeypatch.setattr(spec_module, "_capture_runtime_envelope", capture_runtime)
    created_spec = create_final_evaluation_spec(
        protocol=protocol,
        protocol_path=protocol_path,
        refit_bundle_path=refit_path,
        subgroup_artifact_path=subgroup_path,
        protocol_deviations_path=deviations_path,
        project_root=Path(cast(str, runtime["project_root"])),
        device="cuda:0",
    )
    saved_spec = save_final_evaluation_spec(
        created_spec, tmp_path / "final-evaluation-spec.json"
    )
    spec_path = cast(Path, saved_spec.path)
    # Exercise the real parser/source/runtime verification independently before
    # the release gate consumes the same immutable specification.
    reloaded_spec = load_final_evaluation_spec(
        spec_path,
        protocol=protocol,
        verify_sources=True,
        verify_runtime=True,
    )
    assert reloaded_spec.artifact_sha256 == saved_spec.artifact_sha256

    by_checkpoint = {member.final_checkpoint_path: member for member in members}
    exporter_calls: list[str] = []

    def synthetic_exporter(
        request: PredictionExportRequest,
        *,
        protocol: ExperimentProtocol,
    ) -> PredictionExportResult:
        member = by_checkpoint[request.checkpoint_path.resolve()]
        exporter_calls.append(member.member_id)
        prediction = create_prediction_artifact(
            ecg_id=ecg_id,
            patient_id=patient_id,
            strat_fold=np.full(ecg_id.shape, CALIBRATION_FOLDS[0], dtype=np.int8),
            targets=targets,
            raw_logits=np.full(
                targets.shape,
                fill_value=(member.seed - 2026) * 0.01,
                dtype=np.float32,
            ),
            model_name=member.run_name,
            model_seed=member.seed,
            protocol=protocol,
            config_hash=member.resolved_config_hash,
            manifest_hash=member.manifest_sha256,
            fold_role=FoldRole.CALIBRATION,
            extra_metadata={
                "lineage": "frozen_refit",
                "checkpoint_sha256": member.final_checkpoint_sha256,
                "checkpoint_epoch": member.frozen_epochs - 1,
                "resolved_config_path": str(member.resolved_config_path.resolve()),
                "normalization_sha256": member.normalization_sha256,
                "inference_device": "cuda:0",
                "inference_bf16": True,
                "inference_batch_size": 2,
                "inference_num_workers": 0,
                "refit_run_kind": "post_sweep_frozen_refit",
                "refit_completion_sha256": member.completion_sha256,
                "freeze_artifact_sha256": member.freeze_artifact_sha256,
                "recipe_sha256": member.recipe_sha256,
            },
        )
        files = save_prediction_artifact(
            prediction,
            request.output_path,
            protocol=protocol,
        )
        return PredictionExportResult(
            files=files,
            lineage="frozen_refit",
            fold_role=FoldRole.CALIBRATION,
            folds=CALIBRATION_FOLDS,
            record_count=len(ecg_id),
            model_name=member.run_name,
            model_seed=member.seed,
            checkpoint_sha256=member.final_checkpoint_sha256,
            config_hash=member.resolved_config_hash,
            manifest_hash=member.manifest_sha256,
            normalization_sha256=member.normalization_sha256,
            device="cuda:0",
            bf16_enabled=True,
        )

    output = tmp_path / "fold9_predictions"
    paths = export_fold9_predictions(
        refit_path,
        output,
        protocol=protocol,
        final_evaluation_spec_path=spec_path,
        exporter=synthetic_exporter,
    )

    assert exporter_calls == [member.member_id for member in members]
    assert set(paths) == {member.member_id for member in members}
    assert all(
        path.is_file() and path.with_suffix(".json").is_file()
        for path in paths.values()
    )
    plan = json.loads((output / "fold9-export-plan.json").read_text(encoding="utf-8"))
    completion = json.loads(
        (output / "fold9-export-completion.json").read_text(encoding="utf-8")
    )
    expected_spec_binding = {
        "path": str(spec_path.resolve()),
        "file_sha256": _prefixed_file_hash(spec_path),
        "artifact_sha256": saved_spec.artifact_sha256,
    }
    assert plan["final_evaluation_spec"] == expected_spec_binding
    assert completion["final_evaluation_spec"] == expected_spec_binding
    assert plan["inference"]["device"] == "cuda:0"
    assert plan["inference"]["bf16"] is True
    assert len(completion["members"]) == 6
