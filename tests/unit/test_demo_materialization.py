from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd  # type: ignore[import-untyped]
import pytest

import ecg_trust.demo_materialization as demo_materialization
from ecg_trust.constants import PTBXL_VERSION
from ecg_trust.demo_backend import FrozenDecisionPolicy
from ecg_trust.demo_materialization import (
    DEMO_BINDING_FILENAME,
    DEMO_EXAMPLE_COUNT,
    DEMO_EXAMPLES_FILENAME,
    DEMO_MEMBER_ID,
    DEMO_POLICY_FILENAME,
    DEMO_TARGET_COVERAGE,
    DemoMaterializationError,
    load_and_verify_demo_binding,
    materialize_demo,
)
from ecg_trust.protocol import CALIBRATION_FOLDS, ExperimentProtocol
from ecg_trust.release_gates import canonical_sha256


def _raw_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path.resolve()


def _fake_release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, Path, ExperimentProtocol]:
    protocol = ExperimentProtocol.canonical()
    release = tmp_path / "release"
    refit_bundle_path = _write(release / "refit_bundle.json", b"refit bundle\n")
    calibration_bundle_path = _write(
        release / "calibration_bundle.json", b"calibration bundle\n"
    )
    dataset_root = tmp_path / "dataset"
    rows: list[dict[str, object]] = [
        {
            "ecg_id": 1,
            "patient_id": 1,
            "strat_fold": 10,
            "record_path": "records/00001_lr",
        }
    ]
    for index, ecg_id in enumerate((12, 43, 53, 71, 74, 92), start=1):
        record_path = f"records/{ecg_id:05d}_lr"
        rows.append(
            {
                "ecg_id": ecg_id,
                "patient_id": 100 + index,
                "strat_fold": 8,
                "record_path": record_path,
            }
        )
        root = dataset_root / record_path
        _write(root.with_suffix(".hea"), f"header {ecg_id}\n".encode())
        _write(root.with_suffix(".dat"), f"signal {ecg_id}\n".encode())
    manifest = tmp_path / "manifest.parquet"
    pd.DataFrame(rows).to_parquet(manifest, index=False)
    manifest = manifest.resolve()
    normalization = _write(tmp_path / "normalization.json", b"normalization\n")
    checkpoint = _write(tmp_path / "final.ckpt", b"checkpoint\n")
    decision = _write(tmp_path / "fold9.decisions.json", b"decision\n")
    fold9_npz = _write(tmp_path / "fold9.npz", b"prediction\n")
    fold9_sidecar = _write(tmp_path / "fold9.json", b"sidecar\n")
    config_hash = "sha256:" + "c" * 64
    resolved_config = tmp_path / "resolved_refit_config.json"
    resolved_config.write_text(
        json.dumps(
            {
                "config": {
                    "data": {
                        "dataset_root": str(dataset_root.resolve()),
                        "manifest": str(manifest),
                        "normalization": str(normalization),
                    }
                },
                "config_hash": config_hash,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    resolved_config = resolved_config.resolve()

    manifest_sha256 = "sha256:" + _raw_hash(manifest)
    normalization_sha256 = "sha256:" + _raw_hash(normalization)
    checkpoint_sha256 = "sha256:" + _raw_hash(checkpoint)
    config_file_sha256 = "sha256:" + _raw_hash(resolved_config)
    refit_lineage_sha256 = "sha256:" + "1" * 64
    refit_artifact_sha256 = "sha256:" + "2" * 64
    calibration_artifact_sha256 = "sha256:" + "3" * 64
    refit_member = SimpleNamespace(
        member_id=DEMO_MEMBER_ID,
        architecture="resnet1d",
        seed=2026,
        run_name="resnet1d_refit_folds1-8_seed2026",
        lineage_sha256=refit_lineage_sha256,
        final_checkpoint_path=checkpoint,
        final_checkpoint_sha256=checkpoint_sha256,
        resolved_config_path=resolved_config,
        resolved_config_file_sha256=config_file_sha256,
        resolved_config_hash=config_hash,
        normalization_path=normalization,
        normalization_sha256=normalization_sha256,
        manifest_path=manifest,
        manifest_sha256=manifest_sha256,
    )
    calibration_member = SimpleNamespace(
        member_id=DEMO_MEMBER_ID,
        architecture="resnet1d",
        seed=2026,
        model_name=refit_member.run_name,
        refit_lineage_sha256=refit_lineage_sha256,
        checkpoint_path=checkpoint,
        checkpoint_sha256=checkpoint_sha256,
        resolved_config_path=resolved_config,
        resolved_config_file_sha256=config_file_sha256,
        resolved_config_hash=config_hash,
        normalization_path=normalization,
        normalization_sha256=normalization_sha256,
        prediction_path=fold9_npz,
        prediction_sidecar_path=fold9_sidecar,
        prediction_npz_sha256=_raw_hash(fold9_npz),
        prediction_sidecar_sha256=_raw_hash(fold9_sidecar),
        prediction_artifact_sha256="sha256:" + "4" * 64,
        prediction_alignment_sha256="sha256:" + "5" * 64,
        decision_path=decision,
        decision_file_sha256=_raw_hash(decision),
        decision_artifact_sha256="sha256:" + "6" * 64,
        independent_fit_sha256="sha256:" + "7" * 64,
    )
    refit_bundle = SimpleNamespace(
        artifact_sha256=refit_artifact_sha256,
        protocol_hash=protocol.protocol_hash,
        manifest_sha256=manifest_sha256,
        normalization_sha256=normalization_sha256,
        label_order=protocol.label_order,
        members=(refit_member,),
    )
    calibration_bundle = SimpleNamespace(
        artifact_sha256=calibration_artifact_sha256,
        refit_bundle_sha256=refit_artifact_sha256,
        protocol_hash=protocol.protocol_hash,
        manifest_sha256=manifest_sha256,
        normalization_sha256=normalization_sha256,
        label_order=protocol.label_order,
        members=(calibration_member,),
    )

    calls: list[tuple[str, bool]] = []

    def load_refits(
        path: str | Path, *, protocol: ExperimentProtocol, verify_sources: bool
    ) -> object:
        assert Path(path).resolve() == refit_bundle_path
        assert protocol.protocol_hash == refit_bundle.protocol_hash
        assert verify_sources is True
        calls.append(("refits", verify_sources))
        return refit_bundle

    def load_calibrations(
        path: str | Path, *, protocol: ExperimentProtocol, verify_sources: bool
    ) -> object:
        assert Path(path).resolve() == calibration_bundle_path
        assert protocol.protocol_hash == calibration_bundle.protocol_hash
        assert verify_sources is False
        calls.append(("calibrations", verify_sources))
        return calibration_bundle

    def verify_calibration_sources(bundle: object, *, protocol: ExperimentProtocol) -> None:
        assert bundle is calibration_bundle
        assert protocol.protocol_hash == calibration_bundle.protocol_hash
        calls.append(("calibration_sources", True))

    def fixed_policy(
        bundle: object, member_id: str, *, target_coverage: float
    ) -> dict[str, object]:
        assert bundle is calibration_bundle
        assert member_id == DEMO_MEMBER_ID
        assert target_coverage == DEMO_TARGET_COVERAGE
        assert ("calibration_sources", True) in calls
        calls.append(("policy", True))
        return {
            "schema_version": 1,
            "label_order": list(protocol.label_order),
            "calibration": {"method": "temperature_scaling", "temperature": 1.25},
            "classification_thresholds": [0.1, 0.2, 0.3, 0.4, 0.5],
            "gate": {
                "method": "mean_normalized_binary_entropy",
                "uncertainty_threshold": 0.55,
            },
            "provenance": {
                "dataset_version": PTBXL_VERSION,
                "protocol_hash": protocol.protocol_hash,
                "manifest_hash": manifest_sha256.removeprefix("sha256:"),
                "checkpoint_config_hash": config_hash,
                "checkpoint_sha256": checkpoint_sha256.removeprefix("sha256:"),
                "resolved_config_sha256": config_file_sha256.removeprefix("sha256:"),
                "normalization_sha256": normalization_sha256.removeprefix("sha256:"),
                "calibration_folds": list(CALIBRATION_FOLDS),
            },
        }

    monkeypatch.setattr(demo_materialization, "load_refit_bundle", load_refits)
    monkeypatch.setattr(
        demo_materialization, "load_calibration_bundle", load_calibrations
    )
    monkeypatch.setattr(
        demo_materialization, "materialize_demo_policy_payload", fixed_policy
    )
    monkeypatch.setattr(
        demo_materialization,
        "_verify_calibration_sources_for_demo",
        verify_calibration_sources,
    )
    return refit_bundle_path, calibration_bundle_path, tmp_path / "demo", protocol


def test_materialization_is_fixed_label_free_self_hashed_and_immutable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    refit_path, calibration_path, output, protocol = _fake_release(
        tmp_path, monkeypatch
    )

    result = materialize_demo(
        refit_bundle_path=refit_path,
        calibration_bundle_path=calibration_path,
        output_directory=output,
        protocol=protocol,
    )

    assert result.policy_path.name == DEMO_POLICY_FILENAME
    assert result.examples_path.name == DEMO_EXAMPLES_FILENAME
    assert result.binding_path.name == DEMO_BINDING_FILENAME
    FrozenDecisionPolicy.load(result.policy_path)
    examples = json.loads(result.examples_path.read_text(encoding="utf-8"))["examples"]
    assert len(examples) == DEMO_EXAMPLE_COUNT
    assert [item["id"] for item in examples] == [
        "ptbxl-f8-00012",
        "ptbxl-f8-00043",
        "ptbxl-f8-00053",
        "ptbxl-f8-00071",
        "ptbxl-f8-00074",
    ]
    assert all("patient" not in item for item in examples)

    binding = json.loads(result.binding_path.read_text(encoding="utf-8"))
    stored_hash = binding.pop("artifact_sha256")
    assert canonical_sha256(binding) == stored_hash == result.binding_artifact_sha256
    assert binding["selection"]["member_id"] == DEMO_MEMBER_ID
    assert binding["selection"]["target_coverage"] == DEMO_TARGET_COVERAGE
    assert binding["selection"]["fold10_predictions_read"] is False
    assert binding["selection"]["fold10_performance_used"] is False
    assert binding["examples"]["selection_fold"] == [8]
    assert binding["examples"]["diagnostic_target_columns_read"] is False
    assert len(binding["examples"]["records"]) == DEMO_EXAMPLE_COUNT
    assert binding["sources"]["fold9_decision"]["file_sha256"].startswith(
        "sha256:"
    )

    repeated = materialize_demo(
        refit_bundle_path=refit_path,
        calibration_bundle_path=calibration_path,
        output_directory=output,
        protocol=protocol,
    )
    assert repeated == result
    assert load_and_verify_demo_binding(result.binding_path, protocol=protocol) == result

    result.examples_path.write_text('{"examples": []}\n', encoding="utf-8")
    with pytest.raises(DemoMaterializationError, match="examples differ"):
        load_and_verify_demo_binding(result.binding_path, protocol=protocol)
    with pytest.raises(DemoMaterializationError, match="immutable demo artifact differs"):
        materialize_demo(
            refit_bundle_path=refit_path,
            calibration_bundle_path=calibration_path,
            output_directory=output,
            protocol=protocol,
        )


def test_materialization_refuses_release_directory_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    refit_path, calibration_path, _, protocol = _fake_release(tmp_path, monkeypatch)

    with pytest.raises(DemoMaterializationError, match="outside"):
        materialize_demo(
            refit_bundle_path=refit_path,
            calibration_bundle_path=calibration_path,
            output_directory=refit_path.parent / "demo",
            protocol=protocol,
        )
