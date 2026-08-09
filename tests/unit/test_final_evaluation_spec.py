from __future__ import annotations

import hashlib
import json
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

import ecg_trust.final_evaluation_spec as spec_module
import ecg_trust.release_gates as release_gates
from ecg_trust.final_evaluation_spec import (
    CANONICAL_COVERAGE_TARGETS,
    FinalEvaluationSpec,
    FinalEvaluationSpecError,
    FinalEvaluationSpecIntegrityError,
    canonical_sha256,
    create_final_evaluation_spec,
    freeze_final_evaluation_spec,
    load_final_evaluation_spec,
    save_final_evaluation_spec,
)
from ecg_trust.protocol import LABEL_ORDER, ExperimentProtocol
from ecg_trust.subgroup_artifact import SubgroupArtifact
from scripts.freeze_final_evaluation import build_parser

_REAL_CAPTURE_RUNTIME = spec_module._capture_runtime_envelope


def _file_hash(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class _FakeRefitBundle:
    protocol_hash: str
    manifest_sha256: str
    normalization_sha256: str
    label_order: tuple[str, ...]
    members: tuple[object, ...]
    artifact_sha256: str


@dataclass
class _Fixture:
    protocol: ExperimentProtocol
    protocol_path: Path
    refit_path: Path
    subgroup_path: Path
    deviation_path: Path
    project_root: Path
    bundle: _FakeRefitBundle
    subgroup: SubgroupArtifact
    runtime: dict[str, object]
    refit_calls: list[tuple[Path, bool]]
    subgroup_calls: list[tuple[Path, str, bool]]


def _subgroup(protocol: ExperimentProtocol, manifest_hash: str) -> SubgroupArtifact:
    artifact = SubgroupArtifact(
        dataset_name=protocol.dataset_name,
        dataset_version=protocol.dataset_version,
        protocol_hash=protocol.protocol_hash,
        manifest_path=Path("synthetic-manifest.parquet").resolve(),
        manifest_sha256=manifest_hash,
        ecg_id=(101, 102, 103),
        patient_id=(201, 202, 203),
        sex=("male", "female", "unknown"),
        age_band=("<40", "40-59", "80+"),
        group_counts=(
            {
                "attribute": "sex",
                "group": "male",
                "records": 1,
                "patients": 1,
            },
            {
                "attribute": "sex",
                "group": "female",
                "records": 1,
                "patients": 1,
            },
            {
                "attribute": "sex",
                "group": "unknown",
                "records": 1,
                "patients": 1,
            },
            {"attribute": "age_band", "group": "<40", "records": 1, "patients": 1},
            {"attribute": "age_band", "group": "40-59", "records": 1, "patients": 1},
            {"attribute": "age_band", "group": "60-79", "records": 0, "patients": 0},
            {"attribute": "age_band", "group": "80+", "records": 1, "patients": 1},
            {"attribute": "age_band", "group": "unknown", "records": 0, "patients": 0},
        ),
    )
    return artifact.with_integrity()


def _longitudinal_subgroup(
    protocol: ExperimentProtocol, manifest_hash: str
) -> SubgroupArtifact:
    """One patient legitimately spans two age bands across two ECGs."""

    artifact = SubgroupArtifact(
        dataset_name=protocol.dataset_name,
        dataset_version=protocol.dataset_version,
        protocol_hash=protocol.protocol_hash,
        manifest_path=Path("synthetic-manifest.parquet").resolve(),
        manifest_sha256=manifest_hash,
        ecg_id=(101, 102),
        patient_id=(201, 201),
        sex=("male", "male"),
        age_band=("<40", "40-59"),
        group_counts=(
            {"attribute": "sex", "group": "male", "records": 2, "patients": 1},
            {"attribute": "sex", "group": "female", "records": 0, "patients": 0},
            {"attribute": "sex", "group": "unknown", "records": 0, "patients": 0},
            {"attribute": "age_band", "group": "<40", "records": 1, "patients": 1},
            {
                "attribute": "age_band",
                "group": "40-59",
                "records": 1,
                "patients": 1,
            },
            {
                "attribute": "age_band",
                "group": "60-79",
                "records": 0,
                "patients": 0,
            },
            {"attribute": "age_band", "group": "80+", "records": 0, "patients": 0},
            {
                "attribute": "age_band",
                "group": "unknown",
                "records": 0,
                "patients": 0,
            },
        ),
    )
    return artifact.with_integrity()


def _runtime(project_root: Path) -> dict[str, object]:
    lock_path = project_root / "uv.lock"
    return {
        "project_root": str(project_root.resolve()),
        "git": {"revision": "a" * 40, "dirty": False},
        "dependency_lock": {
            "path": str(lock_path.resolve()),
            "sha256": _file_hash(lock_path),
        },
        "software": {
            "python": "3.12.13",
            "ecg_trust": "0.1.0",
            "torch": "2.13.0+cu130",
            "cuda_runtime": "13.0",
            "cudnn": 91100,
            "nvidia_driver": "596.49",
            "installed_environment": {
                "distribution_count": 100,
                "distributions_sha256": "sha256:" + "e" * 64,
                "core_packages": {
                    "numpy": "2.5.1",
                    "scipy": "1.18.0",
                    "scikit-learn": "1.9.0",
                    "pandas": "3.0.5",
                    "wfdb": "4.3.1",
                    "pyarrow": "25.0.0",
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


@pytest.fixture
def frozen_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> _Fixture:
    protocol = ExperimentProtocol.canonical()
    protocol_path = Path("configs/protocol.yaml").resolve()
    refit_path = tmp_path / "sealed-refit-bundle.json"
    refit_path.write_text('{"sealed":true}\n', encoding="utf-8")
    subgroup_path = tmp_path / "subgroups-v1.json"
    subgroup_path.write_text('{"subgroups":true}\n', encoding="utf-8")
    deviation_path = tmp_path / "PROTOCOL_DEVIATIONS.md"
    deviation_path.write_text("# Disclosed deviations\n\nDEV-001.\n", encoding="utf-8")
    project_root = tmp_path / "repo"
    project_root.mkdir()
    (project_root / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    manifest_hash = "sha256:" + "b" * 64
    bundle = _FakeRefitBundle(
        protocol_hash=protocol.protocol_hash,
        manifest_sha256=manifest_hash,
        normalization_sha256="sha256:" + "c" * 64,
        label_order=LABEL_ORDER,
        members=tuple(SimpleNamespace(member_id=index) for index in range(6)),
        artifact_sha256="sha256:" + "d" * 64,
    )
    subgroup = _subgroup(protocol, manifest_hash)
    runtime = _runtime(project_root)
    refit_calls: list[tuple[Path, bool]] = []
    subgroup_calls: list[tuple[Path, str, bool]] = []

    def fake_refit_loader(
        path: str | Path,
        *,
        protocol: ExperimentProtocol,
        verify_sources: bool = True,
    ) -> _FakeRefitBundle:
        assert protocol.protocol_hash == bundle.protocol_hash
        refit_calls.append((Path(path).resolve(), verify_sources))
        return bundle

    def fake_subgroup_loader(
        path: str | Path,
        *,
        protocol: ExperimentProtocol,
        expected_manifest_sha256: str | None = None,
        verify_source: bool = True,
    ) -> SubgroupArtifact:
        assert protocol.protocol_hash == bundle.protocol_hash
        assert expected_manifest_sha256 is not None
        subgroup_calls.append(
            (Path(path).resolve(), expected_manifest_sha256, verify_source)
        )
        return subgroup

    monkeypatch.setattr(release_gates, "load_refit_bundle", fake_refit_loader)
    monkeypatch.setattr(spec_module, "load_subgroup_artifact", fake_subgroup_loader)
    monkeypatch.setattr(
        spec_module,
        "_capture_runtime_envelope",
        lambda project_root, requested_device: runtime,
    )
    return _Fixture(
        protocol=protocol,
        protocol_path=protocol_path,
        refit_path=refit_path,
        subgroup_path=subgroup_path,
        deviation_path=deviation_path,
        project_root=project_root,
        bundle=bundle,
        subgroup=subgroup,
        runtime=runtime,
        refit_calls=refit_calls,
        subgroup_calls=subgroup_calls,
    )


def _create(inputs: _Fixture) -> FinalEvaluationSpec:
    return create_final_evaluation_spec(
        protocol=inputs.protocol,
        protocol_path=inputs.protocol_path,
        refit_bundle_path=inputs.refit_path,
        subgroup_artifact_path=inputs.subgroup_path,
        protocol_deviations_path=inputs.deviation_path,
        project_root=inputs.project_root,
        device="cuda:0",
    )


def test_spec_is_deterministic_strict_and_reverifies_sources(
    frozen_inputs: _Fixture,
    tmp_path: Path,
) -> None:
    first = _create(frozen_inputs)
    second = _create(frozen_inputs)
    assert first.to_payload() == second.to_payload()
    assert first.artifact_sha256 == second.artifact_sha256
    evaluation = cast(dict[str, object], first.payload["final_evaluation"])
    calibration = cast(dict[str, object], first.payload["calibration_policy"])
    assert calibration["coverage_targets"] == list(CANONICAL_COVERAGE_TARGETS)
    assert evaluation == {
        "final_folds": [10],
        "patient_resampling": "patient_cluster_percentile_bootstrap",
        "bootstrap_resamples": 1_000,
        "bootstrap_base_seed": 20_260_808,
        "bootstrap_confidence": 0.95,
        "bootstrap_minimum_valid": 500,
        "bootstrap_seed_strategy": "base_plus_model_seed",
        "ece_bins": 15,
        "minimum_group_samples": 30,
        "minimum_group_patients": 20,
        "retuning_allowed": False,
    }
    output = tmp_path / "final-evaluation-spec.json"
    saved = save_final_evaluation_spec(first, output)
    loaded = load_final_evaluation_spec(
        output,
        protocol=frozen_inputs.protocol,
        verify_sources=True,
        verify_runtime=True,
    )
    assert loaded.to_payload() == first.to_payload()
    assert saved.path == output.resolve()
    assert frozen_inputs.refit_calls == [
        (frozen_inputs.refit_path.resolve(), True),
        (frozen_inputs.refit_path.resolve(), True),
        (frozen_inputs.refit_path.resolve(), True),
    ]
    assert all(call[2] is True for call in frozen_inputs.subgroup_calls)
    assert all(
        call[1] == frozen_inputs.bundle.manifest_sha256
        for call in frozen_inputs.subgroup_calls
    )


def test_spec_allows_longitudinal_patient_to_span_age_bands(
    frozen_inputs: _Fixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    longitudinal = _longitudinal_subgroup(
        frozen_inputs.protocol, frozen_inputs.bundle.manifest_sha256
    )

    def load_longitudinal_subgroups(
        path: str | Path,
        *,
        protocol: ExperimentProtocol,
        expected_manifest_sha256: str,
        verify_source: bool,
    ) -> SubgroupArtifact:
        assert Path(path).resolve() == frozen_inputs.subgroup_path.resolve()
        assert protocol.protocol_hash == frozen_inputs.protocol.protocol_hash
        assert expected_manifest_sha256 == frozen_inputs.bundle.manifest_sha256
        assert verify_source is True
        return longitudinal

    monkeypatch.setattr(
        spec_module, "load_subgroup_artifact", load_longitudinal_subgroups
    )
    spec = _create(frozen_inputs)
    subgroup = cast(dict[str, object], spec.payload["subgroup_artifact"])
    counts = cast(dict[str, object], subgroup["counts"])
    assert counts["record_count"] == 2
    assert counts["patient_count"] == 1


def test_save_rejects_overwrite(frozen_inputs: _Fixture, tmp_path: Path) -> None:
    spec = _create(frozen_inputs)
    output = tmp_path / "immutable.json"
    save_final_evaluation_spec(spec, output)
    with pytest.raises(FileExistsError, match="already exists"):
        save_final_evaluation_spec(spec, output)


def test_freeze_captures_clean_runtime_once_before_immutable_write(
    frozen_inputs: _Fixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[Path, str]] = []

    def capture(project_root: Path, requested_device: str) -> dict[str, object]:
        calls.append((project_root, requested_device))
        if len(calls) > 1:
            raise AssertionError("runtime must not be recaptured after writing")
        return frozen_inputs.runtime

    monkeypatch.setattr(spec_module, "_capture_runtime_envelope", capture)
    output = tmp_path / "one-shot-final-evaluation-spec.json"
    frozen = freeze_final_evaluation_spec(
        output,
        protocol=frozen_inputs.protocol,
        protocol_path=frozen_inputs.protocol_path,
        refit_bundle_path=frozen_inputs.refit_path,
        subgroup_artifact_path=frozen_inputs.subgroup_path,
        protocol_deviations_path=frozen_inputs.deviation_path,
        project_root=frozen_inputs.project_root,
        device="cuda:0",
    )
    assert frozen.path == output.resolve()
    assert calls == [(frozen_inputs.project_root.resolve(), "cuda:0")]


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    [
        ("final_evaluation", "bootstrap_resamples", 999, "final evaluation policy"),
        ("final_evaluation", "bootstrap_minimum_valid", 499, "final evaluation policy"),
        ("final_evaluation", "ece_bins", 16, "final evaluation policy"),
        ("calibration_policy", "coverage_targets", [1.0, 0.8], "calibration policy"),
        ("report_contract", "retuning_allowed", True, "report contract"),
    ],
)
def test_loader_rejects_alternative_scientific_values_even_when_rehashed(
    frozen_inputs: _Fixture,
    tmp_path: Path,
    section: str,
    field: str,
    value: object,
    message: str,
) -> None:
    payload = _create(frozen_inputs).to_payload()
    cast(dict[str, object], payload[section])[field] = value
    unhashed = dict(payload)
    del unhashed["artifact_sha256"]
    payload["artifact_sha256"] = canonical_sha256(unhashed)
    path = tmp_path / f"tampered-{field}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(FinalEvaluationSpecIntegrityError, match=message):
        load_final_evaluation_spec(
            path,
            protocol=frozen_inputs.protocol,
            verify_sources=False,
            verify_runtime=False,
        )


def test_loader_rejects_unknown_keys_and_nan(
    frozen_inputs: _Fixture,
    tmp_path: Path,
) -> None:
    payload = _create(frozen_inputs).to_payload()
    payload["unknown"] = "not allowed"
    unhashed = dict(payload)
    del unhashed["artifact_sha256"]
    payload["artifact_sha256"] = canonical_sha256(unhashed)
    path = tmp_path / "unknown.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(FinalEvaluationSpecIntegrityError, match="unknown"):
        load_final_evaluation_spec(
            path,
            protocol=frozen_inputs.protocol,
            verify_sources=False,
            verify_runtime=False,
        )
    with pytest.raises(FinalEvaluationSpecError, match="finite JSON"):
        canonical_sha256({"not_finite": float("nan")})


@pytest.mark.parametrize("source_name", ["refit_path", "subgroup_path", "deviation_path"])
def test_loader_rejects_bound_source_drift(
    frozen_inputs: _Fixture,
    tmp_path: Path,
    source_name: str,
) -> None:
    output = tmp_path / f"drift-{source_name}.json"
    save_final_evaluation_spec(_create(frozen_inputs), output)
    source = cast(Path, getattr(frozen_inputs, source_name))
    source.write_text(source.read_text(encoding="utf-8") + "drift\n", encoding="utf-8")
    with pytest.raises(FinalEvaluationSpecIntegrityError, match="changed after freeze"):
        load_final_evaluation_spec(
            output,
            protocol=frozen_inputs.protocol,
            verify_sources=True,
            verify_runtime=False,
        )


def test_runtime_drift_is_rejected(
    frozen_inputs: _Fixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "runtime-drift.json"
    save_final_evaluation_spec(_create(frozen_inputs), output)
    changed = json.loads(json.dumps(frozen_inputs.runtime))
    changed["hardware"]["device_name"] = "Different GPU"
    monkeypatch.setattr(
        spec_module,
        "_capture_runtime_envelope",
        lambda project_root, requested_device: changed,
    )
    with pytest.raises(FinalEvaluationSpecIntegrityError, match="runtime differs"):
        load_final_evaluation_spec(
            output,
            protocol=frozen_inputs.protocol,
            verify_sources=False,
            verify_runtime=True,
        )


def test_loader_rejects_rehashed_nondeterministic_tf32_policy(
    frozen_inputs: _Fixture,
    tmp_path: Path,
) -> None:
    payload = _create(frozen_inputs).to_payload()
    runtime = cast(dict[str, object], payload["runtime_envelope"])
    policy = cast(dict[str, object], runtime["policy"])
    policy["cudnn_allow_tf32"] = True
    unhashed = dict(payload)
    del unhashed["artifact_sha256"]
    payload["artifact_sha256"] = canonical_sha256(unhashed)
    path = tmp_path / "nondeterministic-policy.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(FinalEvaluationSpecIntegrityError, match="policy"):
        load_final_evaluation_spec(
            path,
            protocol=frozen_inputs.protocol,
            verify_sources=False,
            verify_runtime=False,
        )


def test_git_dirty_and_unavailable_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path.resolve()

    def dirty_git(root: Path, *arguments: str) -> str:
        del root
        if arguments == ("rev-parse", "--show-toplevel"):
            return str(project)
        if arguments == ("rev-parse", "HEAD"):
            return "a" * 40
        return " M src/file.py"

    monkeypatch.setattr(spec_module, "_run_git", dirty_git)
    with pytest.raises(FinalEvaluationSpecError, match="clean committed"):
        spec_module._capture_git_envelope(project)
    monkeypatch.setattr(
        spec_module,
        "_run_git",
        lambda root, *arguments: (_ for _ in ()).throw(
            FinalEvaluationSpecError("Git state is unavailable")
        ),
    )
    with pytest.raises(FinalEvaluationSpecError, match="unavailable"):
        spec_module._capture_git_envelope(project)


def test_real_git_subprocess_probe_matches_repository_shape() -> None:
    project = Path.cwd().resolve()
    assert Path(
        spec_module._run_git(project, "rev-parse", "--show-toplevel")
    ).resolve() == project
    revision = spec_module._run_git(project, "rev-parse", "HEAD")
    assert len(revision) in {40, 64}
    status = spec_module._run_git(
        project,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    if status:
        with pytest.raises(FinalEvaluationSpecError, match="clean committed"):
            spec_module._capture_git_envelope(project)
    else:
        assert spec_module._capture_git_envelope(project) == {
            "revision": revision.casefold(),
            "dirty": False,
        }


def test_runtime_requires_explicit_cuda_and_bf16(
    frozen_inputs: _Fixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(FinalEvaluationSpecError, match="auto is forbidden"):
        _REAL_CAPTURE_RUNTIME(frozen_inputs.project_root, "auto")
    with pytest.raises(FinalEvaluationSpecError, match="indexed CUDA"):
        _REAL_CAPTURE_RUNTIME(frozen_inputs.project_root, "cpu")
    with pytest.raises(FinalEvaluationSpecError, match="indexed CUDA"):
        _REAL_CAPTURE_RUNTIME(frozen_inputs.project_root, "cuda")
    monkeypatch.setattr(spec_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(spec_module.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(spec_module.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        spec_module.torch.cuda,
        "device",
        lambda index: nullcontext(),
    )
    monkeypatch.setattr(
        spec_module.torch.cuda,
        "is_bf16_supported",
        lambda including_emulation=False: False,
    )
    with pytest.raises(FinalEvaluationSpecError, match="support BF16"):
        _REAL_CAPTURE_RUNTIME(frozen_inputs.project_root, "cuda:0")
    monkeypatch.setattr(
        spec_module.torch.cuda,
        "is_bf16_supported",
        lambda including_emulation=False: True,
    )
    monkeypatch.setattr(
        spec_module.torch.cuda,
        "get_device_properties",
        lambda index: SimpleNamespace(
            total_memory=12 * 1024**3,
            pci_domain_id=0,
            pci_bus_id=1,
            pci_device_id=0,
        ),
    )
    monkeypatch.setattr(
        spec_module.torch.cuda,
        "get_device_name",
        lambda index: "Synthetic CUDA GPU",
    )
    monkeypatch.setattr(spec_module.torch.backends.cudnn, "version", lambda: 91100)
    with pytest.raises(FinalEvaluationSpecError, match="device UUID"):
        _REAL_CAPTURE_RUNTIME(frozen_inputs.project_root, "cuda:0")


def test_runtime_accepts_indexed_cuda(
    frozen_inputs: _Fixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested_device = "cuda:0"
    monkeypatch.setattr(
        spec_module,
        "_capture_git_envelope",
        lambda project_root: {"revision": "a" * 40, "dirty": False},
    )
    monkeypatch.setattr(spec_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(spec_module.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(spec_module.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        spec_module.torch.cuda,
        "device",
        lambda index: nullcontext(),
    )
    monkeypatch.setattr(
        spec_module.torch.cuda,
        "is_bf16_supported",
        lambda including_emulation=False: True,
    )
    monkeypatch.setattr(
        spec_module.torch.cuda,
        "get_device_properties",
        lambda index: SimpleNamespace(
            total_memory=12 * 1024**3,
            uuid="aba61180-7244-4ad9-bdf5-60c764cb1d59",
            pci_domain_id=0,
            pci_bus_id=1,
            pci_device_id=0,
        ),
    )
    monkeypatch.setattr(
        spec_module.torch.cuda,
        "get_device_name",
        lambda index: "Synthetic CUDA GPU",
    )
    monkeypatch.setattr(
        spec_module.torch.cuda,
        "get_device_capability",
        lambda index: (12, 0),
    )
    monkeypatch.setattr(spec_module.torch.version, "cuda", "13.0")
    monkeypatch.setattr(spec_module.torch.backends.cudnn, "version", lambda: 91100)
    monkeypatch.setattr(
        spec_module,
        "_capture_nvidia_driver",
        lambda gpu_uuid: "596.49",
    )
    runtime = _REAL_CAPTURE_RUNTIME(frozen_inputs.project_root, requested_device)
    hardware = cast(dict[str, object], runtime["hardware"])
    assert hardware["requested_device"] == requested_device
    assert hardware["resolved_device"] == "cuda:0"
    assert hardware["device_uuid"] == (
        "GPU-aba61180-7244-4ad9-bdf5-60c764cb1d59"
    )
    assert {
        field: hardware[field]
        for field in ("pci_domain_id", "pci_bus_id", "pci_device_id")
    } == {"pci_domain_id": 0, "pci_bus_id": 1, "pci_device_id": 0}
    software = cast(dict[str, object], runtime["software"])
    assert software["nvidia_driver"] == "596.49"
    policy = cast(dict[str, object], runtime["policy"])
    assert policy["deterministic_algorithms_enabled"] is True
    assert policy["cuda_matmul_allow_tf32"] is False
    assert policy["cudnn_allow_tf32"] is False
    assert policy["cublas_workspace_config"] == ":4096:8"


def test_bf16_probe_enters_selected_cuda_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = [0]

    class _DeviceContext:
        def __init__(self, index: int) -> None:
            self.index = index
            self.previous = selected[0]

        def __enter__(self) -> None:
            selected[0] = self.index

        def __exit__(self, *args: object) -> None:
            selected[0] = self.previous

    monkeypatch.setattr(
        spec_module.torch.cuda,
        "device",
        lambda index: _DeviceContext(index),
    )
    monkeypatch.setattr(
        spec_module.torch.cuda,
        "current_device",
        lambda: selected[0],
    )
    monkeypatch.setattr(
        spec_module.torch.cuda,
        "is_bf16_supported",
        lambda including_emulation=False: (
            selected[0] == 1 and including_emulation is False
        ),
    )
    assert spec_module._selected_device_supports_bf16(1) is True
    assert selected == [0]


def test_nvidia_driver_is_bound_to_selected_gpu_uuid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        spec_module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout=(
                "GPU-00000000-0000-0000-0000-000000000000, 550.1\n"
                "GPU-aba61180-7244-4ad9-bdf5-60c764cb1d59, 596.49\n"
            )
        ),
    )
    assert (
        spec_module._capture_nvidia_driver(
            "GPU-aba61180-7244-4ad9-bdf5-60c764cb1d59"
        )
        == "596.49"
    )


def test_installed_environment_hash_is_deterministic_and_binds_core_packages() -> None:
    first = spec_module._capture_installed_environment()
    second = spec_module._capture_installed_environment()
    assert first == second
    assert cast(int, first["distribution_count"]) >= 6
    assert cast(str, first["distributions_sha256"]).startswith("sha256:")
    core = cast(dict[str, str], first["core_packages"])
    assert set(core) == {
        "numpy",
        "scipy",
        "scikit-learn",
        "pandas",
        "wfdb",
        "pyarrow",
    }
    assert all(core.values())


def test_cli_help_exposes_only_source_runtime_and_output_controls() -> None:
    help_text = build_parser().format_help()
    for option in (
        "--protocol",
        "--refit-bundle",
        "--subgroups",
        "--protocol-deviations",
        "--project-root",
        "--device",
        "--output",
    ):
        assert option in help_text
    assert "--bootstrap" not in help_text
    assert "--coverage" not in help_text
