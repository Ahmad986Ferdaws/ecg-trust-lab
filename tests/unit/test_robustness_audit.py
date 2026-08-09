from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch

import ecg_trust.post_evaluation as post
import ecg_trust.robustness_audit as audit
import scripts.audit_robustness as robustness_cli
from ecg_trust.audit_runtime import AlignedAuditInference, AuditMemberRuntime, CompletedAuditRuntime
from ecg_trust.post_evaluation import PostEvaluationSpec


def _hash(character: str = "a") -> str:
    return "sha256:" + character * 64


def _spec(tmp_path: Path, *, member_ids: tuple[str, ...] | None = None) -> PostEvaluationSpec:
    members = member_ids or tuple(
        f"{architecture}-seed{seed}"
        for architecture in post.EXPECTED_ARCHITECTURES
        for seed in post.EXPECTED_SEEDS
    )
    root = (tmp_path / "runs" / "post_evaluation" / "comparison").resolve()
    body: dict[str, object] = {
        "members": [
            {"member_id": member_id, "source": _hash(str(index % 10))}
            for index, member_id in enumerate(members)
        ],
        "audit_protocols": {"robustness": post._robustness_protocol()},
        "output_contract": {
            "root": str(root),
            "artifacts": {"robustness_manifest": str(root / "robustness" / "manifest.json")},
        },
    }
    payload = {**body, "artifact_sha256": post.canonical_sha256(body)}
    path = root / "audit_spec.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return PostEvaluationSpec(
        path=path,
        artifact_sha256=cast(str, payload["artifact_sha256"]),
        _canonical_payload=json.dumps(payload, separators=(",", ":"), sort_keys=True),
    )


class _Gate:
    def __init__(self, target: float = 0.8, cutoff: float = 0.8) -> None:
        self.target_coverage = target
        self.maximum_entropy = cutoff

    def to_dict(self) -> dict[str, object]:
        return {
            "target_coverage": self.target_coverage,
            "maximum_entropy": self.maximum_entropy,
            "calibration_coverage": self.target_coverage,
            "selected_count": 8,
            "calibration_count": 10,
        }


def _member(member_id: str = "member-1", *, n_samples: int = 20) -> AuditMemberRuntime:
    rows = np.arange(n_samples, dtype=np.int64)
    targets = np.stack(
        [((rows + index) % (index + 2) == 0).astype(np.int8) for index in range(5)],
        axis=1,
    )
    logits = np.linspace(-2.0, 2.0, n_samples * 5, dtype=np.float64).reshape(n_samples, 5)
    sealed = SimpleNamespace(
        ecg_id=rows + 100,
        patient_id=(rows // 2) + 500,
        strat_fold=np.full(n_samples, 10, dtype=np.int8),
        targets=targets,
        raw_logits=logits,
    )
    decisions = SimpleNamespace(
        temperature_scaling=SimpleNamespace(temperature=1.25),
        threshold_optimization=SimpleNamespace(thresholds=(0.5,) * 5),
        coverage_gates=(_Gate(),),
    )
    return cast(
        AuditMemberRuntime,
        SimpleNamespace(member_id=member_id, sealed_prediction=sealed, decisions=decisions),
    )


def _inference(member: AuditMemberRuntime, *, logit_delta: float = 0.0) -> AlignedAuditInference:
    sealed = member.sealed_prediction
    return AlignedAuditInference(
        member_id=member.member_id,
        ecg_id=sealed.ecg_id,
        patient_id=sealed.patient_id,
        strat_fold=sealed.strat_fold,
        targets=sealed.targets,
        raw_logits=sealed.raw_logits + logit_delta,
    )


def test_case_expansion_is_the_exact_frozen_41_case_grid(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    cases = audit.expand_robustness_cases(spec)

    assert len(cases) == 41
    assert cases[0].to_dict() == {
        "case_id": "clean",
        "corruption": "clean",
        "parameters": {},
    }
    assert len({case.case_id for case in cases}) == 41
    assert sum(case.corruption == "lead_dropout" for case in cases) == 14
    assert cases[-1].case_id == "lead-permutation-reverse-all"
    _, robustness_root = audit._audit_paths(spec)
    assert audit.preflight_robustness_artifact_paths(spec) == cases
    npz_paths = {
        audit._case_npz_path(robustness_root, "resnet1d-seed2026", case.case_id)
        for case in cases
    }
    sidecar_paths = {path.with_suffix(".json") for path in npz_paths}
    assert len(npz_paths) == 41
    assert len(sidecar_paths) == 41
    assert len(npz_paths | sidecar_paths) == 82
    assert all(robustness_root in path.parents for path in npz_paths | sidecar_paths)
    assert any(path.name == "baseline-wander-0.05.npz" for path in npz_paths)


def test_path_collision_fails_before_runtime_access_or_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = _spec(tmp_path)
    _, robustness_root = audit._audit_paths(spec)
    original = audit._case_npz_path

    def colliding_path(root: Path, member_id: str, case_id: str) -> Path:
        if case_id in {"baseline-wander-0.05", "baseline-wander-0.10"}:
            return root / "member_cases" / member_id / "decimal-collision.npz"
        return original(root, member_id, case_id)

    monkeypatch.setattr(audit, "_case_npz_path", colliding_path)
    runtime_accessed = False

    class UntouchedRuntime:
        @property
        def members(self) -> tuple[object, ...]:
            nonlocal runtime_accessed
            runtime_accessed = True
            raise AssertionError("runtime must not be accessed before path preflight")

    with pytest.raises(
        audit.RobustnessAuditIntegrityError,
        match=(
            r"artifact path collision: .*baseline-wander-0\.05/npz and "
            r".*baseline-wander-0\.10/npz"
        ),
    ):
        audit.run_robustness_audit(
            spec=spec,
            runtime=cast(CompletedAuditRuntime, UntouchedRuntime()),
        )

    assert runtime_accessed is False
    assert not robustness_root.exists()


def test_cli_path_preflight_precedes_runtime_loading(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    events: list[str] = []
    spec = cast(PostEvaluationSpec, SimpleNamespace())
    monkeypatch.setattr(robustness_cli, "load_protocol", lambda unused: object())
    monkeypatch.setattr(
        robustness_cli,
        "load_post_evaluation_spec",
        lambda *args, **kwargs: spec,
    )

    def reject_paths(unused: PostEvaluationSpec) -> tuple[object, ...]:
        events.append("path_preflight")
        raise audit.RobustnessAuditIntegrityError("synthetic path collision")

    def forbidden_runtime_load(**kwargs: object) -> object:
        events.append("runtime_load")
        raise AssertionError("runtime loading must follow path preflight")

    monkeypatch.setattr(
        robustness_cli, "preflight_robustness_artifact_paths", reject_paths
    )
    monkeypatch.setattr(
        robustness_cli, "load_completed_audit_runtime", forbidden_runtime_load
    )

    assert robustness_cli.main([]) == 1
    assert events == ["path_preflight"]
    assert "synthetic path collision" in capsys.readouterr().err


def test_decimal_case_artifacts_are_distinct_and_resumable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    member_id = "member-1"
    spec = _spec(tmp_path, member_ids=(member_id,))
    cases = {case.case_id: case for case in audit.expand_robustness_cases(spec)}
    member = _member(member_id)
    cast(Any, member).infer_logits = (
        lambda *, transform=None: _inference(member, logit_delta=0.1)
    )
    _, robustness_root = audit._audit_paths(spec)
    monkeypatch.setattr(
        audit, "_robustness_settings", lambda unused: (1, 3, 0.95, 2)
    )
    clean, _ = audit._load_or_create_case(
        spec=spec,
        member=member,
        case=cases["clean"],
        clean=None,
        robustness_root=robustness_root,
        base_seed=1,
        bootstrap_resamples=3,
        bootstrap_confidence=0.95,
        bootstrap_base_seed=2,
    )
    first, first_created = audit._load_or_create_case(
        spec=spec,
        member=member,
        case=cases["baseline-wander-0.05"],
        clean=clean,
        robustness_root=robustness_root,
        base_seed=1,
        bootstrap_resamples=3,
        bootstrap_confidence=0.95,
        bootstrap_base_seed=2,
    )
    second, second_created = audit._load_or_create_case(
        spec=spec,
        member=member,
        case=cases["baseline-wander-0.10"],
        clean=clean,
        robustness_root=robustness_root,
        base_seed=1,
        bootstrap_resamples=3,
        bootstrap_confidence=0.95,
        bootstrap_base_seed=2,
    )
    resumed, resumed_created = audit._load_or_create_case(
        spec=spec,
        member=member,
        case=cases["baseline-wander-0.05"],
        clean=clean,
        robustness_root=robustness_root,
        base_seed=1,
        bootstrap_resamples=3,
        bootstrap_confidence=0.95,
        bootstrap_base_seed=2,
    )

    assert first_created is True
    assert second_created is True
    assert resumed_created is False
    assert first.artifact.npz_path.name == "baseline-wander-0.05.npz"
    assert second.artifact.npz_path.name == "baseline-wander-0.10.npz"
    assert first.artifact.npz_path != second.artifact.npz_path
    assert resumed.artifact.artifact_sha256 == first.artifact.artifact_sha256


def test_case_expansion_rejects_a_rehashed_severity_change(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    payload = spec.payload
    protocols = cast(dict[str, Any], payload["audit_protocols"])
    robustness = cast(dict[str, Any], protocols["robustness"])
    cases = cast(list[dict[str, Any]], robustness["severity_matrix"])
    cases[1]["parameters"]["amplitude_fraction"] = 0.051
    body = dict(payload)
    del body["artifact_sha256"]
    payload["artifact_sha256"] = post.canonical_sha256(body)
    changed = PostEvaluationSpec(
        path=spec.path,
        artifact_sha256=cast(str, payload["artifact_sha256"]),
        _canonical_payload=json.dumps(payload, separators=(",", ":"), sort_keys=True),
    )

    with pytest.raises(audit.RobustnessAuditIntegrityError, match="canonical frozen"):
        audit.expand_robustness_cases(changed)


@pytest.mark.parametrize(
    ("block", "key", "replacement", "message"),
    [
        (
            "dense_risk_coverage",
            "area_method",
            "trapezoid",
            "dense risk/coverage semantics",
        ),
        (
            "patient_resampling",
            "record_weighting",
            "one_vote_per_patient",
            "bootstrap policy",
        ),
        (
            "execution",
            "metric_precision",
            "cpu_float32",
            "execution semantics",
        ),
    ],
)
def test_case_expansion_rejects_rehashed_execution_semantic_changes(
    tmp_path: Path,
    block: str,
    key: str,
    replacement: str,
    message: str,
) -> None:
    spec = _spec(tmp_path)
    payload = spec.payload
    protocols = cast(dict[str, Any], payload["audit_protocols"])
    robustness = cast(dict[str, Any], protocols["robustness"])
    section = cast(dict[str, Any], robustness[block])
    section[key] = replacement
    body = dict(payload)
    del body["artifact_sha256"]
    payload["artifact_sha256"] = post.canonical_sha256(body)
    changed = PostEvaluationSpec(
        path=spec.path,
        artifact_sha256=cast(str, payload["artifact_sha256"]),
        _canonical_payload=json.dumps(payload, separators=(",", ":"), sort_keys=True),
    )

    with pytest.raises(audit.RobustnessAuditIntegrityError, match=message):
        audit.expand_robustness_cases(changed)


@pytest.mark.parametrize("case_id", ["gaussian-noise-20db", "contiguous-mask-100"])
def test_random_corruptions_are_stateless_by_ecg_id(
    tmp_path: Path, case_id: str
) -> None:
    case = next(
        item for item in audit.expand_robustness_cases(_spec(tmp_path)) if item.case_id == case_id
    )
    transform = audit.build_case_transform(case, base_seed=20_260_808)
    generator = torch.Generator().manual_seed(17)
    signals = torch.randn((4, 12, 1000), generator=generator, dtype=torch.float32)
    ecg_id = np.asarray([101, 205, 309, 411], dtype=np.int64)
    order = np.asarray([2, 0, 3, 1], dtype=np.int64)

    first = transform(signals, ecg_id)
    reordered = transform(signals[order], ecg_id[order])
    inverse = np.argsort(order)

    assert torch.equal(first, reordered[inverse])
    assert torch.equal(first, transform(signals, ecg_id))


def test_runtime_source_cross_binding_rejects_before_output(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    payload = spec.payload
    payload["protocol"] = {"protocol_hash": _hash("a")}
    bound_spec = PostEvaluationSpec(
        path=spec.path,
        artifact_sha256=spec.artifact_sha256,
        _canonical_payload=json.dumps(payload, separators=(",", ":"), sort_keys=True),
    )
    mismatched = SimpleNamespace(
        protocol=SimpleNamespace(protocol_hash=_hash("b")),
        members=tuple(SimpleNamespace(member_id=value) for value in bound_spec.member_ids),
    )

    with pytest.raises(audit.RobustnessAuditIntegrityError, match="runtime protocol"):
        audit.assert_runtime_matches_post_evaluation_spec(
            bound_spec, cast(CompletedAuditRuntime, mismatched)
        )

    assert not (bound_spec.output_root / "robustness").exists()


def test_clean_gate_rejection_occurs_before_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = _spec(tmp_path)
    monkeypatch.setattr(
        audit, "assert_runtime_matches_post_evaluation_spec", lambda unused_spec, unused: None
    )

    class RejectingRuntime:
        members = tuple(SimpleNamespace(member_id=value) for value in spec.member_ids)

        def assert_clean_logit_equivalence(self) -> tuple[object, ...]:
            raise ValueError("synthetic mismatch")

    with pytest.raises(audit.RobustnessCleanGateError, match="synthetic mismatch"):
        audit.run_robustness_audit(
            spec=spec,
            runtime=cast(CompletedAuditRuntime, RejectingRuntime()),
        )

    assert not (spec.output_root / "robustness").exists()


def test_metrics_and_paired_patient_bootstrap_are_deterministic() -> None:
    member = _member()
    gates = tuple(gate.to_dict() for gate in member.decisions.coverage_gates)
    clean_arrays, clean_analysis = audit.compute_member_case_arrays(
        inference=_inference(member),
        clean_logits=member.sealed_prediction.raw_logits,
        temperature=1.25,
        thresholds=(0.5,) * 5,
        gates=gates,
        bootstrap_resamples=9,
        bootstrap_confidence=0.95,
        bootstrap_seed=123,
        clean_case_arrays=None,
    )
    first_arrays, first_analysis = audit.compute_member_case_arrays(
        inference=_inference(member, logit_delta=0.2),
        clean_logits=member.sealed_prediction.raw_logits,
        temperature=1.25,
        thresholds=(0.5,) * 5,
        gates=gates,
        bootstrap_resamples=9,
        bootstrap_confidence=0.95,
        bootstrap_seed=456,
        clean_case_arrays=clean_arrays,
    )
    second_arrays, second_analysis = audit.compute_member_case_arrays(
        inference=_inference(member, logit_delta=0.2),
        clean_logits=member.sealed_prediction.raw_logits,
        temperature=1.25,
        thresholds=(0.5,) * 5,
        gates=gates,
        bootstrap_resamples=9,
        bootstrap_confidence=0.95,
        bootstrap_seed=456,
        clean_case_arrays=clean_arrays,
    )

    assert set(first_arrays) == audit._MEMBER_CASE_ARRAYS
    assert np.array_equal(first_arrays["bootstrap_delta"], second_arrays["bootstrap_delta"])
    assert np.array_equal(first_arrays["bootstrap_valid"], second_arrays["bootstrap_valid"])
    assert first_analysis == second_analysis
    assert np.count_nonzero(clean_arrays["bootstrap_delta"]) == 0
    assert cast(dict[str, Any], clean_analysis["delta_summary"])["direction"] == (
        "corrupted_minus_clean"
    )
    metric_summary = cast(dict[str, Any], first_analysis["metric_summary"])
    assert set(metric_summary) == {
        "raw",
        "calibrated",
        "calibrated_log_loss",
        "uncertainty",
        "raw_logit_drift",
        "full_coverage",
        "frozen_gates",
        "dense_risk_coverage",
    }
    assert first_arrays["dense_coverage"][-1] == 1.0
    dense_summary = cast(dict[str, Any], metric_summary["dense_risk_coverage"])
    assert dense_summary["area_method"] == "arithmetic_mean_over_all_prefix_coverages"
    assert dense_summary["oracle_reference"] == (
        "ascending_per_record_loss_stable_index_tiebreak"
    )
    assert dense_summary["random_reference"] == (
        "analytical_constant_full_coverage_risk"
    )


@pytest.mark.parametrize("summary_name", ["metric_summary", "delta_summary"])
def test_alignment_recomputes_sidecar_summaries(summary_name: str) -> None:
    member = _member()
    gates = tuple(gate.to_dict() for gate in member.decisions.coverage_gates)
    clean_arrays, _ = audit.compute_member_case_arrays(
        inference=_inference(member),
        clean_logits=member.sealed_prediction.raw_logits,
        temperature=1.25,
        thresholds=(0.5,) * 5,
        gates=gates,
        bootstrap_resamples=3,
        bootstrap_confidence=0.95,
        bootstrap_seed=123,
        clean_case_arrays=None,
    )
    arrays, analysis = audit.compute_member_case_arrays(
        inference=_inference(member, logit_delta=0.2),
        clean_logits=member.sealed_prediction.raw_logits,
        temperature=1.25,
        thresholds=(0.5,) * 5,
        gates=gates,
        bootstrap_resamples=3,
        bootstrap_confidence=0.95,
        bootstrap_seed=456,
        clean_case_arrays=clean_arrays,
    )
    metadata: dict[str, object] = {
        "decision_policy": {
            "temperature": 1.25,
            "thresholds": [0.5] * 5,
            "entropy_gates": list(gates),
            "retuned": False,
        },
        **analysis,
    }
    artifact = cast(
        Any,
        SimpleNamespace(arrays=arrays, metadata=metadata),
    )
    audit._assert_artifact_alignment(artifact, member)

    tampered = json.loads(json.dumps(metadata))
    if summary_name == "metric_summary":
        metric_summary = cast(dict[str, Any], tampered[summary_name])
        metric_summary["calibrated_log_loss"] += 0.1
    else:
        delta_summary = cast(dict[str, Any], tampered[summary_name])
        delta_summary["calibrated_log_loss"] += 0.1
    artifact = cast(Any, SimpleNamespace(arrays=arrays, metadata=tampered))
    with pytest.raises(audit.RobustnessAuditIntegrityError, match="summary differs"):
        audit._assert_artifact_alignment(artifact, member)


def test_resume_no_overwrite_and_manifest_self_hash_source_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(audit, "EXPECTED_CASE_COUNT", 1)
    monkeypatch.setattr(audit, "EXPECTED_MEMBER_COUNT", 1)
    monkeypatch.setattr(audit, "EXPECTED_MEMBER_CASE_COUNT", 1)
    monkeypatch.setattr(
        audit, "_robustness_settings", lambda unused: (1, 3, 0.95, 2)
    )
    member_id = "member-1"
    spec = _spec(tmp_path, member_ids=(member_id,))
    clean_case = audit.RobustnessCase("clean", "clean", "{}")
    monkeypatch.setattr(audit, "expand_robustness_cases", lambda unused: (clean_case,))
    member = _member(member_id)
    _, robustness_root = audit._audit_paths(spec)

    created_record, created = audit._load_or_create_case(
        spec=spec,
        member=member,
        case=clean_case,
        clean=None,
        robustness_root=robustness_root,
        base_seed=1,
        bootstrap_resamples=3,
        bootstrap_confidence=0.95,
        bootstrap_base_seed=2,
    )
    resumed_record, resumed_was_created = audit._load_or_create_case(
        spec=spec,
        member=member,
        case=clean_case,
        clean=None,
        robustness_root=robustness_root,
        base_seed=1,
        bootstrap_resamples=3,
        bootstrap_confidence=0.95,
        bootstrap_base_seed=2,
    )

    assert created is True
    assert resumed_was_created is False
    assert resumed_record.artifact.artifact_sha256 == created_record.artifact.artifact_sha256

    manifest_path, _ = audit._audit_paths(spec)
    manifest = audit.save_robustness_manifest(
        manifest_path,
        spec=spec,
        records=(created_record,),
        clean_evidence=(
            {
                "member_id": member_id,
                "exact": True,
                "mismatch_count": 0,
                "maximum_absolute_error": 0.0,
            },
        ),
    )
    original_manifest = manifest.path.read_bytes()
    loaded = audit.load_robustness_manifest(manifest.path, spec=spec, verify_sources=True)
    assert loaded.artifact_sha256 == manifest.artifact_sha256

    decoded = json.loads(original_manifest)
    decoded["case_count"] = 2
    manifest.path.write_text(json.dumps(decoded), encoding="utf-8")
    with pytest.raises(audit.RobustnessAuditIntegrityError, match="self-hash"):
        audit.load_robustness_manifest(manifest.path, spec=spec, verify_sources=True)

    manifest.path.write_bytes(original_manifest)
    with created_record.artifact.npz_path.open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(audit.RobustnessAuditIntegrityError, match="hash mismatch"):
        audit.load_robustness_manifest(manifest.path, spec=spec, verify_sources=True)


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        ("derived_seed", "bootstrap metadata differs from protocol"),
        ("record_weighting", "bootstrap metadata differs from protocol"),
        ("statistics", "bootstrap statistics differ from stored arrays"),
    ],
)
def test_resume_rejects_rehashed_bootstrap_sidecar_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
    message: str,
) -> None:
    member_id = "member-1"
    spec = _spec(tmp_path, member_ids=(member_id,))
    clean_case = audit.RobustnessCase("clean", "clean", "{}")
    member = _member(member_id)
    _, robustness_root = audit._audit_paths(spec)
    monkeypatch.setattr(
        audit, "_robustness_settings", lambda unused: (1, 3, 0.95, 2)
    )
    record, _ = audit._load_or_create_case(
        spec=spec,
        member=member,
        case=clean_case,
        clean=None,
        robustness_root=robustness_root,
        base_seed=1,
        bootstrap_resamples=3,
        bootstrap_confidence=0.95,
        bootstrap_base_seed=2,
    )
    sidecar = json.loads(record.artifact.json_path.read_text(encoding="utf-8"))
    bootstrap = cast(dict[str, Any], sidecar["metadata"]["bootstrap"])
    if tamper == "derived_seed":
        bootstrap["seed"] += 1
    elif tamper == "record_weighting":
        bootstrap["record_weighting"] = "one_vote_per_patient"
    else:
        statistics = cast(list[dict[str, Any]], bootstrap["statistics"])
        statistics[0]["mean_delta"] = 0.25
    body = dict(sidecar)
    del body["artifact_sha256"]
    sidecar["artifact_sha256"] = post.canonical_sha256(body)
    record.artifact.json_path.write_text(
        json.dumps(sidecar, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(audit.RobustnessAuditIntegrityError, match=message):
        audit._load_expected_artifact(
            record.artifact.npz_path,
            spec=spec,
            member_id=member_id,
            case=clean_case,
            member_binding_sha256=audit._member_binding_sha256(spec, member_id),
            robustness_root=robustness_root,
        )
