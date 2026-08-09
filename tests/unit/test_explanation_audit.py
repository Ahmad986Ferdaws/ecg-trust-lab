from __future__ import annotations

import json
import platform
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch

import ecg_trust.explanation_audit as audit
import ecg_trust.post_evaluation as post
from ecg_trust.audit_artifacts import (
    AuditArrayArtifact,
    AuditArrayFiles,
    load_audit_array_artifact,
    save_audit_array_artifact,
)
from ecg_trust.audit_runtime import CompletedAuditRuntime
from ecg_trust.explain import cross_method_temporal_similarity
from ecg_trust.post_evaluation import PostEvaluationSpec, canonical_sha256
from ecg_trust.protocol import LABEL_ORDER


def _hash(character: str = "a") -> str:
    return "sha256:" + character * 64


def _cohort_records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    identifier = 1
    for label_index, label in enumerate(LABEL_ORDER):
        for value, status in ((0, "negative"), (1, "positive")):
            for _ in range(6):
                bits = [0] * len(LABEL_ORDER)
                bits[label_index] = value
                records.append(
                    {
                        "rank": len(records),
                        "ecg_id": identifier,
                        "patient_id": 10_000 + identifier,
                        "target_label": label,
                        "target_index": label_index,
                        "target_status": status,
                        "target_value": value,
                        "target_bits": bits,
                        "selection_sha256": canonical_sha256({"ecg_id": identifier}),
                    }
                )
                identifier += 1
    return records


def _cohort_spec() -> PostEvaluationSpec:
    records = _cohort_records()
    payload = {
        "audit_protocols": {
            "explanations": {
                "cohort": {"records": records},
            }
        }
    }
    return PostEvaluationSpec(
        path=None,
        artifact_sha256=_hash("c"),
        _canonical_payload=json.dumps(payload),
    )


def _bound_spec_and_runtime() -> tuple[PostEvaluationSpec, object]:
    member_id = "resnet1d-seed2026"
    payload = {
        "sealed_evaluation": {
            "final_evaluation_spec": {"artifact_sha256": _hash("1")},
            "refit_bundle": {"artifact_sha256": _hash("2")},
            "calibration_bundle": {"artifact_sha256": _hash("3")},
            "opening_ledger": {
                "ledger_sha256": _hash("4"),
                "batch_sha256": _hash("5"),
            },
        },
        "members": [
            {
                "member_id": member_id,
                "architecture": "resnet1d",
                "seed": 2026,
                "model_name": "model",
                "refit_lineage_sha256": _hash("6"),
                "checkpoint": {"file_sha256": _hash("7")},
                "resolved_config": {"config_hash": _hash("8")},
                "calibration_decision": {"artifact_sha256": _hash("9")},
                "prediction": {
                    "artifact_sha256": _hash("a"),
                    "alignment_sha256": _hash("b"),
                },
            }
        ],
    }
    spec = PostEvaluationSpec(
        path=None,
        artifact_sha256=_hash("c"),
        _canonical_payload=json.dumps(payload),
    )
    member = SimpleNamespace(
        member_id=member_id,
        architecture="resnet1d",
        seed=2026,
        refit=SimpleNamespace(
            run_name="model",
            lineage_sha256=_hash("6"),
            resolved_config_hash=_hash("8"),
        ),
        checkpoint_sha256=_hash("7"),
        decisions=SimpleNamespace(integrity_sha256=_hash("9")),
        sealed_prediction=SimpleNamespace(integrity_sha256=_hash("a"), alignment_sha256=_hash("b")),
    )
    runtime = SimpleNamespace(
        final_evaluation_spec=SimpleNamespace(artifact_sha256=_hash("1")),
        refit_bundle=SimpleNamespace(artifact_sha256=_hash("2")),
        calibration_bundle=SimpleNamespace(artifact_sha256=_hash("3")),
        ledger=SimpleNamespace(ledger_sha256=_hash("4"), batch_sha256=_hash("5")),
        members=(member,),
    )
    return spec, runtime


def _binding(
    files: AuditArrayFiles,
    *,
    summary: Mapping[str, object],
    method: str | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {
        "artifact_sha256": files.artifact_sha256,
        "npz": {"path": str(files.npz_path), "file_sha256": files.npz_sha256},
        "sidecar": {
            "path": str(files.json_path),
            "file_sha256": files.json_file_sha256,
        },
        "summary": dict(summary),
    }
    if method is not None:
        result["method"] = method
    return result


def _manifest_spec(tmp_path: Path) -> PostEvaluationSpec:
    member_ids = tuple(
        f"{architecture}-seed{seed}"
        for architecture in ("resnet1d", "ecg_transformer")
        for seed in (2026, 2027, 2028)
    )
    cohort = {"records": _cohort_records()}
    body: dict[str, object] = {
        "protocol": {"protocol_hash": _hash("f")},
        "audit_protocols": {
            "explanations": {
                "cohort": cohort,
                "settings": post._explanation_settings(),
            }
        },
        "members": [
            {
                "member_id": member_id,
                "architecture": member_id.rsplit("-seed", 1)[0],
                "seed": int(member_id.rsplit("-seed", 1)[1]),
                "checkpoint": {"file_sha256": _hash("1")},
                "calibration_decision": {"artifact_sha256": _hash("2")},
                "prediction": {
                    "artifact_sha256": _hash("3"),
                    "record_count": 2_158,
                },
            }
            for member_id in member_ids
        ],
        "output_contract": {
            "root": str(tmp_path.resolve()),
            "artifacts": {"explanations_manifest": str((tmp_path / "manifest.json").resolve())},
        },
    }
    payload = {**body, "artifact_sha256": canonical_sha256(body)}
    return PostEvaluationSpec(
        path=None,
        artifact_sha256=cast(str, payload["artifact_sha256"]),
        _canonical_payload=json.dumps(payload, separators=(",", ":"), sort_keys=True),
    )


def _attribution_runtime() -> dict[str, object]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "captum": "synthetic-test-fixture",
        "cuda_runtime": None,
        "cudnn": None,
        "device": "cpu",
        "device_name": None,
        "compute_capability": None,
        "model_dtype": "float32",
        "deterministic_algorithms": True,
        "cudnn_benchmark": False,
        "cuda_matmul_tf32": False,
        "cudnn_tf32": False,
        "sealed_clean_equivalence_precision": "bf16",
    }


def _method_metadata(
    *,
    spec: PostEvaluationSpec,
    cohort: audit._Cohort,
    member_id: str,
    architecture: str,
    seed: int,
    method: str,
    runtime: Mapping[str, object],
) -> dict[str, object]:
    return {
        "post_evaluation_spec_sha256": spec.artifact_sha256,
        "cohort_sha256": cohort.payload_sha256,
        "member_id": member_id,
        "architecture": architecture,
        "seed": seed,
        "method": method,
        "checkpoint_sha256": _hash("1"),
        "calibration_decision_sha256": _hash("2"),
        "sealed_prediction_sha256": _hash("3"),
        "temperature": 1.0,
        "target_score": (
            "signed_correct_status_logit; sign=+1 for positive cell and -1 for "
            "negative cell; probability=sigmoid(signed_logit/temperature)"
        ),
        "attribution_runtime": dict(runtime),
        "settings": {
            "ig_steps": 32,
            "ig_internal_batch_size": 8,
            "occlusion_window": 50,
            "occlusion_stride": 25,
            "occlusion_perturbations": 16,
            "fractions": [0.0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0],
            "random_ranking_replicates": 20,
            "random_ranking_seed": 20_261_008,
            "randomization_seeds": [2_026_801, 2_026_802, 2_026_803],
            "stability_snr_db": 40.0,
            "stability_replicates": 3,
            "outer_batch_size": 4,
            "faithfulness_scoring_batch_size": 60,
            "numeric_precision": "float32_model_and_attributions_float64_summaries",
        },
    }


def _method_arrays(cohort: audit._Cohort, method: str) -> dict[str, np.ndarray[Any, Any]]:
    count = cohort.size
    fractions = np.asarray([0.0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0])
    curve_logits = np.zeros((count, fractions.size), dtype=np.float64)
    curve_probabilities = np.full_like(curve_logits, 0.5)
    random_logits = np.zeros((20, count, fractions.size), dtype=np.float64)
    random_probabilities = np.full_like(random_logits, 0.5)
    attribution_channels = 1 if method == "grad_cam_1d" else 12
    temporal_pattern = np.linspace(0.1, 1.0, 1000, dtype=np.float32)
    attributions = np.broadcast_to(
        temporal_pattern, (count, attribution_channels, temporal_pattern.size)
    ).copy()
    deletion_auc = audit._per_example_auc(curve_probabilities, fractions)
    insertion_auc = audit._per_example_auc(curve_probabilities, fractions)
    least_deletion_auc = audit._per_example_auc(curve_probabilities, fractions)
    random_deletion_auc = audit._per_example_auc(random_probabilities, fractions)
    arrays: dict[str, np.ndarray[Any, Any]] = {
        "ecg_id": cohort.ecg_id,
        "patient_id": cohort.patient_id,
        "target_index": cohort.target_index,
        "target_value": cohort.target_value,
        "target_sign": np.where(cohort.target_value == 1, 1, -1).astype(np.int8),
        "target_bits": cohort.target_bits,
        "fp32_raw_logits": np.zeros((count, len(LABEL_ORDER)), dtype=np.float64),
        "sealed_bf16_raw_logits": np.zeros((count, len(LABEL_ORDER)), dtype=np.float64),
        "fp32_minus_sealed_logits": np.zeros((count, len(LABEL_ORDER)), dtype=np.float64),
        "attributions": attributions,
        "deterministic_repeat_exact": np.ones(count, dtype=np.uint8),
        "deterministic_repeat_cosine": np.ones(count, dtype=np.float64),
        "stability_cosine": np.ones((3, count), dtype=np.float64),
        "parameter_randomization_cosine": np.zeros((3, count), dtype=np.float64),
        "fractions": fractions,
        "deletion_logits": curve_logits,
        "deletion_probabilities": curve_probabilities,
        "insertion_logits": curve_logits.copy(),
        "insertion_probabilities": curve_probabilities.copy(),
        "least_deletion_logits": curve_logits.copy(),
        "least_deletion_probabilities": curve_probabilities.copy(),
        "random_deletion_logits": random_logits,
        "random_deletion_probabilities": random_probabilities,
        "deletion_auc": deletion_auc,
        "insertion_auc": insertion_auc,
        "least_deletion_auc": least_deletion_auc,
        "random_deletion_auc": random_deletion_auc,
        "guided_vs_random_deletion_advantage": (random_deletion_auc.mean(axis=0) - deletion_auc),
    }
    if method == "integrated_gradients":
        lead_fractions = np.arange(13, dtype=np.float64) / 12.0
        lead_logits = np.zeros((count, 13), dtype=np.float64)
        lead_probabilities = np.full_like(lead_logits, 0.5)
        random_lead_logits = np.zeros((20, count, 13), dtype=np.float64)
        random_lead_probabilities = np.full_like(random_lead_logits, 0.5)
        lead_auc = audit._per_example_auc(lead_probabilities, lead_fractions)
        random_lead_auc = audit._per_example_auc(random_lead_probabilities, lead_fractions)
        arrays.update(
            {
                "ig_completeness_delta": np.zeros(count, dtype=np.float64),
                "lead_fractions": lead_fractions,
                "lead_ablation_logits": lead_logits,
                "lead_ablation_probabilities": lead_probabilities,
                "least_lead_ablation_logits": lead_logits.copy(),
                "least_lead_ablation_probabilities": lead_probabilities.copy(),
                "random_lead_ablation_logits": random_lead_logits,
                "random_lead_ablation_probabilities": random_lead_probabilities,
                "lead_ablation_auc": lead_auc,
                "least_lead_ablation_auc": lead_auc.copy(),
                "random_lead_ablation_auc": random_lead_auc,
                "guided_vs_random_lead_advantage": (random_lead_auc.mean(axis=0) - lead_auc),
            }
        )
    return arrays


def _cross_binding(
    *,
    tmp_path: Path,
    spec: PostEvaluationSpec,
    cohort: audit._Cohort,
    member_id: str,
    architecture: str,
    seed: int,
    methods: Mapping[str, AuditArrayArtifact],
) -> dict[str, object]:
    method_names = tuple(methods)
    pair_names: list[str] = []
    cosine: list[np.ndarray[Any, Any]] = []
    spearman: list[np.ndarray[Any, Any]] = []
    cosine_valid: list[np.ndarray[Any, Any]] = []
    spearman_valid: list[np.ndarray[Any, Any]] = []
    for left_index, left in enumerate(method_names):
        for right in method_names[left_index + 1 :]:
            pair_names.append(f"{left}__vs__{right}")
            result = cross_method_temporal_similarity(
                torch.from_numpy(np.array(methods[left].arrays["attributions"], copy=True)),
                torch.from_numpy(np.array(methods[right].arrays["attributions"], copy=True)),
            )
            cosine.append(result.cosine.numpy().astype(np.float64, copy=False))
            spearman.append(result.spearman.numpy().astype(np.float64, copy=False))
            cosine_valid.append(result.cosine_valid.numpy().astype(np.bool_, copy=False))
            spearman_valid.append(result.spearman_valid.numpy().astype(np.bool_, copy=False))
    files = save_audit_array_artifact(
        tmp_path / "members" / member_id / "cross_method.npz",
        artifact_type=audit.EXPLANATION_CROSS_METHOD_TYPE,
        arrays={
            "ecg_id": cohort.ecg_id,
            "pair_name": np.asarray(pair_names, dtype=np.str_),
            "cosine": np.stack(cosine),
            "spearman": np.stack(spearman),
            "cosine_valid": np.stack(cosine_valid),
            "spearman_valid": np.stack(spearman_valid),
        },
        metadata={
            "post_evaluation_spec_sha256": spec.artifact_sha256,
            "cohort_sha256": cohort.payload_sha256,
            "member_id": member_id,
            "architecture": architecture,
            "seed": seed,
            "aggregation": "absolute_attribution_mean_across_leads_to_time",
            "metrics": ["cosine", "spearman"],
        },
    )
    artifact = load_audit_array_artifact(
        files.npz_path, expected_artifact_type=audit.EXPLANATION_CROSS_METHOD_TYPE
    )
    return _binding(files, summary=audit._cross_method_summary(artifact))


def _manifest(tmp_path: Path) -> tuple[Path, PostEvaluationSpec]:
    spec = _manifest_spec(tmp_path)
    cohort = audit._cohort_from_spec(spec)
    runtime = _attribution_runtime()
    members: list[dict[str, object]] = []
    clean: list[dict[str, object]] = []
    runtime_blocks: dict[str, object] = {}
    for architecture in ("resnet1d", "ecg_transformer"):
        methods = audit._method_names(architecture)
        for seed in (2026, 2027, 2028):
            member_id = f"{architecture}-seed{seed}"
            runtime_blocks[member_id] = dict(runtime)
            method_bindings: list[dict[str, object]] = []
            method_artifacts: dict[str, AuditArrayArtifact] = {}
            for method in methods:
                files = save_audit_array_artifact(
                    tmp_path / "members" / member_id / f"{method}.npz",
                    artifact_type=audit.EXPLANATION_ARRAY_TYPE,
                    arrays=_method_arrays(cohort, method),
                    metadata=_method_metadata(
                        spec=spec,
                        cohort=cohort,
                        member_id=member_id,
                        architecture=architecture,
                        seed=seed,
                        method=method,
                        runtime=runtime,
                    ),
                )
                artifact = load_audit_array_artifact(
                    files.npz_path,
                    expected_artifact_type=audit.EXPLANATION_ARRAY_TYPE,
                )
                method_artifacts[method] = artifact
                method_bindings.append(
                    _binding(
                        files,
                        method=method,
                        summary=audit._method_summary(artifact),
                    )
                )
            cross_method = _cross_binding(
                tmp_path=tmp_path,
                spec=spec,
                cohort=cohort,
                member_id=member_id,
                architecture=architecture,
                seed=seed,
                methods=method_artifacts,
            )
            members.append(
                {
                    "member_id": member_id,
                    "architecture": architecture,
                    "seed": seed,
                    "methods": method_bindings,
                    "cross_method": cross_method,
                }
            )
            clean.append(
                {
                    "member_id": member_id,
                    "record_count": 2_158,
                    "logit_count": 10_790,
                    "sealed_prediction_sha256": _hash("3"),
                    "exact": True,
                    "mismatch_count": 0,
                    "maximum_absolute_error": 0.0,
                    "mean_absolute_error": 0.0,
                }
            )
    body: dict[str, object] = {
        "schema_version": audit.EXPLANATION_MANIFEST_SCHEMA_VERSION,
        "artifact_type": audit.EXPLANATION_MANIFEST_TYPE,
        "post_evaluation_spec_sha256": spec.artifact_sha256,
        "protocol_hash": _hash("f"),
        "cohort": audit._cohort_summary(cohort),
        "settings": post._explanation_settings(),
        "attribution_runtime": runtime_blocks,
        "clean_logit_equivalence": clean,
        "members": members,
        "limitations": audit._explanation_limitations(),
    }
    payload = {**body, "artifact_sha256": canonical_sha256(body)}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path, spec


def test_cohort_parser_preserves_balanced_unique_target_assignment() -> None:
    cohort = audit._cohort_from_spec(_cohort_spec())

    assert cohort.size == 60
    assert np.unique(cohort.ecg_id).size == 60
    assert np.unique(cohort.patient_id).size == 60
    for label_index in range(len(LABEL_ORDER)):
        for value in (0, 1):
            assert (
                np.count_nonzero(
                    (cohort.target_index == label_index) & (cohort.target_value == value)
                )
                == 6
            )


def test_stability_noise_is_stateless_per_ecg_and_replicate() -> None:
    inputs = torch.arange(2 * 12 * 1000, dtype=torch.float32).reshape(2, 12, 1000)
    identifiers = np.asarray([41, 73], dtype=np.int64)

    first = audit._noisy_inputs(inputs, identifiers, snr_db=40.0, replicate=0)
    second = audit._noisy_inputs(inputs, identifiers, snr_db=40.0, replicate=0)
    changed = audit._noisy_inputs(inputs, identifiers, snr_db=40.0, replicate=1)

    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
    assert not torch.equal(first, changed)
    torch.testing.assert_close(
        inputs, torch.arange(inputs.numel(), dtype=torch.float32).reshape_as(inputs)
    )


def test_auc_and_guided_advantage_use_each_example_axis() -> None:
    fractions = np.asarray([0.0, 0.5, 1.0], dtype=np.float64)
    curves = np.asarray([[1.0, 0.5, 0.0], [0.8, 0.4, 0.0]], dtype=np.float64)

    result = audit._per_example_auc(curves, fractions)

    np.testing.assert_allclose(result, [0.5, 0.4])


def _settings_spec(settings: Mapping[str, object], *, hash_character: str) -> PostEvaluationSpec:
    payload = {"audit_protocols": {"explanations": {"settings": dict(settings)}}}
    return PostEvaluationSpec(
        path=None,
        artifact_sha256=_hash(hash_character),
        _canonical_payload=json.dumps(payload),
    )


def test_frozen_explanation_settings_are_explicit() -> None:
    settings = post._explanation_settings()
    spec = _settings_spec(settings, hash_character="c")

    parsed = audit._settings_from_spec(spec, outer_batch_size=4)
    assert parsed.ig_steps == 32
    assert parsed.ig_internal_batch_size == 8
    assert parsed.occlusion_window == 50
    assert parsed.random_ranking_seed == 20_261_008


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("target_labels",), list(reversed(LABEL_ORDER))),
        (("target_assignment",), "changed"),
        (("baseline",), "changed"),
        (("methods_by_architecture",), {}),
        (("selection_data_policy",), {}),
        (("retuning_or_example_replacement_allowed",), True),
        (("integrated_gradients", "n_steps"), 16),
        (("integrated_gradients", "internal_batch_size"), 4),
        (("integrated_gradients", "multiply_by_inputs"), False),
        (("integrated_gradients", "integration_method"), "riemann_right"),
        (("temporal_occlusion", "window_samples"), 40),
        (("temporal_occlusion", "stride_samples"), 10),
        (("temporal_occlusion", "perturbations_per_eval"), 8),
        (("temporal_occlusion", "normalize"), False),
        (("faithfulness", "random_ranking_seed"), 1),
    ],
)
def test_frozen_explanation_settings_reject_computation_or_policy_changes(
    path: tuple[str, ...], value: object
) -> None:
    changed = cast(dict[str, object], json.loads(json.dumps(post._explanation_settings())))
    cursor = changed
    for component in path[:-1]:
        cursor = cast(dict[str, object], cursor[component])
    cursor[path[-1]] = value
    changed_spec = _settings_spec(changed, hash_character="d")

    with pytest.raises(audit.ExplanationAuditIntegrityError, match="settings differ"):
        audit._settings_from_spec(changed_spec, outer_batch_size=4)


def test_runtime_must_match_every_frozen_release_binding() -> None:
    spec, runtime = _bound_spec_and_runtime()
    audit._assert_runtime_bound_to_spec(spec, cast(CompletedAuditRuntime, runtime))

    cast(SimpleNamespace, runtime).members[0].checkpoint_sha256 = _hash("f")
    with pytest.raises(audit.ExplanationAuditIntegrityError, match="differs"):
        audit._assert_runtime_bound_to_spec(spec, cast(CompletedAuditRuntime, runtime))


def test_manifest_round_trip_verifies_every_array_file(tmp_path: Path) -> None:
    path, spec = _manifest(tmp_path)

    loaded = audit.load_explanation_manifest(
        path,
        expected_spec_sha256=spec.artifact_sha256,
        spec=spec,
        verify_sources=True,
    )

    assert loaded.path == path.resolve()
    assert loaded.artifact_sha256 == cast(str, loaded.payload["artifact_sha256"])
    first_npz = next((tmp_path / "members").rglob("*.npz"))
    first_npz.write_bytes(first_npz.read_bytes() + b"tamper")
    with pytest.raises(audit.ExplanationAuditIntegrityError, match="file hash"):
        audit.load_explanation_manifest(
            path,
            expected_spec_sha256=spec.artifact_sha256,
            spec=spec,
            verify_sources=True,
        )


def test_manifest_rejects_rehashed_missing_clean_gate_and_no_overwrite(
    tmp_path: Path,
) -> None:
    path, spec = _manifest(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["clean_logit_equivalence"][0]["exact"] = False
    body = dict(payload)
    del body["artifact_sha256"]
    payload["artifact_sha256"] = canonical_sha256(body)
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(audit.ExplanationAuditIntegrityError, match="not exact"):
        audit.load_explanation_manifest(
            path,
            expected_spec_sha256=spec.artifact_sha256,
            spec=spec,
            verify_sources=False,
        )
    with pytest.raises(FileExistsError):
        audit._write_new_json(path, {"unused": True})
