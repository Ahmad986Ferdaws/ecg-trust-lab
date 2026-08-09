from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

import ecg_trust.final_results as final
from ecg_trust.protocol import ExperimentProtocol
from scripts.render_final_results import build_parser


def _hash(character: str = "a") -> str:
    return "sha256:" + character * 64


@dataclass(frozen=True)
class _FakeSpec:
    path: Path
    artifact_sha256: str = _hash("a")
    member_ids: tuple[str, ...] = tuple(
        f"{architecture}-seed{seed}"
        for architecture in ("resnet1d", "ecg_transformer")
        for seed in (2026, 2027, 2028)
    )


def _context(tmp_path: Path) -> final._VerifiedContext:
    root = tmp_path / "runs" / "post_evaluation" / "comparison"
    root.mkdir(parents=True)
    spec_path = root / "audit_spec.json"
    spec_path.write_text("synthetic spec\n", encoding="utf-8")
    deviations_path = tmp_path / "reports" / "PROTOCOL_DEVIATIONS.md"
    deviations_path.parent.mkdir(parents=True)
    deviations_path.write_text("# Protocol deviations\n\nDEV-001 disclosure\n", encoding="utf-8")
    artifacts: dict[str, object] = {
        "audit_spec": str(spec_path),
        "derived_manifest": str(root / "derived_artifacts.manifest.json"),
        "final_results_markdown": str(root / "reports" / "FINAL_RESULTS.md"),
        "probability_audit": str(root / "probability_audit.json"),
        "robustness_manifest": str(root / "robustness" / "manifest.json"),
        "explanations_manifest": str(root / "explanations" / "manifest.json"),
        "demo_directory": str(root / "demo"),
        "demo_policy": str(root / "demo" / "resnet1d-seed2026.coverage80.demo-policy.json"),
        "demo_examples": str(root / "demo" / "fold8-label-free.examples.json"),
        "demo_binding": str(root / "demo" / "resnet1d-seed2026.coverage80.demo-binding.json"),
        "publication_tables_directory": str(root / "publication" / "tables"),
        "publication_figures_directory": str(root / "publication" / "figures"),
    }
    payload: dict[str, object] = {
        "analysis_runtime": {"project_root": str(tmp_path)},
        "sealed_evaluation": {
            "protocol_deviations": {
                "path": str(deviations_path),
                "file_sha256": final.sha256_file(deviations_path),
            },
            "final_batch_summary": {
                "path": str(tmp_path / "sealed-summary.json"),
                "file_sha256": _hash("b"),
                "artifact_sha256": _hash("c"),
                "batch_sha256": _hash("d"),
            },
        },
        "audit_protocols": {
            "demo": {
                "member_id": "resnet1d-seed2026",
                "architecture": "resnet1d",
                "seed": 2026,
                "target_coverage": 0.8,
                "entropy_method": "mean_normalized_binary_entropy",
                "gate": {"maximum_entropy": 0.5},
                "checkpoint": {"path": "checkpoint", "file_sha256": _hash("1")},
                "resolved_config": {
                    "path": "config",
                    "file_sha256": _hash("2"),
                    "config_hash": _hash("3"),
                },
                "calibration_decision": {
                    "path": "decision",
                    "file_sha256": _hash("4"),
                    "artifact_sha256": _hash("5"),
                },
                "retuning_allowed": False,
            }
        },
        "output_contract": {"root": str(root), "artifacts": artifacts},
    }
    return final._VerifiedContext(
        spec=cast(Any, _FakeSpec(spec_path)),
        payload=payload,
        root=root.resolve(),
        artifacts=artifacts,
    )


def _write_self_hashed(path: Path, body: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**body, "artifact_sha256": final.canonical_sha256(body)}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fake_analyses() -> tuple[final._MemberAnalysis, ...]:
    result: list[final._MemberAnalysis] = []
    for architecture in ("resnet1d", "ecg_transformer"):
        for seed in (2026, 2027, 2028):
            member_id = f"{architecture}-seed{seed}"
            result.append(
                final._MemberAnalysis(
                    member_id=member_id,
                    architecture=architecture,
                    seed=seed,
                    prediction=cast(
                        Any,
                        SimpleNamespace(
                            integrity_sha256=_hash("6"),
                            alignment_sha256=_hash("7"),
                        ),
                    ),
                    decision=cast(Any, SimpleNamespace(integrity_sha256=_hash("8"))),
                    report={},
                    audit={"member": member_id},
                )
            )
    return tuple(result)


def test_probability_render_publishes_audit_last_and_resumes_exactly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context(tmp_path)
    monkeypatch.setattr(final, "_load_context", lambda *args, **kwargs: context)
    monkeypatch.setattr(
        final,
        "_load_bound_subgroups",
        lambda *args, **kwargs: SimpleNamespace(artifact_sha256=_hash("9")),
    )
    monkeypatch.setattr(final, "_derive_member_analyses", lambda *args, **kwargs: _fake_analyses())
    monkeypatch.setattr(final, "_load_architecture_summaries", lambda *args: ())
    monkeypatch.setattr(final, "_load_paired_reports", lambda *args: ())
    monkeypatch.setattr(
        final,
        "_render_tables",
        lambda *args: {name: f"table:{name}\n".encode() for name in final.TABLE_FILENAMES},
    )
    monkeypatch.setattr(
        final,
        "_render_figures",
        lambda *args, **kwargs: {
            name: b"synthetic-png:" + name.encode() for name in final.FIGURE_FILENAMES
        },
    )

    protocol = ExperimentProtocol.canonical()
    first = final.render_probability_results("ignored", protocol=protocol)
    assert first.audit_path.is_file()
    assert len(first.files) == len(final.TABLE_FILENAMES) + len(final.FIGURE_FILENAMES)
    committed = final.load_and_verify_probability_audit(first.audit_path, context=context)
    assert committed["artifact_sha256"] == first.audit_artifact_sha256

    # A complete exact rerun is read-only and returns the same identities.
    second = final.render_probability_results("ignored", protocol=protocol)
    assert second.to_dict() == first.to_dict()


def test_probability_render_adopts_exact_partial_and_rejects_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context(tmp_path)
    destinations = final._publication_destinations(context)
    partial = destinations[final.TABLE_FILENAMES[0]]
    partial.parent.mkdir(parents=True)
    partial.write_bytes(f"table:{final.TABLE_FILENAMES[0]}\n".encode())
    monkeypatch.setattr(final, "_load_context", lambda *args, **kwargs: context)
    monkeypatch.setattr(
        final,
        "_load_bound_subgroups",
        lambda *args, **kwargs: SimpleNamespace(artifact_sha256=_hash("9")),
    )
    monkeypatch.setattr(final, "_derive_member_analyses", lambda *args, **kwargs: _fake_analyses())
    monkeypatch.setattr(final, "_load_architecture_summaries", lambda *args: ())
    monkeypatch.setattr(final, "_load_paired_reports", lambda *args: ())
    monkeypatch.setattr(
        final,
        "_render_tables",
        lambda *args: {name: f"table:{name}\n".encode() for name in final.TABLE_FILENAMES},
    )
    monkeypatch.setattr(
        final,
        "_render_figures",
        lambda *args, **kwargs: {name: name.encode() for name in final.FIGURE_FILENAMES},
    )
    final.render_probability_results("ignored", protocol=ExperimentProtocol.canonical())

    other_context = _context(tmp_path / "other")
    other = final._publication_destinations(other_context)[final.TABLE_FILENAMES[0]]
    other.parent.mkdir(parents=True)
    other.write_bytes(b"different")
    monkeypatch.setattr(final, "_load_context", lambda *args, **kwargs: other_context)
    with pytest.raises(final.FinalResultsIntegrityError, match="crash-recovery file differs"):
        final.render_probability_results("ignored", protocol=ExperimentProtocol.canonical())
    assert not final._artifact_path(other_context, "probability_audit").exists()


@pytest.mark.parametrize(
    ("branch", "artifact_type"),
    [
        ("robustness", "ecg_trust.robustness_audit_manifest"),
        ("explanation", "ecg_trust.explanation_audit_manifest"),
    ],
)
def test_branch_manifest_requires_exact_type_spec_and_local_file(
    tmp_path: Path, branch: str, artifact_type: str
) -> None:
    context = _context(tmp_path)
    directory = context.root / ("explanations" if branch == "explanation" else branch)
    derived = directory / "result.json"
    derived.parent.mkdir(parents=True)
    derived.write_text("result\n", encoding="utf-8")
    manifest = directory / "manifest.json"
    body: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": artifact_type,
        "post_evaluation_spec_sha256": context.spec.artifact_sha256,
        "files": [
            {
                "path": str(derived.relative_to(context.root)),
                "file_sha256": final.sha256_file(derived),
            }
        ],
    }
    _write_self_hashed(manifest, body)
    loaded = final._load_completed_branch_manifest(manifest, context=context, branch=branch)
    assert loaded["artifact_type"] == artifact_type

    wrong = dict(body)
    wrong["artifact_type"] = "ecg_trust.wrong_manifest"
    manifest.unlink()
    _write_self_hashed(manifest, wrong)
    with pytest.raises(final.FinalResultsIntegrityError, match="artifact type differs"):
        final._load_completed_branch_manifest(manifest, context=context, branch=branch)


def test_canonical_explanation_loader_receives_frozen_spec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context(tmp_path)
    generic = {"artifact_sha256": _hash("e")}
    observed: dict[str, object] = {}

    def fake_loader(
        path: Path,
        *,
        expected_spec_sha256: str,
        spec: object,
        verify_sources: bool,
    ) -> SimpleNamespace:
        observed.update(
            {
                "path": path,
                "expected_spec_sha256": expected_spec_sha256,
                "spec": spec,
                "verify_sources": verify_sources,
            }
        )
        return SimpleNamespace(payload=generic)

    import ecg_trust.explanation_audit as explanation_audit

    monkeypatch.setattr(explanation_audit, "load_explanation_manifest", fake_loader)
    path = context.root / "explanations" / "manifest.json"
    loaded = final._verify_canonical_branch_manifest(
        path,
        context=context,
        branch="explanation",
        generic=generic,
    )

    assert loaded == generic
    assert observed == {
        "path": path,
        "expected_spec_sha256": context.spec.artifact_sha256,
        "spec": context.spec,
        "verify_sources": True,
    }


def test_verified_manifest_summaries_extract_controls(tmp_path: Path) -> None:
    sidecar = tmp_path / "case.json"
    sidecar.write_text(
        json.dumps(
            {
                "metadata": {
                    "delta_summary": {
                        "calibrated": {"macro": {"roc_auc": -0.12, "brier_score": 0.04}},
                        "dense_risk_coverage": {"aurc_hamming": 0.03},
                        "frozen_gates": [{"target_coverage": 0.8, "coverage": -0.07}],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    robustness = final._summarize_robustness(
        {
            "member_count": 1,
            "case_count": 2,
            "member_case_count": 2,
            "artifacts": [
                {
                    "member_id": "resnet1d-seed2026",
                    "case_id": "noise",
                    "sidecar": {"path": str(sidecar)},
                }
            ],
        }
    )
    assert cast(dict[str, object], robustness["minimum_macro_auroc_delta"])[
        "value"
    ] == pytest.approx(-0.12)
    assert cast(dict[str, object], robustness["maximum_absolute_gate_coverage_delta"])[
        "value"
    ] == pytest.approx(0.07)

    members: list[dict[str, object]] = []
    cross_summaries: list[dict[str, object]] = []
    for index in range(6):
        method_count = 3 if index < 3 else 2
        methods: list[dict[str, object]] = []
        for method_index in range(method_count):
            methods.append(
                {
                    "method": f"method-{method_index}",
                    "summary": {
                        "deterministic_repeat_exact": True,
                        "mean_repeat_cosine": 1.0,
                        "mean_stability_cosine_40db": 0.9,
                        "mean_parameter_randomization_cosine": 0.2,
                        "mean_guided_vs_random_deletion_advantage": 0.1,
                        "fp32_vs_sealed_bf16_logit_drift": {
                            "mean_absolute": 0.001,
                            "maximum_absolute": (0.25 if (index, method_index) == (5, 1) else 0.01),
                            "root_mean_square": 0.002,
                        },
                    },
                }
            )
        pair_count = 3 if index < 3 else 1
        mean_cosine: list[float | None] = [0.5] * pair_count
        valid_cosine_examples = [60] * pair_count
        if index == 0:
            mean_cosine[0] = None
            valid_cosine_examples[0] = 0
        cross_summary: dict[str, object] = {
            "pairs": [f"pair-{pair_index}" for pair_index in range(pair_count)],
            "mean_cosine": mean_cosine,
            "mean_spearman": [0.4] * pair_count,
            "valid_cosine_examples": valid_cosine_examples,
            "valid_spearman_examples": [60] * pair_count,
        }
        cross_summaries.append(cross_summary)
        members.append(
            {
                "member_id": f"member-{index}",
                "methods": methods,
                "cross_method": {"summary": cross_summary},
            }
        )
    explanations = final._summarize_explanations(
        {
            "members": members,
            "cohort": {"records": 60},
            "settings": {
                "target_score": {
                    "positive_cell": "+1_times_target_label_logit",
                    "negative_cell": "-1_times_target_label_logit",
                    "probability": ("sigmoid(signed_correct_status_logit_over_frozen_temperature)"),
                    "attribution_orientation": "multiply_target_label_map_by_cell_sign",
                },
                "execution": {
                    "numeric_precision": "float32",
                    "sealed_clean_equivalence_precision": "bf16_as_frozen_in_final_batch",
                    "fp32_vs_sealed_cohort_logit_drift_required": True,
                },
            },
            "attribution_runtime": {
                f"member-{index}": {
                    "model_dtype": "float32",
                    "sealed_clean_equivalence_precision": "bf16",
                }
                for index in range(6)
            },
        }
    )
    assert explanations["method_artifacts"] == 15
    assert explanations["method_ecg_evaluations"] == 900
    assert explanations["cross_method_pairs"] == 12
    assert explanations["cross_method_cosine_pairs_with_valid_values"] == 11
    assert explanations["cross_method_cosine_valid_examples"] == 660
    assert explanations["cross_method_cosine_invalid_examples"] == 60
    assert explanations["mean_cross_method_cosine"] == pytest.approx(0.5)
    assert explanations["cross_method_spearman_valid_examples"] == 720
    maximum_drift = cast(dict[str, object], explanations["maximum_fp32_vs_sealed_bf16_logit_drift"])
    assert maximum_drift == {
        "value": 0.25,
        "member_id": "member-5",
        "method": "method-1",
    }

    cross_summaries[0]["valid_cosine_examples"] = [1, 60, 60]
    with pytest.raises(final.FinalResultsIntegrityError, match="validity count and mean disagree"):
        final._summarize_explanations(
            {
                "members": members,
                "cohort": {"records": 60},
                "settings": {
                    "target_score": {
                        "positive_cell": "+1_times_target_label_logit",
                        "negative_cell": "-1_times_target_label_logit",
                        "probability": (
                            "sigmoid(signed_correct_status_logit_over_frozen_temperature)"
                        ),
                        "attribution_orientation": "multiply_target_label_map_by_cell_sign",
                    },
                    "execution": {
                        "numeric_precision": "float32",
                        "sealed_clean_equivalence_precision": "bf16_as_frozen_in_final_batch",
                        "fp32_vs_sealed_cohort_logit_drift_required": True,
                    },
                },
                "attribution_runtime": {
                    f"member-{index}": {
                        "model_dtype": "float32",
                        "sealed_clean_equivalence_precision": "bf16",
                    }
                    for index in range(6)
                },
            }
        )


def _architecture_summaries() -> tuple[dict[str, object], ...]:
    result: list[dict[str, object]] = []
    for index, architecture in enumerate(("resnet1d", "ecg_transformer")):
        result.append(
            {
                "architecture": architecture,
                "cross_seed_summary": {
                    metric: {"mean": 0.9 - index * 0.05}
                    for metric in ("roc_auc", "average_precision", "brier_score", "ece")
                },
            }
        )
    return tuple(result)


def _paired_reports() -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "seed": seed,
            "comparison": {
                "macro": {"roc_auc": {"estimate": -0.02, "lower": -0.03, "upper": -0.01}}
            },
        }
        for seed in (2026, 2027, 2028)
    )


def test_finalize_writes_disclosures_manifest_and_exact_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context(tmp_path)
    run_log = tmp_path / "reports" / "FINAL_EVALUATION_RUN_LOG.md"
    run_log.write_text("operational disclosure\n", encoding="utf-8")
    probability = {"artifact_sha256": _hash("1")}
    robustness = {"artifact_sha256": _hash("2")}
    explanations = {"artifact_sha256": _hash("3")}
    demo = SimpleNamespace(binding_artifact_sha256=_hash("4"))
    monkeypatch.setattr(final, "_load_context", lambda *args, **kwargs: context)
    monkeypatch.setattr(
        final, "load_and_verify_probability_audit", lambda *args, **kwargs: probability
    )
    monkeypatch.setattr(
        final,
        "_load_completed_branch_manifest",
        lambda path, **kwargs: robustness if "robustness" in str(path) else explanations,
    )
    monkeypatch.setattr(
        final,
        "_verify_canonical_branch_manifest",
        lambda path, **kwargs: kwargs["generic"],
    )
    monkeypatch.setattr(
        final,
        "_summarize_robustness",
        lambda manifest: {
            "members": 6,
            "cases_per_member": 41,
            "member_cases": 246,
            "minimum_macro_auroc_delta": {
                "value": -0.1,
                "member_id": "resnet1d-seed2026",
                "case_id": "noise",
            },
            "maximum_macro_brier_delta": {
                "value": 0.1,
                "member_id": "resnet1d-seed2026",
                "case_id": "noise",
            },
            "maximum_aurc_hamming_delta": {
                "value": 0.1,
                "member_id": "resnet1d-seed2026",
                "case_id": "noise",
            },
            "maximum_absolute_gate_coverage_delta": {
                "value": 0.1,
                "member_id": "resnet1d-seed2026",
                "case_id": "noise",
            },
        },
    )
    monkeypatch.setattr(
        final,
        "_summarize_explanations",
        lambda manifest: {
            "method_artifacts": 15,
            "method_ecg_evaluations": 900,
            "deterministic_repeat_exact_all": True,
            "minimum_mean_repeat_cosine": 1.0,
            "mean_stability_cosine_40db": 0.9,
            "mean_parameter_randomization_cosine": 0.2,
            "mean_guided_vs_random_deletion_advantage": 0.1,
            "maximum_fp32_vs_sealed_bf16_logit_drift": {
                "value": 0.003,
                "member_id": "resnet1d-seed2026",
                "method": "integrated_gradients",
            },
            "target_score_positive_cell": "+1_times_target_label_logit",
            "target_score_negative_cell": "-1_times_target_label_logit",
            "faithfulness_curve_probability": (
                "sigmoid(signed_correct_status_logit_over_frozen_temperature)"
            ),
            "attribution_orientation": "multiply_target_label_map_by_cell_sign",
            "attribution_precision": "float32",
            "sealed_confirmation_precision": "bf16",
            "precision_bridge": "fp32_attribution_vs_sealed_bf16_confirmation",
            "cross_method_pairs": 12,
            "cross_method_examples": 720,
            "cross_method_cosine_pairs_with_valid_values": 0,
            "cross_method_spearman_pairs_with_valid_values": 12,
            "cross_method_cosine_valid_examples": 0,
            "cross_method_cosine_invalid_examples": 720,
            "cross_method_spearman_valid_examples": 720,
            "cross_method_spearman_invalid_examples": 0,
            "mean_cross_method_cosine": None,
            "mean_cross_method_spearman": 0.4,
        },
    )
    monkeypatch.setattr(final, "_load_demo_binding", lambda *args, **kwargs: demo)
    monkeypatch.setattr(
        final,
        "_load_operational_run_log",
        lambda *args: (run_log, final.sha256_file(run_log)),
    )
    monkeypatch.setattr(
        final, "_load_architecture_summaries", lambda *args: _architecture_summaries()
    )
    monkeypatch.setattr(final, "_load_paired_reports", lambda *args: _paired_reports())

    protocol = ExperimentProtocol.canonical()
    first = final.finalize_results("ignored", protocol=protocol)
    text = first.report_path.read_text(encoding="utf-8")
    for required in (
        "confirmatory sealed fold-10 results",
        "post-evaluation descriptive analyses",
        "DEV-001",
        "bias the bootstrap ECE distribution",
        "batch_interrupted",
        "246 member-cases",
        "900 method–ECG evaluations",
        "signed correct-status score",
        "sealed BF16 precision",
        "FP32-attribution-to-sealed-BF16 precision bridge",
        "not estimable",
        "valid n=0/720",
        "0.003",
        "research-only",
        "no clinical or diagnostic claim",
    ):
        assert required in text
    manifest = final._load_self_hashed_json(
        first.manifest_path, expected_type=final.DERIVED_MANIFEST_TYPE
    )
    files = cast(list[dict[str, object]], manifest["files"])
    assert any(item["path"] == "reports/FINAL_RESULTS.md" for item in files)

    # Crash recovery adopts an exact report when the manifest was not yet published.
    expected_manifest_bytes = first.manifest_path.read_bytes()
    first.manifest_path.unlink()
    resumed = final.finalize_results("ignored", protocol=protocol)
    assert resumed.manifest_path.read_bytes() == expected_manifest_bytes

    second = final.finalize_results("ignored", protocol=protocol)
    assert second.to_dict() == first.to_dict()

    first.report_path.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(final.FinalResultsIntegrityError, match="crash-recovery file differs"):
        final.finalize_results("ignored", protocol=protocol)


def test_writes_never_escape_frozen_root(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    with pytest.raises(final.FinalResultsIntegrityError, match="escapes frozen output root"):
        final._write_new_bytes(tmp_path / "outside.txt", b"no", root=root)
    inside = root / "inside.txt"
    final._write_new_bytes(inside, b"first", root=root)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        final._write_new_bytes(inside, b"second", root=root)


def test_cli_exposes_only_probability_and_finalize_modes() -> None:
    parser = build_parser()
    assert parser.parse_args(["probability"]).mode == "probability"
    assert parser.parse_args(["finalize"]).mode == "finalize"
    with pytest.raises(SystemExit):
        parser.parse_args(["unknown"])
