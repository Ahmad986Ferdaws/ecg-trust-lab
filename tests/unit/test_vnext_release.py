from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import NoReturn

import pytest
from pydantic import ValidationError

import ecg_trust.vnext_release as vnext_release_module
from ecg_trust.registry import (
    TRUST_BUNDLE_PARENT_SCHEMA_VERSION,
    ArtifactRole,
    TrustBundleParent,
    bind_parent_file,
    sha256_file,
)
from ecg_trust.source_calibration import load_source_calibration_result_bytes
from ecg_trust.source_calibration.models import (
    LABEL_ORDER,
    ClaimBoundary,
    ConformalFitSummary,
    ConformalValidationSummary,
    EntropyGateSummary,
    EntropyValidationSummary,
    FrozenComponents,
    FrozenComponentsBody,
    LabelConformalCoverage,
    LabelConformalThreshold,
    LabelMetricSummary,
    LabelThresholdSummary,
    MacroMetricSummary,
    OpenWorldPendingSummary,
    PositiveRecords,
    RoleCounts,
    SourceCalibrationResult,
    SourceCalibrationResultBody,
    SourceProvenance,
    SourceRole,
    SourceValidationSummary,
    SplitEvidence,
    TemperatureFitSummary,
    ThresholdFitSummary,
    ThresholdValidationSummary,
    result_json_bytes,
    seal_frozen_components,
    seal_source_calibration_result,
)
from ecg_trust.vnext_release import (
    VNextReleaseAssemblyError,
    guard_vnext_release_assembly_v1,
)

NOW = datetime(2026, 8, 29, 12, 0, tzinfo=UTC)
REVISION = "a" * 40


def _digest(label: str) -> str:
    return "sha256:" + hashlib.sha256(label.encode()).hexdigest()


def _write_release_files(root: Path) -> None:
    payloads = {
        "checkpoint.ckpt": b"synthetic-checkpoint\n",
        "resolved-config.json": b'{"architecture":"resnet1d"}\n',
        "normalization.json": b'{"mean":[0.0],"std":[1.0]}\n',
        "decision-policy.json": b'{"gate":"fold9"}\n',
        "quality-policy.json": b'{"quality_gate":"frozen"}\n',
        "distribution-policy.json": b'{"detector":"shrinkage-mahalanobis"}\n',
        "conformal-policy.json": b'{"coverage_scope":"labelwise_marginal"}\n',
        "label-ontology.json": b'{"labels":["NORM","MI","STTC","CD","HYP"]}\n',
        "safety-document.md": b"Research use only.\n",
        "source-calibration-protocol.yaml": b"protocol: source-calibration-v1\n",
        "dataset-manifest.json": b'{"dataset":"PTB-XL 1.0.3"}\n',
        "uv.lock": b"version = 1\n",
    }
    for name, payload in payloads.items():
        (root / name).write_bytes(payload)


def _frozen_components() -> FrozenComponents:
    thresholds = tuple(
        LabelThresholdSummary(
            label=label,
            threshold=0.5,
            objective="f1",
            objective_value=0.7,
            positives=2,
            negatives=8,
            status="optimized",
        )
        for label in LABEL_ORDER
    )
    conformal_thresholds = tuple(
        LabelConformalThreshold(label=label, threshold=0.9) for label in LABEL_ORDER
    )
    return seal_frozen_components(
        FrozenComponentsBody(
            temperature=TemperatureFitSummary(
                method="single_positive_temperature_binary_nll",
                fit_role=SourceRole.DECISION_FIT,
                n_samples=10,
                temperature=1.1,
                nll_before=0.4,
                nll_after=0.3,
                status="optimized",
                converged=True,
                optimization_steps=5,
                fitted_labels=LABEL_ORDER,
                excluded_degenerate_labels=(),
            ),
            thresholds=ThresholdFitSummary(
                method="per_label_maximum_f1",
                tie_rule="maximum_f1_then_closest_to_0.5_then_higher_threshold",
                fit_role=SourceRole.DECISION_FIT,
                n_samples=10,
                macro_objective=0.7,
                per_label=thresholds,  # type: ignore[arg-type]
            ),
            entropy_gate=EntropyGateSummary(
                method="mean_normalized_binary_entropy",
                fit_role=SourceRole.DECISION_FIT,
                target_coverage=0.8,
                tie_rule="retain_all_scores_less_than_or_equal_to_frozen_order_statistic",
                maximum_entropy=0.6,
                selected_count=8,
                fit_count=10,
                achieved_coverage=0.8,
            ),
            conformal=ConformalFitSummary(
                method="labelwise_binary_split_conformal",
                fit_role=SourceRole.CONFORMAL_AND_OOD_THRESHOLD_FIT,
                alpha=0.1,
                n_samples=10,
                quantile_rank=10,
                quantile_level=1.0,
                coverage_scope="labelwise_marginal_under_exchangeability",
                individual_certainty_guarantee=False,
                per_label=conformal_thresholds,  # type: ignore[arg-type]
            ),
        )
    )


def _source_validation(component_sha256: str) -> SourceValidationSummary:
    per_label = tuple(
        LabelMetricSummary(
            label=label,
            positives=2,
            negatives=8,
            minimum_positive_records=1,
            statement_status="SUFFICIENT_EVIDENCE",
            roc_auc=0.8,
            average_precision=0.7,
            brier_score=0.1,
            ece15=0.05,
            degenerate_reason=None,
        )
        for label in LABEL_ORDER
    )
    conformal_per_label = tuple(
        LabelConformalCoverage(label=label, empirical_coverage=0.9, mean_set_size=1.0)
        for label in LABEL_ORDER
    )
    return SourceValidationSummary(
        evaluation_role=SourceRole.SOURCE_VALIDATION,
        tuning_allowed=False,
        records=10,
        patients=10,
        ece_bins=15,
        per_label=per_label,  # type: ignore[arg-type]
        macro=MacroMetricSummary(
            roc_auc=0.8,
            average_precision=0.7,
            brier_score=0.1,
            ece15=0.05,
            roc_auc_labels=5,
            average_precision_labels=5,
        ),
        threshold_decisions=ThresholdValidationSummary(
            frozen_component_sha256=component_sha256,
            hamming_loss=0.1,
            exact_match_accuracy=0.7,
        ),
        entropy_gate=EntropyValidationSummary(
            frozen_component_sha256=component_sha256,
            maximum_entropy=0.6,
            selected_count=8,
            validation_count=10,
            achieved_coverage=0.8,
            retained_hamming_loss=0.08,
            retained_exact_match_accuracy=0.8,
        ),
        conformal=ConformalValidationSummary(
            frozen_component_sha256=component_sha256,
            coverage_scope="labelwise_marginal_under_exchangeability",
            individual_certainty_guarantee=False,
            marginal_coverage=0.9,
            joint_sample_coverage=0.6,
            mean_set_size=1.0,
            singleton_fraction=1.0,
            empty_fraction=0.0,
            both_fraction=0.0,
            per_label=conformal_per_label,  # type: ignore[arg-type]
        ),
    )


def _source_result(
    root: Path,
    *,
    code_revision: str = REVISION,
) -> SourceCalibrationResult:
    components = _frozen_components()
    positives = PositiveRecords(NORM=2, MI=2, STTC=2, CD=2, HYP=2)
    roles = tuple(
        RoleCounts(role=role, records=10, patients=10, positive_records=positives)
        for role in SourceRole
    )
    provenance = SourceProvenance(
        config_file_sha256=sha256_file(root / "source-calibration-protocol.yaml"),
        source_npz_sha256=_digest("source-npz"),
        source_sidecar_sha256=_digest("source-sidecar"),
        prediction_artifact_sha256=_digest("prediction"),
        source_alignment_sha256=_digest("source-alignment"),
        source_bundle_sha256=_digest("source-bundle"),
        checkpoint_sha256=sha256_file(root / "checkpoint.ckpt"),
        demo_binding_file_sha256=_digest("demo-binding"),
        historical_policy_file_sha256=_digest("historical-policy"),
        experiment_protocol_sha256=_digest("experiment-protocol"),
        code_revision=code_revision,
        model_member_id="resnet1d-seed2026",
        source_artifact_model_name="resnet1d_refit_folds1-8_seed2026",
        architecture="resnet1d",
        seed=2026,
        source_fold=9,
    )
    open_world = OpenWorldPendingSummary(
        method="shrinkage_mahalanobis_embedding_distance",
        status="PENDING",
        artifact_sha256=None,
        threshold_fitted=False,
        source_false_rejection_evaluated=False,
        release_ready=False,
        reference_alignment_verified=False,
        embedding_device=None,
        embedding_precision=None,
        reason_code="REFERENCE_AND_THRESHOLD_EMBEDDINGS_NOT_PROVIDED",
    )
    body = SourceCalibrationResultBody(
        schema_version=1,
        artifact_type="ecg_trust.source_calibration_result",
        protocol_id="trust-sentinel-source-calibration-v1",
        status="PREPARED_NOT_RELEASE_READY",
        frozen_at_utc=NOW,
        provenance=provenance,
        split=SplitEvidence(
            unit="patient",
            algorithm="sha256_first8_uint64_fraction_v1",
            salt_sha256=_digest("salt"),
            assignment_sha256=_digest("assignments"),
            roles=roles,  # type: ignore[arg-type]
        ),
        frozen_components=components,
        source_validation=_source_validation(components.component_sha256),
        open_world=open_world,
        claims=ClaimBoundary(
            scope="retrospective_source_domain_development_only",
            research_only=True,
            clinical_validation=False,
            limitations=("research_only",),
        ),
    )
    return seal_source_calibration_result(body)


def _parents(root: Path) -> tuple[TrustBundleParent, ...]:
    declarations = (
        ("checkpoint", ArtifactRole.CHECKPOINT, "checkpoint.ckpt", "application/octet-stream"),
        (
            "resolved-config",
            ArtifactRole.RESOLVED_CONFIG,
            "resolved-config.json",
            "application/json",
        ),
        (
            "normalization",
            ArtifactRole.NORMALIZATION,
            "normalization.json",
            "application/json",
        ),
        (
            "decision-policy",
            ArtifactRole.DECISION_POLICY,
            "decision-policy.json",
            "application/json",
        ),
        (
            "quality-policy",
            ArtifactRole.QUALITY_POLICY,
            "quality-policy.json",
            "application/json",
        ),
        (
            "distribution-policy",
            ArtifactRole.DISTRIBUTION_POLICY,
            "distribution-policy.json",
            "application/json",
        ),
        (
            "conformal-policy",
            ArtifactRole.CONFORMAL_POLICY,
            "conformal-policy.json",
            "application/json",
        ),
        (
            "label-ontology",
            ArtifactRole.LABEL_ONTOLOGY,
            "label-ontology.json",
            "application/json",
        ),
        (
            "safety-document",
            ArtifactRole.SAFETY_DOCUMENT,
            "safety-document.md",
            "text/markdown",
        ),
        (
            "protocol",
            ArtifactRole.PROTOCOL,
            "source-calibration-protocol.yaml",
            "application/yaml",
        ),
        (
            "dataset-manifest",
            ArtifactRole.DATASET_MANIFEST,
            "dataset-manifest.json",
            "application/json",
        ),
        ("environment-lock", ArtifactRole.ENVIRONMENT_LOCK, "uv.lock", "text/plain"),
        (
            "source-calibration",
            ArtifactRole.EVIDENCE_BUNDLE,
            "source-calibration-result.json",
            "application/json",
        ),
    )
    return tuple(
        bind_parent_file(
            root,
            artifact_id=artifact_id,
            role=role,
            relative_path=relative_path,
            media_type=media_type,
        )
        for artifact_id, role, relative_path, media_type in declarations
    )


def _release_inputs(
    root: Path,
) -> tuple[SourceCalibrationResult, tuple[TrustBundleParent, ...]]:
    _write_release_files(root)
    result = _source_result(root)
    (root / "source-calibration-result.json").write_bytes(result_json_bytes(result))
    loaded = load_source_calibration_result_bytes(
        (root / "source-calibration-result.json").read_bytes()
    )
    return loaded, _parents(root)


def _guard(
    root: Path,
    result: SourceCalibrationResult,
    parents: tuple[TrustBundleParent, ...],
) -> NoReturn:
    guard_vnext_release_assembly_v1(
        source_calibration=result,
        source_calibration_relative_path="source-calibration-result.json",
        artifact_root=root,
        manifest_relative_path="trust-bundle.json",
        parents=parents,
    )


def test_pending_ood_is_rejected_before_manifest_construction(
    tmp_path: Path,
) -> None:
    result, parents = _release_inputs(tmp_path)

    with pytest.raises(VNextReleaseAssemblyError, match="not release-ready"):
        _guard(tmp_path, result, parents)

    assert not (tmp_path / "trust-bundle.json").exists()
    assert not hasattr(vnext_release_module, "seal_trust_bundle")
    assert not hasattr(vnext_release_module, "save_trust_bundle_manifest")
    assert not hasattr(vnext_release_module, "RegisteredVNextRelease")


def test_defensive_guard_has_no_success_path_if_readiness_assertion_regresses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, parents = _release_inputs(tmp_path)
    monkeypatch.setattr(
        vnext_release_module,
        "assert_complete_release_ready",
        lambda _: None,
    )

    with pytest.raises(VNextReleaseAssemblyError, match="new versioned OOD-completion"):
        _guard(tmp_path, result, parents)

    assert not (tmp_path / "trust-bundle.json").exists()


def test_frozen_v1_schema_has_no_ready_or_complete_state(tmp_path: Path) -> None:
    result, _ = _release_inputs(tmp_path)
    body = result.model_dump(mode="python", exclude={"artifact_sha256"})

    complete_status = dict(body)
    complete_status["status"] = "SOURCE_CALIBRATION_COMPLETE"
    with pytest.raises(ValidationError, match="PREPARED_NOT_RELEASE_READY"):
        SourceCalibrationResultBody.model_validate(complete_status)

    ready_ood = dict(body)
    ready_ood["open_world"] = {
        "method": "shrinkage_mahalanobis_embedding_distance",
        "status": "READY",
        "artifact_sha256": _digest("ood"),
        "threshold_fitted": True,
        "source_false_rejection_evaluated": True,
        "release_ready": True,
        "reference_alignment_verified": True,
        "embedding_device": "cuda:0",
        "embedding_precision": "float32",
        "reason_code": None,
    }
    with pytest.raises(ValidationError):
        SourceCalibrationResultBody.model_validate(ready_ood)

    assert result.status == "PREPARED_NOT_RELEASE_READY"
    assert result.open_world.status == "PENDING"
    assert result.open_world.release_ready is False


def test_rebound_parent_tampering_fails_before_registration(tmp_path: Path) -> None:
    result, parents = _release_inputs(tmp_path)
    quality_path = tmp_path / "quality-policy.json"
    quality_path.write_bytes(b"X" * quality_path.stat().st_size)

    with pytest.raises(VNextReleaseAssemblyError, match="changed after binding"):
        _guard(tmp_path, result, parents)

    assert not (tmp_path / "trust-bundle.json").exists()


def test_source_evidence_is_reloaded_and_cannot_differ_from_supplied_result(
    tmp_path: Path,
) -> None:
    result, _ = _release_inputs(tmp_path)
    evidence_path = tmp_path / "source-calibration-result.json"
    evidence_path.write_bytes(evidence_path.read_bytes() + b" ")
    rebound = _parents(tmp_path)

    with pytest.raises(VNextReleaseAssemblyError, match="canonical integrity"):
        _guard(tmp_path, result, rebound)

    assert not (tmp_path / "trust-bundle.json").exists()


def test_supplied_result_must_equal_canonical_evidence_bytes(tmp_path: Path) -> None:
    _, parents = _release_inputs(tmp_path)
    different_valid_result = _source_result(tmp_path, code_revision="b" * 40)

    with pytest.raises(VNextReleaseAssemblyError, match="differs from the registered evidence"):
        _guard(tmp_path, different_valid_result, parents)

    assert not (tmp_path / "trust-bundle.json").exists()


@pytest.mark.parametrize(
    "manifest_path",
    ("../trust-bundle.json", "/trust-bundle.json", "nested\\trust-bundle.json", "bundle.txt"),
)
def test_manifest_registration_path_is_strictly_confined(
    tmp_path: Path,
    manifest_path: str,
) -> None:
    result, parents = _release_inputs(tmp_path)

    with pytest.raises(VNextReleaseAssemblyError, match="manifest"):
        guard_vnext_release_assembly_v1(
            source_calibration=result,
            source_calibration_relative_path="source-calibration-result.json",
            artifact_root=tmp_path,
            manifest_relative_path=manifest_path,
            parents=parents,
        )

    assert not (tmp_path.parent / "trust-bundle.json").exists()


def test_source_calibration_parent_must_be_the_evidence_bundle(tmp_path: Path) -> None:
    result, parents = _release_inputs(tmp_path)
    alternate_evidence = tmp_path / "alternate-evidence.json"
    alternate_evidence.write_bytes(b'{"not":"source-calibration"}\n')
    wrong_evidence = TrustBundleParent(
        schema_version=TRUST_BUNDLE_PARENT_SCHEMA_VERSION,
        artifact_id="source-calibration",
        role=ArtifactRole.EVIDENCE_BUNDLE,
        relative_path="alternate-evidence.json",
        size_bytes=alternate_evidence.stat().st_size,
        file_sha256=sha256_file(alternate_evidence),
        media_type="application/json",
    )
    mismatched = tuple(
        wrong_evidence if parent.role is ArtifactRole.EVIDENCE_BUNDLE else parent
        for parent in parents
    )

    with pytest.raises(VNextReleaseAssemblyError, match="sole EVIDENCE_BUNDLE"):
        _guard(tmp_path, result, mismatched)
