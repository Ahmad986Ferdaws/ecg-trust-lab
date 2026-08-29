from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

import numpy as np
import pytest
from pydantic import ValidationError

import ecg_trust.ood_completion.models as contract_models
from ecg_trust.ood_completion import (
    DistributionPolicy,
    DistributionPolicyBinding,
    DistributionPolicyBody,
    EmbeddingContract,
    EmbeddingRuntimeSummary,
    MahalanobisDetectorArtifact,
    OODBundleMember,
    OODClaimBoundary,
    OODCompletionFailureCode,
    OODCompletionFailureReceiptBody,
    OODCompletionIntegrityError,
    OODCompletionResult,
    OODCompletionResultBody,
    OODCompletionSuccessManifestBody,
    OODIntegritySummary,
    OODLineageProvenance,
    OODPositiveEvaluationSummary,
    PatientClusterBootstrapInterval,
    ReferenceAndThresholdExecutionSummary,
    ReferenceEmbeddingExecutionSummary,
    ScoreQuantiles,
    SourceOODValidationSummary,
    ThresholdEmbeddingExecutionSummary,
    ThresholdFitSummary,
    assert_aggregate_only_ood_result,
    canonical_json_bytes,
    canonical_sha256,
    distribution_policy_json_bytes,
    load_distribution_policy_bytes,
    load_ood_completion_failure_bytes,
    load_ood_completion_result_bytes,
    load_ood_completion_success_bytes,
    ood_completion_failure_json_bytes,
    ood_completion_result_json_bytes,
    ood_completion_success_json_bytes,
    seal_distribution_policy,
    seal_ood_completion_failure_receipt,
    seal_ood_completion_result,
    seal_ood_completion_success_manifest,
)
from ecg_trust.open_world import ShrinkageMahalanobisDetector
from ecg_trust.source_calibration import models as source

NOW = datetime(2026, 8, 29, 12, 0, tzinfo=UTC)
REVISION = "a" * 40
LABELS: tuple[
    Literal["NORM"],
    Literal["MI"],
    Literal["STTC"],
    Literal["CD"],
    Literal["HYP"],
] = ("NORM", "MI", "STTC", "CD", "HYP")
LEAD_ORDER: tuple[
    Literal["I"],
    Literal["II"],
    Literal["III"],
    Literal["aVR"],
    Literal["aVL"],
    Literal["aVF"],
    Literal["V1"],
    Literal["V2"],
    Literal["V3"],
    Literal["V4"],
    Literal["V5"],
    Literal["V6"],
] = (
    "I",
    "II",
    "III",
    "aVR",
    "aVL",
    "aVF",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
)


def _digest(label: str) -> str:
    return "sha256:" + hashlib.sha256(label.encode("utf-8")).hexdigest()


def _file_digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _frozen_source_components() -> source.FrozenComponents:
    labels = LABELS
    thresholds = tuple(
        source.LabelThresholdSummary(
            label=label,
            threshold=0.5,
            objective="f1",
            objective_value=0.7,
            positives=4,
            negatives=16,
            status="optimized",
        )
        for label in labels
    )
    conformal_thresholds = tuple(
        source.LabelConformalThreshold(label=label, threshold=0.9) for label in labels
    )
    return source.seal_frozen_components(
        source.FrozenComponentsBody(
            temperature=source.TemperatureFitSummary(
                method="single_positive_temperature_binary_nll",
                fit_role=source.SourceRole.DECISION_FIT,
                n_samples=847,
                temperature=1.1,
                nll_before=0.4,
                nll_after=0.3,
                status="optimized",
                converged=True,
                optimization_steps=5,
                fitted_labels=labels,
                excluded_degenerate_labels=(),
            ),
            thresholds=source.ThresholdFitSummary(
                method="per_label_maximum_f1",
                tie_rule="maximum_f1_then_closest_to_0.5_then_higher_threshold",
                fit_role=source.SourceRole.DECISION_FIT,
                n_samples=847,
                macro_objective=0.7,
                per_label=thresholds,  # type: ignore[arg-type]
            ),
            entropy_gate=source.EntropyGateSummary(
                method="mean_normalized_binary_entropy",
                fit_role=source.SourceRole.DECISION_FIT,
                target_coverage=0.8,
                tie_rule=("retain_all_scores_less_than_or_equal_to_frozen_order_statistic"),
                maximum_entropy=0.6,
                selected_count=678,
                fit_count=847,
                achieved_coverage=0.8,
            ),
            conformal=source.ConformalFitSummary(
                method="labelwise_binary_split_conformal",
                fit_role=source.SourceRole.CONFORMAL_AND_OOD_THRESHOLD_FIT,
                alpha=0.1,
                n_samples=834,
                quantile_rank=752,
                quantile_level=float(752 / 834),
                coverage_scope="labelwise_marginal_under_exchangeability",
                individual_certainty_guarantee=False,
                per_label=conformal_thresholds,  # type: ignore[arg-type]
            ),
        )
    )


def _source_validation(component_sha256: str) -> source.SourceValidationSummary:
    labels = LABELS
    per_label = tuple(
        source.LabelMetricSummary(
            label=label,
            positives=4,
            negatives=461,
            minimum_positive_records=1,
            statement_status="SUFFICIENT_EVIDENCE",
            roc_auc=0.8,
            average_precision=0.7,
            brier_score=0.1,
            ece15=0.05,
            degenerate_reason=None,
        )
        for label in labels
    )
    conformal = tuple(
        source.LabelConformalCoverage(
            label=label,
            empirical_coverage=0.9,
            mean_set_size=1.0,
        )
        for label in labels
    )
    return source.SourceValidationSummary(
        evaluation_role=source.SourceRole.SOURCE_VALIDATION,
        tuning_allowed=False,
        records=465,
        patients=409,
        ece_bins=15,
        per_label=per_label,  # type: ignore[arg-type]
        macro=source.MacroMetricSummary(
            roc_auc=0.8,
            average_precision=0.7,
            brier_score=0.1,
            ece15=0.05,
            roc_auc_labels=5,
            average_precision_labels=5,
        ),
        threshold_decisions=source.ThresholdValidationSummary(
            frozen_component_sha256=component_sha256,
            hamming_loss=0.1,
            exact_match_accuracy=0.7,
        ),
        entropy_gate=source.EntropyValidationSummary(
            frozen_component_sha256=component_sha256,
            maximum_entropy=0.6,
            selected_count=372,
            validation_count=465,
            achieved_coverage=0.8,
            retained_hamming_loss=0.08,
            retained_exact_match_accuracy=0.8,
        ),
        conformal=source.ConformalValidationSummary(
            frozen_component_sha256=component_sha256,
            coverage_scope="labelwise_marginal_under_exchangeability",
            individual_certainty_guarantee=False,
            marginal_coverage=0.9,
            joint_sample_coverage=0.6,
            mean_set_size=1.0,
            singleton_fraction=1.0,
            empty_fraction=0.0,
            both_fraction=0.0,
            per_label=conformal,  # type: ignore[arg-type]
        ),
    )


def _source_result() -> source.SourceCalibrationResult:
    components = _frozen_source_components()
    roles = (
        source.RoleCounts(
            role=source.SourceRole.DECISION_FIT,
            records=847,
            patients=751,
            positive_records=source.PositiveRecords(NORM=354, MI=209, STTC=230, CD=202, HYP=112),
        ),
        source.RoleCounts(
            role=source.SourceRole.CONFORMAL_AND_OOD_THRESHOLD_FIT,
            records=834,
            patients=757,
            positive_records=source.PositiveRecords(NORM=385, MI=214, STTC=188, CD=189, HYP=111),
        ),
        source.RoleCounts(
            role=source.SourceRole.SOURCE_VALIDATION,
            records=465,
            patients=409,
            positive_records=source.PositiveRecords(NORM=216, MI=117, STTC=110, CD=104, HYP=45),
        ),
    )
    provenance = source.SourceProvenance(
        config_file_sha256=_digest("source-config"),
        source_npz_sha256=_digest("source-npz"),
        source_sidecar_sha256=_digest("source-sidecar"),
        prediction_artifact_sha256=_digest("prediction-artifact"),
        source_alignment_sha256=_digest("source-alignment"),
        source_bundle_sha256=_digest("source-bundle"),
        checkpoint_sha256=_digest("checkpoint"),
        demo_binding_file_sha256=_digest("demo-binding"),
        historical_policy_file_sha256=_digest("historical-policy"),
        experiment_protocol_sha256=_digest("experiment-protocol"),
        code_revision=REVISION,
        model_member_id="resnet1d-seed2026",
        source_artifact_model_name="resnet1d_refit_folds1-8_seed2026",
        architecture="resnet1d",
        seed=2026,
        source_fold=9,
    )
    body = source.SourceCalibrationResultBody(
        schema_version=1,
        artifact_type="ecg_trust.source_calibration_result",
        protocol_id="trust-sentinel-source-calibration-v1",
        status="PREPARED_NOT_RELEASE_READY",
        frozen_at_utc=NOW,
        provenance=provenance,
        split=source.SplitEvidence(
            unit="patient",
            algorithm="sha256_first8_uint64_fraction_v1",
            salt_sha256=_digest("split-salt"),
            assignment_sha256=_digest("split-assignments"),
            roles=roles,
        ),
        frozen_components=components,
        source_validation=_source_validation(components.component_sha256),
        open_world=source.OpenWorldPendingSummary(
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
        ),
        claims=source.ClaimBoundary(
            scope="retrospective_source_domain_development_only",
            research_only=True,
            clinical_validation=False,
            limitations=("research_only",),
        ),
    )
    return source.seal_source_calibration_result(body)


def _detector() -> tuple[ShrinkageMahalanobisDetector, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(2026)
    dimension = 512
    reference = rng.normal(size=(30, dimension))
    threshold = rng.normal(scale=0.05, size=(834, dimension))
    threshold_scores = np.einsum("ni,ni->n", threshold, threshold)
    quantile_rank = 794
    identity = tuple(
        tuple(1.0 if row == column else 0.0 for column in range(dimension))
        for row in range(dimension)
    )
    detector = ShrinkageMahalanobisDetector(
        mean=(0.0,) * dimension,
        precision=identity,
        threshold=float(np.sort(threshold_scores)[quantile_rank - 1]),
        embedding_dim=dimension,
        shrinkage=0.1,
        ridge=0.000001,
        inlier_coverage=0.95,
        n_fit_samples=17084,
        n_threshold_samples=834,
        quantile_rank=quantile_rank,
    )
    return detector, reference, threshold


def _lineage(source_result: source.SourceCalibrationResult) -> OODLineageProvenance:
    provenance = source_result.provenance
    return OODLineageProvenance(
        ood_config_file_sha256=_digest("ood-config"),
        source_calibration_artifact_sha256=source_result.artifact_sha256,
        source_calibration_file_sha256=_digest("source-file"),
        source_calibration_config_file_sha256=provenance.config_file_sha256,
        refit_completion_artifact_sha256=_digest("refit-completion-artifact"),
        refit_completion_file_sha256=_digest("refit-completion-file"),
        checkpoint_file_sha256=provenance.checkpoint_sha256,
        resolved_config_sha256=_digest("resolved-config-logical"),
        resolved_config_file_sha256=_digest("resolved-config-file"),
        dataset_manifest_file_sha256=_digest("manifest-file"),
        normalization_file_sha256=_digest("normalization-file"),
        experiment_protocol_sha256=provenance.experiment_protocol_sha256,
        environment_lock_file_sha256=_digest("environment-lock-file"),
        project_manifest_file_sha256=(
            "sha256:e1de755829678d588784bbcc34becc8c031c742d9b3f05458e76e67f577da3cd"
        ),
        raw_checksum_inventory_file_sha256=_digest("raw-checksums-file"),
        raw_selected_inventory_sha256=_digest("raw-selected-inventory"),
        selected_record_count=18383,
        selected_file_count=36766,
        raw_reference_inventory_sha256=_digest("raw-reference-inventory"),
        raw_source_inventory_sha256=_digest("raw-source-inventory"),
        code_revision=provenance.code_revision,
        model_member_id=provenance.model_member_id,
        architecture=provenance.architecture,
        seed=provenance.seed,
    )


def _policy(
    detector: ShrinkageMahalanobisDetector,
    provenance: OODLineageProvenance,
) -> DistributionPolicy:
    return seal_distribution_policy(
        DistributionPolicyBody(
            schema_version=1,
            artifact_type="ecg_trust.distribution_policy",
            protocol_id="trust-sentinel-ood-completion-v1",
            method="shrinkage_mahalanobis_embedding_distance",
            score_direction="higher_is_more_out_of_distribution",
            threshold_comparison="score_strictly_greater_than_threshold",
            embedding_contract=EmbeddingContract(
                architecture="resnet1d",
                extraction_point="frozen_resnet_preclassifier_global_average_pool",
                pooling="adaptive_average_pool_1d",
                embedding_dimension=512,
                tensor_dtype="float32",
                detector_numeric_dtype="float64",
                lead_order=LEAD_ORDER,
                sampling_frequency_hz=100.0,
                samples_per_lead=1000,
                physical_units="mV",
            ),
            detector=MahalanobisDetectorArtifact.from_detector(detector),
            provenance=provenance,
            research_only=True,
        )
    )


def _runtime(
    *, device_name: str = "NVIDIA GeForce RTX 5070 Ti Laptop GPU"
) -> EmbeddingRuntimeSummary:
    return EmbeddingRuntimeSummary(
        requested_device="cuda:0",
        resolved_device="cuda:0",
        device_type="cuda",
        device_name=device_name,
        compute_capability="12.0",
        python_version="3.12.13",
        torch_version="2.13.0+cu130",
        cuda_runtime_version="13.0",
        cudnn_version=92000,
        nvidia_driver_version="596.49",
        tensor_precision="float32",
        autocast_enabled=False,
        bf16_enabled=False,
        tf32_enabled=False,
        deterministic_algorithms=True,
        cudnn_deterministic=True,
        cudnn_benchmark=False,
        cublas_workspace_config=":4096:8",
        inference_mode=True,
        model_eval_mode=True,
        shuffled=False,
        batch_size=128,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        extraction_passes=2,
        torch_compile_enabled=False,
        seed=2026,
    )


def _quantiles(scores: np.ndarray) -> ScoreQuantiles:
    return ScoreQuantiles(
        min=float(np.min(scores)),
        p01=float(np.quantile(scores, 0.01)),
        p05=float(np.quantile(scores, 0.05)),
        p25=float(np.quantile(scores, 0.25)),
        p50=float(np.quantile(scores, 0.50)),
        p75=float(np.quantile(scores, 0.75)),
        p90=float(np.quantile(scores, 0.90)),
        p95=float(np.quantile(scores, 0.95)),
        p99=float(np.quantile(scores, 0.99)),
        max=float(np.max(scores)),
    )


def _scaled_quantiles(threshold: float, *, maximum_scale: float) -> ScoreQuantiles:
    scales = (0.0, 0.01, 0.05, 0.25, 0.5, 0.7, 0.8, 0.85, 0.9, maximum_scale)
    values = tuple(float(threshold * scale) for scale in scales)
    return ScoreQuantiles(
        min=values[0],
        p01=values[1],
        p05=values[2],
        p25=values[3],
        p50=values[4],
        p75=values[5],
        p90=values[6],
        p95=values[7],
        p99=values[8],
        max=values[9],
    )


def _execution(
    source_result: source.SourceCalibrationResult,
    detector: ShrinkageMahalanobisDetector,
    runtime: EmbeddingRuntimeSummary,
) -> ReferenceAndThresholdExecutionSummary:
    runtime_sha256 = canonical_sha256(runtime.model_dump(mode="json"))
    return ReferenceAndThresholdExecutionSummary(
        runtime=runtime,
        reference=ReferenceEmbeddingExecutionSummary(
            records=detector.n_fit_samples,
            patients=14823,
            embedding_dimension=512,
            alignment_sha256=_digest("reference-alignment"),
            embedding_tensor_sha256=_digest("reference-tensor"),
            repeated_embedding_tensor_sha256=_digest("reference-tensor"),
            embedding_artifact_sha256=_digest("reference-artifact"),
            embedding_npz_file_sha256=_digest("reference-npz-file"),
            embedding_sidecar_file_sha256=_digest("reference-sidecar-file"),
            runtime_sha256=runtime_sha256,
            exact_repeat_verified=True,
            public_contains_row_arrays=False,
            role="ptbxl_folds_1_to_8_training_reference",
            folds=(1, 2, 3, 4, 5, 6, 7, 8),
        ),
        threshold_fit=ThresholdEmbeddingExecutionSummary(
            records=detector.n_threshold_samples,
            patients=757,
            embedding_dimension=512,
            alignment_sha256=_digest("threshold-alignment"),
            embedding_tensor_sha256=_digest("threshold-tensor"),
            repeated_embedding_tensor_sha256=_digest("threshold-tensor"),
            embedding_artifact_sha256=_digest("threshold-artifact"),
            embedding_npz_file_sha256=_digest("threshold-npz-file"),
            embedding_sidecar_file_sha256=_digest("threshold-sidecar-file"),
            runtime_sha256=runtime_sha256,
            exact_repeat_verified=True,
            public_contains_row_arrays=False,
            role="conformal_and_ood_threshold_fit",
            folds=(9,),
            source_assignment_sha256=source_result.split.assignment_sha256,
        ),
    )


def _threshold_fit(
    detector: ShrinkageMahalanobisDetector,
    threshold_embeddings: np.ndarray,
) -> ThresholdFitSummary:
    scores = detector.score(threshold_embeddings)
    rejected = int(np.sum(scores > detector.threshold))
    accepted = detector.n_threshold_samples - rejected
    return ThresholdFitSummary(
        method="shrinkage_mahalanobis_embedding_distance",
        fit_role="conformal_and_ood_threshold_fit",
        n_reference_samples=detector.n_fit_samples,
        n_threshold_samples=detector.n_threshold_samples,
        embedding_dimension=512,
        shrinkage=detector.shrinkage,
        ridge=detector.ridge,
        target_inlier_coverage=0.95,
        quantile_rank=detector.quantile_rank,
        threshold=detector.threshold,
        threshold_comparison="score_strictly_greater_than_threshold",
        observed_inlier_count=accepted,
        observed_rejection_count=rejected,
        observed_inlier_fraction=float(accepted / detector.n_threshold_samples),
        observed_false_rejection_rate=float(rejected / detector.n_threshold_samples),
        score_quantiles=_quantiles(scores),
    )


def _ood_validation(
    source_result: source.SourceCalibrationResult,
    detector: ShrinkageMahalanobisDetector,
    runtime: EmbeddingRuntimeSummary,
    *,
    eligible: bool,
) -> SourceOODValidationSummary:
    rejected = 0 if eligible else 1
    point = float(rejected / 465)
    patient_any_rate = float(rejected / 409)
    upper = 0.04 if eligible else 0.12
    runtime_sha256 = canonical_sha256(runtime.model_dump(mode="json"))
    return SourceOODValidationSummary(
        records=465,
        patients=409,
        embedding_dimension=512,
        alignment_sha256=_digest("validation-alignment"),
        embedding_tensor_sha256=_digest("validation-tensor"),
        repeated_embedding_tensor_sha256=_digest("validation-tensor"),
        embedding_artifact_sha256=_digest("validation-artifact"),
        embedding_npz_file_sha256=_digest("validation-npz-file"),
        embedding_sidecar_file_sha256=_digest("validation-sidecar-file"),
        runtime_sha256=runtime_sha256,
        exact_repeat_verified=True,
        public_contains_row_arrays=False,
        evaluation_role="source_validation",
        tuning_allowed=False,
        rejected_records=rejected,
        accepted_records=465 - rejected,
        rejected_patients_any=rejected,
        record_false_rejection_rate=point,
        source_record_support_coverage=float((465 - rejected) / 465),
        patient_equalized_false_rejection_rate=patient_any_rate,
        patient_any_false_rejection_rate=patient_any_rate,
        maximum_allowed_record_false_rejection_rate=0.05,
        cluster_bootstrap=PatientClusterBootstrapInterval(
            method="patient_cluster_percentile_bootstrap",
            estimator="record_false_rejection_rate",
            resampling_unit="patient",
            sampling_with_replacement=True,
            seed=20260829,
            replicates=10000,
            two_sided_confidence_level=0.95,
            two_sided_lower=0.0,
            two_sided_upper=upper,
            one_sided_confidence_level=0.95,
            one_sided_upper=upper,
        ),
        score_quantiles=_scaled_quantiles(
            detector.threshold,
            maximum_scale=0.9 if eligible else 1.2,
        ),
        threshold=detector.threshold,
        threshold_tie_count=0,
        threshold_comparison="score_strictly_greater_than_threshold",
        source_assignment_sha256=source_result.split.assignment_sha256,
        target_met=eligible,
    )


def _integrity() -> OODIntegritySummary:
    return OODIntegritySummary(
        complete=True,
        verified_input_hashes=True,
        source_calibration_verified=True,
        refit_lineage_verified=True,
        waveform_checksums_verified_before_and_after=True,
        source_alignment_verified=True,
        patient_roles_disjoint=True,
        forbidden_sources_not_accessed=True,
        model_state_unchanged=True,
        deterministic_repeat_verified=True,
        distribution_policy_verified=True,
        aggregate_only_result_verified=True,
    )


def _claims() -> OODClaimBoundary:
    return OODClaimBoundary(
        scope="retrospective_ptbxl_source_domain_development_only",
        research_only=True,
        clinical_validation=False,
        external_ood_positive_validation=False,
        limitations=(
            "no_external_lockbox_evaluation",
            "no_ood_positive_evaluation",
            "no_clinical_validation",
            "ood_score_does_not_identify_unknown_disease",
            "source_false_rejection_target_is_provisional",
            "research_bundle_eligibility_is_not_clinical_release_readiness",
        ),
    )


def _not_evaluated() -> OODPositiveEvaluationSummary:
    return OODPositiveEvaluationSummary(
        status="NOT_EVALUATED",
        evaluation_sources=(),
        records=0,
        semantic_ood_recall="NOT_EVALUATED",
        severe_ood_recall="NOT_EVALUATED",
        ood_auroc="NOT_EVALUATED",
        ood_average_precision="NOT_EVALUATED",
        unseen_site_or_device_performance="NOT_EVALUATED",
        target_site_fitting_performed=False,
        reason_code="NO_FROZEN_OOD_POSITIVE_EVALUATION_SOURCE",
    )


@dataclass(frozen=True)
class _Artifacts:
    source_result: source.SourceCalibrationResult
    detector: ShrinkageMahalanobisDetector
    reference_embeddings: np.ndarray
    threshold_embeddings: np.ndarray
    provenance: OODLineageProvenance
    policy: DistributionPolicy
    body: OODCompletionResultBody
    result: OODCompletionResult


def _artifacts(
    *,
    eligible: bool = True,
    device_name: str = "NVIDIA GeForce RTX 5070 Ti Laptop GPU",
) -> _Artifacts:
    source_result = _source_result()
    detector, reference, threshold = _detector()
    provenance = _lineage(source_result)
    policy = _policy(detector, provenance)
    policy_bytes = distribution_policy_json_bytes(policy)
    runtime = _runtime(device_name=device_name)
    body = OODCompletionResultBody(
        schema_version=1,
        artifact_type="ecg_trust.ood_completion_result",
        protocol_id="trust-sentinel-ood-completion-v1",
        status=(
            "SOURCE_SUPPORT_GATE_COMPLETE"
            if eligible
            else "SOURCE_SUPPORT_GATE_TARGET_MISSED"
        ),
        frozen_at_utc=NOW,
        source_calibration=source_result,
        provenance=provenance,
        distribution_policy=DistributionPolicyBinding(
            filename="distribution-policy.json",
            artifact_sha256=policy.artifact_sha256,
            file_sha256=_file_digest(policy_bytes),
            size_bytes=len(policy_bytes),
        ),
        reference_and_threshold_execution=_execution(source_result, detector, runtime),
        threshold_fit=_threshold_fit(detector, threshold),
        source_validation=_ood_validation(
            source_result,
            detector,
            runtime,
            eligible=eligible,
        ),
        ood_positive_evaluation=_not_evaluated(),
        integrity=_integrity(),
        research_bundle_eligible=eligible,
        claims=_claims(),
    )
    return _Artifacts(
        source_result=source_result,
        detector=detector,
        reference_embeddings=reference,
        threshold_embeddings=threshold,
        provenance=provenance,
        policy=policy,
        body=body,
        result=seal_ood_completion_result(body),
    )


@pytest.fixture(scope="module")
def artifacts() -> _Artifacts:
    return _artifacts()


def _revalidate_body(body: OODCompletionResultBody, **updates: object) -> OODCompletionResultBody:
    payload = body.model_dump(mode="python")
    payload.update(updates)
    return OODCompletionResultBody.model_validate(payload)


def test_distribution_policy_is_canonical_self_hashed_and_restores_detector(
    artifacts: _Artifacts,
) -> None:
    payload = distribution_policy_json_bytes(artifacts.policy)
    loaded = load_distribution_policy_bytes(payload)

    assert payload.endswith(b"\n")
    assert payload == canonical_json_bytes(loaded.model_dump(mode="json")) + b"\n"
    assert loaded.artifact_sha256 == canonical_sha256(
        loaded.model_dump(mode="json", exclude={"artifact_sha256"})
    )
    np.testing.assert_array_equal(
        loaded.to_detector().score(artifacts.reference_embeddings),
        artifacts.detector.score(artifacts.reference_embeddings),
    )


def test_distribution_policy_loader_rejects_tamper_and_noncanonical_bytes(
    artifacts: _Artifacts,
) -> None:
    payload = distribution_policy_json_bytes(artifacts.policy)
    tampered = json.loads(payload)
    tampered["detector"]["threshold"] += 1.0

    with pytest.raises(OODCompletionIntegrityError):
        load_distribution_policy_bytes(canonical_json_bytes(tampered) + b"\n")
    with pytest.raises(OODCompletionIntegrityError):
        load_distribution_policy_bytes(b" " + payload)


def test_completion_result_round_trip_preserves_pending_source_v1(
    artifacts: _Artifacts,
) -> None:
    payload = ood_completion_result_json_bytes(artifacts.result)
    loaded = load_ood_completion_result_bytes(payload)

    assert loaded == artifacts.result
    assert loaded.artifact_sha256 == canonical_sha256(
        loaded.model_dump(mode="json", exclude={"artifact_sha256"})
    )
    assert source.result_json_bytes(loaded.source_calibration) == source.result_json_bytes(
        artifacts.source_result
    )
    assert loaded.source_calibration.open_world.status == "PENDING"
    assert loaded.source_calibration.open_world.release_ready is False
    assert "release_ready" not in loaded.model_fields_set
    assert loaded.distribution_policy.artifact_sha256 == artifacts.policy.artifact_sha256
    assert loaded.distribution_policy.file_sha256 == _file_digest(
        distribution_policy_json_bytes(artifacts.policy)
    )
    assert loaded.ood_positive_evaluation.status == "NOT_EVALUATED"
    contract_models._assert_result_source_support_eligible(loaded)


def test_completion_result_rejects_tamper_and_unsafe_model_copy(
    artifacts: _Artifacts,
) -> None:
    payload = ood_completion_result_json_bytes(artifacts.result)
    tampered = json.loads(payload)
    tampered["research_bundle_eligible"] = False

    with pytest.raises(OODCompletionIntegrityError):
        load_ood_completion_result_bytes(canonical_json_bytes(tampered) + b"\n")
    with pytest.raises(OODCompletionIntegrityError):
        load_ood_completion_result_bytes(payload[:-1])

    unsafe = artifacts.result.model_copy(update={"artifact_sha256": _digest("wrong")})
    with pytest.raises(ValidationError):
        ood_completion_result_json_bytes(unsafe)


def test_status_and_eligibility_are_derived_from_one_sided_upper_bound(
    artifacts: _Artifacts,
) -> None:
    assert artifacts.result.status == "SOURCE_SUPPORT_GATE_COMPLETE"
    assert artifacts.result.research_bundle_eligible is True

    with pytest.raises(ValidationError):
        _revalidate_body(artifacts.body, status="SOURCE_SUPPORT_GATE_TARGET_MISSED")
    with pytest.raises(ValidationError):
        _revalidate_body(artifacts.body, research_bundle_eligible=False)

    missed = _artifacts(eligible=False).result
    assert missed.status == "SOURCE_SUPPORT_GATE_TARGET_MISSED"
    assert missed.research_bundle_eligible is False
    with pytest.raises(OODCompletionIntegrityError):
        contract_models._assert_result_source_support_eligible(missed)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("record_false_rejection_rate", 0.2),
        ("source_record_support_coverage", 0.2),
        ("patient_any_false_rejection_rate", 0.2),
        ("threshold_tie_count", 466),
        ("target_met", False),
        ("repeated_embedding_tensor_sha256", _digest("different-repeat")),
    ),
)
def test_source_validation_rejects_inconsistent_metrics_and_execution_evidence(
    artifacts: _Artifacts,
    field: str,
    value: object,
) -> None:
    payload = artifacts.body.source_validation.model_dump(mode="python")
    payload[field] = value

    with pytest.raises(ValidationError):
        SourceOODValidationSummary.model_validate(payload)


def test_score_quantiles_require_p90_and_monotonic_order(artifacts: _Artifacts) -> None:
    payload = artifacts.body.source_validation.score_quantiles.model_dump(mode="python")
    payload.pop("p90")
    with pytest.raises(ValidationError):
        ScoreQuantiles.model_validate(payload)

    payload = artifacts.body.source_validation.score_quantiles.model_dump(mode="python")
    payload["p90"] = payload["p95"] + 1.0
    with pytest.raises(ValidationError):
        ScoreQuantiles.model_validate(payload)


def test_embedding_dimension_is_fixed_to_512(artifacts: _Artifacts) -> None:
    contract = artifacts.policy.embedding_contract.model_dump(mode="python")
    contract["embedding_dimension"] = 511
    with pytest.raises(ValidationError):
        EmbeddingContract.model_validate(contract)

    validation = artifacts.body.source_validation.model_dump(mode="python")
    validation["embedding_dimension"] = 511
    with pytest.raises(ValidationError):
        SourceOODValidationSummary.model_validate(validation)


def test_bootstrap_contract_fixes_generator_seed_resamples_and_interval_order(
    artifacts: _Artifacts,
) -> None:
    interval = artifacts.body.source_validation.cluster_bootstrap
    assert interval.random_generator == "numpy.random.Generator_PCG64"
    assert interval.seed == 20260829
    assert interval.replicates == 10000
    assert interval.quantile_method == "linear"

    payload = interval.model_dump(mode="python")
    payload["seed"] = 2026
    with pytest.raises(ValidationError):
        PatientClusterBootstrapInterval.model_validate(payload)

    payload = interval.model_dump(mode="python")
    payload["one_sided_upper"] = payload["two_sided_upper"] + 0.01
    with pytest.raises(ValidationError):
        PatientClusterBootstrapInterval.model_validate(payload)


def test_percentile_interval_is_not_required_to_contain_the_point_estimate(
    artifacts: _Artifacts,
) -> None:
    payload = artifacts.body.source_validation.model_dump(mode="python")
    payload["cluster_bootstrap"]["two_sided_lower"] = 0.01

    validated = SourceOODValidationSummary.model_validate(payload)

    assert validated.record_false_rejection_rate == 0.0
    assert validated.cluster_bootstrap.two_sided_lower == 0.01


def test_aggregate_only_privacy_scanner_rejects_paths_and_row_identifiers(
    artifacts: _Artifacts,
) -> None:
    assert_aggregate_only_ood_result(artifacts.result)
    serialized = artifacts.result.model_dump(mode="json")

    def keys(value: object) -> set[str]:
        if isinstance(value, dict):
            return set(value).union(*(keys(child) for child in value.values()))
        if isinstance(value, list):
            return set().union(*(keys(child) for child in value))
        return set()

    assert {"patient_id", "embeddings", "scores", "rows"}.isdisjoint(keys(serialized))

    with pytest.raises(ValidationError):
        _artifacts(device_name=r"C:\private\gpu")

    injected = artifacts.result.model_dump(mode="json")
    injected["patient_id"] = 123
    with pytest.raises(OODCompletionIntegrityError):
        load_ood_completion_result_bytes(canonical_json_bytes(injected) + b"\n")


def test_nested_source_v1_cannot_be_rewritten_as_complete(artifacts: _Artifacts) -> None:
    nested = artifacts.source_result.model_dump(mode="python")
    nested["status"] = "SOURCE_OOD_GATE_COMPLETE"
    body = artifacts.body.model_dump(mode="python")
    body["source_calibration"] = nested

    with pytest.raises(ValidationError):
        OODCompletionResultBody.model_validate(body)


def test_runtime_contract_is_exact_and_bound_across_r_b_and_c(artifacts: _Artifacts) -> None:
    execution = artifacts.body.reference_and_threshold_execution
    runtime = execution.runtime
    expected = canonical_sha256(runtime.model_dump(mode="json"))

    assert runtime.batch_size == 128
    assert runtime.num_workers == 4
    assert runtime.pin_memory is True
    assert runtime.persistent_workers is True
    assert runtime.extraction_passes == 2
    assert runtime.torch_compile_enabled is False
    assert runtime.python_version == "3.12.13"
    assert runtime.nvidia_driver_version == "596.49"
    assert execution.reference.runtime_sha256 == expected
    assert execution.threshold_fit.runtime_sha256 == expected
    assert artifacts.body.source_validation.runtime_sha256 == expected

    payload = execution.model_dump(mode="python")
    payload["runtime"] = {**payload["runtime"], "batch_size": 64}
    with pytest.raises(ValidationError):
        ReferenceAndThresholdExecutionSummary.model_validate(payload)

    runtime_payload = runtime.model_dump(mode="python")
    runtime_payload["python_version"] = "3.12.14"
    with pytest.raises(ValidationError):
        EmbeddingRuntimeSummary.model_validate(runtime_payload)

    runtime_payload = runtime.model_dump(mode="python")
    runtime_payload["nvidia_driver_version"] = "596.50"
    with pytest.raises(ValidationError):
        EmbeddingRuntimeSummary.model_validate(runtime_payload)

    validation = artifacts.body.source_validation.model_dump(mode="python")
    validation["runtime_sha256"] = _digest("different-runtime")
    with pytest.raises(ValidationError):
        _revalidate_body(artifacts.body, source_validation=validation)


def test_lineage_binds_project_manifest_and_selected_checksum_subset(
    artifacts: _Artifacts,
) -> None:
    provenance = artifacts.provenance
    assert provenance.project_manifest_file_sha256 == (
        "sha256:e1de755829678d588784bbcc34becc8c031c742d9b3f05458e76e67f577da3cd"
    )
    assert provenance.raw_selected_inventory_sha256 == _digest("raw-selected-inventory")
    assert provenance.selected_record_count == 18383
    assert provenance.selected_file_count == 36766

    payload = provenance.model_dump(mode="python")
    payload["selected_record_count"] = 18382
    with pytest.raises(ValidationError):
        OODLineageProvenance.model_validate(payload)

    payload = provenance.model_dump(mode="python")
    payload["selected_file_count"] = 36765
    with pytest.raises(ValidationError):
        OODLineageProvenance.model_validate(payload)

    payload = provenance.model_dump(mode="python")
    payload["project_manifest_file_sha256"] = _digest("different-project-manifest")
    with pytest.raises(ValidationError):
        OODLineageProvenance.model_validate(payload)


def test_integrity_must_be_complete_before_any_result_exists(artifacts: _Artifacts) -> None:
    integrity = artifacts.body.integrity.model_dump(mode="python")
    integrity["deterministic_repeat_verified"] = False

    with pytest.raises(ValidationError):
        _revalidate_body(artifacts.body, integrity=integrity)


def test_sanitized_failure_receipt_is_canonical_and_tamper_evident() -> None:
    receipt = seal_ood_completion_failure_receipt(
        OODCompletionFailureReceiptBody(
            schema_version=1,
            artifact_type="ecg_trust.ood_completion_failure",
            protocol_id="trust-sentinel-ood-completion-v1",
            status="FAILED",
            frozen_at_utc=NOW,
            config_file_sha256=_digest("ood-config"),
            code_revision=REVISION,
            failure_code=OODCompletionFailureCode.DETERMINISM_FAILED,
            contains_raw_ids_or_rows=False,
            contains_embeddings=False,
            contains_filesystem_paths=False,
            retry_requires_new_output_root=True,
        )
    )
    payload = ood_completion_failure_json_bytes(receipt)
    assert load_ood_completion_failure_bytes(payload) == receipt
    assert "error" not in json.dumps(receipt.model_dump(mode="json"))

    tampered = json.loads(payload)
    tampered["failure_code"] = "INTERNAL_FAILURE"
    with pytest.raises(OODCompletionIntegrityError):
        load_ood_completion_failure_bytes(canonical_json_bytes(tampered) + b"\n")

    body = receipt.model_dump(mode="python", exclude={"artifact_sha256"})
    body["error_message"] = r"C:\private\record.npz"
    with pytest.raises(ValidationError):
        OODCompletionFailureReceiptBody.model_validate(body)


def test_success_manifest_is_exact_ordered_self_hashed_and_canonical() -> None:
    paths = (
        "distribution-policy.json",
        "ood-completion-result.json",
        "private/reference-embeddings.json",
        "private/reference-embeddings.npz",
        "private/source-validation-embeddings.json",
        "private/source-validation-embeddings.npz",
        "private/threshold-fit-embeddings.json",
        "private/threshold-fit-embeddings.npz",
        "source-validation-access-armed.json",
    )
    members = tuple(
        OODBundleMember.model_validate(
            {
                "relative_path": path,
                "size_bytes": index + 1,
                "file_sha256": _digest(path),
            }
        )
        for index, path in enumerate(paths)
    )
    body = OODCompletionSuccessManifestBody(
        schema_version=1,
        artifact_type="ecg_trust.ood_completion_success_manifest",
        protocol_id="trust-sentinel-ood-completion-v1",
        status="SUCCESS",
        frozen_at_utc=NOW,
        config_file_sha256=_digest("config"),
        code_revision=REVISION,
        result_artifact_sha256=_digest("result"),
        distribution_policy_artifact_sha256=_digest("policy"),
        validation_access_claim_filename=(
            ".ood_completion_v1.source-validation-one-shot-claim.json"
        ),
        validation_access_claim_file_sha256=_digest("claim"),
        member_count=9,
        members=members,
        terminal_checks_complete=True,
        failure_receipt_present=False,
    )
    manifest = seal_ood_completion_success_manifest(body)
    payload = ood_completion_success_json_bytes(manifest)

    assert load_ood_completion_success_bytes(payload) == manifest
    assert payload.endswith(b"\n") and b"\r" not in payload

    reversed_payload = body.model_dump(mode="python")
    reversed_payload["members"] = tuple(reversed(members))
    with pytest.raises(ValidationError):
        OODCompletionSuccessManifestBody.model_validate(reversed_payload)

    tampered = json.loads(payload)
    tampered["status"] = "FAILED"
    with pytest.raises(OODCompletionIntegrityError):
        load_ood_completion_success_bytes(canonical_json_bytes(tampered) + b"\n")


def test_canonical_loaders_enforce_artifact_specific_size_bounds(
    artifacts: _Artifacts,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy_payload = distribution_policy_json_bytes(artifacts.policy)
    result_payload = ood_completion_result_json_bytes(artifacts.result)
    receipt = seal_ood_completion_failure_receipt(
        OODCompletionFailureReceiptBody(
            schema_version=1,
            artifact_type="ecg_trust.ood_completion_failure",
            protocol_id="trust-sentinel-ood-completion-v1",
            status="FAILED",
            frozen_at_utc=NOW,
            config_file_sha256=_digest("ood-config"),
            code_revision=REVISION,
            failure_code=OODCompletionFailureCode.INPUT_CONTRACT_FAILED,
            contains_raw_ids_or_rows=False,
            contains_embeddings=False,
            contains_filesystem_paths=False,
            retry_requires_new_output_root=True,
        )
    )
    failure_payload = ood_completion_failure_json_bytes(receipt)

    monkeypatch.setattr(
        contract_models,
        "MAX_DISTRIBUTION_POLICY_BYTES",
        len(policy_payload) - 1,
    )
    monkeypatch.setattr(
        contract_models,
        "MAX_OOD_COMPLETION_RESULT_BYTES",
        len(result_payload) - 1,
    )
    monkeypatch.setattr(
        contract_models,
        "MAX_OOD_COMPLETION_FAILURE_BYTES",
        len(failure_payload) - 1,
    )

    with pytest.raises(OODCompletionIntegrityError):
        load_distribution_policy_bytes(policy_payload)
    with pytest.raises(OODCompletionIntegrityError):
        load_ood_completion_result_bytes(result_payload)
    with pytest.raises(OODCompletionIntegrityError):
        load_ood_completion_failure_bytes(failure_payload)
