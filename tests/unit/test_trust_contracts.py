from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from ecg_trust.constants import LEADS
from ecg_trust.contracts import (
    ARTIFACT_REFERENCE_SCHEMA_VERSION,
    AUDIT_EVENT_SCHEMA_VERSION,
    CANONICAL_SIGNAL_BYTES,
    CANONICAL_SIGNAL_MEDIA_TYPE,
    CASE_DISTRIBUTION_ASSESSMENT_SCHEMA_VERSION,
    CONFIDENCE_INTERVAL_SCHEMA_VERSION,
    CONFORMAL_COVERAGE_SCOPE,
    CONFORMAL_COVERAGE_SCOPE_TEXT,
    DISTRIBUTION_METRIC_SCHEMA_VERSION,
    DISTRIBUTION_REPORT_SCHEMA_VERSION,
    EVIDENCE_BUNDLE_SCHEMA_VERSION,
    EVIDENCE_COHORT_SCHEMA_VERSION,
    EVIDENCE_METRIC_SCHEMA_VERSION,
    EXPLANATION_ENVELOPE_SCHEMA_VERSION,
    LABEL_PREDICTION_SET_DECISION_SCHEMA_VERSION,
    LABEL_UNCERTAINTY_SCHEMA_VERSION,
    QUALITY_FINDING_SCHEMA_VERSION,
    QUALITY_REPORT_SCHEMA_VERSION,
    SIGNAL_ENVELOPE_SCHEMA_VERSION,
    ArtifactReference,
    AuditAction,
    AuditActorType,
    AuditEvent,
    AuditOutcome,
    CaseDistributionAssessment,
    CaseDistributionStatus,
    ConfidenceInterval,
    DistributionMetric,
    DistributionReport,
    DistributionStatus,
    EvidenceBundle,
    EvidenceCohort,
    EvidenceMetric,
    ExplanationEnvelope,
    ExplanationMethod,
    FindingSeverity,
    LabelPredictionSetDecision,
    LabelUncertaintyDecision,
    MetricDirection,
    PredictionSetDecision,
    PredictionSetUncertaintyKind,
    QualityFinding,
    QualityReport,
    SignalEnvelope,
    SignalSourceKind,
    ThresholdDirection,
    TrustDecision,
)

NOW = datetime(2026, 8, 24, 6, 0, tzinfo=UTC)
DIGEST_A = "sha256:" + "a" * 64
DIGEST_B = "sha256:" + "b" * 64


def _artifact(
    *,
    artifact_id: str = "artifact-1",
    digest: str = DIGEST_A,
    size_bytes: int = CANONICAL_SIGNAL_BYTES,
    media_type: str = CANONICAL_SIGNAL_MEDIA_TYPE,
    sensitive: bool = True,
) -> ArtifactReference:
    return ArtifactReference(
        schema_version=ARTIFACT_REFERENCE_SCHEMA_VERSION,
        artifact_id=artifact_id,
        file_sha256=digest,
        size_bytes=size_bytes,
        media_type=media_type,
        sensitive=sensitive,
    )


def _finding(
    *,
    severity: FindingSeverity = FindingSeverity.ERROR,
    code: str = "EXCESSIVE_NOISE",
) -> QualityFinding:
    return QualityFinding(
        schema_version=QUALITY_FINDING_SCHEMA_VERSION,
        code=code,
        severity=severity,
        message="The signal quality check found excessive noise.",
        affected_leads=("I", "II"),
        sample_start=100,
        sample_end=200,
    )


def _passing_quality() -> QualityReport:
    return QualityReport(
        schema_version=QUALITY_REPORT_SCHEMA_VERSION,
        evaluated_at=NOW,
        passed=True,
        decision=TrustDecision.PREDICTION_ALLOWED,
        findings=(),
    )


def _signal_payload() -> dict[str, object]:
    return {
        "schema_version": SIGNAL_ENVELOPE_SCHEMA_VERSION,
        "signal_id": "signal-001",
        "source_kind": SignalSourceKind.APPROVED_EXAMPLE,
        "payload": _artifact(),
        "lead_order": LEADS,
        "sampling_frequency_hz": 100.0,
        "samples_per_lead": 1000,
        "duration_seconds": 10.0,
        "units": "mV",
        "dtype": "float32",
        "shape": (12, 1000),
        "quality": _passing_quality(),
    }


def test_trust_decision_vocabulary_is_exact_and_contracts_are_frozen() -> None:
    assert {decision.value for decision in TrustDecision} == {
        "INVALID_INPUT",
        "REACQUIRE",
        "UNSUPPORTED_INPUT",
        "ABSTAIN",
        "PREDICTION_ALLOWED",
    }
    quality = _passing_quality()
    with pytest.raises(ValidationError, match="frozen"):
        quality.passed = False


def test_signal_envelope_is_metadata_only_and_enforces_canonical_input() -> None:
    signal = SignalEnvelope.model_validate(_signal_payload())
    assert signal.lead_order == LEADS
    assert "samples" not in signal.model_dump()

    with pytest.raises(ValidationError, match="extra_forbidden"):
        SignalEnvelope.model_validate({**_signal_payload(), "samples": [[0.0]]})
    with pytest.raises(ValidationError, match="canonical 12-lead"):
        SignalEnvelope.model_validate({**_signal_payload(), "lead_order": tuple(reversed(LEADS))})
    with pytest.raises(ValidationError, match="valid number"):
        SignalEnvelope.model_validate({**_signal_payload(), "sampling_frequency_hz": "100"})
    with pytest.raises(ValidationError, match="marked sensitive"):
        SignalEnvelope.model_validate({**_signal_payload(), "payload": _artifact(sensitive=False)})


def test_quality_contract_derives_pass_state_and_validates_localization() -> None:
    failed = QualityReport(
        schema_version=QUALITY_REPORT_SCHEMA_VERSION,
        evaluated_at=NOW,
        passed=False,
        decision=TrustDecision.REACQUIRE,
        findings=(_finding(),),
    )
    assert failed.decision is TrustDecision.REACQUIRE

    with pytest.raises(ValidationError, match="passed quality report"):
        QualityReport(
            schema_version=QUALITY_REPORT_SCHEMA_VERSION,
            evaluated_at=NOW,
            passed=True,
            decision=TrustDecision.PREDICTION_ALLOWED,
            findings=(_finding(),),
        )
    for non_quality_decision in (
        TrustDecision.ABSTAIN,
        TrustDecision.UNSUPPORTED_INPUT,
        TrustDecision.PREDICTION_ALLOWED,
    ):
        with pytest.raises(ValidationError, match="input-quality decision"):
            QualityReport(
                schema_version=QUALITY_REPORT_SCHEMA_VERSION,
                evaluated_at=NOW,
                passed=False,
                decision=non_quality_decision,
                findings=(_finding(),),
            )
    with pytest.raises(ValidationError, match="canonical lead order"):
        QualityFinding(
            schema_version=QUALITY_FINDING_SCHEMA_VERSION,
            code="LEAD_FAILURE",
            severity=FindingSeverity.ERROR,
            message="Lead order is invalid.",
            affected_leads=("II", "I"),
        )
    with pytest.raises(ValidationError, match="both be set"):
        QualityFinding(
            schema_version=QUALITY_FINDING_SCHEMA_VERSION,
            code="CLIPPED_SEGMENT",
            severity=FindingSeverity.WARNING,
            message="A segment may be clipped.",
            sample_start=10,
        )


def test_distribution_report_is_threshold_and_sample_size_derived() -> None:
    metric = DistributionMetric(
        schema_version=DISTRIBUTION_METRIC_SCHEMA_VERSION,
        feature_name="lead_I_amplitude",
        metric_name="population_stability_index",
        value=0.3,
        threshold=0.2,
        threshold_direction=ThresholdDirection.ABOVE,
        threshold_exceeded=True,
    )
    report = DistributionReport(
        schema_version=DISTRIBUTION_REPORT_SCHEMA_VERSION,
        generated_at=NOW,
        reference_cohort_id="ptbxl-reference",
        observed_cohort_id="site-b-window-1",
        reference_count=1000,
        observed_count=100,
        minimum_observed_count=50,
        status=DistributionStatus.SHIFT_DETECTED,
        metrics=(metric,),
    )
    assert report.status is DistributionStatus.SHIFT_DETECTED

    with pytest.raises(ValidationError, match="threshold_exceeded"):
        DistributionMetric(
            schema_version=DISTRIBUTION_METRIC_SCHEMA_VERSION,
            feature_name="lead_I_amplitude",
            metric_name="population_stability_index",
            value=0.1,
            threshold=0.2,
            threshold_direction=ThresholdDirection.ABOVE,
            threshold_exceeded=True,
        )
    with pytest.raises(ValidationError, match="INSUFFICIENT_DATA"):
        DistributionReport(
            schema_version=DISTRIBUTION_REPORT_SCHEMA_VERSION,
            generated_at=NOW,
            reference_cohort_id="ptbxl-reference",
            observed_cohort_id="site-b-window-2",
            reference_count=1000,
            observed_count=4,
            minimum_observed_count=50,
            status=DistributionStatus.WITHIN_REFERENCE,
            metrics=(),
        )
    with pytest.raises(ValidationError, match="at least one metric"):
        DistributionReport(
            schema_version=DISTRIBUTION_REPORT_SCHEMA_VERSION,
            generated_at=NOW,
            reference_cohort_id="ptbxl-reference",
            observed_cohort_id="site-b-window-3",
            reference_count=1000,
            observed_count=100,
            minimum_observed_count=50,
            status=DistributionStatus.WITHIN_REFERENCE,
            metrics=(),
        )


def _case_distribution_payload() -> dict[str, object]:
    return {
        "schema_version": CASE_DISTRIBUTION_ASSESSMENT_SCHEMA_VERSION,
        "assessment_id": "distribution-assessment-001",
        "signal_id": "signal-001",
        "release_id": "release-001",
        "method": "shrinkage_mahalanobis",
        "method_artifact": _artifact(
            artifact_id="mahalanobis-detector",
            size_bytes=500,
            media_type="application/json",
            sensitive=False,
        ),
        "expected_method_schema_version": 1,
        "observed_method_schema_version": 1,
        "artifact_available": True,
        "score_direction": "HIGHER_IS_MORE_OUT_OF_DISTRIBUTION",
        "score": 3.0,
        "threshold": 4.0,
        "is_out_of_distribution": False,
        "status": CaseDistributionStatus.WITHIN_REFERENCE,
        "reason_codes": ("SCORE_WITHIN_FROZEN_THRESHOLD",),
    }


def test_case_distribution_assessment_derives_higher_is_ood_result() -> None:
    within = CaseDistributionAssessment.model_validate(_case_distribution_payload())
    assert within.status is CaseDistributionStatus.WITHIN_REFERENCE
    assert not within.is_out_of_distribution

    outside = CaseDistributionAssessment.model_validate(
        {
            **_case_distribution_payload(),
            "score": 5.0,
            "is_out_of_distribution": True,
            "status": CaseDistributionStatus.OUTSIDE_REFERENCE,
            "reason_codes": ("SCORE_ABOVE_FROZEN_THRESHOLD",),
        }
    )
    assert outside.is_out_of_distribution

    with pytest.raises(ValidationError, match="derived as score > threshold"):
        CaseDistributionAssessment.model_validate(
            {**_case_distribution_payload(), "is_out_of_distribution": True}
        )
    with pytest.raises(ValidationError, match="finite number"):
        CaseDistributionAssessment.model_validate(
            {**_case_distribution_payload(), "score": float("nan")}
        )
    with pytest.raises(ValidationError, match="case_distribution_assessment.v1"):
        CaseDistributionAssessment.model_validate(
            {**_case_distribution_payload(), "schema_version": "v2"}
        )


def test_case_distribution_assessment_fails_closed_when_unavailable_or_incompatible() -> None:
    unavailable_payload = {
        **_case_distribution_payload(),
        "observed_method_schema_version": None,
        "artifact_available": False,
        "score": None,
        "threshold": None,
        "is_out_of_distribution": None,
        "status": CaseDistributionStatus.UNAVAILABLE,
        "reason_codes": ("ARTIFACT_UNAVAILABLE",),
    }
    unavailable = CaseDistributionAssessment.model_validate(unavailable_payload)
    assert unavailable.status is CaseDistributionStatus.UNAVAILABLE

    with pytest.raises(ValidationError, match="fail closed"):
        CaseDistributionAssessment.model_validate(
            {
                **unavailable_payload,
                "status": CaseDistributionStatus.WITHIN_REFERENCE,
            }
        )
    with pytest.raises(ValidationError, match="cannot expose score"):
        CaseDistributionAssessment.model_validate({**unavailable_payload, "score": 0.0})

    mismatch_payload = {
        **_case_distribution_payload(),
        "observed_method_schema_version": 2,
        "score": None,
        "threshold": None,
        "is_out_of_distribution": None,
        "status": CaseDistributionStatus.UNAVAILABLE,
        "reason_codes": ("ARTIFACT_VERSION_MISMATCH",),
    }
    mismatch = CaseDistributionAssessment.model_validate(mismatch_payload)
    assert mismatch.status is CaseDistributionStatus.UNAVAILABLE
    with pytest.raises(ValidationError, match="ARTIFACT_VERSION_MISMATCH"):
        CaseDistributionAssessment.model_validate(
            {**mismatch_payload, "reason_codes": ("SCORING_UNAVAILABLE",)}
        )


def test_per_label_uncertainty_decision_is_fully_derived() -> None:
    decision = LabelUncertaintyDecision(
        schema_version=LABEL_UNCERTAINTY_SCHEMA_VERSION,
        label="MI",
        calibrated_probability=0.8,
        classification_threshold=0.6,
        predicted_positive=True,
        uncertainty_score=0.2,
        uncertainty_threshold=0.3,
        decision=TrustDecision.PREDICTION_ALLOWED,
        reason_codes=("UNCERTAINTY_WITHIN_GATE",),
    )
    assert decision.predicted_positive

    payload = decision.model_dump(mode="python")
    with pytest.raises(ValidationError, match="predicted_positive"):
        LabelUncertaintyDecision.model_validate({**payload, "predicted_positive": False})
    with pytest.raises(ValidationError, match="uncertainty gate"):
        LabelUncertaintyDecision.model_validate({**payload, "decision": TrustDecision.ABSTAIN})
    with pytest.raises(ValidationError, match="ABSTAIN or PREDICTION_ALLOWED"):
        LabelUncertaintyDecision.model_validate(
            {**payload, "decision": TrustDecision.INVALID_INPUT}
        )


def _prediction_set_payload() -> dict[str, object]:
    return {
        "schema_version": LABEL_PREDICTION_SET_DECISION_SCHEMA_VERSION,
        "label": "MI",
        "calibrated_probability": 0.8,
        "include_supported": True,
        "include_not_supported": False,
        "decision": PredictionSetDecision.SUPPORTED,
        "uncertainty_kind": PredictionSetUncertaintyKind.NONE,
        "calibration_artifact": _artifact(
            artifact_id="conformal-calibration",
            size_bytes=500,
            media_type="application/json",
            sensitive=False,
        ),
        "calibration_artifact_type": "ecg_trust.labelwise_binary_conformal",
        "calibration_artifact_schema_version": 1,
        "coverage_scope": CONFORMAL_COVERAGE_SCOPE,
        "coverage_scope_text": CONFORMAL_COVERAGE_SCOPE_TEXT,
    }


@pytest.mark.parametrize(
    ("include_not_supported", "include_supported", "decision", "uncertainty_kind"),
    [
        (False, True, PredictionSetDecision.SUPPORTED, PredictionSetUncertaintyKind.NONE),
        (
            True,
            False,
            PredictionSetDecision.NOT_SUPPORTED,
            PredictionSetUncertaintyKind.NONE,
        ),
        (True, True, PredictionSetDecision.UNCERTAIN, PredictionSetUncertaintyKind.BOTH),
        (False, False, PredictionSetDecision.UNCERTAIN, PredictionSetUncertaintyKind.EMPTY),
    ],
)
def test_label_prediction_set_decision_derives_all_four_membership_states(
    include_not_supported: bool,
    include_supported: bool,
    decision: PredictionSetDecision,
    uncertainty_kind: PredictionSetUncertaintyKind,
) -> None:
    result = LabelPredictionSetDecision.model_validate(
        {
            **_prediction_set_payload(),
            "include_not_supported": include_not_supported,
            "include_supported": include_supported,
            "decision": decision,
            "uncertainty_kind": uncertainty_kind,
        }
    )
    assert result.decision is decision
    assert result.uncertainty_kind is uncertainty_kind


def test_label_prediction_set_rejects_contradictions_and_individual_certainty_copy() -> None:
    with pytest.raises(ValidationError, match="decision contradicts"):
        LabelPredictionSetDecision.model_validate(
            {
                **_prediction_set_payload(),
                "decision": PredictionSetDecision.UNCERTAIN,
            }
        )
    with pytest.raises(ValidationError, match="uncertainty_kind contradicts"):
        LabelPredictionSetDecision.model_validate(
            {
                **_prediction_set_payload(),
                "uncertainty_kind": PredictionSetUncertaintyKind.BOTH,
            }
        )
    with pytest.raises(ValidationError, match="individual certainty guarantee"):
        LabelPredictionSetDecision.model_validate(
            {
                **_prediction_set_payload(),
                "coverage_scope_text": "This individual prediction is 90% certain.",
            }
        )
    with pytest.raises(ValidationError, match="valid boolean"):
        LabelPredictionSetDecision.model_validate(
            {**_prediction_set_payload(), "include_supported": "true"}
        )
    with pytest.raises(ValidationError, match="Input should be 1"):
        LabelPredictionSetDecision.model_validate(
            {**_prediction_set_payload(), "calibration_artifact_schema_version": 2}
        )


def test_explanation_and_audit_contracts_exclude_unstructured_sensitive_data() -> None:
    explanation = ExplanationEnvelope(
        schema_version=EXPLANATION_ENVELOPE_SCHEMA_VERSION,
        explanation_id="explanation-001",
        release_id="release-001",
        signal_id="signal-001",
        method=ExplanationMethod.GRAD_CAM_1D,
        target_label="MI",
        attribution=_artifact(
            artifact_id="attribution-001",
            size_bytes=4000,
            media_type="application/octet-stream",
        ),
        shape=(1, 1000),
        coordinate_space="FROZEN_NORMALIZED_MODEL_INPUT",
        signed=True,
        normalized_to="UNIT_MAX_MAGNITUDE",
        controls_passed=True,
        control_findings=(),
    )
    assert explanation.shape == (1, 1000)
    with pytest.raises(ValidationError, match="explanation shape"):
        ExplanationEnvelope.model_validate(
            {**explanation.model_dump(mode="python"), "shape": (2, 1000)}
        )

    event_payload: dict[str, object] = {
        "schema_version": AUDIT_EVENT_SCHEMA_VERSION,
        "event_id": "event-001",
        "occurred_at": NOW,
        "request_id": "request-001",
        "actor_type": AuditActorType.SERVICE,
        "actor_id": "sentinel-api",
        "action": AuditAction.INFERENCE,
        "resource_type": "TRUST_BUNDLE",
        "resource_id": "signal-token-001",
        "release_id": "release-001",
        "outcome": AuditOutcome.FAILED,
        "reason_code": "QUALITY_GATE_FAILED",
    }
    event = AuditEvent.model_validate(event_payload)
    assert event.outcome is AuditOutcome.FAILED
    with pytest.raises(ValidationError, match="extra_forbidden"):
        AuditEvent.model_validate({**event_payload, "metadata": {"patient_id": "x"}})
    with pytest.raises(ValidationError, match="reason_code"):
        AuditEvent.model_validate({**event_payload, "reason_code": None})
    with pytest.raises(ValidationError, match="timezone"):
        AuditEvent.model_validate({**event_payload, "occurred_at": datetime(2026, 8, 24, 6, 0)})


def test_evidence_bundle_enforces_aggregate_publication_boundary() -> None:
    interval = ConfidenceInterval(
        schema_version=CONFIDENCE_INTERVAL_SCHEMA_VERSION,
        lower=0.88,
        upper=0.94,
        confidence_level=0.95,
        method="patient_bootstrap",
    )
    metric = EvidenceMetric(
        schema_version=EVIDENCE_METRIC_SCHEMA_VERSION,
        metric_id="macro_auc",
        display_name="Macro AUC",
        definition="Unweighted mean of the five superclass ROC AUC values.",
        direction=MetricDirection.HIGHER_IS_BETTER,
        value=0.92,
        unit="area_under_curve",
        sample_count=1000,
        confidence_interval=interval,
        per_seed_values=(0.91, 0.92, 0.93),
    )
    cohort = EvidenceCohort(
        schema_version=EVIDENCE_COHORT_SCHEMA_VERSION,
        cohort_id="ptbxl-fold10",
        dataset_name="PTB-XL",
        dataset_version="1.0.3",
        role="SEALED_TEST",
        record_count=1000,
        patient_count=900,
        definition="The protocol-defined sealed final evaluation cohort.",
    )
    payload: dict[str, object] = {
        "schema_version": EVIDENCE_BUNDLE_SCHEMA_VERSION,
        "evidence_id": "evidence-001",
        "release_id": "release-001",
        "created_at": NOW,
        "title": "Sealed evaluation evidence",
        "cohort": cohort,
        "metrics": (metric,),
        "artifacts": (
            _artifact(
                artifact_id="aggregate-table",
                size_bytes=100,
                media_type="application/json",
                sensitive=False,
            ),
        ),
        "source_artifact_sha256": (DIGEST_A, DIGEST_B),
        "limitations": ("Research use only; no clinical validation was performed.",),
        "aggregate_only": True,
        "contains_direct_identifiers": False,
        "research_only": True,
        "publishable": True,
    }
    bundle = EvidenceBundle.model_validate(payload)
    assert bundle.publishable

    with pytest.raises(ValidationError, match="sensitive artifacts"):
        EvidenceBundle.model_validate(
            {**payload, "artifacts": (_artifact(artifact_id="raw-output"),)}
        )
    with pytest.raises(ValidationError, match="unique"):
        EvidenceBundle.model_validate({**payload, "metrics": (metric, metric)})
