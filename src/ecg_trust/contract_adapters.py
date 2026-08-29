"""Adapters from scientific core results to strict Sentinel boundary contracts."""

from __future__ import annotations

from datetime import datetime
from typing import Final, Literal, cast

import numpy as np
from numpy.typing import ArrayLike

from ecg_trust.conformal import BinaryDecision, BinaryPredictionSets, UncertaintyKind
from ecg_trust.constants import SUPERCLASSES
from ecg_trust.contracts import (
    CASE_DISTRIBUTION_ASSESSMENT_SCHEMA_VERSION,
    LABEL_PREDICTION_SET_DECISION_SCHEMA_VERSION,
    QUALITY_FINDING_SCHEMA_VERSION,
    QUALITY_REPORT_SCHEMA_VERSION,
    ArtifactReference,
    CaseDistributionAssessment,
    CaseDistributionStatus,
    FindingSeverity,
    LabelPredictionSetDecision,
    PredictionSetDecision,
    PredictionSetUncertaintyKind,
    QualityFinding,
    QualityReport,
    TrustDecision,
)
from ecg_trust.contracts.models import LeadName, SuperclassLabel
from ecg_trust.quality.signal_quality import (
    QualityIssue,
    QualityStatus,
    ReasonCode,
    SignalQualityReport,
)

_QUALITY_MESSAGES: dict[ReasonCode, str] = {
    ReasonCode.NON_NUMERIC_SIGNAL: "The waveform is not numeric.",
    ReasonCode.NON_REAL_SIGNAL: "The waveform must contain real-valued samples.",
    ReasonCode.WRONG_SIGNAL_SHAPE: "The waveform does not have the required 12 by 1000 shape.",
    ReasonCode.NONFINITE_SIGNAL: "The waveform contains a non-finite sample.",
    ReasonCode.INVALID_METADATA: "Required ECG metadata is invalid.",
    ReasonCode.LEAD_COUNT_MISMATCH: "The recording does not contain exactly 12 leads.",
    ReasonCode.MISSING_LEADS: "One or more canonical ECG leads are missing.",
    ReasonCode.DUPLICATE_LEADS: "The recording contains duplicate lead identities.",
    ReasonCode.UNEXPECTED_LEADS: "The recording contains an unsupported lead identity.",
    ReasonCode.LEAD_ORDER_MISMATCH: "ECG leads are not in the required canonical order.",
    ReasonCode.SAMPLE_RATE_MISMATCH: "The recording is not sampled at the supported rate.",
    ReasonCode.DURATION_MISMATCH: "The recording does not have the supported duration.",
    ReasonCode.UNIT_COUNT_MISMATCH: "Physical units are missing for one or more leads.",
    ReasonCode.UNSUPPORTED_UNITS: "The recording is not expressed in physical millivolts.",
    ReasonCode.FLATLINE: "A lead is nearly flat or may be disconnected.",
    ReasonCode.CLIPPING_OR_SATURATION: "A lead contains probable clipping or saturation.",
    ReasonCode.EXTREME_AMPLITUDE: "A lead contains an extreme physical amplitude.",
    ReasonCode.EXTREME_SPIKES: "A lead contains an extreme sample-to-sample spike.",
    ReasonCode.BASELINE_WANDER: "A lead contains excessive low-frequency baseline movement.",
    ReasonCode.POWERLINE_INTERFERENCE_50HZ: "A lead contains probable 50 Hz interference.",
    ReasonCode.POWERLINE_INTERFERENCE_60HZ: "A lead contains probable 60 Hz interference.",
    ReasonCode.HIGH_FREQUENCY_NOISE: "A lead contains excessive high-frequency noise.",
    ReasonCode.LIMB_LEAD_INCONSISTENCY: "Limb-lead relationships are inconsistent.",
    ReasonCode.PROBABLE_LIMB_LEAD_REVERSAL: (
        "The recording contains evidence of a probable limb-electrode reversal."
    ),
}

_COVERAGE_SCOPE_TEXT: Final[
    Literal[
        "Coverage is label-wise marginal under exchangeability; "
        "it is not an individual certainty guarantee."
    ]
] = (
    "Coverage is label-wise marginal under exchangeability; "
    "it is not an individual certainty guarantee."
)


def quality_report_to_contract(
    report: SignalQualityReport,
    *,
    evaluated_at: datetime,
) -> QualityReport:
    """Convert a deterministic core report without weakening its disposition."""

    issues = list(report.global_issues)
    for lead in report.leads:
        issues.extend(lead.issues)
    findings = tuple(_quality_finding(issue) for issue in _deduplicate_issues(issues))

    if report.status is QualityStatus.PASS:
        decision = TrustDecision.PREDICTION_ALLOWED
        passed = True
    elif report.status is QualityStatus.INVALID:
        decision = TrustDecision.INVALID_INPUT
        passed = False
    else:
        decision = TrustDecision.REACQUIRE
        passed = False

    return QualityReport(
        schema_version=QUALITY_REPORT_SCHEMA_VERSION,
        evaluated_at=evaluated_at,
        passed=passed,
        decision=decision,
        findings=findings,
    )


def case_distribution_assessment_from_score(
    *,
    assessment_id: str,
    signal_id: str,
    release_id: str,
    method: str,
    method_artifact: ArtifactReference,
    method_schema_version: int,
    score: float,
    threshold: float,
) -> CaseDistributionAssessment:
    """Create a case OOD contract from one frozen higher-is-OOD score."""

    is_ood = score > threshold
    return CaseDistributionAssessment(
        schema_version=CASE_DISTRIBUTION_ASSESSMENT_SCHEMA_VERSION,
        assessment_id=assessment_id,
        signal_id=signal_id,
        release_id=release_id,
        method=method,
        method_artifact=method_artifact,
        expected_method_schema_version=method_schema_version,
        observed_method_schema_version=method_schema_version,
        artifact_available=True,
        score_direction="HIGHER_IS_MORE_OUT_OF_DISTRIBUTION",
        score=score,
        threshold=threshold,
        is_out_of_distribution=is_ood,
        status=(
            CaseDistributionStatus.OUTSIDE_REFERENCE
            if is_ood
            else CaseDistributionStatus.WITHIN_REFERENCE
        ),
        reason_codes=(
            "SCORE_ABOVE_FROZEN_THRESHOLD" if is_ood else "SCORE_WITHIN_FROZEN_THRESHOLD",
        ),
    )


def unavailable_case_distribution_assessment(
    *,
    assessment_id: str,
    signal_id: str,
    release_id: str,
    method: str,
    method_artifact: ArtifactReference,
    expected_method_schema_version: int,
    observed_method_schema_version: int | None,
    artifact_available: bool,
    reason_code: str,
) -> CaseDistributionAssessment:
    """Represent an unavailable or incompatible OOD component without fallback."""

    return CaseDistributionAssessment(
        schema_version=CASE_DISTRIBUTION_ASSESSMENT_SCHEMA_VERSION,
        assessment_id=assessment_id,
        signal_id=signal_id,
        release_id=release_id,
        method=method,
        method_artifact=method_artifact,
        expected_method_schema_version=expected_method_schema_version,
        observed_method_schema_version=observed_method_schema_version,
        artifact_available=artifact_available,
        score_direction="HIGHER_IS_MORE_OUT_OF_DISTRIBUTION",
        score=None,
        threshold=None,
        is_out_of_distribution=None,
        status=CaseDistributionStatus.UNAVAILABLE,
        reason_codes=(reason_code,),
    )


def conformal_prediction_sets_to_contracts(
    prediction_sets: BinaryPredictionSets,
    calibrated_probabilities: ArrayLike,
    *,
    calibration_artifact: ArtifactReference,
) -> tuple[LabelPredictionSetDecision, ...]:
    """Convert exactly one canonical five-label conformal prediction."""

    if prediction_sets.n_samples != 1:
        raise ValueError("case contract conversion requires exactly one prediction row")
    if prediction_sets.label_names != SUPERCLASSES:
        raise ValueError("conformal label order must match the canonical superclasses")
    probabilities = np.asarray(calibrated_probabilities, dtype=np.float64)
    if probabilities.shape != (len(SUPERCLASSES),):
        raise ValueError("calibrated_probabilities must contain five values")
    if not np.isfinite(probabilities).all() or np.any(
        (probabilities < 0.0) | (probabilities > 1.0)
    ):
        raise ValueError("calibrated_probabilities must be finite values in [0, 1]")

    negative = prediction_sets.include_not_supported[0]
    positive = prediction_sets.include_supported[0]
    decisions = prediction_sets.decisions[0]
    uncertainty = prediction_sets.uncertainty_kinds[0]
    result: list[LabelPredictionSetDecision] = []
    for index, label in enumerate(SUPERCLASSES):
        result.append(
            LabelPredictionSetDecision(
                schema_version=LABEL_PREDICTION_SET_DECISION_SCHEMA_VERSION,
                label=cast(SuperclassLabel, label),
                calibrated_probability=float(probabilities[index]),
                include_supported=positive[index],
                include_not_supported=negative[index],
                decision=_prediction_set_decision(decisions[index]),
                uncertainty_kind=_prediction_set_uncertainty(uncertainty[index]),
                calibration_artifact=calibration_artifact,
                calibration_artifact_type="ecg_trust.labelwise_binary_conformal",
                calibration_artifact_schema_version=1,
                coverage_scope="labelwise_marginal_under_exchangeability",
                coverage_scope_text=_COVERAGE_SCOPE_TEXT,
            )
        )
    return tuple(result)


def _quality_finding(issue: QualityIssue) -> QualityFinding:
    affected = () if issue.lead_name is None else (cast(LeadName, issue.lead_name),)
    severity = (
        FindingSeverity.WARNING if issue.status is QualityStatus.LIMITED else FindingSeverity.ERROR
    )
    return QualityFinding(
        schema_version=QUALITY_FINDING_SCHEMA_VERSION,
        code=issue.code.value.upper(),
        severity=severity,
        message=_QUALITY_MESSAGES[issue.code],
        affected_leads=affected,
    )


def _deduplicate_issues(issues: list[QualityIssue]) -> tuple[QualityIssue, ...]:
    result: list[QualityIssue] = []
    observed: set[tuple[ReasonCode, str | None, QualityStatus]] = set()
    for issue in issues:
        identity = (issue.code, issue.lead_name, issue.status)
        if identity not in observed:
            observed.add(identity)
            result.append(issue)
    return tuple(result)


def _prediction_set_decision(value: BinaryDecision) -> PredictionSetDecision:
    return {
        BinaryDecision.SUPPORTED: PredictionSetDecision.SUPPORTED,
        BinaryDecision.NOT_SUPPORTED: PredictionSetDecision.NOT_SUPPORTED,
        BinaryDecision.UNCERTAIN: PredictionSetDecision.UNCERTAIN,
    }[value]


def _prediction_set_uncertainty(
    value: UncertaintyKind | None,
) -> PredictionSetUncertaintyKind:
    if value is UncertaintyKind.BOTH:
        return PredictionSetUncertaintyKind.BOTH
    if value is UncertaintyKind.EMPTY:
        return PredictionSetUncertaintyKind.EMPTY
    return PredictionSetUncertaintyKind.NONE


__all__ = [
    "case_distribution_assessment_from_score",
    "conformal_prediction_sets_to_contracts",
    "quality_report_to_contract",
    "unavailable_case_distribution_assessment",
]
