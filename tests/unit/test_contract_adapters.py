from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from ecg_trust.conformal import BinaryPredictionSets
from ecg_trust.contract_adapters import (
    case_distribution_assessment_from_score,
    conformal_prediction_sets_to_contracts,
    quality_report_to_contract,
    unavailable_case_distribution_assessment,
)
from ecg_trust.contracts import (
    ARTIFACT_REFERENCE_SCHEMA_VERSION,
    ArtifactReference,
    CaseDistributionStatus,
    FindingSeverity,
    PredictionSetDecision,
    PredictionSetUncertaintyKind,
    TrustDecision,
)
from ecg_trust.quality.signal_quality import (
    QualityIssue,
    QualityStatus,
    ReasonCode,
    SignalQualityReport,
)


def _report(status: QualityStatus) -> SignalQualityReport:
    issues = ()
    if status is not QualityStatus.PASS:
        issues = (
            QualityIssue(
                code=ReasonCode.FLATLINE,
                status=status,
                lead_name="V1",
                metric_name="peak_to_peak_mv",
                observed_value=0.0,
                boundary_value=0.03,
            ),
        )
    return SignalQualityReport(
        status=status,
        config_version="test-v1",
        global_issues=issues,
        leads=(),
        reversal_evidence=None,
    )


def _artifact(artifact_id: str = "policy") -> ArtifactReference:
    return ArtifactReference(
        schema_version=ARTIFACT_REFERENCE_SCHEMA_VERSION,
        artifact_id=artifact_id,
        file_sha256="sha256:" + "a" * 64,
        size_bytes=10,
        media_type="application/json",
        sensitive=False,
    )


@pytest.mark.parametrize(
    ("status", "decision", "passed"),
    [
        (QualityStatus.PASS, TrustDecision.PREDICTION_ALLOWED, True),
        (QualityStatus.LIMITED, TrustDecision.REACQUIRE, False),
        (QualityStatus.REACQUIRE, TrustDecision.REACQUIRE, False),
        (QualityStatus.INVALID, TrustDecision.INVALID_INPUT, False),
    ],
)
def test_quality_adapter_never_weakens_the_core_disposition(
    status: QualityStatus,
    decision: TrustDecision,
    passed: bool,
) -> None:
    result = quality_report_to_contract(
        _report(status),
        evaluated_at=datetime(2026, 8, 24, tzinfo=UTC),
    )

    assert result.decision is decision
    assert result.passed is passed
    if passed:
        assert result.findings == ()
    else:
        assert result.findings[0].code == "FLATLINE"
        assert result.findings[0].affected_leads == ("V1",)
        expected_severity = (
            FindingSeverity.WARNING if status is QualityStatus.LIMITED else FindingSeverity.ERROR
        )
        assert result.findings[0].severity is expected_severity


def test_limited_quality_maps_to_warning_but_still_requires_reacquisition() -> None:
    result = quality_report_to_contract(
        _report(QualityStatus.LIMITED),
        evaluated_at=datetime(2026, 8, 24, tzinfo=UTC),
    )

    assert result.decision is TrustDecision.REACQUIRE
    assert result.findings[0].severity is FindingSeverity.WARNING
    assert "nearly flat" in result.findings[0].message


def test_boundary_contract_requires_an_aware_evaluation_timestamp() -> None:
    with pytest.raises(ValueError, match="timezone"):
        quality_report_to_contract(
            _report(QualityStatus.PASS),
            evaluated_at=datetime(2026, 8, 24),
        )


@pytest.mark.parametrize(
    ("score", "expected_status", "expected_reason"),
    [
        (1.0, CaseDistributionStatus.WITHIN_REFERENCE, "SCORE_WITHIN_FROZEN_THRESHOLD"),
        (1.0001, CaseDistributionStatus.OUTSIDE_REFERENCE, "SCORE_ABOVE_FROZEN_THRESHOLD"),
    ],
)
def test_case_distribution_adapter_derives_status_from_frozen_threshold(
    score: float,
    expected_status: CaseDistributionStatus,
    expected_reason: str,
) -> None:
    result = case_distribution_assessment_from_score(
        assessment_id="assessment-1",
        signal_id="signal-1",
        release_id="release-1",
        method="mahalanobis-v1",
        method_artifact=_artifact(),
        method_schema_version=1,
        score=score,
        threshold=1.0,
    )

    assert result.status is expected_status
    assert result.reason_codes == (expected_reason,)


def test_unavailable_distribution_adapter_preserves_fail_closed_reason() -> None:
    result = unavailable_case_distribution_assessment(
        assessment_id="assessment-1",
        signal_id="signal-1",
        release_id="release-1",
        method="mahalanobis-v1",
        method_artifact=_artifact(),
        expected_method_schema_version=1,
        observed_method_schema_version=None,
        artifact_available=False,
        reason_code="ARTIFACT_UNAVAILABLE",
    )

    assert result.status is CaseDistributionStatus.UNAVAILABLE
    assert result.score is None
    assert result.is_out_of_distribution is None


def test_conformal_adapter_preserves_both_and_empty_uncertainty() -> None:
    sets = BinaryPredictionSets.from_masks(
        label_names=("NORM", "MI", "STTC", "CD", "HYP"),
        include_not_supported=np.asarray([[False, True, True, False, True]]),
        include_supported=np.asarray([[True, False, True, False, False]]),
    )

    result = conformal_prediction_sets_to_contracts(
        sets,
        np.asarray([0.9, 0.1, 0.5, 0.5, 0.2]),
        calibration_artifact=_artifact("conformal-v1"),
    )

    assert tuple(item.decision for item in result) == (
        PredictionSetDecision.SUPPORTED,
        PredictionSetDecision.NOT_SUPPORTED,
        PredictionSetDecision.UNCERTAIN,
        PredictionSetDecision.UNCERTAIN,
        PredictionSetDecision.NOT_SUPPORTED,
    )
    assert result[2].uncertainty_kind is PredictionSetUncertaintyKind.BOTH
    assert result[3].uncertainty_kind is PredictionSetUncertaintyKind.EMPTY
    assert "not an individual certainty guarantee" in result[0].coverage_scope_text


def test_conformal_adapter_rejects_noncanonical_case_shapes_and_labels() -> None:
    two_rows = BinaryPredictionSets.from_masks(
        label_names=("NORM", "MI", "STTC", "CD", "HYP"),
        include_not_supported=np.ones((2, 5), dtype=np.bool_),
        include_supported=np.zeros((2, 5), dtype=np.bool_),
    )
    with pytest.raises(ValueError, match="exactly one"):
        conformal_prediction_sets_to_contracts(
            two_rows,
            np.zeros(5),
            calibration_artifact=_artifact(),
        )

    wrong_labels = BinaryPredictionSets.from_masks(
        label_names=("A", "B", "C", "D", "E"),
        include_not_supported=np.ones((1, 5), dtype=np.bool_),
        include_supported=np.zeros((1, 5), dtype=np.bool_),
    )
    with pytest.raises(ValueError, match="canonical"):
        conformal_prediction_sets_to_contracts(
            wrong_labels,
            np.zeros(5),
            calibration_artifact=_artifact(),
        )
