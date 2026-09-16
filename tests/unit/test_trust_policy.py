from __future__ import annotations

from dataclasses import replace
from typing import cast

import pytest

from ecg_trust.conformal import BinaryDecision
from ecg_trust.constants import SUPERCLASSES
from ecg_trust.contracts import TrustDecision
from ecg_trust.quality.signal_quality import (
    QualityIssue,
    QualityStatus,
    ReasonCode,
    SignalQualityReport,
)
from ecg_trust.trust_policy import (
    TrustPolicyConfig,
    TrustPolicyInputs,
    TrustPolicyValidationError,
    TrustReasonCode,
    evaluate_trust_policy,
)


def _quality(status: QualityStatus = QualityStatus.PASS) -> SignalQualityReport:
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
        config_version="test-quality-v1",
        global_issues=issues,
        leads=(),
        reversal_evidence=None,
    )


def _valid_inputs() -> TrustPolicyInputs:
    return TrustPolicyInputs(
        release_integrity_verified=True,
        input_contract_valid=True,
        quality_report=_quality(),
        distribution_supported=True,
        legacy_entropy_gate_accepted=True,
        conformal_decisions=(BinaryDecision.NOT_SUPPORTED,) * len(SUPERCLASSES),
    )


def test_all_gates_must_pass_before_predictions_are_exposed() -> None:
    result = evaluate_trust_policy(_valid_inputs())

    assert result.decision is TrustDecision.PREDICTION_ALLOWED
    assert result.predictions_exposed
    assert result.reason_codes == (TrustReasonCode.ALL_TRUST_GATES_PASSED,)
    assert result.to_dict()["predictions_exposed"] is True


@pytest.mark.parametrize(
    ("changes", "decision", "reason"),
    [
        (
            {"release_integrity_verified": False},
            TrustDecision.INVALID_INPUT,
            TrustReasonCode.RELEASE_INTEGRITY_UNVERIFIED,
        ),
        (
            {"input_contract_valid": False},
            TrustDecision.INVALID_INPUT,
            TrustReasonCode.INPUT_CONTRACT_INVALID,
        ),
        (
            {"quality_report": None},
            TrustDecision.INVALID_INPUT,
            TrustReasonCode.QUALITY_COMPONENT_UNAVAILABLE,
        ),
        (
            {"quality_report": _quality(QualityStatus.INVALID)},
            TrustDecision.INVALID_INPUT,
            TrustReasonCode.SIGNAL_QUALITY_INVALID,
        ),
        (
            {"quality_report": _quality(QualityStatus.LIMITED)},
            TrustDecision.REACQUIRE,
            TrustReasonCode.SIGNAL_REACQUISITION_REQUIRED,
        ),
        (
            {"quality_report": _quality(QualityStatus.REACQUIRE)},
            TrustDecision.REACQUIRE,
            TrustReasonCode.SIGNAL_REACQUISITION_REQUIRED,
        ),
        (
            {"distribution_supported": None},
            TrustDecision.UNSUPPORTED_INPUT,
            TrustReasonCode.DISTRIBUTION_COMPONENT_UNAVAILABLE,
        ),
        (
            {"distribution_supported": False},
            TrustDecision.UNSUPPORTED_INPUT,
            TrustReasonCode.OUTSIDE_VALIDATED_DISTRIBUTION,
        ),
        (
            {"legacy_entropy_gate_accepted": None},
            TrustDecision.ABSTAIN,
            TrustReasonCode.UNCERTAINTY_COMPONENT_UNAVAILABLE,
        ),
        (
            {"legacy_entropy_gate_accepted": False},
            TrustDecision.ABSTAIN,
            TrustReasonCode.LEGACY_ENTROPY_GATE_REJECTED,
        ),
        (
            {"conformal_decisions": None},
            TrustDecision.ABSTAIN,
            TrustReasonCode.UNCERTAINTY_COMPONENT_UNAVAILABLE,
        ),
    ],
)
def test_each_failed_component_closes_the_gate(
    changes: dict[str, object],
    decision: TrustDecision,
    reason: TrustReasonCode,
) -> None:
    result = evaluate_trust_policy(replace(_valid_inputs(), **changes))

    assert result.decision is decision
    assert not result.predictions_exposed
    assert result.reason_codes == (reason,)


def test_conformal_uncertainty_names_each_uncertain_label() -> None:
    decisions = list((BinaryDecision.NOT_SUPPORTED,) * len(SUPERCLASSES))
    decisions[1] = BinaryDecision.UNCERTAIN
    decisions[4] = BinaryDecision.UNCERTAIN

    result = evaluate_trust_policy(replace(_valid_inputs(), conformal_decisions=tuple(decisions)))

    assert result.decision is TrustDecision.ABSTAIN
    assert result.uncertain_labels == ("MI", "HYP")
    assert result.reason_codes == (TrustReasonCode.CONFORMAL_SET_UNCERTAIN,)


def test_quality_failure_has_priority_over_ood_and_uncertainty() -> None:
    result = evaluate_trust_policy(
        replace(
            _valid_inputs(),
            quality_report=_quality(QualityStatus.REACQUIRE),
            distribution_supported=False,
            legacy_entropy_gate_accepted=False,
            conformal_decisions=None,
        )
    )

    assert result.decision is TrustDecision.REACQUIRE
    assert result.quality_reason_codes == ("FLATLINE",)
    assert result.distribution_reason_codes == ()


def test_distribution_failure_has_priority_over_uncertainty() -> None:
    result = evaluate_trust_policy(
        replace(
            _valid_inputs(),
            distribution_supported=False,
            distribution_reason_codes=("MAHALANOBIS_THRESHOLD_EXCEEDED",),
            legacy_entropy_gate_accepted=False,
            conformal_decisions=None,
        )
    )

    assert result.decision is TrustDecision.UNSUPPORTED_INPUT
    assert result.distribution_reason_codes == ("MAHALANOBIS_THRESHOLD_EXCEEDED",)


def test_input_validation_rejects_wrong_label_count_and_duplicate_reasons() -> None:
    with pytest.raises(TrustPolicyValidationError, match="must contain 5 labels"):
        replace(_valid_inputs(), conformal_decisions=(BinaryDecision.SUPPORTED,))
    with pytest.raises(TrustPolicyValidationError, match="must be unique"):
        replace(_valid_inputs(), distribution_reason_codes=("OOD", "OOD"))


def test_v1_configuration_cannot_allow_partial_conformal_results() -> None:
    with pytest.raises(TrustPolicyValidationError, match="must fail closed"):
        TrustPolicyConfig(require_all_label_sets_singleton=False)


@pytest.mark.parametrize("decision", ["uncertain", "supported", "unknown", None, True])
def test_policy_rejects_untyped_conformal_decisions(decision: object) -> None:
    with pytest.raises(TrustPolicyValidationError, match="BinaryDecision"):
        replace(
            _valid_inputs(),
            conformal_decisions=(cast(BinaryDecision, decision),) * len(SUPERCLASSES),
        )


@pytest.mark.parametrize(
    "field",
    [
        "release_integrity_verified",
        "input_contract_valid",
        "distribution_supported",
        "legacy_entropy_gate_accepted",
    ],
)
@pytest.mark.parametrize("value", ["false", 1, 0])
def test_policy_rejects_non_boolean_gate_evidence(field: str, value: object) -> None:
    with pytest.raises(TrustPolicyValidationError, match="boolean"):
        replace(_valid_inputs(), **{field: value})


@pytest.mark.parametrize("status", ["pass", "invalid", "unknown", None])
def test_policy_rejects_untyped_quality_status(status: object) -> None:
    with pytest.raises(TrustPolicyValidationError, match="QualityStatus"):
        replace(
            _valid_inputs(),
            quality_report=replace(_quality(), status=cast(QualityStatus, status)),
        )


@pytest.mark.parametrize(
    "field", ["require_legacy_entropy_gate", "require_all_label_sets_singleton"]
)
@pytest.mark.parametrize("value", ["false", 0, 1, None])
def test_policy_rejects_non_boolean_gate_settings(field: str, value: object) -> None:
    with pytest.raises(TrustPolicyValidationError, match="boolean"):
        replace(TrustPolicyConfig(), **{field: value})


def test_policy_rejects_untyped_quality_report() -> None:
    with pytest.raises(TrustPolicyValidationError, match="SignalQualityReport"):
        replace(_valid_inputs(), quality_report=cast(SignalQualityReport, {"status": "pass"}))
