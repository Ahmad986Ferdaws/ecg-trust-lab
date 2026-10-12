from __future__ import annotations

from dataclasses import replace

import pytest

from ecg_trust.conformal import BinaryDecision
from ecg_trust.contracts import TrustDecision
from ecg_trust.trust_policy import (
    TrustPolicyResult,
    TrustPolicyValidationError,
    TrustReasonCode,
    evaluate_trust_policy,
)
from tests.unit.test_trust_policy import _valid_inputs


def test_mutating_validated_decisions_cannot_turn_abstention_into_permission() -> None:
    decisions = [BinaryDecision.UNCERTAIN] * 5
    evidence = replace(_valid_inputs(), conformal_decisions=decisions)
    decisions[:] = [BinaryDecision.SUPPORTED] * 5
    assert evaluate_trust_policy(evidence).decision is TrustDecision.ABSTAIN


def test_mutating_reason_list_cannot_publish_an_internal_reason() -> None:
    reasons = [TrustReasonCode.CONFORMAL_SET_UNCERTAIN]
    result = TrustPolicyResult("trust-policy-v1", TrustDecision.ABSTAIN, reasons)
    reasons[0] = TrustReasonCode.LABEL_SET_INCOHERENT
    assert result.public_reason_codes == ("CONFORMAL_SET_UNCERTAIN",)


def test_auxiliary_audit_fields_are_snapshotted() -> None:
    quality = ["FLATLINE"]
    distribution = ["DISTANCE"]
    labels = ["MI"]
    result = TrustPolicyResult(
        "trust-policy-v1",
        TrustDecision.ABSTAIN,
        (TrustReasonCode.CONFORMAL_SET_UNCERTAIN,),
        quality_reason_codes=quality,
        distribution_reason_codes=distribution,
        uncertain_labels=labels,
    )
    quality.clear()
    distribution.clear()
    labels.clear()
    assert result.to_dict()["quality_reason_codes"] == ["FLATLINE"]
    assert result.to_dict()["distribution_reason_codes"] == ["DISTANCE"]
    assert result.to_dict()["uncertain_labels"] == ["MI"]


def test_distribution_input_reasons_remain_bound_to_validated_evidence() -> None:
    reasons = ["OUTSIDE_SUPPORT"]
    evidence = replace(
        _valid_inputs(), distribution_supported=False, distribution_reason_codes=reasons
    )
    reasons.clear()
    assert evaluate_trust_policy(evidence).distribution_reason_codes == ("OUTSIDE_SUPPORT",)


@pytest.mark.parametrize(
    "changes",
    [
        {"decision": "ABSTAIN"},
        {"reason_codes": ("CONFORMAL_SET_UNCERTAIN",)},
        {"reason_codes": (None,)},
        {"quality_reason_codes": "FLATLINE"},
        {"distribution_reason_codes": (None,)},
        {"uncertain_labels": ("MI", "MI")},
    ],
)
def test_invalid_result_contracts_raise_policy_error_at_construction(changes) -> None:
    valid = TrustPolicyResult(
        "trust-policy-v1", TrustDecision.ABSTAIN, (TrustReasonCode.CONFORMAL_SET_UNCERTAIN,)
    )
    with pytest.raises(TrustPolicyValidationError):
        replace(valid, **changes)


def test_list_of_valid_allowed_reasons_is_snapshotted_before_exposure_check() -> None:
    result = TrustPolicyResult(
        "trust-policy-v1",
        TrustDecision.PREDICTION_ALLOWED,
        [TrustReasonCode.ALL_TRUST_GATES_PASSED],
    )
    assert result.predictions_exposed
    assert result.reason_codes == (TrustReasonCode.ALL_TRUST_GATES_PASSED,)
