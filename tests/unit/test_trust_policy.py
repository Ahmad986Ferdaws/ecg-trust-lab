from __future__ import annotations

import json
from dataclasses import replace
from itertools import product
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
    DEFAULT_TRUST_POLICY_CONFIG,
    PUBLIC_CONFIDENCE_ABSTENTION_REASON,
    TrustPolicyConfig,
    TrustPolicyInputs,
    TrustPolicyResult,
    TrustPolicyValidationError,
    TrustReasonCode,
    evaluate_trust_policy,
)

COHERENCE_CONFIG = TrustPolicyConfig(
    version="trust-policy-label-coherence-dev",
    require_label_coherence=True,
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
    "field",
    [
        "require_legacy_entropy_gate",
        "require_all_label_sets_singleton",
        "require_label_coherence",
    ],
)
@pytest.mark.parametrize("value", ["false", 0, 1, None])
def test_policy_rejects_non_boolean_gate_settings(field: str, value: object) -> None:
    with pytest.raises(TrustPolicyValidationError, match="boolean"):
        replace(TrustPolicyConfig(), **{field: value})


def test_policy_rejects_untyped_quality_report() -> None:
    with pytest.raises(TrustPolicyValidationError, match="SignalQualityReport"):
        replace(_valid_inputs(), quality_report=cast(SignalQualityReport, {"status": "pass"}))


def _supported(*labels: str) -> tuple[BinaryDecision, ...]:
    return tuple(
        BinaryDecision.SUPPORTED if label in labels else BinaryDecision.NOT_SUPPORTED
        for label in SUPERCLASSES
    )


def test_default_policy_keeps_label_coherence_disabled_and_payload_unchanged() -> None:
    result = evaluate_trust_policy(
        replace(_valid_inputs(), conformal_decisions=_supported("NORM", "MI"))
    )

    assert TrustPolicyConfig() == DEFAULT_TRUST_POLICY_CONFIG
    assert DEFAULT_TRUST_POLICY_CONFIG.version == "trust-policy-v1"
    assert DEFAULT_TRUST_POLICY_CONFIG.require_label_coherence is False
    assert result.decision is TrustDecision.PREDICTION_ALLOWED
    assert result.incoherent_labels == ()
    assert result.label_coherence_required is False
    assert result.public_reason_codes == ("ALL_TRUST_GATES_PASSED",)
    assert json.dumps(result.to_dict()) == (
        '{"policy_version": "trust-policy-v1", "decision": "PREDICTION_ALLOWED", '
        '"predictions_exposed": true, "reason_codes": ["ALL_TRUST_GATES_PASSED"], '
        '"quality_reason_codes": [], "distribution_reason_codes": [], '
        '"uncertain_labels": []}'
    )


def test_coherence_gate_withholds_norm_with_infarction_that_v1_releases() -> None:
    evidence = replace(_valid_inputs(), conformal_decisions=_supported("NORM", "MI"))

    before = evaluate_trust_policy(evidence)
    after = evaluate_trust_policy(evidence, config=COHERENCE_CONFIG)

    assert before.predictions_exposed
    assert after.decision is TrustDecision.ABSTAIN
    assert not after.predictions_exposed
    assert after.reason_codes == (TrustReasonCode.LABEL_SET_INCOHERENT,)
    assert after.incoherent_labels == ("NORM", "MI")
    assert after.uncertain_labels == ()
    assert after.public_reason_codes == (PUBLIC_CONFIDENCE_ABSTENTION_REASON,)
    assert after.to_dict() == {
        "policy_version": "trust-policy-label-coherence-dev",
        "decision": "ABSTAIN",
        "predictions_exposed": False,
        "reason_codes": ["LABEL_SET_INCOHERENT"],
        "quality_reason_codes": [],
        "distribution_reason_codes": [],
        "uncertain_labels": [],
        "label_coherence_required": True,
        "incoherent_labels": ["NORM", "MI"],
    }


@pytest.mark.parametrize(
    ("supported", "incoherent"),
    [
        (("NORM", "CD"), ()),
        (("MI", "STTC", "CD", "HYP"), ()),
        (("NORM", "STTC", "CD"), ("NORM", "STTC")),
        (SUPERCLASSES, ("NORM", "MI", "STTC", "HYP")),
    ],
)
def test_coherence_gate_allows_norm_with_conduction_and_names_conflicts(
    supported: tuple[str, ...],
    incoherent: tuple[str, ...],
) -> None:
    result = evaluate_trust_policy(
        replace(_valid_inputs(), conformal_decisions=_supported(*supported)),
        config=COHERENCE_CONFIG,
    )

    assert result.predictions_exposed is (not incoherent)
    assert result.incoherent_labels == incoherent


def test_coherence_gate_changes_only_otherwise_allowed_singleton_sets() -> None:
    for decisions in product(
        (BinaryDecision.NOT_SUPPORTED, BinaryDecision.SUPPORTED), repeat=len(SUPERCLASSES)
    ):
        evidence = replace(_valid_inputs(), conformal_decisions=decisions)
        default = evaluate_trust_policy(evidence)
        gated = evaluate_trust_policy(evidence, config=COHERENCE_CONFIG)
        supported = {
            label
            for label, decision in zip(SUPERCLASSES, decisions, strict=True)
            if decision is BinaryDecision.SUPPORTED
        }
        conflict = "NORM" in supported and bool(supported & {"MI", "STTC", "HYP"})

        assert default.decision is TrustDecision.PREDICTION_ALLOWED
        assert gated.predictions_exposed is not conflict
        if conflict:
            assert gated.reason_codes == (TrustReasonCode.LABEL_SET_INCOHERENT,)
        else:
            assert (
                replace(
                    gated,
                    policy_version=default.policy_version,
                    label_coherence_required=False,
                )
                == default
            )


@pytest.mark.parametrize(
    "changes",
    [
        {"release_integrity_verified": False},
        {"input_contract_valid": False},
        {"quality_report": None},
        {"quality_report": _quality(QualityStatus.INVALID)},
        {"quality_report": _quality(QualityStatus.REACQUIRE)},
        {"distribution_supported": None},
        {"distribution_supported": False},
        {"legacy_entropy_gate_accepted": None},
        {"legacy_entropy_gate_accepted": False},
        {"conformal_decisions": None},
        {
            "conformal_decisions": (
                BinaryDecision.SUPPORTED,
                BinaryDecision.SUPPORTED,
                BinaryDecision.UNCERTAIN,
                BinaryDecision.NOT_SUPPORTED,
                BinaryDecision.NOT_SUPPORTED,
            )
        },
    ],
)
def test_earlier_gates_keep_precedence_over_label_coherence(changes: dict[str, object]) -> None:
    evidence = replace(
        replace(_valid_inputs(), conformal_decisions=_supported("NORM", "MI")),
        **changes,
    )

    default = evaluate_trust_policy(evidence)
    gated = evaluate_trust_policy(evidence, config=COHERENCE_CONFIG)

    assert TrustReasonCode.LABEL_SET_INCOHERENT not in gated.reason_codes
    assert (
        replace(gated, policy_version=default.policy_version, label_coherence_required=False)
        == default
    )


def test_label_coherence_cannot_be_attributed_to_the_v1_policy() -> None:
    with pytest.raises(TrustPolicyValidationError, match="other than v1"):
        TrustPolicyConfig(require_label_coherence=True)
    with pytest.raises(TrustPolicyValidationError, match="other than v1"):
        TrustPolicyConfig(version=" trust-policy-v1 ", require_label_coherence=True)


@pytest.mark.parametrize(
    ("decision", "reasons", "incoherent", "message"),
    [
        (
            TrustDecision.ABSTAIN,
            (TrustReasonCode.LABEL_SET_INCOHERENT,),
            (),
            "exactly for an incoherent",
        ),
        (
            TrustDecision.ABSTAIN,
            (TrustReasonCode.CONFORMAL_SET_UNCERTAIN,),
            ("NORM", "MI"),
            "exactly for an incoherent",
        ),
        (
            TrustDecision.PREDICTION_ALLOWED,
            (TrustReasonCode.ALL_TRUST_GATES_PASSED,),
            ("NORM", "MI"),
            "exactly for an incoherent",
        ),
        (
            TrustDecision.ABSTAIN,
            (TrustReasonCode.LABEL_SET_INCOHERENT,),
            ("NORM",),
            "at least two",
        ),
        (
            TrustDecision.ABSTAIN,
            (TrustReasonCode.LABEL_SET_INCOHERENT,),
            ("NORM", "AFIB"),
            "unique canonical",
        ),
        (
            TrustDecision.ABSTAIN,
            (TrustReasonCode.LABEL_SET_INCOHERENT,),
            ("NORM", "NORM"),
            "unique canonical",
        ),
        (
            TrustDecision.PREDICTION_ALLOWED,
            (TrustReasonCode.LABEL_SET_INCOHERENT,),
            ("NORM", "MI"),
            "all gates passing",
        ),
    ],
)
def test_result_rejects_incoherent_labels_without_a_matching_reason(
    decision: TrustDecision,
    reasons: tuple[TrustReasonCode, ...],
    incoherent: tuple[str, ...],
    message: str,
) -> None:
    with pytest.raises(TrustPolicyValidationError, match=message):
        TrustPolicyResult(
            policy_version="trust-policy-label-coherence-dev",
            decision=decision,
            reason_codes=reasons,
            incoherent_labels=incoherent,
        )


def _incoherent_result(**changes: object) -> TrustPolicyResult:
    fields: dict[str, object] = {
        "policy_version": "trust-policy-label-coherence-dev",
        "decision": TrustDecision.ABSTAIN,
        "reason_codes": (TrustReasonCode.LABEL_SET_INCOHERENT,),
        "incoherent_labels": ("NORM", "MI"),
        "label_coherence_required": True,
    }
    fields.update(changes)
    return TrustPolicyResult(**fields)  # type: ignore[arg-type]


def test_incoherent_result_matches_what_the_gate_emits() -> None:
    evidence = replace(_valid_inputs(), conformal_decisions=_supported("NORM", "MI"))

    assert evaluate_trust_policy(evidence, config=COHERENCE_CONFIG) == _incoherent_result()


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        (
            {"policy_version": "trust-policy-v1"},
            "other than v1",
        ),
        (
            {"policy_version": " trust-policy-v1 ", "label_coherence_required": False},
            "requires the label-coherence gate",
        ),
        ({"label_coherence_required": False}, "requires the label-coherence gate"),
        ({"label_coherence_required": 1}, "must be boolean"),
        ({"decision": TrustDecision.INVALID_INPUT}, "must end in ABSTAIN"),
        ({"uncertain_labels": ("STTC",)}, "singleton label sets"),
        (
            {
                "reason_codes": (
                    TrustReasonCode.LABEL_SET_INCOHERENT,
                    TrustReasonCode.CONFORMAL_SET_UNCERTAIN,
                )
            },
            "singleton label sets",
        ),
        ({"incoherent_labels": ("MI", "NORM")}, "canonical order"),
        ({"incoherent_labels": ("MI", "CD")}, "listed incompatible pairs"),
        ({"incoherent_labels": ("NORM", "MI", "CD")}, "listed incompatible pairs"),
    ],
)
def test_incoherent_result_rejects_states_the_gate_cannot_emit(
    changes: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(TrustPolicyValidationError, match=message):
        _incoherent_result(**changes)


def test_gated_result_cannot_be_attributed_to_the_v1_policy() -> None:
    with pytest.raises(TrustPolicyValidationError, match="other than v1"):
        TrustPolicyResult(
            policy_version="trust-policy-v1",
            decision=TrustDecision.PREDICTION_ALLOWED,
            reason_codes=(TrustReasonCode.ALL_TRUST_GATES_PASSED,),
            label_coherence_required=True,
        )


@pytest.mark.parametrize(
    "reason",
    [reason for reason in TrustReasonCode if reason is not TrustReasonCode.LABEL_SET_INCOHERENT],
)
def test_public_reason_codes_are_unchanged_without_the_coherence_gate(
    reason: TrustReasonCode,
) -> None:
    exposed = reason is TrustReasonCode.ALL_TRUST_GATES_PASSED
    result = TrustPolicyResult(
        policy_version="trust-policy-v1",
        decision=TrustDecision.PREDICTION_ALLOWED if exposed else TrustDecision.ABSTAIN,
        reason_codes=(reason,),
    )

    assert result.public_reason_codes == (reason.value,)
    assert result.public_reason_codes == tuple(cast(list[str], result.to_dict()["reason_codes"]))


def test_coherence_gate_publishes_one_generic_reason_for_classifier_abstentions() -> None:
    incoherent = evaluate_trust_policy(
        replace(_valid_inputs(), conformal_decisions=_supported("NORM", "STTC", "HYP")),
        config=COHERENCE_CONFIG,
    )
    uncertain = evaluate_trust_policy(
        replace(
            _valid_inputs(),
            conformal_decisions=(BinaryDecision.UNCERTAIN,) + _supported()[1:],
        ),
        config=COHERENCE_CONFIG,
    )
    entropy = evaluate_trust_policy(
        replace(_valid_inputs(), legacy_entropy_gate_accepted=False),
        config=COHERENCE_CONFIG,
    )
    allowed = evaluate_trust_policy(_valid_inputs(), config=COHERENCE_CONFIG)

    assert incoherent.reason_codes == (TrustReasonCode.LABEL_SET_INCOHERENT,)
    assert uncertain.reason_codes == (TrustReasonCode.CONFORMAL_SET_UNCERTAIN,)
    assert incoherent.public_reason_codes == (PUBLIC_CONFIDENCE_ABSTENTION_REASON,)
    assert uncertain.public_reason_codes == incoherent.public_reason_codes
    assert entropy.public_reason_codes == ("LEGACY_ENTROPY_GATE_REJECTED",)
    assert allowed.public_reason_codes == ("ALL_TRUST_GATES_PASSED",)
