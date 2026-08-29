"""Five-state, reason-aware, fail-closed ECG Trust Sentinel policy.

The policy is deliberately small and deterministic.  It does not fit any
threshold, inspect target labels, or run a classifier.  It combines already
frozen evidence in safety order and controls whether downstream code may expose
class results.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from ecg_trust.conformal import BinaryDecision
from ecg_trust.constants import SUPERCLASSES
from ecg_trust.contracts import TrustDecision
from ecg_trust.quality.signal_quality import QualityStatus, SignalQualityReport


class TrustPolicyValidationError(ValueError):
    """Raised when policy inputs violate the frozen decision contract."""


class TrustReasonCode(StrEnum):
    """Stable, user-explainable reasons emitted by the trust router."""

    RELEASE_INTEGRITY_UNVERIFIED = "RELEASE_INTEGRITY_UNVERIFIED"
    INPUT_CONTRACT_INVALID = "INPUT_CONTRACT_INVALID"
    QUALITY_COMPONENT_UNAVAILABLE = "QUALITY_COMPONENT_UNAVAILABLE"
    SIGNAL_QUALITY_INVALID = "SIGNAL_QUALITY_INVALID"
    SIGNAL_REACQUISITION_REQUIRED = "SIGNAL_REACQUISITION_REQUIRED"
    DISTRIBUTION_COMPONENT_UNAVAILABLE = "DISTRIBUTION_COMPONENT_UNAVAILABLE"
    OUTSIDE_VALIDATED_DISTRIBUTION = "OUTSIDE_VALIDATED_DISTRIBUTION"
    UNCERTAINTY_COMPONENT_UNAVAILABLE = "UNCERTAINTY_COMPONENT_UNAVAILABLE"
    LEGACY_ENTROPY_GATE_REJECTED = "LEGACY_ENTROPY_GATE_REJECTED"
    CONFORMAL_SET_UNCERTAIN = "CONFORMAL_SET_UNCERTAIN"
    ALL_TRUST_GATES_PASSED = "ALL_TRUST_GATES_PASSED"


@dataclass(frozen=True, slots=True)
class TrustPolicyConfig:
    """Immutable behavior of the first Sentinel decision release."""

    version: str = "trust-policy-v1"
    require_legacy_entropy_gate: bool = True
    require_all_label_sets_singleton: bool = True

    def __post_init__(self) -> None:
        if not self.version.strip():
            raise TrustPolicyValidationError("policy version must be non-empty")
        if not self.require_all_label_sets_singleton:
            raise TrustPolicyValidationError(
                "v1 must fail closed when any label prediction set is uncertain"
            )


DEFAULT_TRUST_POLICY_CONFIG = TrustPolicyConfig()


@dataclass(frozen=True, slots=True)
class TrustPolicyInputs:
    """One case's frozen evidence, with missing components represented explicitly."""

    release_integrity_verified: bool
    input_contract_valid: bool
    quality_report: SignalQualityReport | None
    distribution_supported: bool | None
    distribution_reason_codes: tuple[str, ...] = ()
    legacy_entropy_gate_accepted: bool | None = None
    conformal_decisions: tuple[BinaryDecision, ...] | None = None

    def __post_init__(self) -> None:
        if len(set(self.distribution_reason_codes)) != len(self.distribution_reason_codes):
            raise TrustPolicyValidationError("distribution reason codes must be unique")
        if any(not value.strip() for value in self.distribution_reason_codes):
            raise TrustPolicyValidationError("distribution reason codes must be non-empty")
        if self.conformal_decisions is not None and len(self.conformal_decisions) != len(
            SUPERCLASSES
        ):
            raise TrustPolicyValidationError(
                f"conformal_decisions must contain {len(SUPERCLASSES)} labels"
            )


@dataclass(frozen=True, slots=True)
class TrustPolicyResult:
    """Final disposition; only one state permits result exposure."""

    policy_version: str
    decision: TrustDecision
    reason_codes: tuple[TrustReasonCode, ...]
    quality_reason_codes: tuple[str, ...] = ()
    distribution_reason_codes: tuple[str, ...] = ()
    uncertain_labels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.policy_version.strip():
            raise TrustPolicyValidationError("policy_version must be non-empty")
        if not self.reason_codes or len(set(self.reason_codes)) != len(self.reason_codes):
            raise TrustPolicyValidationError("reason_codes must be non-empty and unique")
        if len(set(self.quality_reason_codes)) != len(self.quality_reason_codes):
            raise TrustPolicyValidationError("quality_reason_codes must be unique")
        if len(set(self.distribution_reason_codes)) != len(self.distribution_reason_codes):
            raise TrustPolicyValidationError("distribution_reason_codes must be unique")
        if any(label not in SUPERCLASSES for label in self.uncertain_labels):
            raise TrustPolicyValidationError("uncertain_labels must use canonical labels")
        if self.predictions_exposed != (
            self.reason_codes == (TrustReasonCode.ALL_TRUST_GATES_PASSED,)
        ):
            raise TrustPolicyValidationError(
                "prediction exposure must be justified only by all gates passing"
            )

    @property
    def predictions_exposed(self) -> bool:
        """Whether an API or UI is permitted to reveal class results."""

        return self.decision is TrustDecision.PREDICTION_ALLOWED

    def to_dict(self) -> dict[str, object]:
        """Return a finite JSON-safe policy result without model probabilities."""

        return {
            "policy_version": self.policy_version,
            "decision": self.decision.value,
            "predictions_exposed": self.predictions_exposed,
            "reason_codes": [value.value for value in self.reason_codes],
            "quality_reason_codes": list(self.quality_reason_codes),
            "distribution_reason_codes": list(self.distribution_reason_codes),
            "uncertain_labels": list(self.uncertain_labels),
        }


def evaluate_trust_policy(
    evidence: TrustPolicyInputs,
    *,
    config: TrustPolicyConfig = DEFAULT_TRUST_POLICY_CONFIG,
) -> TrustPolicyResult:
    """Apply the five-state policy in strict safety order.

    Release/input failures precede quality, which precedes distribution support,
    which precedes uncertainty.  Later evidence can never override an earlier
    blocking state.
    """

    if not evidence.release_integrity_verified:
        return _result(
            config,
            TrustDecision.INVALID_INPUT,
            TrustReasonCode.RELEASE_INTEGRITY_UNVERIFIED,
        )
    if not evidence.input_contract_valid:
        return _result(
            config,
            TrustDecision.INVALID_INPUT,
            TrustReasonCode.INPUT_CONTRACT_INVALID,
        )

    quality = evidence.quality_report
    if quality is None:
        return _result(
            config,
            TrustDecision.INVALID_INPUT,
            TrustReasonCode.QUALITY_COMPONENT_UNAVAILABLE,
        )
    quality_reasons = tuple(code.value.upper() for code in quality.reason_codes)
    if quality.status is QualityStatus.INVALID:
        return _result(
            config,
            TrustDecision.INVALID_INPUT,
            TrustReasonCode.SIGNAL_QUALITY_INVALID,
            quality_reason_codes=quality_reasons,
        )
    if quality.status in {QualityStatus.LIMITED, QualityStatus.REACQUIRE}:
        return _result(
            config,
            TrustDecision.REACQUIRE,
            TrustReasonCode.SIGNAL_REACQUISITION_REQUIRED,
            quality_reason_codes=quality_reasons,
        )

    if evidence.distribution_supported is None:
        return _result(
            config,
            TrustDecision.UNSUPPORTED_INPUT,
            TrustReasonCode.DISTRIBUTION_COMPONENT_UNAVAILABLE,
            distribution_reason_codes=evidence.distribution_reason_codes,
        )
    if not evidence.distribution_supported:
        return _result(
            config,
            TrustDecision.UNSUPPORTED_INPUT,
            TrustReasonCode.OUTSIDE_VALIDATED_DISTRIBUTION,
            distribution_reason_codes=evidence.distribution_reason_codes,
        )

    if config.require_legacy_entropy_gate:
        if evidence.legacy_entropy_gate_accepted is None:
            return _result(
                config,
                TrustDecision.ABSTAIN,
                TrustReasonCode.UNCERTAINTY_COMPONENT_UNAVAILABLE,
            )
        if not evidence.legacy_entropy_gate_accepted:
            return _result(
                config,
                TrustDecision.ABSTAIN,
                TrustReasonCode.LEGACY_ENTROPY_GATE_REJECTED,
            )

    decisions = evidence.conformal_decisions
    if decisions is None:
        return _result(
            config,
            TrustDecision.ABSTAIN,
            TrustReasonCode.UNCERTAINTY_COMPONENT_UNAVAILABLE,
        )
    uncertain = tuple(
        label
        for label, decision in zip(SUPERCLASSES, decisions, strict=True)
        if decision is BinaryDecision.UNCERTAIN
    )
    if uncertain:
        return _result(
            config,
            TrustDecision.ABSTAIN,
            TrustReasonCode.CONFORMAL_SET_UNCERTAIN,
            uncertain_labels=uncertain,
        )

    return _result(
        config,
        TrustDecision.PREDICTION_ALLOWED,
        TrustReasonCode.ALL_TRUST_GATES_PASSED,
    )


def _result(
    config: TrustPolicyConfig,
    decision: TrustDecision,
    reason: TrustReasonCode,
    *,
    quality_reason_codes: tuple[str, ...] = (),
    distribution_reason_codes: tuple[str, ...] = (),
    uncertain_labels: tuple[str, ...] = (),
) -> TrustPolicyResult:
    return TrustPolicyResult(
        policy_version=config.version,
        decision=decision,
        reason_codes=(reason,),
        quality_reason_codes=quality_reason_codes,
        distribution_reason_codes=distribution_reason_codes,
        uncertain_labels=uncertain_labels,
    )


__all__ = [
    "DEFAULT_TRUST_POLICY_CONFIG",
    "TrustPolicyConfig",
    "TrustPolicyInputs",
    "TrustPolicyResult",
    "TrustPolicyValidationError",
    "TrustReasonCode",
    "evaluate_trust_policy",
]
