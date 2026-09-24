"""Five-state, reason-aware, fail-closed ECG Trust Sentinel policy.

The policy is deliberately small and deterministic.  It does not fit any
threshold, inspect target labels, or run a classifier.  It combines already
frozen evidence in safety order and controls whether downstream code may expose
class results.  An optional, default-off label-coherence gate (see
:mod:`ecg_trust.label_coherence`) can also withhold a jointly implausible set of
singleton decisions; it is not part of ``trust-policy-v1``.  Its specific reason
and labels are audit-only: :attr:`TrustPolicyResult.public_reason_codes` is the
only reason vocabulary that may cross a public boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from ecg_trust.conformal import BinaryDecision
from ecg_trust.constants import SUPERCLASSES
from ecg_trust.contracts import TrustDecision
from ecg_trust.label_coherence import find_incoherent_labels
from ecg_trust.quality.signal_quality import QualityStatus, SignalQualityReport

_V1_POLICY_VERSION: Final = "trust-policy-v1"
PUBLIC_CONFIDENCE_ABSTENTION_REASON: Final = "CONFIDENCE_GATE_ABSTAINED"


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
    LABEL_SET_INCOHERENT = "LABEL_SET_INCOHERENT"
    ALL_TRUST_GATES_PASSED = "ALL_TRUST_GATES_PASSED"


# Under the coherence gate these two abstentions share one public code, so a
# caller cannot tell a jointly supported NORM plus MI/STTC/HYP set (the only way
# LABEL_SET_INCOHERENT arises) from an uncertain label-wise set.
_COHERENCE_MASKED_REASONS: Final = frozenset(
    {TrustReasonCode.CONFORMAL_SET_UNCERTAIN, TrustReasonCode.LABEL_SET_INCOHERENT}
)


@dataclass(frozen=True, slots=True)
class TrustPolicyConfig:
    """Immutable Sentinel decision-policy behavior.

    The defaults are ``trust-policy-v1``, the first Sentinel decision release.
    Any other configuration is development-only.  ``require_label_coherence`` is
    an opt-in development gate that is off by default.  It cannot be enabled
    under the v1 version string, so its results are never attributed to
    ``trust-policy-v1``; enabling it for any release requires a new
    preregistered protocol.
    """

    version: str = _V1_POLICY_VERSION
    require_legacy_entropy_gate: bool = True
    require_all_label_sets_singleton: bool = True
    require_label_coherence: bool = False

    def __post_init__(self) -> None:
        if not self.version.strip():
            raise TrustPolicyValidationError("policy version must be non-empty")
        if (
            not isinstance(self.require_legacy_entropy_gate, bool)
            or not isinstance(self.require_all_label_sets_singleton, bool)
            or not isinstance(self.require_label_coherence, bool)
        ):
            raise TrustPolicyValidationError("policy gate settings must be boolean")
        if not self.require_all_label_sets_singleton:
            raise TrustPolicyValidationError(
                "v1 must fail closed when any label prediction set is uncertain"
            )
        if self.require_label_coherence and self.version.strip() == _V1_POLICY_VERSION:
            raise TrustPolicyValidationError(
                "the label-coherence gate requires a policy version other than v1"
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
        for name in ("release_integrity_verified", "input_contract_valid"):
            if not isinstance(getattr(self, name), bool):
                raise TrustPolicyValidationError(f"{name} must be boolean")
        for name in ("distribution_supported", "legacy_entropy_gate_accepted"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, bool):
                raise TrustPolicyValidationError(f"{name} must be boolean or None")
        if self.quality_report is not None:
            if not isinstance(self.quality_report, SignalQualityReport):
                raise TrustPolicyValidationError("quality_report must be a SignalQualityReport")
            if not isinstance(self.quality_report.status, QualityStatus):
                raise TrustPolicyValidationError("quality status must use QualityStatus")
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
        if self.conformal_decisions is not None and any(
            not isinstance(decision, BinaryDecision) for decision in self.conformal_decisions
        ):
            raise TrustPolicyValidationError("conformal_decisions must use BinaryDecision")


@dataclass(frozen=True, slots=True)
class TrustPolicyResult:
    """Final disposition; only one state permits result exposure.

    This is the internal audit record.  ``uncertain_labels`` and
    ``incoherent_labels`` name labels of a withheld case, and
    ``LABEL_SET_INCOHERENT`` implies which labels were supported, so public
    boundaries publish :attr:`public_reason_codes` instead of these fields.
    """

    policy_version: str
    decision: TrustDecision
    reason_codes: tuple[TrustReasonCode, ...]
    quality_reason_codes: tuple[str, ...] = ()
    distribution_reason_codes: tuple[str, ...] = ()
    uncertain_labels: tuple[str, ...] = ()
    incoherent_labels: tuple[str, ...] = ()
    label_coherence_required: bool = False

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
        if any(label not in SUPERCLASSES for label in self.incoherent_labels) or len(
            set(self.incoherent_labels)
        ) != len(self.incoherent_labels):
            raise TrustPolicyValidationError("incoherent_labels must be unique canonical labels")
        if (TrustReasonCode.LABEL_SET_INCOHERENT in self.reason_codes) != bool(
            self.incoherent_labels
        ):
            raise TrustPolicyValidationError(
                "incoherent_labels must be reported exactly for an incoherent label set"
            )
        if len(self.incoherent_labels) == 1:
            raise TrustPolicyValidationError("an incoherent label set names at least two labels")
        if self.predictions_exposed != (
            self.reason_codes == (TrustReasonCode.ALL_TRUST_GATES_PASSED,)
        ):
            raise TrustPolicyValidationError(
                "prediction exposure must be justified only by all gates passing"
            )
        if not isinstance(self.label_coherence_required, bool):
            raise TrustPolicyValidationError("label_coherence_required must be boolean")
        if self.label_coherence_required and self.policy_version.strip() == _V1_POLICY_VERSION:
            raise TrustPolicyValidationError(
                "the label-coherence gate requires a policy version other than v1"
            )
        if TrustReasonCode.LABEL_SET_INCOHERENT in self.reason_codes:
            self._validate_incoherent_label_set()

    def _validate_incoherent_label_set(self) -> None:
        if not self.label_coherence_required:
            raise TrustPolicyValidationError(
                "an incoherent label set requires the label-coherence gate"
            )
        if self.decision is not TrustDecision.ABSTAIN:
            raise TrustPolicyValidationError("an incoherent label set must end in ABSTAIN")
        if self.uncertain_labels or TrustReasonCode.CONFORMAL_SET_UNCERTAIN in self.reason_codes:
            raise TrustPolicyValidationError(
                "label coherence is defined only for singleton label sets"
            )
        supported = tuple(
            BinaryDecision.SUPPORTED
            if label in self.incoherent_labels
            else BinaryDecision.NOT_SUPPORTED
            for label in SUPERCLASSES
        )
        if find_incoherent_labels(supported) != self.incoherent_labels:
            raise TrustPolicyValidationError(
                "incoherent_labels must be exactly the labels of listed incompatible pairs, "
                "in canonical order"
            )

    @property
    def predictions_exposed(self) -> bool:
        """Whether an API or UI is permitted to reveal class results."""

        return self.decision is TrustDecision.PREDICTION_ALLOWED

    @property
    def public_reason_codes(self) -> tuple[str, ...]:
        """Return the reason codes that may cross a public boundary.

        Without the coherence gate these are the policy reasons unchanged.
        ``LABEL_SET_INCOHERENT`` arises only when ``NORM`` and at least one of
        ``MI``, ``STTC``, or ``HYP`` are ``SUPPORTED``, so publishing it would
        disclose class results of a withheld case.  Under the gate, that reason and
        ``CONFORMAL_SET_UNCERTAIN`` are therefore both published as the generic
        ``CONFIDENCE_GATE_ABSTAINED``, which does not say which gate fired.
        """

        if not self.label_coherence_required:
            return tuple(reason.value for reason in self.reason_codes)
        public: list[str] = []
        for reason in self.reason_codes:
            value = (
                PUBLIC_CONFIDENCE_ABSTENTION_REASON
                if reason in _COHERENCE_MASKED_REASONS
                else reason.value
            )
            if value not in public:
                public.append(value)
        return tuple(public)

    def to_dict(self) -> dict[str, object]:
        """Return a finite JSON-safe policy result without model probabilities."""

        payload: dict[str, object] = {
            "policy_version": self.policy_version,
            "decision": self.decision.value,
            "predictions_exposed": self.predictions_exposed,
            "reason_codes": [value.value for value in self.reason_codes],
            "quality_reason_codes": list(self.quality_reason_codes),
            "distribution_reason_codes": list(self.distribution_reason_codes),
            "uncertain_labels": list(self.uncertain_labels),
        }
        # Only the opt-in coherence gate populates these fields, so default-policy
        # payloads keep their existing keys exactly.  This is an audit record: the
        # labels reveal supported decisions of a withheld case, so public
        # boundaries use ``public_reason_codes`` and never this payload.
        if self.label_coherence_required:
            payload["label_coherence_required"] = True
        if self.incoherent_labels:
            payload["incoherent_labels"] = list(self.incoherent_labels)
        return payload


def evaluate_trust_policy(
    evidence: TrustPolicyInputs,
    *,
    config: TrustPolicyConfig = DEFAULT_TRUST_POLICY_CONFIG,
) -> TrustPolicyResult:
    """Apply the five-state policy in strict safety order.

    Release/input failures precede quality, which precedes distribution support,
    which precedes uncertainty.  Later evidence can never override an earlier
    blocking state.  The opt-in label-coherence gate runs last, only after every
    conformal set is a singleton.
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

    if config.require_label_coherence:
        # Inputs validate each decision's type and count, not the container type.
        incoherent = find_incoherent_labels(tuple(decisions))
        if incoherent:
            return _result(
                config,
                TrustDecision.ABSTAIN,
                TrustReasonCode.LABEL_SET_INCOHERENT,
                incoherent_labels=incoherent,
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
    incoherent_labels: tuple[str, ...] = (),
) -> TrustPolicyResult:
    return TrustPolicyResult(
        policy_version=config.version,
        decision=decision,
        reason_codes=(reason,),
        quality_reason_codes=quality_reason_codes,
        distribution_reason_codes=distribution_reason_codes,
        uncertain_labels=uncertain_labels,
        incoherent_labels=incoherent_labels,
        label_coherence_required=config.require_label_coherence,
    )


__all__ = [
    "DEFAULT_TRUST_POLICY_CONFIG",
    "PUBLIC_CONFIDENCE_ABSTENTION_REASON",
    "TrustPolicyConfig",
    "TrustPolicyInputs",
    "TrustPolicyResult",
    "TrustPolicyValidationError",
    "TrustReasonCode",
    "evaluate_trust_policy",
]
