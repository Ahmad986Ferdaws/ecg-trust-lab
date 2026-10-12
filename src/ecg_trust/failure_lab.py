"""Controlled single-case stress experiments over the complete trust pipeline."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, cast

from torch import Tensor

from ecg_trust.constants import SUPERCLASSES
from ecg_trust.contracts import SignalSourceKind, TrustDecision
from ecg_trust.quality.signal_quality import SignalMetadata
from ecg_trust.sentinel_engine import SentinelCaseResult
from ecg_trust.stress import (
    CONTROLLED_SENSITIVITY_LABEL,
    StressProvenance,
    StressScenario,
    apply_stress_scenario,
)
from ecg_trust.trust_policy import PUBLIC_CONFIDENCE_ABSTENTION_REASON, TrustReasonCode

_PUBLIC_REASONS = frozenset(
    reason.value for reason in TrustReasonCode if reason is not TrustReasonCode.LABEL_SET_INCOHERENT
) | {PUBLIC_CONFIDENCE_ABSTENTION_REASON}


class FailureLabError(RuntimeError):
    """Raised when a controlled comparison cannot be interpreted safely."""


class FailureLabAnalyzer(Protocol):
    """Complete Sentinel analysis required for each side of a comparison."""

    def analyze(
        self,
        *,
        signal_id: str,
        signal_mv: object,
        metadata: SignalMetadata,
        evaluated_at: datetime,
        release_integrity_verified: bool,
    ) -> SentinelCaseResult: ...


@dataclass(frozen=True, slots=True)
class FailureLabDecision:
    """Disclosure-safe snapshot of one side of a controlled experiment."""

    decision: TrustDecision
    reason_codes: tuple[str, ...]
    probabilities: tuple[float, ...] | None

    def __post_init__(self) -> None:
        if not isinstance(self.decision, TrustDecision):
            raise FailureLabError("decision must use the closed trust-decision vocabulary")
        if (
            not isinstance(self.reason_codes, (tuple, list))
            or not 1 <= len(self.reason_codes) <= len(_PUBLIC_REASONS)
        ):
            raise FailureLabError("reason codes must be a non-empty bounded tuple or list")
        reasons = tuple(self.reason_codes)
        if any(not isinstance(reason, str) or reason not in _PUBLIC_REASONS for reason in reasons):
            raise FailureLabError("reason codes must use the public trust vocabulary")
        if len(set(reasons)) != len(reasons):
            raise FailureLabError("reason codes must be unique")
        object.__setattr__(self, "reason_codes", reasons)
        exposed = self.decision is TrustDecision.PREDICTION_ALLOWED
        if exposed != (self.probabilities is not None):
            raise FailureLabError("probabilities must follow the trust disclosure decision")
        if self.probabilities is not None:
            if not isinstance(self.probabilities, (tuple, list)) or len(self.probabilities) != len(
                SUPERCLASSES
            ):
                raise FailureLabError("an allowed Failure Lab result requires five probabilities")
            probabilities = tuple(self.probabilities)
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not 0 <= value <= 1
                for value in probabilities
            ):
                raise FailureLabError("probabilities must be finite real numbers in [0, 1]")
            object.__setattr__(
                self, "probabilities", tuple(float(value) for value in probabilities)
            )

    @classmethod
    def from_case_result(cls, result: SentinelCaseResult) -> FailureLabDecision:
        return cls(
            decision=result.decision,
            reason_codes=result.policy.public_reason_codes,
            probabilities=result.calibrated_probabilities,
        )

    def to_public_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "decision": self.decision.value,
            "reason_codes": list(self.reason_codes),
            "predictions_exposed": self.probabilities is not None,
        }
        if self.probabilities is not None:
            payload["probabilities"] = dict(zip(SUPERCLASSES, self.probabilities, strict=True))
        return payload


@dataclass(frozen=True, slots=True)
class FailureLabTrial:
    """One baseline-versus-stress comparison without waveform or record IDs."""

    source_kind: SignalSourceKind
    scenario: StressScenario
    provenance: StressProvenance
    baseline: FailureLabDecision
    stressed: FailureLabDecision
    probability_delta: tuple[float, ...] | None

    def __post_init__(self) -> None:
        if not isinstance(self.source_kind, SignalSourceKind):
            raise FailureLabError("source_kind must use the closed signal-source vocabulary")
        if not isinstance(self.baseline, FailureLabDecision) or not isinstance(
            self.stressed, FailureLabDecision
        ):
            raise FailureLabError("trial decisions must be validated FailureLabDecision values")
        both_exposed = (
            self.baseline.probabilities is not None and self.stressed.probabilities is not None
        )
        if both_exposed != (self.probability_delta is not None):
            raise FailureLabError(
                "probability deltas require prediction permission on both waveforms"
            )
        if self.probability_delta is not None:
            if not isinstance(self.probability_delta, (tuple, list)) or len(
                self.probability_delta
            ) != len(SUPERCLASSES):
                raise FailureLabError("probability deltas must contain five values")
            delta = tuple(self.probability_delta)
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not -1 <= value <= 1
                for value in delta
            ):
                raise FailureLabError("probability deltas must be finite real numbers in [-1, 1]")
            baseline = cast(tuple[float, ...], self.baseline.probabilities)
            stressed = cast(tuple[float, ...], self.stressed.probabilities)
            expected = tuple(
                after - before for before, after in zip(baseline, stressed, strict=True)
            )
            if any(
                not math.isclose(value, actual, rel_tol=1e-12, abs_tol=1e-12)
                for value, actual in zip(delta, expected, strict=True)
            ):
                raise FailureLabError("probability deltas do not match the two decisions")
            object.__setattr__(self, "probability_delta", tuple(float(value) for value in delta))
        if self.scenario.scenario_sha256 != self.provenance.scenario_sha256:
            raise FailureLabError("scenario and applied provenance do not match")

    @property
    def decision_changed(self) -> bool:
        return self.baseline.decision is not self.stressed.decision

    def to_public_dict(self) -> dict[str, object]:
        """Exclude waveform hashes and every source identifier from public output."""

        if self.source_kind is not SignalSourceKind.SYNTHETIC:
            raise FailureLabError("only synthetic Failure Lab trials may cross the public boundary")

        payload: dict[str, object] = {
            "schema_version": "ecg_trust.failure_lab_trial.v1",
            "scenario": {
                "scenario_id": self.scenario.scenario_id,
                "scenario_sha256": self.scenario.scenario_sha256,
                "kind": self.scenario.kind.value,
                "parameters": self.scenario.parameters,
                "affected_leads": list(self.provenance.affected_leads),
            },
            "baseline": self.baseline.to_public_dict(),
            "stressed": self.stressed.to_public_dict(),
            "decision_changed": self.decision_changed,
            "interpretation": CONTROLLED_SENSITIVITY_LABEL,
            "limitations": [
                "This is a controlled model-sensitivity probe, not clinical distribution shift.",
                "It does not establish diagnostic safety, prevalence, or causal physiology.",
                "Research use only; not for clinical decision-making.",
            ],
        }
        if self.probability_delta is not None:
            payload["probability_delta"] = dict(
                zip(SUPERCLASSES, self.probability_delta, strict=True)
            )
        return payload


class FailureLabRunner:
    """Apply one deterministic scenario and rerun every Sentinel trust gate."""

    def __init__(self, analyzer: FailureLabAnalyzer) -> None:
        self._analyzer = analyzer

    def run(
        self,
        *,
        signal_id: str,
        waveform_mv: Tensor,
        metadata: SignalMetadata,
        source_kind: SignalSourceKind,
        scenario: StressScenario,
        evaluated_at: datetime,
        release_integrity_verified: bool,
    ) -> FailureLabTrial:
        baseline_result = self._analyzer.analyze(
            signal_id=signal_id,
            signal_mv=waveform_mv.detach().cpu().numpy(),
            metadata=metadata,
            evaluated_at=evaluated_at,
            release_integrity_verified=release_integrity_verified,
        )
        baseline = FailureLabDecision.from_case_result(baseline_result)
        if baseline.decision is not TrustDecision.PREDICTION_ALLOWED:
            raise FailureLabError("baseline must pass every trust gate before a stress comparison")

        applied = apply_stress_scenario(waveform_mv, scenario)
        stressed_result = self._analyzer.analyze(
            signal_id=signal_id,
            signal_mv=applied.waveform.detach().cpu().numpy(),
            metadata=metadata,
            evaluated_at=evaluated_at,
            release_integrity_verified=release_integrity_verified,
        )
        stressed = FailureLabDecision.from_case_result(stressed_result)
        delta: tuple[float, ...] | None = None
        if stressed.probabilities is not None:
            baseline_probabilities = cast(tuple[float, ...], baseline.probabilities)
            delta = tuple(
                stressed_value - baseline_value
                for stressed_value, baseline_value in zip(
                    stressed.probabilities,
                    baseline_probabilities,
                    strict=True,
                )
            )
        return FailureLabTrial(
            source_kind=source_kind,
            scenario=scenario,
            provenance=applied.provenance,
            baseline=baseline,
            stressed=stressed,
            probability_delta=delta,
        )


__all__ = [
    "FailureLabAnalyzer",
    "FailureLabDecision",
    "FailureLabError",
    "FailureLabRunner",
    "FailureLabTrial",
]
