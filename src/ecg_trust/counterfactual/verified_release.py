"""Counterfactual sensitivity checks bound to one verified Sentinel release."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from numpy.typing import ArrayLike

from ecg_trust.contracts import CaseDistributionStatus, TrustDecision
from ecg_trust.counterfactual.counterfactual_review import (
    CounterfactualEvaluation,
    CounterfactualEvaluator,
    CounterfactualProposal,
    CounterfactualStatus,
    CounterfactualValidationError,
    FloatArray,
    PlausibilityConfig,
    QualityGateStatus,
    SentinelAnalysis,
    TargetSuperclass,
    canonical_json,
)
from ecg_trust.quality.signal_quality import SignalMetadata
from ecg_trust.sentinel_engine import SentinelArtifacts, TrustSentinelEngine

VERIFIED_RESULT_SCHEMA = "counterfactual-release-validation-summary-v1"


class ReleaseProvenanceStatus(StrEnum):
    VERIFIED = "verified"
    UNAVAILABLE = "unavailable"
    NOT_EVALUATED = "not_evaluated"


class ReleaseProvenanceReason(StrEnum):
    RELEASE_IDENTITY_CHANGED = "RELEASE_IDENTITY_CHANGED"
    RUNTIME_IDENTITY_UNAVAILABLE = "RUNTIME_IDENTITY_UNAVAILABLE"
    RUNTIME_ANALYSIS_UNAVAILABLE = "RUNTIME_ANALYSIS_UNAVAILABLE"


def _identity(artifacts: SentinelArtifacts) -> dict[str, object]:
    """Expose release-level digests only, never paths or waveform identities."""

    return {
        "release_id": artifacts.release_id,
        "manifest_sha256": artifacts.manifest_sha256,
        "checkpoint_sha256s": [item.file_sha256 for item in artifacts.checkpoint_artifacts],
        "resolved_config_sha256": artifacts.resolved_config_artifact.file_sha256,
        "normalization_sha256": artifacts.normalization_artifact.file_sha256,
        "quality_policy_sha256": artifacts.quality_policy_artifact.file_sha256,
        "decision_policy_sha256": artifacts.decision_policy_artifact.file_sha256,
        "distribution_policy_sha256": artifacts.distribution_artifact.file_sha256,
        "conformal_policy_sha256": artifacts.conformal_artifact.file_sha256,
    }


@dataclass(frozen=True, slots=True)
class ReleaseBoundCounterfactualEvaluation:
    """Add release provenance around the unchanged, disclosure-gated v1 result."""

    artifacts: SentinelArtifacts
    evaluation: CounterfactualEvaluation
    provenance_status: ReleaseProvenanceStatus
    provenance_reasons: tuple[ReleaseProvenanceReason, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.artifacts, SentinelArtifacts) or not isinstance(
            self.evaluation, CounterfactualEvaluation
        ):
            raise CounterfactualValidationError("verified evaluation requires release and result")
        if not isinstance(self.provenance_status, ReleaseProvenanceStatus):
            raise CounterfactualValidationError("invalid release provenance status")
        reasons = tuple(self.provenance_reasons)
        if any(not isinstance(reason, ReleaseProvenanceReason) for reason in reasons):
            raise CounterfactualValidationError("invalid release provenance reason")
        object.__setattr__(self, "provenance_reasons", tuple(dict.fromkeys(reasons)))
        if (self.provenance_status is ReleaseProvenanceStatus.UNAVAILABLE) != bool(reasons):
            raise CounterfactualValidationError("unavailable provenance requires failure reasons")
        if self.evaluation.status is CounterfactualStatus.ACCEPTED_MODEL_SENSITIVITY and (
            self.provenance_status is not ReleaseProvenanceStatus.VERIFIED
        ):
            raise CounterfactualValidationError("accepted sensitivity requires verified provenance")

    def to_public_summary(self) -> dict[str, object]:
        return {
            "schema_version": VERIFIED_RESULT_SCHEMA,
            "release": _identity(self.artifacts),
            "provenance_status": self.provenance_status.value,
            "provenance_reasons": [reason.value for reason in self.provenance_reasons],
            "evaluation": self.evaluation.to_public_summary(),
        }

    def canonical_public_json(self) -> str:
        return canonical_json(self.to_public_summary())


class _VerifiedAnalyzer:
    def __init__(
        self,
        provider: Callable[[], TrustSentinelEngine],
        expected: SentinelArtifacts,
        evaluated_at: datetime,
    ) -> None:
        self.provider = provider
        self.expected = expected
        self.evaluated_at = evaluated_at
        self.reasons: list[ReleaseProvenanceReason] = []
        self.verified_analyses = 0
        self.calls = 0

    def _check_identity(self, engine: TrustSentinelEngine) -> None:
        try:
            observed = engine.verified_artifacts()
        except Exception:
            self.reasons.append(ReleaseProvenanceReason.RUNTIME_IDENTITY_UNAVAILABLE)
            raise CounterfactualValidationError("verified runtime identity unavailable") from None
        if observed != self.expected:
            self.reasons.append(ReleaseProvenanceReason.RELEASE_IDENTITY_CHANGED)
            raise CounterfactualValidationError("counterfactual release identity changed")

    def analyze(self, signal_mv: FloatArray) -> SentinelAnalysis:
        self.calls += 1
        try:
            engine = self.provider()
            if not isinstance(engine, TrustSentinelEngine):
                raise TypeError
        except Exception:
            self.reasons.append(ReleaseProvenanceReason.RUNTIME_IDENTITY_UNAVAILABLE)
            raise CounterfactualValidationError("verified runtime unavailable") from None
        self._check_identity(engine)
        try:
            result = engine.analyze(
                signal_id=f"counterfactual-input-{self.calls}",
                signal_mv=signal_mv,
                metadata=SignalMetadata.canonical(),
                evaluated_at=self.evaluated_at,
                release_integrity_verified=True,
            )
        except Exception:
            self.reasons.append(ReleaseProvenanceReason.RUNTIME_ANALYSIS_UNAVAILABLE)
            raise CounterfactualValidationError("verified runtime analysis unavailable") from None
        self._check_identity(engine)
        if result.release_id != self.expected.release_id:
            self.reasons.append(ReleaseProvenanceReason.RELEASE_IDENTITY_CHANGED)
            raise CounterfactualValidationError("analysis release identity changed")
        allowed = result.decision is TrustDecision.PREDICTION_ALLOWED
        quality = (
            QualityGateStatus.PASS
            if result.quality.passed
            else QualityGateStatus.INVALID
            if result.quality.decision is TrustDecision.INVALID_INPUT
            else QualityGateStatus.REACQUIRE
        )
        analysis = SentinelAnalysis(
            decision=result.decision,
            reason_codes=result.policy.public_reason_codes,
            quality_status=quality,
            artifact_free=result.quality.passed,
            ood_supported=(
                result.distribution is not None
                and result.distribution.status is CaseDistributionStatus.WITHIN_REFERENCE
            ),
            uncertainty_passed=allowed,
            superclass_scores=(
                tuple(zip(TargetSuperclass, result.calibrated_probabilities, strict=True))
                if allowed and result.calibrated_probabilities is not None
                else None
            ),
        )
        self.verified_analyses += 1
        return analysis


class VerifiedCounterfactualEvaluator:
    """Run both waveforms through the full engine under one pinned artifact graph.

    The provider may resolve an active engine, but switching its release between
    calls fails closed. Component loaders remain trusted in-process code; these
    checks verify their artifact identities, not arbitrary changes to model RAM.
    """

    def __init__(
        self,
        engine_provider: Callable[[], TrustSentinelEngine],
        *,
        expected_manifest_sha256: str,
        config: PlausibilityConfig | None = None,
    ) -> None:
        if (
            not isinstance(expected_manifest_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", expected_manifest_sha256) is None
        ):
            raise CounterfactualValidationError("expected manifest must be a lowercase SHA-256")
        try:
            engine = engine_provider()
            if not isinstance(engine, TrustSentinelEngine):
                raise TypeError
            artifacts = engine.verified_artifacts()
        except Exception:
            raise CounterfactualValidationError("verified release is unavailable") from None
        if artifacts.manifest_sha256 != expected_manifest_sha256:
            raise CounterfactualValidationError("expected release manifest does not match")
        self._provider = engine_provider
        self._artifacts = artifacts
        self._config = config or PlausibilityConfig()

    def evaluate(
        self,
        proposal: CounterfactualProposal,
        original_signal_mv: ArrayLike,
        candidate_signal_mv: ArrayLike,
        *,
        evaluated_at: datetime,
    ) -> ReleaseBoundCounterfactualEvaluation:
        if not isinstance(evaluated_at, datetime) or evaluated_at.utcoffset() is None:
            raise CounterfactualValidationError("evaluated_at must be a timezone-aware datetime")
        analyzer = _VerifiedAnalyzer(self._provider, self._artifacts, evaluated_at)
        evaluation = CounterfactualEvaluator(analyzer, config=self._config).evaluate(
            proposal, original_signal_mv, candidate_signal_mv
        )
        if analyzer.calls and analyzer.verified_analyses != 2 and not analyzer.reasons:
            analyzer.reasons.append(ReleaseProvenanceReason.RUNTIME_ANALYSIS_UNAVAILABLE)
        status = (
            ReleaseProvenanceStatus.UNAVAILABLE
            if analyzer.reasons
            else ReleaseProvenanceStatus.VERIFIED
            if analyzer.verified_analyses == 2
            else ReleaseProvenanceStatus.NOT_EVALUATED
        )
        return ReleaseBoundCounterfactualEvaluation(
            artifacts=self._artifacts,
            evaluation=evaluation,
            provenance_status=status,
            provenance_reasons=tuple(analyzer.reasons),
        )
