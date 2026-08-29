"""Adapter from the trust-core engine to the identifier-only HTTP service."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol, cast

from numpy.typing import ArrayLike

from ecg_trust.constants import SUPERCLASSES
from ecg_trust.contracts import TrustDecision
from ecg_trust.quality.signal_quality import SignalMetadata
from ecg_trust.sentinel_engine import SentinelCaseResult
from ecg_trust.service.sentinel_service import (
    AnalysisOutcome,
    ReasonCode,
    ResolvedCase,
    VerifiedRelease,
)


class CoreSentinelEngine(Protocol):
    """Narrow view needed to adapt the core without coupling storage to it."""

    @property
    def release_id(self) -> str: ...

    @property
    def bound_manifest_sha256(self) -> str: ...

    def is_ready(self) -> bool: ...

    def analyze(
        self,
        *,
        signal_id: str,
        signal_mv: ArrayLike,
        metadata: SignalMetadata,
        evaluated_at: datetime,
        release_integrity_verified: bool,
    ) -> SentinelCaseResult: ...


@dataclass(frozen=True, slots=True)
class SentinelCasePayload:
    """Private resolver payload; never serialized by the service boundary."""

    signal_mv: ArrayLike
    metadata: SignalMetadata

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, SignalMetadata):
            raise ValueError("metadata must use the Sentinel physical-signal contract")


class SentinelServiceAnalysisEngine:
    """Run the same complete trust decision for validation and inference routes.

    The validation route intentionally calculates the complete disposition but
    the HTTP layer omits label results.  This prevents a superficial file check
    from being mistaken for permission to reveal predictions.
    """

    def __init__(
        self,
        engine: CoreSentinelEngine,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._engine = engine
        self._clock = clock or (lambda: datetime.now(UTC))

    def is_ready(self) -> bool:
        try:
            return self._engine.is_ready() is True
        except Exception:
            return False

    def is_ready_for_release(self, release: VerifiedRelease) -> bool:
        """Require the provider's fresh release identity to match the loaded engine."""

        try:
            return (
                isinstance(release, VerifiedRelease)
                and release.verified
                and release.locked
                and release.release_id == self._engine.release_id
                and release.artifact_sha256 == self._engine.bound_manifest_sha256
                and self._engine.is_ready() is True
            )
        except Exception:
            return False

    def validate_case(
        self,
        case: ResolvedCase,
        release: VerifiedRelease,
    ) -> AnalysisOutcome:
        return self._analyze(case, release)

    def infer(
        self,
        case: ResolvedCase,
        release: VerifiedRelease,
    ) -> AnalysisOutcome:
        return self._analyze(case, release)

    def _analyze(
        self,
        case: ResolvedCase,
        release: VerifiedRelease,
    ) -> AnalysisOutcome:
        if not release.verified or not release.locked:
            raise RuntimeError("release did not pass the external integrity gate")
        if release.release_id != self._engine.release_id:
            raise RuntimeError("requested release does not match the loaded engine")
        if release.artifact_sha256 != self._engine.bound_manifest_sha256:
            raise RuntimeError("verified release digest does not match the loaded engine")
        if self._engine.is_ready() is not True:
            raise RuntimeError("loaded engine is not ready")
        payload = case.analysis_handle
        if not isinstance(payload, SentinelCasePayload):
            raise TypeError("case resolver returned an incompatible private handle")
        evaluated_at = self._clock()
        if evaluated_at.tzinfo is None or evaluated_at.utcoffset() is None:
            raise RuntimeError("analysis clock must return a timezone-aware timestamp")

        result = self._engine.analyze(
            signal_id=case.case_id,
            signal_mv=payload.signal_mv,
            metadata=payload.metadata,
            evaluated_at=evaluated_at,
            release_integrity_verified=True,
        )
        reasons = tuple(ReasonCode(reason.value) for reason in result.policy.reason_codes)
        if result.decision is not TrustDecision.PREDICTION_ALLOWED:
            return AnalysisOutcome(decision=result.decision, reason_codes=reasons)

        probabilities = cast(tuple[float, ...], result.calibrated_probabilities)
        return AnalysisOutcome(
            decision=result.decision,
            reason_codes=reasons,
            labels=SUPERCLASSES,
            probabilities=probabilities,
        )


__all__ = [
    "CoreSentinelEngine",
    "SentinelCasePayload",
    "SentinelServiceAnalysisEngine",
]
