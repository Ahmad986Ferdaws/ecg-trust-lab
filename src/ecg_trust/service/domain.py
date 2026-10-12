"""Transport-independent Sentinel case, release and analysis contracts.

Storage providers and analysis adapters depend on these contracts; HTTP routes
remain in sentinel_service, which re-exports the original public names.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Final, Protocol

from ecg_trust.contracts import TrustDecision

OPAQUE_ID_PATTERN: Final = r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$"
LABEL_PATTERN: Final = re.compile(r"^[A-Z][A-Z0-9_]{0,31}$")
SHA256_PATTERN: Final = re.compile(r"^[a-f0-9]{64}$")
OPAQUE_ID_RE: Final = re.compile(OPAQUE_ID_PATTERN)


class ReasonCode(StrEnum):
    """Stable, non-sensitive reason codes safe to expose to API consumers."""

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
    CONFIDENCE_GATE_ABSTAINED = "CONFIDENCE_GATE_ABSTAINED"
    ALL_TRUST_GATES_PASSED = "ALL_TRUST_GATES_PASSED"
    RELEASE_NOT_READY = "RELEASE_NOT_READY"
    BACKEND_UNAVAILABLE = "BACKEND_UNAVAILABLE"
    SERVICE_BUSY = "SERVICE_BUSY"


class CaseKind(StrEnum):
    """Permitted provenance classes; arbitrary public uploads are out of scope."""

    AUTHORIZED = "AUTHORIZED"
    SYNTHETIC = "SYNTHETIC"


def _is_opaque_id(value: str) -> bool:
    return OPAQUE_ID_RE.fullmatch(value) is not None


@dataclass(frozen=True, slots=True)
class ResolvedCase:
    """Internal authorized case handle; ``analysis_handle`` is never serialized."""

    case_id: str
    kind: CaseKind
    analysis_handle: object

    def __post_init__(self) -> None:
        if not _is_opaque_id(self.case_id):
            raise ValueError("case_id must be an opaque identifier")
        if not isinstance(self.kind, CaseKind):
            raise ValueError("kind must identify an authorized or synthetic case")


@dataclass(frozen=True, slots=True)
class VerifiedRelease:
    """Minimal safe release metadata returned by a verification boundary."""

    release_id: str
    artifact_sha256: str
    verified: bool = True
    locked: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.verified, bool) or not isinstance(self.locked, bool):
            raise ValueError("verified and locked must be booleans")
        if not _is_opaque_id(self.release_id):
            raise ValueError("release_id must be an opaque identifier")
        if SHA256_PATTERN.fullmatch(self.artifact_sha256) is None:
            raise ValueError("artifact_sha256 must be 64 lowercase hexadecimal characters")


@dataclass(frozen=True, slots=True)
class AnalysisOutcome:
    """Validated engine result with a structural no-leakage invariant."""

    decision: TrustDecision
    reason_codes: tuple[ReasonCode, ...]
    labels: tuple[str, ...] | None = None
    probabilities: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.decision, TrustDecision):
            raise ValueError("decision must use the closed trust-decision vocabulary")
        if self.decision is not TrustDecision.PREDICTION_ALLOWED and (
            self.labels is not None or self.probabilities is not None
        ):
            raise ValueError("non-allowed outcomes cannot contain prediction results")
        # A frozen dataclass does not freeze caller-owned lists. Snapshot first
        # within the existing size/disclosure limits, then validate the snapshot.
        for name, limit in (("reason_codes", 16), ("labels", 71), ("probabilities", 71)):
            value = getattr(self, name)
            if value is None and name != "reason_codes":
                continue
            if not isinstance(value, (tuple, list)):
                raise ValueError(f"{name} must be a tuple or list")
            if len(value) > limit:
                raise ValueError(f"{name} exceeds its {limit}-item bound")
            object.__setattr__(self, name, tuple(value))
        if not self.reason_codes or len(self.reason_codes) > 16:
            raise ValueError("one to sixteen reason codes are required")
        if any(not isinstance(reason, ReasonCode) for reason in self.reason_codes):
            raise ValueError("reason codes must use the closed reason vocabulary")
        if len(set(self.reason_codes)) != len(self.reason_codes):
            raise ValueError("reason codes must be unique")

        if self.decision is not TrustDecision.PREDICTION_ALLOWED:
            return

        if not self.labels or not self.probabilities:
            raise ValueError("allowed outcomes require labels and probabilities")
        if len(self.labels) != len(self.probabilities) or len(self.labels) > 71:
            raise ValueError("labels and probabilities must have matching bounded lengths")
        if len(set(self.labels)) != len(self.labels):
            raise ValueError("labels must be unique")
        if any(LABEL_PATTERN.fullmatch(label) is None for label in self.labels):
            raise ValueError("labels must use the bounded research-label vocabulary format")
        if any(
            isinstance(probability, bool)
            or not isinstance(probability, (float, int))
            or not math.isfinite(float(probability))
            or not 0.0 <= float(probability) <= 1.0
            for probability in self.probabilities
        ):
            raise ValueError("probabilities must be finite values between zero and one")


class CaseResolver(Protocol):
    """Resolves only pre-authorized or explicitly synthetic opaque cases."""

    def is_ready(self) -> bool: ...

    def resolve_case(self, case_id: str) -> ResolvedCase: ...


class VerifiedReleaseProvider(Protocol):
    """Provides immutable release records that passed an external verification gate."""

    def is_ready(self) -> bool: ...

    def get_active_release(self) -> VerifiedRelease: ...

    def get_release(self, release_id: str) -> VerifiedRelease: ...


class AnalysisEngine(Protocol):
    """Runs contract validation and gated research inference."""

    def is_ready(self) -> bool: ...

    def is_ready_for_release(self, release: VerifiedRelease) -> bool: ...

    def validate_case(self, case: ResolvedCase, release: VerifiedRelease) -> AnalysisOutcome: ...

    def infer(self, case: ResolvedCase, release: VerifiedRelease) -> AnalysisOutcome: ...


class ResourceNotFoundError(LookupError):
    """Expected private-backend miss, translated to a generic public 404."""


class DependencyUnavailableError(RuntimeError):
    """Expected dependency outage, translated to a fail-closed response."""
