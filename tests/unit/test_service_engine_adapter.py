from __future__ import annotations

from datetime import UTC, datetime
from typing import cast

import numpy as np
import pytest

from ecg_trust.conformal import BinaryPredictionSets
from ecg_trust.constants import SUPERCLASSES
from ecg_trust.contract_adapters import conformal_prediction_sets_to_contracts
from ecg_trust.contracts import (
    ARTIFACT_REFERENCE_SCHEMA_VERSION,
    QUALITY_REPORT_SCHEMA_VERSION,
    ArtifactReference,
    QualityReport,
    TrustDecision,
)
from ecg_trust.quality.signal_quality import SignalMetadata
from ecg_trust.sentinel_engine import SentinelCaseResult
from ecg_trust.service.engine_adapter import (
    SentinelCasePayload,
    SentinelServiceAnalysisEngine,
)
from ecg_trust.service.sentinel_service import (
    CaseKind,
    ReasonCode,
    ResolvedCase,
    VerifiedRelease,
)
from ecg_trust.trust_policy import TrustPolicyResult, TrustReasonCode

NOW = datetime(2026, 8, 24, 12, 0, tzinfo=UTC)
RELEASE = VerifiedRelease(
    release_id="release-vnext",
    artifact_sha256="a" * 64,
    verified=True,
    locked=True,
)


def _artifact() -> ArtifactReference:
    return ArtifactReference(
        schema_version=ARTIFACT_REFERENCE_SCHEMA_VERSION,
        artifact_id="conformal-artifact",
        file_sha256=f"sha256:{'b' * 64}",
        size_bytes=128,
        media_type="application/json",
        sensitive=False,
    )


def _result(decision: TrustDecision) -> SentinelCaseResult:
    allowed = decision is TrustDecision.PREDICTION_ALLOWED
    reason = (
        TrustReasonCode.ALL_TRUST_GATES_PASSED
        if allowed
        else TrustReasonCode.CONFORMAL_SET_UNCERTAIN
    )
    policy = TrustPolicyResult(
        policy_version="trust-policy-v1",
        decision=decision,
        reason_codes=(reason,),
        uncertain_labels=() if allowed else ("MI",),
    )
    quality = QualityReport(
        schema_version=QUALITY_REPORT_SCHEMA_VERSION,
        evaluated_at=NOW,
        passed=True,
        decision=TrustDecision.PREDICTION_ALLOWED,
        findings=(),
    )
    probabilities = (0.9, 0.1, 0.1, 0.1, 0.1)
    if not allowed:
        return SentinelCaseResult(
            release_id="release-vnext",
            decision=decision,
            policy=policy,
            quality=quality,
            distribution=None,
            label_prediction_sets=None,
            calibrated_probabilities=None,
        )
    sets = BinaryPredictionSets.from_masks(
        label_names=SUPERCLASSES,
        include_not_supported=[[False, True, True, True, True]],
        include_supported=[[True, False, False, False, False]],
    )
    contracts = conformal_prediction_sets_to_contracts(
        sets,
        probabilities,
        calibration_artifact=_artifact(),
    )
    return SentinelCaseResult(
        release_id="release-vnext",
        decision=decision,
        policy=policy,
        quality=quality,
        distribution=None,
        label_prediction_sets=contracts,
        calibrated_probabilities=probabilities,
    )


class FakeCore:
    release_id = "release-vnext"
    bound_manifest_sha256 = "a" * 64

    def __init__(self, result: SentinelCaseResult, *, ready: bool = True) -> None:
        self.result = result
        self.ready = ready
        self.calls = 0

    def is_ready(self) -> bool:
        return self.ready

    def analyze(self, **kwargs: object) -> SentinelCaseResult:
        self.calls += 1
        assert kwargs["signal_id"] == "case-001"
        assert kwargs["evaluated_at"] == NOW
        assert kwargs["release_integrity_verified"] is True
        return self.result


def _case(handle: object | None = None) -> ResolvedCase:
    payload = handle or SentinelCasePayload(
        signal_mv=np.zeros((12, 1_000), dtype=np.float64),
        metadata=SignalMetadata.canonical(),
    )
    return ResolvedCase(
        case_id="case-001",
        kind=CaseKind.SYNTHETIC,
        analysis_handle=payload,
    )


def test_allowed_core_result_maps_to_service_labels_and_probabilities() -> None:
    core = FakeCore(_result(TrustDecision.PREDICTION_ALLOWED))
    adapter = SentinelServiceAnalysisEngine(core, clock=lambda: NOW)

    validation = adapter.validate_case(_case(), RELEASE)
    inference = adapter.infer(_case(), RELEASE)

    for outcome in (validation, inference):
        assert outcome.decision is TrustDecision.PREDICTION_ALLOWED
        assert outcome.reason_codes == (ReasonCode.ALL_TRUST_GATES_PASSED,)
        assert outcome.labels == SUPERCLASSES
        assert outcome.probabilities == (0.9, 0.1, 0.1, 0.1, 0.1)
    assert core.calls == 2


def test_blocked_core_result_never_maps_label_outputs() -> None:
    adapter = SentinelServiceAnalysisEngine(
        FakeCore(_result(TrustDecision.ABSTAIN)), clock=lambda: NOW
    )

    outcome = adapter.infer(_case(), RELEASE)

    assert outcome.decision is TrustDecision.ABSTAIN
    assert outcome.reason_codes == (ReasonCode.CONFORMAL_SET_UNCERTAIN,)
    assert outcome.labels is None
    assert outcome.probabilities is None


def test_release_and_private_handle_mismatch_fail_closed() -> None:
    core = FakeCore(_result(TrustDecision.ABSTAIN))
    adapter = SentinelServiceAnalysisEngine(core, clock=lambda: NOW)
    wrong_release = VerifiedRelease(
        release_id="other-release",
        artifact_sha256="c" * 64,
        verified=True,
        locked=True,
    )
    wrong_digest = VerifiedRelease(
        release_id="release-vnext",
        artifact_sha256="c" * 64,
        verified=True,
        locked=True,
    )

    with pytest.raises(RuntimeError, match="does not match"):
        adapter.infer(_case(), wrong_release)
    with pytest.raises(RuntimeError, match="digest does not match"):
        adapter.infer(_case(), wrong_digest)
    assert core.calls == 0
    with pytest.raises(TypeError, match="incompatible"):
        adapter.infer(_case(cast(object, "C:\\private\\case.npy")), RELEASE)


def test_readiness_and_timezone_aware_clock_are_enforced() -> None:
    unavailable = FakeCore(_result(TrustDecision.ABSTAIN), ready=False)
    assert not SentinelServiceAnalysisEngine(unavailable).is_ready()

    core = FakeCore(_result(TrustDecision.ABSTAIN))
    adapter = SentinelServiceAnalysisEngine(core, clock=lambda: NOW.replace(tzinfo=None))
    with pytest.raises(RuntimeError, match="timezone-aware"):
        adapter.infer(_case(), RELEASE)


def test_release_specific_readiness_requires_exact_bound_identity() -> None:
    adapter = SentinelServiceAnalysisEngine(FakeCore(_result(TrustDecision.ABSTAIN)))
    wrong_digest = VerifiedRelease(
        release_id="release-vnext",
        artifact_sha256="b" * 64,
        verified=True,
        locked=True,
    )
    wrong_release = VerifiedRelease(
        release_id="other-release",
        artifact_sha256="a" * 64,
        verified=True,
        locked=True,
    )

    assert adapter.is_ready_for_release(RELEASE)
    assert not adapter.is_ready_for_release(wrong_digest)
    assert not adapter.is_ready_for_release(wrong_release)
