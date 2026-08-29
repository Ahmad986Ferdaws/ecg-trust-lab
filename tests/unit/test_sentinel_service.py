from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import Response

from ecg_trust.service.sentinel_service import (
    AnalysisEngine,
    AnalysisOutcome,
    CaseKind,
    CaseResolver,
    ReasonCode,
    ResolvedCase,
    ResourceNotFoundError,
    SentinelServiceConfig,
    TrustDecision,
    VerifiedRelease,
    VerifiedReleaseProvider,
    create_sentinel_app,
)

RELEASE_ID = "release-v1"
ARTIFACT_SHA256 = "a" * 64


class FakeCaseResolver:
    def __init__(self, *, ready: bool = True, readiness_raises: bool = False) -> None:
        self.ready = ready
        self.readiness_raises = readiness_raises
        self.resolve_calls = 0

    def is_ready(self) -> bool:
        if self.readiness_raises:
            raise RuntimeError("C:\\private\\resolver.env token=do-not-leak")
        return self.ready

    def resolve_case(self, case_id: str) -> ResolvedCase:
        self.resolve_calls += 1
        if case_id == "missing-case":
            raise ResourceNotFoundError("C:\\private\\missing.npy")
        if case_id == "resolver-boom":
            raise RuntimeError("C:\\private\\cases.db password=do-not-leak")
        kind = CaseKind.SYNTHETIC if case_id.startswith("synthetic") else CaseKind.AUTHORIZED
        return ResolvedCase(case_id=case_id, kind=kind, analysis_handle=case_id)


class FakeReleaseProvider:
    def __init__(self, *, ready: bool = True, readiness_raises: bool = False) -> None:
        self.ready = ready
        self.readiness_raises = readiness_raises

    def is_ready(self) -> bool:
        if self.readiness_raises:
            raise RuntimeError("C:\\private\\registry.json secret=do-not-leak")
        return self.ready

    def get_release(self, release_id: str) -> VerifiedRelease:
        if release_id == "missing-release":
            raise ResourceNotFoundError("s3://private-bucket/model")
        if release_id == "release-boom":
            raise RuntimeError("C:\\private\\model.pt credential=do-not-leak")
        if release_id == "release-unverified":
            return VerifiedRelease(
                release_id=release_id,
                artifact_sha256="b" * 64,
                verified=False,
                locked=True,
            )
        return VerifiedRelease(
            release_id=release_id,
            artifact_sha256=ARTIFACT_SHA256,
            verified=True,
            locked=True,
        )

    def get_active_release(self) -> VerifiedRelease:
        return self.get_release(RELEASE_ID)


class FakeAnalysisEngine:
    def __init__(
        self,
        *,
        ready: bool = True,
        readiness_raises: bool = False,
        expected_artifact_sha256: str = ARTIFACT_SHA256,
    ) -> None:
        self.ready = ready
        self.readiness_raises = readiness_raises
        self.expected_artifact_sha256 = expected_artifact_sha256
        self.infer_calls = 0
        self.validate_calls = 0

    def is_ready(self) -> bool:
        if self.readiness_raises:
            raise RuntimeError("C:\\private\\engine.cfg api_key=do-not-leak")
        return self.ready

    def is_ready_for_release(self, release: VerifiedRelease) -> bool:
        return self.is_ready() and release.artifact_sha256 == self.expected_artifact_sha256

    def validate_case(
        self,
        case: ResolvedCase,
        release: VerifiedRelease,
    ) -> AnalysisOutcome:
        del release
        self.validate_calls += 1
        return self._outcome(case)

    def infer(self, case: ResolvedCase, release: VerifiedRelease) -> AnalysisOutcome:
        del release
        self.infer_calls += 1
        return self._outcome(case)

    @staticmethod
    def _outcome(case: ResolvedCase) -> AnalysisOutcome:
        case_id = cast(str, case.analysis_handle)
        if case_id == "backend-failure":
            raise RuntimeError("C:\\private\\waveform.npy bearer=do-not-leak")
        if case_id == "malformed-backend":
            return cast(AnalysisOutcome, {"raw_waveform": [1.0], "path": "C:\\private"})
        if case_id == "invalid":
            return AnalysisOutcome(
                decision=TrustDecision.INVALID_INPUT,
                reason_codes=(ReasonCode.INPUT_CONTRACT_INVALID,),
            )
        if case_id == "reacquire":
            return AnalysisOutcome(
                decision=TrustDecision.REACQUIRE,
                reason_codes=(ReasonCode.SIGNAL_REACQUISITION_REQUIRED,),
            )
        if case_id == "unsupported":
            return AnalysisOutcome(
                decision=TrustDecision.UNSUPPORTED_INPUT,
                reason_codes=(ReasonCode.OUTSIDE_VALIDATED_DISTRIBUTION,),
            )
        if case_id == "abstain":
            return AnalysisOutcome(
                decision=TrustDecision.ABSTAIN,
                reason_codes=(ReasonCode.CONFIDENCE_GATE_ABSTAINED,),
            )
        return AnalysisOutcome(
            decision=TrustDecision.PREDICTION_ALLOWED,
            reason_codes=(ReasonCode.ALL_TRUST_GATES_PASSED,),
            labels=("NORM", "MI"),
            probabilities=(0.91, 0.08),
        )


def _app(
    *,
    resolver: CaseResolver | None = None,
    provider: VerifiedReleaseProvider | None = None,
    engine: AnalysisEngine | None = None,
    config: SentinelServiceConfig | None = None,
) -> FastAPI:
    return create_sentinel_app(
        case_resolver=resolver or FakeCaseResolver(),
        release_provider=provider or FakeReleaseProvider(),
        analysis_engine=engine or FakeAnalysisEngine(),
        config=config,
    )


def _body(response: Response) -> dict[str, object]:
    return cast(dict[str, object], response.json())


def _assert_boundary(body: Mapping[str, object]) -> None:
    assert body["api_version"] == "v1"
    assert body["research_only"] is True
    assert "Not for diagnosis" in cast(str, body["boundary"])


def _assert_security_headers(response: Response) -> None:
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["pragma"] == "no-cache"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["referrer-policy"] == "no-referrer"
    assert response.headers["content-security-policy"] == (
        "default-src 'none'; frame-ancestors 'none'"
    )
    assert response.headers["permissions-policy"] == "camera=(), microphone=(), geolocation=()"


def test_health_is_live_without_backends_and_sets_security_boundary() -> None:
    with TestClient(create_sentinel_app()) as client:
        response = client.get("/api/v1/healthz")

    assert response.status_code == 200
    body = _body(response)
    assert body["status"] == "alive"
    _assert_boundary(body)
    _assert_security_headers(response)


def test_readiness_requires_every_injected_dependency() -> None:
    with TestClient(create_sentinel_app()) as client:
        unavailable = client.get("/api/v1/readyz")
    assert unavailable.status_code == 503
    assert _body(unavailable)["status"] == "not_ready"
    assert _body(unavailable)["reason_codes"] == [ReasonCode.BACKEND_UNAVAILABLE]

    with TestClient(_app()) as client:
        ready = client.get("/api/v1/readyz")
    assert ready.status_code == 200
    assert _body(ready)["status"] == "ready"
    assert _body(ready)["reason_codes"] == []


def test_readiness_fails_closed_when_release_and_engine_digests_differ() -> None:
    engine = FakeAnalysisEngine(expected_artifact_sha256="b" * 64)
    with TestClient(_app(engine=engine)) as client:
        response = client.get("/api/v1/readyz")

    assert response.status_code == 503
    assert _body(response)["status"] == "not_ready"
    assert _body(response)["reason_codes"] == [ReasonCode.BACKEND_UNAVAILABLE]


@pytest.mark.parametrize(
    "dependency",
    [
        FakeCaseResolver(ready=False),
        FakeReleaseProvider(readiness_raises=True),
        FakeAnalysisEngine(ready=False),
    ],
)
def test_readiness_fails_closed_for_false_or_raising_dependency(dependency: object) -> None:
    resolver: CaseResolver = (
        cast(CaseResolver, dependency)
        if isinstance(dependency, FakeCaseResolver)
        else FakeCaseResolver()
    )
    provider: VerifiedReleaseProvider = (
        cast(VerifiedReleaseProvider, dependency)
        if isinstance(dependency, FakeReleaseProvider)
        else FakeReleaseProvider()
    )
    engine: AnalysisEngine = (
        cast(AnalysisEngine, dependency)
        if isinstance(dependency, FakeAnalysisEngine)
        else FakeAnalysisEngine()
    )
    with TestClient(_app(resolver=resolver, provider=provider, engine=engine)) as client:
        response = client.get("/api/v1/readyz")

    assert response.status_code == 503
    serialized = response.text
    assert "private" not in serialized
    assert "do-not-leak" not in serialized


def test_release_endpoint_exposes_only_safe_verified_metadata() -> None:
    with TestClient(_app()) as client:
        response = client.get(f"/api/v1/releases/{RELEASE_ID}")

    assert response.status_code == 200
    body = _body(response)
    assert body == {
        "api_version": "v1",
        "service_version": "sentinel-v1",
        "research_only": True,
        "boundary": "Research use only. Not for diagnosis, treatment, or clinical decision-making.",
        "release_id": RELEASE_ID,
        "artifact_sha256": ARTIFACT_SHA256,
        "verified": True,
        "locked": True,
    }
    assert "path" not in response.text.lower()
    assert "waveform" not in response.text.lower()


@pytest.mark.parametrize(
    ("case_id", "decision", "reason_code", "results_allowed"),
    [
        (
            "invalid",
            TrustDecision.INVALID_INPUT,
            ReasonCode.INPUT_CONTRACT_INVALID,
            False,
        ),
        (
            "reacquire",
            TrustDecision.REACQUIRE,
            ReasonCode.SIGNAL_REACQUISITION_REQUIRED,
            False,
        ),
        (
            "unsupported",
            TrustDecision.UNSUPPORTED_INPUT,
            ReasonCode.OUTSIDE_VALIDATED_DISTRIBUTION,
            False,
        ),
        (
            "abstain",
            TrustDecision.ABSTAIN,
            ReasonCode.CONFIDENCE_GATE_ABSTAINED,
            False,
        ),
        (
            "allowed",
            TrustDecision.PREDICTION_ALLOWED,
            ReasonCode.ALL_TRUST_GATES_PASSED,
            True,
        ),
    ],
)
def test_inference_supports_five_states_and_discloses_results_only_when_allowed(
    case_id: str,
    decision: TrustDecision,
    reason_code: ReasonCode,
    results_allowed: bool,
) -> None:
    with TestClient(_app()) as client:
        response = client.post(
            "/api/v1/inferences",
            json={"case_id": case_id, "release_id": RELEASE_ID},
            headers={"Idempotency-Key": f"request-{case_id}"},
        )

    assert response.status_code == 200
    body = _body(response)
    assert body["decision"] == decision
    assert body["reason_codes"] == [reason_code]
    assert ("labels" in body) is results_allowed
    assert ("probabilities" in body) is results_allowed
    if results_allowed:
        assert body["labels"] == ["NORM", "MI"]
        assert body["probabilities"] == [0.91, 0.08]
    _assert_boundary(body)


def test_case_validation_never_returns_prediction_results() -> None:
    engine = FakeAnalysisEngine()
    with TestClient(_app(engine=engine)) as client:
        response = client.post(
            "/api/v1/cases:validate",
            json={"case_id": "allowed", "release_id": RELEASE_ID},
        )

    assert response.status_code == 200
    body = _body(response)
    assert body["decision"] == TrustDecision.PREDICTION_ALLOWED
    assert "labels" not in body
    assert "probabilities" not in body
    assert engine.validate_calls == 1
    assert engine.infer_calls == 0


def test_inference_replay_is_identical_and_executes_engine_once() -> None:
    engine = FakeAnalysisEngine()
    request = {"case_id": "allowed", "release_id": RELEASE_ID}
    headers = {"Idempotency-Key": "stable-request-001"}
    with TestClient(_app(engine=engine)) as client:
        first = client.post("/api/v1/inferences", json=request, headers=headers)
        replay = client.post("/api/v1/inferences", json=request, headers=headers)

    assert first.status_code == replay.status_code == 200
    assert first.json() == replay.json()
    assert first.headers["idempotency-replayed"] == "false"
    assert replay.headers["idempotency-replayed"] == "true"
    assert engine.infer_calls == 1


def test_idempotency_key_reuse_with_different_request_is_conflict() -> None:
    engine = FakeAnalysisEngine()
    headers = {"Idempotency-Key": "stable-request-002"}
    with TestClient(_app(engine=engine)) as client:
        first = client.post(
            "/api/v1/inferences",
            json={"case_id": "allowed", "release_id": RELEASE_ID},
            headers=headers,
        )
        conflict = client.post(
            "/api/v1/inferences",
            json={"case_id": "abstain", "release_id": RELEASE_ID},
            headers=headers,
        )

    assert first.status_code == 200
    assert conflict.status_code == 409
    assert _body(conflict)["error"] == {
        "code": "idempotency_key_conflict",
        "message": "The idempotency key was already used for another request.",
    }
    assert engine.infer_calls == 1
    assert "allowed" not in conflict.text
    assert "abstain" not in conflict.text


@pytest.mark.parametrize("case_id", ["backend-failure", "malformed-backend", "resolver-boom"])
def test_backend_failures_abstain_without_sensitive_result_leakage(case_id: str) -> None:
    engine = FakeAnalysisEngine()
    with TestClient(_app(engine=engine)) as client:
        response = client.post(
            "/api/v1/inferences",
            json={"case_id": case_id, "release_id": RELEASE_ID},
            headers={"Idempotency-Key": f"failure-{case_id}"},
        )

    assert response.status_code == 503
    body = _body(response)
    assert body["decision"] == TrustDecision.ABSTAIN
    assert body["reason_codes"] == [ReasonCode.BACKEND_UNAVAILABLE]
    assert "labels" not in body
    assert "probabilities" not in body
    assert "raw_waveform" not in response.text
    assert "private" not in response.text.lower()
    assert "do-not-leak" not in response.text


def test_unverified_release_abstains_and_omits_results() -> None:
    with TestClient(_app()) as client:
        response = client.post(
            "/api/v1/inferences",
            json={"case_id": "allowed", "release_id": "release-unverified"},
            headers={"Idempotency-Key": "unverified-release-request"},
        )

    assert response.status_code == 503
    body = _body(response)
    assert body["decision"] == TrustDecision.ABSTAIN
    assert body["reason_codes"] == [ReasonCode.RELEASE_NOT_READY]
    assert "labels" not in body
    assert "probabilities" not in body


def test_failure_is_not_cached_as_an_idempotent_success() -> None:
    engine = FakeAnalysisEngine()
    request = {"case_id": "backend-failure", "release_id": RELEASE_ID}
    headers = {"Idempotency-Key": "retryable-failure-request"}
    with TestClient(_app(engine=engine)) as client:
        first = client.post("/api/v1/inferences", json=request, headers=headers)
        second = client.post("/api/v1/inferences", json=request, headers=headers)

    assert first.status_code == second.status_code == 503
    assert first.headers["idempotency-replayed"] == "false"
    assert second.headers["idempotency-replayed"] == "false"
    assert engine.infer_calls == 2


def test_not_found_is_generic_and_does_not_reveal_backend_details() -> None:
    with TestClient(_app()) as client:
        response = client.post(
            "/api/v1/cases:validate",
            json={"case_id": "missing-case", "release_id": RELEASE_ID},
        )

    assert response.status_code == 404
    body = _body(response)
    assert body["error"] == {
        "code": "resource_not_found",
        "message": "The requested resource was not found.",
    }
    assert "missing-case" not in response.text
    assert "private" not in response.text.lower()


@pytest.mark.parametrize(
    "payload",
    [
        {"case_id": 123, "release_id": RELEASE_ID},
        {"case_id": "../private", "release_id": RELEASE_ID},
        {"case_id": "allowed", "release_id": RELEASE_ID, "raw_waveform": [0.0, 1.0]},
        {"case_id": "allowed", "release_id": RELEASE_ID, "local_path": "C:\\secret"},
    ],
)
def test_request_contract_is_strict_and_validation_errors_are_sanitized(
    payload: dict[str, object],
) -> None:
    with TestClient(_app()) as client:
        response = client.post(
            "/api/v1/inferences",
            json=payload,
            headers={"Idempotency-Key": "contract-check-001"},
        )

    assert response.status_code == 422
    body = _body(response)
    assert body["error"] == {
        "code": "request_validation_failed",
        "message": "The request does not match the API contract.",
    }
    assert "raw_waveform" not in response.text
    assert "local_path" not in response.text
    assert "private" not in response.text.lower()
    assert "secret" not in response.text.lower()
    _assert_security_headers(response)


@pytest.mark.parametrize("headers", [{}, {"Idempotency-Key": "C:\\secret"}])
def test_idempotency_key_is_required_and_bounded(headers: dict[str, str]) -> None:
    with TestClient(_app()) as client:
        response = client.post(
            "/api/v1/inferences",
            json={"case_id": "allowed", "release_id": RELEASE_ID},
            headers=headers,
        )

    assert response.status_code == 422
    assert _body(response)["error"] == {
        "code": "request_validation_failed",
        "message": "The request does not match the API contract.",
    }
    assert "secret" not in response.text.lower()


def test_release_path_identifier_is_bounded_and_sanitized() -> None:
    with TestClient(_app()) as client:
        response = client.get("/api/v1/releases/..private")

    assert response.status_code == 422
    assert "..private" not in response.text
    assert _body(response)["error"] == {
        "code": "request_validation_failed",
        "message": "The request does not match the API contract.",
    }


def test_analysis_outcome_rejects_result_leakage_and_missing_allowed_results() -> None:
    with pytest.raises(ValueError, match="cannot contain prediction results"):
        AnalysisOutcome(
            decision=TrustDecision.ABSTAIN,
            reason_codes=(ReasonCode.CONFIDENCE_GATE_ABSTAINED,),
            labels=("NORM",),
            probabilities=(0.5,),
        )

    with pytest.raises(ValueError, match="require labels and probabilities"):
        AnalysisOutcome(
            decision=TrustDecision.PREDICTION_ALLOWED,
            reason_codes=(ReasonCode.ALL_TRUST_GATES_PASSED,),
        )


def test_idempotency_store_is_bounded() -> None:
    engine = FakeAnalysisEngine()
    app = _app(
        engine=engine,
        config=SentinelServiceConfig(max_idempotency_entries=1),
    )
    with TestClient(app) as client:
        client.post(
            "/api/v1/inferences",
            json={"case_id": "allowed", "release_id": RELEASE_ID},
            headers={"Idempotency-Key": "bounded-cache-001"},
        )
        client.post(
            "/api/v1/inferences",
            json={"case_id": "synthetic-case", "release_id": RELEASE_ID},
            headers={"Idempotency-Key": "bounded-cache-002"},
        )
        replay_after_eviction = client.post(
            "/api/v1/inferences",
            json={"case_id": "allowed", "release_id": RELEASE_ID},
            headers={"Idempotency-Key": "bounded-cache-001"},
        )

    assert replay_after_eviction.status_code == 200
    assert replay_after_eviction.headers["idempotency-replayed"] == "false"
    assert engine.infer_calls == 3
