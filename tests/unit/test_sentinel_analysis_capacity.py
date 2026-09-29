from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest
from fastapi.testclient import TestClient

from ecg_trust.service.sentinel_service import SentinelServiceConfig, create_sentinel_app
from tests.unit.test_sentinel_service import (
    FakeAnalysisEngine,
    FakeCaseResolver,
    FakeReleaseProvider,
)


class BlockingEngine(FakeAnalysisEngine):
    def __init__(self):
        super().__init__()
        self.started = Event()
        self.release = Event()

    def _outcome(self, case):
        if case.case_id == "blocked":
            self.started.set()
            assert self.release.wait(5)
        return super()._outcome(case)


def client_for(engine, limit=1):
    return TestClient(
        create_sentinel_app(
            case_resolver=FakeCaseResolver(),
            release_provider=FakeReleaseProvider(),
            analysis_engine=engine,
            config=SentinelServiceConfig(max_concurrent_analyses=limit),
        )
    )


def infer(client, case, key):
    return client.post(
        "/api/v1/inferences",
        json={"case_id": case, "release_id": "release-v1"},
        headers={"Idempotency-Key": key},
    )


@pytest.mark.parametrize("limit", [0, -1, 65, True, False, 1.5, "1"])
def test_capacity_configuration_rejects_invalid_limits(limit):
    with pytest.raises(ValueError, match="max_concurrent_analyses"):
        SentinelServiceConfig(max_concurrent_analyses=limit)


@pytest.mark.parametrize("first_endpoint", ["/api/v1/inferences", "/api/v1/cases:validate"])
def test_saturation_fails_promptly_and_releases_capacity(first_endpoint):
    engine = BlockingEngine()
    with client_for(engine) as client, ThreadPoolExecutor(2) as pool:
        first = pool.submit(
            client.post,
            first_endpoint,
            json={"case_id": "blocked", "release_id": "release-v1"},
            headers={"Idempotency-Key": "blocked-key"},
        )
        try:
            assert engine.started.wait(2)
            busy = pool.submit(infer, client, "ordinary", "retry-key").result(timeout=2)
            assert busy.status_code == 503
            assert busy.headers["Retry-After"] == "1"
            assert busy.headers["Cache-Control"] == "no-store"
            assert busy.headers["Idempotency-Replayed"] == "false"
            assert busy.json()["reason_codes"] == ["SERVICE_BUSY"]
            assert busy.json()["decision"] == "ABSTAIN"
            assert "probabilities" not in busy.json()
            validation = client.post(
                "/api/v1/cases:validate",
                json={"case_id": "ordinary", "release_id": "release-v1"},
            )
            assert validation.status_code == 503
            assert validation.headers["Retry-After"] == "1"
            assert client.get("/api/v1/healthz").status_code == 200
        finally:
            engine.release.set()
        assert first.result(timeout=2).status_code == 200
        recovered = infer(client, "ordinary", "retry-key")
        assert recovered.status_code == 200
        assert recovered.headers["Idempotency-Replayed"] == "false"
        assert "Retry-After" not in recovered.headers


def test_cached_replay_does_not_consume_analysis_capacity():
    engine = BlockingEngine()
    with client_for(engine) as client, ThreadPoolExecutor(2) as pool:
        cached = infer(client, "ordinary", "cached-key")
        first = pool.submit(infer, client, "blocked", "blocked-key")
        try:
            assert engine.started.wait(2)
            replay = pool.submit(infer, client, "ordinary", "cached-key").result(timeout=2)
            assert replay.status_code == 200
            assert replay.json() == cached.json()
            assert replay.headers["Idempotency-Replayed"] == "true"
        finally:
            engine.release.set()
        first.result(timeout=2)
    assert engine.infer_calls == 2


@pytest.mark.parametrize("limit", [None, 2])
def test_unlimited_default_or_free_second_slot_allows_independent_requests(limit):
    engine = BlockingEngine()
    with client_for(engine, limit) as client, ThreadPoolExecutor(2) as pool:
        first = pool.submit(infer, client, "blocked", "blocked-key")
        try:
            assert engine.started.wait(2)
            second = pool.submit(infer, client, "ordinary", "second-key").result(timeout=2)
            assert second.status_code == 200
        finally:
            engine.release.set()
        first.result(timeout=2)


@pytest.mark.parametrize("failure", ["backend-failure", "resolver-boom", "malformed-backend"])
def test_failed_backend_paths_release_the_slot(failure):
    with client_for(FakeAnalysisEngine()) as client:
        assert infer(client, failure, "failure-key").status_code == 503
        assert infer(client, "ordinary", "healthy-key").status_code == 200
