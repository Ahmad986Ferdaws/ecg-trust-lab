from __future__ import annotations

import anyio
import anyio.to_thread
import pytest
from httpx import ASGITransport, AsyncClient

from ecg_trust.service.sentinel_service import create_sentinel_app
from tests.unit.test_sentinel_analysis_capacity import BlockingEngine
from tests.unit.test_sentinel_service import (
    FakeCaseResolver,
    FakeReleaseProvider,
    _assert_boundary,
    _assert_security_headers,
)


@pytest.mark.parametrize("endpoint", ["/api/v1/inferences", "/api/v1/cases:validate"])
def test_liveness_responds_while_analysis_occupies_the_worker_pool(endpoint: str) -> None:
    async def scenario() -> None:
        limiter = anyio.to_thread.current_default_thread_limiter()
        previous_tokens = limiter.total_tokens
        limiter.total_tokens = 1
        engine = BlockingEngine()
        app = create_sentinel_app(
            case_resolver=FakeCaseResolver(),
            release_provider=FakeReleaseProvider(),
            analysis_engine=engine,
        )
        completed = []
        try:
            async with AsyncClient(transport=ASGITransport(app), base_url="http://test") as client:

                async def blocked_analysis() -> None:
                    response = await client.post(
                        endpoint,
                        json={"case_id": "blocked", "release_id": "release-v1"},
                        headers={"Idempotency-Key": "blocked-key"},
                    )
                    completed.append(response.status_code)

                async with anyio.create_task_group() as group:
                    group.start_soon(blocked_analysis)
                    try:
                        with anyio.fail_after(2):
                            while not engine.started.is_set():
                                await anyio.sleep(0.001)
                        assert limiter.borrowed_tokens == 1
                        with anyio.fail_after(2):
                            health = await client.get("/api/v1/healthz")
                        assert health.status_code == 200
                        _assert_boundary(health.json())
                        _assert_security_headers(health)
                        assert not engine.release.is_set()
                    finally:
                        engine.release.set()
            assert completed == [200]
        finally:
            engine.release.set()
            limiter.total_tokens = previous_tokens

    anyio.run(scenario)
