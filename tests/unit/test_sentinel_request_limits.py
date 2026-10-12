from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient
from starlette.types import ASGIApp, Message, Scope

from ecg_trust.service.sentinel_service import SentinelServiceConfig, create_sentinel_app
from tests.unit.test_sentinel_service import (
    FakeAnalysisEngine,
    FakeCaseResolver,
    FakeReleaseProvider,
    _assert_boundary,
    _assert_security_headers,
)

PAYLOAD = b'{"case_id":"ordinary","release_id":"release-v1"}'


@pytest.mark.parametrize("endpoint", ["/api/v1/cases:validate", "/api/v1/inferences"])
def test_oversized_json_never_reaches_a_case_or_engine(endpoint: str) -> None:
    resolver, engine = FakeCaseResolver(), FakeAnalysisEngine()
    app = create_sentinel_app(
        case_resolver=resolver,
        release_provider=FakeReleaseProvider(),
        analysis_engine=engine,
    )
    with TestClient(app) as client:
        response = client.post(
            endpoint,
            content=PAYLOAD + b" " * 5000,
            headers={"Content-Type": "application/json", "Idempotency-Key": "retry-key"},
        )
        assert response.status_code == 413
        assert response.json()["error"]["code"] == "request_body_too_large"
        _assert_boundary(response.json())
        _assert_security_headers(response)
        assert resolver.resolve_calls == engine.validate_calls == engine.infer_calls == 0
        assert "ordinary" not in response.text
        retry = client.post(
            endpoint,
            content=PAYLOAD,
            headers={"Content-Type": "application/json", "Idempotency-Key": "retry-key"},
        )
        assert retry.status_code == 200
        assert resolver.resolve_calls == 1


@pytest.mark.parametrize("limit", [0, 255, 1_048_577, True, False, 4096.0, "4096", None])
def test_request_limit_requires_a_bounded_integer(limit: object) -> None:
    with pytest.raises(ValueError, match="max_request_body_bytes"):
        SentinelServiceConfig(max_request_body_bytes=limit)  # type: ignore[arg-type]


@pytest.mark.parametrize("size, expected", [(256, 200), (257, 413)])
def test_configured_limit_counts_the_complete_body(size: int, expected: int) -> None:
    app = create_sentinel_app(
        case_resolver=FakeCaseResolver(),
        release_provider=FakeReleaseProvider(),
        analysis_engine=FakeAnalysisEngine(),
        config=SentinelServiceConfig(max_request_body_bytes=256),
    )
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/cases:validate",
            content=PAYLOAD + b" " * (size - len(PAYLOAD)),
            headers={"Content-Type": "application/json"},
        )
    assert response.status_code == expected


async def _stream_request(
    app: ASGIApp,
    chunks: list[bytes],
    headers: list[tuple[bytes, bytes]],
    *,
    disconnect: bool = False,
) -> tuple[list[Message], int]:
    scope: Scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.4"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/api/v1/cases:validate",
        "raw_path": b"/api/v1/cases:validate",
        "query_string": b"",
        "root_path": "",
        "headers": [(b"content-type", b"application/json"), *headers],
        "client": ("127.0.0.1", 1234),
        "server": ("test", 80),
    }
    incoming = [
        {"type": "http.request", "body": chunk, "more_body": index < len(chunks) - 1 or disconnect}
        for index, chunk in enumerate(chunks)
    ]
    if disconnect:
        incoming.append({"type": "http.disconnect"})
    read = 0
    sent: list[Message] = []

    async def receive() -> Message:
        nonlocal read
        if read < len(incoming):
            message = incoming[read]
            read += 1
            return message
        # Keep the connection live until the response cancels its disconnect listener.
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    async def send(message: Message) -> None:
        sent.append(message)

    await asyncio.wait_for(app(scope, receive, send), timeout=2)
    return sent, read


@pytest.mark.parametrize(
    "headers", [[], [(b"content-length", b"1")], [(b"content-length", b"bad")]]
)
def test_chunked_body_limit_does_not_trust_length_headers(
    headers: list[tuple[bytes, bytes]],
) -> None:
    resolver, engine = FakeCaseResolver(), FakeAnalysisEngine()
    app = create_sentinel_app(
        case_resolver=resolver,
        release_provider=FakeReleaseProvider(),
        analysis_engine=engine,
        config=SentinelServiceConfig(max_request_body_bytes=256),
    )
    sent, read = asyncio.run(_stream_request(app, [PAYLOAD, b" " * 256, b"private-tail"], headers))
    assert sent[0]["status"] == 413
    assert read == 2
    assert resolver.resolve_calls == engine.validate_calls == 0
    assert b"private-tail" not in b"".join(message.get("body", b"") for message in sent)
    assert (b"cache-control", b"no-store") in sent[0]["headers"]


def test_declared_oversize_is_rejected_without_reading_the_stream() -> None:
    app = create_sentinel_app()
    sent, read = asyncio.run(_stream_request(app, [PAYLOAD], [(b"Content-Length", b"4097")]))
    assert sent[0]["status"] == 413
    assert read == 0


def test_chunked_valid_json_is_replayed_exactly_to_the_parser() -> None:
    resolver = FakeCaseResolver()
    app = create_sentinel_app(
        case_resolver=resolver,
        release_provider=FakeReleaseProvider(),
        analysis_engine=FakeAnalysisEngine(),
    )
    sent, read = asyncio.run(_stream_request(app, [PAYLOAD[:9], b"", PAYLOAD[9:]], []))
    assert sent[0]["status"] == 200
    assert read == 3
    assert resolver.resolve_calls == 1


def test_incomplete_disconnected_body_never_reaches_analysis() -> None:
    resolver, engine = FakeCaseResolver(), FakeAnalysisEngine()
    app = create_sentinel_app(
        case_resolver=resolver,
        release_provider=FakeReleaseProvider(),
        analysis_engine=engine,
    )
    sent, read = asyncio.run(_stream_request(app, [PAYLOAD[:9]], [], disconnect=True))
    assert read == 2
    assert resolver.resolve_calls == engine.validate_calls == 0
    assert not sent


def test_openapi_describes_the_shared_body_limit_error() -> None:
    schema = create_sentinel_app().openapi()
    for path in ("/api/v1/cases:validate", "/api/v1/inferences"):
        assert schema["paths"][path]["post"]["responses"]["413"]["content"]["application/json"][
            "schema"
        ]["$ref"].endswith("/ErrorResponse")
