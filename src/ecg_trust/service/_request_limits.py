"""Bound HTTP bodies before routing, JSON parsing or case resolution."""

from __future__ import annotations

from collections.abc import Callable

from starlette.responses import Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send


class RequestBodyLimitMiddleware:
    """Buffer at most max_bytes and replay a complete body to the application.

    Count received bytes even when Content-Length is absent or incorrect. The
    ASGI server owns transport decoding, receive timeouts and connection limits.
    """

    def __init__(
        self, app: ASGIApp, *, max_bytes: int, on_reject: Callable[[], Response]
    ) -> None:
        self.app = app
        self.max_bytes = max_bytes
        self.on_reject = on_reject

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        for name, value in scope.get("headers", []):
            if name.lower() == b"content-length":
                try:
                    declared = int(value)
                except ValueError:
                    # The server validates HTTP framing. An invalid declaration
                    # never disables the actual byte count below.
                    continue
                if declared > self.max_bytes:
                    await self.on_reject()(scope, receive, send)
                    return

        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            if len(body) + len(chunk) > self.max_bytes:
                await self.on_reject()(scope, receive, send)
                return
            body.extend(chunk)
            if not message.get("more_body", False):
                break

        # Hand the bounded content to the parser once. Later receive calls retain
        # the original channel so response streaming can observe disconnection.
        payload = bytes(body)
        body.clear()
        replayed = False

        async def replay_receive() -> Message:
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": payload, "more_body": False}
            return await receive()

        await self.app(scope, replay_receive, send)
