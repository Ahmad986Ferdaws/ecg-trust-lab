from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from ecg_trust import demo_app


@pytest.mark.parametrize("route", ["/metadata", "/assets/plotly.min.js"])
def test_unexpected_demo_errors_are_private_and_never_cached(
    route: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail() -> Any:
        raise RuntimeError("private artifact path must never appear in a response")

    class BrokenBackend:
        @property
        def policy(self) -> Any:
            return fail()

    monkeypatch.setattr(demo_app, "get_plotlyjs", fail)
    app = demo_app.create_app(backend=BrokenBackend())  # type: ignore[arg-type]
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get(route)

    assert response.status_code == 500
    assert response.json() == {"detail": "internal server error"}
    assert "private artifact" not in response.text
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["pragma"] == "no-cache"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["referrer-policy"] == "no-referrer"
    assert "frame-ancestors 'none'" in response.headers["content-security-policy"]
