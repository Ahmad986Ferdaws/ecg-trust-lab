from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from ecg_trust.demo_app import DemoAppConfig, DemoExample, DemoWebError, create_app


@pytest.mark.parametrize(
    "field",
    ["max_header_bytes", "max_signal_bytes", "integrated_gradients_steps", "inference_concurrency"],
)
@pytest.mark.parametrize(
    "value", [True, False, 2048.0, 2.5, float("nan"), float("inf"), "2048", None]
)
def test_operational_limits_require_integers(field: str, value: Any) -> None:
    with pytest.raises(DemoWebError, match=field):
        DemoAppConfig(**{field: value})


def test_example_registry_is_snapshotted_before_app_creation() -> None:
    example = DemoExample("original", "Original example", Path("record"))
    registry = [example]
    settings = DemoAppConfig(examples=registry)  # type: ignore[arg-type]
    registry.append(DemoExample("later", "Later mutation", Path("other")))

    with TestClient(create_app(config=settings)) as client:
        assert client.get("/examples").json() == {"examples": [example.public_dict()]}
    assert settings.examples == (example,)


@pytest.mark.parametrize("value", [None, "registry", [None], [{}]])
def test_malformed_example_registry_has_a_configuration_error(value: Any) -> None:
    with pytest.raises(DemoWebError, match="examples"):
        DemoAppConfig(examples=value)
