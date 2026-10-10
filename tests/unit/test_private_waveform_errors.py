from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from ecg_trust import demo_backend
from ecg_trust.demo_app import DemoAppConfig, DemoExample, create_app


def test_example_decoder_error_does_not_disclose_local_paths_or_library_details(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    stem = tmp_path / "private-record"
    stem.with_suffix(".hea").touch()
    stem.with_suffix(".dat").touch()

    def fail(_: str) -> Any:
        raise ValueError("sensitive decoder detail from private subject metadata")

    monkeypatch.setattr(demo_backend.wfdb, "rdrecord", fail)
    config = DemoAppConfig(examples=(DemoExample("public-example", "Example", stem),))
    app = create_app(config=config, backend=object())  # type: ignore[arg-type]
    with TestClient(app) as client:
        response = client.post("/predict/example/public-example")

    assert response.status_code == 422
    assert response.json() == {
        "detail": "could not read WFDB record; verify the matched header and signal files"
    }
    assert str(tmp_path) not in response.text
    assert "sensitive decoder" not in response.text
    assert response.headers["cache-control"] == "no-store"


def test_local_decoder_failure_retains_exception_chain_for_debugging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    stem = tmp_path / "record"
    stem.with_suffix(".hea").touch()
    stem.with_suffix(".dat").touch()
    cause = OSError("private filesystem detail")

    def fail(_: str) -> Any:
        raise cause

    monkeypatch.setattr(demo_backend.wfdb, "rdrecord", fail)
    with pytest.raises(demo_backend.DemoInputError) as captured:
        demo_backend.load_wfdb_physical_signal(stem)
    assert captured.value.__cause__ is cause
    assert "private filesystem" not in str(captured.value)
