from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import benchmark_models


def test_benchmark_writer_does_not_follow_stale_temp_symlink(tmp_path: Path) -> None:
    victim = tmp_path / "unrelated"
    victim.write_bytes(b"preserve")
    stale = tmp_path / "result.json.tmp"
    stale.symlink_to(victim)
    output = tmp_path / "result.json"
    benchmark_models._write_json({"result": 1}, output)
    assert victim.read_bytes() == b"preserve"
    assert stale.is_symlink()
    assert not output.is_symlink()
    assert json.loads(output.read_text()) == {"result": 1}


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_benchmark_cannot_replace_existing_result(tmp_path: Path, value: float) -> None:
    output = tmp_path / "result.json"
    output.write_bytes(b"previous result")
    with pytest.raises(ValueError):
        benchmark_models._write_json({"nested": [{"metric": value}]}, output)
    assert output.read_bytes() == b"previous result"
    assert list(tmp_path.iterdir()) == [output]


def test_failed_benchmark_replace_preserves_previous_result_and_cleans_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "result.json"
    output.write_bytes(b"previous result")

    def fail_replace(*args: object) -> None:
        raise OSError("destination is unavailable")

    monkeypatch.setattr(benchmark_models.os, "replace", fail_replace)
    with pytest.raises(OSError, match="unavailable"):
        benchmark_models._write_json({"metric": 1.0}, output)
    assert output.read_bytes() == b"previous result"
    assert list(tmp_path.iterdir()) == [output]


def test_benchmark_json_keeps_original_format(tmp_path: Path) -> None:
    payload = {"z": [None, True, 1.25], "a": "synthetic"}
    output = tmp_path / "nested" / "result.json"
    benchmark_models._write_json(payload, output)
    assert output.read_text() == json.dumps(payload, indent=2, sort_keys=True) + "\n"
    assert list(output.parent.iterdir()) == [output]
