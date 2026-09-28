from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ecg_trust.passport import (
    MAX_CANONICAL_PASSPORT_BYTES,
    cli,
    model_passport_to_json_bytes,
    render_model_passport_markdown,
)
from tests.unit.test_model_passport import _hash, _passport

ROOT = Path(__file__).resolve().parents[2]
EXPECTED = [
    "--expected-release-sha256",
    _hash(1),
    "--expected-bundle-sha256",
    _hash(2),
    "--expected-protocol-sha256",
    _hash(5),
]


def _run(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "ecg_trust.passport.cli", *args],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )


@pytest.fixture
def passport_path(tmp_path: Path) -> Path:
    path = tmp_path / "passport.json"
    # Existing synthetic unit fixture; never a promoted or scientific artifact.
    path.write_bytes(model_passport_to_json_bytes(_passport()))
    return path


def test_verify_subprocess_requires_independent_identities_and_writes_no_export(
    tmp_path: Path,
    passport_path: Path,
) -> None:
    result = _run(
        tmp_path, "verify", str(passport_path), *EXPECTED, "--expected-release-id", "release-001"
    )
    assert result.returncode == 0
    assert json.loads(result.stdout) == {
        "verified": True,
        "passport_sha256": _passport().passport_sha256,
    }
    assert result.stderr == ""
    assert list(tmp_path.iterdir()) == [passport_path]
    missing = _run(tmp_path, "verify", str(passport_path))
    assert missing.returncode == 2
    assert missing.stdout == ""


@pytest.mark.parametrize("output_format", ["json", "markdown"])
def test_export_subprocess_matches_canonical_serializer_or_existing_renderer(
    tmp_path: Path,
    passport_path: Path,
    output_format: str,
) -> None:
    output = tmp_path / "export"
    result = _run(
        tmp_path,
        "export",
        str(passport_path),
        *EXPECTED,
        "--format",
        output_format,
        "--output",
        str(output),
    )
    assert result.returncode == 0, result.stderr
    expected = (
        passport_path.read_bytes()
        if output_format == "json"
        else render_model_passport_markdown(_passport()).encode("utf-8")
    )
    assert output.read_bytes() == expected
    assert not list(tmp_path.glob(".passport-*.tmp"))


@pytest.mark.parametrize(
    "mutation",
    [
        "tampered",
        "noncanonical",
        "privacy",
        "schema",
        "duplicate_key",
        "missing_newline",
        "malformed",
        "encoding",
        "oversize",
    ],
)
def test_invalid_input_never_exports_or_echoes_payload(
    tmp_path: Path,
    passport_path: Path,
    mutation: str,
) -> None:
    payload = passport_path.read_bytes()
    body = json.loads(payload)
    if mutation == "tampered":
        body["limitations"] = ["PRIVATE_PAYLOAD_SENTINEL"]
    elif mutation == "privacy":
        body["patient_id"] = "PRIVATE_PAYLOAD_SENTINEL"
    elif mutation == "schema":
        body["schema_version"] = "PRIVATE_PAYLOAD_SENTINEL"
    if mutation in {"tampered", "privacy", "schema"}:
        payload = json.dumps(body, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    elif mutation == "noncanonical":
        payload = json.dumps(body, indent=2).encode() + b"\n"
    elif mutation == "duplicate_key":
        payload = b'{"release_id":"PRIVATE_PAYLOAD_SENTINEL",' + payload[1:]
    elif mutation == "missing_newline":
        payload = payload.rstrip(b"\n")
    elif mutation == "malformed":
        payload = b'{"PRIVATE_PAYLOAD_SENTINEL": '
    elif mutation == "encoding":
        payload = b"\xffPRIVATE_PAYLOAD_SENTINEL"
    elif mutation == "oversize":
        payload = b" " * (MAX_CANONICAL_PASSPORT_BYTES + 1)
    passport_path.write_bytes(payload)
    output = tmp_path / "export.json"
    result = _run(tmp_path, "export", str(passport_path), *EXPECTED, "--output", str(output))
    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr == "error: passport verification failed; no output was written\n"
    assert not output.exists()
    assert list(tmp_path.iterdir()) == [passport_path]


@pytest.mark.parametrize("field", ["release", "bundle", "protocol", "release-id"])
def test_mismatched_expected_identity_fails_closed(
    tmp_path: Path,
    passport_path: Path,
    field: str,
) -> None:
    arguments = EXPECTED.copy()
    if field == "release-id":
        arguments += ["--expected-release-id", "PRIVATE_EXPECTATION"]
    else:
        arguments[arguments.index(f"--expected-{field}-sha256") + 1] = _hash(99)
    output = tmp_path / "export.json"
    result = _run(tmp_path, "export", str(passport_path), *arguments, "--output", str(output))
    assert result.returncode == 1
    assert result.stdout == ""
    assert "PRIVATE_EXPECTATION" not in result.stderr
    assert list(tmp_path.iterdir()) == [passport_path]


def test_invalid_digest_argument_does_not_echo_its_value(
    tmp_path: Path, passport_path: Path
) -> None:
    arguments = EXPECTED.copy()
    arguments[1] = "PRIVATE_EXPECTATION"
    result = _run(tmp_path, "verify", str(passport_path), *arguments)
    assert result.returncode == 2
    assert "PRIVATE_EXPECTATION" not in result.stderr


@pytest.mark.parametrize("existing_kind", ["file", "symlink", "dangling_symlink", "directory"])
def test_export_never_replaces_existing_destination(
    tmp_path: Path,
    passport_path: Path,
    existing_kind: str,
) -> None:
    output = tmp_path / "export.json"
    target = tmp_path / "unrelated"
    if existing_kind == "file":
        output.write_bytes(b"preserve")
    elif existing_kind == "directory":
        output.mkdir()
    else:
        if existing_kind == "symlink":
            target.write_bytes(b"preserve")
        try:
            output.symlink_to(target)
        except OSError:
            pytest.skip("symlink creation unavailable")
    result = _run(tmp_path, "export", str(passport_path), *EXPECTED, "--output", str(output))
    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr == "error: export destination already exists; no file was replaced\n"
    if existing_kind == "file":
        assert output.read_bytes() == b"preserve"
    elif existing_kind == "directory":
        assert output.is_dir()
    else:
        assert output.is_symlink()
        if existing_kind == "symlink":
            assert target.read_bytes() == b"preserve"
        else:
            assert not target.exists()
    assert not list(tmp_path.glob(".passport-*.tmp"))


def test_concurrent_destination_creation_is_not_clobbered(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "export.json"
    link = os.link

    def create_competing_output(source: Path, destination: Path) -> None:
        destination.write_bytes(b"concurrent result")
        link(source, destination)

    monkeypatch.setattr(cli.os, "link", create_competing_output)
    with pytest.raises(FileExistsError):
        cli._write_new_output(output, b"new result")
    assert output.read_bytes() == b"concurrent result"
    assert list(tmp_path.iterdir()) == [output]


@pytest.mark.parametrize("failure", [OSError, KeyboardInterrupt])
def test_staged_output_is_cleaned_on_write_failure_or_interrupt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: type[BaseException],
) -> None:
    def fail_sync(_descriptor: int) -> None:
        raise failure()

    monkeypatch.setattr(cli.os, "fsync", fail_sync)
    with pytest.raises(failure):
        cli._write_new_output(tmp_path / "export.json", b"new result")
    assert list(tmp_path.iterdir()) == []


def test_regular_file_input_is_bounded_even_when_stat_is_stale(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "oversized.json"
    source.write_bytes(b" " * 1024)
    monkeypatch.setattr(cli, "MAX_CANONICAL_PASSPORT_BYTES", 16)
    original = os.fstat

    def stale_size(descriptor: int) -> os.stat_result:
        details = list(original(descriptor))
        details[6] = 0
        return os.stat_result(details)

    monkeypatch.setattr(cli.os, "fstat", stale_size)
    assert len(cli._read_payload(source)) == 17


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO requires a POSIX platform")
def test_fifo_input_is_rejected_without_waiting_for_writer(tmp_path: Path) -> None:
    source = tmp_path / "fifo"
    os.mkfifo(source)
    result = _run(tmp_path, "verify", str(source), *EXPECTED)
    assert result.returncode == 1
    assert "Traceback" not in result.stderr
