from __future__ import annotations

import hashlib
import io
from http.client import IncompleteRead
from pathlib import Path
from urllib.request import Request

import pytest

from ecg_trust.data.manifest import ManifestError
from scripts import download_ptbxl as downloader


class _Response(io.BytesIO):
    def __init__(self, body: bytes, status: int = 200, **headers: str) -> None:
        super().__init__(body)
        self.status = status
        self.headers = headers

    def getcode(self) -> int:
        return self.status


@pytest.mark.parametrize(
    "status,headers,body",
    [
        (206, {}, b"def"),
        (206, {"Content-Range": "bytes 0-2/6"}, b"def"),
        (206, {"Content-Range": "bytes 3-2/6"}, b"def"),
        (206, {"Content-Range": "bytes 3-6/6"}, b"def"),
        (206, {"Content-Range": "bytes 3-5/*"}, b"def"),
        (206, {"Content-Range": "bytes 3-4/6"}, b"de"),
        (206, {"Content-Range": "bytes 3-5/6", "Content-Length": "2"}, b"def"),
        (200, {"Content-Length": "6"}, b"ab"),
        (200, {"Content-Length": "2"}, b"abcdef"),
        (200, {"Content-Length": "-1"}, b"abc"),
        (200, {"Content-Length": "many"}, b"abc"),
        (204, {}, b""),
    ],
)
def test_malformed_or_incomplete_response_is_never_promoted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    status: int,
    headers: dict[str, str],
    body: bytes,
) -> None:
    (tmp_path / "record.dat.part").write_bytes(b"abc")
    monkeypatch.setattr(
        downloader, "urlopen", lambda *args, **kwargs: _Response(body, status, **headers)
    )
    with pytest.raises(ManifestError):
        downloader.download_file("record.dat", tmp_path, expected_sha256=None, timeout=1, retries=1)
    assert not (tmp_path / "record.dat").exists()


def test_valid_resume_checks_offset_and_preserves_existing_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "record.dat.part").write_bytes(b"abc")

    def open_response(request: Request, *, timeout: float) -> _Response:
        assert request.get_header("Range") == "bytes=3-"
        return _Response(b"def", 206, **{"Content-Range": "bytes 3-5/6", "Content-Length": "3"})

    monkeypatch.setattr(downloader, "urlopen", open_response)
    downloader.download_file("record.dat", tmp_path, expected_sha256=None, timeout=1, retries=1)
    assert (tmp_path / "record.dat").read_bytes() == b"abcdef"


def test_short_body_is_preserved_and_resumed_on_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    responses = iter(
        [
            _Response(b"abc", **{"Content-Length": "6"}),
            _Response(b"def", 206, **{"Content-Range": "bytes 3-5/6"}),
        ]
    )
    ranges: list[str | None] = []

    def open_response(request: Request, *, timeout: float) -> _Response:
        ranges.append(request.get_header("Range"))
        return next(responses)

    monkeypatch.setattr(downloader, "urlopen", open_response)
    monkeypatch.setattr(downloader.time, "sleep", lambda seconds: None)
    downloader.download_file("record.dat", tmp_path, expected_sha256=None, timeout=1, retries=2)
    assert ranges == [None, "bytes=3-"]
    assert (tmp_path / "record.dat").read_bytes() == b"abcdef"


def test_checksum_mismatch_discards_corrupt_partial_before_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    responses = iter([b"bad", b"good"])
    ranges: list[str | None] = []

    def open_response(request: Request, *, timeout: float) -> _Response:
        ranges.append(request.get_header("Range"))
        body = next(responses)
        return _Response(body, **{"Content-Length": str(len(body))})

    monkeypatch.setattr(downloader, "urlopen", open_response)
    monkeypatch.setattr(downloader.time, "sleep", lambda seconds: None)
    downloader.download_file(
        "record.dat",
        tmp_path,
        expected_sha256=hashlib.sha256(b"good").hexdigest(),
        timeout=1,
        retries=2,
    )
    assert ranges == [None, None]
    assert (tmp_path / "record.dat").read_bytes() == b"good"


def test_incomplete_http_read_is_retried(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = 0

    def open_response(*args: object, **kwargs: object) -> _Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise IncompleteRead(b"", 4)
        return _Response(b"good", **{"Content-Length": "4"})

    monkeypatch.setattr(downloader, "urlopen", open_response)
    monkeypatch.setattr(downloader.time, "sleep", lambda seconds: None)
    downloader.download_file("record.dat", tmp_path, expected_sha256=None, timeout=1, retries=2)
    assert attempts == 2


@pytest.mark.parametrize("timeout", [float("nan"), float("inf"), 0, -1, True])
def test_invalid_timeout_is_rejected_before_any_io(
    tmp_path: Path, timeout: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        downloader, "_download_once", lambda *args, **kwargs: downloader.DownloadResult("x", "x", 0)
    )
    with pytest.raises(ValueError, match="timeout"):
        downloader.download_file(
            "record.dat", tmp_path, expected_sha256=None, timeout=timeout, retries=1
        )


@pytest.mark.parametrize("retries", [0, -1, True, 1.5])
def test_invalid_retry_count_is_rejected(
    tmp_path: Path, retries: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        downloader, "_download_once", lambda *args, **kwargs: downloader.DownloadResult("x", "x", 0)
    )
    with pytest.raises(ValueError, match="retries"):
        downloader.download_file(
            "record.dat", tmp_path, expected_sha256=None, timeout=1, retries=retries
        )


@pytest.mark.parametrize("timeout", ["nan", "inf", "-inf"])
def test_cli_rejects_nonfinite_timeout(timeout: str) -> None:
    with pytest.raises(SystemExit) as error:
        downloader.parse_args([f"--timeout={timeout}"])
    assert error.value.code == 2
