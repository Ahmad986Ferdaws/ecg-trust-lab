from datetime import UTC, datetime
from email.message import Message
from io import BytesIO
from urllib.error import HTTPError

import pytest

from scripts import download_ptbxl as downloader


def http_error(status, retry_after=None):
    headers = Message()
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    return HTTPError("https://example.invalid/record", status, "synthetic", headers, BytesIO())


@pytest.mark.parametrize("status", [400, 401, 403, 404, 405, 410, 422, 501, 505])
def test_permanent_http_failure_is_not_retried(tmp_path, monkeypatch, status):
    error = http_error(status)
    attempts = []

    def fail(*args, **kwargs):
        attempts.append(1)
        raise error

    monkeypatch.setattr(downloader, "_download_once", fail)
    monkeypatch.setattr(downloader.time, "sleep", lambda delay: pytest.fail("retry wait"))
    with pytest.raises(HTTPError) as caught:
        downloader.download_file("record.dat", tmp_path, expected_sha256=None, timeout=1, retries=3)
    assert caught.value is error
    assert attempts == [1]
    assert error.closed


@pytest.mark.parametrize("status", [408, 416, 429, 500, 502, 503, 504])
def test_transient_http_failure_uses_bounded_exponential_fallback(tmp_path, monkeypatch, status):
    attempts, delays = [], []

    def download(*args, **kwargs):
        attempts.append(1)
        if len(attempts) < 3:
            raise http_error(status)
        return downloader.DownloadResult("record.dat", "downloaded", 1)

    monkeypatch.setattr(downloader, "_download_once", download)
    monkeypatch.setattr(downloader.time, "sleep", delays.append)
    result = downloader.download_file(
        "record.dat",
        tmp_path,
        expected_sha256=None,
        timeout=1,
        retries=3,
    )
    assert result.bytes_written == 1
    assert delays == [1, 2]


@pytest.mark.parametrize(
    "value,expected",
    [
        ("120", 120),
        ("0", 0),
        (" 0300 ", 300),
        ("00000120", 120),
        ("301", None),
        ("9" * 10000, None),
        ("-1", 2),
        ("1.5", 2),
        ("", 2),
        ("tomorrow", 2),
        ("nan", 2),
        ("inf", 2),
        ("１２", 2),
        ("Sun, 06 Nov 1994 08:49:37 GMT", 120),
        ("Sunday, 06-Nov-94 08:49:37 GMT", 120),
        ("Sun Nov  6 08:49:37 1994", 120),
        ("Sun, 06 Nov 1994 09:49:37 GMT", None),
        ("Sat, 05 Nov 1994 08:49:37 GMT", 0),
    ],
)
def test_retry_after_values(value, expected, monkeypatch):
    now = datetime(1994, 11, 6, 8, 47, 37, tzinfo=UTC).timestamp()
    monkeypatch.setattr(downloader.time, "time", lambda: now)
    assert downloader._http_retry_delay(http_error(503, value), 2) == expected


@pytest.mark.parametrize(
    "retry_after,attempt_count,delays",
    [
        ("120", 3, [120, 120]),
        ("86400", 1, []),
    ],
)
def test_retry_after_controls_real_retry_loop(
    tmp_path, monkeypatch, retry_after, attempt_count, delays
):
    attempts, waits = [], []

    def fail(*args, **kwargs):
        attempts.append(1)
        raise http_error(429, retry_after)

    monkeypatch.setattr(downloader, "_download_once", fail)
    monkeypatch.setattr(downloader.time, "sleep", waits.append)
    with pytest.raises(HTTPError):
        downloader.download_file("record.dat", tmp_path, expected_sha256=None, timeout=1, retries=3)
    assert len(attempts) == attempt_count
    assert waits == delays


def test_complete_partial_416_still_finishes_without_retry(tmp_path, monkeypatch):
    import hashlib

    content = b"complete partial"
    (tmp_path / "record.dat.part").write_bytes(content)
    error = http_error(416)

    def refuse(*args, **kwargs):
        raise error

    monkeypatch.setattr(downloader, "urlopen", refuse)
    monkeypatch.setattr(downloader.time, "sleep", lambda delay: pytest.fail("retry wait"))
    result = downloader.download_file(
        "record.dat",
        tmp_path,
        expected_sha256=hashlib.sha256(content).hexdigest(),
        timeout=1,
        retries=3,
    )
    assert result.status == "resumed-complete"
    assert (tmp_path / "record.dat").read_bytes() == content
    assert error.closed
