"""Queue and interruption contracts without network access or patient data."""

from concurrent.futures import CancelledError
from io import BytesIO
from threading import Event, Thread

import pytest

from scripts import download_ptbxl as downloader


def run_many(tmp_path, *, workers=2, count=50):
    return downloader._download_many(
        [f"file-{index}" for index in range(count)],
        tmp_path,
        checksums={},
        workers=workers,
        timeout=1,
        retries=1,
        force=False,
    )


@pytest.mark.parametrize("workers", [1, 2, 8])
def test_in_flight_work_is_bounded_and_every_file_reported(tmp_path, monkeypatch, capsys, workers):
    actual_wait = downloader.wait
    window_sizes = []
    paths = []

    def observe_wait(futures, **kwargs):
        window_sizes.append(len(futures))
        return actual_wait(futures, **kwargs)

    def download(path, *args, **kwargs):
        paths.append(path)
        return downloader.DownloadResult(path, "downloaded", 1)

    monkeypatch.setattr(downloader, "wait", observe_wait)
    monkeypatch.setattr(downloader, "download_file", download)
    assert run_many(tmp_path, workers=workers) == {"downloaded": 50}
    assert len(set(paths)) == 50
    assert max(window_sizes) <= 2 * workers
    assert "progress: 50/50 files" in capsys.readouterr().out


def test_interrupt_cancels_queue_and_joins_active_workers(tmp_path, monkeypatch):
    started = Event()
    entered = []
    exited = []

    def download(path, *args, cancellation, **kwargs):
        downloader._check_cancelled(cancellation)
        entered.append(path)
        started.set()
        assert cancellation.wait(2)
        exited.append(path)
        raise CancelledError()

    def interrupt(*args, **kwargs):
        assert started.wait(2)
        raise KeyboardInterrupt()

    monkeypatch.setattr(downloader, "download_file", download)
    monkeypatch.setattr(downloader, "wait", interrupt)
    with pytest.raises(KeyboardInterrupt):
        run_many(tmp_path, workers=1, count=1000)
    assert len(entered) == 1
    assert entered == exited


@pytest.mark.parametrize("error", [CancelledError, KeyboardInterrupt, SystemExit])
def test_worker_control_flow_is_not_collected_as_transfer_failure(tmp_path, monkeypatch, error):
    def download(*args, **kwargs):
        raise error()

    monkeypatch.setattr(downloader, "download_file", download)
    with pytest.raises(error):
        run_many(tmp_path, workers=1)


def test_ordinary_failures_are_collected_and_other_files_finish(tmp_path, monkeypatch):
    visited = []

    def download(path, *args, **kwargs):
        visited.append(path)
        if path == "file-1":
            raise OSError("synthetic failure")
        return downloader.DownloadResult(path, "verified-existing", 0)

    monkeypatch.setattr(downloader, "download_file", download)
    with pytest.raises(downloader.ManifestError, match="1 downloads failed"):
        run_many(tmp_path)
    assert len(set(visited)) == 50


def test_cancellation_wakes_retry_backoff(tmp_path, monkeypatch):
    cancellation, attempted = Event(), Event()
    errors = []

    def fail(*args, **kwargs):
        attempted.set()
        raise OSError("retryable")

    def work():
        try:
            downloader.download_file(
                "record.dat",
                tmp_path,
                expected_sha256=None,
                timeout=1,
                retries=20,
                cancellation=cancellation,
            )
        except CancelledError as exc:
            errors.append(exc)

    monkeypatch.setattr(downloader, "_download_once", fail)
    worker = Thread(target=work)
    worker.start()
    try:
        assert attempted.wait(2)
    finally:
        cancellation.set()
        worker.join(2)
    assert not worker.is_alive()
    assert len(errors) == 1


def test_cancelled_body_preserves_resumable_partial_and_existing_destination(tmp_path, monkeypatch):
    cancellation = Event()
    destination = tmp_path / "record.dat"
    destination.write_bytes(b"old-result")

    class Response(BytesIO):
        status = 200
        headers = {}
        reads = 0

        def getcode(self):
            return 200

        def read1(self, length):
            self.reads += 1
            if self.reads == 1:
                return b"prefix"
            cancellation.set()
            return b"unpublished"

    monkeypatch.setattr(downloader, "urlopen", lambda *args, **kwargs: Response())
    with pytest.raises(CancelledError):
        downloader.download_file(
            "record.dat",
            tmp_path,
            expected_sha256=None,
            timeout=1,
            retries=3,
            force=True,
            cancellation=cancellation,
        )
    assert destination.read_bytes() == b"old-result"
    assert (tmp_path / "record.dat.part").read_bytes() == b"prefix"


def test_pre_cancelled_download_never_opens_network(tmp_path, monkeypatch):
    cancellation = Event()
    cancellation.set()
    monkeypatch.setattr(
        downloader, "urlopen", lambda *args, **kwargs: pytest.fail("network opened")
    )
    with pytest.raises(CancelledError):
        downloader.download_file(
            "record.dat",
            tmp_path,
            expected_sha256=None,
            timeout=1,
            retries=1,
            cancellation=cancellation,
        )
    assert list(tmp_path.iterdir()) == []
