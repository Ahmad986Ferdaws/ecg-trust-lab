from __future__ import annotations

import hashlib
import io
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ecg_trust.data.manifest import ManifestError
from scripts import download_ptbxl as downloader


class _Response(io.BytesIO):
    status = 200
    headers: dict[str, str] = {}

    def getcode(self) -> int:
        return self.status


@pytest.mark.parametrize("kind", ["symlink", "dangling", "hardlink", "directory"])
@pytest.mark.parametrize("force", [False, True])
def test_download_rejects_aliased_or_special_partial_before_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str, force: bool
) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    unrelated = tmp_path / "unrelated"
    unrelated.write_bytes(b"preserve me")
    partial = root / "record.dat.part"
    if kind == "symlink":
        partial.symlink_to(unrelated)
    elif kind == "dangling":
        partial.symlink_to(tmp_path / "absent")
    elif kind == "hardlink":
        os.link(unrelated, partial)
    else:
        partial.mkdir()

    def unexpected_network(*args: object, **kwargs: object) -> None:
        pytest.fail("unsafe partial must be rejected before requesting bytes")

    monkeypatch.setattr(downloader, "urlopen", unexpected_network)
    with pytest.raises(ManifestError, match="partial"):
        downloader.download_file(
            "record.dat", root, expected_sha256=None, timeout=1, retries=1, force=force
        )
    assert unrelated.read_bytes() == b"preserve me"
    assert not (root / "record.dat").exists()
    assert not (tmp_path / "absent").exists()


@pytest.mark.parametrize("existing", [None, b"old partial"])
def test_regular_partial_is_replaced_by_verified_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, existing: bytes | None
) -> None:
    if existing is not None:
        (tmp_path / "record.dat.part").write_bytes(existing)
    monkeypatch.setattr(downloader, "urlopen", lambda *args, **kwargs: _Response(b"complete"))
    result = downloader.download_file(
        "record.dat",
        tmp_path,
        expected_sha256=hashlib.sha256(b"complete").hexdigest(),
        timeout=1,
        retries=1,
    )
    assert result.status == "downloaded"
    assert (tmp_path / "record.dat").read_bytes() == b"complete"
    assert not (tmp_path / "record.dat.part").exists()


def test_open_partial_rechecks_file_before_truncating(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    partial = tmp_path / "record.part"
    partial.write_bytes(b"old")
    unrelated = tmp_path / "unrelated"
    unrelated.write_bytes(b"preserve me")
    original_open = os.open

    def replace_with_hardlink(path: Path, flags: int, mode: int) -> int:
        partial.unlink()
        os.link(unrelated, partial)
        return original_open(path, flags, mode)

    monkeypatch.setattr(downloader.os, "open", replace_with_hardlink)
    with pytest.raises(ManifestError, match="partial"):
        downloader._open_partial(partial, append=False)
    assert unrelated.read_bytes() == b"preserve me"


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO race requires POSIX")
def test_replacing_partial_with_fifo_cannot_block_the_downloader(tmp_path: Path) -> None:
    # Use a subprocess so a regression becomes a bounded test failure, rather
    # than permanently hanging the pytest worker in a blocking FIFO open.
    partial = tmp_path / "record.part"
    partial.write_bytes(b"old")
    code = """
import os
import sys
from pathlib import Path
from scripts.download_ptbxl import _open_partial

partial = Path(sys.argv[1])
original_open = os.open
def replace_with_fifo(path, flags, mode):
    partial.unlink()
    os.mkfifo(partial)
    return original_open(path, flags, mode)
os.open = replace_with_fifo
try:
    handle = _open_partial(partial, append=False)
except OSError:
    pass
else:
    handle.close()
    raise AssertionError("FIFO should not be opened for a download")
"""
    completed = subprocess.run(
        [sys.executable, "-c", code, str(partial)], capture_output=True, text=True, timeout=3
    )
    assert completed.returncode == 0, completed.stderr
