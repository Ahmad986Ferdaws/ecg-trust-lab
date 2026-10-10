from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from ecg_trust.data.manifest import ManifestError
from scripts import download_ptbxl as downloader


@pytest.mark.parametrize("verify_only", [False, True])
@pytest.mark.parametrize("allow_missing_checksums", [False, True])
@pytest.mark.parametrize("inventory", ["mismatch", "missing"])
def test_unverified_metadata_cannot_construct_the_download_worklist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    verify_only: bool,
    allow_missing_checksums: bool,
    inventory: str,
) -> None:
    metadata = b"filename_lr\nrecords100/untrusted\n"
    checksums = "0" * 64 + "  other.txt\n"
    if inventory == "mismatch":
        checksums += "0" * 64 + "  ptbxl_database.csv\n"
    (tmp_path / "ptbxl_database.csv").write_bytes(metadata)
    (tmp_path / "SHA256SUMS.txt").write_text(checksums, encoding="utf-8")
    monkeypatch.setattr(downloader, "_bootstrap_file", lambda *args, **kwargs: None)

    def unsafe_worklist(path: Path) -> list[str]:
        pytest.fail("unverified metadata reached worklist construction")

    monkeypatch.setattr(downloader, "selected_paths", unsafe_worklist)
    with pytest.raises(ManifestError, match="checksum|SHA-256"):
        downloader.acquire(
            tmp_path,
            workers=1,
            timeout=1,
            retries=1,
            verify_only=verify_only,
            allow_missing_checksums=allow_missing_checksums,
            force=False,
        )


def test_valid_bootstrap_allows_worklist_and_waveform_downloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metadata = b"filename_lr\nrecords100/example\n"
    payloads = {
        "ptbxl_database.csv": metadata,
        "records100/example.hea": b"header",
        "records100/example.dat": b"signal",
    }
    inventory = "".join(
        f"{hashlib.sha256(value).hexdigest()}  {name}\n" for name, value in payloads.items()
    ).encode()
    payloads["SHA256SUMS.txt"] = inventory
    requests: list[str] = []

    def download(relative_path: str, root: Path, **kwargs: object) -> downloader.DownloadResult:
        requests.append(relative_path)
        destination = root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payloads[relative_path])
        return downloader.DownloadResult(relative_path, "downloaded", len(payloads[relative_path]))

    monkeypatch.setattr(downloader, "download_file", download)
    monkeypatch.setattr(downloader, "EXPECTED_RECORDS", 1)
    monkeypatch.setattr(downloader, "ROOT_FILES", ("SHA256SUMS.txt", "ptbxl_database.csv"))
    downloader.acquire(
        tmp_path,
        workers=1,
        timeout=1,
        retries=1,
        verify_only=False,
        allow_missing_checksums=False,
        force=False,
    )
    assert requests[:2] == ["SHA256SUMS.txt", "ptbxl_database.csv"]
    assert set(requests[2:]) == {"records100/example.hea", "records100/example.dat"}
