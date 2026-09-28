import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import download_ptbxl as downloader


@pytest.mark.parametrize(
    "content,diagnostic",
    [
        (b"\xff", "UTF-8 CSV"),
        (b'filename_lr\n"records100/unclosed\n', "UTF-8 CSV"),
        (b'filename_lr\n"records100/a"trailing\n', "UTF-8 CSV"),
        (b"filename_lr,filename_lr\nrecords100/a,records100/b\n", "duplicate column"),
        (b"ecg_id,filename_lr\n1\n", "wrong field count"),
        (b"filename_lr\nrecords100/a,extra\n", "wrong field count"),
    ],
)
def test_malformed_metadata_becomes_manifest_error(tmp_path, content, diagnostic):
    path = tmp_path / "ptbxl_database.csv"
    path.write_bytes(content)
    with pytest.raises(downloader.ManifestError, match=diagnostic):
        downloader.read_record_stems(path)


def test_valid_quoted_metadata_preserves_paths(tmp_path, monkeypatch):
    path = tmp_path / "ptbxl_database.csv"
    path.write_text('filename_lr,extra\n"records100/b","quoted, value"\nrecords100/a,x\n')
    monkeypatch.setattr(downloader, "EXPECTED_RECORDS", 2)
    assert downloader.read_record_stems(path) == ["records100/a", "records100/b"]


@pytest.mark.parametrize("invalid_inventory", [True, False])
def test_cli_returns_concise_failure_without_creating_waveforms(tmp_path, invalid_inventory):
    metadata = b'filename_lr\n"records100/unclosed\n'
    (tmp_path / "ptbxl_database.csv").write_bytes(metadata)
    inventory = (
        b"\xff"
        if invalid_inventory
        else f"{hashlib.sha256(metadata).hexdigest()}  ptbxl_database.csv\n".encode()
    )
    (tmp_path / "SHA256SUMS.txt").write_bytes(inventory)
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/download_ptbxl.py"),
            "--destination",
            str(tmp_path),
            "--verify-only",
        ],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONPATH": str(root / "src")},
    )
    assert result.returncode == 1
    assert result.stderr.startswith("error: ")
    assert "UTF-8" in result.stderr
    assert "Traceback" not in result.stderr
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "SHA256SUMS.txt",
        "ptbxl_database.csv",
    ]
