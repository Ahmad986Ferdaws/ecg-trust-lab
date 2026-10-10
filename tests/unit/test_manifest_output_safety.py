from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import ecg_trust.data.manifest as manifest_module
from ecg_trust.data.manifest import ManifestError, write_manifest_artifacts


@pytest.mark.parametrize(
    "stem",
    [
        "../escaped",
        "/escaped",
        "nested/name",
        "nested\\name",
        "C:drive",
        "",
        ".",
        "..",
        " leading",
        "trailing ",
        "trailing.",
        "line\nbreak",
        "bad?name",
        "NUL",
        "con.data",
    ],
)
def test_invalid_artifact_stem_fails_before_creating_output(tmp_path: Path, stem: str) -> None:
    output = tmp_path / "output"
    with pytest.raises(ManifestError, match="stem"):
        write_manifest_artifacts(pd.DataFrame({"value": [1]}), {}, output, stem=stem)
    assert not output.exists()
    assert list(tmp_path.iterdir()) == []


def test_stale_temp_symlink_cannot_overwrite_unrelated_file(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    unrelated = tmp_path / "unrelated"
    unrelated.write_bytes(b"preserve me")
    stale = output / "fixture.csv.tmp"
    stale.symlink_to(unrelated)

    artifacts = write_manifest_artifacts(pd.DataFrame({"value": [1]}), {}, output, stem="fixture")

    assert unrelated.read_bytes() == b"preserve me"
    assert stale.is_symlink()
    assert not artifacts.csv_path.is_symlink()
    assert pd.read_csv(artifacts.csv_path)["value"].tolist() == [1]
    assert not list(output.glob(".fixture.*"))


def test_serialization_failure_cleans_staging_without_replacing_existing_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "output"
    original = write_manifest_artifacts(pd.DataFrame({"value": [1]}), {}, output, stem="fixture")
    before = {path.name: path.read_bytes() for path in output.iterdir()}

    def fail_parquet(*args: object, **kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(manifest_module.pq, "write_table", fail_parquet)
    with pytest.raises(OSError, match="disk full"):
        write_manifest_artifacts(pd.DataFrame({"value": [2]}), {}, output, stem="fixture")
    assert {path.name: path.read_bytes() for path in output.iterdir()} == before
    assert original.csv_path.read_bytes() == before[original.csv_path.name]
