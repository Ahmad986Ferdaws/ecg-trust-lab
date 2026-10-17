from __future__ import annotations

import importlib.metadata as metadata
import json
from pathlib import Path

import pytest

from scripts.dependency_inventory import inventory


def install_metadata(root: Path, folder: str, body: str) -> None:
    target = root / (folder + ".dist-info")
    target.mkdir()
    (target / "METADATA").write_text("Metadata-Version: 2.4\n" + body)
    (target / "direct_url.json").write_text('{"url":"https://private:secret@example.invalid/repo"}')


def test_inventory_reads_real_metadata_without_importing_packages_or_source_urls(tmp_path) -> None:
    install_metadata(tmp_path, "zeta", "Name: Zeta_Pkg\nVersion: 1.2\nLicense-Expression: MIT\n")
    install_metadata(tmp_path, "alpha", "Name: alpha\nVersion: 2.0\nLicense: BSD\n"
                     "Classifier: License :: OSI Approved :: BSD License\nLicense-File: LICENSE\n")
    result = inventory(metadata.distributions(path=[str(tmp_path)]))
    rows = result["distributions"]
    assert result["distribution_count"] == 2
    assert [row["name"] for row in rows] == ["alpha", "zeta-pkg"]
    assert rows[0]["license_expression"] is None
    assert rows[0]["legacy_license_summary"] == "BSD"
    assert rows[0]["license_files_declared"] is True
    assert rows[1]["license_expression"] == "MIT"
    assert all(row["manual_review_required"] for row in rows)
    assert "secret" not in json.dumps(result)
    assert str(tmp_path) not in json.dumps(result)


def test_missing_license_is_unknown_and_conflicting_fields_remain_visible(tmp_path) -> None:
    install_metadata(tmp_path, "a", "Name: a\nVersion: 1\n")
    install_metadata(tmp_path, "b", "Name: b\nVersion: 1\nLicense: BSD\nLicense-Expression: MIT\n")
    rows = inventory(metadata.distributions(path=[str(tmp_path)]))["distributions"]
    assert rows[0]["license_expression"] is None
    assert rows[0]["legacy_license_summary"] is None
    assert rows[1]["conflicting_license_fields"] is True


def test_long_license_text_is_flagged_without_copying_it_into_summary(tmp_path) -> None:
    install_metadata(tmp_path, "a", "Name: a\nVersion: 1\nLicense: " + "text " * 100 + "\n")
    row = inventory(metadata.distributions(path=[str(tmp_path)]))["distributions"][0]
    assert row["legacy_license_text_present"] is True
    assert row["legacy_license_summary"] is None


def test_duplicate_normalized_names_are_not_silently_overwritten(tmp_path) -> None:
    install_metadata(tmp_path, "a", "Name: some_pkg\nVersion: 1\n")
    install_metadata(tmp_path, "b", "Name: Some-Pkg\nVersion: 2\n")
    with pytest.raises(ValueError, match="duplicate"):
        inventory(metadata.distributions(path=[str(tmp_path)]))


@pytest.mark.parametrize("body", ["Version: 1\n", "Name: a\n", "Name: ../a\nVersion: 1\n"])
def test_malformed_metadata_does_not_silently_reduce_inventory(tmp_path, body: str) -> None:
    install_metadata(tmp_path, "a", body)
    with pytest.raises(ValueError, match="metadata"):
        inventory(metadata.distributions(path=[str(tmp_path)]))
