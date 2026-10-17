"""Inventory installed Python distribution metadata for manual license review.

This is an environment inventory, not a complete SBOM or a license clearance.
It reads distribution metadata without importing package code or direct_url.json.
"""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import platform
import re
from collections.abc import Iterable


def inventory(distributions: Iterable[metadata.Distribution]) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for distribution in distributions:
        info = distribution.metadata
        name, version = info.get("Name", ""), info.get("Version", "")
        if not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?", name):
            raise ValueError("distribution has missing or malformed Name metadata")
        if not re.fullmatch(r"[A-Za-z0-9.!+_-]+", version):
            raise ValueError(f"{name} has missing or malformed Version metadata")
        normalized = re.sub(r"[-_.]+", "-", name).lower()
        if normalized in seen:
            raise ValueError(f"duplicate distribution metadata for {normalized}")
        seen.add(normalized)
        expression = info.get("License-Expression", "").strip()
        legacy = info.get("License", "").strip()
        classifiers = sorted(set(
            value for value in info.get_all("Classifier", []) if value.startswith("License ::")
        ))
        rows.append({
            "name": normalized,
            "version": version,
            "license_expression": expression or None,
            "legacy_license_summary": (
                legacy if len(legacy) <= 120 and "\n" not in legacy and legacy else None
            ),
            "legacy_license_text_present": bool(legacy),
            "license_classifiers": classifiers,
            "license_files_declared": bool(info.get_all("License-File", [])),
            "conflicting_license_fields": bool(expression and legacy),
            "manual_review_required": True,
        })
    rows.sort(key=lambda row: str(row["name"]))
    return {
        "schema_version": "ecg_trust.dependency_inventory.v1",
        "scope": "installed Python environment, including development tools",
        "python_version": platform.python_version(),
        "platform": platform.system(),
        "architecture": platform.machine(),
        "distribution_count": len(rows),
        "distributions": rows,
        "limitations": [
            "Metadata is publisher-declared, not verified legal permission.",
            "No archive hashes, dependency graph, JS packages or model/data licenses are included.",
            "License files are not opened and SPDX expressions are not parsed or approved.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    try:
        result = inventory(metadata.distributions())
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
