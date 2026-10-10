#!/usr/bin/env python3
"""Verify a wheel installed with --no-deps --target, without importing source files.

Run with ``python -I scripts/smoke_installed_wheel.py <installation-directory>``
inside the CPU development environment so third-party dependencies are available.
The isolated interpreter ignores PYTHONPATH and the checkout working directory.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import importlib.resources
import json
import sys
from pathlib import Path


def verify(installation: Path) -> dict[str, str]:
    if not sys.flags.isolated:
        raise RuntimeError("run the wheel smoke check with python -I")
    installation = installation.resolve(strict=True)
    sys.path.insert(0, str(installation))
    package = importlib.import_module("ecg_trust")
    package_path = Path(package.__file__).resolve()
    if not package_path.is_relative_to(installation):
        raise RuntimeError("ecg_trust was imported from outside the wheel installation")

    distributions = list(importlib.metadata.distributions(path=[str(installation)]))
    package_distributions = [item for item in distributions if item.metadata["Name"] == "ecg-trust"]
    if len(package_distributions) != 1:
        raise RuntimeError("wheel installation must contain exactly one ecg-trust distribution")
    distribution = package_distributions[0]
    entry_points = [
        entry
        for entry in distribution.entry_points
        if entry.group == "console_scripts" and entry.name == "ecg-verify"
    ]
    if len(entry_points) != 1 or entry_points[0].value != "ecg_trust.system_check:main":
        raise RuntimeError("wheel is missing the ecg-verify console entry point")
    if not callable(entry_points[0].load()):
        raise RuntimeError("ecg-verify entry point cannot be loaded")
    resources = importlib.resources.files(package)
    if not resources.joinpath("py.typed").is_file():
        raise RuntimeError("wheel is missing its py.typed marker")
    if not resources.joinpath("templates", "index.html").is_file():
        raise RuntimeError("wheel is missing the demo HTML template")

    from fastapi.testclient import TestClient

    from ecg_trust.demo_app import create_app

    with TestClient(create_app()) as client:
        page = client.get("/")
        if page.status_code != 200 or "Research use only" not in page.text:
            raise RuntimeError("installed demo did not render its research-only page")
        javascript = client.get("/assets/plotly.min.js")
        if javascript.status_code != 200 or "plotly" not in javascript.text.lower():
            raise RuntimeError("installed demo cannot serve its local plotting dependency")
        health = client.get("/health")
        if health.status_code != 503 or health.json().get("model_loaded") is not False:
            raise RuntimeError("installed demo incorrectly reports a loaded model")
    return {"status": "PASS", "version": distribution.version, "package": str(package_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("installation", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.installation), indent=2))


if __name__ == "__main__":
    main()
