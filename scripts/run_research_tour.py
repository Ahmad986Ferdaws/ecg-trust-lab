#!/usr/bin/env python3
"""Run three fixed synthetic research examples and report their context together."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, TypedDict

import numpy as np

import ecg_trust
from ecg_trust.gate_power import BoundMethod, plan_gate
from scripts.compare_abstention_scores import compare, synthetic_demo
from scripts.quality_gate_stress_report import build_report as quality_report

NOTICE = (
    "Synthetic examples only. No patient data or model weights are loaded. "
    "Completion is not clinical validation, a release decision or independent acceptance."
)
SEED = 0


class PlanningControls(TypedDict):
    sample_size: int
    maximum_rate: float
    true_rate: float
    confidence: float


PLANNING_CONTROLS: PlanningControls = {
    "sample_size": 200,
    "maximum_rate": 0.05,
    "true_rate": 0.02,
    "confidence": 0.95,
}


class TourEnvironmentError(RuntimeError):
    """The imported package does not belong to this source checkout."""


def _require_local_package(root: Path) -> None:
    expected = (root / "src" / "ecg_trust" / "__init__.py").resolve()
    if ecg_trust.__file__ is None or Path(ecg_trust.__file__).resolve() != expected:
        raise TourEnvironmentError("package is outside the source checkout")


def _source_context(root: Path) -> dict[str, object]:
    """Record checkout context without exposing paths or changed filenames."""

    try:
        revision = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--verify", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()
        if re.fullmatch(r"(?:[a-f0-9]{40}|[a-f0-9]{64})", revision) is None:
            raise ValueError("invalid revision")
        changed = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=normal"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout
    except (OSError, subprocess.SubprocessError, ValueError, UnicodeError):
        return {"revision": None, "has_uncommitted_changes": None}
    return {"revision": revision, "has_uncommitted_changes": bool(changed)}


def _environment() -> dict[str, object]:
    versions: dict[str, str | None] = {}
    for name in ("ecg-trust", "numpy", "scipy"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "python": platform.python_version(),
        "system": platform.system(),
        "architecture": platform.machine(),
        "packages": versions,
    }


def build_report() -> dict[str, Any]:
    """Evaluate fixed examples without exposing a data-input or fitting mode."""

    root = Path(__file__).resolve().parents[1]
    _require_local_package(root)
    quality = quality_report()
    probabilities, targets, members = synthetic_demo(seed=SEED)
    abstention = compare(
        probabilities,
        targets,
        thresholds=np.full(5, 0.5),
        member_probabilities=members,
    )
    abstention.update(
        {
            "source": "synthetic demo (not model output)",
            "seed": SEED,
            "decision_thresholds": [0.5] * 5,
            "notice": (
                "Score rankings describe this constructed cohort only; no gate is selected. "
                "Lower area metrics are better on the same predictions and loss scale."
            ),
            "loss_units": {
                "mean_binary_log_loss": "nats per label",
                "thresholded_hamming_loss": "fraction of misclassified labels",
            },
        }
    )
    planning = plan_gate(**PLANNING_CONTROLS, method=BoundMethod.CLOPPER_PEARSON).to_dict()
    planning["notice"] = (
        "Fixed illustrative binomial controls, not a study recommendation. "
        "Record-level count approximation; patient clustering is not modeled."
    )
    return {
        "schema_version": 1,
        "report_type": "ecg_trust.synthetic_research_tour",
        "status": "examples_completed",
        "notice": NOTICE,
        "environment": _environment(),
        "source_checkout": _source_context(root),
        "quality": quality,
        "abstention": abstention,
        "gate_planning": planning,
    }


def markdown(report: dict[str, Any]) -> str:
    quality = report["quality"]
    abstention = report["abstention"]
    planning = report["gate_planning"]
    context = report["source_checkout"]
    environment = report["environment"]
    lines = [
        "# Synthetic research tour",
        "",
        NOTICE,
        "",
        "## Execution context",
        "",
        "```json",
        json.dumps({"environment": environment, "source_checkout": context}, indent=2),
        "```",
        "",
        "Checkout context is informational; it is not a software attestation.",
        "",
        "## Constructed signal-quality faults",
        "",
        quality["notice"],
        "",
        "Presets: "
        + "; ".join(f"{key} = `{value}`" for key, value in quality["preset_versions"].items()),
        "",
        "| Case | Should refuse | v1 | v2 |",
        "|---|---|---|---|",
    ]
    for row in quality["cases"]:
        outcomes = [
            row[preset]["status"] + (" (fail-open)" if row[preset]["fail_open"] else "")
            for preset in ("v1", "v2")
        ]
        lines.append(
            f"| {row['description']} | {'yes' if row['should_refuse'] else 'no'} | "
            f"{outcomes[0]} | {outcomes[1]} |"
        )
    lines.extend(
        [
            "",
            "## Synthetic abstention-score comparison",
            "",
            f"{abstention['records']} constructed records; seed {abstention['seed']}; "
            "all decision thresholds are 0.5.",
            "",
            abstention["notice"],
        ]
    )
    for loss, values in abstention["losses"].items():
        lines.extend(
            [
                "",
                f"### {loss}",
                "",
                f"Mean loss: {values['mean_loss']:.6g} ({abstention['loss_units'][loss]}).",
                "",
                "| Score | AURC | Oracle AURC | Excess AURC | AUGRC |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for row in values["scores"]:
            lines.append(
                f"| {row['score']} | {row['aurc']:.6g} | {row['oracle_aurc']:.6g} | "
                f"{row['excess_aurc']:.6g} | {row['augrc']:.6g} |"
            )
    lines.extend(
        [
            "",
            "## Illustrative gate planning",
            "",
            planning["notice"],
            "",
            f"At n={planning['sample_size']}, true rate={planning['true_rate']}, "
            f"rate cap={planning['maximum_rate']}, one-sided confidence={planning['confidence']}; "
            f"the {planning['method']} gate passes at most {planning['max_passing_count']} events.",
            "",
            f"Pass probability under that assumed rate: {planning['pass_probability']:.6g}.",
            "",
            "No scientific release, customer handover or clinical-use approval "
            "follows from this report.",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=("json", "markdown"), default="markdown")
    args = parser.parse_args(argv)
    try:
        report = build_report()
        output = (
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
            if args.format == "json"
            else markdown(report)
        )
    except TourEnvironmentError:
        print(
            "error: install this checkout in editable mode using CONTRIBUTING.md", file=sys.stderr
        )
        return 1
    except Exception:
        print("error: synthetic tour failed; no report was produced", file=sys.stderr)
        return 1
    print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
