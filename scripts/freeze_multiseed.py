#!/usr/bin/env python3
"""Freeze six explicit fold-8 confirmation receipts and generate refit recipes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ecg_trust.multiseed_freeze import (
    FreezeCreation,
    MultiSeedFreezeError,
    default_freeze_creation,
    publish_multiseed_freeze_bundle,
)
from ecg_trust.protocol import ProtocolValidationError, load_protocol


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep-summary",
        type=Path,
        default=(
            project_root
            / "runs"
            / "sweeps"
            / "ptbxl_matched_equal_budget_v1"
            / "sweep_summary.json"
        ),
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=project_root / "configs" / "protocol.yaml",
    )
    parser.add_argument(
        "--completion",
        action="append",
        type=Path,
        required=True,
        help="Explicit member_completion.json path; pass exactly six times.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            project_root
            / "runs"
            / "confirmation"
            / "ptbxl_matched_equal_budget_v1"
            / "multiseed_freeze.json"
        ),
    )
    parser.add_argument(
        "--recipes-dir",
        type=Path,
        default=None,
        help="Defaults to a refit_recipes directory beside the freeze artifact.",
    )
    parser.add_argument(
        "--refit-output-root",
        type=Path,
        default=project_root / "runs" / "refit" / "ptbxl_matched_equal_budget_v1",
    )
    parser.add_argument(
        "--created-at-utc",
        default=None,
        help="Optional deterministic YYYY-MM-DDTHH:MM:SSZ timestamp.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    project_root = Path(__file__).resolve().parents[1]
    if len(args.completion) != 6:
        print("error: --completion must be supplied exactly six times", file=sys.stderr)
        return 1
    try:
        protocol = load_protocol(args.protocol)
        creation = None
        if args.created_at_utc is not None:
            default = default_freeze_creation(project_root)
            creation = FreezeCreation(
                timestamp_utc=args.created_at_utc,
                code_revision=default.code_revision,
                dependency_lock_sha256=default.dependency_lock_sha256,
                software_versions=default.software_versions,
            )
        recipes_dir = args.recipes_dir or (args.output.parent / "refit_recipes")
        freeze, recipes = publish_multiseed_freeze_bundle(
            args.output,
            recipes_dir=recipes_dir,
            sweep_summary_path=args.sweep_summary,
            member_completion_paths=args.completion,
            protocol=protocol,
            refit_output_root=args.refit_output_root,
            creation=creation,
        )
    except (
        OSError,
        ValueError,
        MultiSeedFreezeError,
        ProtocolValidationError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"freeze artifact: {freeze.path}")
    print(f"freeze SHA-256: {freeze.artifact_sha256}")
    for recipe in recipes:
        print(f"refit recipe: {recipe}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
