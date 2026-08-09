#!/usr/bin/env python3
"""Render or finalize immutable post-evaluation publication artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from ecg_trust.final_results import finalize_results, render_probability_results
from ecg_trust.protocol import load_protocol


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[1]
    comparison_id = "ptbxl_matched_equal_budget_v1"
    default_spec = project_root / "runs" / "post_evaluation" / comparison_id / "audit_spec.json"
    parser = argparse.ArgumentParser(
        description=(
            "Render frozen post-evaluation probability outputs or finalize the report "
            "after all branch manifests verify. Output paths come only from audit_spec.json."
        )
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)
    for mode in ("probability", "finalize"):
        command = subparsers.add_parser(mode)
        command.add_argument(
            "--protocol", type=Path, default=project_root / "configs" / "protocol.yaml"
        )
        command.add_argument("--spec", type=Path, default=default_spec)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        protocol = load_protocol(args.protocol)
        if args.mode == "probability":
            result_payload = render_probability_results(args.spec, protocol=protocol).to_dict()
        else:
            result_payload = finalize_results(args.spec, protocol=protocol).to_dict()
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result_payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
