#!/usr/bin/env python3
"""Plan, inspect, or run the fixed three-seed fold-8 confirmation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOT = _PROJECT_ROOT / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from ecg_trust.multiseed_runner import (  # noqa: E402
    MultiSeedRunnerError,
    create_multiseed_plan,
    read_multiseed_status,
    run_multiseed_confirmation,
)
from ecg_trust.protocol import ProtocolValidationError, load_protocol  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    comparison_id = "ptbxl_matched_equal_budget_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=("plan", "status", "run"),
        default="status",
        nargs="?",
        help="status is read-only and is the safe default",
    )
    parser.add_argument(
        "--sweep-summary",
        type=Path,
        default=(
            project_root
            / "runs"
            / "sweeps"
            / comparison_id
            / "sweep_summary.json"
        ),
        help="completed schema-v2 paired sweep summary; used by plan only",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=project_root / "runs" / "confirmation",
        help="parent directory for the comparison-specific immutable plan",
    )
    parser.add_argument(
        "--plan",
        type=Path,
        default=(
            project_root
            / "runs"
            / "confirmation"
            / comparison_id
            / "multiseed_plan.json"
        ),
        help="persisted plan used by status and run",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=project_root / "configs" / "protocol.yaml",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume only the exact immutable plan; valid only with run",
    )
    args = parser.parse_args(argv)
    if args.resume and args.action != "run":
        parser.error("--resume is valid only with the run action")
    return args


def _print_json(payload: dict[str, object]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        protocol = load_protocol(args.protocol)
        if args.action == "plan":
            plan_result = create_multiseed_plan(
                args.sweep_summary,
                output_root=args.output_root,
                protocol=protocol,
            )
            _print_json({"action": "plan", **plan_result.to_dict()})
            return 0
        if args.action == "status":
            _print_json(
                {
                    "action": "status",
                    **read_multiseed_status(args.plan, protocol=protocol),
                }
            )
            return 0
        run_result = run_multiseed_confirmation(
            args.plan,
            protocol=protocol,
            resume=args.resume,
        )
        _print_json(
            {
                "action": "run",
                "mode": "resume" if args.resume else "fresh",
                **run_result.to_dict(),
            }
        )
        return 0
    except (
        OSError,
        ProtocolValidationError,
        MultiSeedRunnerError,
        RuntimeError,
        ValueError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
