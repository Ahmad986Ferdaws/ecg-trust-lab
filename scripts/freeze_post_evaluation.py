#!/usr/bin/env python
"""Freeze the read-only post-evaluation audit plan after final-batch completion."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from ecg_trust.post_evaluation import (
    PostEvaluationError,
    freeze_post_evaluation_spec,
)
from ecg_trust.protocol import ProtocolValidationError, load_protocol


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[1]
    comparison_id = "ptbxl_matched_equal_budget_v1"
    release_root = project_root / "runs" / "release" / comparison_id
    parser = argparse.ArgumentParser(
        description=(
            "Verify and bind the completed exact-six fold-10 release, then atomically "
            "freeze robustness, explanation, probability-audit, and demo choices."
        )
    )
    parser.add_argument("--protocol", type=Path, default=project_root / "configs" / "protocol.yaml")
    parser.add_argument(
        "--final-batch-summary",
        type=Path,
        default=release_root / "fold10_final" / "final-batch-summary.json",
    )
    parser.add_argument(
        "--opening-ledger",
        type=Path,
        default=None,
        help="Optional explicit canonical ledger path; otherwise inferred from the sealed spec.",
    )
    parser.add_argument("--refit-bundle", type=Path, default=release_root / "refit_bundle.json")
    parser.add_argument(
        "--calibration-bundle",
        type=Path,
        default=release_root / "calibration_bundle.json",
    )
    parser.add_argument(
        "--final-evaluation-spec",
        type=Path,
        default=release_root / "final_evaluation_spec.json",
    )
    parser.add_argument(
        "--protocol-deviations",
        type=Path,
        default=project_root / "reports" / "PROTOCOL_DEVIATIONS.md",
    )
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Optional exact output root. Defaults to the legacy comparison root, or to "
            "the next __audit-rN sibling when --supersedes-spec is supplied."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional canonical audit_spec.json path under the resolved output root.",
    )
    parser.add_argument(
        "--supersedes-spec",
        type=Path,
        default=None,
        help=(
            "Explicit immutable prior post-evaluation spec binding for a versioned "
            "replacement freeze."
        ),
    )
    parser.add_argument(
        "--supersession-reason",
        default=None,
        help=(
            "Required with --supersedes-spec; must be one of the fixed canonical "
            "supersession reasons."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        protocol = load_protocol(args.protocol)
        saved = freeze_post_evaluation_spec(
            args.output,
            protocol=protocol,
            final_batch_summary_path=args.final_batch_summary,
            opening_ledger_path=args.opening_ledger,
            refit_bundle_path=args.refit_bundle,
            calibration_bundle_path=args.calibration_bundle,
            final_evaluation_spec_path=args.final_evaluation_spec,
            protocol_deviations_path=args.protocol_deviations,
            project_root=args.project_root,
            output_root=args.output_root,
            supersedes_spec_path=args.supersedes_spec,
            supersession_reason=args.supersession_reason,
        )
    except (OSError, PostEvaluationError, ProtocolValidationError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "path": str(saved.path),
                "artifact_sha256": saved.artifact_sha256,
                "output_root": str(saved.output_root),
                "member_ids": list(saved.member_ids),
                "release_inputs_modified": False,
                "schema_version": saved.payload["schema_version"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
