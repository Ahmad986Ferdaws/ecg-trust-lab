#!/usr/bin/env python3
"""Run or safely resume the frozen exact-six PTB-XL robustness audit."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

from ecg_trust.audit_artifacts import AuditArtifactError
from ecg_trust.audit_runtime import AuditRuntimeError, load_completed_audit_runtime
from ecg_trust.post_evaluation import (
    PostEvaluationError,
    load_post_evaluation_spec,
)
from ecg_trust.protocol import ProtocolValidationError, load_protocol
from ecg_trust.robustness_audit import RobustnessAuditError, run_robustness_audit


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[1]
    comparison_id = "ptbxl_matched_equal_budget_v1"
    post_root = project_root / "runs" / "post_evaluation" / comparison_id
    parser = argparse.ArgumentParser(
        description=(
            "Verify the completed exact-six release and clean-logit gate, then run or "
            "resume the preregistered 41-case physical-mV robustness audit."
        )
    )
    parser.add_argument(
        "--protocol", type=Path, default=project_root / "configs" / "protocol.yaml"
    )
    parser.add_argument(
        "--post-evaluation-spec", type=Path, default=post_root / "audit_spec.json"
    )
    parser.add_argument(
        "--member-id",
        action="append",
        default=None,
        help=(
            "Run only this release member (repeatable). The exact-six clean gate still "
            "runs, and the final manifest remains withheld until all 246 units exist."
        ),
    )
    parser.add_argument(
        "--case-id",
        action="append",
        default=None,
        help=(
            "Run only this frozen case (repeatable). The member clean artifact is always "
            "verified first."
        ),
    )
    parser.add_argument(
        "--no-finalize",
        action="store_true",
        help="Verify/write member-case units but do not publish the final manifest.",
    )
    return parser


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise RobustnessAuditError(f"{context} must be a mapping")
    return cast(Mapping[str, object], value)


def _bound_path(value: object, context: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise RobustnessAuditError(f"{context} path is invalid")
    return Path(value).resolve()


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        protocol = load_protocol(args.protocol)
        spec = load_post_evaluation_spec(
            args.post_evaluation_spec,
            protocol=protocol,
            verify_sources=True,
            verify_git=True,
        )
        sealed = _mapping(spec.payload["sealed_evaluation"], "sealed_evaluation")
        final_spec = _mapping(
            sealed["final_evaluation_spec"], "final_evaluation_spec"
        )
        refit = _mapping(sealed["refit_bundle"], "refit_bundle")
        calibration = _mapping(sealed["calibration_bundle"], "calibration_bundle")
        ledger = _mapping(sealed["opening_ledger"], "opening_ledger")
        runtime = load_completed_audit_runtime(
            protocol=protocol,
            final_evaluation_spec_path=_bound_path(final_spec["path"], "final spec"),
            refit_bundle_path=_bound_path(refit["path"], "refit bundle"),
            calibration_bundle_path=_bound_path(
                calibration["path"], "calibration bundle"
            ),
            ledger_path=_bound_path(ledger["path"], "opening ledger"),
        )
        progress = run_robustness_audit(
            spec=spec,
            runtime=runtime,
            member_ids=args.member_id,
            case_ids=args.case_id,
            finalize=not args.no_finalize,
            progress=lambda message: print(message, file=sys.stderr, flush=True),
        )
    except (
        AuditArtifactError,
        AuditRuntimeError,
        FileExistsError,
        OSError,
        PostEvaluationError,
        ProtocolValidationError,
        RobustnessAuditError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(progress.to_dict(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
