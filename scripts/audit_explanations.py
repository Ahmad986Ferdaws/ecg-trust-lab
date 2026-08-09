#!/usr/bin/env python
"""Run the immutable post-evaluation explanation and faithfulness audit."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a mapping")
    return cast(Mapping[str, object], value)


def _path(value: object, context: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} path is invalid")
    return Path(value).resolve()


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[1]
    comparison_id = "ptbxl_matched_equal_budget_v1"
    parser = argparse.ArgumentParser(
        description=(
            "Verify the completed exact-six release and run the frozen 60-record "
            "explanation, repeatability, faithfulness, and randomization audits."
        )
    )
    parser.add_argument("--protocol", type=Path, default=project_root / "configs" / "protocol.yaml")
    parser.add_argument(
        "--spec",
        type=Path,
        default=(project_root / "runs" / "post_evaluation" / comparison_id / "audit_spec.json"),
    )
    parser.add_argument(
        "--outer-batch-size",
        type=int,
        default=4,
        help="Attribution examples per outer GPU batch; frozen method settings are unchanged.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from ecg_trust.audit_runtime import load_completed_audit_runtime
    from ecg_trust.explanation_audit import run_explanation_audit
    from ecg_trust.post_evaluation import load_post_evaluation_spec
    from ecg_trust.protocol import load_protocol

    try:
        protocol = load_protocol(args.protocol)
        spec = load_post_evaluation_spec(
            args.spec,
            protocol=protocol,
            verify_sources=True,
            verify_git=True,
        )
        sealed = _mapping(spec.payload["sealed_evaluation"], "sealed_evaluation")
        final_spec = _mapping(sealed["final_evaluation_spec"], "final_evaluation_spec")
        refit = _mapping(sealed["refit_bundle"], "refit_bundle")
        calibration = _mapping(sealed["calibration_bundle"], "calibration_bundle")
        ledger = _mapping(sealed["opening_ledger"], "opening_ledger")
        runtime = load_completed_audit_runtime(
            protocol=protocol,
            final_evaluation_spec_path=_path(final_spec["path"], "final spec"),
            refit_bundle_path=_path(refit["path"], "refit bundle"),
            calibration_bundle_path=_path(calibration["path"], "calibration bundle"),
            ledger_path=_path(ledger["path"], "opening ledger"),
        )
        manifest = run_explanation_audit(
            spec=spec,
            runtime=runtime,
            outer_batch_size=args.outer_batch_size,
            progress=lambda message: print(message, flush=True),
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "path": str(manifest.path),
                "artifact_sha256": manifest.artifact_sha256,
                "post_evaluation_spec_sha256": spec.artifact_sha256,
                "release_inputs_modified": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
