#!/usr/bin/env python
"""Freeze all scientific and runtime choices before fold-9 inference."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from ecg_trust.final_evaluation_spec import freeze_final_evaluation_spec
from ecg_trust.protocol import load_protocol


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create the immutable final-evaluation specification before fold 9. "
            "The worktree must be clean and the CUDA/BF16 device must be explicit."
        )
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/protocol.yaml"),
    )
    parser.add_argument("--refit-bundle", type=Path, required=True)
    parser.add_argument("--subgroups", type=Path, required=True)
    parser.add_argument(
        "--protocol-deviations",
        type=Path,
        default=Path("reports/PROTOCOL_DEVIATIONS.md"),
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument(
        "--device",
        required=True,
        help="Explicit indexed CUDA device, for example cuda:0; auto/bare cuda are forbidden.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    protocol = load_protocol(args.protocol)
    spec = freeze_final_evaluation_spec(
        args.output,
        protocol=protocol,
        protocol_path=args.protocol,
        refit_bundle_path=args.refit_bundle,
        subgroup_artifact_path=args.subgroups,
        protocol_deviations_path=args.protocol_deviations,
        project_root=args.project_root,
        device=args.device,
    )
    print(
        json.dumps(
            {
                "path": str(spec.path),
                "artifact_sha256": spec.artifact_sha256,
                "protocol_hash": spec.protocol_hash,
                "refit_bundle_sha256": spec.refit_bundle_sha256,
                "subgroup_artifact_sha256": spec.subgroup_artifact_sha256,
                "manifest_sha256": spec.manifest_sha256,
                "device": spec.requested_device,
                "formal_fold9_or_fold10_inference_performed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
