#!/usr/bin/env python3
"""Export immutable PTB-XL predictions from a validated training checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ecg_trust.prediction_export import (
    PredictionExportRequest,
    export_checkpoint_predictions,
)
from ecg_trust.protocol import (
    FINAL_TEST_CONFIRMATION,
    FoldRole,
    authorize_final_test_access,
    load_protocol,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--resolved-config", type=Path, required=True)
    parser.add_argument("--run-metadata", type=Path)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=project_root / "configs" / "protocol.yaml",
    )
    parser.add_argument(
        "--role",
        choices=(
            FoldRole.MODEL_SELECTION.value,
            FoldRole.CALIBRATION.value,
            FoldRole.FINAL_TEST.value,
        ),
        required=True,
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--normalization", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--bf16",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--final-test-purpose")
    parser.add_argument(
        "--final-test-confirmation",
        help=f"required verbatim for fold 10: {FINAL_TEST_CONFIRMATION}",
    )
    args = parser.parse_args(argv)
    if args.role == FoldRole.FINAL_TEST.value:
        if args.final_test_purpose is None or args.final_test_confirmation is None:
            parser.error(
                "final_test requires --final-test-purpose and --final-test-confirmation"
            )
    elif args.final_test_purpose is not None or args.final_test_confirmation is not None:
        parser.error("final-test authorization options are valid only for final_test")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        protocol = load_protocol(args.protocol)
        role = FoldRole(args.role)
        token = (
            authorize_final_test_access(
                protocol,
                purpose=args.final_test_purpose,
                confirmation=args.final_test_confirmation,
            )
            if role is FoldRole.FINAL_TEST
            else None
        )
        request = PredictionExportRequest(
            checkpoint_path=args.checkpoint,
            resolved_config_path=args.resolved_config,
            run_metadata_path=args.run_metadata,
            manifest_path=args.manifest,
            dataset_root=args.dataset_root,
            normalization_path=args.normalization,
            output_path=args.output,
            fold_role=role,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            persistent_workers=args.persistent_workers,
            device=args.device,
            bf16=args.bf16,
        )
        result = export_checkpoint_predictions(
            request,
            protocol=protocol,
            test_access=token,
        )
    except (OSError, PermissionError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
