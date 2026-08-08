#!/usr/bin/env python
"""Lifecycle CLI for sealed refit, calibration, and one-time final batches."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from ecg_trust.final_batch import FinalBatchSettings, run_final_batch
from ecg_trust.protocol import FINAL_TEST_CONFIRMATION, load_protocol
from ecg_trust.release_gates import (
    create_refit_bundle,
    export_fold9_predictions,
    fit_calibration_bundle,
    load_calibration_bundle,
    load_refit_bundle,
    save_calibration_bundle,
    save_refit_bundle,
)


def _member_paths(values: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        member_id, separator, raw_path = value.partition("=")
        if not separator or not member_id.strip() or not raw_path.strip():
            raise argparse.ArgumentTypeError(
                "member paths must use MEMBER_ID=PATH"
            )
        if member_id in result:
            raise argparse.ArgumentTypeError(f"duplicate member ID {member_id!r}")
        result[member_id] = Path(raw_path)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Seal six-member PTB-XL release stages without post-freeze tuning."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    seal = subparsers.add_parser(
        "seal-refits", help="Verify and seal exactly six completed folds-1-8 refits."
    )
    seal.add_argument("--protocol", type=Path, default=Path("configs/protocol.yaml"))
    seal.add_argument(
        "--refit-completion", type=Path, action="append", required=True
    )
    seal.add_argument("--output", type=Path, required=True)

    verify_refits = subparsers.add_parser(
        "verify-refits", help="Re-hash a saved refit bundle and all bound source files."
    )
    verify_refits.add_argument(
        "--protocol", type=Path, default=Path("configs/protocol.yaml")
    )
    verify_refits.add_argument("--bundle", type=Path, required=True)

    export = subparsers.add_parser(
        "export-fold9",
        help="Export the complete fold-9 batch after the refit gate passes.",
    )
    export.add_argument("--protocol", type=Path, default=Path("configs/protocol.yaml"))
    export.add_argument("--refit-bundle", type=Path, required=True)
    export.add_argument("--output-dir", type=Path, required=True)
    export.add_argument("--batch-size", type=int)
    export.add_argument("--num-workers", type=int)
    export.add_argument("--device", default="auto")
    export.add_argument("--no-bf16", action="store_true")

    calibrate = subparsers.add_parser(
        "fit-calibration",
        help="Fit six independent fold-9 policies and seal one calibration bundle.",
    )
    calibrate.add_argument(
        "--protocol", type=Path, default=Path("configs/protocol.yaml")
    )
    calibrate.add_argument("--refit-bundle", type=Path, required=True)
    calibrate.add_argument(
        "--prediction",
        action="append",
        required=True,
        metavar="MEMBER_ID=PATH",
    )
    calibrate.add_argument("--decision-output-dir", type=Path, required=True)
    calibrate.add_argument("--bundle-output", type=Path, required=True)
    calibrate.add_argument("--coverage", type=float, action="append")

    verify_calibration = subparsers.add_parser(
        "verify-calibration",
        help="Re-hash a calibration bundle and all six fold-9 policy sources.",
    )
    verify_calibration.add_argument(
        "--protocol", type=Path, default=Path("configs/protocol.yaml")
    )
    verify_calibration.add_argument("--bundle", type=Path, required=True)

    final = subparsers.add_parser(
        "run-final",
        help=(
            "Create the persistent opening ledger, then run the exact six-member "
            "fold-10 batch. No calibration or threshold fitting occurs."
        ),
    )
    final.add_argument("--protocol", type=Path, default=Path("configs/protocol.yaml"))
    final.add_argument("--refit-bundle", type=Path, required=True)
    final.add_argument("--calibration-bundle", type=Path, required=True)
    final.add_argument("--subgroups", type=Path, required=True)
    final.add_argument("--output-dir", type=Path, required=True)
    final.add_argument("--ledger", type=Path, required=True)
    final.add_argument("--purpose", required=True)
    final.add_argument("--operator", required=True)
    final.add_argument(
        "--confirmation",
        required=True,
        help=f"Must exactly equal: {FINAL_TEST_CONFIRMATION}",
    )
    final.add_argument("--resume", action="store_true")
    final.add_argument("--batch-size", type=int)
    final.add_argument("--num-workers", type=int)
    final.add_argument("--device", default="auto")
    final.add_argument("--no-bf16", action="store_true")
    final.add_argument("--bootstrap-resamples", type=int, default=1_000)
    final.add_argument("--bootstrap-seed", type=int, default=20_260_808)
    final.add_argument("--bootstrap-confidence", type=float, default=0.95)
    final.add_argument("--bootstrap-minimum-valid", type=int)
    final.add_argument("--minimum-group-samples", type=int, default=30)
    final.add_argument("--minimum-group-patients", type=int, default=20)
    final.add_argument("--ece-bins", type=int, default=15)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    protocol = load_protocol(args.protocol)

    if args.command == "seal-refits":
        bundle = create_refit_bundle(args.refit_completion, protocol=protocol)
        path, digest = save_refit_bundle(bundle, args.output)
        print(json.dumps({"path": str(path), "artifact_sha256": digest}))
        return 0
    if args.command == "verify-refits":
        bundle = load_refit_bundle(args.bundle, protocol=protocol, verify_sources=True)
        print(json.dumps(bundle.to_payload(), sort_keys=True))
        return 0
    if args.command == "export-fold9":
        paths = export_fold9_predictions(
            args.refit_bundle,
            args.output_dir,
            protocol=protocol,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device,
            bf16=not args.no_bf16,
        )
        print(json.dumps({key: str(value) for key, value in paths.items()}))
        return 0
    if args.command == "fit-calibration":
        predictions = _member_paths(args.prediction)
        bundle = fit_calibration_bundle(
            args.refit_bundle,
            predictions,
            args.decision_output_dir,
            protocol=protocol,
            coverage_targets=args.coverage or (1.0, 0.9, 0.8, 0.7, 0.5),
        )
        path, digest = save_calibration_bundle(bundle, args.bundle_output)
        print(json.dumps({"path": str(path), "artifact_sha256": digest}))
        return 0
    if args.command == "verify-calibration":
        bundle = load_calibration_bundle(
            args.bundle, protocol=protocol, verify_sources=True
        )
        print(json.dumps(bundle.to_payload(), sort_keys=True))
        return 0
    if args.command == "run-final":
        settings = FinalBatchSettings.create(
            output_directory=args.output_dir,
            subgroup_path=args.subgroups,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device,
            bf16=not args.no_bf16,
            bootstrap_resamples=args.bootstrap_resamples,
            bootstrap_seed=args.bootstrap_seed,
            bootstrap_confidence=args.bootstrap_confidence,
            bootstrap_minimum_valid=args.bootstrap_minimum_valid,
            minimum_group_samples=args.minimum_group_samples,
            minimum_group_patients=args.minimum_group_patients,
            ece_bins=args.ece_bins,
        )
        result = run_final_batch(
            refit_bundle_path=args.refit_bundle,
            calibration_bundle_path=args.calibration_bundle,
            settings=settings,
            ledger_path=args.ledger,
            protocol=protocol,
            purpose=args.purpose,
            operator=args.operator,
            confirmation=args.confirmation,
            resume=args.resume,
        )
        print(
            json.dumps(
                {
                    "ledger_path": str(result.ledger_path),
                    "batch_summary_path": str(result.batch_summary_path),
                    "paired_manifest_path": str(result.paired_manifest_path),
                    "batch_sha256": result.batch_sha256,
                },
                sort_keys=True,
            )
        )
        return 0
    raise AssertionError(f"unhandled command {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
