#!/usr/bin/env python
"""Lifecycle CLI for sealed refit, calibration, and one-time final batches."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

from ecg_trust.final_batch import FinalBatchSettings, run_final_batch
from ecg_trust.final_evaluation_spec import (
    FinalEvaluationSpec,
    load_final_evaluation_spec,
)
from ecg_trust.protocol import (
    FINAL_TEST_CONFIRMATION,
    ExperimentProtocol,
    load_protocol,
)
from ecg_trust.release_gates import (
    create_refit_bundle,
    export_fold9_predictions,
    fit_calibration_bundle,
    load_calibration_bundle,
    load_refit_bundle,
    save_calibration_bundle,
    save_refit_bundle,
)


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{context} must be a string-keyed mapping")
    return cast(Mapping[str, object], value)


def _string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a non-empty string")
    return value


def _integer(value: object, context: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{context} must be an integer >= {minimum}")
    return value


def _number(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be numeric")
    return float(value)


def _loaded_evaluation_spec(
    path: Path, *, protocol: ExperimentProtocol
) -> FinalEvaluationSpec:
    return load_final_evaluation_spec(
        path,
        protocol=protocol,
        verify_sources=True,
        verify_runtime=True,
    )


def _release_root(spec: FinalEvaluationSpec) -> Path:
    if spec.path is None:  # pragma: no cover - loaded artifacts always have a path
        raise ValueError("final-evaluation specification must be saved")
    return spec.path.resolve().parent


def _final_settings(spec: FinalEvaluationSpec) -> FinalBatchSettings:
    payload = spec.payload
    evaluation = _mapping(payload["final_evaluation"], "final_evaluation")
    subgroup = _mapping(payload["subgroup_artifact"], "subgroup_artifact")
    return FinalBatchSettings.create(
        output_directory=_release_root(spec) / "fold10_final",
        subgroup_path=Path(_string(subgroup["path"], "subgroup_artifact.path")),
        device=spec.requested_device,
        bf16=True,
        bootstrap_resamples=_integer(
            evaluation["bootstrap_resamples"], "bootstrap_resamples", minimum=2
        ),
        bootstrap_seed=_integer(
            evaluation["bootstrap_base_seed"], "bootstrap_base_seed", minimum=0
        ),
        bootstrap_confidence=_number(
            evaluation["bootstrap_confidence"], "bootstrap_confidence"
        ),
        bootstrap_minimum_valid=_integer(
            evaluation["bootstrap_minimum_valid"],
            "bootstrap_minimum_valid",
            minimum=1,
        ),
        minimum_group_samples=_integer(
            evaluation["minimum_group_samples"],
            "minimum_group_samples",
            minimum=1,
        ),
        minimum_group_patients=_integer(
            evaluation["minimum_group_patients"],
            "minimum_group_patients",
            minimum=1,
        ),
        ece_bins=_integer(evaluation["ece_bins"], "ece_bins", minimum=2),
    )


def _assert_calibration_binds_spec(
    calibration_bundle: object,
    spec: FinalEvaluationSpec,
) -> None:
    provenance = getattr(calibration_bundle, "stage_provenance", None)
    stage = _mapping(provenance, "calibration stage_provenance")
    binding = _mapping(
        stage["final_evaluation_spec"], "final_evaluation_spec binding"
    )
    if spec.path is None or Path(
        _string(binding["path"], "final_evaluation_spec.path")
    ).resolve() != spec.path.resolve():
        raise ValueError(
            "calibration bundle is bound to a different final-evaluation specification"
        )
    if binding["artifact_sha256"] != spec.artifact_sha256:
        raise ValueError(
            "calibration bundle final-evaluation specification hash differs"
        )


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
    export.add_argument("--evaluation-spec", type=Path, required=True)

    calibrate = subparsers.add_parser(
        "fit-calibration",
        help="Fit six independent fold-9 policies and seal one calibration bundle.",
    )
    calibrate.add_argument(
        "--protocol", type=Path, default=Path("configs/protocol.yaml")
    )
    calibrate.add_argument("--refit-bundle", type=Path, required=True)
    calibrate.add_argument("--evaluation-spec", type=Path, required=True)

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
    final.add_argument("--evaluation-spec", type=Path, required=True)
    final.add_argument("--purpose", required=True)
    final.add_argument("--operator", required=True)
    final.add_argument(
        "--confirmation",
        required=True,
        help=f"Must exactly equal: {FINAL_TEST_CONFIRMATION}",
    )
    final.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    protocol = load_protocol(args.protocol)

    if args.command == "seal-refits":
        refit_bundle = create_refit_bundle(args.refit_completion, protocol=protocol)
        path, digest = save_refit_bundle(refit_bundle, args.output)
        print(json.dumps({"path": str(path), "artifact_sha256": digest}))
        return 0
    if args.command == "verify-refits":
        verified_refits = load_refit_bundle(
            args.bundle, protocol=protocol, verify_sources=True
        )
        print(json.dumps(verified_refits.to_payload(), sort_keys=True))
        return 0
    if args.command == "export-fold9":
        evaluation_spec = _loaded_evaluation_spec(
            args.evaluation_spec, protocol=protocol
        )
        paths = export_fold9_predictions(
            args.refit_bundle,
            _release_root(evaluation_spec) / "fold9_predictions",
            protocol=protocol,
            final_evaluation_spec_path=args.evaluation_spec,
        )
        print(json.dumps({key: str(value) for key, value in paths.items()}))
        return 0
    if args.command == "fit-calibration":
        evaluation_spec = _loaded_evaluation_spec(
            args.evaluation_spec, protocol=protocol
        )
        release_root = _release_root(evaluation_spec)
        verified_refits = load_refit_bundle(
            args.refit_bundle, protocol=protocol, verify_sources=True
        )
        predictions = {
            member.member_id: (
                release_root / "fold9_predictions" / f"{member.member_id}.fold9.npz"
            )
            for member in verified_refits.members
        }
        calibration_bundle = fit_calibration_bundle(
            args.refit_bundle,
            predictions,
            release_root / "calibration",
            protocol=protocol,
            fold9_export_completion_path=(
                release_root
                / "fold9_predictions"
                / "fold9-export-completion.json"
            ),
        )
        path, digest = save_calibration_bundle(
            calibration_bundle, release_root / "calibration_bundle.json"
        )
        print(json.dumps({"path": str(path), "artifact_sha256": digest}))
        return 0
    if args.command == "verify-calibration":
        verified_calibration = load_calibration_bundle(
            args.bundle, protocol=protocol, verify_sources=True
        )
        print(json.dumps(verified_calibration.to_payload(), sort_keys=True))
        return 0
    if args.command == "run-final":
        evaluation_spec = _loaded_evaluation_spec(
            args.evaluation_spec, protocol=protocol
        )
        verified_calibration = load_calibration_bundle(
            args.calibration_bundle,
            protocol=protocol,
            verify_sources=True,
        )
        _assert_calibration_binds_spec(verified_calibration, evaluation_spec)
        settings = _final_settings(evaluation_spec)
        result = run_final_batch(
            refit_bundle_path=args.refit_bundle,
            calibration_bundle_path=args.calibration_bundle,
            settings=settings,
            ledger_path=None,
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
