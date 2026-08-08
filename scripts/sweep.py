#!/usr/bin/env python3
"""Preflight, inspect, or run the paired folds-1-7/8 development sweep."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ecg_trust.protocol import ProtocolValidationError, load_protocol
from ecg_trust.sweep_config import SweepConfigError, load_equal_budget_pair
from ecg_trust.sweep_runner import (
    ArchitectureSweepResult,
    BestCandidateResult,
    EqualBudgetSweepResult,
    SweepRunnerError,
    preflight_equal_budget_sweeps,
    read_sweep_status,
    run_equal_budget_sweeps,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=("preflight", "status", "run"),
        default="preflight",
        nargs="?",
        help="preflight is read-only and is the safe default; status is also read-only",
    )
    parser.add_argument(
        "--configs",
        type=Path,
        nargs=2,
        metavar=("RESNET_CONFIG", "TRANSFORMER_CONFIG"),
        default=(
            project_root / "configs" / "sweep_resnet_equal_budget.yaml",
            project_root / "configs" / "sweep_transformer_equal_budget.yaml",
        ),
        help="the jointly validated, equal-budget sweep configuration pair",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=project_root / "configs" / "protocol.yaml",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume the exact persisted paired plan; valid only with the run action",
    )
    args = parser.parse_args(argv)
    if args.resume and args.action != "run":
        parser.error("--resume is valid only with the run action")
    return args


def _best_payload(best: BestCandidateResult | None) -> dict[str, object] | None:
    if best is None:
        return None
    return {
        "architecture": best.architecture,
        "candidate_index": best.candidate_index,
        "trial_number": best.trial_number,
        "best_fold8_macro_auroc": best.best_macro_auroc,
        "best_epoch": best.best_epoch,
        "completed_epochs": best.completed_epochs,
        "run_dir": str(best.run_dir),
        "parameters": dict(best.parameters),
        "experiment_config_hash": best.experiment_config_hash,
        "resolved_config_hash": best.resolved_config_hash,
    }


def _study_payload(study: ArchitectureSweepResult) -> dict[str, object]:
    return {
        "architecture": study.architecture,
        "study_name": study.study_name,
        "completed_candidates": study.completed_candidates,
        "failed_attempts": study.failed_attempts,
        "total_attempts": study.total_attempts,
        "budget_complete": study.budget_complete,
        "study_summary_path": str(study.study_summary_path),
        "best": _best_payload(study.best),
    }


def _run_payload(
    result: EqualBudgetSweepResult,
    *,
    resume: bool,
    preflight: dict[str, object],
) -> dict[str, object]:
    return {
        "action": "run",
        "mode": "resume" if resume else "fresh",
        "comparison_id": result.comparison_id,
        "candidate_plan_path": str(result.candidate_plan_path),
        "summary_path": str(result.summary_path),
        "preflight": preflight,
        "studies": [_study_payload(study) for study in result.studies],
    }


def _print_json(payload: dict[str, object]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    project_root = Path(__file__).resolve().parents[1]
    try:
        pair = load_equal_budget_pair(args.configs, base_dir=project_root)
        protocol = load_protocol(args.protocol)
        if args.action == "preflight":
            preflight = preflight_equal_budget_sweeps(pair, protocol=protocol)
            _print_json({"action": "preflight", **preflight.to_dict()})
            return 0
        if args.action == "status":
            status = read_sweep_status(pair, protocol=protocol)
            _print_json({"action": "status", **status})
            return 0

        preflight = preflight_equal_budget_sweeps(pair, protocol=protocol)
        if args.resume and not preflight.candidate_plan_path.is_file():
            raise SweepRunnerError(
                "--resume requires the existing immutable candidate-plan artifact"
            )
        result = run_equal_budget_sweeps(
            pair,
            protocol=protocol,
            resume=args.resume,
        )
        _print_json(
            _run_payload(
                result,
                resume=args.resume,
                preflight=preflight.to_dict(),
            )
        )
        return 0
    except (
        OSError,
        ProtocolValidationError,
        SweepConfigError,
        SweepRunnerError,
        ValueError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
