#!/usr/bin/env python3
"""Run a leakage-safe PTB-XL development experiment on folds 1-7/8."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ecg_trust.experiment_config import ExperimentConfigError, load_experiment_config
from ecg_trust.experiment_runner import DevelopmentRunnerError, run_development_experiment
from ecg_trust.protocol import ProtocolValidationError, load_protocol


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=project_root / "configs" / "train_smoke.yaml",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=project_root / "configs" / "protocol.yaml",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    project_root = Path(__file__).resolve().parents[1]
    try:
        config = load_experiment_config(args.config, base_dir=project_root)
        protocol = load_protocol(args.protocol)
        result = run_development_experiment(config, protocol=protocol)
    except (
        OSError,
        ExperimentConfigError,
        ProtocolValidationError,
        DevelopmentRunnerError,
        ValueError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(f"run directory: {result.run_dir}")
    print(f"completed epochs: {result.completed_epochs}")
    print(f"best fold-8 macro AUROC: {result.best_macro_auroc:.6f}")
    print(f"best checkpoint: {result.best_checkpoint_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
