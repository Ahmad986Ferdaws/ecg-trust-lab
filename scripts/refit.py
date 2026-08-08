#!/usr/bin/env python3
"""Run a legacy or freeze-bound fresh refit on PTB-XL folds 1-8."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ecg_trust.protocol import ProtocolValidationError, load_protocol
from ecg_trust.refit_config import RefitConfigError, load_refit_config
from ecg_trust.refit_runner import FrozenRefitError, run_frozen_refit


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=project_root / "configs" / "refit_resnet_frozen.yaml",
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
        config = load_refit_config(args.config, base_dir=project_root)
        protocol = load_protocol(args.protocol)
        result = run_frozen_refit(config, protocol=protocol)
    except (
        OSError,
        RefitConfigError,
        ProtocolValidationError,
        FrozenRefitError,
        ValueError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(f"refit directory: {result.run_dir}")
    print(f"completed frozen epochs: {result.frozen_epochs}")
    print(f"final checkpoint: {result.final_checkpoint_path}")
    print(
        "diagnostic best training loss: "
        f"{result.best_training_loss:.6f} at epoch {result.best_training_loss_epoch}"
    )
    if result.completion_path is not None:
        print(f"refit completion: {result.completion_path}")
        print(f"completion SHA-256: {result.completion_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
