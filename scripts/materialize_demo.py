#!/usr/bin/env python3
"""Materialize the fixed provenance-bound PTB-XL research demo artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from ecg_trust.demo_backend import DemoArtifactError
from ecg_trust.demo_materialization import DemoMaterializationError, materialize_demo
from ecg_trust.protocol import ProtocolValidationError, load_protocol
from ecg_trust.release_gates import ReleaseGateError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create the immutable resnet1d-seed2026, fold-9 coverage-0.8 demo binding. "
            "The member, gate, example fold, and example-selection rule are not CLI options."
        )
    )
    parser.add_argument("--protocol", type=Path, default=Path("configs/protocol.yaml"))
    parser.add_argument("--refit-bundle", type=Path, required=True)
    parser.add_argument("--calibration-bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        protocol = load_protocol(args.protocol)
        result = materialize_demo(
            refit_bundle_path=args.refit_bundle,
            calibration_bundle_path=args.calibration_bundle,
            output_directory=args.output_dir,
            protocol=protocol,
        )
    except (
        DemoArtifactError,
        DemoMaterializationError,
        FileExistsError,
        OSError,
        ProtocolValidationError,
        ReleaseGateError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result.to_dict(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
