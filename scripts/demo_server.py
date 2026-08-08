#!/usr/bin/env python3
"""Serve the local provenance-checked PTB-XL ECG research demo."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import uvicorn

from ecg_trust.demo_app import DemoAppConfig, DemoWebError, create_app
from ecg_trust.demo_backend import DemoArtifactError, DemoInferenceBackend


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--resolved-config", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--decision-policy", type=Path, required=True)
    parser.add_argument("--examples", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        backend = DemoInferenceBackend.load(
            checkpoint_path=args.checkpoint,
            resolved_config_path=args.resolved_config,
            normalization_path=args.normalization,
            decision_policy_path=args.decision_policy,
        )
        config = (
            DemoAppConfig()
            if args.examples is None
            else DemoAppConfig.with_example_manifest(args.examples)
        )
        app = create_app(backend=backend, config=config)
    except (OSError, DemoArtifactError, DemoWebError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    if not 1 <= args.port <= 65535:
        print("error: --port must be in [1, 65535]", file=sys.stderr)
        return 1
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
