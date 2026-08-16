#!/usr/bin/env python3
"""Run the frozen, identifier-separated SPH external-transport study."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from ecg_trust.audit_artifacts import AuditArtifactError
from ecg_trust.audit_runtime import AuditRuntimeError
from ecg_trust.data.sph import SPHMetadataValidationError, SPHRecordValidationError
from ecg_trust.protocol import ProtocolValidationError
from ecg_trust.sph_transport import (
    SPHTransportError,
    load_sph_transport_spec,
    run_sph_transport,
)
from ecg_trust.sph_transport_metrics import SPHTransportMetricsError


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Run the preregistered SPH exploratory external-transport stress test. "
            "All scientific settings come from the frozen YAML; no fitting or "
            "scientific command-line overrides are available."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=project_root / "configs" / "external_transport_sph_frozen.yaml",
        help="Path to the frozen SPH protocol YAML.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        spec = load_sph_transport_spec(args.config)
        result = run_sph_transport(
            spec,
            progress=lambda message: print(message, file=sys.stderr, flush=True),
        )
    except (
        AuditArtifactError,
        AuditRuntimeError,
        FileExistsError,
        OSError,
        ProtocolValidationError,
        SPHMetadataValidationError,
        SPHRecordValidationError,
        SPHTransportError,
        SPHTransportMetricsError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result.to_dict(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
