"""Freeze aggregate monitoring provenance and replay windows against it offline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ecg_trust.monitoring.provenance import (
    FrozenMonitoringReference,
    monitoring_config_from_dict,
)
from ecg_trust.monitoring.trust_monitoring import (
    AggregateTelemetryWindow,
    TelemetryValidationError,
    TrustMonitoringConfig,
    _require_mapping,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser(
        "freeze", help="Seal a v1 aggregate reference with its configuration"
    )
    freeze.add_argument("--reference", type=Path, required=True)
    freeze.add_argument("--config", type=Path)
    replay = commands.add_parser("replay", help="Replay a JSON array of v1 windows")
    replay.add_argument("--frozen-reference", type=Path, required=True)
    replay.add_argument("--expected-sha256", required=True)
    replay.add_argument("--windows", type=Path, required=True)
    replay.add_argument("--config", type=Path, help="Optional configuration to check for drift")
    for command in (freeze, replay):
        command.add_argument(
            "--output", type=Path, required=True, help="New output file (must not exist)"
        )
    args = parser.parse_args(argv)
    try:
        config = (
            monitoring_config_from_dict(
                _require_mapping(json.loads(args.config.read_text()), "configuration")
            )
            if args.config is not None
            else None
        )
        if args.command == "freeze":
            window = AggregateTelemetryWindow.from_dict(
                _require_mapping(json.loads(args.reference.read_text()), "reference")
            )
            result = FrozenMonitoringReference(window, config or TrustMonitoringConfig()).to_json()
        else:
            frozen = FrozenMonitoringReference.from_json(
                args.frozen_reference.read_text(), expected_sha256=args.expected_sha256
            )
            rows = json.loads(args.windows.read_text())
            if not isinstance(rows, list):
                raise TelemetryValidationError("replay windows must be a JSON array")
            windows = tuple(
                AggregateTelemetryWindow.from_dict(_require_mapping(row, "window")) for row in rows
            )
            result = frozen.replay_to_json(windows, config=config)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(result)
    except (OSError, ValueError, RecursionError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
