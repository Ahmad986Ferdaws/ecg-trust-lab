"""Prepare the frozen, development-only Trust Sentinel source policy."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from ecg_trust.source_calibration import (
    prepare_source_calibration,
    verify_clean_git_revision,
)

_DEFAULT_CONFIG = "configs/trust_sentinel_source_calibration_v1.yaml"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a self-hashed, aggregate-only source calibration artifact. "
            "This command never creates a complete release while OOD evidence is pending."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Clean committed repository root (defaults to this repository).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(_DEFAULT_CONFIG),
        help=f"Frozen YAML path relative to the project root (default: {_DEFAULT_CONFIG}).",
    )
    parser.add_argument(
        "--require-complete-release",
        action="store_true",
        help="Fail closed: this preparation cannot claim a complete release without OOD evidence.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.require_complete_release:
        print(
            "COMPLETE_RELEASE_FORBIDDEN: OOD evidence is pending; no preparation was run.",
            file=sys.stderr,
        )
        return 2
    try:
        project_root = arguments.project_root.resolve(strict=True)
        config_path = arguments.config
        if not config_path.is_absolute():
            config_path = project_root / config_path
        config_path = config_path.resolve(strict=True)
        config_path.relative_to(project_root)
        revision = verify_clean_git_revision(project_root)
        result = prepare_source_calibration(
            config_path=config_path,
            project_root=project_root,
            code_revision=revision,
        )
    except Exception:
        print(
            "SOURCE_CALIBRATION_FAILED: inspect the local preflight and immutable output state.",
            file=sys.stderr,
        )
        return 1
    print(
        json.dumps(
            {
                "artifact_sha256": result.artifact_sha256,
                "open_world_status": result.open_world.status,
                "release_ready": result.open_world.release_ready,
                "status": result.status,
            },
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
