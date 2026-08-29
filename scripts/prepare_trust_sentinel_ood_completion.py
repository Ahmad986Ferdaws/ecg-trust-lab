"""Execute the frozen Trust Sentinel OOD-completion protocol once."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from ecg_trust.ood_completion.pipeline import prepare_ood_completion
from ecg_trust.source_calibration import verify_clean_git_revision

_DEFAULT_CONFIG = "configs/trust_sentinel_ood_completion_v1.yaml"


def _absolute_lexical(path: Path) -> Path:
    """Make a path absolute without dereferencing any filesystem component."""

    return Path(os.path.abspath(os.fspath(path)))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Execute the immutable, source-domain OOD-completion protocol on its exact "
            "CUDA runtime. No scientific or runtime overrides are accepted."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=_absolute_lexical(Path(__file__)).parents[1],
        help="Clean committed repository root (defaults to this repository).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(_DEFAULT_CONFIG),
        help=f"Frozen YAML path relative to the project root (default: {_DEFAULT_CONFIG}).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        project_root = _absolute_lexical(arguments.project_root)
        config_path = arguments.config
        if not config_path.is_absolute():
            config_path = project_root / config_path
        config_path = _absolute_lexical(config_path)
        config_path.relative_to(project_root)
        revision = verify_clean_git_revision(project_root)
        result = prepare_ood_completion(
            config_path=config_path,
            project_root=project_root,
            code_revision=revision,
        )
    except Exception:
        print(
            "OOD_COMPLETION_FAILED: inspect the private local preflight "
            "and immutable output state.",
            file=sys.stderr,
        )
        return 1
    print(
        json.dumps(
            {
                "artifact_sha256": result.artifact_sha256,
                "distribution_policy_sha256": result.distribution_policy.artifact_sha256,
                "ood_positive_evaluation": result.ood_positive_evaluation.status,
                "research_bundle_eligible": result.research_bundle_eligible,
                "source_false_rejection_rate": (
                    result.source_validation.record_false_rejection_rate
                ),
                "source_record_support_coverage": (
                    result.source_validation.source_record_support_coverage
                ),
                "source_false_rejection_upper_95": (
                    result.source_validation.cluster_bootstrap.one_sided_upper
                ),
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
