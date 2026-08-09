#!/usr/bin/env python
"""Freeze label-free PTB-XL fold-10 subgroup metadata before final opening."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from ecg_trust.protocol import load_protocol
from ecg_trust.release_gates import load_refit_bundle
from ecg_trust.subgroup_artifact import (
    build_subgroup_artifact,
    load_subgroup_artifact,
    save_subgroup_artifact,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create the immutable sex/age-band artifact from non-label PTB-XL "
            "metadata bound to the sealed refit bundle."
        )
    )
    parser.add_argument(
        "--protocol", type=Path, default=Path("configs/protocol.yaml")
    )
    parser.add_argument("--refit-bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    protocol = load_protocol(args.protocol)
    bundle = load_refit_bundle(
        args.refit_bundle, protocol=protocol, verify_sources=True
    )
    manifest_paths = {member.manifest_path.resolve() for member in bundle.members}
    if len(manifest_paths) != 1:
        raise RuntimeError("refit bundle members do not share one manifest path")
    artifact = build_subgroup_artifact(
        manifest_paths.pop(),
        protocol=protocol,
        expected_manifest_sha256=bundle.manifest_sha256,
    )
    path, digest = save_subgroup_artifact(artifact, args.output)
    verified = load_subgroup_artifact(
        path,
        protocol=protocol,
        expected_manifest_sha256=bundle.manifest_sha256,
        verify_source=True,
    )
    print(
        json.dumps(
            {
                "path": str(path),
                "artifact_sha256": digest,
                "records": verified.record_count,
                "patients": verified.patient_count,
                "folds": [10],
                "attributes": ["sex", "age_band"],
                "diagnostic_target_columns_read": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
