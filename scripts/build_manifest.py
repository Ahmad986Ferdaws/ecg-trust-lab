#!/usr/bin/env python3
"""Build the deterministic five-superclass PTB-XL manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ecg_trust.constants import PTBXL_VERSION
from ecg_trust.data.manifest import ManifestError, build_manifest, write_manifest_artifacts


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=project_root / "data" / "raw" / "ptb-xl" / PTBXL_VERSION,
        help="verified PTB-XL dataset root (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "data" / "manifests",
        help="artifact directory (default: %(default)s)",
    )
    parser.add_argument(
        "--include-unlabeled",
        action="store_true",
        help="retain records with no mapped diagnostic superclass",
    )
    parser.add_argument(
        "--skip-file-checks",
        action="store_true",
        help="skip .hea/.dat existence checks (development only)",
    )
    parser.add_argument(
        "--allow-noncanonical-counts",
        action="store_true",
        help="disable official record/patient/label count gates (fixtures only)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest, summary = build_manifest(
            args.dataset_root,
            strict_official_counts=not args.allow_noncanonical_counts,
            verify_files=not args.skip_file_checks,
            include_unlabeled=args.include_unlabeled,
        )
        artifacts = write_manifest_artifacts(manifest, summary, args.output_dir)
    except (ManifestError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"manifest rows: {len(manifest)}")
    print(f"CSV:     {artifacts.csv_path}  sha256={artifacts.csv_sha256}")
    print(f"Parquet: {artifacts.parquet_path}  sha256={artifacts.parquet_sha256}")
    print(f"Summary: {artifacts.summary_path}  sha256={artifacts.summary_sha256}")
    print(f"Hashes:  {artifacts.checksums_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
