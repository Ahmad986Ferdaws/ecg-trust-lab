#!/usr/bin/env python3
"""Compute leakage-safe per-lead normalization from PTB-XL folds 1-7 only."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from ecg_trust.constants import PTBXL_VERSION
from ecg_trust.data.dataset import compute_normalization_stats
from ecg_trust.protocol import FoldRole, load_protocol


def _read_manifest(path: Path) -> pd.DataFrame:
    if path.suffix.casefold() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.casefold() == ".csv":
        return pd.read_csv(path)
    raise ValueError("manifest must be a .parquet or .csv file")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=(
            project_root / "data" / "manifests" / f"ptbxl_superclasses_v{PTBXL_VERSION}.parquet"
        ),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=project_root / "data" / "raw" / "ptb-xl" / PTBXL_VERSION,
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=project_root / "configs" / "protocol.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            project_root
            / "artifacts"
            / "preprocessing"
            / f"ptbxl_v{PTBXL_VERSION}_train_folds_1-7_normalization.json"
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        protocol = load_protocol(args.protocol)
        training_folds = protocol.folds_for(FoldRole.TRAIN)
        manifest = _read_manifest(args.manifest)
        stats = compute_normalization_stats(
            manifest,
            args.dataset_root,
            training_folds=training_folds,
            dataset_version=protocol.dataset_version,
        )
        stats.save(args.output)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"saved: {args.output.resolve()}")
    print(f"records: {stats.provenance.record_count}")
    print(f"samples: {stats.provenance.sample_count}")
    print(f"folds: {stats.provenance.training_folds}")
    print(f"manifest SHA-256: {stats.provenance.manifest_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
