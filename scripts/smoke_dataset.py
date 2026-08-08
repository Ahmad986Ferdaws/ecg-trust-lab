#!/usr/bin/env python3
"""Load real PTB-XL records through the strict dataset contract."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

from ecg_trust.constants import PTBXL_VERSION
from ecg_trust.data.dataset import PTBXLDataset
from ecg_trust.protocol import load_protocol


def _read_manifest(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix.casefold() == ".parquet" else pd.read_csv(path)


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
    parser.add_argument("--folds", nargs="+", type=int, default=[1, 8, 9])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = _read_manifest(args.manifest)
        protocol = load_protocol(args.protocol)
        for fold in args.folds:
            dataset = PTBXLDataset(
                manifest,
                args.dataset_root,
                folds=fold,
                protocol=protocol,
            )
            signal, target = dataset[0]
            if signal.shape != (12, 1_000) or target.shape != (5,):
                raise RuntimeError(
                    f"fold {fold} returned unexpected shapes {signal.shape}, {target.shape}"
                )
            if signal.dtype != torch.float32 or not torch.isfinite(signal).all():
                raise RuntimeError(f"fold {fold} returned an invalid signal tensor")
            print(
                f"fold {fold}: records={len(dataset)}, first={dataset.record_path(0)}, "
                f"shape={tuple(signal.shape)}, labels={target.int().tolist()}"
            )
    except (OSError, PermissionError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
