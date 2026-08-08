#!/usr/bin/env python3
"""Create reproducible PTB-XL label, fold, and waveform overview figures."""

# The writable Matplotlib cache must be configured before third-party imports.
# ruff: noqa: E402

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(_PROJECT_ROOT / "artifacts" / "matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ecg_trust.constants import LEADS, PTBXL_VERSION, SUPERCLASSES, TARGET_COLUMNS
from ecg_trust.data.dataset import PTBXLDataset
from ecg_trust.protocol import ExperimentProtocol


def _read_manifest(path: Path) -> pd.DataFrame:
    if path.suffix.casefold() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.casefold() == ".csv":
        return pd.read_csv(path)
    raise ValueError("manifest must be a .parquet or .csv file")


def _plot_counts(manifest: pd.DataFrame, destination: Path) -> None:
    label_counts = manifest.loc[:, list(TARGET_COLUMNS)].sum().to_numpy(dtype=int)
    fold_counts = manifest.groupby("strat_fold", sort=True).size()

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    axes[0].bar(SUPERCLASSES, label_counts, color="#2563eb")
    axes[0].set_title("Diagnostic-superclass positives")
    axes[0].set_ylabel("ECG records")
    axes[0].grid(axis="y", alpha=0.25)
    for index, count in enumerate(label_counts):
        axes[0].text(index, count, f"{count:,}", ha="center", va="bottom", fontsize=9)

    axes[1].bar(fold_counts.index.astype(str), fold_counts.to_numpy(), color="#0f766e")
    axes[1].set_title("Official patient-respecting folds")
    axes[1].set_xlabel("strat_fold")
    axes[1].set_ylabel("ECG records")
    axes[1].grid(axis="y", alpha=0.25)

    figure.suptitle(f"PTB-XL {PTBXL_VERSION}: canonical five-superclass manifest")
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def _plot_waveform(
    manifest: pd.DataFrame,
    dataset_root: Path,
    destination: Path,
    *,
    ecg_id: int | None,
) -> int:
    development = PTBXLDataset(
        manifest,
        dataset_root,
        folds=ExperimentProtocol.canonical().folds_for("train"),
    )
    selected = development.manifest
    if ecg_id is None:
        candidates = np.flatnonzero(selected["label_MI"].to_numpy(dtype=int) == 1)
        position = int(candidates[0]) if len(candidates) else 0
    else:
        matches = np.flatnonzero(selected["ecg_id"].to_numpy(dtype=int) == ecg_id)
        if not len(matches):
            raise ValueError(f"ecg_id {ecg_id} is not present in development folds 1-7")
        position = int(matches[0])

    signal, target = development[position]
    row = selected.iloc[position]
    time_seconds = np.arange(signal.shape[1], dtype=np.float64) / 100.0
    positive_labels = [
        label for label, value in zip(SUPERCLASSES, target.tolist(), strict=True) if value == 1
    ]

    figure, axes = plt.subplots(
        len(LEADS),
        1,
        figsize=(14, 14),
        sharex=True,
        constrained_layout=True,
    )
    waveform = signal.numpy()
    for lead_index, (lead, axis) in enumerate(zip(LEADS, axes, strict=True)):
        axis.plot(time_seconds, waveform[lead_index], color="#111827", linewidth=0.65)
        axis.set_ylabel(lead, rotation=0, labelpad=16)
        axis.grid(alpha=0.18, linewidth=0.5)
    axes[-1].set_xlabel("Time (seconds)")
    figure.suptitle(
        f"PTB-XL ECG {int(row['ecg_id'])} — labels: {', '.join(positive_labels)}\n"
        "Physical 100 Hz signal; research visualization only"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)
    return int(row["ecg_id"])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    project_root = _PROJECT_ROOT
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
        "--output-dir",
        type=Path,
        default=project_root / "reports" / "figures" / "data",
    )
    parser.add_argument("--ecg-id", type=int)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = _read_manifest(args.manifest)
        counts_path = args.output_dir / "ptbxl_label_and_fold_counts.png"
        waveform_path = args.output_dir / "ptbxl_representative_ecg.png"
        _plot_counts(manifest, counts_path)
        selected_id = _plot_waveform(
            manifest,
            args.dataset_root,
            waveform_path,
            ecg_id=args.ecg_id,
        )
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"saved: {counts_path.resolve()}")
    print(f"saved: {waveform_path.resolve()} (ecg_id={selected_id})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
