#!/usr/bin/env python3
"""Create reproducible PTB-XL label, fold, and waveform overview figures."""

# The writable Matplotlib cache must be configured before third-party imports.
# ruff: noqa: E402

from __future__ import annotations

import argparse
import os
import sys
import tempfile
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


def _validated_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    """Validate count-bearing metadata without opening any waveform."""

    required = {"ecg_id", "patient_id", "strat_fold", "record_path", *TARGET_COLUMNS}
    if manifest.empty or not manifest.columns.is_unique:
        raise ValueError("overview manifest must be non-empty with unique columns")
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"overview manifest is missing required columns: {sorted(missing)}")
    validated = manifest.copy()
    for column in ("ecg_id", "patient_id", "strat_fold", *TARGET_COLUMNS):
        try:
            numeric = pd.to_numeric(manifest[column], errors="raise").to_numpy()
            if np.iscomplexobj(numeric):
                raise ValueError("complex values are forbidden")
            integer_dtype = numeric.dtype.kind in {"i", "u"}
            values = numeric if integer_dtype else np.asarray(numeric, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"overview column {column!r} must contain real numbers") from error
        if not np.isfinite(values).all() or (
            not integer_dtype and not np.equal(values, np.floor(values)).all()
        ):
            raise ValueError(f"overview column {column!r} must contain finite integers")
        if column in TARGET_COLUMNS:
            if not np.isin(values, (0, 1)).all():
                raise ValueError("overview targets must contain only 0 or 1")
        elif (
            numeric.dtype.kind == "b"
            or (values < 1).any()
            or (numeric.dtype.kind == "u" and (values > np.iinfo(np.int64).max).any())
            or (not integer_dtype and (values >= 2**63).any())
        ):
            raise ValueError(f"overview column {column!r} must contain positive 64-bit integers")
        if column == "strat_fold" and (values > 10).any():
            raise ValueError("overview folds must be in 1-10")
        validated[column] = values.astype(np.int64)
    if validated["ecg_id"].duplicated().any():
        raise ValueError("overview ECG identifiers must be unique")
    if (validated.groupby("patient_id")["strat_fold"].nunique() > 1).any():
        raise ValueError("overview patients must not span multiple folds")
    if (validated.loc[:, list(TARGET_COLUMNS)].sum(axis=1) == 0).any():
        raise ValueError("overview records must have at least one positive target")
    return validated


def _plot_counts(manifest: pd.DataFrame, destination: Path) -> None:
    label_counts = manifest.loc[:, list(TARGET_COLUMNS)].sum().to_numpy(dtype=int)
    fold_counts = manifest.groupby("strat_fold", sort=True).size()

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    try:
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
    finally:
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
    try:
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
    finally:
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
        manifest = _validated_manifest(_read_manifest(args.manifest))
        counts_path = args.output_dir / "ptbxl_label_and_fold_counts.png"
        waveform_path = args.output_dir / "ptbxl_representative_ecg.png"
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".overview-", dir=args.output_dir) as staging:
            staged_counts = Path(staging) / counts_path.name
            staged_waveform = Path(staging) / waveform_path.name
            selected_id = _plot_waveform(
                manifest,
                args.dataset_root,
                staged_waveform,
                ecg_id=args.ecg_id,
            )
            _plot_counts(manifest, staged_counts)
            os.replace(staged_counts, counts_path)
            os.replace(staged_waveform, waveform_path)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"saved: {counts_path.resolve()}")
    print(f"saved: {waveform_path.resolve()} (ecg_id={selected_id})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
