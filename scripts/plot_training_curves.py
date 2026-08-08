#!/usr/bin/env python3
"""Plot aligned development histories for the matched PTB-XL models."""

# The writable Matplotlib cache must be configured before third-party imports.
# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import cast

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(_PROJECT_ROOT / "artifacts" / "matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def _load_history(run_dir: Path) -> list[dict[str, object]]:
    history_path = run_dir / "history.jsonl"
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(history_path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            decoded: object = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON at {history_path}:{line_number}") from error
        if not isinstance(decoded, dict):
            raise ValueError(f"history row {line_number} must be an object")
        rows.append(cast(dict[str, object], decoded))
    if not rows:
        raise ValueError(f"history is empty: {history_path}")
    return rows


def _number(row: dict[str, object], key: str) -> float:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"history field {key!r} must be numeric")
    return float(value)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dirs",
        nargs="*",
        type=Path,
        default=[
            _PROJECT_ROOT / "runs" / "development" / "resnet1d_matched_seed2026",
            _PROJECT_ROOT / "runs" / "development" / "ecg_transformer_matched_seed2026",
        ],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            _PROJECT_ROOT
            / "reports"
            / "figures"
            / "development"
            / "matched_seed2026_training_curves.png"
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        histories = [(run_dir.name, _load_history(run_dir)) for run_dir in args.run_dirs]
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for name, rows in histories:
        epochs = [_number(row, "epoch") + 1 for row in rows]
        axes[0, 0].plot(
            epochs,
            [_number(row, "validation_macro_auroc") for row in rows],
            label=name,
        )
        axes[0, 1].plot(
            epochs,
            [_number(row, "train_loss") for row in rows],
            label=f"{name} train",
        )
        axes[0, 1].plot(
            epochs,
            [_number(row, "validation_loss") for row in rows],
            linestyle="--",
            label=f"{name} validation",
        )
        axes[1, 0].plot(
            epochs,
            [_number(row, "learning_rate") for row in rows],
            label=name,
        )
        axes[1, 1].plot(
            epochs,
            [_number(row, "train_samples_per_second") for row in rows],
            label=name,
        )

    axes[0, 0].set_title("Fold-8 macro AUROC")
    axes[0, 0].set_ylabel("AUROC")
    axes[0, 1].set_title("BCEWithLogits loss")
    axes[0, 1].set_ylabel("Loss")
    axes[1, 0].set_title("Warmup-cosine learning rate")
    axes[1, 0].set_ylabel("Learning rate")
    axes[1, 1].set_title("Training throughput")
    axes[1, 1].set_ylabel("Samples / second")
    for axis in axes.flat:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle("Matched-capacity PTB-XL development runs — seed 2026")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    plt.close(figure)
    print(f"saved: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
