#!/usr/bin/env python3
"""Compare abstention (uncertainty) scores on one development prediction file.

Reads an ``.npz`` with:

* ``probabilities``: ``[records, 5]`` calibrated probabilities (canonical order);
* ``targets``: ``[records, 5]`` binary labels;
* optional ``member_probabilities``: ``[members, records, 5]`` for ensemble scores;
  ``probabilities`` must then be their mean, so every score describes the same
  predictions whose losses are reported;
* ``fold_ids``: ``[records]`` PTB-XL folds; every row must be fold 9, the only
  fold the research blueprint (section 9.3) allows for abstention-score
  selection. Training (1-7), model-selection (8), and sealed (10) rows are
  refused. Fold IDs can't identify spent one-shot cohorts inside fold 9, so
  never pass rows from one (such as the source-support cohort C);
* optional ``thresholds``: ``[5]`` decision thresholds (default 0.5).

Following the research blueprint (section 9.3), the primary per-record loss is
mean per-label binary log loss and the secondary loss is thresholded Hamming
loss; a record-level correct/incorrect flag is too coarse. Every score is
ranked from least to most uncertain and summarized with AURC, oracle AURC,
E-AURC, and AUGRC per loss (lower is better), without fitting anything.

    uv run --no-sync python scripts/compare_abstention_scores.py --predictions fold9.npz
    uv run --no-sync python scripts/compare_abstention_scores.py --synthetic-demo
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import zipfile
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ecg_trust.ensemble_uncertainty import (
    EnsembleUncertaintyError,
    decompose_ensemble_uncertainty,
)
from ecg_trust.evaluation import EvaluationValidationError, validate_multilabel_arrays
from ecg_trust.post_analysis import mean_normalized_binary_entropy
from ecg_trust.protocol import CALIBRATION_FOLDS, FINAL_TEST_FOLDS
from ecg_trust.selective_metrics import SelectiveMetricError, augrc, excess_aurc

FloatArray = NDArray[np.float64]


class ComparisonError(ValueError):
    """Raised when the prediction file cannot be compared safely."""


def _binary_entropy_bits(probabilities: FloatArray) -> FloatArray:
    epsilon = np.finfo(np.float64).eps
    clipped = np.clip(probabilities, epsilon, 1.0 - epsilon)
    entropy = -(clipped * np.log(clipped) + (1.0 - clipped) * np.log(1.0 - clipped))
    return np.asarray(entropy / math.log(2.0), dtype=np.float64)


def scores_for(
    probabilities: FloatArray, member_probabilities: object | None
) -> dict[str, FloatArray]:
    """Named per-record uncertainty scores; higher means more uncertain."""

    scores: dict[str, FloatArray] = {
        "mean_label_entropy (frozen gate score)": mean_normalized_binary_entropy(probabilities),
        "worst_label_entropy": _binary_entropy_bits(probabilities).max(axis=1),
    }
    if member_probabilities is not None:
        ensemble = decompose_ensemble_uncertainty(member_probabilities)  # type: ignore[arg-type]
        if ensemble.total.shape != probabilities.shape:
            raise ComparisonError("member_probabilities must be [members, records, 5]")
        member_mean = np.asarray(member_probabilities, dtype=np.float64).mean(axis=0)
        if not np.allclose(member_mean, probabilities, rtol=0.0, atol=1e-6):
            raise ComparisonError("probabilities must equal the mean of member_probabilities")
        scores["ensemble_epistemic_mean"] = ensemble.per_record("epistemic")
        scores["ensemble_epistemic_max"] = ensemble.per_record("epistemic", reduction="max")
    return scores


def record_losses(
    probabilities: FloatArray, targets: NDArray[np.int64], thresholds: FloatArray
) -> dict[str, FloatArray]:
    """Planned per-record losses: mean binary log loss (primary) and Hamming loss."""

    epsilon = np.finfo(np.float64).eps
    clipped = np.clip(probabilities, epsilon, 1.0 - epsilon)
    outcome = targets.astype(np.float64)
    log_loss = -(outcome * np.log(clipped) + (1.0 - outcome) * np.log1p(-clipped)).mean(axis=1)
    hamming = ((probabilities >= thresholds[None, :]) != targets.astype(bool)).mean(axis=1)
    return {
        "mean_binary_log_loss": np.asarray(log_loss, dtype=np.float64),
        "thresholded_hamming_loss": np.asarray(hamming, dtype=np.float64),
    }


def compare(
    probabilities: FloatArray,
    targets: NDArray[np.int64],
    *,
    thresholds: FloatArray,
    member_probabilities: object | None = None,
) -> dict[str, object]:
    losses = record_losses(probabilities, targets, thresholds)
    scores = scores_for(probabilities, member_probabilities)
    results: dict[str, object] = {}
    for loss_name, loss in losses.items():
        rows = []
        for name, score in scores.items():
            summary = excess_aurc(loss, score)
            rows.append(
                {
                    "score": name,
                    "aurc": summary.aurc,
                    "oracle_aurc": summary.oracle_aurc,
                    "excess_aurc": summary.excess_aurc,
                    "augrc": augrc(loss, score),
                }
            )
        rows.sort(key=lambda row: float(row["excess_aurc"]))  # type: ignore[arg-type]
        results[loss_name] = {"mean_loss": float(loss.mean()), "scores": rows}
    return {
        "records": int(probabilities.shape[0]),
        "primary_loss": "mean_binary_log_loss",
        "losses": results,
    }


def load(path: Path) -> tuple[FloatArray, NDArray[np.int64], FloatArray, object | None]:
    with np.load(path, allow_pickle=False) as payload:
        arrays = {name: payload[name] for name in payload.files}
    missing = {"probabilities", "targets", "fold_ids"} - set(arrays)
    if missing:
        raise ComparisonError(f"prediction file is missing {sorted(missing)}")
    targets, probabilities = validate_multilabel_arrays(arrays["targets"], arrays["probabilities"])
    folds = arrays["fold_ids"]
    if folds.shape != (probabilities.shape[0],) or not np.issubdtype(folds.dtype, np.integer):
        raise ComparisonError("fold_ids must be integers with one entry per record")
    if np.isin(folds, FINAL_TEST_FOLDS).any():
        raise ComparisonError("fold-10 rows are sealed; compare development predictions only")
    if not np.isin(folds, CALIBRATION_FOLDS).all():
        raise ComparisonError("fold_ids must all be fold 9, the abstention-selection fold")
    raw_thresholds = np.asarray(arrays.get("thresholds", np.full(5, 0.5)))
    if (
        raw_thresholds.dtype == np.object_
        or np.issubdtype(raw_thresholds.dtype, np.bool_)
        or np.iscomplexobj(raw_thresholds)
        or not np.issubdtype(raw_thresholds.dtype, np.number)
    ):
        raise ComparisonError("thresholds must be five finite values in [0, 1]")
    thresholds = np.asarray(raw_thresholds, dtype=np.float64)
    if (
        thresholds.shape != (5,)
        or not np.isfinite(thresholds).all()
        or np.any((thresholds < 0.0) | (thresholds > 1.0))
    ):
        raise ComparisonError("thresholds must be five finite values in [0, 1]")
    # Leave member arrays unconverted so the ensemble validator sees their dtype.
    return probabilities, targets.astype(np.int64), thresholds, arrays.get("member_probabilities")


def synthetic_demo(seed: int = 0) -> tuple[FloatArray, NDArray[np.int64], FloatArray]:
    """Clearly synthetic records: three members that disagree more on hard cases."""

    rng = np.random.default_rng(seed)
    count = 2_000
    targets = (rng.uniform(size=(count, 5)) < 0.25).astype(np.int64)
    hardness = rng.uniform(0.2, 2.5, size=(count, 1))
    base = np.where(targets == 1, 2.0, -2.0) / hardness
    members = np.stack(
        [
            1.0 / (1.0 + np.exp(-(base + rng.normal(0.0, 0.8, size=base.shape) * hardness)))
            for _ in range(3)
        ]
    )
    return members.mean(axis=0), targets, members


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--predictions", type=Path)
    source.add_argument("--synthetic-demo", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.synthetic_demo:
            probabilities, targets, members = synthetic_demo()
            report = compare(
                probabilities, targets, thresholds=np.full(5, 0.5), member_probabilities=members
            )
            report["source"] = "synthetic demo (not model output)"
        else:
            probabilities, targets, thresholds, member_array = load(args.predictions)
            report = compare(
                probabilities, targets, thresholds=thresholds, member_probabilities=member_array
            )
            report["source"] = args.predictions.name
    except (
        ComparisonError,
        EnsembleUncertaintyError,
        EvaluationValidationError,
        SelectiveMetricError,
        OSError,
        ValueError,
        EOFError,
        zipfile.BadZipFile,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
