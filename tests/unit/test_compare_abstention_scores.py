from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts import compare_abstention_scores as tool


def _save(path: Path, **arrays: np.ndarray) -> Path:
    np.savez(path, **arrays)
    return path


def test_synthetic_demo_ranks_every_score_by_excess_aurc(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert tool.main(["--synthetic-demo"]) == 0
    report = json.loads(capsys.readouterr().out)

    names = [row["score"] for row in report["scores"]]
    excess = [row["excess_aurc"] for row in report["scores"]]
    assert report["source"] == "synthetic demo (not model output)"
    assert set(names) == {
        "mean_label_entropy (frozen gate score)",
        "worst_label_entropy",
        "ensemble_total_mean",
        "ensemble_epistemic_mean",
        "ensemble_epistemic_max",
    }
    assert excess == sorted(excess)
    for row in report["scores"]:
        assert row["aurc"] >= row["oracle_aurc"]
        assert 0.0 <= row["augrc"] <= report["exact_match_error_rate"]


def test_prediction_file_round_trip_without_members(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    probabilities, targets, _ = tool.synthetic_demo(seed=1)
    path = _save(
        tmp_path / "fold9.npz",
        probabilities=probabilities,
        targets=targets,
        fold_ids=np.full(2000, 9),
    )

    assert tool.main(["--predictions", str(path)]) == 0
    report = json.loads(capsys.readouterr().out)

    assert report["source"] == "fold9.npz"
    assert len(report["scores"]) == 2


@pytest.mark.parametrize(
    ("arrays", "message"),
    [
        ({"probabilities": np.full((4, 5), 0.5)}, "missing"),
        (
            {
                "probabilities": np.full((4, 5), 0.5),
                "targets": np.zeros((4, 5), dtype=np.int64),
                "fold_ids": np.array([9, 9, 10, 9]),
            },
            "fold-10",
        ),
        (
            {
                "probabilities": np.full((4, 5), 0.5),
                "targets": np.zeros((4, 5), dtype=np.int64),
                "thresholds": np.full(4, 0.5),
            },
            "thresholds",
        ),
        (
            {
                "probabilities": np.full((4, 5), 0.5),
                "targets": np.zeros((4, 5), dtype=np.int64),
                "member_probabilities": np.full((2, 3, 5), 0.5),
            },
            "member_probabilities",
        ),
        (
            {"probabilities": np.full((4, 5), 1.5), "targets": np.zeros((4, 5), dtype=np.int64)},
            "probabilities",
        ),
    ],
)
def test_unsafe_or_malformed_files_exit_with_an_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], arrays: dict[str, np.ndarray], message: str
) -> None:
    path = _save(tmp_path / "bad.npz", **arrays)

    assert tool.main(["--predictions", str(path)]) == 2
    assert message in capsys.readouterr().err


def test_missing_file_is_reported(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert tool.main(["--predictions", str(tmp_path / "absent.npz")]) == 2
    assert "error" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ({"fold_ids": np.array([9, 9])}, "one entry per record"),
        ({"member_probabilities": np.full((2, 4, 5), 1.5)}, "[0, 1]"),
        ({"thresholds": np.full(5, 2.0)}, "thresholds"),
    ],
)
def test_misaligned_folds_bad_members_and_thresholds_are_refused(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    extra: dict[str, np.ndarray],
    message: str,
) -> None:
    path = _save(
        tmp_path / "bad.npz",
        probabilities=np.full((4, 5), 0.5),
        targets=np.zeros((4, 5), dtype=np.int64),
        **extra,
    )

    assert tool.main(["--predictions", str(path)]) == 2
    assert message in capsys.readouterr().err
