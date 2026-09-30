from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts import compare_abstention_scores as tool


def _save(path: Path, **arrays: np.ndarray) -> Path:
    np.savez(path, **arrays)
    return path


def test_synthetic_demo_reports_both_planned_losses(capsys: pytest.CaptureFixture[str]) -> None:
    assert tool.main(["--synthetic-demo"]) == 0
    report = json.loads(capsys.readouterr().out)

    assert report["source"] == "synthetic demo (not model output)"
    assert report["primary_loss"] == "mean_binary_log_loss"
    assert set(report["losses"]) == {"mean_binary_log_loss", "thresholded_hamming_loss"}
    for block in report["losses"].values():
        names = [row["score"] for row in block["scores"]]
        excess = [row["excess_aurc"] for row in block["scores"]]
        assert set(names) == {
            "mean_label_entropy (frozen gate score)",
            "worst_label_entropy",
            "ensemble_total_mean",
            "ensemble_epistemic_mean",
            "ensemble_epistemic_max",
        }
        assert excess == sorted(excess)
        for row in block["scores"]:
            assert row["aurc"] >= row["oracle_aurc"]
            assert 0.0 <= row["augrc"] <= block["mean_loss"]


def test_losses_distinguish_one_wrong_label_from_five() -> None:
    probabilities = np.array([[0.9, 0.1, 0.1, 0.1, 0.1], [0.1, 0.9, 0.9, 0.9, 0.9]])
    targets = np.array([[0, 0, 0, 0, 0], [1, 0, 0, 0, 0]])

    losses = tool.record_losses(probabilities, targets, np.full(5, 0.5))

    np.testing.assert_allclose(losses["thresholded_hamming_loss"], [0.2, 1.0])
    assert losses["mean_binary_log_loss"][1] > losses["mean_binary_log_loss"][0]


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
    assert len(report["losses"]["mean_binary_log_loss"]["scores"]) == 2


def test_missing_file_is_reported(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert tool.main(["--predictions", str(tmp_path / "absent.npz")]) == 2
    assert "error" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ({"fold_ids": None}, "missing"),
        ({"fold_ids": np.array([9, 9])}, "one entry per record"),
        ({"fold_ids": np.array([9.0, 9.0, 9.0, 9.0])}, "integers"),
        ({"fold_ids": np.array([9, 9, 10, 9])}, "fold-10"),
        ({"fold_ids": np.array([9, 9, 11, 9])}, "development folds"),
        ({"member_probabilities": np.full((2, 3, 5), 0.5)}, "member_probabilities"),
        ({"member_probabilities": np.full((2, 4, 5), 1.5)}, "[0, 1]"),
        ({"member_probabilities": np.ones((2, 4, 5), dtype=bool)}, "real numeric"),
        ({"member_probabilities": np.full((2, 4, 5), 0.5 + 0.1j)}, "real-valued"),
        ({"thresholds": np.full(4, 0.5)}, "thresholds"),
        ({"thresholds": np.full(5, 2.0)}, "thresholds"),
        ({"probabilities": np.full((4, 5), 1.5)}, "probabilities"),
    ],
)
def test_unsafe_or_malformed_files_exit_with_an_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    extra: dict[str, np.ndarray | None],
    message: str,
) -> None:
    arrays: dict[str, np.ndarray | None] = {
        "probabilities": np.full((4, 5), 0.5),
        "targets": np.zeros((4, 5), dtype=np.int64),
        "fold_ids": np.full(4, 9),
        **extra,
    }
    path = _save(
        tmp_path / "bad.npz",
        **{name: value for name, value in arrays.items() if value is not None},
    )

    assert tool.main(["--predictions", str(path)]) == 2
    assert message in capsys.readouterr().err
