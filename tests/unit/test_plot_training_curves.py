from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
from matplotlib.figure import Figure

from scripts import plot_training_curves as plotter


def _row(epoch: int = 0) -> dict[str, object]:
    return {
        "epoch": epoch,
        "validation_macro_auroc": 0.7,
        "train_loss": 0.5,
        "validation_loss": 0.6,
        "learning_rate": 0.001,
        "train_samples_per_second": 25.0,
    }


def _write_history(root: Path, rows: list[dict[str, object]]) -> Path:
    run = root / "fixture_run"
    run.mkdir()
    (run / "history.jsonl").write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return run


@pytest.mark.parametrize(
    "field,value",
    [
        ("train_loss", float("nan")),
        ("validation_loss", float("inf")),
        ("learning_rate", -0.5),
        ("train_samples_per_second", -1),
        ("validation_macro_auroc", 1.1),
        ("validation_macro_auroc", -0.1),
        ("train_loss", True),
        ("train_loss", None),
        ("epoch", 1.5),
        ("epoch", -1),
        ("epoch", True),
    ],
)
def test_invalid_history_fails_before_creating_a_figure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], field: str, value: object
) -> None:
    row = _row()
    row[field] = value
    run = _write_history(tmp_path, [row])
    output = tmp_path / "history.png"
    figures_before = plt.get_fignums()
    assert plotter.main([str(run), "--output", str(output)]) == 1
    assert not output.exists()
    assert plt.get_fignums() == figures_before
    diagnostic = capsys.readouterr().err
    assert field in diagnostic
    assert "history.jsonl:1" in diagnostic


@pytest.mark.parametrize("epochs", [[1, 1], [2, 1]])
def test_history_requires_strictly_increasing_epochs(tmp_path: Path, epochs: list[int]) -> None:
    run = _write_history(tmp_path, [_row(epoch) for epoch in epochs])
    with pytest.raises(ValueError, match="increasing"):
        plotter._load_history(run)


def test_finite_float_conversion_rejects_overflow() -> None:
    with pytest.raises(ValueError, match="finite"):
        plotter._number({"train_loss": 10**1000}, "train_loss")


def test_failed_save_preserves_existing_output_and_closes_figure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    run = _write_history(tmp_path, [_row()])
    output = tmp_path / "history.png"
    output.write_bytes(b"previous result")
    figures_before = plt.get_fignums()

    def failed_save(self: Figure, path: Path, **kwargs: object) -> None:
        Path(path).write_bytes(b"incomplete image")
        raise OSError("disk full")

    monkeypatch.setattr(Figure, "savefig", failed_save)
    assert plotter.main([str(run), "--output", str(output)]) == 1
    assert output.read_bytes() == b"previous result"
    assert plt.get_fignums() == figures_before
    assert "disk full" in capsys.readouterr().err
    assert list(tmp_path.glob(".history.*")) == []


@pytest.mark.parametrize("filename", ["history.png", "history"])
def test_valid_history_renders_png_without_leaking_figures(tmp_path: Path, filename: str) -> None:
    run = _write_history(tmp_path, [_row(0), _row(1)])
    output = tmp_path / filename
    figures_before = plt.get_fignums()
    assert plotter.main([str(run), "--output", str(output)]) == 0
    assert output.with_suffix(".png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert plt.get_fignums() == figures_before
