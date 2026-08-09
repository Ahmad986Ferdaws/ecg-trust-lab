"""Command-line entry points for locked calibration and final reporting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from ecg_trust.decisioning import (
    fit_calibration_decisions,
    save_calibration_decisions,
)
from ecg_trust.predictions import load_prediction_artifact
from ecg_trust.protocol import load_protocol

app = typer.Typer(
    name="ecg-decisions",
    help=(
        "Low-level fold-9 decision fitting. Use scripts/release_pipeline.py "
        "for sealed release calibration and all fold-10 evaluation."
    ),
    no_args_is_help=True,
)


@app.callback()
def root_command() -> None:
    """Keep fold-9 fitting as an explicit subcommand."""


@app.command("fit")
def fit_command(
    predictions: Annotated[
        Path, typer.Option(exists=True, dir_okay=False)
    ],
    protocol_path: Annotated[
        Path, typer.Option("--protocol", exists=True, dir_okay=False)
    ],
    output: Annotated[Path, typer.Option(dir_okay=False)],
    coverage: Annotated[list[float] | None, typer.Option("--coverage")] = None,
) -> None:
    """Fit temperature, thresholds, and entropy gates from fold 9 only."""

    protocol = load_protocol(protocol_path)
    prediction = load_prediction_artifact(predictions, protocol=protocol)
    decisions = fit_calibration_decisions(
        prediction,
        protocol=protocol,
        coverage_targets=coverage or [1.0, 0.9, 0.8, 0.7, 0.5],
    )
    saved = save_calibration_decisions(decisions, output)
    typer.echo(json.dumps(saved.to_dict(), sort_keys=True))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
