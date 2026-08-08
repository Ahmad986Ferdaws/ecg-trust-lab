"""Command-line entry points for locked calibration and final reporting."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, cast

import numpy as np
import typer
from numpy.typing import NDArray

from ecg_trust.decisioning import (
    fit_calibration_decisions,
    generate_final_report,
    load_calibration_decisions,
    save_calibration_decisions,
    save_final_report,
)
from ecg_trust.predictions import load_prediction_artifact
from ecg_trust.protocol import authorize_final_test_access, load_protocol

app = typer.Typer(
    name="ecg-decisions",
    help="Fit fold-9 decisions and run a token-gated fold-10 report.",
    no_args_is_help=True,
)


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


@app.command("final-report")
def final_report_command(
    decisions_path: Annotated[
        Path, typer.Option("--decisions", exists=True, dir_okay=False)
    ],
    predictions: Annotated[
        Path, typer.Option(exists=True, dir_okay=False)
    ],
    subgroups_path: Annotated[
        Path, typer.Option("--subgroups", exists=True, dir_okay=False)
    ],
    protocol_path: Annotated[
        Path, typer.Option("--protocol", exists=True, dir_okay=False)
    ],
    output: Annotated[Path, typer.Option(dir_okay=False)],
    final_test_purpose: Annotated[str, typer.Option("--final-test-purpose")],
    final_test_confirmation: Annotated[
        str, typer.Option("--final-test-confirmation")
    ],
    bootstrap_resamples: Annotated[int, typer.Option(min=2)] = 1_000,
    bootstrap_seed: Annotated[int, typer.Option(min=0)] = 20_260_808,
    minimum_valid_resamples: Annotated[int | None, typer.Option(min=1)] = None,
    minimum_group_samples: Annotated[int, typer.Option(min=1)] = 30,
    minimum_group_patients: Annotated[int, typer.Option(min=1)] = 20,
) -> None:
    """Apply frozen decisions to fold 10; this command performs no tuning."""

    protocol = load_protocol(protocol_path)
    token = authorize_final_test_access(
        protocol,
        purpose=final_test_purpose,
        confirmation=final_test_confirmation,
    )
    decisions = load_calibration_decisions(decisions_path, protocol=protocol)
    prediction = load_prediction_artifact(
        predictions, protocol=protocol, test_access=token
    )
    subgroup_ids, subgroups = _load_subgroups(subgroups_path)
    report = generate_final_report(
        decisions,
        prediction,
        protocol=protocol,
        test_access=token,
        subgroup_ecg_id=subgroup_ids,
        subgroups=subgroups,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
        bootstrap_minimum_valid=minimum_valid_resamples,
        minimum_group_samples=minimum_group_samples,
        minimum_group_patients=minimum_group_patients,
    )
    saved = save_final_report(
        report, output, protocol=protocol, test_access=token
    )
    typer.echo(json.dumps(saved.to_dict(), sort_keys=True))


def _load_subgroups(
    path: Path,
) -> tuple[NDArray[np.object_], dict[str, NDArray[np.object_]]]:
    try:
        decoded: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise typer.BadParameter(f"could not decode subgroup JSON: {exc}") from exc
    if not isinstance(decoded, Mapping):
        raise typer.BadParameter("subgroup JSON must be a mapping")
    root = cast(Mapping[str, object], decoded)
    if set(root) != {"ecg_id", "attributes"}:
        raise typer.BadParameter(
            "subgroup JSON requires exactly 'ecg_id' and 'attributes'"
        )
    ids = root["ecg_id"]
    attributes = root["attributes"]
    if not isinstance(ids, list) or not isinstance(attributes, Mapping):
        raise typer.BadParameter("invalid subgroup ecg_id or attributes")
    result: dict[str, NDArray[np.object_]] = {}
    for name, values in attributes.items():
        if not isinstance(name, str) or not isinstance(values, list):
            raise typer.BadParameter("subgroup attributes must map names to lists")
        result[name] = np.asarray(values, dtype=object)
    return np.asarray(cast(list[object], ids), dtype=object), result


def main() -> None:
    app()


if __name__ == "__main__":
    main()
