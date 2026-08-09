from __future__ import annotations

import pytest

from ecg_trust.protocol import FINAL_TEST_CONFIRMATION
from scripts.release_pipeline import build_parser


def test_release_cli_exposes_only_spec_derived_fold9_and_final_settings() -> None:
    parser = build_parser()

    export = parser.parse_args(
        [
            "export-fold9",
            "--refit-bundle",
            "refits.json",
            "--evaluation-spec",
            "spec.json",
        ]
    )
    assert export.command == "export-fold9"
    assert not hasattr(export, "output_dir")
    assert not hasattr(export, "device")
    assert not hasattr(export, "batch_size")

    calibration = parser.parse_args(
        [
            "fit-calibration",
            "--refit-bundle",
            "refits.json",
            "--evaluation-spec",
            "spec.json",
        ]
    )
    assert calibration.command == "fit-calibration"
    assert not hasattr(calibration, "coverage")
    assert not hasattr(calibration, "decision_output_dir")
    assert not hasattr(calibration, "prediction")

    final = parser.parse_args(
        [
            "run-final",
            "--refit-bundle",
            "refits.json",
            "--calibration-bundle",
            "calibration.json",
            "--evaluation-spec",
            "spec.json",
            "--purpose",
            "one-time final evaluation",
            "--operator",
            "operator",
            "--confirmation",
            FINAL_TEST_CONFIRMATION,
        ]
    )
    assert final.command == "run-final"
    for removed_override in (
        "batch_size",
        "num_workers",
        "device",
        "no_bf16",
        "bootstrap_resamples",
        "bootstrap_seed",
        "bootstrap_confidence",
        "bootstrap_minimum_valid",
        "minimum_group_samples",
        "minimum_group_patients",
        "ece_bins",
        "output_dir",
        "subgroups",
        "ledger",
    ):
        assert not hasattr(final, removed_override)


def test_release_cli_rejects_removed_final_scientific_override() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "run-final",
                "--refit-bundle",
                "refits.json",
                "--calibration-bundle",
                "calibration.json",
                "--evaluation-spec",
                "spec.json",
                "--purpose",
                "one-time final evaluation",
                "--operator",
                "operator",
                "--confirmation",
                FINAL_TEST_CONFIRMATION,
                "--device",
                "cpu",
            ]
        )
