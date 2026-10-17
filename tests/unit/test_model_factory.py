from __future__ import annotations

import subprocess
import sys

import pytest

from ecg_trust.experiment_config import ModelConfig
from ecg_trust.experiment_runner import DevelopmentRunnerError, build_experiment_model
from ecg_trust.models import factory


@pytest.mark.parametrize("module", ["ecg_trust.models.factory", "ecg_trust.demo_backend"])
def test_inference_model_construction_does_not_import_experiment_execution(module: str) -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            f"import {module}; import sys; assert 'ecg_trust.experiment_runner' not in sys.modules",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def test_capacity_drift_preserves_new_and_legacy_error_contracts(monkeypatch) -> None:
    monkeypatch.setattr(factory, "count_parameters", lambda model: 0)
    config = ModelConfig(architecture="resnet1d", preset="matched_capacity")
    with pytest.raises(factory.ModelConstructionError, match="matched-capacity model drift"):
        factory.build_model(config)
    with pytest.raises(DevelopmentRunnerError, match="matched-capacity model drift"):
        build_experiment_model(config)
