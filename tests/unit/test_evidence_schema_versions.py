from __future__ import annotations

import pytest

from ecg_trust.conformal import (
    BinaryPredictionSets,
    ClassConditionalLabelwiseConformal,
    ConformalMetrics,
    ConformalValidationError,
    LabelwiseBinaryConformal,
    evaluate_prediction_sets,
)
from ecg_trust.open_world.scores import (
    MaxNormalizedBernoulliEntropyScorer,
    NormalizedBernoulliEntropyScorer,
    OODScoreValidationError,
    SymmetricBinaryEnergyScorer,
)

Artifact = (
    NormalizedBernoulliEntropyScorer
    | MaxNormalizedBernoulliEntropyScorer
    | SymmetricBinaryEnergyScorer
    | LabelwiseBinaryConformal
    | ClassConditionalLabelwiseConformal
    | BinaryPredictionSets
    | ConformalMetrics
)


def _artifacts() -> list[Artifact]:
    calibrator = LabelwiseBinaryConformal.fit(
        [[0.9], [0.8], [0.2], [0.1]], [[1], [1], [0], [0]],
        label_names=("label",), alpha=0.5,
    )
    predictions = calibrator.predict([[0.8], [0.2]])
    class_conditional = ClassConditionalLabelwiseConformal.fit(
        [[0.9], [0.8], [0.2], [0.1]], [[1], [1], [0], [0]],
        label_names=("label",),
        negative_alphas={"label": 0.5},
        positive_alphas={"label": 0.5},
    )
    return [
        NormalizedBernoulliEntropyScorer(),
        MaxNormalizedBernoulliEntropyScorer(),
        SymmetricBinaryEnergyScorer(),
        calibrator,
        class_conditional,
        predictions,
        evaluate_prediction_sets(predictions, [[1], [0]]),
    ]


@pytest.mark.parametrize("artifact", _artifacts(), ids=lambda value: type(value).__name__)
@pytest.mark.parametrize("version", [True, False, 1.0, "1", None, 0, 2])
def test_schema_version_requires_the_supported_integer(artifact: Artifact, version: object) -> None:
    payload = artifact.to_dict()
    payload["schema_version"] = version
    error = (
        OODScoreValidationError
        if isinstance(
            artifact,
            (
                NormalizedBernoulliEntropyScorer,
                MaxNormalizedBernoulliEntropyScorer,
                SymmetricBinaryEnergyScorer,
            ),
        )
        else ConformalValidationError
    )
    with pytest.raises(error, match="schema_version"):
        type(artifact).from_dict(payload)


@pytest.mark.parametrize("artifact", _artifacts(), ids=lambda value: type(value).__name__)
def test_integer_schema_version_round_trips(artifact: Artifact) -> None:
    payload = artifact.to_dict()
    assert type(payload["schema_version"]) is int
    assert payload["schema_version"] == 1
    assert type(artifact).from_dict(payload) == artifact
