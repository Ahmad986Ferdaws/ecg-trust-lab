from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from ecg_trust.open_world import MahalanobisValidationError, ShrinkageMahalanobisDetector


def _fitted() -> ShrinkageMahalanobisDetector:
    return ShrinkageMahalanobisDetector.fit(
        [[0.0, 1.0], [1.0, 0.0], [2.0, 1.0]],
        [[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]], inlier_coverage=0.5,
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [("threshold", float("nan")), ("threshold", -1), ("embedding_dim", True),
     ("mean", [0.0]), ("precision", [[0., 0.], [0., 0.]]),
     ("precision", [[1., 2.], [0., 1.]]), ("quantile_rank", 1),
     ("n_fit_samples", 1), ("ridge", 0), ("shrinkage", 2), ("inlier_coverage", 1)],
)
def test_direct_construction_enforces_the_fitted_state(field: str, value: object) -> None:
    with pytest.raises(MahalanobisValidationError):
        replace(_fitted(), **{field: value})


def test_detector_snapshots_nested_sequences() -> None:
    fitted = _fitted()
    mean = list(fitted.mean)
    precision = [list(row) for row in fitted.precision]
    copied = replace(fitted, mean=mean, precision=precision)
    before = copied.score([[2.0, 3.0]])
    mean[0] = float("nan")
    precision[0][0] = -1.0
    assert copied == fitted
    np.testing.assert_array_equal(copied.score([[2.0, 3.0]]), before)


@pytest.mark.parametrize("version", [True, 1.0])
def test_detector_schema_version_requires_integer(version: object) -> None:
    payload = _fitted().to_dict()
    payload["schema_version"] = version
    with pytest.raises(MahalanobisValidationError, match="schema_version"):
        ShrinkageMahalanobisDetector.from_dict(payload)


@pytest.mark.parametrize("object_dtype", [False, True])
def test_complex_embeddings_are_rejected_in_fit_and_score(object_dtype: bool) -> None:
    raw = np.array([np.complex128(1 + 3j)] * 6, dtype=object if object_dtype else complex)
    embeddings = raw.reshape(3, 2)
    with pytest.raises(MahalanobisValidationError):
        _fitted().score(embeddings)
    with pytest.raises(MahalanobisValidationError):
        ShrinkageMahalanobisDetector.fit(embeddings, np.ones((3, 2)), inlier_coverage=0.5)
    with pytest.raises(MahalanobisValidationError):
        ShrinkageMahalanobisDetector.fit(np.ones((3, 2)), embeddings, inlier_coverage=0.5)


def test_overflowing_embeddings_use_domain_error() -> None:
    with pytest.raises(MahalanobisValidationError):
        _fitted().score([[10**1000, 0]])
