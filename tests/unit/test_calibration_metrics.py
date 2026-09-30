from __future__ import annotations

import numpy as np
import pytest

from ecg_trust.calibration_metrics import (
    CalibrationMetricError,
    adaptive_calibration_error,
    monotonic_sweep_calibration_error,
)
from ecg_trust.evaluation import EvaluationValidationError, fixed_bin_ece
from ecg_trust.protocol import LABEL_ORDER


def _matrix(column: np.ndarray) -> np.ndarray:
    return np.repeat(column[:, None], len(LABEL_ORDER), axis=1)


def test_calibrated_bins_have_zero_error_and_a_constant_offset_is_exact() -> None:
    probabilities = _matrix(np.full(100, 0.2))
    targets = _matrix((np.arange(100) % 5 == 0).astype(np.int64))

    calibrated = adaptive_calibration_error(targets, probabilities, n_bins=10)
    offset = adaptive_calibration_error(
        np.zeros((100, 5), dtype=np.int64), _matrix(np.full(100, 0.5))
    )

    assert calibrated.macro == pytest.approx(0.0)
    assert offset.macro == pytest.approx(0.5)
    assert [item.label for item in calibrated.labels] == list(LABEL_ORDER)
    assert calibrated.to_dict()["method"] == "equal_mass_ace"


def test_matches_the_equal_width_reference_when_bins_coincide() -> None:
    rng = np.random.default_rng(3)
    probabilities = np.concatenate(
        (rng.uniform(0.0, 0.49, size=(30, 5)), rng.uniform(0.51, 1.0, size=(30, 5)))
    )
    targets = (rng.uniform(size=(60, 5)) < probabilities).astype(np.int64)

    equal_mass = adaptive_calibration_error(targets, probabilities, n_bins=2)
    single = adaptive_calibration_error(targets, probabilities, n_bins=1)

    for index, item in enumerate(equal_mass.labels):
        assert item.calibration_error == pytest.approx(
            fixed_bin_ece(targets[:, index], probabilities[:, index], n_bins=2)
        )
        assert single.labels[index].calibration_error == pytest.approx(
            abs(targets[:, index].mean() - probabilities[:, index].mean())
        )


def test_bin_count_is_capped_by_sample_count() -> None:
    result = adaptive_calibration_error(
        np.array([[0, 1, 0, 1, 0], [1, 0, 1, 0, 1]]),
        np.full((2, 5), 0.5),
        n_bins=15,
    )

    assert {item.bins for item in result.labels} == {2}
    assert result.macro == pytest.approx(0.5)


def test_sweep_chooses_the_largest_monotone_count_across_all_candidates() -> None:
    probabilities = _matrix(np.linspace(0.1, 0.9, 5))
    nonnested = _matrix(np.array([0, 1, 1, 0, 1]))
    monotone = _matrix(np.array([0, 0, 1, 1, 1]))
    never = _matrix(np.array([1, 1, 0, 0, 0]))

    nonnested_result = monotonic_sweep_calibration_error(nonnested, probabilities, max_bins=3)
    monotone_result = monotonic_sweep_calibration_error(monotone, probabilities)
    never_result = monotonic_sweep_calibration_error(never, probabilities)

    assert {item.bins for item in nonnested_result.labels} == {3}
    assert {item.bins for item in monotone_result.labels} == {5}
    assert {item.bins for item in never_result.labels} == {1}
    assert monotone_result.to_dict()["method"] == "equal_mass_monotonic_sweep"


@pytest.mark.parametrize("n_bins", [0, -1, True, 2.0])
def test_invalid_bin_controls_are_rejected(n_bins: object) -> None:
    with pytest.raises(CalibrationMetricError, match="bin count"):
        adaptive_calibration_error(np.zeros((3, 5)), np.zeros((3, 5)), n_bins=n_bins)  # type: ignore[arg-type]


def test_sweep_requires_at_least_two_candidate_bins() -> None:
    with pytest.raises(CalibrationMetricError, match="at least 2"):
        monotonic_sweep_calibration_error(np.zeros((3, 5)), np.zeros((3, 5)), max_bins=1)


def test_invalid_probabilities_reuse_the_evaluation_contract() -> None:
    with pytest.raises(EvaluationValidationError):
        adaptive_calibration_error(np.zeros((3, 5)), np.full((3, 5), 1.5))
