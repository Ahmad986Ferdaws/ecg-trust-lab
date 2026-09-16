from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from ecg_trust.open_world import (
    OODScoreValidationError,
    normalized_bernoulli_entropy,
    symmetric_binary_energy,
)


@pytest.mark.parametrize("score", [normalized_bernoulli_entropy, symmetric_binary_energy])
@pytest.mark.parametrize(
    "invalid", [np.complex128(0.5 + 3j), 10**1000], ids=["complex", "overflow"]
)
def test_scorers_reject_lossy_object_conversion(score: Callable, invalid: object) -> None:
    with pytest.raises(OODScoreValidationError):
        score(np.array([invalid], dtype=object).reshape(1, 1))


def test_energy_temperature_overflow_is_a_domain_error() -> None:
    with pytest.raises(OODScoreValidationError):
        symmetric_binary_energy([[0.0]], temperature=10**1000)
