from __future__ import annotations

import random
import statistics
from collections.abc import Mapping, Sequence
from typing import overload

import pytest

from ecg_trust.counterfactual.counterfactual_review import (
    BlindedCardiologyReview,
    _pairwise_review_statistics,
)


def _review(rating: int, useful: bool) -> BlindedCardiologyReview:
    return BlindedCardiologyReview("a" * 64, "REV-AAAAAAAA", rating, False, useful)


def _pairwise_oracle(
    groups: Mapping[str, Sequence[BlindedCardiologyReview]],
) -> tuple[float | None, float | None, float | None]:
    pairs = [
        (left, right)
        for reviews in groups.values()
        for index, left in enumerate(reviews)
        for right in reviews[index + 1 :]
    ]
    if not pairs:
        return None, None, None
    agreement = sum(a.useful_for_model_review == b.useful_for_model_review for a, b in pairs)
    differences = sum(
        abs(a.morphology_plausibility_rating - b.morphology_plausibility_rating) for a, b in pairs
    )
    observed = agreement / len(pairs)
    prevalence = statistics.fmean(
        review.useful_for_model_review for reviews in groups.values() for review in reviews
    )
    expected = prevalence**2 + (1.0 - prevalence) ** 2
    kappa = (observed - expected) / (1.0 - expected) if expected < 1.0 else None
    return observed, kappa, differences / len(pairs)


@pytest.mark.parametrize("seed", range(20))
def test_agreement_matches_pairwise_oracle_for_unequal_panels(seed: int) -> None:
    rng = random.Random(seed)
    groups = {
        str(index): [_review(rng.randint(1, 5), bool(rng.randrange(2))) for _ in range(size)]
        for index, size in enumerate([0, 1, 2, 3, 7, 17, 40])
    }
    assert _pairwise_review_statistics(groups) == _pairwise_oracle(groups)


@pytest.mark.parametrize("useful", [False, True])
@pytest.mark.parametrize("rating", [1, 2, 3, 4, 5])
def test_unanimous_panels_preserve_undefined_kappa(useful: bool, rating: int) -> None:
    assert _pairwise_review_statistics({"p": [_review(rating, useful)] * 10}) == (1.0, None, 0.0)


def test_no_pairs_remains_undefined() -> None:
    assert _pairwise_review_statistics({}) == (None, None, None)
    assert _pairwise_review_statistics({"p": [_review(3, True)]}) == (None, None, None)


class _AccessBudgetSequence(Sequence[BlindedCardiologyReview]):
    def __init__(self, reviews: list[BlindedCardiologyReview]) -> None:
        self.reviews = reviews
        self.remaining = 4 * len(reviews)

    def __len__(self) -> int:
        return len(self.reviews)

    @overload
    def __getitem__(self, index: int) -> BlindedCardiologyReview: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[BlindedCardiologyReview]: ...

    def __getitem__(self, index: int | slice) -> BlindedCardiologyReview | Sequence[
        BlindedCardiologyReview
    ]:
        result = self.reviews[index]
        self.remaining -= len(result) if isinstance(result, list) else 1
        assert self.remaining >= 0, "agreement must require only a bounded number of passes"
        return result


def test_large_panel_requires_linear_input_access() -> None:
    reviews = _AccessBudgetSequence(
        [_review(index % 5 + 1, bool(index % 2)) for index in range(4000)]
    )
    result = _pairwise_review_statistics({"p": reviews})
    assert result == (0.49987496874218557, -0.0002500625156288683, 1.6004001000250063)
