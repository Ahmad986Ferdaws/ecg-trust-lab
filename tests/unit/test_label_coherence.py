from __future__ import annotations

from itertools import product
from typing import cast

import pytest

from ecg_trust.conformal import BinaryDecision
from ecg_trust.constants import SUPERCLASSES
from ecg_trust.label_coherence import (
    INCOMPATIBLE_LABEL_PAIRS,
    IncompatibleLabelPair,
    LabelCoherenceValidationError,
    find_incoherent_labels,
)

SUPPORTED = BinaryDecision.SUPPORTED
NOT_SUPPORTED = BinaryDecision.NOT_SUPPORTED


def _decisions(*supported: str) -> tuple[BinaryDecision, ...]:
    return tuple(SUPPORTED if label in supported else NOT_SUPPORTED for label in SUPERCLASSES)


def test_rule_lists_only_rare_norm_pairs_and_allows_norm_with_conduction() -> None:
    assert tuple((pair.first, pair.second) for pair in INCOMPATIBLE_LABEL_PAIRS) == (
        ("NORM", "MI"),
        ("NORM", "STTC"),
        ("NORM", "HYP"),
    )
    assert IncompatibleLabelPair("NORM", "CD") not in INCOMPATIBLE_LABEL_PAIRS


@pytest.mark.parametrize(
    ("supported", "expected"),
    [
        ((), ()),
        (("NORM",), ()),
        (("NORM", "CD"), ()),
        (("MI", "STTC", "CD", "HYP"), ()),
        (("NORM", "MI"), ("NORM", "MI")),
        (("NORM", "HYP"), ("NORM", "HYP")),
        (("NORM", "MI", "STTC", "CD"), ("NORM", "MI", "STTC")),
        (SUPERCLASSES, ("NORM", "MI", "STTC", "HYP")),
    ],
)
def test_conflicts_name_supported_pair_members_in_canonical_order(
    supported: tuple[str, ...],
    expected: tuple[str, ...],
) -> None:
    assert find_incoherent_labels(_decisions(*supported)) == expected


def test_every_singleton_label_set_follows_the_norm_exclusion_rule() -> None:
    for decisions in product((NOT_SUPPORTED, SUPPORTED), repeat=len(SUPERCLASSES)):
        supported = {
            label
            for label, decision in zip(SUPERCLASSES, decisions, strict=True)
            if decision is SUPPORTED
        }
        partners = supported & {"MI", "STTC", "HYP"}
        expected = {"NORM", *partners} if "NORM" in supported and partners else set()

        result = find_incoherent_labels(decisions)

        assert set(result) == expected
        assert result == tuple(label for label in SUPERCLASSES if label in expected)


def test_custom_pairs_are_applied_without_the_default_rule() -> None:
    pairs = (IncompatibleLabelPair("MI", "CD"),)

    assert find_incoherent_labels(_decisions("NORM", "MI"), incompatible_pairs=pairs) == ()
    assert find_incoherent_labels(_decisions("MI", "CD"), incompatible_pairs=pairs) == (
        "MI",
        "CD",
    )


@pytest.mark.parametrize(
    ("decisions", "message"),
    [
        (_decisions("NORM", "MI")[:-1], "tuple of 5"),
        (list(_decisions("NORM", "MI")), "tuple of 5"),
        (("supported",) * len(SUPERCLASSES), "BinaryDecision"),
        ((True,) * len(SUPERCLASSES), "BinaryDecision"),
        (
            (SUPPORTED, SUPPORTED, BinaryDecision.UNCERTAIN, NOT_SUPPORTED, NOT_SUPPORTED),
            "singleton",
        ),
    ],
)
def test_rejects_malformed_or_non_singleton_decisions(decisions: object, message: str) -> None:
    with pytest.raises(LabelCoherenceValidationError, match=message):
        find_incoherent_labels(cast(tuple[BinaryDecision, ...], decisions))


@pytest.mark.parametrize(
    ("first", "second", "message"),
    [
        ("NORM", "AFIB", "canonical"),
        ("norm", "MI", "canonical"),
        ("NORM", "NORM", "distinct"),
        ("MI", "NORM", "canonical order"),
        ("NORM", None, "strings"),
    ],
)
def test_pair_rejects_unknown_repeated_or_reordered_labels(
    first: object,
    second: object,
    message: str,
) -> None:
    with pytest.raises(LabelCoherenceValidationError, match=message):
        IncompatibleLabelPair(cast(str, first), cast(str, second))


@pytest.mark.parametrize(
    "pairs",
    [
        (),
        [IncompatibleLabelPair("NORM", "MI")],
        (("NORM", "MI"),),
        (IncompatibleLabelPair("NORM", "MI"), IncompatibleLabelPair("NORM", "MI")),
    ],
)
def test_rejects_empty_untyped_or_duplicate_rules(pairs: object) -> None:
    with pytest.raises(LabelCoherenceValidationError, match="incompatible_pairs"):
        find_incoherent_labels(
            _decisions("NORM", "MI"),
            incompatible_pairs=cast(tuple[IncompatibleLabelPair, ...], pairs),
        )
