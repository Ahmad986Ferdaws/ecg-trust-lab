"""Cross-label coherence rule for singleton superclass decisions.

The five superclass outputs are independent sigmoid heads, and label-wise
conformal sets are calibrated separately for each label.  Neither makes a
released label *set* jointly plausible: every label can be a confident
singleton while the combination is one the source cohort almost never records.

This module names such combinations as a fixed rule.  ``NORM`` asserted
together with ``MI``, ``STTC``, or ``HYP`` is incoherent.  In the descriptive
pairwise table of ``docs/DATA_CARD.md`` (21,388 included PTB-XL v1.0.3
records), ``NORM`` co-occurs with ``MI`` in 1 record, ``HYP`` in 5, and
``STTC`` in 33, out of 9,514 ``NORM`` records.  ``NORM`` with ``CD`` (415
records) is deliberately allowed because conduction findings are commonly
recorded on an otherwise normal ECG.

Guarantees: :func:`find_incoherent_labels` is a pure, deterministic function of
singleton decisions.  It reads no probabilities, targets, or thresholds and
fits nothing.  A label joins a conflict only when it and its paired label are
both ``SUPPORTED``; ``NOT_SUPPORTED`` never conflicts, so an all-negative set is
always coherent by definition.  "Coherent" means only that no listed pair is
jointly ``SUPPORTED``, not that the set is plausible in the cohort.

Limits: the pairs are a hand-stated rule motivated by whole-cohort descriptive
counts that include the already observed fold 10.  They are not fitted or
validated, and genuine co-occurrences exist (at most 39 ``NORM`` records across
the listed pairs, which can overlap), so applying the rule trades some correct
releases for fewer implausible ones.  It cannot detect other implausible
combinations or establish that a coherent set is clinically correct.  In
particular, the empty supported set has zero support in the included cohort
(``docs/DATA_CARD.md``: no included row has an all-zero target), less than any
listed pair, yet it is deliberately out of scope and never flagged.  It is not
part of ``trust-policy-v1``; using it in any release requires a new
preregistered protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from ecg_trust.conformal import BinaryDecision
from ecg_trust.constants import SUPERCLASSES


class LabelCoherenceValidationError(ValueError):
    """Raised when a coherence rule or singleton decision vector is malformed."""


@dataclass(frozen=True, slots=True)
class IncompatibleLabelPair:
    """Two canonical superclasses that must not both be released as supported."""

    first: str
    second: str

    def __post_init__(self) -> None:
        if not isinstance(self.first, str) or not isinstance(self.second, str):
            raise LabelCoherenceValidationError("incompatible labels must be strings")
        if self.first not in SUPERCLASSES or self.second not in SUPERCLASSES:
            raise LabelCoherenceValidationError("incompatible labels must be canonical")
        if SUPERCLASSES.index(self.first) >= SUPERCLASSES.index(self.second):
            raise LabelCoherenceValidationError(
                "incompatible labels must be distinct and in canonical order"
            )


INCOMPATIBLE_LABEL_PAIRS: Final[tuple[IncompatibleLabelPair, ...]] = (
    IncompatibleLabelPair("NORM", "MI"),
    IncompatibleLabelPair("NORM", "STTC"),
    IncompatibleLabelPair("NORM", "HYP"),
)


def find_incoherent_labels(
    decisions: tuple[BinaryDecision, ...],
    *,
    incompatible_pairs: tuple[IncompatibleLabelPair, ...] = INCOMPATIBLE_LABEL_PAIRS,
) -> tuple[str, ...]:
    """Return every label in a supported incompatible pair, in canonical order.

    ``decisions`` must hold one singleton decision per label of ``SUPERCLASSES``.
    ``UNCERTAIN`` is rejected because coherence is defined only after every
    label-wise prediction set has resolved to a singleton.  An empty result
    means the supported set contains no listed pair.
    """

    if not isinstance(decisions, tuple) or len(decisions) != len(SUPERCLASSES):
        raise LabelCoherenceValidationError(
            f"decisions must be a tuple of {len(SUPERCLASSES)} label decisions"
        )
    if any(not isinstance(decision, BinaryDecision) for decision in decisions):
        raise LabelCoherenceValidationError("decisions must use BinaryDecision")
    if any(decision is BinaryDecision.UNCERTAIN for decision in decisions):
        raise LabelCoherenceValidationError("coherence requires singleton label decisions")
    if (
        not isinstance(incompatible_pairs, tuple)
        or not incompatible_pairs
        or any(not isinstance(pair, IncompatibleLabelPair) for pair in incompatible_pairs)
    ):
        raise LabelCoherenceValidationError(
            "incompatible_pairs must be a non-empty tuple of IncompatibleLabelPair"
        )
    if len(set(incompatible_pairs)) != len(incompatible_pairs):
        raise LabelCoherenceValidationError("incompatible_pairs must be unique")

    supported = {
        label
        for label, decision in zip(SUPERCLASSES, decisions, strict=True)
        if decision is BinaryDecision.SUPPORTED
    }
    conflicting = {
        label
        for pair in incompatible_pairs
        if pair.first in supported and pair.second in supported
        for label in (pair.first, pair.second)
    }
    return tuple(label for label in SUPERCLASSES if label in conflicting)


__all__ = [
    "INCOMPATIBLE_LABEL_PAIRS",
    "IncompatibleLabelPair",
    "LabelCoherenceValidationError",
    "find_incoherent_labels",
]
