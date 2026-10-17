from __future__ import annotations

import json

import pytest

from ecg_trust.contracts import TrustDecision
from ecg_trust.failure_lab import FailureLabDecision, FailureLabError

ALLOWED = TrustDecision.PREDICTION_ALLOWED
REASON = "ALL_TRUST_GATES_PASSED"


def test_public_snapshot_does_not_retain_caller_owned_lists() -> None:
    reasons = [REASON]
    probabilities = [0.5] * 5
    result = FailureLabDecision(ALLOWED, reasons, probabilities)  # type: ignore[arg-type]
    reasons[0] = "private-note"
    probabilities[0] = float("nan")
    payload = result.to_public_dict()
    assert payload["reason_codes"] == [REASON]
    assert payload["probabilities"]["NORM"] == 0.5
    json.dumps(payload, allow_nan=False)


@pytest.mark.parametrize("value", [True, -0.1, 1.1, float("nan"), float("inf"), "0.5", 1j])
def test_invalid_public_probabilities_are_refused(value: object) -> None:
    with pytest.raises(FailureLabError, match="probabilities"):
        FailureLabDecision(ALLOWED, (REASON,), (value, 0.5, 0.5, 0.5, 0.5))


@pytest.mark.parametrize("reasons", [(), ("private-note",), (REASON, REASON), "ABSTAIN", (None,)])
def test_reasons_require_a_nonempty_unique_public_vocabulary(reasons: object) -> None:
    with pytest.raises(FailureLabError, match="reason"):
        FailureLabDecision(ALLOWED, reasons, (0.5,) * 5)


def test_internal_label_coherence_reason_cannot_be_published_directly() -> None:
    with pytest.raises(FailureLabError, match="reason"):
        FailureLabDecision(TrustDecision.ABSTAIN, ("LABEL_SET_INCOHERENT",), None)
    result = FailureLabDecision(TrustDecision.ABSTAIN, ("CONFIDENCE_GATE_ABSTAINED",), None)
    assert "probabilities" not in result.to_public_dict()


def test_unknown_decision_is_refused_before_serialization() -> None:
    with pytest.raises(FailureLabError, match="decision"):
        FailureLabDecision("ABSTAIN", ("CONFIDENCE_GATE_ABSTAINED",), None)


def test_serialized_collections_are_independent() -> None:
    result = FailureLabDecision(ALLOWED, (REASON,), (0, 1, 0.5, 0.25, 0.75))
    first = result.to_public_dict()
    first["reason_codes"].clear()
    first["probabilities"]["NORM"] = 1
    second = result.to_public_dict()
    assert second["reason_codes"] == [REASON]
    assert second["probabilities"]["NORM"] == 0
