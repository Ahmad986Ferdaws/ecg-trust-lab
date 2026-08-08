from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from ecg_trust.protocol import (
    ALL_FOLDS,
    CALIBRATION_FOLDS,
    DEFAULT_PROTOCOL_PATH,
    FINAL_TEST_CONFIRMATION,
    FINAL_TEST_FOLDS,
    LABEL_ORDER,
    MODEL_SELECTION_FOLDS,
    TRAIN_FOLDS,
    ExperimentProtocol,
    FinalTestAccessError,
    FinalTestAccessToken,
    FoldRole,
    ProtocolValidationError,
    authorize_final_test_access,
    load_protocol,
)


def _raw_protocol() -> dict[str, object]:
    return ExperimentProtocol.canonical().to_resolved_dict()


def _roles(raw: dict[str, object]) -> dict[str, object]:
    folds = raw["folds"]
    assert isinstance(folds, dict)
    roles = folds["roles"]
    assert isinstance(roles, dict)
    return roles


def test_repository_protocol_is_canonical_and_immutable() -> None:
    protocol = load_protocol()

    assert DEFAULT_PROTOCOL_PATH.exists()
    assert protocol.label_order == LABEL_ORDER == ("NORM", "MI", "STTC", "CD", "HYP")
    assert protocol.all_folds == ALL_FOLDS == tuple(range(1, 11))
    assert protocol.folds_for(FoldRole.TRAIN) == TRAIN_FOLDS == tuple(range(1, 8))
    assert protocol.folds_for(FoldRole.MODEL_SELECTION) == MODEL_SELECTION_FOLDS == (8,)
    assert protocol.folds_for(FoldRole.CALIBRATION) == CALIBRATION_FOLDS == (9,)
    assert protocol.development_folds == tuple(range(1, 10))

    with pytest.raises(FrozenInstanceError):
        protocol.protocol_id = "changed"  # type: ignore[misc]


def test_fold_roles_must_be_disjoint() -> None:
    raw = _raw_protocol()
    raw.pop("protocol_hash")
    _roles(raw)["train"] = [1, 2, 3, 4, 5, 6, 7, 8]

    with pytest.raises(ProtocolValidationError, match="disjoint.*fold 8"):
        ExperimentProtocol.from_mapping(raw)


def test_fold_roles_must_exhaust_universe() -> None:
    raw = _raw_protocol()
    raw.pop("protocol_hash")
    _roles(raw)["train"] = [1, 2, 3, 4, 5, 6]

    with pytest.raises(ProtocolValidationError, match=r"exhaust.*missing=\[7\]"):
        ExperimentProtocol.from_mapping(raw)


def test_disjoint_exhaustive_but_reassigned_roles_are_rejected() -> None:
    raw = _raw_protocol()
    raw.pop("protocol_hash")
    _roles(raw)["train"] = [1, 2, 3, 4, 5, 6, 8]
    _roles(raw)["model_selection"] = [7]

    with pytest.raises(ProtocolValidationError, match="immutable"):
        ExperimentProtocol.from_mapping(raw)


def test_label_order_is_fixed() -> None:
    raw = _raw_protocol()
    raw.pop("protocol_hash")
    task = raw["task"]
    assert isinstance(task, dict)
    task["label_order"] = ["MI", "NORM", "STTC", "CD", "HYP"]

    with pytest.raises(ProtocolValidationError, match="label_order is immutable"):
        ExperimentProtocol.from_mapping(raw)


def test_fold_10_is_blocked_from_ordinary_development_apis() -> None:
    protocol = ExperimentProtocol.canonical()

    with pytest.raises(FinalTestAccessError, match="fold 10 is sealed"):
        protocol.folds_for(FoldRole.FINAL_TEST)
    with pytest.raises(FinalTestAccessError, match="fold 10 is sealed"):
        protocol.guard_fold_access([1, 10])

    assert protocol.guard_fold_access(protocol.development_folds) == tuple(range(1, 10))


def test_final_test_token_requires_explicit_confirmation_and_purpose() -> None:
    protocol = ExperimentProtocol.canonical()

    with pytest.raises(FinalTestAccessError, match="non-empty"):
        authorize_final_test_access(protocol, purpose="  ", confirmation=FINAL_TEST_CONFIRMATION)
    with pytest.raises(FinalTestAccessError, match="did not match"):
        authorize_final_test_access(protocol, purpose="locked final evaluation", confirmation="yes")
    with pytest.raises(TypeError, match="cannot be constructed directly"):
        FinalTestAccessToken(protocol.protocol_hash, "forged", _seal=object())


def test_explicit_protocol_bound_token_unlocks_only_its_protocol() -> None:
    protocol = ExperimentProtocol.canonical()
    token = authorize_final_test_access(
        protocol,
        purpose="locked final evaluation after model selection",
        confirmation=FINAL_TEST_CONFIRMATION,
    )

    assert protocol.folds_for(FoldRole.FINAL_TEST, test_access=token) == FINAL_TEST_FOLDS
    assert protocol.guard_fold_access([9, 10], test_access=token) == (9, 10)
    assert token.purpose == "locked final evaluation after model selection"
    assert token.protocol_hash == protocol.protocol_hash

    other_raw = _raw_protocol()
    other_raw.pop("protocol_hash")
    other_raw["protocol_id"] = "different-analysis-protocol"
    other_protocol = ExperimentProtocol.from_mapping(other_raw)
    with pytest.raises(FinalTestAccessError, match="fold 10 is sealed"):
        other_protocol.folds_for(FoldRole.FINAL_TEST, test_access=token)


def test_resolved_serialization_round_trips_and_verifies_hash(tmp_path: Path) -> None:
    protocol = load_protocol()
    resolved = protocol.to_resolved_dict()

    assert resolved["protocol_hash"] == protocol.protocol_hash
    assert protocol.protocol_hash == (
        "sha256:ebfdb588615bfa22eedc6d936d7b0155a33702878cbe0258ebb84aaa88567e09"
    )
    assert ExperimentProtocol.from_json(protocol.to_json()) == protocol
    assert protocol == ExperimentProtocol.canonical()

    reordered = json.loads(protocol.to_json())
    assert isinstance(reordered, dict)
    round_trip_path = tmp_path / "resolved-protocol.json"
    round_trip_path.write_text(json.dumps(reordered), encoding="utf-8")
    assert load_protocol(round_trip_path) == protocol

    reordered["protocol_hash"] = "sha256:" + ("0" * 64)
    with pytest.raises(ProtocolValidationError, match="does not match"):
        ExperimentProtocol.from_mapping(reordered)


def test_unknown_fields_and_invalid_fold_requests_fail_closed() -> None:
    raw = _raw_protocol()
    raw.pop("protocol_hash")
    raw["typo"] = True

    with pytest.raises(ProtocolValidationError, match=r"unknown=\['typo'\]"):
        ExperimentProtocol.from_mapping(raw)

    protocol = ExperimentProtocol.canonical()
    with pytest.raises(ProtocolValidationError, match="outside"):
        protocol.guard_fold_access([11])
    with pytest.raises(ProtocolValidationError, match="duplicates"):
        protocol.guard_fold_access([8, 8])
    with pytest.raises(ProtocolValidationError, match="unknown fold role"):
        protocol.folds_for("validation")
