from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import cast

import pytest
from pydantic import ValidationError

from ecg_trust.audit_log import (
    AuditAction,
    AuditActorType,
    AuditEvent,
    AuditEventAdapter,
    AuditLedger,
    AuditLedgerConfig,
    AuditOutcome,
    AuditPrivacyError,
    AuditValidationError,
    LedgerCheckpoint,
    LedgerCorruptionError,
    assert_privacy_safe,
)
from ecg_trust.audit_log.audit_ledger import GENESIS_HASH, canonical_json
from ecg_trust.contracts import AUDIT_EVENT_SCHEMA_VERSION, TrustDecision

FIXED_UTC = datetime(2026, 8, 24, 12, 30, 45, 123456, tzinfo=UTC)
RELEASE_ID = "release-v1"


def _reason_for(decision: TrustDecision) -> str:
    return {
        TrustDecision.INVALID_INPUT: "INPUT_CONTRACT_INVALID",
        TrustDecision.REACQUIRE: "SIGNAL_REACQUISITION_REQUIRED",
        TrustDecision.UNSUPPORTED_INPUT: "OUTSIDE_VALIDATED_DISTRIBUTION",
        TrustDecision.ABSTAIN: "CONFIDENCE_GATE_ABSTAINED",
        TrustDecision.PREDICTION_ALLOWED: "ALL_TRUST_GATES_PASSED",
    }[decision]


def _event(
    request_id: str = "request-001",
    *,
    event_id: str | None = None,
    action: AuditAction = AuditAction.INFERENCE,
    actor_type: AuditActorType = AuditActorType.SERVICE,
    outcome: AuditOutcome = AuditOutcome.SUCCESS,
    reason_code: str | None = None,
    decision: TrustDecision | None = TrustDecision.PREDICTION_ALLOWED,
    release_id: str | None = RELEASE_ID,
    occurred_at: datetime = FIXED_UTC,
) -> AuditEvent:
    if action not in {AuditAction.CASE_VALIDATION, AuditAction.INFERENCE}:
        decision = None
    if reason_code is None and decision is not None:
        reason_code = _reason_for(decision)
    if outcome is not AuditOutcome.SUCCESS and reason_code is None:
        reason_code = "OPERATION_FAILED"
    return AuditEvent(
        schema_version=AUDIT_EVENT_SCHEMA_VERSION,
        event_id=event_id or f"event-{request_id}",
        occurred_at=occurred_at,
        request_id=request_id,
        actor_type=actor_type,
        actor_id="sentinel-service",
        action=action,
        resource_type="TRUST_DECISION",
        resource_id=f"resource-{request_id}",
        release_id=release_id,
        outcome=outcome,
        reason_code=reason_code,
        decision=decision,
    )


def _ledger_with_entries(path: Path, count: int = 3) -> AuditLedger:
    ledger = AuditLedger(path)
    for index in range(count):
        ledger.append(
            _event(
                f"request-{index:03d}",
                decision=TrustDecision.ABSTAIN,
            )
        )
    return ledger


def test_canonical_event_appends_and_round_trips_through_explicit_adapter(
    tmp_path: Path,
) -> None:
    event = _event()
    AuditEventAdapter.validate(event)
    ledger = AuditLedger(tmp_path / "audit.jsonl")
    receipt = ledger.append(event)

    assert receipt.entry.sequence == 0
    assert receipt.entry.previous_entry_hash == GENESIS_HASH
    assert receipt.entry.to_audit_event() == event
    assert AuditEventAdapter.restore(receipt.entry) == event
    assert receipt.checkpoint == LedgerCheckpoint(
        entry_count=1,
        head_hash=receipt.entry.entry_hash,
    )
    assert ledger.path.read_bytes() == receipt.entry.canonical_line()
    assert ledger.checkpoint_path.read_bytes() == receipt.checkpoint.canonical_line()
    assert ledger.read_entries()[0].to_audit_event() == event


def test_audit_log_exports_the_single_canonical_event_name() -> None:
    from ecg_trust.audit_log import AuditEvent as LedgerPublicEvent
    from ecg_trust.contracts import AuditEvent as ContractEvent

    assert LedgerPublicEvent is ContractEvent


@pytest.mark.parametrize("action", list(AuditAction))
def test_every_canonical_action_round_trips(tmp_path: Path, action: AuditAction) -> None:
    release_id = RELEASE_ID if action is AuditAction.RELEASE_VERIFICATION else None
    if action in {AuditAction.CASE_VALIDATION, AuditAction.INFERENCE}:
        event = _event(action=action)
    else:
        event = _event(
            action=action,
            release_id=release_id,
            reason_code=("RELEASE_VERIFIED" if release_id else None),
        )
    ledger = AuditLedger(tmp_path / f"{action.value}.jsonl")

    entry = ledger.append(event).entry

    assert entry.action is action
    assert entry.to_audit_event() == event


@pytest.mark.parametrize("actor_type", list(AuditActorType))
def test_every_canonical_actor_round_trips(
    tmp_path: Path,
    actor_type: AuditActorType,
) -> None:
    event = _event(actor_type=actor_type)
    entry = AuditLedger(tmp_path / f"{actor_type.value}.jsonl").append(event).entry

    assert entry.actor_type is actor_type
    assert entry.to_audit_event().actor_type is actor_type


@pytest.mark.parametrize("outcome", list(AuditOutcome))
def test_every_outcome_and_failure_reason_round_trips(
    tmp_path: Path,
    outcome: AuditOutcome,
) -> None:
    event = _event(
        action=AuditAction.POLICY_CHANGE,
        outcome=outcome,
        reason_code=(None if outcome is AuditOutcome.SUCCESS else "POLICY_OPERATION_FAILED"),
        release_id=None,
    )
    entry = AuditLedger(tmp_path / f"{outcome.value}.jsonl").append(event).entry

    restored = entry.to_audit_event()
    assert restored.outcome is outcome
    assert restored.reason_code == event.reason_code


@pytest.mark.parametrize("decision", list(TrustDecision))
def test_every_exact_trust_decision_round_trips_without_scores_or_labels(
    tmp_path: Path,
    decision: TrustDecision,
) -> None:
    ledger = AuditLedger(tmp_path / f"{decision.value}.jsonl")
    event = _event(decision=decision)
    ledger.append(event)

    entry = ledger.read_entries()[0]
    assert entry.decision is decision
    assert entry.to_audit_event() == event
    serialized = ledger.path.read_text(encoding="utf-8")
    assert "probabilities" not in serialized
    assert "labels" not in serialized
    assert "waveform" not in serialized


def test_non_utc_aware_timestamp_is_canonically_normalized_and_round_trips(
    tmp_path: Path,
) -> None:
    offset_time = datetime(
        2026,
        8,
        24,
        7,
        30,
        45,
        123456,
        tzinfo=timezone(timedelta(hours=-5)),
    )
    event = _event(occurred_at=offset_time)
    entry = AuditLedger(tmp_path / "offset.jsonl").append(event).entry

    assert entry.timestamp_utc == "2026-08-24T12:30:45.123456Z"
    assert entry.to_audit_event().occurred_at == event.occurred_at


def test_canonical_contract_and_adapter_fail_closed_on_unknown_input() -> None:
    payload = _event().model_dump(mode="python")
    with pytest.raises(ValidationError, match="action"):
        AuditEvent.model_validate({**payload, "action": "RUN_INFERENCE"})
    with pytest.raises(ValidationError, match="actor_type"):
        AuditEvent.model_validate({**payload, "actor_type": "REVIEWER"})
    with pytest.raises(ValidationError, match="outcome"):
        AuditEvent.model_validate({**payload, "outcome": "UNKNOWN"})
    with pytest.raises(AuditValidationError, match="contracts.AuditEvent"):
        AuditEventAdapter.validate(cast(AuditEvent, object()))


def test_failed_and_denied_contracts_require_a_reason() -> None:
    payload = _event(
        action=AuditAction.POLICY_CHANGE,
        release_id=None,
    ).model_dump(mode="python")
    for outcome in (AuditOutcome.FAILED, AuditOutcome.DENIED):
        with pytest.raises(ValidationError, match="reason_code"):
            AuditEvent.model_validate({**payload, "outcome": outcome, "reason_code": None})


def test_successful_decision_contract_requires_release_decision_and_reason() -> None:
    payload = _event().model_dump(mode="python")
    for missing in ("release_id", "decision", "reason_code"):
        with pytest.raises(ValidationError, match="release, decision, and reason"):
            AuditEvent.model_validate({**payload, missing: None})


def test_prediction_allowed_reason_is_unambiguous() -> None:
    payload = _event().model_dump(mode="python")
    with pytest.raises(ValidationError, match="ALL_TRUST_GATES_PASSED"):
        AuditEvent.model_validate({**payload, "reason_code": "CONFIDENCE_GATE_ABSTAINED"})
    with pytest.raises(ValidationError, match="reserved"):
        AuditEvent.model_validate(
            {
                **payload,
                "decision": TrustDecision.ABSTAIN,
                "reason_code": "ALL_TRUST_GATES_PASSED",
            }
        )


@pytest.mark.parametrize(
    "safe_attributes",
    [
        {"raw_waveform": {"point": 1.0}},
        {"rows": {"row_one": 1}},
        {"patient_id": "patient-001"},
        {"record_id": "record-001"},
        {"local_path": "model.bin"},
        {"password": "not-allowed"},
        {"aggregate_counts": [1, 2, 3]},
        {"note": "C:\\private\\ecg.npy"},
        {"note": "/var/private/ecg.npy"},
        {"note": "../private/ecg.npy"},
        {"note": "~/private/ecg.npy"},
        {"note": "artifact at s3://private-bucket/ecg.npy"},
        {"note": "api_key=do-not-leak"},
        {"note": "Bearer do-not-leak"},
        {"note": "sk-proj-do-not-leak"},
        {"nested": {"subject_id": "subject-001"}},
    ],
)
def test_privacy_gate_rejects_identifiers_arrays_paths_and_secrets(
    safe_attributes: dict[str, object],
) -> None:
    with pytest.raises(AuditPrivacyError) as captured:
        assert_privacy_safe(safe_attributes)

    message = str(captured.value).lower()
    assert "patient-001" not in message
    assert "record-001" not in message
    assert "private" not in message
    assert "do-not-leak" not in message


def test_canonical_event_structurally_rejects_arbitrary_metadata() -> None:
    payload = _event().model_dump(mode="python")
    for field_name, value in (
        ("metadata", {"patient_id": "patient-1"}),
        ("safe_attributes", {"count": 3}),
        ("waveform", [0.1, 0.2]),
    ):
        with pytest.raises(ValidationError, match="extra_forbidden"):
            AuditEvent.model_validate({**payload, field_name: value})


def test_privacy_scan_accepts_scalar_aggregate_metadata() -> None:
    assert_privacy_safe(
        {
            "quality_status": "pass",
            "aggregate": {
                "event_count": 500,
                "ood_rate": 0.04,
                "restricted": False,
                "comment": None,
            },
        }
    )


def test_reason_code_cannot_encode_patient_or_record_identifiers() -> None:
    event = _event(
        action=AuditAction.POLICY_CHANGE,
        release_id=None,
        reason_code="PATIENT_ID_123",
    )
    with pytest.raises(AuditPrivacyError):
        AuditEventAdapter.validate(event)


def test_canonical_bytes_are_deterministic(tmp_path: Path) -> None:
    event = _event()
    first = AuditLedger(tmp_path / "first.jsonl")
    second = AuditLedger(tmp_path / "second.jsonl")

    first_receipt = first.append(event)
    second_receipt = second.append(event)

    assert first.path.read_bytes() == second.path.read_bytes()
    assert first_receipt.entry.entry_hash == second_receipt.entry.entry_hash
    assert first_receipt.checkpoint == second_receipt.checkpoint


def test_all_vocabularies_are_summarized_without_linkage_identifiers(
    tmp_path: Path,
) -> None:
    ledger = AuditLedger(tmp_path / "summary.jsonl")
    events = [
        _event("validation-001", action=AuditAction.CASE_VALIDATION),
        _event(
            "policy-001",
            action=AuditAction.POLICY_CHANGE,
            actor_type=AuditActorType.USER,
            release_id=None,
            reason_code=None,
        ),
        _event(
            "monitor-001",
            action=AuditAction.MONITORING_ACTION,
            actor_type=AuditActorType.SYSTEM,
            outcome=AuditOutcome.DENIED,
            release_id=None,
            reason_code="MONITORING_ACTION_DENIED",
        ),
    ]
    for event in events:
        ledger.append(event)

    summary = ledger.summarize()
    assert summary.event_count == 3
    assert summary.action_counts[AuditAction.CASE_VALIDATION.value] == 1
    assert summary.actor_type_counts[AuditActorType.USER.value] == 1
    assert summary.outcome_counts[AuditOutcome.SUCCESS.value] == 2
    assert summary.outcome_counts[AuditOutcome.DENIED.value] == 1
    assert summary.decision_counts[TrustDecision.PREDICTION_ALLOWED.value] == 1
    assert summary.events_without_decision == 2

    public_summary = canonical_json(summary.to_json_dict())
    for event in events:
        assert event.event_id not in public_summary
        assert event.request_id not in public_summary
        assert event.resource_id not in public_summary
    assert RELEASE_ID not in public_summary
    assert '"aggregate_only":true' in public_summary


def test_verifier_detects_mutation(tmp_path: Path) -> None:
    ledger = _ledger_with_entries(tmp_path / "mutated.jsonl")
    original = ledger.path.read_bytes()
    mutated = original.replace(b'"actor_type":"SERVICE"', b'"actor_type":"SYSTEM"', 1)
    assert mutated != original
    ledger.path.write_bytes(mutated)

    with pytest.raises(LedgerCorruptionError):
        ledger.verify()


def test_verifier_detects_reordered_or_deleted_middle_entries(tmp_path: Path) -> None:
    reordered = _ledger_with_entries(tmp_path / "reordered.jsonl")
    lines = reordered.path.read_bytes().splitlines(keepends=True)
    reordered.path.write_bytes(lines[1] + lines[0] + lines[2])
    with pytest.raises(LedgerCorruptionError):
        reordered.verify()

    deleted = _ledger_with_entries(tmp_path / "middle-delete.jsonl")
    lines = deleted.path.read_bytes().splitlines(keepends=True)
    deleted.path.write_bytes(lines[0] + lines[2])
    with pytest.raises(LedgerCorruptionError):
        deleted.verify()


def test_checkpoint_detects_deleted_valid_tail(tmp_path: Path) -> None:
    ledger = _ledger_with_entries(tmp_path / "tail-delete.jsonl")
    trusted_checkpoint = ledger.verify().checkpoint
    lines = ledger.path.read_bytes().splitlines(keepends=True)
    ledger.path.write_bytes(b"".join(lines[:-1]))

    with pytest.raises(LedgerCorruptionError, match="checkpoint"):
        ledger.verify(expected_checkpoint=trusted_checkpoint)


def test_verifier_detects_truncation_and_noncanonical_json(tmp_path: Path) -> None:
    truncated = _ledger_with_entries(tmp_path / "truncated.jsonl")
    truncated.path.write_bytes(truncated.path.read_bytes()[:-8])
    with pytest.raises(LedgerCorruptionError, match="truncated"):
        truncated.verify()

    pretty = _ledger_with_entries(tmp_path / "pretty.jsonl", count=1)
    parsed = cast(dict[str, object], json.loads(pretty.path.read_text(encoding="utf-8")))
    pretty.path.write_text(json.dumps(parsed, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(LedgerCorruptionError, match="canonical"):
        pretty.verify()


def test_missing_or_corrupted_checkpoint_fails_closed(tmp_path: Path) -> None:
    missing = _ledger_with_entries(tmp_path / "missing-checkpoint.jsonl", count=1)
    missing.checkpoint_path.unlink()
    with pytest.raises(LedgerCorruptionError, match="missing its checkpoint"):
        missing.verify()

    corrupted = _ledger_with_entries(tmp_path / "corrupt-checkpoint.jsonl", count=1)
    corrupted.checkpoint_path.write_bytes(b'{"entry_count":1')
    with pytest.raises(LedgerCorruptionError, match="checkpoint"):
        corrupted.verify()


def test_append_refuses_to_extend_corrupted_ledger(tmp_path: Path) -> None:
    ledger = _ledger_with_entries(tmp_path / "refuse.jsonl", count=2)
    ledger.path.write_bytes(ledger.path.read_bytes() + b'{"partial":')
    corrupted_bytes = ledger.path.read_bytes()

    with pytest.raises(LedgerCorruptionError):
        ledger.append(_event("request-new"))

    assert ledger.path.read_bytes() == corrupted_bytes


def test_concurrent_append_is_serialized_without_loss_or_duplicate_sequence(
    tmp_path: Path,
) -> None:
    ledger_path = tmp_path / "concurrent.jsonl"

    def append_one(index: int) -> None:
        AuditLedger(ledger_path).append(
            _event(
                f"request-{index:04d}",
                decision=TrustDecision.ABSTAIN,
            )
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(append_one, range(32)))

    ledger = AuditLedger(ledger_path)
    report = ledger.verify()
    entries = ledger.read_entries()
    assert report.entry_count == 32
    assert [entry.sequence for entry in entries] == list(range(32))
    assert {entry.request_id for entry in entries} == {
        f"request-{index:04d}" for index in range(32)
    }
    assert entries[0].previous_entry_hash == GENESIS_HASH
    assert all(
        entries[index].previous_entry_hash == entries[index - 1].entry_hash
        for index in range(1, len(entries))
    )


def test_empty_ledger_verifies_and_summarizes_without_identifiers(tmp_path: Path) -> None:
    ledger = AuditLedger(tmp_path / "empty.jsonl")
    report = ledger.verify()
    summary = ledger.summarize().to_json_dict()

    assert report.entry_count == 0
    assert report.head_hash == GENESIS_HASH
    assert summary["event_count"] == 0
    assert summary["head_hash"] == GENESIS_HASH
    assert summary["events_without_decision"] == 0


@pytest.mark.parametrize(
    "config",
    [
        AuditLedgerConfig(lock_timeout_seconds=0.0),
        AuditLedgerConfig(max_event_bytes=1024),
    ],
)
def test_valid_ledger_configuration_is_frozen(config: AuditLedgerConfig) -> None:
    assert config.lock_timeout_seconds >= 0.0
    assert config.max_event_bytes >= 1024


def test_invalid_ledger_configuration_is_rejected() -> None:
    with pytest.raises(AuditValidationError):
        AuditLedgerConfig(lock_timeout_seconds=-1.0)
    with pytest.raises(AuditValidationError):
        AuditLedgerConfig(max_event_bytes=100)
