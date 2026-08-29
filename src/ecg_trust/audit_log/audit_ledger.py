"""Tamper-evident, privacy-safe local audit ledger for Sentinel decisions.

The JSONL ledger is append-only. Each canonical entry commits to the full previous
entry hash, while a separately and atomically committed checkpoint makes suffix
deletion detectable. This is tamper evidence, not access control or a digital
signature; deployment must still protect the ledger and checkpoint permissions.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import re
import tempfile
from collections import Counter
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import RLock
from types import MappingProxyType
from typing import Final, cast

from filelock import FileLock
from filelock import Timeout as FileLockTimeout
from pydantic import ValidationError

from ecg_trust.contracts import (
    AUDIT_EVENT_SCHEMA_VERSION,
    AuditAction,
    AuditActorType,
    AuditEvent,
    AuditOutcome,
    TrustDecision,
)

LEDGER_SCHEMA_VERSION: Final = "sentinel-audit-ledger-v2"
CHECKPOINT_SCHEMA_VERSION: Final = "sentinel-audit-checkpoint-v1"
SUMMARY_SCHEMA_VERSION: Final = "sentinel-audit-summary-v1"
GENESIS_HASH: Final = hashlib.sha256(b"ecg-trust:sentinel-audit:genesis:v1").hexdigest()

_OPAQUE_ID_RE: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,159}$")
_REASON_CODE_RE: Final = re.compile(r"^[A-Z][A-Z0-9_]{0,95}$")
_SENSITIVE_REASON_RE: Final = re.compile(
    r"(?:^|_)(?:PATIENT|RECORD|SUBJECT|ECG|MRN)_(?:ID|IDENTIFIER|\d)(?:_|$)"
)
_SAFE_KEY_RE: Final = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_HASH_RE: Final = re.compile(r"^[a-f0-9]{64}$")
_UTC_TIMESTAMP_RE: Final = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$")
_ABSOLUTE_PATH_RE: Final = re.compile(
    r"(?:^|[\s\"'=])(?:[A-Za-z]:[\\/]|[/\\]{2}|/|~[\\/]|\.{1,2}[\\/]|"
    r"[A-Za-z][A-Za-z0-9+.-]*://)",
    re.I,
)
_SECRET_VALUE_RE: Final = re.compile(
    r"(?:\b(?:password|passwd|secret|api[_-]?key|access[_-]?token|refresh[_-]?token|"
    r"bearer|authorization|credential)\b\s*(?::|=|\s)\s*\S+|"
    r"\b(?:sk|pk)_(?:live|test)_[A-Za-z0-9]{8,}|\bsk-[A-Za-z0-9_-]{8,}|"
    r"\bgh[pousr]_[A-Za-z0-9]{8,}|\bAKIA[A-Z0-9]{12,}|"
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----|"
    r"^eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.)",
    re.I,
)
_FORBIDDEN_EXACT_KEYS: Final = frozenset(
    {
        "absolute_path",
        "access_key",
        "api_key",
        "authorization",
        "credential",
        "ecg_id",
        "file_path",
        "filepath",
        "filename",
        "lead_values",
        "local_path",
        "mrn",
        "password",
        "passwd",
        "patient",
        "patient_id",
        "private_key",
        "raw",
        "raw_signal",
        "raw_waveform",
        "record_id",
        "request_id",
        "release_id",
        "row",
        "row_data",
        "rows",
        "sample_values",
        "samples",
        "secret",
        "signal",
        "signals",
        "signing_key",
        "subject_id",
        "token",
        "waveform",
        "waveforms",
    }
)
_FORBIDDEN_KEY_SUFFIXES: Final = (
    "_absolute_path",
    "_api_key",
    "_credential",
    "_file_path",
    "_local_path",
    "_path",
    "_password",
    "_patient_id",
    "_record_id",
    "_request_id",
    "_release_id",
    "_secret",
    "_subject_id",
    "_token",
)
_ENTRY_KEYS: Final = frozenset(
    {
        "schema_version",
        "sequence",
        "event_id",
        "timestamp_utc",
        "action",
        "actor_type",
        "actor_id",
        "request_id",
        "resource_type",
        "resource_id",
        "release_id",
        "outcome",
        "decision",
        "reason_codes",
        "safe_attributes",
        "previous_entry_hash",
        "entry_hash",
    }
)
_CHECKPOINT_KEYS: Final = frozenset({"schema_version", "entry_count", "head_hash"})


class AuditLedgerError(RuntimeError):
    """Base class for audit-ledger failures."""


class AuditValidationError(AuditLedgerError):
    """An event violates the closed ledger contract."""


class AuditPrivacyError(AuditValidationError):
    """An event attempts to include forbidden or non-aggregate content."""


class LedgerCorruptionError(AuditLedgerError):
    """The ledger or its durable checkpoint cannot be trusted."""


class AuditStorageError(AuditLedgerError):
    """A local durability or locking operation failed."""


def canonical_json(value: object) -> str:
    """Return the sole permitted deterministic JSON representation."""

    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise AuditValidationError("value is not canonical JSON data") from error


def _validated_opaque_id(value: object, *, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or _OPAQUE_ID_RE.fullmatch(value) is None:
        raise AuditValidationError(f"{field_name} must be a bounded opaque identifier")
    return value


def _validated_reason_codes(value: object) -> tuple[str, ...]:
    if not isinstance(value, (tuple, list)) or len(value) > 1:
        raise AuditValidationError("zero or one canonical reason code is permitted")
    reasons: list[str] = []
    for reason in value:
        if not isinstance(reason, str) or _REASON_CODE_RE.fullmatch(reason) is None:
            raise AuditValidationError("reason codes must use the bounded machine format")
        if _SENSITIVE_REASON_RE.search(reason) is not None:
            raise AuditPrivacyError("reason codes cannot encode patient or record identifiers")
        reasons.append(reason)
    return tuple(reasons)


def _validated_utc_datetime(value: object) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise AuditValidationError("timestamp_utc must be timezone-aware UTC")
    try:
        offset = value.utcoffset()
    except Exception as error:
        raise AuditValidationError("timestamp_utc must have a valid UTC offset") from error
    if offset != timedelta(0):
        raise AuditValidationError("timestamp_utc must use UTC")
    return value.astimezone(UTC)


def _format_utc(value: datetime) -> str:
    validated = _validated_utc_datetime(value)
    return validated.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _parse_utc(value: object) -> datetime:
    if not isinstance(value, str) or _UTC_TIMESTAMP_RE.fullmatch(value) is None:
        raise LedgerCorruptionError("ledger entry has a non-canonical UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except ValueError as error:
        raise LedgerCorruptionError("ledger entry has an invalid UTC timestamp") from error
    if _format_utc(parsed) != value:
        raise LedgerCorruptionError("ledger entry has a non-canonical UTC timestamp")
    return parsed


def _forbidden_key(key: str) -> bool:
    normalized = key.lower()
    return (
        normalized in _FORBIDDEN_EXACT_KEYS
        or normalized == "id"
        or normalized.endswith(("_id", "_identifier"))
        or normalized.endswith(_FORBIDDEN_KEY_SUFFIXES)
    )


def _validated_safe_string(value: str) -> str:
    if len(value) > 256:
        raise AuditPrivacyError("safe attribute strings must be at most 256 characters")
    if _ABSOLUTE_PATH_RE.search(value) is not None:
        raise AuditPrivacyError("absolute path values are forbidden in audit events")
    if _SECRET_VALUE_RE.search(value) is not None:
        raise AuditPrivacyError("secret-like values are forbidden in audit events")
    return value


def _normalize_safe_mapping(
    value: object,
    *,
    depth: int = 0,
    node_count: list[int] | None = None,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise AuditPrivacyError("safe_attributes must be a mapping")
    if depth > 4:
        raise AuditPrivacyError("safe_attributes nesting exceeds the privacy limit")
    if len(value) > 64:
        raise AuditPrivacyError("safe_attributes contains too many keys")
    counter = node_count if node_count is not None else [0]
    normalized: dict[str, object] = {}
    for raw_key, raw_value in value.items():
        counter[0] += 1
        if counter[0] > 256:
            raise AuditPrivacyError("safe_attributes exceeds the aggregate-field limit")
        if not isinstance(raw_key, str) or _SAFE_KEY_RE.fullmatch(raw_key) is None:
            raise AuditPrivacyError("safe attribute keys must use the bounded machine format")
        if _forbidden_key(raw_key):
            raise AuditPrivacyError("a forbidden audit attribute key was supplied")
        if isinstance(raw_value, Mapping):
            normalized[raw_key] = _normalize_safe_mapping(
                raw_value,
                depth=depth + 1,
                node_count=counter,
            )
        elif isinstance(raw_value, (list, tuple, set, frozenset)):
            # Audit metadata uses scalar leaves only. This prevents waveforms and row arrays
            # even when a caller attempts to hide them under an innocuous key.
            raise AuditPrivacyError("array values are forbidden in audit events")
        elif isinstance(raw_value, str):
            normalized[raw_key] = _validated_safe_string(raw_value)
        elif raw_value is None or isinstance(raw_value, bool):
            normalized[raw_key] = raw_value
        elif isinstance(raw_value, int):
            if not -(2**63) <= raw_value < 2**63:
                raise AuditValidationError("safe attribute integers must fit in signed 64 bits")
            normalized[raw_key] = raw_value
        elif isinstance(raw_value, float):
            if not math.isfinite(raw_value):
                raise AuditValidationError("safe attribute numbers must be finite")
            normalized[raw_key] = raw_value
        else:
            raise AuditPrivacyError("non-scalar audit attribute values are forbidden")
    return normalized


def assert_privacy_safe(attributes: Mapping[str, object]) -> None:
    """Raise if attributes could contain raw rows, identifiers, paths, or secrets."""

    _normalize_safe_mapping(attributes)


def _attributes_from_canonical(value: str) -> dict[str, object]:
    parsed = cast(object, json.loads(value))
    if not isinstance(parsed, dict):
        raise AuditValidationError("canonical safe attributes must decode to an object")
    return cast(dict[str, object], parsed)


@dataclass(frozen=True, slots=True)
class _LedgerEvent:
    """Validated storage envelope produced only from the canonical contract."""

    event_id: str
    action: AuditAction
    actor_type: AuditActorType
    actor_id: str
    timestamp_utc: datetime
    request_id: str
    resource_type: str
    resource_id: str
    release_id: str | None
    outcome: AuditOutcome
    decision: TrustDecision | None
    reason_codes: tuple[str, ...]
    _canonical_safe_attributes: str = field(repr=False)

    def safe_attributes_dict(self) -> dict[str, object]:
        return _attributes_from_canonical(self._canonical_safe_attributes)


def _adapt_contract_audit_event(
    event: AuditEvent,
    *,
    safe_attributes: Mapping[str, object] | None = None,
) -> _LedgerEvent:
    """Strictly translate the public event into the private ledger envelope."""

    if not isinstance(event, AuditEvent):
        raise AuditValidationError("adapter requires ecg_trust.contracts.AuditEvent")
    event_id = _validated_opaque_id(event.event_id, field_name="event_id")
    request_id = _validated_opaque_id(event.request_id, field_name="request_id")
    actor_id = _validated_opaque_id(event.actor_id, field_name="actor_id")
    resource_id = _validated_opaque_id(event.resource_id, field_name="resource_id")
    release_id = _validated_opaque_id(event.release_id, field_name="release_id")
    if event_id is None or request_id is None or actor_id is None or resource_id is None:
        raise AuditValidationError("canonical audit identifiers must be present")
    if not isinstance(event.action, AuditAction):
        raise AuditValidationError("action must use the canonical audit vocabulary")
    if not isinstance(event.actor_type, AuditActorType):
        raise AuditValidationError("actor_type must use the canonical actor vocabulary")
    if not isinstance(event.outcome, AuditOutcome):
        raise AuditValidationError("outcome must use the canonical outcome vocabulary")
    if event.decision is not None and not isinstance(event.decision, TrustDecision):
        raise AuditValidationError("decision must use the canonical five-state vocabulary")
    reasons = _validated_reason_codes(() if event.reason_code is None else (event.reason_code,))
    timestamp = event.occurred_at.astimezone(UTC)
    _validated_utc_datetime(timestamp)
    normalized = _normalize_safe_mapping(safe_attributes or {})
    return _LedgerEvent(
        event_id=event_id,
        action=event.action,
        actor_type=event.actor_type,
        actor_id=actor_id,
        timestamp_utc=timestamp,
        request_id=request_id,
        resource_type=event.resource_type,
        resource_id=resource_id,
        release_id=release_id,
        outcome=event.outcome,
        decision=event.decision,
        reason_codes=reasons,
        _canonical_safe_attributes=canonical_json(normalized),
    )


@dataclass(frozen=True, slots=True)
class AuditEntry:
    """One immutable, hash-chained JSONL envelope."""

    sequence: int
    event_id: str
    timestamp_utc: str
    action: AuditAction
    actor_type: AuditActorType
    actor_id: str
    request_id: str
    resource_type: str
    resource_id: str
    release_id: str | None
    outcome: AuditOutcome
    decision: TrustDecision | None
    reason_codes: tuple[str, ...]
    _canonical_safe_attributes: str = field(repr=False)
    previous_entry_hash: str
    entry_hash: str

    def safe_attributes_dict(self) -> dict[str, object]:
        return _attributes_from_canonical(self._canonical_safe_attributes)

    def to_mapping(self, *, include_entry_hash: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": LEDGER_SCHEMA_VERSION,
            "sequence": self.sequence,
            "event_id": self.event_id,
            "timestamp_utc": self.timestamp_utc,
            "action": self.action.value,
            "actor_type": self.actor_type.value,
            "actor_id": self.actor_id,
            "request_id": self.request_id,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "release_id": self.release_id,
            "outcome": self.outcome.value,
            "decision": self.decision.value if self.decision is not None else None,
            "reason_codes": list(self.reason_codes),
            "safe_attributes": self.safe_attributes_dict(),
            "previous_entry_hash": self.previous_entry_hash,
        }
        if include_entry_hash:
            payload["entry_hash"] = self.entry_hash
        return payload

    def canonical_line(self) -> bytes:
        return (canonical_json(self.to_mapping()) + "\n").encode("utf-8")

    def to_audit_event(self) -> AuditEvent:
        """Restore the single canonical public audit contract."""

        reason_code = self.reason_codes[0] if self.reason_codes else None
        try:
            return AuditEvent(
                schema_version=AUDIT_EVENT_SCHEMA_VERSION,
                event_id=self.event_id,
                occurred_at=_parse_utc(self.timestamp_utc),
                request_id=self.request_id,
                actor_type=self.actor_type,
                actor_id=self.actor_id,
                action=self.action,
                resource_type=self.resource_type,
                resource_id=self.resource_id,
                release_id=self.release_id,
                outcome=self.outcome,
                reason_code=reason_code,
                decision=self.decision,
            )
        except ValidationError as error:
            raise LedgerCorruptionError(
                "ledger entry cannot be restored as the canonical audit contract"
            ) from error


class AuditEventAdapter:
    """Explicit boundary between the public contract and private storage envelope."""

    @staticmethod
    def validate(event: AuditEvent) -> None:
        """Validate canonical vocabulary, identifier bounds, reasons, and privacy."""

        _adapt_contract_audit_event(event)

    @staticmethod
    def restore(entry: AuditEntry) -> AuditEvent:
        """Round-trip a verified ledger entry to the canonical public contract."""

        if not isinstance(entry, AuditEntry):
            raise AuditValidationError("restore requires a verified AuditEntry")
        return entry.to_audit_event()


@dataclass(frozen=True, slots=True)
class LedgerCheckpoint:
    """Durable external anchor needed to detect deletion of a valid suffix."""

    entry_count: int
    head_hash: str

    def __post_init__(self) -> None:
        if isinstance(self.entry_count, bool) or not isinstance(self.entry_count, int):
            raise AuditValidationError("checkpoint entry_count must be an integer")
        if self.entry_count < 0:
            raise AuditValidationError("checkpoint entry_count cannot be negative")
        if not isinstance(self.head_hash, str) or _HASH_RE.fullmatch(self.head_hash) is None:
            raise AuditValidationError("checkpoint head_hash must be a SHA-256 digest")
        if self.entry_count == 0 and self.head_hash != GENESIS_HASH:
            raise AuditValidationError("an empty checkpoint must use the genesis hash")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "entry_count": self.entry_count,
            "head_hash": self.head_hash,
        }

    def canonical_line(self) -> bytes:
        return (canonical_json(self.to_mapping()) + "\n").encode("utf-8")


@dataclass(frozen=True, slots=True)
class VerificationReport:
    """Successful integrity result. Corruption is represented by an exception."""

    entry_count: int
    head_hash: str

    @property
    def checkpoint(self) -> LedgerCheckpoint:
        return LedgerCheckpoint(entry_count=self.entry_count, head_hash=self.head_hash)


@dataclass(frozen=True, slots=True)
class AuditAppendReceipt:
    """Receipt that callers may retain as an additional trusted anchor."""

    entry: AuditEntry
    checkpoint: LedgerCheckpoint


@dataclass(frozen=True, slots=True)
class AuditSummary:
    """Aggregate-only audit view; identifiers and per-event attributes are omitted."""

    event_count: int
    head_hash: str
    action_counts: Mapping[str, int]
    actor_type_counts: Mapping[str, int]
    outcome_counts: Mapping[str, int]
    decision_counts: Mapping[str, int]
    events_without_decision: int
    reason_code_counts: Mapping[str, int]

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "aggregate_only": True,
            "event_count": self.event_count,
            "head_hash": self.head_hash,
            "action_counts": dict(sorted(self.action_counts.items())),
            "actor_type_counts": dict(sorted(self.actor_type_counts.items())),
            "outcome_counts": dict(sorted(self.outcome_counts.items())),
            "decision_counts": dict(sorted(self.decision_counts.items())),
            "events_without_decision": self.events_without_decision,
            "reason_code_counts": dict(sorted(self.reason_code_counts.items())),
        }


@dataclass(frozen=True, slots=True)
class AuditLedgerConfig:
    """Validated durability and resource limits."""

    lock_timeout_seconds: float = 10.0
    max_event_bytes: int = 32_768

    def __post_init__(self) -> None:
        if (
            isinstance(self.lock_timeout_seconds, bool)
            or not isinstance(self.lock_timeout_seconds, (int, float))
            or not math.isfinite(float(self.lock_timeout_seconds))
            or float(self.lock_timeout_seconds) < 0.0
        ):
            raise AuditValidationError("lock_timeout_seconds must be finite and non-negative")
        if (
            isinstance(self.max_event_bytes, bool)
            or not isinstance(self.max_event_bytes, int)
            or not 1024 <= self.max_event_bytes <= 1_048_576
        ):
            raise AuditValidationError("max_event_bytes must be between 1024 and 1048576")


def _entry_from_event(
    event: _LedgerEvent,
    *,
    sequence: int,
    previous_entry_hash: str,
) -> AuditEntry:
    unsigned_payload: dict[str, object] = {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "sequence": sequence,
        "event_id": event.event_id,
        "timestamp_utc": _format_utc(event.timestamp_utc),
        "action": event.action.value,
        "actor_type": event.actor_type.value,
        "actor_id": event.actor_id,
        "request_id": event.request_id,
        "resource_type": event.resource_type,
        "resource_id": event.resource_id,
        "release_id": event.release_id,
        "outcome": event.outcome.value,
        "decision": event.decision.value if event.decision is not None else None,
        "reason_codes": list(event.reason_codes),
        "safe_attributes": event.safe_attributes_dict(),
        "previous_entry_hash": previous_entry_hash,
    }
    entry_hash = hashlib.sha256(canonical_json(unsigned_payload).encode("utf-8")).hexdigest()
    return AuditEntry(
        sequence=sequence,
        event_id=event.event_id,
        timestamp_utc=cast(str, unsigned_payload["timestamp_utc"]),
        action=event.action,
        actor_type=event.actor_type,
        actor_id=event.actor_id,
        request_id=event.request_id,
        resource_type=event.resource_type,
        resource_id=event.resource_id,
        release_id=event.release_id,
        outcome=event.outcome,
        decision=event.decision,
        reason_codes=event.reason_codes,
        _canonical_safe_attributes=canonical_json(event.safe_attributes_dict()),
        previous_entry_hash=previous_entry_hash,
        entry_hash=entry_hash,
    )


def _required_mapping(value: object, *, context: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise LedgerCorruptionError(f"{context} must be a JSON object")
    return cast(dict[str, object], value)


def _required_string(mapping: Mapping[str, object], key: str) -> str:
    value = mapping[key]
    if not isinstance(value, str):
        raise LedgerCorruptionError(f"ledger entry field {key} has an invalid type")
    return value


def _entry_from_mapping(mapping: dict[str, object]) -> AuditEntry:
    if frozenset(mapping) != _ENTRY_KEYS:
        raise LedgerCorruptionError("ledger entry fields do not match the closed schema")
    if mapping["schema_version"] != LEDGER_SCHEMA_VERSION:
        raise LedgerCorruptionError("ledger entry schema version is unsupported")
    sequence = mapping["sequence"]
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
        raise LedgerCorruptionError("ledger entry sequence is invalid")
    timestamp = _required_string(mapping, "timestamp_utc")
    parsed_timestamp = _parse_utc(timestamp)
    try:
        action = AuditAction(_required_string(mapping, "action"))
        actor_type = AuditActorType(_required_string(mapping, "actor_type"))
        outcome = AuditOutcome(_required_string(mapping, "outcome"))
    except ValueError as error:
        raise LedgerCorruptionError("ledger entry uses an unsupported enum value") from error
    event_id = _validated_opaque_id(mapping["event_id"], field_name="event_id")
    actor_id = _validated_opaque_id(mapping["actor_id"], field_name="actor_id")
    request_id = _validated_opaque_id(mapping["request_id"], field_name="request_id")
    resource_type = _required_string(mapping, "resource_type")
    resource_id = _validated_opaque_id(mapping["resource_id"], field_name="resource_id")
    release_id = _validated_opaque_id(mapping["release_id"], field_name="release_id")
    if event_id is None or actor_id is None or request_id is None or resource_id is None:
        raise LedgerCorruptionError("ledger entry is missing a canonical identifier")
    raw_decision = mapping["decision"]
    if raw_decision is None:
        decision = None
    elif isinstance(raw_decision, str):
        try:
            decision = TrustDecision(raw_decision)
        except ValueError as error:
            raise LedgerCorruptionError("ledger entry decision is unsupported") from error
    else:
        raise LedgerCorruptionError("ledger entry decision has an invalid type")
    reasons = _validated_reason_codes(mapping["reason_codes"])
    attributes = _required_mapping(mapping["safe_attributes"], context="safe_attributes")
    try:
        contract_event = AuditEvent(
            schema_version=AUDIT_EVENT_SCHEMA_VERSION,
            event_id=event_id,
            occurred_at=parsed_timestamp,
            request_id=request_id,
            actor_type=actor_type,
            actor_id=actor_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            release_id=release_id,
            outcome=outcome,
            reason_code=reasons[0] if reasons else None,
            decision=decision,
        )
        event = _adapt_contract_audit_event(contract_event, safe_attributes=attributes)
    except (AuditValidationError, ValidationError) as error:
        raise LedgerCorruptionError("ledger entry violates the event contract") from error
    previous_hash = _required_string(mapping, "previous_entry_hash")
    entry_hash = _required_string(mapping, "entry_hash")
    if _HASH_RE.fullmatch(previous_hash) is None or _HASH_RE.fullmatch(entry_hash) is None:
        raise LedgerCorruptionError("ledger entry contains an invalid hash")
    return AuditEntry(
        sequence=sequence,
        event_id=event.event_id,
        timestamp_utc=timestamp,
        action=action,
        actor_type=actor_type,
        actor_id=event.actor_id,
        request_id=request_id,
        resource_type=event.resource_type,
        resource_id=event.resource_id,
        release_id=release_id,
        outcome=outcome,
        decision=decision,
        reason_codes=reasons,
        _canonical_safe_attributes=canonical_json(event.safe_attributes_dict()),
        previous_entry_hash=previous_hash,
        entry_hash=entry_hash,
    )


def _checkpoint_from_mapping(mapping: dict[str, object]) -> LedgerCheckpoint:
    if frozenset(mapping) != _CHECKPOINT_KEYS:
        raise LedgerCorruptionError("checkpoint fields do not match the closed schema")
    if mapping["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise LedgerCorruptionError("checkpoint schema version is unsupported")
    entry_count = mapping["entry_count"]
    head_hash = mapping["head_hash"]
    try:
        return LedgerCheckpoint(
            entry_count=cast(int, entry_count),
            head_hash=cast(str, head_hash),
        )
    except AuditValidationError as error:
        raise LedgerCorruptionError("checkpoint values are invalid") from error


class AuditLedger:
    """Single-host, file-locked, durable append interface for a JSONL ledger."""

    def __init__(self, path: Path, *, config: AuditLedgerConfig | None = None) -> None:
        self._path = path
        self._checkpoint_path = path.with_name(path.name + ".checkpoint")
        self._lock_path = path.with_name(path.name + ".lock")
        self._config = config or AuditLedgerConfig()
        self._thread_lock = RLock()
        self._file_lock = FileLock(
            str(self._lock_path),
            timeout=float(self._config.lock_timeout_seconds),
        )

    @property
    def path(self) -> Path:
        return self._path

    @property
    def checkpoint_path(self) -> Path:
        return self._checkpoint_path

    @contextmanager
    def _exclusive(self) -> Iterator[None]:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise AuditStorageError("audit ledger parent directory is unavailable") from error
        with self._thread_lock:
            try:
                self._file_lock.acquire()
            except FileLockTimeout as error:
                raise AuditStorageError("timed out acquiring the audit ledger lock") from error
            try:
                yield
            finally:
                self._file_lock.release()

    def _read_entries_unlocked(self) -> tuple[AuditEntry, ...]:
        if not self._path.exists():
            return ()
        entries: list[AuditEntry] = []
        expected_previous = GENESIS_HASH
        try:
            with self._path.open("rb") as stream:
                while True:
                    raw_line = stream.readline(self._config.max_event_bytes + 2)
                    if raw_line == b"":
                        break
                    line_number = len(entries) + 1
                    if len(raw_line) > self._config.max_event_bytes + 1:
                        raise LedgerCorruptionError(
                            f"ledger entry {line_number} exceeds the byte limit"
                        )
                    if not raw_line.endswith(b"\n"):
                        raise LedgerCorruptionError(f"ledger entry {line_number} is truncated")
                    canonical_bytes = raw_line[:-1]
                    if not canonical_bytes:
                        raise LedgerCorruptionError(f"ledger entry {line_number} is blank")
                    try:
                        parsed = cast(object, json.loads(canonical_bytes.decode("utf-8")))
                    except (UnicodeDecodeError, json.JSONDecodeError) as error:
                        raise LedgerCorruptionError(
                            f"ledger entry {line_number} is not valid UTF-8 JSON"
                        ) from error
                    mapping = _required_mapping(parsed, context="ledger entry")
                    try:
                        recanonicalized = canonical_json(mapping).encode("utf-8")
                    except AuditValidationError as error:
                        raise LedgerCorruptionError(
                            f"ledger entry {line_number} is not canonical JSON"
                        ) from error
                    if recanonicalized != canonical_bytes:
                        raise LedgerCorruptionError(
                            f"ledger entry {line_number} is not canonical JSON"
                        )
                    try:
                        entry = _entry_from_mapping(mapping)
                    except AuditValidationError as error:
                        raise LedgerCorruptionError(
                            f"ledger entry {line_number} violates the ledger contract"
                        ) from error
                    if entry.sequence != len(entries):
                        raise LedgerCorruptionError(
                            f"ledger entry {line_number} has a discontinuous sequence"
                        )
                    if not hmac.compare_digest(entry.previous_entry_hash, expected_previous):
                        raise LedgerCorruptionError(
                            f"ledger entry {line_number} breaks the previous-hash chain"
                        )
                    expected_hash = hashlib.sha256(
                        canonical_json(entry.to_mapping(include_entry_hash=False)).encode("utf-8")
                    ).hexdigest()
                    if not hmac.compare_digest(entry.entry_hash, expected_hash):
                        raise LedgerCorruptionError(
                            f"ledger entry {line_number} has an invalid self hash"
                        )
                    entries.append(entry)
                    expected_previous = entry.entry_hash
        except LedgerCorruptionError:
            raise
        except OSError as error:
            raise AuditStorageError("audit ledger could not be read") from error
        return tuple(entries)

    def _read_checkpoint_unlocked(self) -> LedgerCheckpoint | None:
        if not self._checkpoint_path.exists():
            return None
        try:
            raw = self._checkpoint_path.read_bytes()
        except OSError as error:
            raise AuditStorageError("audit checkpoint could not be read") from error
        if not raw.endswith(b"\n") or raw.count(b"\n") != 1:
            raise LedgerCorruptionError("audit checkpoint is truncated or contains extra data")
        try:
            parsed = cast(object, json.loads(raw[:-1].decode("utf-8")))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise LedgerCorruptionError("audit checkpoint is not valid UTF-8 JSON") from error
        mapping = _required_mapping(parsed, context="audit checkpoint")
        if canonical_json(mapping).encode("utf-8") != raw[:-1]:
            raise LedgerCorruptionError("audit checkpoint is not canonical JSON")
        return _checkpoint_from_mapping(mapping)

    def _verified_state_unlocked(self) -> tuple[tuple[AuditEntry, ...], VerificationReport]:
        entries = self._read_entries_unlocked()
        head_hash = entries[-1].entry_hash if entries else GENESIS_HASH
        report = VerificationReport(entry_count=len(entries), head_hash=head_hash)
        persisted = self._read_checkpoint_unlocked()
        if persisted is None:
            if entries:
                raise LedgerCorruptionError("non-empty audit ledger is missing its checkpoint")
        elif persisted != report.checkpoint:
            raise LedgerCorruptionError("audit ledger does not match its durable checkpoint")
        return entries, report

    def verify(
        self,
        *,
        expected_checkpoint: LedgerCheckpoint | None = None,
    ) -> VerificationReport:
        """Verify schema, privacy, canonical bytes, chain, hashes, and checkpoint."""

        with self._exclusive():
            _, report = self._verified_state_unlocked()
            if expected_checkpoint is not None and report.checkpoint != expected_checkpoint:
                raise LedgerCorruptionError("audit ledger does not match the expected checkpoint")
            return report

    def read_entries(self) -> tuple[AuditEntry, ...]:
        """Return entries only after complete integrity and checkpoint verification."""

        with self._exclusive():
            entries, _ = self._verified_state_unlocked()
            return entries

    def append(self, event: AuditEvent) -> AuditAppendReceipt:
        """Adapt a canonical event, then durably append its private envelope."""

        if not isinstance(event, AuditEvent):
            raise AuditValidationError("append requires ecg_trust.contracts.AuditEvent")
        adapted = _adapt_contract_audit_event(event)
        with self._exclusive():
            entries, report = self._verified_state_unlocked()
            entry = _entry_from_event(
                adapted,
                sequence=len(entries),
                previous_entry_hash=report.head_hash,
            )
            line = entry.canonical_line()
            if len(line) > self._config.max_event_bytes + 1:
                raise AuditValidationError("canonical audit event exceeds the byte limit")
            self._append_line_unlocked(line)
            checkpoint = LedgerCheckpoint(
                entry_count=len(entries) + 1,
                head_hash=entry.entry_hash,
            )
            self._write_checkpoint_unlocked(checkpoint)
            return AuditAppendReceipt(entry=entry, checkpoint=checkpoint)

    def _append_line_unlocked(self, line: bytes) -> None:
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_BINARY", 0)
        descriptor: int | None = None
        try:
            descriptor = os.open(self._path, flags, 0o600)
            written = os.write(descriptor, line)
            if written != len(line):
                raise OSError("partial audit append")
            os.fsync(descriptor)
        except OSError as error:
            raise AuditStorageError("audit entry could not be durably appended") from error
        finally:
            if descriptor is not None:
                os.close(descriptor)

    def _write_checkpoint_unlocked(self, checkpoint: LedgerCheckpoint) -> None:
        descriptor: int | None = None
        temporary_name: str | None = None
        try:
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{self._checkpoint_path.name}.",
                suffix=".tmp",
                dir=self._checkpoint_path.parent,
            )
            payload = checkpoint.canonical_line()
            written = os.write(descriptor, payload)
            if written != len(payload):
                raise OSError("partial checkpoint write")
            os.fsync(descriptor)
            os.close(descriptor)
            descriptor = None
            os.replace(temporary_name, self._checkpoint_path)
            temporary_name = None
            self._fsync_parent_directory()
        except OSError as error:
            raise AuditStorageError("audit checkpoint could not be atomically committed") from error
        finally:
            if descriptor is not None:
                os.close(descriptor)
            if temporary_name is not None:
                with suppress(OSError):
                    Path(temporary_name).unlink(missing_ok=True)

    def _fsync_parent_directory(self) -> None:
        if os.name == "nt":
            return
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        descriptor: int | None = None
        try:
            descriptor = os.open(self._path.parent, flags)
            os.fsync(descriptor)
        except OSError as error:
            raise AuditStorageError("audit directory metadata could not be synchronized") from error
        finally:
            if descriptor is not None:
                os.close(descriptor)

    def summarize(self) -> AuditSummary:
        """Return aggregate counts without IDs, timestamps, or event attributes."""

        with self._exclusive():
            entries, report = self._verified_state_unlocked()
            action_counts = Counter(entry.action.value for entry in entries)
            actor_counts = Counter(entry.actor_type.value for entry in entries)
            outcome_counts = Counter(entry.outcome.value for entry in entries)
            decision_counts = Counter(
                entry.decision.value for entry in entries if entry.decision is not None
            )
            for action in AuditAction:
                action_counts.setdefault(action.value, 0)
            for actor_type in AuditActorType:
                actor_counts.setdefault(actor_type.value, 0)
            for outcome in AuditOutcome:
                outcome_counts.setdefault(outcome.value, 0)
            for decision in TrustDecision:
                decision_counts.setdefault(decision.value, 0)
            reason_counts = Counter(reason for entry in entries for reason in entry.reason_codes)
            without_decision = sum(entry.decision is None for entry in entries)
            return AuditSummary(
                event_count=len(entries),
                head_hash=report.head_hash,
                action_counts=MappingProxyType(dict(action_counts)),
                actor_type_counts=MappingProxyType(dict(actor_counts)),
                outcome_counts=MappingProxyType(dict(outcome_counts)),
                decision_counts=MappingProxyType(dict(decision_counts)),
                events_without_decision=without_decision,
                reason_code_counts=MappingProxyType(dict(reason_counts)),
            )
