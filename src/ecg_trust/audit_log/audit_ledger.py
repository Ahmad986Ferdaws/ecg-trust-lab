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
import tempfile
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from types import MappingProxyType
from typing import cast

from filelock import FileLock
from filelock import Timeout as FileLockTimeout

from ecg_trust.audit_log._codec import CHECKPOINT_SCHEMA_VERSION as CHECKPOINT_SCHEMA_VERSION
from ecg_trust.audit_log._codec import GENESIS_HASH as GENESIS_HASH
from ecg_trust.audit_log._codec import LEDGER_SCHEMA_VERSION as LEDGER_SCHEMA_VERSION
from ecg_trust.audit_log._codec import SUMMARY_SCHEMA_VERSION as SUMMARY_SCHEMA_VERSION
from ecg_trust.audit_log._codec import AuditAppendReceipt as AuditAppendReceipt
from ecg_trust.audit_log._codec import AuditEntry as AuditEntry
from ecg_trust.audit_log._codec import AuditEventAdapter as AuditEventAdapter
from ecg_trust.audit_log._codec import AuditLedgerError as AuditLedgerError
from ecg_trust.audit_log._codec import AuditPrivacyError as AuditPrivacyError
from ecg_trust.audit_log._codec import AuditStorageError as AuditStorageError
from ecg_trust.audit_log._codec import AuditSummary as AuditSummary
from ecg_trust.audit_log._codec import AuditValidationError as AuditValidationError
from ecg_trust.audit_log._codec import LedgerCheckpoint as LedgerCheckpoint
from ecg_trust.audit_log._codec import LedgerCorruptionError as LedgerCorruptionError
from ecg_trust.audit_log._codec import VerificationReport as VerificationReport
from ecg_trust.audit_log._codec import _adapt_contract_audit_event as _adapt_contract_audit_event
from ecg_trust.audit_log._codec import _attributes_from_canonical as _attributes_from_canonical
from ecg_trust.audit_log._codec import _checkpoint_from_mapping as _checkpoint_from_mapping
from ecg_trust.audit_log._codec import _entry_from_event as _entry_from_event
from ecg_trust.audit_log._codec import _entry_from_mapping as _entry_from_mapping
from ecg_trust.audit_log._codec import _forbidden_key as _forbidden_key
from ecg_trust.audit_log._codec import _format_utc as _format_utc
from ecg_trust.audit_log._codec import _LedgerEvent as _LedgerEvent
from ecg_trust.audit_log._codec import _normalize_safe_mapping as _normalize_safe_mapping
from ecg_trust.audit_log._codec import _parse_utc as _parse_utc
from ecg_trust.audit_log._codec import _required_mapping as _required_mapping
from ecg_trust.audit_log._codec import _required_string as _required_string
from ecg_trust.audit_log._codec import _validated_opaque_id as _validated_opaque_id
from ecg_trust.audit_log._codec import _validated_reason_codes as _validated_reason_codes
from ecg_trust.audit_log._codec import _validated_safe_string as _validated_safe_string
from ecg_trust.audit_log._codec import _validated_utc_datetime as _validated_utc_datetime
from ecg_trust.audit_log._codec import assert_privacy_safe as assert_privacy_safe
from ecg_trust.audit_log._codec import canonical_json as canonical_json
from ecg_trust.contracts import (
    AuditAction,
    AuditActorType,
    AuditEvent,
    AuditOutcome,
    TrustDecision,
)


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
                    except (UnicodeDecodeError, ValueError, RecursionError) as error:
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
            with self._checkpoint_path.open("rb") as stream:
                raw = stream.read(self._config.max_event_bytes + 2)
        except OSError as error:
            raise AuditStorageError("audit checkpoint could not be read") from error
        if len(raw) > self._config.max_event_bytes + 1:
            raise LedgerCorruptionError("audit checkpoint exceeds the byte limit")
        if not raw.endswith(b"\n") or raw.count(b"\n") != 1:
            raise LedgerCorruptionError("audit checkpoint is truncated or contains extra data")
        try:
            parsed = cast(object, json.loads(raw[:-1].decode("utf-8")))
        except (UnicodeDecodeError, ValueError, RecursionError) as error:
            raise LedgerCorruptionError("audit checkpoint is not valid UTF-8 JSON") from error
        mapping = _required_mapping(parsed, context="audit checkpoint")
        try:
            canonical = canonical_json(mapping).encode("utf-8")
        except AuditValidationError as error:
            raise LedgerCorruptionError("audit checkpoint is not canonical JSON") from error
        if canonical != raw[:-1]:
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
