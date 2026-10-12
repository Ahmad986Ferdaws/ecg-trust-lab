"""Process-local replay and concurrent request coordination for Sentinel.

The HTTP adapter owns request validation and response serialization. This module
owns in-flight coordination, fingerprint conflicts, cacheability and LRU eviction.
It does not provide cross-process or durable idempotency.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from threading import Event, RLock


class _IdempotencyConflictError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class _ServiceResult:
    content: dict[str, object]
    status_code: int
    cacheable: bool = False
    retry_after_seconds: int | None = None


@dataclass(frozen=True, slots=True)
class _StoredInference:
    fingerprint: str
    result: _ServiceResult


@dataclass(slots=True)
class _PendingInference:
    fingerprint: str
    completed: Event
    result: _ServiceResult | None = None


class _InMemoryIdempotencyStore:
    """Process-local, bounded replay store containing only already-safe JSON."""

    def __init__(self, max_entries: int) -> None:
        self._max_entries = max_entries
        self._entries: OrderedDict[str, _StoredInference] = OrderedDict()
        self._pending: dict[str, _PendingInference] = {}
        self._lock = RLock()

    def execute(
        self,
        *,
        key: str,
        fingerprint: str,
        operation: Callable[[], _ServiceResult],
    ) -> tuple[_ServiceResult, bool]:
        while True:
            with self._lock:
                existing = self._entries.get(key)
                if existing is not None:
                    if existing.fingerprint != fingerprint:
                        raise _IdempotencyConflictError
                    self._entries.move_to_end(key)
                    return existing.result, True

                pending = self._pending.get(key)
                if pending is None:
                    pending = _PendingInference(fingerprint, Event())
                    self._pending[key] = pending
                    break
                if pending.fingerprint != fingerprint:
                    raise _IdempotencyConflictError

            # Waiting duplicates retain their flight even if another key evicts
            # its eventual cached entry before this thread resumes.
            pending.completed.wait()
            if pending.result is not None and pending.result.cacheable:
                return pending.result, True

        try:
            result = operation()
            with self._lock:
                pending.result = result
                if result.cacheable:
                    self._entries[key] = _StoredInference(fingerprint=fingerprint, result=result)
                    while len(self._entries) > self._max_entries:
                        self._entries.popitem(last=False)
            return result, False
        finally:
            with self._lock:
                del self._pending[key]
                pending.completed.set()
