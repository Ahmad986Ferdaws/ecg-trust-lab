from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock

import pytest

from ecg_trust.service.sentinel_service import (
    _IdempotencyConflictError,
    _InMemoryIdempotencyStore,
    _ServiceResult,
)


def test_distinct_keys_execute_while_another_request_is_running() -> None:
    store = _InMemoryIdempotencyStore(2)
    started, release, independent = Event(), Event(), Event()

    def blocked() -> _ServiceResult:
        started.set()
        assert release.wait(5)
        return _ServiceResult({"first": True}, 200, True)

    def other() -> _ServiceResult:
        independent.set()
        return _ServiceResult({"second": True}, 200, True)

    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(store.execute, key="a", fingerprint="a", operation=blocked)
        try:
            assert started.wait(5)
            second = pool.submit(store.execute, key="b", fingerprint="b", operation=other)
            assert independent.wait(1), "unrelated keys must not wait for the first operation"
        finally:
            release.set()
        assert first.result()[1] is False
        assert second.result()[1] is False


def test_concurrent_duplicates_execute_once_and_replay() -> None:
    store = _InMemoryIdempotencyStore(2)
    started, release = Event(), Event()
    calls = 0
    count_lock = Lock()

    def operation() -> _ServiceResult:
        nonlocal calls
        with count_lock:
            calls += 1
        started.set()
        assert release.wait(5)
        return _ServiceResult({"result": "stable"}, 200, True)

    with ThreadPoolExecutor(8) as pool:
        futures = [
            pool.submit(store.execute, key="same", fingerprint="same", operation=operation)
            for _ in range(8)
        ]
        try:
            assert started.wait(5)
        finally:
            release.set()
        results = [future.result() for future in futures]
    assert calls == 1
    assert sum(replayed for _, replayed in results) == 7
    assert all(result.content == {"result": "stable"} for result, _ in results)


def test_conflicting_key_fails_while_original_is_running() -> None:
    store = _InMemoryIdempotencyStore(2)
    started, release = Event(), Event()

    def operation() -> _ServiceResult:
        started.set()
        assert release.wait(5)
        return _ServiceResult({}, 200, True)

    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(store.execute, key="same", fingerprint="original", operation=operation)
        try:
            assert started.wait(5)
            conflict = pool.submit(
                store.execute, key="same", fingerprint="conflict", operation=operation
            )
            with pytest.raises(_IdempotencyConflictError):
                conflict.result(timeout=1)
        finally:
            release.set()
        first.result()


@pytest.mark.parametrize("raises", [False, True])
def test_retry_after_uncacheable_result_or_exception(raises: bool) -> None:
    store = _InMemoryIdempotencyStore(2)
    started, release = Event(), Event()
    calls = 0

    def operation() -> _ServiceResult:
        nonlocal calls
        calls += 1
        if calls == 1:
            started.set()
            assert release.wait(5)
            if raises:
                raise RuntimeError("temporary failure")
            return _ServiceResult({}, 503)
        return _ServiceResult({"recovered": True}, 200, True)

    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(store.execute, key="same", fingerprint="same", operation=operation)
        try:
            assert started.wait(5)
            second = pool.submit(store.execute, key="same", fingerprint="same", operation=operation)
        finally:
            release.set()
        if raises:
            with pytest.raises(RuntimeError, match="temporary failure"):
                first.result()
        else:
            assert first.result()[0].status_code == 503
        result, replayed = second.result(timeout=2)
    assert calls == 2
    assert result.content == {"recovered": True}
    assert replayed is False
    assert store.execute(key="same", fingerprint="same", operation=operation)[1] is True
