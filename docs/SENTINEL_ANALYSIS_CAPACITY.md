# Bounding active Sentinel analyses

An application can set a per-process limit on active backend analyses when it
constructs the service:

```python
app = create_sentinel_app(
    case_resolver=case_resolver,
    release_provider=release_provider,
    analysis_engine=analysis_engine,
    config=SentinelServiceConfig(max_concurrent_analyses=2),
)
```

The limit accepts integers from 1 through 64. The default is `None`, preserving
the existing uncapped behavior. Case validation and inference share the limit,
which covers resolving a case, loading its verified release, and running the
backend. Health and readiness requests do not consume these slots.

The liveness endpoint (`/api/v1/healthz`) only builds a static boundary response
and runs on the event loop, so it does not wait for a free analysis worker.
[Starlette runs synchronous endpoints in a shared worker pool](https://starlette.dev/threadpool/);
an occupied pool must not make this lightweight liveness probe wait behind
analysis. Liveness does not check backend health or available analysis capacity.
Readiness (`/api/v1/readyz`) performs dependency checks and can still wait for a
worker; event-loop stalls and server connection limits can also delay probes.

When every slot is occupied, a new analysis returns HTTP 503 immediately with
`decision: ABSTAIN`, `reason_codes: [SERVICE_BUSY]`, and `Retry-After: 1`. The
response has the usual research boundary and no-store headers, with no model
probabilities. Busy results are not cached: the same idempotency key can be
retried after capacity becomes available. A completed cached replay does not
consume a slot. Concurrent duplicates still share their original operation.

Slots are released on success and on failure. This is an active-analysis limit
inside one application process; it does not limit the ASGI server's connection
queue, duplicate waiters, or the combined capacity of multiple worker processes.
Choose the limit for the actual model's memory and runtime requirements. No
clinical behavior, trust thresholds, or scientific artifact formats change.
