# Sentinel request-body limits

Sentinel accepts small JSON requests containing opaque case and release IDs.
The service now limits each HTTP request body to **4,096 bytes by default**,
before JSON parsing, case resolution, idempotency registration or analysis.
It counts bytes, including whitespace, rather than characters or JSON fields.

To set a different limit when constructing an app:

```python
from ecg_trust.service.sentinel_service import SentinelServiceConfig, create_sentinel_app

app = create_sentinel_app(
    config=SentinelServiceConfig(max_request_body_bytes=8192),
    # Supply the authorized resolver, verified release provider and engine
    # for your research integration, as usual.
)
```

The setting must be an integer from 256 to 1,048,576. It applies to every HTTP
route, including bodies on requests that will otherwise be refused. A body at
the configured limit can proceed to ordinary contract validation; one byte over
is rejected with HTTP 413 and `error.code: request_body_too_large`. The response
includes the usual research boundary and no-store/security headers, and never
echoes body content or identifiers. OpenAPI describes this shared error.

A declared `Content-Length` above the limit is refused before receiving the
body. Missing, invalid or understated declarations do not disable counting: the
middleware accumulates received chunks only up to the limit and stops when a
chunk would exceed it. Valid content is replayed to the ordinary request parser.
If the client disconnects before completing the body, no case analysis runs.
Rejected bodies do not reserve an idempotency key, so a smaller valid retry can
use the same key.

## Operational scope

This bounds the body accumulated by one application request. It does not bound
the ASGI server's receive chunk, total concurrent connections, headers or URL
size, time spent waiting for a slow sender, or total process memory. Configure
transport timeouts, connection limits and any proxy limits at their owning
layer. The separate [analysis capacity setting](SENTINEL_ANALYSIS_CAPACITY.md)
controls active backend work after request validation.

The [ASGI HTTP specification](https://asgi.readthedocs.io/en/stable/specs/www.html#request-receive-event)
defines request chunks and the `more_body` completion signal; transport
chunk-decoding belongs to the server. These controls do not add authentication,
tenant isolation or support for a public deployment. The repository's
[security scope](../SECURITY.md) and research restrictions still apply.
