# Maintaining audit records and ledger storage

The public entry point remains `ecg_trust.audit_log`. Existing imports from
`ecg_trust.audit_log.audit_ledger` also remain available.

Two internal responsibilities now have separate owners:

| Module | Owns | Does not establish |
|---|---|---|
| [`_codec.py`](../src/ecg_trust/audit_log/_codec.py) | Event adaptation, bounded metadata checks, record/checkpoint structures, canonical JSON and record hashing | A record's membership in a verified chain, storage durability, source authenticity |
| [`audit_ledger.py`](../src/ecg_trust/audit_log/audit_ledger.py) | Operational limits, file locks, bounded reads, whole-chain/checkpoint verification, append/fsync and aggregate summary construction | Authentication, authorization, a digital signature or multi-host coordination |

The codec imports no filesystem, locking or temporary-file APIs. Its record
parsers remain internal implementation details. Use `AuditLedger.read_entries`
or `verify` to check the complete ledger and checkpoint rather than treating
successful decoding as evidence that an entry belongs to a verified chain.
The package initializer still exposes the storage API; this refactor makes no
claim of a lighter package import or a separate deployable component.

## Preserve the byte and error contracts

Changing canonical serialization, schema tags, UTC formatting, enum vocabulary,
privacy rules or hash inputs can change how retained evidence is interpreted.
Review such a change as a format/behavior change with an explicit migration
plan. A code move alone must preserve the existing canonical bytes and hashes.

The storage module re-exports the moved classes, exceptions, constants and
helpers. Public callers do not need to change their imports, and existing
pickles naming the old module can resolve the same class aliases. New pickles
will name the internal codec module; pickle is not the portable ledger format
and must never be loaded from an untrusted source. Canonical JSONL and its
checkpoint remain the storage contract.

Do not bypass the append interface to write a decoded record directly. The
ledger coordinates verification, append and checkpoint work under its existing
single-host lock. This change does not alter failure recovery, fsync behavior,
checkpoint anchoring or permission requirements.

## Verification for changes to this boundary

The [ledger tests](../tests/unit/test_audit_ledger.py) cover canonical round trips,
privacy refusals, tampering, suffix deletion, invalid checkpoints and concurrent
append. Monitoring and Sentinel tests exercise downstream consumers. For a
mechanical extraction, compare the moved definitions and the storage methods,
then read a baseline ledger under the new code and compare exact entry,
checkpoint and aggregate-summary bytes. Test legacy imports and record pickles
using locally generated trusted fixtures.

The project-source closure discovers Python modules recursively, so the codec
is included in new source-tree identities. Do not edit a frozen protocol's
explicit bindings or regenerate sealed evidence merely because code moved.
Existing research limits and the [contribution boundary](../CONTRIBUTING.md#respect-the-scientific-boundary)
continue to apply.
