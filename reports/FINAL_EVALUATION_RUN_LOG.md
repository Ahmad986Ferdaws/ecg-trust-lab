# Final evaluation operational run log

This document records how the preregistered PTB-XL fold-10 batch was executed.
It is an operational record created after completion; it is not part of the
pre-evaluation specification and does not replace the bound protocol-deviation
log.

## Frozen identities

- Evaluation-spec artifact: `sha256:1f73c021a544ffeb119ffe8e490a16e32ec84247e30bce1ffd895fcffed6c762`
- Evaluation code revision: `b0334730d8ab287a364f5978003dbe961770867c`
- Refit bundle: `sha256:7a0bbd07bfeec1cfb68599829921fee0ad4a3d97968520c43c952f1ea3bb59dd`
- Calibration bundle: `sha256:12b63f0ca20c0c8a901166ff3fa3bc8ed707cecb467413326e057d69c588b0ec`
- Final batch: `sha256:a4da85d5272b1634baaf953496c3d9efd8917777ad8de13b1b0c6dc754699e62`
- Completed ledger: `sha256:3bb83554a08832212989ea8f3ea212f6af42c08460edbde9ba130065b1115a57`
- Final batch summary: `sha256:f075d45f03ae43127c22546c95b1c9f4dd2357dbe6b2c4c9d948d159c8ce1cbf`
- Required scientific disclosure: `DEV-001` in `reports/PROTOCOL_DEVIATIONS.md`

## Opening and command

The global spec-keyed ledger was created before fold-10 access and the
canonical opening marker was committed on 2026-08-09 at approximately 00:52
UTC. The operator was `devha`; the recorded purpose was
`Preregistered one-time PTB-XL fold-10 final evaluation`.

The batch was opened once with `scripts/release_pipeline.py run-final` using
the exact refit bundle, calibration bundle, evaluation specification, purpose,
operator, and required confirmation phrase. Every continuation used the same
command inputs with only the required `--resume` flag added. No model,
checkpoint, calibration transform, threshold, abstention gate, subgroup,
metric, seed, or report setting changed after the opening.

## Interruption and recovery

Immediately after each newly exported prediction pair was atomically saved, a
validator compared two equivalent SHA-256 representations directly:
`sha256:<hex>` from the prediction writer versus bare `<hex>` from the release
file checker. The pair's sidecar, size, artifact hash, raw-file hash, lineage,
fold, checkpoint, configuration, manifest, and inference settings all verified;
the failure occurred only in the redundant in-memory post-save comparison.

The ledger recorded six `batch_interrupted` events with the error
`final exporter NPZ hash differs from saved file`. The designed crash-recovery
path was used without deleting, renaming, or overwriting any prediction. On
each exact resume, the complete pair was loaded and fully validated, then
adopted into the ledger before its report was generated. The next member was
then exported and encountered the same representation-only check. This
continued deterministically until all six members and aggregate reports were
complete.

One supervised shell invocation also reached its external two-minute command
limit during report computation. It terminated before a new ledger event or
artifact commit; the ledger and existing prediction pair were unchanged. The
next exact resume repeated only unfinished deterministic computation.

## Completion evidence

The terminal ledger event is `exact_six_member_final_batch_complete` at
2026-08-09 01:12:37 UTC. The completed ledger binds:

- six immutable fold-10 prediction NPZ/JSON pairs;
- six immutable member reports;
- two architecture summaries;
- three within-seed paired patient-bootstrap reports;
- the paired-bootstrap manifest; and
- the final batch summary.

After completion, an additional exact `--resume` was executed from the frozen
`b033473` tree. It returned the same batch identity after read-only validation,
without running inference, creating files, or mutating the complete ledger.

## Post-evaluation correction

Only after completion and read-only replay, commit `c75a12b` normalized the
two SHA-256 representations in the redundant exporter-result validator and
added a regression test. This later revision did not produce or alter any
sealed evaluation artifact. All confirmatory results remain attributable to
the code revision frozen in the evaluation specification.

## Interpretation boundary

The interruptions were operational and did not cause retuning or repeated
scientific test queries: each model produced one immutable prediction artifact,
and every report was computed from that artifact under its frozen fold-9
decisions. Nevertheless, the full history is disclosed here so readers can
distinguish a single scientific opening with ledgered recovery from an
uninterrupted process execution.
