# Replaying aggregate monitoring with frozen configuration provenance

The existing v1 telemetry and comparison formats remain unchanged. For an audited
replay, seal the aggregate reference together with **every** monitoring configuration
field. A version label alone does not identify threshold values.

```bash
python -m scripts.replay_trust_monitoring freeze \
  --reference reference-window.json --output frozen-reference.json
```

The default configuration is used unless `--config full-config.json` is supplied.
The configuration file must contain every field returned by
`monitoring_config_to_dict(TrustMonitoringConfig())`; missing values are rejected.
Keep the generated `envelope_sha256` in the review record independently of the file.
The envelope is canonical JSON; preserve its exact bytes.

```bash
python -m scripts.replay_trust_monitoring replay \
  --frozen-reference frozen-reference.json --expected-sha256 RETAINED_DIGEST \
  --windows ordered-windows.json --output monitoring-replay.json
```

`ordered-windows.json` is a JSON array of existing v1 aggregate telemetry windows.
Replay uses the sealed thresholds. An optional `--config` checks the intended runtime
configuration against the sealed fingerprint and fails on any drift, including a
threshold change under the same version label. Outputs must be new files.

The replay envelope records the frozen reference, configuration, and each input
window digest alongside the unmodified v1 comparisons. Recommendations remain
advisory; replay does not retrain, change thresholds, or execute governance actions.
Only aggregate counters and histograms are accepted. These hashes detect changes
relative to a retained identity; they do not authenticate who created the reference.

Python callers can use `FrozenMonitoringReference(reference, config)`, then
`compare`, `replay`, or `replay_to_json`. Reload with `from_json(...,
expected_sha256=retained_digest)` to verify a separately retained reference identity.
