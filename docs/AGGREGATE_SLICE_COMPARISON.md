# Compare aggregate subgroup metrics

Use `scripts/compare_slices.py` to apply the existing subgroup regression gate to
two sets of aggregate results. It needs a source checkout and the Python
environment in [CONTRIBUTING.md](../CONTRIBUTING.md), but no waveforms, model
weights or fitting. The command is suitable for reviewing already authorized
research results under limits chosen before inspecting the comparison.

## Input and example

Save this **synthetic example** as `groups.json` in a disposable working area:

```json
{
  "schema_version": 1,
  "reference": [
    {"group": "group_a", "count": 900, "value": 0.9},
    {"group": "group_b", "count": 100, "value": 0.8}
  ],
  "candidate": [
    {"group": "group_a", "count": 900, "value": 0.9},
    {"group": "group_b", "count": 100, "value": 0.7}
  ]
}
```

Run from the repository root, substituting the example file's path:

```bash
uv run --no-sync python scripts/compare_slices.py groups.json --metric retention --direction higher --max-overall-drop 0.02 --max-slice-drop 0.05 --min-count 30
```

The overall drop is about 0.01 and passes the illustrative overall limit, while
`group_b` drops by 0.10 and fails its slice limit. The report therefore says
`passed: false` and the process exits **1**. This is a completed comparison with
a failed gate, not a program error. The example limits are not recommendations
or changes to a frozen protocol.

Both sides must contain the same unique, non-empty group names. Each row has
exactly `group`, `count` and `value`; counts are non-negative integers and values
are finite real numbers. The root has exactly the three fields shown above.
Inputs must be regular UTF-8 JSON files no larger than 1 MiB. Duplicate JSON keys,
unknown fields, booleans in numeric fields and nonfinite values are refused.
Use aggregate group labels, not patient identifiers or row-level records.

## Choose controls deliberately

`--metric`, `--direction`, both maximum drops and `--min-count` are required.
Use `higher` for a quantity such as retention and `lower` for one such as loss.
Drops are absolute differences in the supplied metric's units: a retention drop
of 0.05 means five percentage points when values are fractions.

The default `--insufficient-policy block` prevents a passing result when either
side of a group has fewer than the minimum records. Explicitly selecting
`report` lists those groups as insufficient while excluding them from the
overall calculation. If no group is sufficient, the gate still fails.

The overall summary uses reference counts to weight **only sufficient groups**
on both sides. It holds group composition fixed; it is not the candidate's raw
population average. Use mutually exclusive groups from a coherent population
when interpreting this weighted summary. Overlapping slices double-count
membership. Non-additive metrics such as AUROC cannot be reconstructed as a
pooled metric by averaging subgroup values; interpret that output only as the
stated weighted summary.

## Output and automation

Output is JSON on stdout. It contains the metric, all controls, group counts,
individual and overall verdicts, and SHA-256 of the exact input bytes. Retain
the input and the repository commit with the report; whitespace changes alter
that digest. It binds this report to supplied input bytes, not to an independently
verified data source or an authenticated author.

| Exit status | Meaning |
|---|---|
| 0 | Comparison completed and passed the selected gate |
| 1 | Comparison completed but failed or lacked sufficient evidence |
| 2 | Invalid arguments, unreadable/malformed input, invalid controls or nonfinite computed output; no report produced |

The gate compares point estimates. It supplies no confidence interval,
significance test, multiplicity correction or clinical validation. A passing
result does not promote a model release. Define any needed uncertainty analysis
and acceptance rule in the relevant protocol; do not retune thresholds using
sealed evaluation results or spent one-shot cohorts. See the
[evaluation toolkit](EVALUATION_TOOLKIT.md) and [known limitations](KNOWN_LIMITATIONS.md).
