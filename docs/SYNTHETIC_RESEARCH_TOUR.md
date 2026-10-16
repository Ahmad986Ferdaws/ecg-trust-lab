# Your first research reports

Use this walkthrough to evaluate the toolkit's workflow before arranging access
to a dataset or trained model. It runs three existing commands on synthetic
inputs or explicit planning assumptions. It performs no training, fitting on
real data, model inference, or network requests once the environment is installed.

These commands require a **source checkout**, Python 3.12, and the disposable
CPU environment in [CONTRIBUTING.md](../CONTRIBUTING.md). Run them from the
repository root. The `scripts/` directory is not part of the installed wheel.
Keep `--no-sync` so uv does not replace the CPU environment with the scientific
CUDA environment. `ecg-verify` is a separate CUDA training-stack check.

## 1. Inspect acquisition failures

```bash
uv run --no-sync python scripts/quality_gate_stress_report.py
uv run --no-sync python scripts/quality_gate_stress_report.py --format json
```

The report starts from one deterministic 12-lead waveform and applies eight
acquisition faults. Read `should_refuse` as the synthetic scenario's expected
behavior, `classification_allowed` as the gate's actual decision, and `fail_open`
as a mismatch between them. The unaltered `clean` case is a passing control.

The current bank exposes five v1 fail-open cases and none for v2. Both presets
allow the clean control. This demonstrates behavior on nine constructed cases;
it does not estimate sensitivity, specificity, or field reliability. Preset v1
remains frozen, and this demonstration does not promote v2 to a release.

## 2. Compare abstention scores

```bash
uv run --no-sync python scripts/compare_abstention_scores.py --synthetic-demo
```

The JSON identifies its source as `synthetic demo (not model output)` and reports
2,000 simulated records. It compares five uncertainty scores on the same
predictions. `mean_binary_log_loss` is primary; `thresholded_hamming_loss` is
secondary. Lower AURC, excess AURC, and AUGRC indicate better loss ordering under
their respective definitions; they are not diagnostic accuracy percentages.

Check both losses: their score rankings need not agree. The synthetic ranking
does not establish which score works best on PTB-XL or another population.
See the [evaluation toolkit](EVALUATION_TOOLKIT.md) for the metrics' definitions
and limitations. Do not replace `--synthetic-demo` with held-out predictions
without a separate protocol; the real-input command has fold restrictions and
cannot identify every already-spent cohort within a fold.

## 3. Explore a proposed study gate

```bash
uv run --no-sync python scripts/plan_gate.py --sample-size 200 --maximum-rate 0.05 --true-rate 0.01 --search-limit 1000
```

This hypothetical binomial plan asks about a one-sided 95% upper-bound gate at
5%, assuming independent trials with a true event rate of 1%. It allows at most
four events among 200 trials and gives a pass probability of about 0.9483.
The search reports 124 as the first sample size reaching its default 0.8 target.

Those are mathematical planning outputs under the supplied assumptions, not
observed clinical results or an approved sample-size recommendation. Repeated
ECGs within patients and shared fitted thresholds can violate the independent
binomial model. Read the [source-support planning note](TRUST_SENTINEL_OOD_COMPLETION_GATE_POWER_NOTE.md)
before applying this tool. No change to an existing frozen gate is implied.

## Keep a reviewable trial record

Save outputs in a fresh directory under the Git-ignored `outputs/` directory if
needed. Record the commit (`git rev-parse HEAD`), Python version, installed
package versions, command, exit status, and any warnings alongside each report.
Record setup time, unclear steps, and assistance needed; these are observations
about adoption effort, not evidence of clinical usefulness or commercial demand.

Successful completion means you can explain the refused synthetic faults,
compare score rankings, and distinguish a planning assumption from a measured
result. A follow-up evaluation with real data needs an agreed data contract,
scope and acceptance criteria. The [known limitations](KNOWN_LIMITATIONS.md)
and [model card](MODEL_CARD_PTBXL_SUPERCLASS_R3.md) remain part of that review.

If an import fails, confirm the CPU environment installation and current working
directory first. Missing private data or weights should not affect this tour.
Report a failure with its commit and sanitized command output through the
[contribution workflow](../CONTRIBUTING.md).
