# Run and retain the synthetic research tour

A prospective contributor or research partner can run the core examples with
one command, using the disposable CPU environment in
[Contributing](../CONTRIBUTING.md). Run from the repository root:

```bash
uv run --no-sync python -m scripts.run_research_tour
```

This prints a Markdown report. To retain a report under a new filename:

```bash
uv run --no-sync python -m scripts.run_research_tour > synthetic-tour.md
uv run --no-sync python -m scripts.run_research_tour --format json > synthetic-tour.json
```

The command has no patient-data, checkpoint or fitting mode. It runs the
existing synthetic quality example, the fixed-seed abstention comparison, and
one illustrative proportion-gate calculation. It uses the current source
checkout and requires its Python dependencies; it is not a wheel entry point.
No dataset download, GPU, credentials or private scientific artifact is needed.
The imported ECG Trust package must belong to this checkout; a wheel or another
editable checkout is refused before the examples run. Follow the editable CPU
installation in Contributing if that guard reports an environment mismatch.

## What the report establishes

| Part | Included context | Interpretation |
|---|---|---|
| Quality | Nine constructed cases, both full preset versions, statuses and fail-open markers | Shows behavior on these faults; preserves the five unfavorable v1 examples |
| Abstention | 2,000 constructed records, seed 0, decision thresholds 0.5, both losses and their units | Compares scores on this synthetic cohort; lower area metrics are better on the same predictions and loss scale |
| Gate planning | n=200, assumed error rate 0.02, maximum rate 0.05, one-sided confidence 0.95 and Clopper–Pearson method | Record-level binomial planning example; patient clustering is not modeled and no study size is recommended |
| Environment | Python, operating-system family, architecture and relevant installed package versions | Helps describe the observed run; does not replace a complete dependency record |
| Checkout | Git revision and whether uncommitted changes are present (including untracked files) | Informational provenance; local paths and changed filenames are omitted, and unavailable Git context is explicitly null |

The JSON schema version is 1. `status: examples_completed` and exit 0 mean that
all three examples produced a report; **they do not mean that every quality
case passed**, a release is eligible, or an independent person accepted the
software. Runtime failure produces a generic error, exit 1 and no partial
report on stdout. Invalid command arguments use exit 2.

Outputs are deterministic for the same code and numerical environment, except
for the recorded environment and checkout state. Comparing reports across
platforms should allow for numerical-library differences. The report preserves
research-only notices in both formats and contains aggregate demonstrations,
not individual model predictions or waveform samples.

For a handover exercise, retain the report with the actual command, elapsed
setup/run time, assistance needed, errors and the recipient's interpretation.
A real recipient must decide whether the agreed workflow was understood and
completed; this command does not supply that acceptance. Consult the
[evaluation toolkit](EVALUATION_TOOLKIT.md) for metric definitions and the
[known limitations](KNOWN_LIMITATIONS.md) for scientific evidence gaps.
