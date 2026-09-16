# Contributing to ECG Trust Lab

Start with a concrete problem or an existing issue. Useful contributions improve
correctness, verification, usability, or maintainability. Split independently
reviewable changes by responsibility; do not create duplicate issues or split
one fix just to increase contribution counts.

## Record the evidence

Use a [bug report](.github/ISSUE_TEMPLATE/bug_report.yml) for an observed defect
or an [improvement proposal](.github/ISSUE_TEMPLATE/improvement.yml) for a gap.
Include the affected revision, expected behavior, observed behavior, and clear
acceptance criteria. Separate observations from hypotheses. Prefer a small
synthetic reproduction over private data or a costly scientific rerun.

For uncertain behavior, state the hypothesis, run the cheapest test that could
refute it, and preserve the outcome. A passing control is useful evidence; do
not describe it as a reproduced failure.

## Validate the change

The [CPU quality workflow](.github/workflows/cpu-quality.yml) is the authoritative
engineering setup. In a disposable checkout, with Python 3.12 and uv 0.12.15:

```bash
uv venv --python 3.12
uv pip install --no-sources --torch-backend cpu -e . --group dev
uv pip check
```

Run the relevant tests first. Before merging code changes, run the quality gates:

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg uv run --no-sync pytest -q --basetemp=.pytest-tmp-cpu
uv run --no-sync ruff check src tests scripts
uv run --no-sync mypy --platform win32
git diff --exit-code -- pyproject.toml uv.lock
```

The Windows mypy target reflects the scientific runtime contract; it does not
mean the CPU tests ran on Windows. Report skipped tests and environment limits.
The CPU setup bypasses the scientific CUDA package source without rewriting
the tracked dependency files. Do not use a later syncing command to replace it.

For public result presentation, Node.js 22.13+ can run the dependency-free check:

```bash
node --experimental-strip-types --test results-experience/tests/evidence-parity.test.mjs
```

For UI changes, also follow the lint, typecheck, build, and rendered checks in
[the results experience guide](results-experience/README.md). For documentation
changes, verify links and syntax rather than adding tests that repeat the prose.

## Respect the scientific boundary

Read the [project overview](README.md) and the protocol relevant to your change.
Engineering tests do not authorize new scientific runs or establish clinical
validity. Preserve frozen splits, thresholds, model states, hashes, and sealed
results. Do not regenerate evidence to make a test pass, tune against an opened
evaluation cohort, or turn an unfavorable result into a favorable claim.

The [source-support completion report](docs/TRUST_SENTINEL_OOD_COMPLETION_RESULT.md)
records a missed target and research ineligibility. The
[frozen SPH protocol](docs/EXTERNAL_TRANSPORT_SPH_R2.md) defines its transport
boundary. New scientific work needs its own explicit scope and protocol.
Keep patient identifiers, waveforms, row-level embeddings, credentials, and
private artifacts out of issues, logs, commits, and reviews.

## Submit and review

Use the [PR template](.github/pull_request_template.md). Link the issue and use
`Closes #N` only when the acceptance criteria are met. Explain what changed,
why, how it was checked, and the limits of those checks. Include a failing-before
and passing-after reproduction for a bug when practical.

A documented technical review should name the reviewed commit, trace the
important data/control flow, examine failure paths and compatibility, cite
validation evidence, and identify actionable findings or remaining limits.
Mark agent-assisted reviews as such. An author's COMMENT review is a record of
technical inspection, not an independent approval.

Resolve findings with evidence, run the relevant checks on the final changes,
and merge only after CI passes and conflicts are resolved. Close an issue when
its actual work is complete; leave blocked scientific decisions open with an
accurate explanation of what is still needed.
