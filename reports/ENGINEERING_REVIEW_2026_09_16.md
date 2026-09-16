# Engineering review: Sentinel boundaries and CPU regression checks

Reviewed baseline: `7127977` on `main`. This is a source and synthetic-fixture
review, not a new scientific evaluation. No patient dataset, checkpoint,
fold-10 outcome, or private run artifact was loaded.

## Findings and changes

### 1. Malformed policy evidence could permit predictions

`TrustPolicyInputs` checked the number of conformal decisions but not their
runtime types. `evaluate_trust_policy` compares decisions by enum identity,
so five plain `"uncertain"` strings were not recognized as uncertainty. With
the other gates passing, that input reached `PREDICTION_ALLOWED`. Non-boolean
gate values such as `"false"` were also interpreted by Python truthiness,
and untyped quality-status strings could miss the blocking branches.

The policy now rejects non-boolean gates and configuration flags, invalid
quality report/status types, and non-`BinaryDecision` entries at construction.
Missing optional evidence still follows the existing fail-closed routes.
This hardens the internal Python integration contract; it is not evidence of
an externally exploitable HTTP input path. Correctly typed policy behavior,
thresholds, and gate ordering are unchanged.

### 2. Request execution did not enforce engine/release readiness

The readiness endpoint checked `is_ready_for_release`, but the validation and
inference routes called the injected engine without that check. A provider
could report a verified release while an injected engine reported that its
loaded release did not match, and the service could still return predictions.

Both routes now require a literal `True` readiness result for the requested
release before calling the engine. False, truthy non-boolean results, and
exceptions produce a sanitized HTTP 503 abstention with no label outputs.
Those failures remain retryable with the same idempotency key. Successful
responses retain their existing replay semantics.

The production `SentinelServiceAnalysisEngine` already checked release
identity internally. The change brings the generic service boundary into
alignment with that adapter rather than asserting the adapter was unsafe.

### 3. Pull requests had no automated Python quality gate

Issue #2 identified the missing workflow. The new `CPU quality` workflow
installs Python 3.12 development dependencies with CPU PyTorch and runs the
complete test suite, Ruff, strict mypy, and dependency consistency checks.
Actions are pinned to immutable revisions, the token is read-only, and
checkout credentials are not persisted. No data download, GPU smoke test,
artifact upload, or scientific execution is configured.

The existing CUDA-only source cannot install on macOS. CPU checks explicitly
use `--no-sources --torch-backend cpu` and `uv run --no-sync`; they do not
rewrite `pyproject.toml` or `uv.lock`. This is a separate engineering
environment resolving the declared dependency ranges, not a locked scientific
reproduction. See [the commands and limitations](../docs/REPRODUCIBILITY.md#cpu-pull-request-checks).

Running the full suite exposed existing fixture assumptions that needed
correction for CPU checks:

- The synthetic CUDA UUID test did not supply a synthetic CUDA runtime version,
  so CPU PyTorch failed an earlier guard instead of exercising the UUID guard.
  The synthetic acceptance test also now supplies its own device visibility
  instead of inheriting the CI host's empty `CUDA_VISIBLE_DEVICES` setting.
- The module-origin fixtures replaced global `sys.modules`, preventing pytest
  from reporting failures safely. They now replace only the audited module's
  view of `sys`, and model the frozen Windows `ntpath` alias explicitly.
- A checkout-local pytest temporary directory avoids macOS's indirect system
  temporary paths, which the filesystem-integrity tests deliberately reject.
- Sweep-runner fixtures now provide their own synthetic bytes for provenance
  hashing instead of reading the private PTB-XL manifest and normalization.
- Tests that actually require the frozen Windows runtime now declare that
  platform prerequisite. Tests that require private frozen inventory,
  normalization, or decision artifacts skip explicitly when those inputs are
  absent, consistent with the existing source-calibration test. These are
  visible coverage limits, not successful executions of those private checks.
- Synthetic subprocess fixtures resolve the Python executable and use direct
  temporary paths; Git command expectations account for the host null device.

The frozen Windows OOD implementation remains unchanged. Mypy runs in strict
mode with an explicit Windows target so that it checks Windows-only `ctypes`
APIs even on the Linux CI host; Windows runtime skips remain visible.

## Broader review observations

- Inspected representative paths through manifest validation, dataset fold
  guards, prediction alignment, calibration fitting, OOD patient partitioning,
  service release binding, and audit-ledger persistence. The code contains
  explicit patient-fold checks, fold-9 calibration restrictions, hash-bound
  prediction provenance, and fail-closed artifact validation. This is not a
  line-by-line certification of all source files.
- Compared all 96 macro mean/SD values in `results-experience/lib/results.ts`
  and `results-experience/app/components/story/storyData.ts` with
  `publication/results/tables/architecture_metrics.csv` and
  `publication/external_transport_sph_r2/FINAL_RESULTS.md`. All match at
  six-decimal presentation precision: 64 values across the library's four
  cohorts and 32 in the story's two cohorts. No copy change was needed for
  those metrics. This comparison did not recompute any research metric.
- Read the open issues before choosing scope. The parked studies and the
  execute-or-terminate OOD decision remain separate from engineering fixes.
  The sealed publication tables, scientific configurations, and lockfile
  are unchanged.

## Validation record

The first regression run against the unchanged implementation had **30 new
failures and 47 existing passes**, reproducing both boundary gaps. After the
fixes and additional input-contract cases, the focused policy/service suite
passed **86 tests**. The three module-origin fixtures also passed in isolation.

Local environment: macOS arm64, Python 3.12.14, CPU PyTorch 2.13.0,
pytest 9.1.1, Ruff 0.16.7, mypy 2.3.1, uv 0.12.15. Strict Windows-target
mypy passed all 111 source files; Ruff and `uv pip check` passed.

The final local full-suite run passed **1,672 tests**, with **110 explicit
prerequisite skips** and two upstream Starlette/AnyIO deprecation warnings in
63.65 seconds. The 110 skips include 49 already present in the initial run;
61 additional unavailable-platform/private-input cases now report their
prerequisites instead of failing on missing resources. No test is marked xfail.
The hosted CI result is recorded in the pull request.
CPU checks do not verify CUDA numerics, Windows handle/process behavior,
private artifact availability, or scientific result reproducibility.
