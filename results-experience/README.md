# ECG Trust Lab — The Signal Ledger

An editorial, interactive account of the audited PTB-XL classifier comparison
and frozen SPH external-transport stress test. The design behaves like a
scientific feature: matte paper, ruled data tables, direct labels, and motion
that explains the evidence instead of decorating it.

## Run locally

Prerequisites: Node.js 22.13 or newer and pnpm 11.

```bash
pnpm install --frozen-lockfile
pnpm dev
```

Open `http://localhost:3000`.

## Quality gates

```bash
pnpm lint
pnpm typecheck
pnpm test
```

`pnpm test` first checks public evidence parity, then performs a production build and verifies the server-rendered
evidence, research boundary, frozen metric source, 12-lead scene contract, and
the absence of the previous neon/glass visual patterns.

Run `pnpm test:evidence` for the fast, dependency-free data check. From the
repository root, the equivalent command is:

```bash
node --experimental-strip-types --test results-experience/tests/evidence-parity.test.mjs
```

It compares all 96 displayed macro mean/SD values and 43 SPH cohort/positive
counts with the sealed public CSV/JSON artifacts, checks PTB-XL ECG/patient
totals against the model card and member table, and verifies both public
checksum inventories. It also binds source-support role counts, published
outcomes, missed-target status, research ineligibility, and unevaluated OOD
claims to the [public completion report](../docs/TRUST_SENTINEL_OOD_COMPLETION_RESULT.md).
Percentages are compared at the report’s precision; this does not recompute
private scientific results. It needs Node.js 22.13+ but no package installation,
site build, private artifacts, or patient data. The explicit type-stripping
flag supports the minimum Node version. Read-only push/PR CI runs this check
on Node 22.13 and 24. A mismatch should be investigated against the sealed
evidence; do not update scientific results or their inventories to make a
presentation test pass. This check does not certify the prose or clinical claims. The full rendered-HTML
suite also asserts that the visible frozen decision remains “Not eligible”;
the component derives that label from the checked eligibility field.

## Architecture

- `app/components/ResultsUniverse.tsx` — a one-pass 12-lead acquisition built
  with Three.js `Line2`, `LineGeometry`, and `LineMaterial`
- `app/components/ExperienceMotion.tsx` — shared play/pause and reduced-motion
  state
- `app/components/story/` — ruled metric ledger, transport sequence, and
  research boundary
- `lib/results.ts` — typed audited result values and caveats
- `.openai/hosting.json` — OpenAI Sites hosting binding

The WebGL scene draws only while its short acquisition is running. It switches
to demand rendering when complete, paused, offscreen, or hidden. A non-WebGL
fallback remains available, reduced-motion users receive the completed static
state, and every material result is present in semantic server-rendered HTML.

## Scientific boundary

This is a retrospective research visualization, not a medical device and not
clinical validation. It must not be used for diagnosis, treatment, or emergency
decisions. SPH was evaluated with frozen models and no target-domain training,
selection, preprocessing adaptation, recalibration, thresholds, or confidence
gate tuning. The hero waveform is explicitly schematic; the reported numbers
come from the audited evaluation artifacts.

## Continuous integration

The [results-experience workflow](../.github/workflows/results-experience-quality.yml)
installs the existing lockfile with pnpm 11.19.0 on Node 24, then runs lint,
TypeScript, data parity, a production build, and rendered-HTML checks. It runs on
changes to the UI, public evidence, documentation, or its own workflow, with
read-only repository permissions and a 15-minute limit.

`pnpm-workspace.yaml` explicitly disables install scripts for the three locked
packages that ship prebuilt native binaries and makes stale dependency state an
error before scripts run. Run `pnpm install --frozen-lockfile` after dependency
configuration changes. This follows pnpm's
[explicit build-script policy](https://github.com/pnpm/pnpm.io/blob/main/blog/releases/11.0.md).
The CI check also confirms that dependency declarations were not rewritten.
