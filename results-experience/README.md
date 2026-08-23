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

`pnpm test` performs a production build and verifies the server-rendered
evidence, research boundary, frozen metric source, 12-lead scene contract, and
the absence of the previous neon/glass visual patterns.

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
