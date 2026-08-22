# ECG Trust Lab — Results in Motion

An immersive WebGL results experience for the audited PTB-XL classifier and
frozen SPH external-transport study. It turns the project’s scientific evidence
into a scroll-driven visual narrative without changing or simplifying the
underlying result values.

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

`pnpm test` performs a production build and verifies the server-rendered story,
headline evidence, research boundary, social card, and frozen metric source.

## Architecture

- `app/components/ResultsUniverse.tsx` — live React Three Fiber scene
- `app/components/story/` — interactive metrics, cohort orbit, and safety ending
- `lib/results.ts` — typed audited result values and caveats
- `public/og.png` — generated social preview image
- `.openai/hosting.json` — OpenAI Sites hosting binding

The WebGL scene has a non-WebGL fallback, animation honors reduced-motion
preferences, and every material result remains available as semantic HTML.

## Scientific boundary

This is a retrospective research visualization, not a medical device and not
clinical validation. It must not be used for diagnosis, treatment, or emergency
decisions. SPH was evaluated with frozen models and no target-domain training,
selection, preprocessing adaptation, recalibration, thresholds, or confidence
gate tuning.
