import assert from "node:assert/strict";
import { readFile, stat } from "node:fs/promises";
import test from "node:test";

async function render(path = "/") {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);

  return worker.fetch(
    new Request(`https://results.example.test${path}`, {
      headers: {
        accept: "text/html",
        host: "results.example.test",
        "x-forwarded-proto": "https",
      },
    }),
    {
      ASSETS: {
        fetch: async () => new Response("Not found", { status: 404 }),
      },
    },
    {
      waitUntil() {},
      passThroughOnException() {},
    },
  );
}

test("server-renders the complete Signal Ledger evidence experience", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /<title>ECG Trust Lab — The Signal Ledger<\/title>/i);
  assert.match(html, /<link rel="icon" href="data:,"\s*\/?>/i);
  assert.match(html, /og:image/i);
  assert.match(html, /twitter:image:alt/i);
  assert.match(
    html,
    /https:\/\/ecg-trust-results-motion\.byw-123\.chatgpt\.site\/og\.png/i,
  );
  assert.doesNotMatch(html, /localhost(?::\d+)?\/og\.png/i);
  assert.match(html, /A ResNet kept its ranking lead/);
  assert.match(html, /Study sequence from training to transport/);
  assert.match(html, /Discrimination and calibration results/);
  assert.match(html, /Frozen-model transport from PTB-XL to SPH/);
  assert.match(html, /no internal ECE winner is claimed/i);
  assert.match(html, /Research system/);
  assert.match(html, /Not a medical device/);
  assert.match(html, /0\.9219/);
  assert.match(html, /0\.9309/);
  assert.match(html, /15,193/);
  assert.match(html, /12 leads × 10 seconds/);
  assert.match(html, /No SPH tuning/);
  assert.match(html, /id="evidence-sources"/);
  assert.match(html, /Unadjudicated ontology bridge/);
  assert.match(html, /unknown, not verified negatives/);
  assert.match(html, /Rare transported endpoints/);
  assert.match(html, /complete operator-level outcome blindness is not claimed/);
  for (const path of [
    "reports/FINAL_RESULTS_PUBLIC.md",
    "publication/external_transport_sph_r2/FINAL_RESULTS.md",
    "docs/TRUST_SENTINEL_OOD_COMPLETION_RESULT.md",
    "reports/PROTOCOL_DEVIATIONS.md",
  ]) {
    assert.ok(html.includes(`https://github.com/Ahmad986Ferdaws/ecg-trust-lab/blob/43cf519ad1e0a664a5689a913e9123fc147294c9/${path}`));
  }
  assert.match(html, /href="\/reported-results\.csv"[^>]*download/);
  assert.match(html, /Download aggregate results \(CSV\)/);
  assert.match(html, /id="metric-panel-sph-broad"/);
  assert.match(html, /id="metric-panel-sph-no-ambiguous"/);
  assert.match(html, /Missing mappings are unknown, not verified negative diagnoses/);
  assert.match(html, /underlying ontology bridge remains unadjudicated/);
  assert.match(html, /0\.904714/);
  assert.match(html, /0\.928405/);
  assert.match(html, /15,066/);
  assert.match(html, /id="source-support"/);
  assert.match(html, /The experiment completed\. The gate did not pass\./);
  assert.match(html, /94\.62%/);
  assert.match(html, /False rejection on a 0 to 10% axis/);
  assert.match(html, /5\.38%; one-sided 95% upper bound 7\.30%; frozen maximum 5\.00%/);
  assert.match(html, /--maximum-rejection:50%/);
  assert.match(html, /--observed-rejection:53\.76344/);
  assert.match(html, /--upper-rejection:72\.96137/);
  assert.match(html, /Source-support target missed/);
  const supportSection = /<section\b[^>]*id="source-support"[\s\S]*?<\/section>/.exec(html);
  assert.ok(supportSection, "source-support evidence must be rendered");
  assert.match(
    supportSection[0],
    /<span>Frozen decision<\/span>\s*<strong>Not eligible<\/strong>/,
    "rendered eligibility must preserve the public report's unfavorable conclusion",
  );
  assert.match(html, /NOT EVALUATED/);
  assert.match(html, /No tuning · no retry/);
  assert.match(html, /id="failure-lab"/);
  assert.match(html, /Challenge the trust gates before trusting a score/);
  assert.match(html, /Illustrative synthetic scenario preview/);
  assert.match(html, /FailureLabRunner/);
  assert.match(html, /Baseline wander/);
  assert.match(html, /Mains interference/);
  assert.match(html, /Lead order swap/);
  assert.match(html, /Designed to challenge:[\s\S]{0,40}Signal-quality gate/);
  assert.match(html, /INVALID_INPUT/);
  assert.match(html, /REACQUIRE/);
  assert.match(html, /UNSUPPORTED_INPUT/);
  assert.match(html, /ABSTAIN/);
  assert.match(html, /PREDICTION_ALLOWED/);
  assert.match(html, /Not model output/);
  assert.equal((html.match(/data-lead="/g) ?? []).length, 12);
  assert.equal((html.match(/data-schematic-lead="/g) ?? []).length, 12,
    "the schematic remains visible before JavaScript or WebGL is available");
  assert.equal((html.match(/name="failure-scenario"/g) ?? []).length, 9);
  assert.match(html, /type="range"/);
  assert.doesNotMatch(
    html,
    /id="metric-panel-sph"[^>]*\shidden(?:=""|="hidden")?/i,
  );
  assert.doesNotMatch(
    html,
    /codex-preview|Your site is taking shape|Building your site|signal universe|neon/i,
  );
});

test("serves the public aggregate CSV as a named download", async () => {
  const response = await render("/reported-results.csv");
  assert.equal(response.status, 200);
  assert.equal(response.headers.get("content-type"), "text/csv; charset=utf-8");
  assert.equal(response.headers.get("content-disposition"),
    'attachment; filename="ecg-trust-reported-results-2026-08-29.csv"');
  const csv = await response.text();
  assert.equal(csv.trimEnd().split("\r\n").length, 33);
  assert.match(csv, /0\.904714,0\.001118,3,2026-08-29/);
  assert.match(csv, /0\.928405,0\.001382,3,2026-08-29/);
});

test("ships audited data and a matte, evidence-bearing visual system", async () => {
  const [
    resultsSource,
    sceneSource,
    pageSource,
    rootStyles,
    storyStyles,
    failureLabSource,
    failureLabStyles,
    sourceSupportSource,
  ] =
    await Promise.all([
      readFile(new URL("../lib/results.ts", import.meta.url), "utf8"),
      readFile(
        new URL("../app/components/ResultsWebGL.tsx", import.meta.url),
        "utf8",
      ),
      readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
      readFile(new URL("../app/globals.css", import.meta.url), "utf8"),
      readFile(
        new URL("../app/components/story/story.module.css", import.meta.url),
        "utf8",
      ),
      readFile(
        new URL("../app/components/story/FailureLabSection.tsx", import.meta.url),
        "utf8",
      ),
      readFile(
        new URL(
          "../app/components/story/FailureLabSection.module.css",
          import.meta.url,
        ),
        "utf8",
      ),
      readFile(
        new URL(
          "../app/components/story/SourceSupportSection.tsx",
          import.meta.url,
        ),
        "utf8",
      ),
    ]);

  const socialPreview = await stat(new URL("../public/og.png", import.meta.url));
  assert.ok(socialPreview.size > 100_000);

  assert.match(resultsSource, /mean: 0\.921921, sd: 0\.000913/);
  assert.match(resultsSource, /mean: 0\.930912, sd: 0\.000964/);
  assert.match(resultsSource, /records: 15_698, patients: 15_193/);
  assert.match(resultsSource, /supportCoverage: 0\.946236559139785/);
  assert.match(resultsSource, /oneSidedUpper95: 0\.07296137339055794/);
  assert.match(resultsSource, /researchBundleEligible: false/);
  assert.match(
    resultsSource,
    /Neither the PTB-XL benchmark nor the SPH transport study establishes clinical validity/,
  );

  assert.match(sceneSource, /Line2/);
  assert.match(sceneSource, /LineGeometry/);
  assert.match(sceneSource, /LineMaterial/);
  assert.match(sceneSource, /frameloop=\{shouldAnimate \? "always" : "demand"\}/);
  assert.doesNotMatch(
    sceneSource,
    /CatmullRomCurve3|TubeGeometry|AdditiveBlending|UnrealBloomPass|AfterimagePass/i,
  );

  const visualSource = `${pageSource}\n${rootStyles}\n${storyStyles}\n${failureLabStyles}`;
  assert.doesNotMatch(
    visualSource,
    /text-shadow|backdrop-filter|drop-shadow|radial-gradient|background-clip:\s*text/i,
  );
  assert.doesNotMatch(visualSource, /\bcyan\b|\bviolet\b|\bportal\b|\bparticle/i);
  assert.doesNotMatch(
    failureLabSource,
    /Math\.random|setInterval|requestAnimationFrame|modelProbability/i,
  );
  assert.match(failureLabSource, /const SAMPLE_COUNT = 960/);
  assert.match(failureLabSource, /2 \* Math\.PI \* mainsFrequencyHz \* time/);
  assert.match(failureLabSource, /2 \* Math\.PI \* (?:23|31|43) \* time/);
  assert.doesNotMatch(
    `${failureLabSource}\n${failureLabStyles}`,
    /(?:linear|radial)-gradient|backdrop-filter|box-shadow|animation\s*:/i,
  );
  assert.equal((failureLabSource.match(/id: "[a-z-]+"/g) ?? []).length, 9);
  assert.match(sourceSupportSource, /OOD-positive evaluation/);
  assert.match(sourceSupportSource, /result\.oodPositiveEvaluation/);
  assert.doesNotMatch(
    sourceSupportSource,
    /patient_id|ecg_id|embedding|filesystem|source-validation-one-shot-claim/i,
  );
});

test("keeps the WebGL renderer outside initial page preloads", async () => {
  const { default: assets } = await import("../dist/server/vinext-client-assets.js");
  const rendererFiles = assets.dynamicPreloads["app/components/ResultsWebGL.tsx"];
  const initialFiles = assets.dynamicPreloads["app/components/ResultsUniverse.tsx"];
  const renderer = rendererFiles?.find((file) => /ResultsWebGL.*\.js$/.test(file));
  assert.ok(renderer, "the renderer must be a separately loadable JavaScript chunk");
  assert.ok(initialFiles, "the schematic shell must remain in the initial client graph");
  assert.ok(!initialFiles.includes(renderer), "the shell must not preload the heavy renderer");

  const shell = initialFiles.find((file) => /ResultsUniverse.*\.js$/.test(file));
  assert.ok(shell);
  const { size } = await stat(new URL(`../dist/client/${shell}`, import.meta.url));
  assert.ok(size < 50_000, `the initial schematic shell should stay small, found ${size} bytes`);

  const response = await render();
  const html = await response.text();
  assert.ok(!html.includes(renderer), "server HTML must not eagerly request the optional renderer");
});
