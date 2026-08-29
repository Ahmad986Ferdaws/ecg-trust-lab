import assert from "node:assert/strict";
import { readFile, stat } from "node:fs/promises";
import test from "node:test";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);

  return worker.fetch(
    new Request("https://results.example.test/", {
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
  assert.match(html, /id="source-support"/);
  assert.match(html, /The experiment completed\. The gate did not pass\./);
  assert.match(html, /94\.62%/);
  assert.match(html, /Source-support target missed/);
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
        new URL("../app/components/ResultsUniverse.tsx", import.meta.url),
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
  assert.equal((sceneSource.match(/\{ name: "/g) ?? []).length, 12);
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
