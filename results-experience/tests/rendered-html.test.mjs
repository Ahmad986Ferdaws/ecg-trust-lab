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
  assert.doesNotMatch(html, /og:image|twitter:image|localhost(?::\d+)?\/og\.png/i);
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
  const [resultsSource, sceneSource, pageSource, rootStyles, storyStyles] =
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
    ]);

  await assert.rejects(
    stat(new URL("../public/og.png", import.meta.url)),
    (error) => error?.code === "ENOENT",
  );

  assert.match(resultsSource, /mean: 0\.921921, sd: 0\.000913/);
  assert.match(resultsSource, /mean: 0\.930912, sd: 0\.000964/);
  assert.match(resultsSource, /records: 15_698, patients: 15_193/);
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

  const visualSource = `${pageSource}\n${rootStyles}\n${storyStyles}`;
  assert.doesNotMatch(
    visualSource,
    /text-shadow|backdrop-filter|drop-shadow|radial-gradient|background-clip:\s*text/i,
  );
  assert.doesNotMatch(visualSource, /\bcyan\b|\bviolet\b|\bportal\b|\bparticle/i);
});
