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

test("server-renders the complete ECG evidence experience", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /<title>ECG Trust Lab — Results in Motion<\/title>/i);
  assert.match(
    html,
    /property="og:image" content="https:\/\/results\.example\.test\/og\.png"/i,
  );
  assert.match(
    html,
    /name="twitter:image" content="https:\/\/results\.example\.test\/og\.png"/i,
  );
  assert.doesNotMatch(html, /localhost(?::\d+)?\/og\.png/i);
  assert.match(html, /Trust,/);
  assert.match(html, /One race\./);
  assert.match(html, /Keep the model frozen\./);
  assert.match(html, /Strong evidence\./);
  assert.match(html, /Research system · not a medical device/);
  assert.match(html, /0\.922/);
  assert.match(html, /0\.931/);
  assert.match(html, /15,193/);
  assert.match(html, /494/);
  assert.doesNotMatch(html, /codex-preview|Your site is taking shape|Building your site/i);
});

test("ships the social card and frozen audited result source", async () => {
  const [image, imageStats, resultsSource] = await Promise.all([
    readFile(new URL("../public/og.png", import.meta.url)),
    stat(new URL("../public/og.png", import.meta.url)),
    readFile(new URL("../lib/results.ts", import.meta.url), "utf8"),
  ]);

  assert.deepEqual([...image.subarray(0, 8)], [137, 80, 78, 71, 13, 10, 26, 10]);
  assert.ok(imageStats.size > 1_000_000);
  assert.match(resultsSource, /mean: 0\.921921, sd: 0\.000913/);
  assert.match(resultsSource, /mean: 0\.930912, sd: 0\.000964/);
  assert.match(resultsSource, /records: 15_698, patients: 15_193/);
  assert.match(resultsSource, /Neither the PTB-XL benchmark nor the SPH transport study establishes clinical validity/);
});
