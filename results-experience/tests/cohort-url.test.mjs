import assert from "node:assert/strict";
import test from "node:test";
import { COHORT_IDS } from "../lib/results.ts";
import { cohortFromSearch, urlForCohort } from "../lib/cohort-url.ts";

test("each audited cohort survives a shareable URL round trip", () => {
  for (const cohort of COHORT_IDS) {
    const path = urlForCohort("https://results.example.test/?ref=paper#evidence", cohort);
    const url = new URL(path, "https://results.example.test");
    assert.equal(cohortFromSearch(url.search), cohort);
    assert.equal(url.searchParams.get("ref"), "paper");
    assert.equal(url.hash, "#evidence");
  }
});

test("missing, empty, malformed and unknown choices request the default view", () => {
  for (const search of ["", "?cohort=", "?cohort=unknown", "?cohort=SPH_BROAD", "?cohort=%E0%A4%A"]) {
    assert.equal(cohortFromSearch(search), null, search);
  }
});

test("selection replaces duplicated cohort parameters and retains unrelated navigation state", () => {
  const url = new URL(urlForCohort(
    "https://results.example.test/notebook?ref=paper&tag=one&cohort=sph_primary&tag=two&cohort=bad#source-support",
    "sph_broad",
  ), "https://results.example.test");
  assert.equal(url.pathname, "/notebook");
  assert.equal(url.hash, "#source-support");
  assert.deepEqual(url.searchParams.getAll("tag"), ["one", "two"]);
  assert.deepEqual(url.searchParams.getAll("cohort"), ["sph_broad"]);
  assert.equal(url.searchParams.get("ref"), "paper");
});

test("selecting the current cohort does not change the URL", () => {
  const path = "/?cohort=sph_no_ambiguous#failure-lab";
  assert.equal(urlForCohort(`https://results.example.test${path}`, "sph_no_ambiguous"), path);
});
