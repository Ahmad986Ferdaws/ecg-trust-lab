import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { COHORTS } from "../lib/results.ts";
import {
  AUDITED_BENCHMARKS,
  AUDITED_TRANSPORT_COHORT,
} from "../app/components/story/storyData.ts";

const publication = new URL("../../publication/", import.meta.url);
const metricNames = {
  auroc: "roc_auc",
  averagePrecision: "average_precision",
  brier: "brier_score",
  ece: "ece",
};
const cohortNames = {
  sph_primary: "primary_mapped",
  sph_broad: "broad_exact10",
  sph_no_ambiguous: "no_ambiguous_mapped",
};
const modelNames = ["resnet1d", "ecg_transformer"];
const readJson = async (path) =>
  JSON.parse(await readFile(new URL(path, publication), "utf8"));

// This sealed table contains unquoted numeric/machine-name fields. Fail rather
// than silently misparse if a future editorial change introduces CSV quoting.
const csv = await readFile(
  new URL("results/tables/architecture_metrics.csv", publication), "utf8",
);
assert.ok(!csv.includes('"'), "architecture table requires an unquoted CSV schema");
const [header, ...lines] = csv.trim().split(/\r?\n/).map((line) => line.split(","));
const architectureRows = lines.map((fields) => {
  assert.equal(fields.length, header.length, "architecture CSV column count");
  return Object.fromEntries(header.map((key, i) => [key, fields[i]]));
});

async function expectedMetrics(cohort, model) {
  const statistics = cohort === "ptbxl_fold10" ? null :
    (await readJson(`external_transport_sph_r2/architecture_summaries/${cohortNames[cohort]}__${model}.json`)).summary.statistics;
  return Object.fromEntries(Object.entries(metricNames).map(([display, source]) => {
    let mean, sd;
    if (statistics === null) {
      const rows = architectureRows.filter((row) =>
        row.architecture === model && row.metric === source);
      assert.equal(rows.length, 1, `${model}/${source}: one sealed aggregate required`);
      mean = Number(rows[0].mean);
      sd = Number(rows[0].sample_sd);
    } else {
      const value = statistics[`frozen_temperature_calibrated.macro.${source}`];
      mean = value.mean;
      sd = value.sample_standard_deviation;
    }
    assert.ok(Number.isFinite(mean) && Number.isFinite(sd), `${cohort}/${model}/${source}`);
    return [display, { mean: Number(mean.toFixed(6)), sd: Number(sd.toFixed(6)) }];
  }));
}

test("library contains exactly the four audited cohorts", () => {
  assert.deepEqual(Object.keys(COHORTS).sort(), ["ptbxl_fold10", ...Object.keys(cohortNames)].sort());
});

for (const cohort of ["ptbxl_fold10", ...Object.keys(cohortNames)]) {
  test(`${cohort}: every library metric matches sealed mean and sample SD`, async () => {
    const expected = Object.fromEntries(await Promise.all(modelNames.map(async (model) =>
      [model, await expectedMetrics(cohort, model)])));
    assert.deepEqual(COHORTS[cohort].results, expected);
  });
}

test("both story benchmarks match the sealed artifacts independently", async () => {
  assert.deepEqual(AUDITED_BENCHMARKS.map(({ id }) => id).sort(), ["ptb-xl", "sph"]);
  for (const dataset of AUDITED_BENCHMARKS) {
    const cohort = dataset.id === "ptb-xl" ? "ptbxl_fold10" : "sph_primary";
    assert.deepEqual(dataset.models.map(({ id }) => id).sort(), ["resnet", "transformer"]);
    for (const model of dataset.models) {
      const modelId = model.id === "resnet" ? "resnet1d" : "ecg_transformer";
      const expected = await expectedMetrics(cohort, modelId);
      const presentation = Object.fromEntries(Object.entries(expected).map(([metric, value]) =>
        [metric, { value: value.mean, spread: value.sd }]));
      assert.deepEqual(model.metrics, presentation, `${dataset.id}/${model.id}`);
    }
  }
});

test("SPH cohort totals and all positive counts match the frozen cohort summary", async () => {
  const { cohorts } = await readJson("external_transport_sph_r2/cohort_summary.json");
  for (const [display, source] of Object.entries(cohortNames)) {
    const expected = cohorts[source];
    assert.deepEqual(COHORTS[display].count, {
      records: expected.records, patients: expected.patients, allZeroRows: expected.all_zero_rows,
    }, `${display}/count`);
    const positives = Object.fromEntries(Object.entries(expected.positive_records).map(([label, records]) =>
      [label, { records, patients: expected.positive_patients[label] }]));
    assert.deepEqual(COHORTS[display].positiveCounts, positives, `${display}/positiveCounts`);
  }
  for (const [display, source] of [["primary", "primary_mapped"], ["broad", "broad_exact10"]]) {
    assert.equal(AUDITED_TRANSPORT_COHORT[display].ecgs, cohorts[source].records, `${display}/ecgs`);
    assert.equal(AUDITED_TRANSPORT_COHORT[display].patients, cohorts[source].patients, `${display}/patients`);
  }
});

for (const directory of ["results", "external_transport_sph_r2"]) {
  test(`${directory}: published assets match their sealed checksum inventory`, async () => {
    const base = new URL(`${directory}/`, publication);
    const inventory = await readFile(new URL("SHA256SUMS.txt", base), "utf8");
    for (const line of inventory.trim().split(/\r?\n/)) {
      const match = /^([a-f0-9]{64})  ([A-Za-z0-9_./-]+)$/.exec(line);
      assert.ok(match, "invalid public checksum inventory entry");
      const [, expected, path] = match;
      assert.ok(!path.startsWith("/") && !path.split("/").includes(".."), "unsafe inventory path");
      const actual = createHash("sha256").update(await readFile(new URL(path, base))).digest("hex");
      assert.equal(actual, expected, `${directory}/${path}: sealed artifact checksum`);
    }
  });
}
