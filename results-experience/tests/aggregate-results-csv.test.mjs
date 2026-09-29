import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { COHORT_IDS, COHORTS, METRIC_IDS, MODEL_IDS, RESULTS_STORY } from "../lib/results.ts";
import { aggregateResultsCsv, serializeCsv } from "../lib/aggregate-results-csv.ts";

// Independent parser keeps parity checks sensitive to quoting and field alignment.
function parseCsv(csv) {
  const rows = [];
  let row = [], field = "", quoted = false;
  for (let i = 0; i < csv.length; i += 1) {
    const char = csv[i];
    if (char === '"') {
      if (quoted && csv[i + 1] === '"') {
        field += '"';
        i += 1;
      } else quoted = !quoted;
    } else if (!quoted && char === ",") {
      row.push(field);
      field = "";
    } else if (!quoted && char === "\r" && csv[i + 1] === "\n") {
      rows.push([...row, field]);
      row = [];
      field = "";
      i += 1;
    } else field += char;
  }
  assert.equal(quoted, false, "CSV quotes must close");
  assert.equal(field, "", "every row must end with CRLF");
  assert.deepEqual(row, []);
  return rows;
}

test("CSV escaping preserves commas, quotes, line breaks and empty fields", () => {
  const rows = [["comma,inside", 'quote"inside', "line\nbreak", "carriage\rreturn", "", 0.25]];
  const csv = serializeCsv(rows);
  assert.equal(csv, '"comma,inside","quote""inside","line\nbreak","carriage\rreturn",,0.25\r\n');
  assert.deepEqual(parseCsv(csv), [rows[0].map(String)]);
});

test("download contains each audited aggregate once, with exact library numbers and provenance", () => {
  const csv = aggregateResultsCsv();
  assert.equal(csv, aggregateResultsCsv(), "export bytes must not depend on time or interaction state");
  const [headers, ...data] = parseCsv(csv);
  assert.deepEqual(headers, [
    "cohort_id", "cohort", "cohort_role", "cohort_description",
    "model_id", "model", "metric_id", "metric", "mean", "sample_sd",
    "seed_count", "audit_date", "source_url", "scope",
  ]);
  assert.equal(data.length, COHORT_IDS.length * MODEL_IDS.length * METRIC_IDS.length);
  const rows = data.map((fields) => {
    assert.equal(fields.length, headers.length);
    return Object.fromEntries(headers.map((header, index) => [header, fields[index]]));
  });
  const expectedOrder = COHORT_IDS.flatMap((cohort) => MODEL_IDS.flatMap((model) =>
    METRIC_IDS.map((metric) => `${cohort}/${model}/${metric}`)));
  assert.deepEqual(rows.map((row) => `${row.cohort_id}/${row.model_id}/${row.metric_id}`), expectedOrder);

  for (const row of rows) {
    const cohort = COHORTS[row.cohort_id];
    const estimate = cohort.results[row.model_id][row.metric_id];
    assert.equal(row.mean, String(estimate.mean));
    assert.equal(row.sample_sd, String(estimate.sd));
    assert.equal(Number(row.mean), estimate.mean, "CSV parsing must round-trip the library mean");
    assert.equal(Number(row.sample_sd), estimate.sd, "CSV parsing must round-trip the library sample SD");
    assert.equal(Number(row.seed_count), RESULTS_STORY.seeds.length);
    assert.equal(row.cohort_role, cohort.role);
    assert.equal(row.cohort_description, cohort.description);
    assert.equal(row.audit_date, RESULTS_STORY.updatedFromAudit);
    assert.equal(row.source_url,
      `https://github.com/Ahmad986Ferdaws/ecg-trust-lab/blob/43cf519ad1e0a664a5689a913e9123fc147294c9/${cohort.evidencePath}`);
    assert.match(row.scope, /aggregate retrospective research.*not clinical validation/);
  }
});

test("parsed exported values independently match the frozen publication aggregates", async () => {
  const publication = new URL("../../publication/", import.meta.url);
  const sourceCsv = await readFile(new URL("results/tables/architecture_metrics.csv", publication), "utf8");
  const [sourceHeaders, ...sourceRows] = parseCsv(sourceCsv.trimEnd().replace(/\r?\n/g, "\r\n") + "\r\n");
  const architectureRows = sourceRows.map((fields) =>
    Object.fromEntries(sourceHeaders.map((header, i) => [header, fields[i]])));
  const sourceMetrics = { auroc: "roc_auc", averagePrecision: "average_precision", brier: "brier_score", ece: "ece" };
  const sourceCohorts = { sph_primary: "primary_mapped", sph_broad: "broad_exact10", sph_no_ambiguous: "no_ambiguous_mapped" };
  const summaries = new Map();
  const [headers, ...data] = parseCsv(aggregateResultsCsv());
  for (const fields of data) {
    const row = Object.fromEntries(headers.map((header, i) => [header, fields[i]]));
    let mean, sd;
    if (row.cohort_id === "ptbxl_fold10") {
      const matching = architectureRows.filter((source) =>
        source.architecture === row.model_id && source.metric === sourceMetrics[row.metric_id]);
      assert.equal(matching.length, 1);
      mean = Number(matching[0].mean);
      sd = Number(matching[0].sample_sd);
    } else {
      const path = `external_transport_sph_r2/architecture_summaries/${sourceCohorts[row.cohort_id]}__${row.model_id}.json`;
      if (!summaries.has(path)) summaries.set(path, JSON.parse(await readFile(new URL(path, publication), "utf8")));
      const value = summaries.get(path).summary.statistics[`frozen_temperature_calibrated.macro.${sourceMetrics[row.metric_id]}`];
      mean = value.mean;
      sd = value.sample_standard_deviation;
    }
    // The audited public library reports six decimal places from these artifacts.
    assert.equal(Number(row.mean), Number(mean.toFixed(6)));
    assert.equal(Number(row.sample_sd), Number(sd.toFixed(6)));
  }
});
