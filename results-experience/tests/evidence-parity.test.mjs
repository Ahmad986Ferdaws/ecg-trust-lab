import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { COHORTS, SOURCE_SUPPORT_COMPLETION } from "../lib/results.ts";
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
async function readNumericCsv(path) {
  const csv = await readFile(new URL(path, publication), "utf8");
  assert.ok(!csv.includes('"'), `${path}: expected unquoted CSV schema`);
  const [header, ...lines] = csv.trim().split(/\r?\n/).map((line) => line.split(","));
  return lines.map((fields) => {
    assert.equal(fields.length, header.length, `${path}: CSV column count`);
    return Object.fromEntries(header.map((key, i) => [key, fields[i]]));
  });
}
const architectureRows = await readNumericCsv("results/tables/architecture_metrics.csv");

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

test("PTB-XL displayed cohort totals match the model card and member table", async () => {
  const card = await readFile(new URL("../../docs/MODEL_CARD_PTBXL_SUPERCLASS_R3.md", import.meta.url), "utf8");
  const row = /^\| Final cohort \| ([\d,]+) ECGs from ([\d,]+) patients \|$/m.exec(card);
  assert.ok(row, "model card must contain the final-cohort evidence row");
  const records = Number(row[1].replaceAll(",", ""));
  const patients = Number(row[2].replaceAll(",", ""));
  assert.deepEqual(COHORTS.ptbxl_fold10.count, { records, patients, allZeroRows: null });
  const members = await readNumericCsv("results/tables/member_metrics.csv");
  assert.equal(members.length, 6, "six sealed model members required");
  for (const member of members) {
    assert.equal(Number(member.n_samples), records, `${member.member_id}: sealed cohort size`);
  }
});

for (const directory of ["results", "external_transport_sph_r2"]) {
  test(`${directory}: published assets match their sealed checksum inventory`, async () => {
    const base = new URL(`${directory}/`, publication);
    const inventory = await readFile(new URL("SHA256SUMS.txt", base), "utf8");
    for (const line of inventory.trim().split(/\r?\n/)) {
      const match = /^([a-f0-9]{64}) {2}([A-Za-z0-9_./-]+)$/.exec(line);
      assert.ok(match, "invalid public checksum inventory entry");
      const [, expected, path] = match;
      assert.ok(!path.startsWith("/") && !path.split("/").includes(".."), "unsafe inventory path");
      const actual = createHash("sha256").update(await readFile(new URL(path, base))).digest("hex");
      assert.equal(actual, expected, `${directory}/${path}: sealed artifact checksum`);
    }
  });
}

const supportReport = await readFile(
  new URL("../../docs/TRUST_SENTINEL_OOD_COMPLETION_RESULT.md", import.meta.url), "utf8",
);
const supportRows = new Map(supportReport.split(/\r?\n/)
  .filter((line) => line.startsWith("| "))
  .map((line) => line.split("|").slice(1, -1).map((cell) => cell.trim()))
  .map(([key, ...values]) => [key, values]));
function supportField(name) {
  const prefix = `- **${name}:** `;
  const line = supportReport.split(/\r?\n/).find((line) => line.startsWith(prefix));
  assert.ok(line, `completion report: missing ${name}`);
  return line.slice(prefix.length).replaceAll("`", "");
}
const percent = (value) => `${(value * 100).toFixed(4)}%`;

test("source-support identity and unfavorable eligibility match the completion report", () => {
  const s = SOURCE_SUPPORT_COMPLETION;
  assert.equal(s.protocol, supportField("Protocol"));
  assert.equal(`${s.completed} UTC`, supportField("Completed"));
  assert.equal(s.status, supportField("Status"));
  assert.equal(s.status, "SOURCE_SUPPORT_GATE_TARGET_MISSED");
  assert.equal(supportField("Research-bundle eligible"), "no");
  assert.equal(s.researchBundleEligible, false);
  assert.equal(supportField("Scope"), "retrospective PTB-XL source-domain development only");
  assert.equal(s.claimScope, "retrospective_ptbxl_source_domain_development_only");
  assert.equal(s.oodPositiveEvaluation, "NOT_EVALUATED");
  assert.match(supportReport, /following outcomes remain exactly `NOT_EVALUATED`/);
  const bootstrap = /with ([\d,]+) patient-cluster bootstrap/.exec(supportReport);
  assert.ok(bootstrap, "completion report: missing bootstrap count");
  assert.equal(s.bootstrapReplicates, Number(bootstrap[1].replaceAll(",", "")));
  assert.ok(s.validation.oneSidedUpper95 > s.validation.targetMaximum);
});

test("source-support role counts and outcomes match public report precision", () => {
  const { roles, validation: v } = SOURCE_SUPPORT_COMPLETION;
  assert.deepEqual(Object.keys(roles).sort(), ["reference", "sourceValidation", "thresholdFit"]);
  for (const [name, code] of [["reference", "R"], ["thresholdFit", "B"], ["sourceValidation", "C"]]) {
    const role = roles[name];
    assert.equal(role.code, code);
    const row = supportRows.get(code);
    assert.ok(row, `completion report: missing role ${code}`);
    assert.equal(role.records, Number(row[2].replaceAll(",", "")), `${code}/records`);
    assert.equal(role.patients, Number(row[3].replaceAll(",", "")), `${code}/patients`);
  }
  const expected = {
    "Source-validation ECGs retained": `${v.retained} / ${roles.sourceValidation.records}`,
    "Source-validation ECGs rejected": `${v.rejected} / ${roles.sourceValidation.records}`,
    "Source-support coverage": percent(v.supportCoverage),
    "Record-level false rejection": percent(v.recordFalseRejection),
    "Patient-equalized false rejection": percent(v.patientEqualizedFalseRejection),
    "Patients with any rejected ECG": `${v.patientsWithAnyRejection} / ${roles.sourceValidation.patients} (${percent(v.patientAnyFalseRejection)})`,
    "Two-sided 95% patient-cluster interval": v.twoSided95.map(percent).join("–"),
    "One-sided 95% patient-cluster upper bound": percent(v.oneSidedUpper95),
    "Preregistered maximum upper bound": percent(v.targetMaximum),
    "Validation observations tied at the threshold": String(v.thresholdTies),
  };
  for (const [outcome, displayed] of Object.entries(expected)) {
    assert.deepEqual(supportRows.get(outcome), [displayed], outcome);
  }
});
