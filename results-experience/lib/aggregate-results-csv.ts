import {
  COHORT_IDS,
  COHORTS,
  METRIC_IDS,
  METRICS,
  MODEL_IDS,
  MODELS,
  RESULTS_STORY,
} from "./results.ts";

// Match the public revision used by the evidence register, rather than mutable main.
const SOURCE_BASE = "https://github.com/Ahmad986Ferdaws/ecg-trust-lab/blob/43cf519ad1e0a664a5689a913e9123fc147294c9/";
export const AGGREGATE_RESULTS_FILENAME = `ecg-trust-reported-results-${RESULTS_STORY.updatedFromAudit}.csv`;

export function serializeCsv(rows: readonly (readonly (string | number)[])[]): string {
  return rows.map((row) => row.map((value) => {
    const text = String(value);
    return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
  }).join(",")).join("\r\n") + "\r\n";
}

export function aggregateResultsCsv(): string {
  const rows: Array<Array<string | number>> = [[
    "cohort_id", "cohort", "cohort_role", "cohort_description",
    "model_id", "model", "metric_id", "metric", "mean", "sample_sd",
    "seed_count", "audit_date", "source_url", "scope",
  ]];

  for (const cohortId of COHORT_IDS) {
    const cohort = COHORTS[cohortId];
    for (const modelId of MODEL_IDS) {
      for (const metricId of METRIC_IDS) {
        const estimate = cohort.results[modelId][metricId];
        rows.push([
          cohort.id, cohort.name, cohort.role, cohort.description,
          modelId, MODELS[modelId].name, metricId, METRICS[metricId].label,
          String(estimate.mean), String(estimate.sd), RESULTS_STORY.seeds.length,
          RESULTS_STORY.updatedFromAudit, `${SOURCE_BASE}${cohort.evidencePath}`,
          "Reported aggregate retrospective research results; not clinical validation",
        ]);
      }
    }
  }
  return serializeCsv(rows);
}
