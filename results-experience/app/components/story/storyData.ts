import { COHORTS, METRIC_IDS, MODELS, type CohortId, type ModelId } from "../../../lib/results.ts";

export type MetricKey = "auroc" | "averagePrecision" | "brier" | "ece";

export type MetricEstimate = {
  value: number;
  spread: number;
};

export type ModelBenchmark = {
  id: "resnet" | "transformer";
  name: string;
  descriptor: string;
  accent: "resnet" | "transformer";
  metrics: Record<MetricKey, MetricEstimate>;
};

export type BenchmarkDataset = {
  id: "ptb-xl" | "sph" | "sph-broad" | "sph-no-ambiguous";
  cohortId: CohortId;
  tabLabel: string;
  eyebrow: string;
  title: string;
  description: string;
  models: [ModelBenchmark, ModelBenchmark];
};

export type CohortSnapshot = {
  sourceLabel: string;
  destinationLabel: string;
  primary: {
    ecgs: number;
    patients: number;
    label: string;
  };
  broad: {
    ecgs: number;
    patients: number;
    label: string;
  };
  noAmbiguous: {
    ecgs: number;
    patients: number;
    label: string;
  };
};

export const METRIC_META: Record<
  MetricKey,
  {
    label: string;
    longLabel: string;
    direction: "higher" | "lower";
    plotDomain: readonly [number, number];
  }
> = {
  auroc: {
    label: "AUROC",
    longLabel: "Area under the receiver operating curve",
    direction: "higher",
    plotDomain: [0.85, 0.95],
  },
  averagePrecision: {
    label: "AP",
    longLabel: "Average precision",
    direction: "higher",
    plotDomain: [0.6, 0.85],
  },
  brier: {
    label: "Brier",
    longLabel: "Brier score",
    direction: "lower",
    plotDomain: [0.05, 0.11],
  },
  ece: {
    label: "ECE",
    longLabel: "Expected calibration error",
    direction: "lower",
    plotDomain: [0, 0.1],
  },
};

/** Every displayed estimate is adapted from the same audited result model. */
function modelBenchmark(cohortId: CohortId, modelId: ModelId): ModelBenchmark {
  const id = modelId === "resnet1d" ? "resnet" : "transformer";
  const result = COHORTS[cohortId].results[modelId];
  return {
    id,
    name: MODELS[modelId].name,
    descriptor: "Three frozen seeds",
    accent: id,
    metrics: Object.fromEntries(METRIC_IDS.map((metric) => [
      metric, { value: result[metric].mean, spread: result[metric].sd },
    ])) as Record<MetricKey, MetricEstimate>,
  };
}

const benchmarkViews: Array<Omit<BenchmarkDataset, "models">> = [
  {
    id: "ptb-xl",
    cohortId: "ptbxl_fold10",
    tabLabel: "Sealed PTB-XL",
    eyebrow: "In-distribution benchmark",
    title: "The ResNet clears the 0.92 AUROC benchmark.",
    description:
      "Two architectures face the same sealed superclass task. Discrimination, precision, and calibration are shown together—not collapsed into a single score.",
  },
  {
    id: "sph",
    cohortId: "sph_primary",
    tabLabel: "SPH primary",
    eyebrow: "Frozen external transport",
    title: "Frozen-model performance on the SPH primary cohort.",
    description:
      "The same frozen models are transported to the primary SPH cohort. Strong AUROC persists while average precision and calibration expose the dataset shift.",
  },
  {
    id: "sph-broad",
    cohortId: "sph_broad",
    tabLabel: "SPH broad",
    eyebrow: "Sensitivity · unknown absences",
    title: "Broad sensitivity includes unmapped records.",
    description:
      `${COHORTS.sph_broad.description} Missing mappings are unknown, not verified negative diagnoses. This sensitivity analysis does not replace the primary transport result.`,
  },
  {
    id: "sph-no-ambiguous",
    cohortId: "sph_no_ambiguous",
    tabLabel: "SPH no ambiguity",
    eyebrow: "Sensitivity · restricted mapping",
    title: "Sensitivity to removing ambiguous mapped codes.",
    description:
      `${COHORTS.sph_no_ambiguous.description} Removing these codes changes the label mix and reduces rare endpoint counts; the underlying ontology bridge remains unadjudicated.`,
  },
];

export const AUDITED_BENCHMARKS: BenchmarkDataset[] = benchmarkViews.map((view) => ({
  ...view,
  models: [modelBenchmark(view.cohortId, "resnet1d"), modelBenchmark(view.cohortId, "ecg_transformer")],
}));

function cohortSnapshot(cohortId: CohortId, label: string) {
  const { records, patients } = COHORTS[cohortId].count;
  return { ecgs: records, patients, label };
}

export const AUDITED_TRANSPORT_COHORT: CohortSnapshot = {
  sourceLabel: "PTB-XL",
  destinationLabel: "SPH",
  primary: cohortSnapshot("sph_primary", "Primary transport cohort"),
  broad: cohortSnapshot("sph_broad", "Broad sensitivity cohort"),
  noAmbiguous: cohortSnapshot("sph_no_ambiguous", "No-ambiguous sensitivity cohort"),
};
