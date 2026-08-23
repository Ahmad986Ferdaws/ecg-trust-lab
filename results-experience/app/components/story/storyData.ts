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
  id: "ptb-xl" | "sph";
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
    plotDomain: [0, 0.07],
  },
};

/** Audited research outputs. Values are means ± sample SD across three frozen seeds. */
export const AUDITED_BENCHMARKS: BenchmarkDataset[] = [
  {
    id: "ptb-xl",
    tabLabel: "Sealed PTB-XL",
    eyebrow: "In-distribution benchmark",
    title: "The ResNet clears the 0.92 AUROC benchmark.",
    description:
      "Two architectures face the same sealed superclass task. Discrimination, precision, and calibration are shown together—not collapsed into a single score.",
    models: [
      {
        id: "resnet",
        name: "1D ResNet",
        descriptor: "Convolutional baseline",
        accent: "resnet",
        metrics: {
          auroc: { value: 0.921921, spread: 0.000913 },
          averagePrecision: { value: 0.810248, spread: 0.003327 },
          brier: { value: 0.08504, spread: 0.000718 },
          ece: { value: 0.022744, spread: 0.002984 },
        },
      },
      {
        id: "transformer",
        name: "ECG Transformer",
        descriptor: "Attention architecture",
        accent: "transformer",
        metrics: {
          auroc: { value: 0.89742, spread: 0.00327 },
          averagePrecision: { value: 0.76527, spread: 0.007739 },
          brier: { value: 0.096807, spread: 0.001968 },
          ece: { value: 0.025646, spread: 0.001163 },
        },
      },
    ],
  },
  {
    id: "sph",
    tabLabel: "SPH transport",
    eyebrow: "Frozen external transport",
    title: "Frozen-model performance on the SPH primary cohort.",
    description:
      "The same frozen models are transported to the primary SPH cohort. Strong AUROC persists while average precision and calibration expose the dataset shift.",
    models: [
      {
        id: "resnet",
        name: "1D ResNet",
        descriptor: "Frozen for transport",
        accent: "resnet",
        metrics: {
          auroc: { value: 0.930912, spread: 0.000964 },
          averagePrecision: { value: 0.698955, spread: 0.006752 },
          brier: { value: 0.061301, spread: 0.000248 },
          ece: { value: 0.052477, spread: 0.000877 },
        },
      },
      {
        id: "transformer",
        name: "ECG Transformer",
        descriptor: "Frozen for transport",
        accent: "transformer",
        metrics: {
          auroc: { value: 0.924088, spread: 0.001231 },
          averagePrecision: { value: 0.657838, spread: 0.007557 },
          brier: { value: 0.064153, spread: 0.003962 },
          ece: { value: 0.06148, spread: 0.006313 },
        },
      },
    ],
  },
];

export const AUDITED_TRANSPORT_COHORT: CohortSnapshot = {
  sourceLabel: "PTB-XL",
  destinationLabel: "SPH",
  primary: {
    ecgs: 15_698,
    patients: 15_193,
    label: "Primary transport cohort",
  },
  broad: {
    ecgs: 18_842,
    patients: 18_157,
    label: "Broad sensitivity cohort",
  },
};
