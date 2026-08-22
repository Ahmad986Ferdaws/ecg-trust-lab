/**
 * Audited, presentation-ready result data for the ECG Trust experience.
 *
 * Values are copied from the sealed PTB-XL r3 report and the frozen SPH r2
 * external-transport report. `as const` keeps the complete graph deeply
 * readonly at compile time while `satisfies` guards its public shape.
 */

export const MODEL_IDS = ["resnet1d", "ecg_transformer"] as const;
export type ModelId = (typeof MODEL_IDS)[number];

export const METRIC_IDS = ["auroc", "averagePrecision", "brier", "ece"] as const;
export type MetricId = (typeof METRIC_IDS)[number];
export type MetricDirection = "higher" | "lower";

export const LABEL_CODES = ["NORM", "MI", "STTC", "CD", "HYP"] as const;
export type LabelCode = (typeof LABEL_CODES)[number];

export const COHORT_IDS = [
  "ptbxl_fold10",
  "sph_primary",
  "sph_broad",
  "sph_no_ambiguous",
] as const;
export type CohortId = (typeof COHORT_IDS)[number];
export type DatasetId = "ptbxl" | "sph";
export type EvidenceRole = "confirmatory" | "exploratory" | "sensitivity";

export interface ModelDefinition {
  readonly id: ModelId;
  readonly name: string;
  readonly shortName: string;
  readonly description: string;
  readonly color: {
    readonly accent: string;
    readonly secondary: string;
    readonly glow: string;
    readonly surface: string;
  };
}

export interface MetricDefinition {
  readonly id: MetricId;
  readonly label: string;
  readonly shortLabel: string;
  readonly description: string;
  readonly direction: MetricDirection;
  readonly unit: "score";
  readonly domain: readonly [minimum: number, maximum: number];
  readonly decimals: number;
}

export interface MetricEstimate {
  readonly mean: number;
  readonly sd: number;
}

export type MetricSet = Readonly<Record<MetricId, MetricEstimate>>;
export type ArchitectureResults = Readonly<Record<ModelId, MetricSet>>;

export interface CohortCount {
  readonly records: number;
  readonly patients: number;
  readonly allZeroRows: number | null;
}

export interface PositiveCount {
  readonly records: number;
  readonly patients: number;
}

export interface CohortDefinition {
  readonly id: CohortId;
  readonly dataset: DatasetId;
  readonly role: EvidenceRole;
  readonly name: string;
  readonly shortName: string;
  readonly description: string;
  readonly count: CohortCount;
  readonly positiveCounts: Readonly<Record<LabelCode, PositiveCount>> | null;
  readonly adaptation: "sealed internal evaluation" | "none";
  readonly evidencePath: string;
  readonly results: ArchitectureResults;
}

export interface ScientificCaveat {
  readonly id: string;
  readonly title: string;
  readonly detail: string;
  readonly severity: "critical" | "important" | "context";
}

export const MODELS = {
  resnet1d: {
    id: "resnet1d",
    name: "1D ResNet",
    shortName: "ResNet",
    description: "A residual convolutional network operating directly on all 12 ECG leads.",
    color: {
      accent: "#55F6D1",
      secondary: "#2AB8FF",
      glow: "rgba(85, 246, 209, 0.45)",
      surface: "rgba(38, 203, 181, 0.12)",
    },
  },
  ecg_transformer: {
    id: "ecg_transformer",
    name: "ECG Transformer",
    shortName: "Transformer",
    description: "A capacity-matched patch transformer trained under the same protocol and budget.",
    color: {
      accent: "#B28CFF",
      secondary: "#FF62D6",
      glow: "rgba(178, 140, 255, 0.45)",
      surface: "rgba(154, 105, 255, 0.12)",
    },
  },
} as const satisfies Readonly<Record<ModelId, ModelDefinition>>;

export const METRICS = {
  auroc: {
    id: "auroc",
    label: "Macro AUROC",
    shortLabel: "AUROC",
    description: "How well predictions rank positive cases above negative cases across the five labels.",
    direction: "higher",
    unit: "score",
    domain: [0, 1],
    decimals: 6,
  },
  averagePrecision: {
    id: "averagePrecision",
    label: "Macro average precision",
    shortLabel: "AP",
    description: "Precision–recall quality averaged equally across the five diagnostic superclasses.",
    direction: "higher",
    unit: "score",
    domain: [0, 1],
    decimals: 6,
  },
  brier: {
    id: "brier",
    label: "Macro Brier score",
    shortLabel: "Brier",
    description: "Squared probability error; lower values indicate more accurate probabilities.",
    direction: "lower",
    unit: "score",
    domain: [0, 1],
    decimals: 6,
  },
  ece: {
    id: "ece",
    label: "Macro expected calibration error",
    shortLabel: "ECE",
    description: "Fixed-bin gap between stated confidence and observed frequency; lower is better.",
    direction: "lower",
    unit: "score",
    domain: [0, 1],
    decimals: 6,
  },
} as const satisfies Readonly<Record<MetricId, MetricDefinition>>;

export const LABELS = [
  {
    code: "NORM",
    name: "Normal ECG",
    description: "Recordings labeled as normal within the benchmark ontology.",
    color: "#63F5D2",
  },
  {
    code: "MI",
    name: "Myocardial infarction",
    description: "ECG patterns associated with myocardial infarction labels.",
    color: "#FF6F8F",
  },
  {
    code: "STTC",
    name: "ST/T change",
    description: "Repolarization-related ST-segment or T-wave changes.",
    color: "#FFCC66",
  },
  {
    code: "CD",
    name: "Conduction disturbance",
    description: "Electrical conduction abnormalities represented by the superclass.",
    color: "#66B8FF",
  },
  {
    code: "HYP",
    name: "Hypertrophy",
    description: "ECG labels associated with cardiac chamber hypertrophy.",
    color: "#CF8CFF",
  },
] as const satisfies readonly {
  readonly code: LabelCode;
  readonly name: string;
  readonly description: string;
  readonly color: string;
}[];

export const SOURCE_POPULATIONS = {
  ptbxlPublishedRelease: {
    records: 21_799,
    patients: 18_869,
    description: "Published PTB-XL source release.",
  },
  ptbxlCanonicalManifest: {
    records: 21_388,
    patients: 18_617,
    description: "Provenance-checked labeled manifest used by the project.",
  },
  sphCompleteMetadata: {
    records: 25_770,
    patients: 24_666,
    description: "Complete official SPH metadata population before frozen cohort filters.",
  },
} as const;
export const COHORTS = {
  ptbxl_fold10: {
    id: "ptbxl_fold10",
    dataset: "ptbxl",
    role: "confirmatory",
    name: "PTB-XL sealed fold 10",
    shortName: "PTB-XL · internal",
    description: "One sealed, patient-isolated final evaluation after model, calibration, and policy choices were frozen.",
    count: { records: 2_158, patients: 1_877, allZeroRows: null },
    positiveCounts: null,
    adaptation: "sealed internal evaluation",
    evidencePath: "reports/FINAL_RESULTS_PUBLIC.md",
    results: {
      resnet1d: {
        auroc: { mean: 0.921921, sd: 0.000913 },
        averagePrecision: { mean: 0.810248, sd: 0.003327 },
        brier: { mean: 0.08504, sd: 0.000718 },
        ece: { mean: 0.022744, sd: 0.002984 },
      },
      ecg_transformer: {
        auroc: { mean: 0.89742, sd: 0.00327 },
        averagePrecision: { mean: 0.76527, sd: 0.007739 },
        brier: { mean: 0.096807, sd: 0.001968 },
        ece: { mean: 0.025646, sd: 0.001163 },
      },
    },
  },
  sph_primary: {
    id: "sph_primary",
    dataset: "sph",
    role: "exploratory",
    name: "SPH primary mapped cohort",
    shortName: "SPH · primary",
    description: "Exact 10-second ECGs with at least one conservative direct mapping to a PTB-XL superclass.",
    count: { records: 15_698, patients: 15_193, allZeroRows: 0 },
    positiveCounts: {
      NORM: { records: 11_172, patients: 10_874 },
      MI: { records: 138, patients: 131 },
      STTC: { records: 3_030, patients: 2_947 },
      CD: { records: 1_510, patients: 1_453 },
      HYP: { records: 113, patients: 110 },
    },
    adaptation: "none",
    evidencePath: "publication/external_transport_sph_r2/FINAL_RESULTS.md",
    results: {
      resnet1d: {
        auroc: { mean: 0.930912, sd: 0.000964 },
        averagePrecision: { mean: 0.698955, sd: 0.006752 },
        brier: { mean: 0.061301, sd: 0.000248 },
        ece: { mean: 0.052477, sd: 0.000877 },
      },
      ecg_transformer: {
        auroc: { mean: 0.924088, sd: 0.001231 },
        averagePrecision: { mean: 0.657838, sd: 0.007557 },
        brier: { mean: 0.064153, sd: 0.003962 },
        ece: { mean: 0.06148, sd: 0.006313 },
      },
    },
  },
  sph_broad: {
    id: "sph_broad",
    dataset: "sph",
    role: "sensitivity",
    name: "SPH broad exact-10-second cohort",
    shortName: "SPH · broad",
    description: "All exact-10-second SPH records; 3,144 records without a direct target mapping are treated operationally as all-zero only for this sensitivity view.",
    count: { records: 18_842, patients: 18_157, allZeroRows: 3_144 },
    positiveCounts: {
      NORM: { records: 11_172, patients: 10_874 },
      MI: { records: 138, patients: 131 },
      STTC: { records: 3_030, patients: 2_947 },
      CD: { records: 1_510, patients: 1_453 },
      HYP: { records: 113, patients: 110 },
    },
    adaptation: "none",
    evidencePath: "publication/external_transport_sph_r2/FINAL_RESULTS.md",
    results: {
      resnet1d: {
        auroc: { mean: 0.904714, sd: 0.001118 },
        averagePrecision: { mean: 0.642155, sd: 0.006472 },
        brier: { mean: 0.076688, sd: 0.000146 },
        ece: { mean: 0.071016, sd: 0.000858 },
      },
      ecg_transformer: {
        auroc: { mean: 0.898652, sd: 0.001028 },
        averagePrecision: { mean: 0.601741, sd: 0.008373 },
        brier: { mean: 0.078817, sd: 0.003018 },
        ece: { mean: 0.078034, sd: 0.004649 },
      },
    },
  },
  sph_no_ambiguous: {
    id: "sph_no_ambiguous",
    dataset: "sph",
    role: "sensitivity",
    name: "SPH no-ambiguous mapped cohort",
    shortName: "SPH · no ambiguity",
    description: "The primary cohort after removing every record carrying a pre-specified ambiguous primary code.",
    count: { records: 15_563, patients: 15_066, allZeroRows: 0 },
    positiveCounts: {
      NORM: { records: 11_172, patients: 10_874 },
      MI: { records: 131, patients: 124 },
      STTC: { records: 2_981, patients: 2_899 },
      CD: { records: 1_470, patients: 1_417 },
      HYP: { records: 64, patients: 63 },
    },
    adaptation: "none",
    evidencePath: "publication/external_transport_sph_r2/FINAL_RESULTS.md",
    results: {
      resnet1d: {
        auroc: { mean: 0.928405, sd: 0.001382 },
        averagePrecision: { mean: 0.666044, sd: 0.005249 },
        brier: { mean: 0.060795, sd: 0.000245 },
        ece: { mean: 0.052181, sd: 0.000772 },
      },
      ecg_transformer: {
        auroc: { mean: 0.920511, sd: 0.000971 },
        averagePrecision: { mean: 0.627274, sd: 0.003484 },
        brier: { mean: 0.063547, sd: 0.003912 },
        ece: { mean: 0.061313, sd: 0.006297 },
      },
    },
  },
} as const satisfies Readonly<Record<CohortId, CohortDefinition>>;

export const STUDY_NARRATIVE = {
  eyebrow: "One benchmark. Two architectures. A second hospital.",
  headline: "Trust is measured after the model leaves home.",
  summary:
    "The 1D ResNet led the matched transformer on sealed PTB-XL evaluation, then retained a smaller advantage when all six frozen models crossed into SPH without tuning or recalibration.",
  chapters: [
    {
      id: "seal",
      index: "01",
      title: "Seal the test",
      body: "Patients were isolated by fold, decisions were frozen on earlier folds, and fold 10 was opened once for the confirmatory comparison.",
    },
    {
      id: "compare",
      index: "02",
      title: "Compare more than ranking",
      body: "AUROC and average precision measure discrimination; Brier score and ECE test whether the probabilities themselves deserve confidence.",
    },
    {
      id: "transport",
      index: "03",
      title: "Cross the boundary unchanged",
      body: "The same three seeds per architecture were transported to SPH with no target-domain training, selection, preprocessing adaptation, or recalibration.",
    },
    {
      id: "limit",
      index: "04",
      title: "Keep the claim honest",
      body: "The result is strong retrospective research evidence—not proof of diagnostic safety, clinical validity, or deployment readiness.",
    },
  ],
  conclusion:
    "The result is not simply that one network scored higher. It is that the advantage survived a governed, no-adaptation transport stress test while every limitation stayed visible.",
} as const;

export const SCIENTIFIC_CAVEATS = [
  {
    id: "research-only",
    title: "Research only",
    detail: "Neither the PTB-XL benchmark nor the SPH transport study establishes clinical validity, diagnostic safety, medical-device performance, or fitness for patient care.",
    severity: "critical",
  },
  {
    id: "auroc-not-accuracy",
    title: "AUROC is not accuracy",
    detail: "A macro AUROC of 0.921921 describes ranking discrimination across labels; it must not be presented as 92.2% diagnostic accuracy.",
    severity: "critical",
  },
  {
    id: "retrospective-transport",
    title: "External transport, not clinical validation",
    detail: "SPH was a retrospective, one-pass stress test. No SPH data were used for training, model selection, preprocessing adaptation, recalibration, thresholds, or confidence gates.",
    severity: "important",
  },
  {
    id: "ontology-bridge",
    title: "Unadjudicated ontology bridge",
    detail: "The SPH AHA-to-PTB-superclass mapping was deliberately conservative but was not clinically adjudicated.",
    severity: "important",
  },
  {
    id: "broad-unknowns",
    title: "Broad-cohort absences are unknown",
    detail: "The broad sensitivity treats 3,144 records without a direct mapping as operationally all-zero; those missing mappings are unknown, not verified negatives.",
    severity: "important",
  },
  {
    id: "rare-endpoints",
    title: "Rare transported endpoints",
    detail: "The SPH primary cohort has only 138 MI-positive and 113 HYP-positive ECGs, so estimates and some bootstrap replicates can be unstable.",
    severity: "important",
  },
  {
    id: "calibration-comparison",
    title: "No internal ECE winner claimed",
    detail: "Every paired PTB-XL ECE interval crossed zero, and fixed-bin ECE has resampling and binning sensitivity.",
    severity: "context",
  },
  {
    id: "blindness-deviation",
    title: "Recorded protocol deviation",
    detail: "DEV-001 documents bounded exposure to raw fold-10 label-bearing metadata; no exposed value informed a model, calibration, gate, or reporting choice, but complete operator-level outcome blindness is not claimed.",
    severity: "context",
  },
] as const satisfies readonly ScientificCaveat[];

export const RESULTS_STORY = {
  models: MODELS,
  metrics: METRICS,
  labels: LABELS,
  sourcePopulations: SOURCE_POPULATIONS,
  cohorts: COHORTS,
  narrative: STUDY_NARRATIVE,
  caveats: SCIENTIFIC_CAVEATS,
  seeds: [2026, 2027, 2028],
  statistic: "mean ± sample standard deviation across three frozen seeds",
  updatedFromAudit: "2026-08-22",
} as const;
