"use client";

import {
  type CSSProperties,
  type KeyboardEvent,
  useRef,
  useState,
  useSyncExternalStore,
} from "react";
import styles from "./story.module.css";
import {
  AUDITED_BENCHMARKS,
  METRIC_META,
  type BenchmarkDataset,
  type MetricKey,
} from "./storyData";

type MetricComparisonSectionProps = {
  datasets?: BenchmarkDataset[];
  initialDatasetId?: BenchmarkDataset["id"];
  className?: string;
};

type CSSVars = CSSProperties & Record<`--${string}`, string | number>;

const METRIC_ORDER: MetricKey[] = [
  "auroc",
  "averagePrecision",
  "brier",
  "ece",
];

const subscribeToHydration = () => () => {};
const getHydratedSnapshot = () => true;
const getServerHydratedSnapshot = () => false;

function formatEstimate(value: number) {
  return value.toFixed(6);
}

function comparisonSummary(
  dataset: BenchmarkDataset,
  metricKey: MetricKey,
) {
  const [resnet, transformer] = dataset.models;
  const meta = METRIC_META[metricKey];
  const raw = resnet.metrics[metricKey].value - transformer.metrics[metricKey].value;
  const resnetLeads = meta.direction === "higher" ? raw > 0 : raw < 0;

  return {
    leader: resnetLeads ? resnet.name : transformer.name,
    difference: Math.abs(raw),
    relation: meta.direction === "higher" ? "higher" : "lower",
  };
}

function plotPosition(value: number, domain: readonly [number, number]) {
  const [minimum, maximum] = domain;
  if (maximum === minimum) return 50;
  return Math.min(100, Math.max(0, ((value - minimum) / (maximum - minimum)) * 100));
}

function ScoreLedger({ dataset }: { dataset: BenchmarkDataset }) {
  return (
    <table className={styles.ledgerTable}>
      <caption className={styles.visuallyHidden}>
        {dataset.tabLabel} comparison of the 1D ResNet and ECG Transformer.
        Values are audited means with sample standard deviation across three
        frozen seeds.
      </caption>
      <colgroup>
        <col className={styles.metricColumn} />
        <col className={styles.plotColumn} />
        <col className={styles.valueColumn} />
        <col className={styles.valueColumn} />
        <col className={styles.deltaColumn} />
      </colgroup>
      <thead>
        <tr>
          <th scope="col">Metric</th>
          <th scope="col">Direct comparison</th>
          <th scope="col">
            <span className={styles.modelHeaderMark} data-model="resnet" aria-hidden="true" />
            1D ResNet
          </th>
          <th scope="col">
            <span
              className={styles.modelHeaderMark}
              data-model="transformer"
              aria-hidden="true"
            />
            ECG Transformer
          </th>
          <th scope="col">Difference</th>
        </tr>
      </thead>
      <tbody>
        {METRIC_ORDER.map((metricKey) => {
          const meta = METRIC_META[metricKey];
          const [resnet, transformer] = dataset.models;
          const resnetEstimate = resnet.metrics[metricKey];
          const transformerEstimate = transformer.metrics[metricKey];
          const comparison = comparisonSummary(dataset, metricKey);
          const resnetPosition = plotPosition(resnetEstimate.value, meta.plotDomain);
          const transformerPosition = plotPosition(
            transformerEstimate.value,
            meta.plotDomain,
          );
          const left = Math.min(resnetPosition, transformerPosition);
          const width = Math.abs(resnetPosition - transformerPosition);

          return (
            <tr key={metricKey}>
              <th scope="row" className={styles.metricName}>
                <strong>{meta.label}</strong>
                <span>{meta.longLabel}</span>
                <small>{meta.direction} is better</small>
              </th>
              <td className={styles.plotCell}>
                <span className={styles.mobileCellLabel}>Direct comparison</span>
                <div
                  className={styles.scorePlot}
                  aria-label={`${resnet.name} ${formatEstimate(
                    resnetEstimate.value,
                  )}; ${transformer.name} ${formatEstimate(
                    transformerEstimate.value,
                  )}. Detail axis from ${meta.plotDomain[0]} to ${meta.plotDomain[1]}.`}
                  style={
                    {
                      "--resnet-position": `${resnetPosition}%`,
                      "--transformer-position": `${transformerPosition}%`,
                      "--difference-left": `${left}%`,
                      "--difference-width": `${width}%`,
                    } as CSSVars
                  }
                >
                  <div className={styles.scorePlotTrack} aria-hidden="true">
                    <i className={styles.differenceLine} />
                    <i className={styles.resnetMarker}>R</i>
                    <i className={styles.transformerMarker}>T</i>
                  </div>
                  <div className={styles.plotAxis} aria-hidden="true">
                    <span>{meta.plotDomain[0].toFixed(2)}</span>
                    <span>detail axis</span>
                    <span>{meta.plotDomain[1].toFixed(2)}</span>
                  </div>
                </div>
              </td>
              <td className={styles.numericCell} data-model="resnet">
                <span className={styles.mobileCellLabel}>1D ResNet</span>
                <strong>{formatEstimate(resnetEstimate.value)}</strong>
                <small>± {formatEstimate(resnetEstimate.spread)}</small>
              </td>
              <td className={styles.numericCell} data-model="transformer">
                <span className={styles.mobileCellLabel}>ECG Transformer</span>
                <strong>{formatEstimate(transformerEstimate.value)}</strong>
                <small>± {formatEstimate(transformerEstimate.spread)}</small>
              </td>
              <td className={styles.deltaCell}>
                <span className={styles.mobileCellLabel}>Difference</span>
                <strong>{formatEstimate(comparison.difference)}</strong>
                <span>
                  {comparison.leader} {comparison.relation} mean
                </span>
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

export function MetricComparisonSection({
  datasets = AUDITED_BENCHMARKS,
  initialDatasetId = "ptb-xl",
  className = "",
}: MetricComparisonSectionProps) {
  const safeInitial = datasets.some((dataset) => dataset.id === initialDatasetId)
    ? initialDatasetId
    : datasets[0]?.id;
  const [selectedId, setSelectedId] = useState(safeInitial);
  const hasHydrated = useSyncExternalStore(
    subscribeToHydration,
    getHydratedSnapshot,
    getServerHydratedSnapshot,
  );
  const selected =
    datasets.find((dataset) => dataset.id === selectedId) ?? datasets[0];
  const tabRefs = useRef<Array<HTMLButtonElement | null>>([]);

  if (!selected) return null;

  const handleTabKeys = (
    event: KeyboardEvent<HTMLButtonElement>,
    currentIndex: number,
  ) => {
    if (
      !["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Home", "End"].includes(
        event.key,
      )
    ) {
      return;
    }

    event.preventDefault();
    const lastIndex = datasets.length - 1;
    const nextIndex =
      event.key === "Home"
        ? 0
        : event.key === "End"
          ? lastIndex
          : event.key === "ArrowRight" || event.key === "ArrowDown"
            ? (currentIndex + 1) % datasets.length
            : (currentIndex - 1 + datasets.length) % datasets.length;
    const nextDataset = datasets[nextIndex];
    if (!nextDataset) return;
    setSelectedId(nextDataset.id);
    tabRefs.current[nextIndex]?.focus();
  };

  return (
    <section
      className={`${styles.storySection} ${styles.metricsSection} ${className}`}
      aria-labelledby="results-ledger-heading"
    >
      <div className={styles.sectionInner}>
        <header className={styles.sectionHeader}>
          <p className={styles.kicker}>Results ledger / 02</p>
          <div>
            <h2 id="results-ledger-heading">
              Discrimination and calibration results
            </h2>
            <p className={styles.headerCopy}>
              Every value below is an audited mean ± sample standard deviation
              across three frozen seeds. Inferential claims are stated separately.
            </p>
          </div>
        </header>

        <aside className={styles.inferenceNote} aria-label="Calibration inference note">
          <strong>Inference note.</strong> Point estimates are descriptive. Every
          paired PTB-XL ECE interval crossed zero, so no internal ECE winner is
          claimed.
        </aside>

        <div className={styles.datasetTabs} role="tablist" aria-label="Dataset results">
          {datasets.map((dataset, index) => {
            const active = dataset.id === selected.id;
            return (
              <button
                key={dataset.id}
                ref={(node) => {
                  tabRefs.current[index] = node;
                }}
                type="button"
                role="tab"
                id={`metric-tab-${dataset.id}`}
                aria-controls={`metric-panel-${dataset.id}`}
                aria-selected={active}
                tabIndex={hasHydrated ? (active ? 0 : -1) : 0}
                onClick={() => setSelectedId(dataset.id)}
                onKeyDown={(event) => handleTabKeys(event, index)}
                className={active ? styles.activeTab : ""}
              >
                <span>{dataset.tabLabel}</span>
                <small>{dataset.eyebrow}</small>
              </button>
            );
          })}
        </div>

        {datasets.map((dataset) => {
          const active = dataset.id === selected.id;
          return (
            <div
              key={dataset.id}
              id={`metric-panel-${dataset.id}`}
              role="tabpanel"
              aria-labelledby={`metric-tab-${dataset.id}`}
              className={styles.metricPanel}
              tabIndex={hasHydrated ? (active ? 0 : -1) : 0}
              hidden={hasHydrated && !active}
            >
              <div className={styles.panelIntro}>
                <h3>{dataset.title}</h3>
                <p>{dataset.description}</p>
              </div>
              <ScoreLedger dataset={dataset} />
              <p className={styles.precisionNote}>
                R = 1D ResNet. T = ECG Transformer. Detail axes are labeled and
                vary by metric; exact values are the primary record.
              </p>
            </div>
          );
        })}
      </div>
    </section>
  );
}
