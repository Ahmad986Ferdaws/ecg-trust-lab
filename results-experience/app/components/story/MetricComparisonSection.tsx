"use client";

import {
  type CSSProperties,
  type KeyboardEvent,
  useId,
  useRef,
  useState,
} from "react";
import styles from "./story.module.css";
import {
  AUDITED_BENCHMARKS,
  METRIC_META,
  type BenchmarkDataset,
  type MetricKey,
} from "./storyData";
import { useStoryMotion } from "./useStoryMotion";

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

const numberFormatter = new Intl.NumberFormat("en-US", {
  minimumFractionDigits: 4,
  maximumFractionDigits: 4,
});

function scoreDifference(
  dataset: BenchmarkDataset,
  metricKey: MetricKey,
) {
  const [resnet, transformer] = dataset.models;
  const meta = METRIC_META[metricKey];
  const raw = resnet.metrics[metricKey].value - transformer.metrics[metricKey].value;
  const magnitude = Math.abs(raw);
  const resnetWins = meta.direction === "higher" ? raw > 0 : raw < 0;

  if (magnitude < 0.0000005) return "Models are tied";
  return `${resnetWins ? resnet.name : transformer.name} ${numberFormatter.format(
    magnitude,
  )} ${meta.direction === "higher" ? "higher" : "lower"}`;
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
  const selected =
    datasets.find((dataset) => dataset.id === selectedId) ?? datasets[0];
  const { ref, isVisible } = useStoryMotion<HTMLElement>();
  const tabId = useId();
  const tabRefs = useRef<Array<HTMLButtonElement | null>>([]);

  if (!selected) return null;

  const handleTabKeys = (
    event: KeyboardEvent<HTMLButtonElement>,
    currentIndex: number,
  ) => {
    if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) {
      return;
    }
    event.preventDefault();
    const lastIndex = datasets.length - 1;
    const nextIndex =
      event.key === "Home"
        ? 0
        : event.key === "End"
          ? lastIndex
          : event.key === "ArrowRight"
            ? (currentIndex + 1) % datasets.length
            : (currentIndex - 1 + datasets.length) % datasets.length;
    const next = datasets[nextIndex];
    if (!next) return;
    setSelectedId(next.id);
    tabRefs.current[nextIndex]?.focus();
  };

  return (
    <section
      ref={ref}
      className={`${styles.storySection} ${styles.metricsSection} ${
        isVisible ? styles.isVisible : ""
      } ${className}`}
      aria-labelledby={`${tabId}-heading`}
    >
      <div className={styles.sectionWash} aria-hidden="true" />
      <div className={styles.sectionInner}>
        <header className={styles.sectionHeader}>
          <div>
            <p className={styles.kicker}>02 · Model evidence</p>
            <h2 id={`${tabId}-heading`}>
              One race. <span>Four ways to measure trust.</span>
            </h2>
          </div>
          <p className={styles.headerCopy}>
            AUROC and average precision measure discrimination. Brier score and
            ECE test whether confidence deserves to be believed.
          </p>
        </header>

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
                id={`${tabId}-${dataset.id}-tab`}
                aria-controls={`${tabId}-${dataset.id}-panel`}
                aria-selected={active}
                tabIndex={active ? 0 : -1}
                onClick={() => setSelectedId(dataset.id)}
                onKeyDown={(event) => handleTabKeys(event, index)}
                className={active ? styles.activeTab : ""}
              >
                <span>{dataset.tabLabel}</span>
                <i aria-hidden="true" />
              </button>
            );
          })}
        </div>

        <div
          key={selected.id}
          id={`${tabId}-${selected.id}-panel`}
          role="tabpanel"
          aria-labelledby={`${tabId}-${selected.id}-tab`}
          className={styles.metricPanel}
          tabIndex={0}
        >
          <div className={styles.panelIntro}>
            <p>{selected.eyebrow}</p>
            <h3>{selected.title}</h3>
            <span>{selected.description}</span>
          </div>

          <div className={styles.modelLegend} aria-label="Compared models">
            {selected.models.map((model) => (
              <div key={model.id} className={styles.legendItem}>
                <i className={styles[model.accent]} aria-hidden="true" />
                <span>
                  <strong>{model.name}</strong>
                  <small>{model.descriptor}</small>
                </span>
              </div>
            ))}
          </div>

          <div className={styles.metricGrid}>
            {METRIC_ORDER.map((metricKey, index) => {
              const meta = METRIC_META[metricKey];
              return (
                <article
                  className={styles.metricCard}
                  key={metricKey}
                  style={{ "--card-index": index } as CSSVars}
                >
                  <div className={styles.metricCardTop}>
                    <div>
                      <span>{meta.longLabel}</span>
                      <h4>{meta.label}</h4>
                    </div>
                    <p className={styles.directionTag}>
                      {meta.direction} is better
                    </p>
                  </div>

                  <div
                    className={styles.scoreRail}
                    aria-label={`${meta.longLabel}, ${scoreDifference(
                      selected,
                      metricKey,
                    )}`}
                  >
                    <div className={styles.railTrack} aria-hidden="true">
                      {selected.models.map((model, modelIndex) => (
                        <i
                          key={model.id}
                          className={`${styles.scoreMarker} ${
                            styles[model.accent]
                          }`}
                          style={
                            {
                              "--score": `${model.metrics[metricKey].value * 100}%`,
                              "--model-index": modelIndex,
                            } as CSSVars
                          }
                        />
                      ))}
                    </div>
                    <div className={styles.scoreScale} aria-hidden="true">
                      <span>0</span>
                      <span>1</span>
                    </div>
                  </div>

                  <dl className={styles.scoreList}>
                    {selected.models.map((model) => (
                      <div key={model.id}>
                        <dt>
                          <i className={styles[model.accent]} aria-hidden="true" />
                          {model.name}
                        </dt>
                        <dd title={`${model.metrics[metricKey].value} ± ${model.metrics[metricKey].spread}`}>
                          {numberFormatter.format(model.metrics[metricKey].value)}
                          <small>
                            ± {numberFormatter.format(model.metrics[metricKey].spread)}
                          </small>
                        </dd>
                      </div>
                    ))}
                  </dl>

                  <p className={styles.delta}>{scoreDifference(selected, metricKey)}</p>
                </article>
              );
            })}
          </div>

          <p className={styles.precisionNote}>
            Values are audited means with the reported ± spread. Rails use a
            common 0–1 scale; they do not exaggerate small differences.
          </p>
        </div>
      </div>
    </section>
  );
}
