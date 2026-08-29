"use client";

import { useState } from "react";

import styles from "./FailureLabSection.module.css";

const LEADS = [
  "I",
  "II",
  "III",
  "aVR",
  "aVL",
  "aVF",
  "V1",
  "V2",
  "V3",
  "V4",
  "V5",
  "V6",
] as const;

const SCENARIOS = [
  {
    id: "baseline-wander",
    label: "Baseline wander",
    short: "Low-frequency drift",
    checkpoint: "Signal-quality gate",
    challenge:
      "Whether slow baseline displacement is identified before uncertainty or prediction release is considered.",
    affected: "All 12 leads",
  },
  {
    id: "mains-interference",
    label: "Mains interference",
    short: "50/60 Hz-like contamination",
    checkpoint: "Signal-quality gate",
    challenge:
      "Whether periodic electrical contamination is surfaced as a quality finding rather than mistaken for physiology.",
    affected: "All 12 leads",
  },
  {
    id: "deterministic-noise",
    label: "Broadband noise",
    short: "Seeded high-frequency mixture",
    checkpoint: "Quality → uncertainty",
    challenge:
      "Whether a degraded trace is stopped by quality checks or later withheld by the uncertainty policy.",
    affected: "All 12 leads",
  },
  {
    id: "gain-change",
    label: "Gain change",
    short: "Amplitude rescaling",
    checkpoint: "Distribution support",
    challenge:
      "Whether a plausible-looking but shifted amplitude scale remains inside the frozen reference envelope.",
    affected: "All 12 leads",
  },
  {
    id: "contiguous-mask",
    label: "Masked segment",
    short: "Contiguous samples replaced",
    checkpoint: "Signal-quality gate",
    challenge:
      "Whether a missing interval is detected before downstream evidence can authorize prediction disclosure.",
    affected: "Selected precordial leads",
  },
  {
    id: "lead-dropout",
    label: "Lead dropout",
    short: "One or more flat leads",
    checkpoint: "Input quality → reacquire",
    challenge:
      "Whether lost channels produce an explicit reacquisition path instead of a silent partial-input prediction.",
    affected: "Severity-selected leads",
  },
  {
    id: "lead-order",
    label: "Lead order swap",
    short: "Canonical labels, permuted signals",
    checkpoint: "Quality → distribution",
    challenge:
      "Whether physiologic lead relationships or frozen distribution evidence flag signals whose metadata still claims canonical order.",
    affected: "All 12 lead positions",
  },
  {
    id: "bounded-clipping",
    label: "Bounded clipping",
    short: "Peaks limited at fixed bounds",
    checkpoint: "Quality → distribution",
    challenge:
      "Whether truncated morphology is caught by the quality policy or rejected as outside supported distribution.",
    affected: "All 12 leads",
  },
  {
    id: "time-shift",
    label: "Global time shift",
    short: "All leads displaced with zero padding",
    checkpoint: "Distribution → uncertainty",
    challenge:
      "Whether a displaced acquisition window remains supported or triggers abstention downstream.",
    affected: "All 12 leads",
  },
] as const;

type ScenarioId = (typeof SCENARIOS)[number]["id"];

const SEVERITY_LABELS = ["", "Mild", "Moderate", "Strong", "Severe"] as const;

const DECISION_STATES = [
  {
    state: "INVALID_INPUT",
    meaning:
      "The release, input contract, or required quality evidence is invalid or unavailable. Stop before prediction.",
  },
  {
    state: "REACQUIRE",
    meaning:
      "Signal quality is limited enough that a new recording is requested. No prediction is released.",
  },
  {
    state: "UNSUPPORTED_INPUT",
    meaning:
      "The input falls outside supported distribution evidence, or that check is unavailable. Stop before uncertainty claims.",
  },
  {
    state: "ABSTAIN",
    meaning:
      "Earlier gates passed, but frozen uncertainty or conformal evidence does not support disclosure.",
  },
  {
    state: "PREDICTION_ALLOWED",
    meaning:
      "Every required trust gate passed, so research probabilities may be shown. This is not a diagnosis or safety claim.",
  },
] as const;

const SVG_WIDTH = 840;
const SVG_HEIGHT = 528;
const PLOT_LEFT = 74;
const PLOT_RIGHT = 820;
const ROW_HEIGHT = 42;
// The SVG preview samples at ~155 Hz over 6.2 seconds, high enough to render
// the illustrative 50/60 Hz and >20 Hz components without display aliasing.
const SAMPLE_COUNT = 960;
const VERTICAL_RULES = Array.from({ length: 24 }, (_, index) => PLOT_LEFT + index * 32);
const DROPOUT_INDEXES: ReadonlyArray<ReadonlySet<number>> = [
  new Set(),
  new Set([2]),
  new Set([2, 7]),
  new Set([2, 7, 10]),
  new Set([2, 7, 10, 5]),
];

function gaussian(value: number, center: number, width: number): number {
  const distance = (value - center) / width;
  return Math.exp(-0.5 * distance * distance);
}

function baseEcg(time: number, profileIndex: number): number {
  const phase = ((time % 1) + 1) % 1;
  const polarity = profileIndex === 3 || profileIndex === 6 ? -1 : 1;
  const scale = 0.74 + (profileIndex % 5) * 0.08;
  const p = 0.12 * gaussian(phase, 0.18, 0.035);
  const q = -0.2 * gaussian(phase, 0.355, 0.012);
  const r = 1.08 * gaussian(phase, 0.38, 0.014);
  const s = -0.32 * gaussian(phase, 0.415, 0.018);
  const t = 0.3 * gaussian(phase, 0.66, 0.07);
  return polarity * scale * (p + q + r + s + t);
}

function dropoutIndexes(severity: number): ReadonlySet<number> {
  return DROPOUT_INDEXES[severity] ?? DROPOUT_INDEXES[1];
}

function scenarioValue(
  scenarioId: ScenarioId,
  severity: number,
  time: number,
  leadIndex: number,
): number {
  const chestLead = leadIndex >= 6;
  const sourceIndex =
    scenarioId === "lead-order" ? (leadIndex + severity) % LEADS.length : leadIndex;
  const shiftedTime = scenarioId === "time-shift" ? time + severity * 0.055 : time;
  let value = baseEcg(shiftedTime, sourceIndex);

  if (scenarioId === "baseline-wander") {
    value += severity * 0.1 * Math.sin(time * 0.72 + leadIndex * 0.08);
  } else if (scenarioId === "mains-interference") {
    const mainsFrequencyHz = leadIndex % 2 === 0 ? 50 : 60;
    value +=
      severity *
      0.035 *
      Math.sin(2 * Math.PI * mainsFrequencyHz * time + leadIndex * 0.31);
  } else if (scenarioId === "deterministic-noise") {
    const mixture =
      Math.sin(2 * Math.PI * 23 * time + leadIndex * 0.7) +
      0.62 * Math.sin(2 * Math.PI * 31 * time + leadIndex * 0.19) +
      0.38 * Math.sin(2 * Math.PI * 43 * time + leadIndex * 0.43);
    value += severity * 0.045 * mixture;
  } else if (scenarioId === "gain-change") {
    value *= 1 + severity * 0.24;
  } else if (
    scenarioId === "contiguous-mask" &&
    chestLead &&
    time >= 2.35 &&
    time <= 2.35 + severity * 0.28
  ) {
    value = 0;
  } else if (scenarioId === "lead-dropout" && dropoutIndexes(severity).has(leadIndex)) {
    value = 0;
  } else if (scenarioId === "bounded-clipping") {
    const bound = 0.86 - severity * 0.12;
    value = Math.max(-bound, Math.min(bound, value));
  }

  return Math.max(-1.55, Math.min(1.55, value));
}

function buildLeadPath(scenarioId: ScenarioId, severity: number, leadIndex: number): string {
  const centerY = 18 + leadIndex * ROW_HEIGHT + ROW_HEIGHT / 2;
  const plotWidth = PLOT_RIGHT - PLOT_LEFT;
  const commands: string[] = [];

  for (let sample = 0; sample < SAMPLE_COUNT; sample += 1) {
    const progress = sample / (SAMPLE_COUNT - 1);
    const x = PLOT_LEFT + progress * plotWidth;
    const time = progress * 6.2;
    const y = centerY - scenarioValue(scenarioId, severity, time, leadIndex) * 13.5;
    commands.push(`${sample === 0 ? "M" : "L"}${x.toFixed(2)} ${y.toFixed(2)}`);
  }

  return commands.join(" ");
}

function isAffectedLead(scenarioId: ScenarioId, severity: number, leadIndex: number): boolean {
  if (scenarioId === "lead-dropout") {
    return dropoutIndexes(severity).has(leadIndex);
  }
  if (scenarioId === "contiguous-mask") {
    return leadIndex >= 6;
  }
  return true;
}

export function FailureLabSection() {
  const [scenarioId, setScenarioId] = useState<ScenarioId>("baseline-wander");
  const [severity, setSeverity] = useState(2);
  const selected = SCENARIOS.find((scenario) => scenario.id === scenarioId) ?? SCENARIOS[0];
  const severityLabel = SEVERITY_LABELS[severity];

  return (
    <section className={styles.section} id="failure-lab" aria-labelledby="failure-lab-title">
      <div className={styles.inner}>
        <header className={styles.header}>
          <p className={styles.kicker}>Failure Lab / controlled sensitivity</p>
          <div>
            <h2 id="failure-lab-title">Challenge the trust gates before trusting a score.</h2>
            <p>
              Explore deterministic signal corruptions and the checkpoint each scenario is
              designed to test. The lab asks where the pipeline should stop; it does not
              predict what any model will do.
            </p>
          </div>
        </header>

        <aside className={styles.previewBoundary} aria-label="Illustrative preview boundary">
          <strong>Illustrative synthetic scenario preview</strong>
          <p>
            The browser changes a generated teaching waveform only. It runs no classifier and
            shows no measured quality, OOD, calibration, conformal, or decision output. The
            tracked Python <code>FailureLabRunner</code> reruns the real complete trust pipeline
            for controlled experiments.
          </p>
        </aside>

        <div className={styles.workbench}>
          <form
            className={styles.controls}
            onReset={() => {
              setScenarioId("baseline-wander");
              setSeverity(2);
            }}
          >
            <fieldset>
              <legend>Choose a synthetic challenge</legend>
              <div className={styles.scenarioChoices}>
                {SCENARIOS.map((scenario, index) => (
                  <label key={scenario.id} className={styles.scenarioChoice}>
                    <input
                      type="radio"
                      name="failure-scenario"
                      value={scenario.id}
                      checked={scenarioId === scenario.id}
                      onChange={() => setScenarioId(scenario.id)}
                    />
                    <span aria-hidden="true">{String(index + 1).padStart(2, "0")}</span>
                    <span>
                      <strong>{scenario.label}</strong>
                      <small>{scenario.short}</small>
                    </span>
                  </label>
                ))}
              </div>
            </fieldset>

            <div className={styles.severityControl}>
              <div>
                <label htmlFor="failure-severity">Illustrative severity</label>
                <output htmlFor="failure-severity">
                  {severity}/4 · {severityLabel}
                </output>
              </div>
              <input
                id="failure-severity"
                type="range"
                min="1"
                max="4"
                step="1"
                value={severity}
                aria-valuetext={`${severityLabel}, level ${severity} of 4`}
                onChange={(event) => setSeverity(Number(event.currentTarget.value))}
              />
              <div className={styles.severityScale} aria-hidden="true">
                <span>Mild</span>
                <span>Severe</span>
              </div>
            </div>

            <button className={styles.resetButton} type="reset">
              Reset synthetic preview
            </button>
          </form>

          <figure className={styles.previewFigure}>
            <div className={styles.previewHeading}>
              <div>
                <span>Synthetic strip / {severityLabel.toLowerCase()}</span>
                <strong>{selected.label}</strong>
              </div>
              <span>Not model output</span>
            </div>

            <div
              className={styles.svgScroller}
              role="region"
              aria-label="Scrollable synthetic 12-lead ECG preview"
            >
              <svg
                className={styles.waveform}
                viewBox={`0 0 ${SVG_WIDTH} ${SVG_HEIGHT}`}
                role="img"
                aria-labelledby="failure-waveform-title failure-waveform-description"
              >
                <title id="failure-waveform-title">
                  {`Synthetic 12-lead preview of ${selected.label.toLowerCase()}`}
                </title>
                <desc id="failure-waveform-description">
                  Deterministic teaching waveform at {severityLabel.toLowerCase()} illustrative
                  severity. It is not a patient ECG and contains no model result.
                </desc>

                <g className={styles.grid} aria-hidden="true">
                  {VERTICAL_RULES.map((x) => (
                    <line key={x} x1={x} x2={x} y1="12" y2={SVG_HEIGHT - 8} />
                  ))}
                  {LEADS.map((lead, index) => (
                    <line
                      key={lead}
                      x1={PLOT_LEFT}
                      x2={PLOT_RIGHT}
                      y1={18 + (index + 1) * ROW_HEIGHT}
                      y2={18 + (index + 1) * ROW_HEIGHT}
                    />
                  ))}
                </g>

                {scenarioId === "contiguous-mask" ? (
                  <rect
                    className={styles.maskBand}
                    x={PLOT_LEFT + ((2.35 / 6.2) * (PLOT_RIGHT - PLOT_LEFT))}
                    y={18 + 6 * ROW_HEIGHT}
                    width={(severity * 0.28 * (PLOT_RIGHT - PLOT_LEFT)) / 6.2}
                    height={6 * ROW_HEIGHT}
                    aria-hidden="true"
                  />
                ) : null}

                {LEADS.map((lead, leadIndex) => (
                  <g key={lead} data-lead={lead}>
                    <text className={styles.leadLabel} x="18" y={44 + leadIndex * ROW_HEIGHT}>
                      {lead}
                    </text>
                    <path
                      className={
                        isAffectedLead(scenarioId, severity, leadIndex)
                          ? styles.affectedTrace
                          : styles.trace
                      }
                      d={buildLeadPath(scenarioId, severity, leadIndex)}
                      vectorEffect="non-scaling-stroke"
                    />
                  </g>
                ))}
              </svg>
            </div>

            <figcaption>
              A deterministic teaching trace generated in the page. No patient signal, record
              identifier, probability, or clinical claim is present.
            </figcaption>
          </figure>
        </div>

        <section
          className={styles.checkpoint}
          aria-labelledby="checkpoint-title"
          aria-live="polite"
        >
          <p className={styles.kicker}>Selected experiment</p>
          <div>
            <h3 id="checkpoint-title">
              Designed to challenge: {selected.checkpoint}
            </h3>
            <p>{selected.challenge}</p>
            <dl>
              <div>
                <dt>Intervention</dt>
                <dd>{selected.short}</dd>
              </div>
              <div>
                <dt>Illustrative severity</dt>
                <dd>{severityLabel} · {severity}/4</dd>
              </div>
              <div>
                <dt>Preview scope</dt>
                <dd>{selected.affected}</dd>
              </div>
              <div>
                <dt>Displayed result</dt>
                <dd>None—the runner must measure it</dd>
              </div>
            </dl>
          </div>
        </section>

        <section className={styles.vocabulary} aria-labelledby="decision-vocabulary-title">
          <header>
            <p className={styles.kicker}>Frozen safety order</p>
            <h3 id="decision-vocabulary-title">The five states are stops, not confidence badges.</h3>
            <p>
              Earlier blocking evidence wins. A later gate can never override an invalid input,
              a reacquisition request, or unsupported distribution evidence.
            </p>
          </header>
          <dl className={styles.stateLedger}>
            {DECISION_STATES.map((item, index) => (
              <div key={item.state}>
                <dt>
                  <span>{String(index + 1).padStart(2, "0")}</span>
                  <code>{item.state}</code>
                </dt>
                <dd>{item.meaning}</dd>
              </div>
            ))}
          </dl>
        </section>

        <section className={styles.scenarioRegister} aria-labelledby="scenario-register-title">
          <header>
            <p className={styles.kicker}>Scenario register</p>
            <h3 id="scenario-register-title">Every browser option remains a hypothesis.</h3>
          </header>
          <div
            className={styles.tableScroller}
            role="region"
            aria-label="Scrollable Failure Lab scenario register"
          >
            <table>
              <thead>
                <tr>
                  <th scope="col">Scenario</th>
                  <th scope="col">Designed checkpoint</th>
                  <th scope="col">What the tracked run asks</th>
                </tr>
              </thead>
              <tbody>
                {SCENARIOS.map((scenario) => (
                  <tr key={scenario.id}>
                    <th scope="row">{scenario.label}</th>
                    <td>{scenario.checkpoint}</td>
                    <td>{scenario.challenge}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        <footer className={styles.runnerNote}>
          <strong>Browser preview ≠ pipeline result</strong>
          <p>
            Reproducible evidence comes from a versioned scenario passed to the tracked Python
            <code>FailureLabRunner</code>, which compares the untouched baseline with the stressed
            signal through release integrity, input validation, quality, distribution, model,
            calibration, uncertainty, conformal, and disclosure policy checks.
          </p>
        </footer>
      </div>
    </section>
  );
}
