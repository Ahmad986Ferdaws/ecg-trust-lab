import { MotionControl } from "./components/ExperienceMotion";
import { ResultsUniverse } from "./components/ResultsUniverse";
import { FailureLabSection } from "./components/story/FailureLabSection";
import { MetricComparisonSection } from "./components/story/MetricComparisonSection";
import { ResearchSafetySection } from "./components/story/ResearchSafetySection";
import { SourceSupportSection } from "./components/story/SourceSupportSection";
import { TransportCohortSection } from "./components/story/TransportCohortSection";
import { COHORTS, SOURCE_POPULATIONS } from "../lib/results";

const ptb = COHORTS.ptbxl_fold10;
const sph = COHORTS.sph_primary;

const headlineFacts = [
  {
    value: ptb.results.resnet1d.auroc.mean.toFixed(4),
    label: "PTB-XL macro-AUROC",
    detail: `${ptb.count.records.toLocaleString("en-US")} sealed ECGs`,
  },
  {
    value: sph.results.resnet1d.auroc.mean.toFixed(4),
    label: "SPH macro-AUROC",
    detail: "frozen external transport",
  },
  {
    value: sph.count.patients.toLocaleString("en-US"),
    label: "SPH primary patients",
    detail: `${sph.count.records.toLocaleString("en-US")} ECGs`,
  },
] as const;

const methodSteps = [
  {
    index: "01",
    title: "Build on PTB-XL",
    body: `${SOURCE_POPULATIONS.ptbxlCanonicalManifest.records.toLocaleString("en-US")} provenance-checked ECGs formed the labeled project manifest. Patients stayed isolated by fold.`,
  },
  {
    index: "02",
    title: "Freeze every decision",
    body: "Architecture, preprocessing, calibration, thresholds, and confidence gates were fixed before the final evaluations.",
  },
  {
    index: "03",
    title: "Open the sealed test",
    body: `${ptb.count.records.toLocaleString("en-US")} fold-10 ECGs compared the ResNet and transformer across the same three seeds.`,
  },
  {
    index: "04",
    title: "Transport without adaptation",
    body: `${sph.count.records.toLocaleString("en-US")} mapped SPH ECGs were scored once—without target-domain training, selection, preprocessing changes, or recalibration.`,
  },
] as const;

const verdictDeltas = [
  {
    label: "PTB-XL AUROC lead",
    value: `+${(
      ptb.results.resnet1d.auroc.mean -
      ptb.results.ecg_transformer.auroc.mean
    ).toFixed(4)}`,
  },
  {
    label: "SPH primary AUROC lead",
    value: `+${(
      sph.results.resnet1d.auroc.mean -
      sph.results.ecg_transformer.auroc.mean
    ).toFixed(4)}`,
  },
  {
    label: "SPH primary AP lead",
    value: `+${(
      sph.results.resnet1d.averagePrecision.mean -
      sph.results.ecg_transformer.averagePrecision.mean
    ).toFixed(4)}`,
  },
] as const;

export default function Home() {
  return (
    <>
      <nav className="site-nav" aria-label="Primary navigation">
        <a className="wordmark" href="#top" aria-label="ECG Trust Lab home">
          <span>ECG Trust Lab</span>
          <small>Results notebook / 2026</small>
        </a>
        <div className="nav-context" aria-label="Study path">
          <span>PTB-XL</span>
          <i aria-hidden="true">→</i>
          <span>SPH</span>
          <i aria-hidden="true">→</i>
          <span>Support gate</span>
        </div>
        <div className="nav-actions">
          <a href="#method">Method</a>
          <a href="#evidence">Results</a>
          <a href="#source-support">Support gate</a>
          <a href="#failure-lab">Failure lab</a>
          <MotionControl />
        </div>
      </nav>

      <main className="experience-shell">

      <section className="hero" id="top" aria-labelledby="hero-title">
        <div className="hero-copy">
          <p className="eyebrow">
            <span>Retrospective research</span>
            <span>Five diagnostic superclasses</span>
          </p>
          <h1 id="hero-title">
            A ResNet kept its ranking lead <em>after transport.</em>
          </h1>
          <p className="hero-deck">
            Three fixed seeds per architecture. One sealed PTB-XL test. One
            no-adaptation SPH stress test. The discrimination ordering stayed
            unchanged when the data source changed.
          </p>

          <dl className="hero-facts" aria-label="Headline audited results">
            {headlineFacts.map((fact) => (
              <div key={fact.label}>
                <dt>{fact.label}</dt>
                <dd>{fact.value}</dd>
                <small>{fact.detail}</small>
              </div>
            ))}
          </dl>

          <p className="hero-boundary">
            External transport is evidence of robustness—not clinical
            validation. This is a research system, not a medical device.
          </p>
        </div>

        <figure className="hero-figure">
          <div className="hero-scene" aria-hidden="true">
            <ResultsUniverse />
          </div>
          <figcaption>
            <span>Figure 01</span>
            <p>
              Schematic 12-lead evidence instrument. The waveform geometry is
              illustrative; every reported metric is drawn from the audited
              evaluation artifacts.
            </p>
          </figcaption>
        </figure>
      </section>

      <section className="study-rail" id="method" aria-labelledby="method-title">
        <header>
          <p className="section-label">Study design</p>
          <h2 id="method-title">Study sequence from training to transport</h2>
          <p>
            The scientific claim depends on the order of operations. Nothing
            from SPH was allowed to alter the models or their decision policy.
          </p>
        </header>
        <ol>
          {methodSteps.map((step) => (
            <li key={step.index}>
              <span>{step.index}</span>
              <div>
                <h3>{step.title}</h3>
                <p>{step.body}</p>
              </div>
            </li>
          ))}
        </ol>
        <dl className="method-ledger" aria-label="Study dimensions">
          <div>
            <dt>Signal</dt>
            <dd>12 leads × 10 seconds</dd>
          </div>
          <div>
            <dt>Labels</dt>
            <dd>5 superclasses</dd>
          </div>
          <div>
            <dt>Models</dt>
            <dd>2 architectures × 3 seeds</dd>
          </div>
          <div>
            <dt>SPH adaptation</dt>
            <dd>None</dd>
          </div>
        </dl>
      </section>

      <div id="evidence">
        <MetricComparisonSection />
      </div>

      <TransportCohortSection />

      <SourceSupportSection />

      <section className="verdict" aria-labelledby="verdict-title">
        <header>
          <p className="section-label">Primary result</p>
          <h2 id="verdict-title">
            The ResNet led on discrimination. Calibration was less conclusive.
          </h2>
          <p>
            AUROC and average-precision advantages remained directionally
            consistent across the three fixed seeds. Transported Brier differences
            were seed-dependent, and no internal ECE winner is claimed.
          </p>
        </header>

        <dl className="verdict-deltas">
          {verdictDeltas.map((delta) => (
            <div key={delta.label}>
              <dt>{delta.label}</dt>
              <dd>{delta.value}</dd>
            </div>
          ))}
        </dl>

        <aside className="verdict-note">
          <span>Interpretation note</span>
          <p>
            AUROC measures ranking, not diagnostic accuracy. A higher transported
            score does not establish safety, clinical utility, or deployment
            readiness.
          </p>
        </aside>
      </section>

      <FailureLabSection />

      <ResearchSafetySection />

      </main>

      <footer className="site-footer">
        <div>
          <strong>ECG Trust Lab</strong>
          <span>Audited model evidence / 2026</span>
        </div>
        <p>
          Retrospective research only. Not for diagnosis, treatment, triage, or
          emergency decisions.
        </p>
        <a href="#top">Back to the beginning ↑</a>
      </footer>
    </>
  );
}
