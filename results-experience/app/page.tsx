import { ResultsUniverse } from "./components/ResultsUniverse";
import {
  MetricComparisonSection,
  ResearchSafetySection,
  TransportCohortSection,
} from "./components/story";

const heroFacts = [
  { value: "0.922", label: "PTB-XL macro AUROC", detail: "sealed internal test" },
  { value: "0.931", label: "SPH macro AUROC", detail: "frozen transport" },
  { value: "15,193", label: "external patients", detail: "primary cohort" },
  { value: "494", label: "checks passing", detail: "verified release" },
] as const;

export default function Home() {
  return (
    <main className="experience-shell">
      <div className="ambient-grid" aria-hidden="true" />
      <div className="film-grain" aria-hidden="true" />

      <nav className="site-nav" aria-label="Primary navigation">
        <a className="wordmark" href="#top" aria-label="ECG Trust Lab home">
          <span className="wordmark-pulse" aria-hidden="true" />
          ECG / TRUST LAB
        </a>
        <div className="nav-status">
          <span className="status-dot" aria-hidden="true" />
          Audited research build
        </div>
        <a className="nav-link" href="#evidence">
          Enter the evidence <span aria-hidden="true">↘</span>
        </a>
      </nav>

      <section className="hero" id="top" aria-labelledby="hero-title">
        <div className="hero-scene" aria-hidden="true">
          <ResultsUniverse />
        </div>

        <div className="hero-copy">
          <p className="eyebrow">
            <span>12-lead ECG intelligence</span>
            <span className="eyebrow-line" aria-hidden="true" />
            <span>Two models. Two populations.</span>
          </p>
          <h1 id="hero-title">
            <span>Trust,</span>
            <span>rendered in</span>
            <span className="text-glow">three dimensions.</span>
          </h1>
          <p className="hero-deck">
            A living map of what our models learned, where they travelled, and
            exactly how their confidence held up under pressure.
          </p>
          <div className="hero-actions">
            <a className="primary-action" href="#evidence">
              <span>Explore the signal</span>
              <span className="action-orbit" aria-hidden="true">↗</span>
            </a>
            <p>
              ResNet leads the sealed test and the independent transport study.
              <strong> Research only.</strong>
            </p>
          </div>
        </div>

        <div className="hero-index" aria-label="Experience chapter">
          <span>01</span>
          <span>Signal universe</span>
        </div>

        <div className="hero-facts" aria-label="Headline verified results">
          {heroFacts.map((fact) => (
            <article className="hero-fact" key={fact.label}>
              <strong>{fact.value}</strong>
              <span>{fact.label}</span>
              <small>{fact.detail}</small>
            </article>
          ))}
        </div>

        <a className="scroll-cue" href="#evidence" aria-label="Scroll to the evidence">
          <span>Scroll to enter</span>
          <i aria-hidden="true" />
        </a>
      </section>

      <section className="manifesto" aria-labelledby="manifesto-title">
        <div className="manifesto-orbit" aria-hidden="true">
          <i />
          <i />
          <i />
        </div>
        <p className="chapter-label">The governing idea / 01.5</p>
        <h2 id="manifesto-title">
          The bar was never one score.
          <span>It was whether the evidence could survive leaving home.</span>
        </h2>
        <div className="study-geometry" aria-label="Study design at a glance">
          <article>
            <strong>12</strong>
            <span>simultaneous ECG leads</span>
          </article>
          <article>
            <strong>5</strong>
            <span>diagnostic superclasses</span>
          </article>
          <article>
            <strong>3 × 2</strong>
            <span>frozen seeds × architectures</span>
          </article>
          <article>
            <strong>0</strong>
            <span>SPH tuning decisions</span>
          </article>
        </div>
      </section>

      <div id="evidence">
        <MetricComparisonSection />
      </div>

      <aside className="signal-marquee" aria-label="Five classified superclasses">
        <div aria-hidden="true">
          <span>NORM</span><i>◆</i><span>MI</span><i>◆</i><span>STTC</span><i>◆</i>
          <span>CD</span><i>◆</i><span>HYP</span><i>◆</i><span>NORM</span><i>◆</i>
          <span>MI</span><i>◆</i><span>STTC</span><i>◆</i><span>CD</span><i>◆</i><span>HYP</span>
        </div>
      </aside>

      <TransportCohortSection />

      <section className="verdict" aria-labelledby="verdict-title">
        <div className="verdict-number" aria-hidden="true">02</div>
        <div className="verdict-copy">
          <p className="chapter-label">What the experiment says / 03.5</p>
          <h2 id="verdict-title">The simpler architecture won twice.</h2>
          <p>
            The 1D ResNet ranked cases better than the transformer in every
            frozen seed across the primary and sensitivity cohorts. Its
            advantage was not enormous. It was consistent—which is the more
            interesting result.
          </p>
        </div>
        <dl className="verdict-deltas">
          <div>
            <dt>PTB-XL AUROC lead</dt>
            <dd>+0.0245</dd>
          </div>
          <div>
            <dt>SPH primary AUROC lead</dt>
            <dd>+0.0068</dd>
          </div>
          <div>
            <dt>SPH primary AP lead</dt>
            <dd>+0.0411</dd>
          </div>
        </dl>
      </section>

      <ResearchSafetySection />

      <footer className="site-footer">
        <div>
          <span className="wordmark-pulse" aria-hidden="true" />
          <strong>ECG / TRUST LAB</strong>
        </div>
        <p>
          Audited retrospective research experience. Not a medical device. Not
          for diagnosis, treatment, or emergency decisions.
        </p>
        <a href="#top">Return to signal ↑</a>
      </footer>
    </main>
  );
}
