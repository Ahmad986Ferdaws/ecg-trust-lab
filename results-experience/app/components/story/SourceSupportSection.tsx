import { SOURCE_SUPPORT_COMPLETION } from "../../../lib/results";
import styles from "./story.module.css";

const result = SOURCE_SUPPORT_COMPLETION;
const roles = Object.values(result.roles);

function percent(value: number, digits = 2) {
  return `${(value * 100).toFixed(digits)}%`;
}

export function SourceSupportSection() {
  return (
    <section
      id="source-support"
      className={`${styles.storySection} ${styles.sourceSupportSection}`}
      aria-labelledby="source-support-heading"
    >
      <div className={styles.sectionInner}>
        <div className={styles.supportStatus}>
          <span>One-shot completion / 2026-08-29</span>
          <strong>Source-support target missed</strong>
          <span>No tuning · no retry</span>
        </div>

        <header className={styles.supportHeader}>
          <p className={styles.kicker}>Trust Sentinel / evidence gate</p>
          <div>
            <h2 id="source-support-heading">
              The experiment completed. The gate did not pass.
            </h2>
            <p>
              A frozen distance rule tested whether known-source ECGs still
              looked familiar to the ResNet. The point estimate narrowly missed
              the target, and its uncertainty crossed farther beyond it—so the
              unfavorable result was preserved exactly as observed.
            </p>
          </div>
        </header>

        <div className={styles.supportHero}>
          <div className={styles.coverageHeadline}>
            <span>Observed source support</span>
            <strong>{percent(result.validation.supportCoverage)}</strong>
            <p>
              {result.validation.retained} of {result.roles.sourceValidation.records}
              {" "}known-source ECGs retained
            </p>
          </div>

          <div className={styles.gateDecision}>
            <span>Frozen decision</span>
            <strong>Not eligible</strong>
            <p>
              The preregistered rule required the one-sided 95% false-rejection
              upper bound to stay at or below 5.00%.
            </p>
          </div>

          <div
            className={styles.supportPlot}
            role="img"
            aria-label={`Observed source-support coverage ${percent(
              result.validation.supportCoverage,
            )}; target coverage 95 percent.`}
          >
            <div className={styles.supportPlotLabels} aria-hidden="true">
              <span>0%</span>
              <span>Observed {percent(result.validation.supportCoverage)}</span>
              <span>100%</span>
            </div>
            <div className={styles.supportTrack} aria-hidden="true">
              <span className={styles.observedSupport} />
              <span className={styles.targetMarker}>
                <i>95% target</i>
              </span>
            </div>
          </div>
        </div>

        <div className={styles.uncertaintyLedger}>
          <div className={styles.uncertaintyIntro}>
            <p className={styles.ledgerLabel}>Why the gate stayed closed</p>
            <h3>Confidence matters more than a near miss.</h3>
            <p>
              The observed false-rejection rate was 5.38%. Across 10,000
              patient-cluster bootstrap samples, the one-sided upper bound was
              7.30%—above the frozen 5.00% ceiling.
            </p>
          </div>

          <dl className={styles.gateNumbers}>
            <div>
              <dt>Observed false rejection</dt>
              <dd className={styles.metricValue}>
                {percent(result.validation.recordFalseRejection)}
              </dd>
              <dd className={styles.metricContext}>
                {result.validation.rejected} / {result.roles.sourceValidation.records} ECGs
              </dd>
            </div>
            <div className={styles.upperBoundMetric}>
              <dt>One-sided 95% upper bound</dt>
              <dd className={styles.metricValue}>
                {percent(result.validation.oneSidedUpper95)}
              </dd>
              <dd className={styles.metricContext}>patient-cluster bootstrap</dd>
            </div>
            <div className={styles.targetMetric}>
              <dt>Frozen maximum</dt>
              <dd className={styles.metricValue}>
                {percent(result.validation.targetMaximum)}
              </dd>
              <dd className={styles.metricContext}>required for eligibility</dd>
            </div>
          </dl>
        </div>

        <ol className={styles.supportRoles} aria-label="Patient-disjoint experiment roles">
          {roles.map((role) => (
            <li key={role.code}>
              <span className={styles.roleCode}>{role.code}</span>
              <div>
                <p>{role.label}</p>
                <h3>{role.purpose}</h3>
                <dl>
                  <div>
                    <dt>ECGs</dt>
                    <dd>{role.records.toLocaleString("en-US")}</dd>
                  </div>
                  <div>
                    <dt>Patients</dt>
                    <dd>{role.patients.toLocaleString("en-US")}</dd>
                  </div>
                </dl>
                <small>{role.folds}</small>
              </div>
            </li>
          ))}
        </ol>

        <aside className={styles.supportBoundary}>
          <div>
            <span>OOD-positive evaluation</span>
            <strong>{result.oodPositiveEvaluation.replace("_", " ")}</strong>
          </div>
          <p>
            This measures retention of known-source PTB-XL ECGs—not detection
            of unfamiliar diseases, sites, or devices. It is retrospective
            research evidence, not clinical validation.
          </p>
        </aside>
      </div>
    </section>
  );
}
