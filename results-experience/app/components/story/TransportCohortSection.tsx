import styles from "./story.module.css";
import {
  AUDITED_TRANSPORT_COHORT,
  type CohortSnapshot,
} from "./storyData";

type TransportCohortSectionProps = {
  cohort?: CohortSnapshot;
  className?: string;
};

const countFormatter = new Intl.NumberFormat("en-US");

export function TransportCohortSection({
  cohort = AUDITED_TRANSPORT_COHORT,
  className = "",
}: TransportCohortSectionProps) {
  return (
    <section
      id="transport"
      className={`${styles.storySection} ${styles.transportSection} ${className}`}
      aria-labelledby="transport-evidence-heading"
    >
      <div className={styles.sectionInner}>
        <header className={styles.sectionHeader}>
          <p className={styles.kicker}>External transport / 03</p>
          <div>
            <h2 id="transport-evidence-heading">
              Frozen-model transport from PTB-XL to SPH
            </h2>
            <p className={styles.headerCopy}>
              Frozen models leave PTB-XL and enter an independently assembled SPH
              cohort. No SPH tuning decision is added between those two points.
            </p>
          </div>
        </header>

        <ol className={styles.transportSequence} aria-label="External transport sequence">
          <li>
            <span>01 / source</span>
            <strong>{cohort.sourceLabel}</strong>
            <p>Training and sealed in-distribution evaluation.</p>
          </li>
          <li className={styles.frozenStep}>
            <span>02 / transfer</span>
            <strong>Frozen models</strong>
            <p>Same weights and architectures. No destination retuning.</p>
            <dl>
              <div>
                <dt>Architectures</dt>
                <dd>1D ResNet + ECG Transformer</dd>
              </div>
              <div>
                <dt>SPH tuning decisions</dt>
                <dd>0</dd>
              </div>
            </dl>
          </li>
          <li>
            <span>03 / destination</span>
            <strong>{cohort.destinationLabel}</strong>
            <p>External discrimination and calibration reread.</p>
          </li>
        </ol>

        <div className={styles.transportLedger}>
          <div className={styles.transportFinding}>
            <p className={styles.ledgerLabel}>What the sequence establishes</p>
            <h3>SPH preserved AUROC while other metrics shifted.</h3>
            <p>
              Strong AUROC persisted in SPH. Average precision and calibration
              metrics changed with the destination cohort. That is evidence of
              robustness under dataset shift—not clinical validation.
            </p>
          </div>

          <table className={styles.cohortTable}>
            <caption>Audited SPH cohort sizes</caption>
            <thead>
              <tr>
                <th scope="col">Analysis cohort</th>
                <th scope="col">ECGs</th>
                <th scope="col">Patients</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <th scope="row">{cohort.primary.label}</th>
                <td>{countFormatter.format(cohort.primary.ecgs)}</td>
                <td>{countFormatter.format(cohort.primary.patients)}</td>
              </tr>
              <tr>
                <th scope="row">{cohort.broad.label}</th>
                <td>{countFormatter.format(cohort.broad.ecgs)}</td>
                <td>{countFormatter.format(cohort.broad.patients)}</td>
              </tr>
            </tbody>
          </table>
        </div>

        <aside className={styles.transportMethodNote} aria-label="Transport interpretation note">
          <strong>Read transport as a stress test.</strong>
          <p>
            It asks whether a locked research result remains informative in a
            different dataset. It does not test prospective workflow, clinician
            behavior, patient outcomes, or medical-device performance.
          </p>
        </aside>
      </div>
    </section>
  );
}
