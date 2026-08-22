"use client";

import type { CSSProperties } from "react";
import styles from "./story.module.css";
import {
  AUDITED_TRANSPORT_COHORT,
  type CohortSnapshot,
} from "./storyData";
import { useCountUp, useStoryMotion } from "./useStoryMotion";

type TransportCohortSectionProps = {
  cohort?: CohortSnapshot;
  className?: string;
};

type CSSVars = CSSProperties & Record<`--${string}`, string | number>;

const countFormatter = new Intl.NumberFormat("en-US");

function AnimatedCount({ value, active }: { value: number; active: boolean }) {
  const visibleValue = useCountUp(value, active);
  return (
    <span aria-label={countFormatter.format(value)}>
      <span aria-hidden="true">{countFormatter.format(visibleValue)}</span>
    </span>
  );
}

export function TransportCohortSection({
  cohort = AUDITED_TRANSPORT_COHORT,
  className = "",
}: TransportCohortSectionProps) {
  const { ref, isVisible } = useStoryMotion<HTMLElement>();

  return (
    <section
      ref={ref}
      className={`${styles.storySection} ${styles.transportSection} ${
        isVisible ? styles.isVisible : ""
      } ${className}`}
      aria-labelledby="transport-heading"
    >
      <div className={styles.sectionInner}>
        <header className={styles.sectionHeader}>
          <div>
            <p className={styles.kicker}>03 · External transport</p>
            <h2 id="transport-heading">
              Leave the home dataset. <span>Keep the model frozen.</span>
            </h2>
          </div>
          <p className={styles.headerCopy}>
            Generalization is tested by moving the trained models from PTB-XL
            into a distinct SPH cohort—without tuning them to the destination.
          </p>
        </header>

        <div className={styles.transportStage}>
          <div className={styles.transportCopy}>
            <p className={styles.transportRoute}>
              <span>{cohort.sourceLabel}</span>
              <i aria-hidden="true" />
              <span>{cohort.destinationLabel}</span>
            </p>
            <h3>A different hospital system. A harder question.</h3>
            <p>
              High AUROC survives transport, but changes in precision and ECE
              make the shift visible. That is evidence of robustness—not proof
              of clinical readiness.
            </p>

            <dl className={styles.cohortStats}>
              <div>
                <dt>{cohort.primary.label}</dt>
                <dd>
                  <strong>
                    <AnimatedCount value={cohort.primary.ecgs} active={isVisible} />
                  </strong>
                  <span>ECGs</span>
                  <small>
                    <AnimatedCount
                      value={cohort.primary.patients}
                      active={isVisible}
                    />{" "}
                    patients
                  </small>
                </dd>
              </div>
              <div>
                <dt>{cohort.broad.label}</dt>
                <dd>
                  <strong>
                    <AnimatedCount value={cohort.broad.ecgs} active={isVisible} />
                  </strong>
                  <span>ECGs</span>
                  <small>
                    <AnimatedCount
                      value={cohort.broad.patients}
                      active={isVisible}
                    />{" "}
                    patients
                  </small>
                </dd>
              </div>
            </dl>
          </div>

          <div
            className={styles.orbitScene}
            aria-label={`Transport from ${cohort.sourceLabel} to ${cohort.destinationLabel}; primary cohort ${countFormatter.format(
              cohort.primary.ecgs,
            )} ECGs from ${countFormatter.format(cohort.primary.patients)} patients`}
            role="img"
          >
            <div className={styles.orbitGlow} aria-hidden="true" />
            {[0, 1, 2].map((ring) => (
              <div
                key={ring}
                className={`${styles.orbitRing} ${styles[`orbitRing${ring + 1}`]}`}
                style={{ "--ring-index": ring } as CSSVars}
                aria-hidden="true"
              >
                <i />
                <i />
              </div>
            ))}
            <div className={`${styles.datasetNode} ${styles.sourceNode}`} aria-hidden="true">
              <span>TRAIN</span>
              <strong>{cohort.sourceLabel}</strong>
              <i />
            </div>
            <div className={`${styles.datasetNode} ${styles.targetNode}`} aria-hidden="true">
              <span>TRANSPORT</span>
              <strong>{cohort.destinationLabel}</strong>
              <i />
            </div>
            <div className={styles.orbitCore} aria-hidden="true">
              <span>FROZEN</span>
              <strong>MODEL</strong>
              <small>no retuning</small>
            </div>
            <div className={styles.signalParticles} aria-hidden="true">
              {Array.from({ length: 12 }, (_, index) => (
                <i
                  key={index}
                  style={
                    {
                      "--particle-index": index,
                      "--particle-delay": `${index * -0.23}s`,
                    } as CSSVars
                  }
                />
              ))}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
