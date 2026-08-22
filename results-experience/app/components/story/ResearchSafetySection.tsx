"use client";

import styles from "./story.module.css";
import { useStoryMotion } from "./useStoryMotion";

export type SafetyPrinciple = {
  index: string;
  title: string;
  description: string;
};

type ResearchSafetySectionProps = {
  eyebrow?: string;
  title?: string;
  description?: string;
  principles?: SafetyPrinciple[];
  className?: string;
};

const DEFAULT_PRINCIPLES: SafetyPrinciple[] = [
  {
    index: "01",
    title: "Retrospective evidence",
    description:
      "The results describe performance on historical research datasets, not prospective patient care.",
  },
  {
    index: "02",
    title: "External transport ≠ validation",
    description:
      "SPH transport probes robustness under dataset shift; it does not establish clinical safety or efficacy.",
  },
  {
    index: "03",
    title: "No diagnostic use",
    description:
      "This system is a research demonstration and must not guide diagnosis, treatment, or emergency decisions.",
  },
];

export function ResearchSafetySection({
  eyebrow = "04 · The boundary",
  title = "Strong evidence. A deliberate limit.",
  description =
    "Trustworthy machine learning is as explicit about what the experiment cannot prove as it is about every score it can measure.",
  principles = DEFAULT_PRINCIPLES,
  className = "",
}: ResearchSafetySectionProps) {
  const { ref, isVisible } = useStoryMotion<HTMLElement>();

  return (
    <section
      ref={ref}
      className={`${styles.storySection} ${styles.safetySection} ${
        isVisible ? styles.isVisible : ""
      } ${className}`}
      aria-labelledby="safety-heading"
    >
      <div className={styles.safetyGrid} aria-hidden="true" />
      <div className={styles.sectionInner}>
        <div className={styles.boundaryLabel}>
          <i aria-hidden="true" />
          Research system · not a medical device
        </div>

        <div className={styles.safetyHeadline}>
          <p className={styles.kicker}>{eyebrow}</p>
          <h2 id="safety-heading">{title}</h2>
          <p>{description}</p>
        </div>

        <div className={styles.principleDeck}>
          {principles.map((principle, index) => (
            <article key={`${principle.index}-${principle.title}`}>
              <div className={styles.principleFace}>
                <span>{principle.index}</span>
                <h3>{principle.title}</h3>
                <p>{principle.description}</p>
              </div>
              <i
                className={styles.principleEdge}
                style={{ "--edge-index": index } as React.CSSProperties}
                aria-hidden="true"
              />
            </article>
          ))}
        </div>

        <div className={styles.finalStatement}>
          <div className={styles.finalPulse} aria-hidden="true">
            <i />
            <i />
            <i />
          </div>
          <p>
            <span>Research conclusion</span>
            Across the audited experiments, the 1D ResNet is the stronger of the
            two tested architectures. The next claim requires prospective,
            clinically governed validation.
          </p>
        </div>
      </div>
    </section>
  );
}
