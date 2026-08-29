import styles from "./story.module.css";

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
    title: "Stress tests ≠ validation",
    description:
      "SPH transport and PTB-XL source-support analysis probe narrow failure modes; neither establishes clinical safety or efficacy.",
  },
  {
    index: "03",
    title: "No diagnostic use",
    description:
      "This system is a research demonstration and must not guide diagnosis, treatment, or emergency decisions.",
  },
];

export function ResearchSafetySection({
  eyebrow = "Research boundary / 04",
  title = "Research evidence, not clinical validation",
  description =
    "Trustworthy machine learning is as explicit about what an experiment cannot prove as it is about every score it can measure.",
  principles = DEFAULT_PRINCIPLES,
  className = "",
}: ResearchSafetySectionProps) {
  return (
    <section
      id="research-boundary"
      className={`${styles.storySection} ${styles.safetySection} ${className}`}
      aria-labelledby="research-boundary-heading"
    >
      <div className={styles.sectionInner}>
        <div className={styles.researchStatus}>
          <span>Research system</span>
          <strong>Not a medical device</strong>
        </div>

        <header className={styles.safetyHeader}>
          <p className={styles.kicker}>{eyebrow}</p>
          <div>
            <h2 id="research-boundary-heading">{title}</h2>
            <p>{description}</p>
          </div>
        </header>

        <ol className={styles.boundaryList}>
          {principles.map((principle) => (
            <li key={`${principle.index}-${principle.title}`}>
              <span>{principle.index}</span>
              <h3>{principle.title}</h3>
              <p>{principle.description}</p>
            </li>
          ))}
        </ol>

        <div className={styles.researchConclusion}>
          <span>Research conclusion</span>
          <p>
            Across the audited experiments, the 1D ResNet has the stronger
            discrimination results of the two tested architectures. Calibration
            conclusions are narrower, the source-support gate missed its frozen
            target, and any clinical claim requires prospective, governed
            validation.
          </p>
        </div>
      </div>
    </section>
  );
}
