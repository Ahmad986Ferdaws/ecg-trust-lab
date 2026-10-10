import { RESULTS_STORY, SCIENTIFIC_CAVEATS } from "../../../lib/results";
import styles from "./EvidenceSourcesSection.module.css";

// Pin the links to the public revision verified when this register was added.
const evidenceBase = "https://github.com/Ahmad986Ferdaws/ecg-trust-lab/blob/43cf519ad1e0a664a5689a913e9123fc147294c9/";
const sources = [
  {
    title: "Sealed PTB-XL results",
    description: "Architecture comparison, paired inference, and the final evaluation boundary.",
    path: "reports/FINAL_RESULTS_PUBLIC.md",
  },
  {
    title: "Frozen SPH transport",
    description: "Primary and sensitivity cohorts, transported metrics, and label-mapping limits.",
    path: "publication/external_transport_sph_r2/FINAL_RESULTS.md",
  },
  {
    title: "Source-support completion",
    description: "The one-shot validation, bootstrap uncertainty, and preserved ineligible outcome.",
    path: "docs/TRUST_SENTINEL_OOD_COMPLETION_RESULT.md",
  },
] as const;

const limitationSources: Readonly<Record<string, { path: string; label: string }>> = {
  "ontology-bridge": { path: "docs/EXTERNAL_TRANSPORT_SPH_R2.md", label: "Read the mapping protocol" },
  "broad-unknowns": { path: "reports/SPH_EXTERNAL_TRANSPORT_AUDIT.md", label: "Read the cohort audit" },
  "rare-endpoints": { path: "reports/SPH_EXTERNAL_TRANSPORT_AUDIT.md", label: "Read the endpoint counts" },
  "blindness-deviation": { path: "reports/PROTOCOL_DEVIATIONS.md", label: "Read deviation DEV-001" },
};
const limitations = SCIENTIFIC_CAVEATS.filter((caveat) => caveat.id in limitationSources);

export function EvidenceSourcesSection() {
  return (
    <section className={styles.section} id="evidence-sources" aria-labelledby="evidence-sources-title">
      <header className={styles.header}>
        <p>Evidence register</p>
        <h2 id="evidence-sources-title">Follow the result back to the record.</h2>
        <p>
          Open the versioned public reports behind this notebook. Reported values
          reflect the audit of <time dateTime={RESULTS_STORY.updatedFromAudit}>August 29, 2026</time>.
        </p>
      </header>

      <ol className={styles.sources}>
        {sources.map((source, index) => (
          <li key={source.path}>
            <span aria-hidden="true">0{index + 1}</span>
            <h3><a href={`${evidenceBase}${source.path}`}>{source.title} <span aria-hidden="true">↗</span></a></h3>
            <p>{source.description}</p>
          </li>
        ))}
      </ol>

      <div className={styles.limitations}>
        <header>
          <p>Conditions on interpretation</p>
          <h3>The limits travel with the numbers.</h3>
        </header>
        <dl>
          {limitations.map((caveat) => (
            <div key={caveat.id}>
              <dt>{caveat.title}</dt>
              <dd>
                <p>{caveat.detail}</p>
                <a href={`${evidenceBase}${limitationSources[caveat.id].path}`}>
                  {limitationSources[caveat.id].label} <span aria-hidden="true">↗</span>
                </a>
              </dd>
            </div>
          ))}
        </dl>
      </div>
    </section>
  );
}
