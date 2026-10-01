# Known limitations

A single, current list of what this repository does **not** yet establish, with
where the evidence lives and what would resolve each item. It complements the
[model card](MODEL_CARD_PTBXL_SUPERCLASS_R3.md), the
[protocol deviations](../reports/PROTOCOL_DEVIATIONS.md), and the Trust Sentinel
[design](TRUST_SENTINEL_VNEXT.md). Nothing here is clinical validation, and the
project is not authorized for patient care.

Last reviewed against `main` on 2026-09-30.

## Trust Sentinel (vNext)

| Limitation | Evidence | What would resolve it |
|---|---|---|
| The five-state Sentinel runs only in tests. No script builds `SentinelEngine` on real inputs, and the local demo still uses the historical entropy gate. | `scripts/demo_server.py` → `ecg_trust.demo_app`; README "Demo walkthrough" | An integration entry point that runs the engine end to end on synthetic Failure Lab inputs and reports decision and reason-code counts, without inspecting outcomes. Any scientific evaluation needs a new protocol-scoped cohort and must never reuse the spent source-validation cohort C |
| There is no end-to-end Sentinel evaluation. The only quantitative evidence is a component-level source-calibration run (temperature, thresholds, entropy gate, conformal outputs) on 465 fold-9 source-validation ECGs, with the OOD component pending; that artifact was explicitly not a complete release. | [Source calibration report](../reports/TRUST_SENTINEL_SOURCE_CALIBRATION.md) | A scoped evaluation on data not used for calibration, under its own protocol |
| Unfamiliar-input (OOD) detection is unvalidated. v1 missed its preregistered support gate; v2 was terminated before inference; the v2.1 decision is open. | [v1 result](TRUST_SENTINEL_OOD_COMPLETION_RESULT.md), [pass-probability note](TRUST_SENTINEL_OOD_COMPLETION_GATE_POWER_NOTE.md), [v2 infeasibility](TRUST_SENTINEL_OOD_EXTERNAL_V2_INFEASIBILITY.md), issue #1 | Execute or terminate v2.1 (issue #1). Any successor should state its gate pass probability before freezing (`scripts/plan_gate.py`) and must not reuse cohort C |
| The default quality gate (`canonical-12x1000-mv-v1`) passes partial lead dropout (e.g. all leads flat for the last 5 s) and duplicated lead waveforms. | `scripts/quality_gate_stress_report.py` | The opt-in `SENTINEL_V2_SIGNAL_QUALITY_CONFIG` refuses them, but no release binds it yet. Adopt it in the next release with its own validation |
| New calibration and abstention options (class-conditional conformal budgets, worst-label entropy, classwise sigmoid scaling, label coherence, ensemble disagreement) are implemented but have never been run on data. | Modules under `src/ecg_trust/` and `src/ecg_trust/conformal/` | One preregistered development experiment on fold-9 roles not already spent by a one-shot protocol (never cohort C or fold 10), scored with E-AURC/AUGRC, per-label coverage, and per-group gates |

## Sealed PTB-XL and SPH evidence

| Limitation | Evidence | What would resolve it |
|---|---|---|
| Public fold-10 discrimination results (AUROC, average precision, paired deltas) are published only as macro summaries; per-label calibration curves exist (`reliability_seed2026.csv`), but there are no per-label discrimination summaries with patient-cluster intervals. | `publication/results/tables/` | Publish per-label discrimination tables with patient-cluster intervals from the existing sealed artifacts |
| Class imbalance was not addressed in the frozen models (unweighted BCE). | Frozen training configs; `ecg_trust.class_balance` now computes train-fold weights | A scoped weighted-training experiment on folds 1–8, recalibrated on fold 9 |
| One global temperature calibrates all five labels. | [Model card](MODEL_CARD_PTBXL_SUPERCLASS_R3.md) | Compare per-label calibrators on fold 9 before any new release |
| Global abstention gates under-covered patients aged 80+. | README "Current status" (r3 package) | A group-aware abstention policy with an absolute per-group retention (coverage) gate, validated separately in the next release protocol. `ecg_trust.slice_gates` only prevents further regression and can't fix existing under-coverage; group-conditional conformal sets address per-group set *coverage*, not abstention retention, and aren't wired into the Sentinel |
| Strict operator-level outcome-label blindness to fold 10 was breached before evaluation (DEV-001). | [Protocol deviations](../reports/PROTOCOL_DEVIATIONS.md) | Not resolvable retroactively; disclosed |
| SPH transport is exploratory. The cohort is label-selected, the cross-ontology map was not clinically adjudicated, and prevalence differs sharply from PTB-XL. | [SPH r2 protocol](EXTERNAL_TRANSPORT_SPH_R2.md), issue #6 | A clinically reviewed ontology map (issue #6); decompose prevalence shift separately from other shift |
| No foundation-model, longitudinal, or human-factors evidence. | Issue #7 | The parked studies in issue #7 |

## Engineering and reproducibility

| Limitation | Evidence | What would resolve it |
|---|---|---|
| Scientific runs were made on one Windows laptop, and the OOD v2 line pins that host's runtime. | [Environment](ENVIRONMENT.md), `REQUIRED_RUNTIME_BINDING_PATHS` in `src/ecg_trust/ood_v2/pipeline.py` | Decide issue #1; afterwards the v2 line can be simplified or archived |
| CPU CI skips about 108 tests that need Windows, CUDA, or private artifacts. | `.github/workflows/cpu-quality.yml` run summaries | Expected; report skips with results as CONTRIBUTING requires |
| A fresh clone contains no PTB-XL records, checkpoints, or predictions. | [Reproducibility guide](REPRODUCIBILITY.md) | Rebuild from the official dataset; don't treat the repository as a downloadable model |
