# Portfolio case study: trustworthy ECG classification

## One-sentence summary

I built and audited a reproducible 12-lead ECG research system that compares a
capacity-matched 1D ResNet with an ECG transformer on PTB-XL, then evaluates
calibration, abstention, subgroup coverage, corruption sensitivity, and
explanation controls before running a frozen, no-adaptation transport stress
test on a second retrospective ECG dataset.

## The problem

Many machine-learning projects report only whether a model separates classes
well. In a high-stakes setting, that is incomplete: probabilities can be
miscalibrated, confidence gates can fail unevenly across groups, minor input
changes can destabilize predictions, and attractive saliency maps may not be
faithful. This project asks both **which matched model performs better?** and
**what evidence would be needed to trust its behavior?**

The task is multi-label prediction of five PTB-XL diagnostic superclasses from
a ten-second, 100 Hz, canonical 12-lead ECG:

- normal ECG (NORM);
- myocardial infarction (MI);
- ST/T change (STTC);
- conduction disturbance (CD); and
- hypertrophy (HYP).

This is benchmark research, not a diagnostic system or medical device.

## What I built

- A provenance-checked PTB-XL 1.0.3 pipeline with 21,388 labeled ECGs from
  18,617 patients and patient-isolated official folds.
- A capacity-matched 1D ResNet and patch-based ECG transformer trained under
  the same search budget and three fixed confirmation seeds.
- A strict split protocol: folds 1-7 for training, fold 8 for model selection,
  fold 9 for calibration and decision policies, and fold 10 for one sealed
  final evaluation.
- Temperature calibration, per-label thresholds, and an entropy-based
  accept/defer gate fitted without fold-10 tuning.
- Paired patient-cluster bootstrap comparisons, reliability and risk-coverage
  analyses, subgroup coverage audits, 246 controlled-corruption member-cases,
  and 900 explanation-control evaluations.
- A frozen external-transport protocol that applied the same six members to
  SPH once, without SPH-based training, model selection, preprocessing
  adaptation, recalibration, threshold fitting, or gate fitting.
- A local FastAPI/Plotly viewer that presents raw and calibrated probabilities,
  thresholds, the frozen gate decision, a 12-lead waveform, and optional
  Grad-CAM or Integrated Gradients sensitivity overlays.
- Immutable specifications, ledgers, hashes, run logs, and regression tests so
  failures and recoveries remain visible instead of being silently rewritten.

## Main result

Values are mean +/- sample standard deviation across seeds 2026, 2027, and
2028 on the sealed fold-10 evaluation.

| Architecture | Macro AUROC | Macro average precision | Brier score | ECE |
|---|---:|---:|---:|---:|
| 1D ResNet | `0.921921 +/- 0.000913` | `0.810248 +/- 0.003327` | `0.085040 +/- 0.000718` | `0.022744 +/- 0.002984` |
| ECG transformer | `0.897420 +/- 0.003270` | `0.765270 +/- 0.007739` | `0.096807 +/- 0.001968` | `0.025646 +/- 0.001163` |

Within each seed, paired patient-bootstrap intervals supported the ResNet for
macro AUROC, average precision, and Brier score. Every paired ECE interval
crossed zero, so the project makes no comparative ECE claim. AUROC is not
classification accuracy; `0.9219` must not be described as “92% accurate.”

## Exploratory SPH transport result

After the PTB-XL r3 analysis was complete, the six frozen members were applied
unchanged to the retrospective SPH dataset. The primary conservative mapping
cohort contained 15,698 ECGs from 15,193 patients; the broader exact-10-second
cohort contained 18,842 ECGs from 18,157 patients.

| Architecture | Calibrated macro AUROC | Macro average precision | Brier score |
|---|---:|---:|---:|
| 1D ResNet | `0.930912 +/- 0.000964` | `0.698955 +/- 0.006752` | `0.061301 +/- 0.000248` |
| ECG transformer | `0.924088 +/- 0.001231` | `0.657838 +/- 0.007557` | `0.064153 +/- 0.003962` |

Values are mean +/- sample standard deviation across the same three seeds.
Paired patient-cluster bootstrap intervals favored the ResNet for macro AUROC
in all three seeds. This is only an exploratory retrospective external
transport stress test: there was no SPH tuning or recalibration, the new
AHA-to-PTB-superclass ontology bridge was not clinically adjudicated, and MI
and HYP are rare. It is not prospective or clinical validation and does not
establish diagnostic safety or deployment readiness.

## What the trustworthiness audit revealed

- At the nominal 0.8 coverage gate, realized coverage was about 0.79-0.81.
  Accepted cases still contained errors; abstention is a workload-risk tradeoff,
  not a safety guarantee.
- Coverage for age 80+ was only about 0.60-0.65, versus roughly 0.93-0.95 below
  age 40. A single global gate therefore cannot support a fairness claim.
- Full lead reversal produced the largest observed discrimination and
  calibration degradation, while dropping all precordial leads caused a large
  coverage shift. Controlled corruptions are sensitivity tests, not estimates
  of real-world deployment shift.
- Explanation repeats were exact and 40 dB noise stability was high, but
  cross-method rank agreement was limited. Saliency shows model sensitivity;
  it does not establish physiological causality or clinician-like reasoning.

## Engineering and research lessons

1. **Split governance is part of the model.** Patient isolation and a truly
   untouched final fold matter more than squeezing another decimal from a
   development result.
2. **Calibration and abstention need their own data.** Fold 9 was reserved for
   temperatures, thresholds, and coverage gates so the final test could not
   become another tuning set.
3. **Audit failures should be preserved.** Two post-evaluation packaging defects
   were superseded in new immutable roots; neither failed tree was edited or
   reused.
4. **A good-looking explanation is not validation.** Repeatability, deletion,
   stability, randomization, and cross-method controls changed the interpretation
   of the saliency maps.
5. **Operational transparency strengthens the result.** DEV-001 records a
   bounded metadata exposure that prevents claiming complete operator-level
   outcome-label blindness, even though no exposed value informed a decision.

## Verification record

- The historical August 9 r3 gate passed 434 automated tests.
- The current post-SPH gate separately passed 494 automated tests.
- Ruff and strict mypy passed.
- A real RTX 5070 Ti CUDA/BF16 forward-and-backward smoke test passed.
- Isolated Chromium verified ordinary inference and Grad-CAM with all 12 SVG
  waveform traces, no console errors, and no failed network requests.
- The r3 finalizer reverified the 554-file derived manifest and reproduced the
  existing final report bytes exactly.

## Portfolio-ready descriptions

### Short résumé bullet

Built a provenance-locked PTB-XL ECG benchmark comparing capacity-matched 1D
ResNet and transformer models across three seeds; achieved `0.9219` sealed
macro-AUROC and added calibrated abstention, patient-bootstrap inference,
subgroup/robustness audits, explanation sanity checks, a verified local demo,
and a frozen no-adaptation SPH transport stress test backed by 494-test
reproducibility coverage.

### Abstract-length summary

We compared capacity-matched 1D ResNet and ECG-transformer classifiers for
five-superclass, multi-label PTB-XL prediction under a patient-isolated,
four-stage data protocol. Models were selected on fold 8, calibrated on fold 9,
and evaluated once on sealed fold 10 across three fixed seeds. The ResNet
achieved mean macro AUROC `0.9219`, compared with `0.8974` for the transformer;
paired patient-bootstrap intervals favored the ResNet for AUROC, average
precision, and Brier score, while ECE differences remained inconclusive.
Post-evaluation audits exposed unequal abstention coverage for age 80+,
substantial sensitivity to destructive lead corruptions, and limited agreement
between attribution methods despite strong repeatability. The work demonstrates
an integrity-focused methodology for clinical-adjacent ML. A later frozen
SPH stress test retained the ResNet's mean macro-AUROC advantage on a
conservative 15,698-ECG mapped cohort without target-domain adaptation. Because
that bridge was unadjudicated, rare endpoints were sparse, and the study was
retrospective, the work explicitly makes no claim of clinical validity,
diagnostic safety, or medical-device performance.

## Authoritative evidence

- [Completed model card](MODEL_CARD_PTBXL_SUPERCLASS_R3.md)
- [Reproducibility guide](REPRODUCIBILITY.md)
- [Post-evaluation run log](../reports/POST_EVALUATION_RUN_LOG.md)
- [Required protocol deviations](../reports/PROTOCOL_DEVIATIONS.md)
- [Frozen SPH transport protocol](EXTERNAL_TRANSPORT_SPH_R2.md)
- [Sanitized SPH result](../publication/external_transport_sph_r2/FINAL_RESULTS.md)
- [SPH external-transport audit](../reports/SPH_EXTERNAL_TRANSPORT_AUDIT.md)
