# Evaluation and trust toolkit

A map of the analysis tools added for development work: what question each
answers, when to use it, and where it comes from. None of them changes a frozen
model, release, or sealed result. Two kinds are mixed below:

- **Descriptive** tools only summarize given predictions (the metrics, the Brier
  decomposition, ensemble disagreement, slice gates, gate planning).
- **Estimating** tools fit parameters from the data you give them:
  `GroupConditionalConformal.fit` (per-group thresholds), `weak_calibration`
  (logistic calibration coefficients), `estimate_label_shift` (EM priors and
  adjusted probabilities), and `training_class_balance` (loss weights).

Treat estimation on held-out data as fitting, not read-only analysis. Using any
tool on held-out data needs its own protocol; see
[known limitations](KNOWN_LIMITATIONS.md) for what is still open.

## Selective prediction (abstention)

| Tool | Question it answers | Notes |
|---|---|---|
| `selective_metrics.aurc`, `oracle_aurc`, `excess_aurc` | How good is the risk-coverage trade-off (AURC), what is the best achievable for these losses (oracle AURC, which depends on the losses only), and how far is a score from that best (E-AURC)? | E-AURC (Geifman et al., ICLR 2019) is zero for a perfect ranking at any accuracy, but its scale still depends on the loss distribution, so compare scores on the same predictions rather than across models with different error rates. Ties use their random-tie-break expectation. |
| `selective_metrics.augrc`, `generalized_risk_curve` | How often does a wrong prediction get through, averaged over coverage? | AUGRC (Traub et al., NeurIPS 2024) divides accepted loss by all cases, so tiny low-coverage prefixes aren't over-weighted. |
| `ensemble_uncertainty.decompose_ensemble_uncertainty` | Do the frozen members disagree on this record? | Total = aleatoric + epistemic (mutual information). Seeds of one architecture share blind spots; cross-architecture disagreement is more informative. |
| `scripts/compare_abstention_scores.py` | Does any score (worst-label entropy, threshold proximity, ensemble disagreement) beat the frozen mean-entropy gate score? | Uses the planned losses (mean binary log loss primary, thresholded Hamming secondary) on fold-9 rows only; members must average to the scored probabilities. `--synthetic-demo` for a data-free run. |

## Calibration

| Tool | Question it answers | Notes |
|---|---|---|
| `calibration_metrics.adaptive_calibration_error`, `monotonic_sweep_calibration_error` | How far are probabilities from observed rates, with less bias for rare labels? | Equal-mass bins (Nixon et al., 2019) and ECE-sweep (Roelofs et al., AISTATS 2022). |
| `calibration_hierarchy.weak_calibration` | Is the average risk right, and are predictions too extreme? | Calibration-in-the-large, calibration slope, and O/E per label (Van Calster et al., J Clin Epidemiol 2016). Reports non-convergence under separation. |
| `brier_decomposition.brier_decomposition` | Did a Brier change come from calibration or discrimination? | Exact five-term decomposition (Murphy; Stephenson, Coelho & Jolliffe, 2008). Tied forecasts share a bin. |

## Shift, subgroups, and release gates

| Tool | Question it answers | Notes |
|---|---|---|
| `prior_shift.estimate_label_shift`, `adjust_to_priors` | How much of an external miscalibration is just prevalence? | Saerens et al. EM (Neural Computation, 2002) under the label-shift assumption. Exploratory for SPH. |
| `conformal.GroupConditionalConformal` | Does coverage hold inside each subgroup, not just on average? | Mondrian conformal; small groups get trivial `{0, 1}` sets and are listed. Don't convert its sets with the v1 case-contract adapter: the adapter can't detect them and would stamp pooled provenance and marginal coverage on them. |
| `slice_gates.evaluate_slice_gate` | Did a candidate regress overall or in any group? | Compares at a fixed reference composition; small groups block by default. |
| `class_balance.training_class_balance` | What BCE `pos_weight` would folds 1–7 imply? | Refuses rows from folds 8–10. |

## Protocol planning and input quality

| Tool | Question it answers | Notes |
|---|---|---|
| `gate_power.plan_gate`, `scripts/plan_gate.py` | How likely is a preregistered proportion gate to pass if the system works as designed? | Binomial or split-conformal (beta-binomial) count approximation; see the [source-support note](TRUST_SENTINEL_OOD_COMPLETION_GATE_POWER_NOTE.md). |
| `SENTINEL_V2_SIGNAL_QUALITY_CONFIG`, `scripts/quality_gate_stress_report.py` | Which acquisition faults does each quality preset refuse? | v2 adds the flat-run and duplicated-waveform checks; v1 stays frozen. |
