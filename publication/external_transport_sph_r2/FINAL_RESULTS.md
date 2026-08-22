# SPH exploratory external transport stress test

No tuning or recalibration on SPH was performed. This is not clinical validation; all outputs are research only.

Frozen protocol: `sha256:840acb758d50dbf1a04bf704b16a58d4d29d668370ce7ef91ef7a44860bf311b`

## Cohorts

| Cohort | Records | Patients |
|---|---:|---:|
| primary_mapped | 15698 | 15193 |
| broad_exact10 | 18842 | 18157 |
| no_ambiguous_mapped | 15563 | 15066 |

## Primary calibrated architecture results

Mean +/- sample SD across the three frozen seeds.

| Architecture | Macro AUROC | Macro AP | Macro Brier | Macro ECE |
|---|---:|---:|---:|---:|
| resnet1d | 0.930912 +/- 0.000964 | 0.698955 +/- 0.006752 | 0.061301 +/- 0.000248 | 0.052477 +/- 0.000877 |
| ecg_transformer | 0.924088 +/- 0.001231 | 0.657838 +/- 0.007557 | 0.064153 +/- 0.003962 | 0.061480 +/- 0.006313 |

## Primary calibrated per-class results

| Architecture | Label | AUROC | AP | Brier | ECE |
|---|---|---:|---:|---:|---:|
| resnet1d | NORM | 0.887893 +/- 0.001127 | 0.937523 +/- 0.001669 | 0.118254 +/- 0.000596 | 0.065973 +/- 0.001640 |
| resnet1d | MI | 0.980216 +/- 0.001208 | 0.619412 +/- 0.017834 | 0.035017 +/- 0.000585 | 0.090696 +/- 0.001966 |
| resnet1d | STTC | 0.920910 +/- 0.003316 | 0.758012 +/- 0.006022 | 0.081601 +/- 0.001492 | 0.016408 +/- 0.002955 |
| resnet1d | CD | 0.892561 +/- 0.004201 | 0.697833 +/- 0.002607 | 0.052958 +/- 0.000909 | 0.037849 +/- 0.004539 |
| resnet1d | HYP | 0.972978 +/- 0.001489 | 0.481996 +/- 0.026212 | 0.018676 +/- 0.002035 | 0.051459 +/- 0.003699 |
| ecg_transformer | NORM | 0.883921 +/- 0.003715 | 0.935566 +/- 0.001950 | 0.123225 +/- 0.006455 | 0.070551 +/- 0.013063 |
| ecg_transformer | MI | 0.958665 +/- 0.006363 | 0.460699 +/- 0.040070 | 0.045393 +/- 0.008322 | 0.117334 +/- 0.016543 |
| ecg_transformer | STTC | 0.915601 +/- 0.003742 | 0.740547 +/- 0.013332 | 0.083979 +/- 0.003084 | 0.017004 +/- 0.002567 |
| ecg_transformer | CD | 0.893508 +/- 0.007894 | 0.693236 +/- 0.026338 | 0.055302 +/- 0.003031 | 0.058718 +/- 0.006851 |
| ecg_transformer | HYP | 0.968744 +/- 0.003287 | 0.459141 +/- 0.026029 | 0.012868 +/- 0.000746 | 0.043795 +/- 0.000943 |

## Frozen entropy gates on the primary cohort

Each target is the nominal PTB fold-9 coverage; observed SPH coverage can differ.

| Target | Architecture | Observed coverage | Hamming risk | Exact-match accuracy |
|---:|---|---:|---:|---:|
| 1.0 | resnet1d | 1.000000 +/- 0.000000 | 0.093460 +/- 0.005504 | 0.708264 +/- 0.011761 |
| 1.0 | ecg_transformer | 1.000000 +/- 0.000000 | 0.098412 +/- 0.004306 | 0.703402 +/- 0.007923 |
| 0.9 | resnet1d | 0.961927 +/- 0.002639 | 0.085927 +/- 0.005424 | 0.728147 +/- 0.011532 |
| 0.9 | ecg_transformer | 0.966195 +/- 0.010345 | 0.091318 +/- 0.002339 | 0.721290 +/- 0.002837 |
| 0.8 | resnet1d | 0.911624 +/- 0.003844 | 0.077840 +/- 0.003800 | 0.751430 +/- 0.006583 |
| 0.8 | ecg_transformer | 0.914618 +/- 0.017132 | 0.082098 +/- 0.001353 | 0.746928 +/- 0.001394 |
| 0.7 | resnet1d | 0.841508 +/- 0.004072 | 0.068172 +/- 0.002060 | 0.781522 +/- 0.003185 |
| 0.7 | ecg_transformer | 0.844672 +/- 0.021061 | 0.071700 +/- 0.001120 | 0.778913 +/- 0.002507 |
| 0.5 | resnet1d | 0.665775 +/- 0.006905 | 0.045608 +/- 0.001087 | 0.860951 +/- 0.003648 |
| 0.5 | ecg_transformer | 0.671041 +/- 0.028108 | 0.048219 +/- 0.000848 | 0.859579 +/- 0.003332 |

## Paired Transformer-minus-ResNet primary differences

Calibrated point estimates and 95% paired patient-cluster bootstrap CIs.

| Seed | Metric | Estimate | 95% CI |
|---:|---|---:|---:|
| 2026 | roc_auc | -0.007596 | [-0.011126, -0.004146] |
| 2026 | average_precision | -0.056505 | [-0.076874, -0.037532] |
| 2026 | brier_score | 0.007232 | [0.006177, 0.008225] |
| 2026 | ece | 0.015173 | [0.013692, 0.016799] |
| 2027 | roc_auc | -0.007571 | [-0.012374, -0.002923] |
| 2027 | average_precision | -0.034377 | [-0.059769, -0.011066] |
| 2027 | brier_score | 0.000492 | [-0.000338, 0.001313] |
| 2027 | ece | 0.008057 | [0.006309, 0.009576] |
| 2028 | roc_auc | -0.005304 | [-0.009869, -0.000695] |
| 2028 | average_precision | -0.032469 | [-0.054194, -0.012668] |
| 2028 | brier_score | 0.000832 | [-0.000122, 0.001692] |
| 2028 | ece | 0.003780 | [0.002324, 0.005377] |

## Sensitivity summaries

Calibrated macro mean +/- sample SD across frozen seeds.

| Cohort | Architecture | AUROC | AP | Brier | ECE |
|---|---|---:|---:|---:|---:|
| broad_exact10 | resnet1d | 0.904714 +/- 0.001118 | 0.642155 +/- 0.006472 | 0.076688 +/- 0.000146 | 0.071016 +/- 0.000858 |
| broad_exact10 | ecg_transformer | 0.898652 +/- 0.001028 | 0.601741 +/- 0.008373 | 0.078817 +/- 0.003018 | 0.078034 +/- 0.004649 |
| no_ambiguous_mapped | resnet1d | 0.928405 +/- 0.001382 | 0.666044 +/- 0.005249 | 0.060795 +/- 0.000245 | 0.052181 +/- 0.000772 |
| no_ambiguous_mapped | ecg_transformer | 0.920511 +/- 0.000971 | 0.627274 +/- 0.003484 | 0.063547 +/- 0.003912 | 0.061313 +/- 0.006297 |

## Interpretation limits

The broad all-zero sensitivity treats diagnoses without a direct mapping as operationally all-zero; those absences are unknown, not verified negatives.

SPH labels are a conservative cross-ontology mapping without expert adjudication. Rare MI and HYP labels can yield unstable or undefined bootstrap replicates. Frozen PTB coverage cutoffs need not achieve their nominal coverage after transport.

Finite physical-amplitude extremes were retained without clipping, rejection, rescaling, model selection, or outcome-informed cleaning.
