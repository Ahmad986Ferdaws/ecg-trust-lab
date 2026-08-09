# Post-evaluation audit run log

This is the operational record for the descriptive analyses performed after
the preregistered fold-10 batch was complete. It complements the
[final-evaluation run log](FINAL_EVALUATION_RUN_LOG.md) and the
[protocol-deviation log](PROTOCOL_DEVIATIONS.md); it does not replace either
record or convert post-evaluation analyses into confirmatory evidence.

## Interpretation boundary

The six-member architecture results and within-seed paired patient-bootstrap
comparisons came from the sealed fold-10 batch. Reliability, dense
risk-coverage, subgroup views, controlled corruptions, attribution controls,
publication rendering, and the local viewer were specified and run only
afterward. They are descriptive audits of frozen models and frozen fold-9
policies. They were not used to select a model, seed, temperature, threshold,
coverage gate, subgroup, or headline metric.

This is research-only benchmark evidence. It does not establish clinical or
external validity, diagnostic safety, fairness, medical-device performance,
or fitness for patient care.

## Immutable revision lineage

No failed output tree was edited, deleted, or reused. Each successor used a
new sibling directory, a new clean Git revision, and a specification that
cryptographically binds the preceding tree and sets
`derived_artifact_reuse_allowed: false`.

| Revision | Analysis Git revision | Specification SHA-256 | Outcome |
|---|---|---|---|
| r1: `runs/post_evaluation/ptbxl_matched_equal_budget_v1/` | `a29ae5830f15b2b677a07c87f6bc29b045c82644` | `sha256:3a86035ecfaebafc0bfa8da4713fde90cb62345f151996f5ff79e8a68fa3043e` | Aborted. Decimal corruption IDs were passed through `Path.with_suffix`, so names such as `baseline-wander-0.05` collided at `baseline-wander-0.npz`. The 31-file tree is preserved with tree hash `sha256:32c7c242fc74de9917286d9de879364aba130ee4427af47c9d80f0c5f28fb387`; no robustness, explanation, or derived manifest was published. |
| r2: `runs/post_evaluation/ptbxl_matched_equal_budget_v1__audit-r2/` | `e280a43bc7bf75bd0219b1bbaa38ffd8c6d88ef5` | `sha256:c0b07012265b5bd63ffb2c08438b64db7d74877e88050673e3192e354d9a0073` | Superseded r1 for `decimal_case_id_suffix_collision`. All 246 robustness cases and all explanation artifacts were regenerated. Finalization then stopped because the loader treated the JSON key order of `attribution_runtime` as semantic, even though the production serializer canonically sorts mapping keys. The 550-file tree is preserved with tree hash `sha256:55e8ecb005ec8fb1a9679a8e49221972725dbdc3b7bb037ba3b0d448ce0a3b3d`; no derived manifest or final report was written. |
| r3: `runs/post_evaluation/ptbxl_matched_equal_budget_v1__audit-r3/` | `731753796e3c26d68bfff072583c5ea54851e8c1` | `sha256:5727858f0c22b6311152749e0d9a3d20b3c14f4ee1c72ef9d1cf6e1943434200` | Superseded r2 for `attribution_runtime_mapping_order_validation`. Validation was corrected to compare mapping key sets, and every probability, robustness, explanation, demo, table, figure, and report artifact was regenerated from its frozen sources. Finalization completed. |

The first r3 finalization attempt stopped at the required-disclosure marker
check because two phrases in `FINAL_EVALUATION_RUN_LOG.md` were split by
Markdown line wrapping. Commit `99602a7` reflowed only those operational
phrases. Finalization then ran at that clean descendant with Git-envelope
equality disabled intentionally: the frozen analysis revision `7317537`
remained its ancestor, scientific code and every bound source were reverified,
and the existing r3 artifacts were finalized without a new scientific query.

The r2 branch manifests remained valid data under the corrected order-insensitive
loader, but none of those artifacts was copied into r3. The complete r3
derived manifest inventories 554 generated files and separately binds its
verified prerequisites.

## r3 identity and integrity anchors

| Artifact | Canonical artifact SHA-256 |
|---|---|
| Post-evaluation specification | `sha256:5727858f0c22b6311152749e0d9a3d20b3c14f4ee1c72ef9d1cf6e1943434200` |
| Probability audit | `sha256:e5f5261b2ff95dfe11596e544227effad5d5e98a4e547251fa5246d32ad26528` |
| Robustness manifest | `sha256:be9ff204f85f2ca11ed78ed93d4bf066029fd5c68f7c61fcfb0e1d9cfa6d906a` |
| Explanation manifest | `sha256:0803fe333b8f736d8c61babc2c50fe26194d3a75a7e06d123640ee2e1fc9733d` |
| Demo binding | `sha256:c8f7417875f646d1cabe4af4cb420813b2bde6cf6c201a0e1f1442ce133edb31` |
| Derived-artifact manifest | `sha256:fae6df30090ee59425a347034a7f4272cac5b799582a5742fff9c62b92a092f8` |

The final rendered report is
`runs/post_evaluation/ptbxl_matched_equal_budget_v1__audit-r3/reports/FINAL_RESULTS.md`
with file SHA-256
`c4363d89a43c59b7e7185a01b13a22377a90e35eb356e8a640bf7517fe721d0c`.
The specification itself binds the sealed evaluation specification
`sha256:1f73c021a544ffeb119ffe8e490a16e32ec84247e30bce1ffd895fcffed6c762`,
final batch
`sha256:a4da85d5272b1634baaf953496c3d9efd8917777ad8de13b1b0c6dc754699e62`,
completed ledger
`sha256:3bb83554a08832212989ea8f3ea212f6af42c08460edbde9ba130065b1115a57`,
refit bundle
`sha256:7a0bbd07bfeec1cfb68599829921fee0ad4a3d97968520c43c952f1ea3bb59dd`,
and calibration bundle
`sha256:12b63f0ca20c0c8a901166ff3fa3bc8ed707cecb467413326e057d69c588b0ec`.

## Sealed result carried into the audit

The table is the descriptive mean and sample standard deviation across the
three preregistered seeds, not a pooled patient estimate.

| Architecture | Macro AUROC | Macro AP | Brier | ECE |
|---|---:|---:|---:|---:|
| 1D ResNet | `0.921921 +/- 0.000913` | `0.810248 +/- 0.003327` | `0.085040 +/- 0.000718` | `0.022744 +/- 0.002984` |
| ECG transformer | `0.897420 +/- 0.003270` | `0.765270 +/- 0.007739` | `0.096807 +/- 0.001968` | `0.025646 +/- 0.001163` |

Paired differences below are transformer minus ResNet on aligned patients.
The AUROC and AP intervals are wholly below zero and the Brier intervals wholly
above zero for every seed, supporting the ResNet on those metrics under the
frozen comparison. Every ECE interval crosses zero, so the calibration-error
difference is inconclusive.

| Seed | Macro AUROC, 95% CI | Macro AP, 95% CI | Brier, 95% CI | ECE, 95% CI |
|---:|---:|---:|---:|---:|
| 2026 | `-0.0280 [-0.0342, -0.0222]` | `-0.0515 [-0.0638, -0.0390]` | `+0.0143 [+0.0113, +0.0174]` | `+0.0048 [-0.0006, +0.0108]` |
| 2027 | `-0.0244 [-0.0304, -0.0185]` | `-0.0467 [-0.0590, -0.0340]` | `+0.0116 [+0.0085, +0.0144]` | `-0.0001 [-0.0056, +0.0055]` |
| 2028 | `-0.0211 [-0.0272, -0.0150]` | `-0.0367 [-0.0483, -0.0247]` | `+0.0094 [+0.0063, +0.0124]` | `+0.0040 [-0.0024, +0.0080]` |

ECE needs an additional caveat: patient resampling duplicates rows while the
fixed 15-bin estimator repartitions those rows. This can bias the bootstrap
distribution and can place an ECE point estimate outside its percentile
interval. The ECE intervals are sensitivity summaries, not conventional
inferential guarantees.

## Probability, selective-prediction, and subgroup audit

Only the frozen fold-9 temperatures, label thresholds, and entropy cutoffs
were applied. Tightening the global gate generally reduced Hamming risk, but
realized coverage did not exactly equal the target and accepted predictions
still contained errors. At the nominal 0.8 gate, realized global coverage was
approximately 0.79-0.81 across members; Hamming risk was approximately
0.09-0.10 for the ResNet and 0.11 for the transformer. This is a workload-risk
tradeoff, not a clinical safety guarantee.

Coverage was not uniform across age bands. At the nominal 0.8 gate, coverage
for age 80+ was approximately 0.60-0.65, versus approximately 0.93-0.95 for
age under 40. The age-80+ group also had the lowest age-band AUROC for every
member: approximately 0.881-0.885 for the ResNet and 0.834-0.857 for the
transformer. Sex and age results are descriptive internal-cohort observations;
they do not establish fairness or discrimination, and the available metadata
cannot support a comprehensive fairness claim.

## Controlled-corruption audit

The grid contains 246 verified member-cases: six members times 41 cases.
Clean logits matched the sealed sources exactly for all six members before any
corruption was evaluated. Across the 240 non-clean cases:

- the worst macro-AUROC delta was `-0.2939` for transformer seed 2027 under
  full lead reversal;
- the largest macro-Brier delta was `+0.2213` for the same member and case;
- the largest Hamming-AURC delta was `+0.3857` for ResNet seed 2026 under full
  lead reversal; and
- the largest absolute frozen-gate coverage delta was `0.5714` for ResNet
  seed 2027 after dropping all precordial leads.

These deliberately controlled and sometimes semantically destructive
perturbations expose sensitivity. They are not estimates of deployment shift
or evidence of real-world robustness.

## Explanation-control audit

The fixed, one-ECG-per-patient cohort contains 60 ECGs balanced across ten
label/status cells. It produced 15 method artifacts and 900 method-ECG
evaluations across the six members. Repeat attributions were exact for every
method. Aggregate controls were:

| Control | Result |
|---|---:|
| Minimum method-level mean repeat cosine | `1.0000` |
| Mean stability cosine at 40 dB | `0.9929` |
| Mean parameter-randomization cosine | `-0.0067` |
| Mean random-minus-guided deletion AUC | `0.0495` (positive favors guided ranking) |
| Mean cross-method cosine | `0.5636` (`720/720` valid) |
| Mean cross-method Spearman | `0.2012` (`720/720` valid) |
| Maximum FP32-attribution versus sealed-BF16 raw-logit drift | `0.0913756` |

The attribution audit uses a signed correct-status target and has no clinical
localization ground truth. Grad-CAM, Integrated Gradients, and temporal
occlusion characterize model sensitivity; they are not causal explanations,
physiological evidence, or proof of clinician-like reasoning. The modest
random-minus-guided deletion-AUC difference and limited cross-method rank
agreement should be reported alongside the strong repeat and noise-stability
controls.

## Required disclosures and operational limitations

- [DEV-001](PROTOCOL_DEVIATIONS.md) records accidental pre-evaluation exposure
  of a bounded set of raw fold-10 label-bearing metadata rows. No exposed value
  was used to choose a model, policy, subgroup, or report, and no prediction or
  metric was seen then. Strict operator-level outcome-label blindness was
  nevertheless breached; this project must not claim a completely blind test.
- The sealed batch itself used ledgered immutable predictions, but six
  representation-only SHA-256 comparison failures required exact resumes.
  Existing predictions were revalidated and adopted without overwrite or a
  repeated scientific query. See the
  [final-evaluation run log](FINAL_EVALUATION_RUN_LOG.md).
- Focused post-evaluation suites passed. The final August 9, 2026 repository
  gate then passed all 434 tests in 156.41 seconds, Ruff, and strict mypy.
  Pytest emitted one upstream Starlette/httpx deprecation warning. A fresh
  workspace `--basetemp` and an unrestricted local test process were required
  because the managed sandbox could not traverse its default
  installed-package or temporary-directory paths.

## Viewer verification

An isolated headless-Chromium session verified the final local viewer against
the r3 demo binding. The first diagnostic navigation exposed two operational
defects: the Content Security Policy blocked the inline controller, and the
WebGL waveform trace did not render in that browser. The viewer was corrected
to use a fresh per-request script nonce, a no-request data favicon, and SVG
waveform traces. These changes did not alter model, calibration, decision, or
audit artifacts.

The clean replay reached `Model ready`, exposed exactly five label-free fold-8
examples, and completed both ordinary inference and ResNet Grad-CAM for the
first example. It rendered two probability traces, one threshold trace, all 12
canonical leads with 1,000 samples each, and three highlighted attribution
regions. The frozen gate returned `Accept` for that example. No local path,
traceback, error overlay, console error, failed request, or WebGL fallback was
observed. Screenshots were retained as `ecg-demo-r3-initial.png` and
`ecg-demo-r3-gradcam.png` in the external verification workspace rather than
added to the immutable r3 package.
