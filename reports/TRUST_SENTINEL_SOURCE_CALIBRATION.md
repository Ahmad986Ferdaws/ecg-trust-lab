# Trust Sentinel source-calibration status

**Protocol:** `trust-sentinel-source-calibration-v1`
**Execution status:** `PREPARED_NOT_RELEASE_READY`
**Scope:** retrospective source-domain development only

The frozen preparation completed from clean commit
`5919adb47113ef2d75d83b844073359ab939a874`. Independent read-only audits
collectively verified its hashes, patient-level partitions, fitting-role
isolation, and aggregate validation results. The local result is ignored by
Git and contains no ECG or patient identifiers, row-level outputs, logits,
probabilities, waveforms, or filesystem paths.

## Patient-separated roles

| Role | ECGs | Patients | Permitted use |
|---|---:|---:|---|
| Decision fit | 847 | 751 | Temperature, class thresholds, entropy cutoff |
| Conformal/OOD fit | 834 | 757 | Conformal calibration; OOD role reserved |
| Source validation | 465 | 409 | Evaluation only; no tuning |

The roles are mutually patient-disjoint and exhaust all 2,146 fold-9 source
records in the frozen input. PTB-XL fold 10 and SPH were not used.

## Frozen development components

- Temperature: `1.3196200524`; decision-fit binary NLL changed from `0.295477`
  to `0.284835`.
- Label thresholds for NORM, MI, STTC, CD, and HYP: `0.414990`, `0.375395`,
  `0.364357`, `0.362302`, and `0.308924`.
- Entropy cutoff: `0.597575`, fitted to approximately 80% decision-set
  retention with the preregistered tie rule.
- Label-wise split conformal: `alpha=0.1`, fitted only on its reserved role.

## Untuned source-validation point estimates

| Measure | Value |
|---|---:|
| Macro AUROC | 0.914492 |
| Macro average precision | 0.797413 |
| Brier score | 0.085276 |
| ECE, 15 bins | 0.043885 |
| Frozen-threshold Hamming loss | 0.126452 |
| Frozen-threshold exact-match accuracy | 0.593548 |
| Entropy-retained coverage | 0.819355 |
| Entropy-retained exact-match accuracy | 0.669291 |
| Conformal marginal coverage | 0.900215 |
| Mean conformal set size | 1.042581 |

These are development point estimates; confidence intervals are pending. The
conformal coverage is label-wise marginal coverage under exchangeability, not
individual certainty or simultaneous five-label coverage. Joint record
coverage was `0.647312`.

## Why this is not a release

Embedding-based unfamiliar-input detection remains explicitly `PENDING`: no
reference embedding artifact, threshold, device/precision binding, or source
false-rejection estimate exists. The release-readiness assertion therefore
fails closed, as designed. No complete Trust Sentinel release, external vNext
validation, clinical validation, diagnosis, or patient-care authorization is
claimed.

The next scientific step is a new immutable OOD-completion protocol. It must
bind the frozen ResNet checkpoint and preprocessing, use folds 1–8 only for
reference embeddings, fit the threshold only on the reserved fold-9 role, and
evaluate false rejection only on source validation before a complete release
can be assembled.

## Integrity references

- Frozen config: [`configs/trust_sentinel_source_calibration_v1.yaml`](../configs/trust_sentinel_source_calibration_v1.yaml)
- Config SHA-256: `3dbef163757807c442276b80631e0c83a6c07b241c62974fc64ba91bbedb8178`
- Logical result SHA-256: `b9063fd2965b194806f9e544f3ea6390cc19bc8a93b27d3e88a674bf0aa7c839`
- Physical result-file SHA-256: `8bae3acdebac42504167afc7bb7d2051b7ac2c48019aa429ed6544f14a59f38f`
- Frozen-component SHA-256: `a3180e2f5da6cfd7b44499e590202416fe669d51099298353926d828ab2004ca`
