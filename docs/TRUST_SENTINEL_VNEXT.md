# ECG Trust Sentinel vNext

**Status:** active implementation
**Protocol:** `trust-sentinel-vnext-dev-v1`
**Scope:** retrospective research and education only

## Objective

The vNext line extends the completed PTB-XL and SPH work into a fail-closed
ECG assurance system. Its primary question is not merely which label has the
largest score. It asks whether an input is valid, readable, represented by the
evaluated data, and sufficiently certain for the frozen research classifier to
show a result.

The sealed PTB-XL fold-10 and SPH studies remain immutable historical evidence.
They are already observed and cannot be reused as untouched confirmation sets
for vNext methods. The machine-readable development contract is
[`configs/trust_sentinel_vnext.yaml`](../configs/trust_sentinel_vnext.yaml).

## Required decision states

Every analysis ends in exactly one state, evaluated in this order:

1. `INVALID_INPUT`: the file, metadata, shape, units, or processing contract is
   invalid.
2. `REACQUIRE`: the signal is unreadable or has a suspected lead problem.
3. `UNSUPPORTED_INPUT`: the signal is valid but outside the evaluated data
   distribution.
4. `ABSTAIN`: the signal is valid and supported, but the classifier is not
   sufficiently certain.
5. `PREDICTION_ALLOWED`: every preceding gate passed.

Only `PREDICTION_ALLOWED` may expose class results. Quality, distribution
support, and prediction uncertainty remain separate evidence streams because
they represent different hazards. A missing component, artifact mismatch, or
runtime failure closes the gate rather than falling back to classification.

## Architecture

The first implementation is a modular Python application with one local GPU
worker. It deliberately avoids a microservice fleet. The existing
provenance-checked inference, decisioning, robustness, explanation, and
artifact modules remain the domain kernel and are extended through typed
adapters.

The new boundaries are:

- `contracts`: versioned external schemas;
- `registry`: immutable decision-system release bundles;
- `quality`: deterministic input and per-lead signal assurance;
- `open_world`: reproducible unfamiliar-input baselines;
- `conformal`: label-wise uncertainty sets and coverage reports;
- `trust_policy`: the five-state fail-closed router;
- `stress`: controlled original-versus-perturbed comparisons;
- `monitoring`: aggregate quality, shift, abstention, and operational signals;
- `service`: versioned API and background research jobs; and
- `evidence`: privacy-reviewed aggregate publication bundles.

A release binds the input contract, preprocessing, model members, calibration,
classification thresholds, uncertainty policy, quality policy, distribution
policy, ontology, source-code revision, dependency lock, evaluation evidence,
and safety language. The service must refuse readiness if any bound artifact or
compatibility check fails.

## Implemented development surface

The current source tree includes strict contracts and release-bundle
verification; deterministic signal-quality checks; conformal and open-world
scoring primitives; the five-state Sentinel policy and service boundary; a
synthetic-only Failure Lab; aggregate drift monitoring and a tamper-evident
audit ledger; multi-site governance and model-passport contracts; and bounded
research scaffolds for foundation representations, counterfactual review,
longitudinal studies, and human-factors studies. These modules are tested
research infrastructure, not completed clinical studies or claims of benefit.

The source-calibration protocol is frozen as a development-only preparation
step. Its embedding-based unfamiliar-input component remains explicitly
`PENDING`, so no complete vNext release is ready. The historical Plotly demo is
a separate entropy-gated baseline and must not be presented as the Sentinel
service. No vNext component is authorized for patient care.

## Delivery sequence

### R1 — Architecture and input assurance

- Add strict contracts and an immutable release manifest.
- Add canonical ECG validation and per-lead quality findings.
- Detect flatline, clipping, spikes, baseline wander, electrical interference,
  high-frequency noise, and probable limb-lead reversal.
- Never silently repair or relabel a suspected reversal.
- Add deterministic synthetic and malformed-input tests.

### R2 — Reason-aware deferral

- Preserve the frozen entropy gate as a baseline.
- Add label-wise split-conformal decisions.
- Add entropy, energy, and embedding-distance unfamiliar-input baselines.
- Implement the five-state policy with stable reason codes.
- Report coverage, set size, selective risk, unfamiliar-input performance, and
  subgroup coverage without fitting on target-site results.

### R3 — Failure Lab

- Add an authorized or synthetic-case workflow.
- Apply versioned lead dropout, reversal, noise, drift, clipping, gain, and
  timing perturbations.
- Compare quality findings, distribution support, calibrated scores, deferral,
  and explanation stability before and after each perturbation.
- Keep the public evidence site aggregate-only; do not create a public patient
  upload service.

### R4 — Multi-site passport and drift replay

- Add isolated external dataset adapters without weakening the PTB-XL loader.
- Preserve native labels and freeze a clinically reviewed shared ontology.
- Exclude overlapping PTB/PTB-XL sources.
- Assign development, calibration, observed, and untouched lockbox roles before
  viewing model results.
- Publish per-site quality, unfamiliarity, discrimination, calibration,
  abstention, and subgroup evidence with patient-level uncertainty.
- Replay abrupt and gradual shifts through an investigate, restrict, pause, and
  rollback action ladder. Do not retrain automatically.

### R5 and later research

- Compare open pretrained ECG representations using frozen heads and
  parameter-efficient tuning; do not attempt foundation pretraining locally.
- Develop physiologically constrained model-sensitivity counterfactuals and
  require blinded cardiology review before making usefulness claims.
- Begin longitudinal trajectory research only with strict temporal splits,
  leakage audits, censoring-aware evaluation, and independent validation.
- Any prospective work begins with formative usability and silent observation,
  requires appropriate governance, and must not influence care.

## Release gates

Each release must pass source and frontend static checks, unit and integration
tests, schema compatibility, dataset-role validation, patient-isolation gates,
artifact hashing, malformed-input and controlled-corruption cases, privacy
allowlisting, and a real GPU smoke test before promotion. Scientific results
must report confidence intervals and per-site or per-subtype behavior; pooled
averages cannot hide a critical failure.

The numerical quality, coverage, and unfamiliar-input thresholds in the YAML
are provisional preregistration candidates. They are not universal medical
standards and require cardiology and statistical review before any study that
could affect clinical claims.

## Non-goals

vNext does not provide diagnoses, treatment recommendations, emergency
screening, individual certainty guarantees, autonomous retraining, or evidence
of clinical safety. A model score is not a disease probability, an
unfamiliar-input flag is not discovery of a new condition, and an attribution
or generated waveform is not a causal physiological explanation.
