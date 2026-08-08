# Multi-seed confirmation and freeze plan

## Purpose and stage boundary

This stage starts only after both equal-budget studies are complete and their
selection artifacts are integrity-verified. It uses folds 1–7 for fitting and
fold 8 for confirmation. It must finish with an immutable freeze artifact and
generated folds-1–8 refit recipes **before any fold-9 artifact is opened or
created**.

The fixed confirmatory seed set is:

```text
2026, 2027, 2028
```

These seeds are paired across architectures and may not be replaced after a
metric is observed. Seed `2026` intentionally repeats the fixed HPO seed so
the selected trial is one member of the confirmation set; seeds `2027` and
`2028` measure additional initialization/data-order sensitivity. A
hardware/interruption failure is rerun with the same seed and exact config.

## Audit of the current refit implementation

The existing refit runner already enforces important invariants:

- refit folds are exactly 1–8 and normalization remains fitted on folds 1–7;
- manifest, protocol, normalization provenance, patient isolation, model
  architecture/preset, optimizer values, batch size, and selected checkpoint
  config hash are checked;
- the model is initialized afresh, no validation dataset is constructed, and
  no early stopping or model selection occurs;
- `final.ckpt`, rather than minimum-training-loss diagnostics, is authoritative.

It is not yet sufficient as the post-sweep multi-seed contract:

- `FrozenRefitConfig` points directly to one editable checkpoint path and does
  not bind a sweep/confirmation freeze artifact or expected checkpoint hash;
- the runner does not verify a sibling completed-run metadata artifact or
  history, so it cannot prove that the checkpoint came from a complete
  confirmatory run;
- the refit seed and BF16 policy are not compared with the selected development
  run, and only batch size—not the full loader/reproducibility policy—is bound;
- lineage does not contain comparison ID, winning sweep trial, candidate
  manifest, confirmatory seed set, architecture decision, or code/lock hashes;
- final refit metadata does not bind the SHA-256 of `final.ckpt`;
- `last.ckpt` is labelled crash-recovery state, but there is no implemented
  mid-run resume path;
- the checked refit YAMLs are placeholders for the seed-2026 matched defaults.
  Their `frozen_epochs: 50`, checkpoint paths, and optimizer values cannot be
  used after a sweep whose development ceiling is 30.

Do not manually mutate the two checked placeholder YAMLs into post-sweep
recipes. Generate new, hash-bound recipes from the freeze artifact.

## Exact confirmation procedure

1. Load both completed sweep-selection artifacts and reject any mismatch in
   comparison ID, protocol, manifest, normalization, candidate-manifest,
   source revision, environment, folds, labels, input shape, or budget.
2. Materialize six development configs: two architectures times the three
   fixed seeds. Apart from architecture/model preset, each config must equal
   its architecture's winning resolved sweep config; only seed, run name, and
   output path may vary.
3. Train all six runs on folds 1–7 with fold-8 macro ROC-AUC early stopping.
   Require complete run metadata, a verified `best.ckpt`, and a fold-8
   prediction artifact for every run. No partial set is selectable.
4. Recompute each score from its integrity-checked fold-8 prediction artifact,
   using sigmoid probabilities and the canonical labels
   `[NORM, MI, STTC, CD, HYP]`. Require all five labels to contain both classes
   and require exact ECG-ID, patient-ID, target, and fold alignment across all
   six artifacts.
5. For architecture `a`, define the confirmation score exactly as

   ```text
   S_a = mean(AUROC_a,2026, AUROC_a,2027, AUROC_a,2028)
   delta = S_transformer - S_resnet
   ```

   Individual seed scores and the paired seed differences remain reportable;
   no seed is selected or discarded.

## Architecture decision rule

Predeclare a practical development margin of `0.005` macro-AUROC:

- `delta >= 0.005`: transformer is the development-selected primary;
- `delta <= -0.005`: ResNet is the development-selected primary;
- otherwise: record `practical_tie` and designate ResNet as the operational
  primary by the predeclared parsimony fallback. Do not describe this fallback
  as statistical superiority.

Both architectures and all three seeds remain frozen comparators and proceed
through refit, calibration, and the single authorized final-test batch. This
preserves the promised ResNet-versus-transformer study. The primary designation
is for the demo and headline ordering; fold 9 or fold 10 must not change it.

If resource constraints later require carrying only one architecture forward,
that restriction must be declared before confirmation results are computed.
It creates a different protocol version and forfeits a direct final-test
architecture comparison.

## Epoch-budget rule

Use one robust, architecture-specific epoch budget for all three refit seeds:

```text
e_a,s = selected zero-based best epoch + 1
E_a = max(warmup_epochs + 1, median(e_a,2026, e_a,2027, e_a,2028))
```

With three integer values, the median is an integer. Require `E_a <= 30`; any
violation is an integrity error. Each architecture's three folds-1–8 refits use
its same `E_a`, its original seed, no validation, and no early stopping. The
warmup-cosine scheduler is rebuilt with `E_a` as its horizon; it is not a
truncated 30-epoch schedule. Record this consequence explicitly.

This replaces the current runner's single-run rule that every refit's epoch
count must equal its own source checkpoint epoch plus one. The runner must
instead verify `E_a` against the loaded multi-seed freeze artifact. Using the
median prevents a lucky or unstable seed from setting a different training
budget for each final refit while retaining a deterministic fold-8-only rule.

## Freeze artifact schema

Add a strict module such as `ecg_trust.multiseed_freeze` with an immutable
`MultiSeedFreezeArtifact`. Its canonical JSON payload should contain:

```text
schema_version, artifact_type, comparison_id
protocol_hash, manifest_hash, normalization_hash, label_order
input_resolution, input_shape, lead_order, fold_roles
sweep_sources:
  candidate_manifest_hash
  resnet_selection_artifact_hash
  transformer_selection_artifact_hash
confirmation_plan:
  seeds: [2026, 2027, 2028]
  objective, direction, practical_margin, tie_policy
  epoch_budget_rule, carry_forward_policy
architectures:
  resnet1d / ecg_transformer:
    winning_sweep_trial and resolved-hyperparameter hash
    model metadata and parameter count
    for each seed:
      complete run-metadata hash and resolved-config hash
      history hash, best-checkpoint hash, fold-8-prediction hash
      selected epoch/count and all-five-label macro AUROC
    mean score, per-seed scores, paired differences, frozen refit epochs
decision:
  delta, status, primary_architecture, frozen_comparators
refit_recipes:
  architecture, seed, run name, source checkpoint/hash
  complete resolved optimizer/loader/runtime policy and frozen epochs
created timestamp, code revision, lockfile hash, software versions
artifact_sha256
```

The artifact writer must be atomic and non-overwriting. The SHA-256 covers the
canonical payload excluding only its own hash field. Its loader must use exact
schema keys, recompute every derived mean/delta/median/decision, and reject any
duplicate/missing seed, non-finite value, incomplete run, alignment mismatch,
role other than fold-8 model selection, or provenance drift.

The freeze builder should accept explicit development artifact paths only. It
must not enumerate prediction directories, accept a calibration artifact, or
have a fold-9/fold-10 option. Synthetic tests should pass deliberately supplied
fold-9 artifacts and prove rejection.

## Required refit changes

Introduce a versioned post-sweep refit config rather than weakening the
existing strict config silently. Each generated recipe must include
`freeze_artifact`, `freeze_artifact_sha256`, `comparison_id`, architecture,
confirmatory seed, expected source checkpoint SHA-256, and the frozen
architecture-level epoch budget.

Before dataset construction, the runner must:

- load and verify the freeze artifact, confirm that the recipe appears in it
  byte-for-byte, and confirm the architecture/seed is one of the six frozen
  members;
- verify source `run_metadata.json` has `status: complete`, then cross-check its
  hash, resolved config, history, best epoch/score, protocol, manifest,
  normalization, seed, BF16/determinism policy, and checkpoint hash;
- require all scientific fields to match the winning sweep config. Operational
  worker count may differ only if explicitly classified as non-scientific in
  the freeze schema; batch size and precision may not differ;
- use `E_a` from the artifact, never an editable YAML epoch or a fold-9 metric.

After training, add the final checkpoint SHA-256 and freeze-artifact SHA-256 to
refit metadata. Because JSON cannot bind a file written afterward without an
additional commit, write a separate immutable `refit_completion.json` after
`final.ckpt`; bind the final checkpoint, resolved config, metadata, protocol,
manifest, normalization, source development checkpoint, and freeze artifact.
Downstream prediction export should require this completion artifact.

## Fold-9 release gate

Fold 9 remains unavailable throughout confirmation, winner selection, freeze
creation, and refit. The only transition into calibration is a separate
`refit_bundle.json`, written after all six expected `refit_completion.json`
artifacts verify. It lists their final checkpoint hashes and the freeze hash.

Only then may an explicit calibration-export command create six fold-9
prediction artifacts, one per frozen architecture/seed. Calibration decisions
are fitted independently for each member unless an ensemble method and its
aggregation order were already specified in the freeze artifact. Fold-9
discrimination must not be used to change the architecture, seed set, epoch
budgets, hyperparameters, or ensemble membership.

## Acceptance gates

- [ ] Exactly six complete, aligned fold-8 confirmation artifacts exist for
      the fixed paired seeds; no replacement seed or best-seed selection.
- [ ] Architecture score, margin decision, practical-tie fallback, and median
      epoch budgets recompute exactly from source artifacts.
- [ ] Freeze JSON is immutable, integrity-bound, and contains no fold-9/10
      source or metric.
- [ ] Generated refit recipes are exact members of the freeze artifact; manual
      placeholder configs are rejected.
- [ ] Refit verification binds complete source-run metadata, checkpoint hash,
      seed, precision, scientific config, and architecture-level epoch budget.
- [ ] All six refit completions bind authoritative final checkpoints and the
      same freeze hash before the fold-9 release bundle can be created.
- [ ] CPU-only tests cover tampering, missing seeds, misalignment, incomplete
      runs, seed/config drift, premature fold 9, and deterministic regeneration.
