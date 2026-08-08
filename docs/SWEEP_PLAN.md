# Equal-budget sweep plan

## Scientific status

The checked YAML pair and schema-v2 executor form the frozen primary sweep
protocol. The sampler design, objective, pruning policy, seed semantics,
tie-breaking rule, failure accounting, paired execution order, provenance
checks, and winner-release gate are typed, hashed, and tested. Fold-8 sweep
results remain development evidence and cannot be described as final-test
performance.

"Equal budget" currently means equal trial count, epoch ceiling, search-space
domain, fold roles, parameter-matched preset, loader policy, early-stopping
policy, device policy, and precision policy.  It does **not** mean equal wall
time, GPU energy, FLOPs, optimizer-step count, or completed epochs.  Results
must be described as a fixed-model, common-optimizer, equal-attempt comparison,
not as a search for each architecture family's unconstrained best model.

## Landed configuration contract

The repository now contains a strict, typed configuration layer for a paired
development sweep:

- `configs/sweep_resnet_equal_budget.yaml`
- `configs/sweep_transformer_equal_budget.yaml`

Both sides use the existing fixed `matched_capacity` models, folds 1–7 for
training, and fold 8 for model selection. They share one immutable 12-row
SciPy Latin-hypercube candidate plan, a 30-epoch ceiling, fixed experiment seed
`2026`, one search space, one SQLite path, and one output root. The search is
limited to fields already validated by
`DevelopmentExperimentConfig`:
learning rate, weight decay, batch size, gradient clipping, warmup epochs, and
minimum learning-rate ratio.

Validate the paired contract without starting training:

```powershell
uv run --frozen pytest tests/unit/test_sweep_config.py
```

## Execution layer

`ecg_trust.sweep_runner` and `scripts/sweep.py` implement read-only preflight
and status, fresh launch, and explicit resume. The runner uses persistent
SQLite state, an exclusive comparison writer lock, immutable candidate/attempt
directories, alternating within-pair architecture order, non-consuming retry
of the same failed candidate, and filesystem reconciliation of every COMPLETE
row. It releases winners only after 12 verified candidates per architecture.

```powershell
uv run python scripts/sweep.py preflight
uv run python scripts/sweep.py status
uv run python scripts/sweep.py run
uv run python scripts/sweep.py run --resume
```

The last command is recovery-only and requires the existing immutable
candidate plan. The executor composes the development runner, so folds 9 and
10 are unreachable from this workflow.

Architecture-specific searches over dropout, ResNet width/kernels, and
transformer patch/depth/dimension/head compatibility are also deferred. They
require typed model overrides and refit provenance checks; adding them as an
unvalidated side channel would weaken the current experiment contract.

## Required protocol decisions before execution

### Candidate sampling and search-space fairness

- Freeze a sampler implementation and version, not only `sampler_seed`.
  Independent adaptive TPE studies can diverge after observing different
  architecture-specific objectives even when they share a seed and domain.
- For the cleanest paired comparison, generate one immutable 12-row candidate
  manifest from the comparison seed, hash it, and enqueue the exact same row
  at the same trial index in both studies.  Record each row in every trial
  artifact.  A seeded random or QMC design is preferable to adaptive TPE for
  this small six-dimensional budget.
- Preflight all three batch sizes on both models without consulting fold-8
  scores.  If any candidate is infeasible for either model, revise and
  version the common search space before the study.  Do not silently give one
  architecture a replacement candidate after an OOM.
- Varying batch size changes optimizer steps per epoch.  The primary budget is
  therefore equal examples/epoch and equal epoch ceiling, not equal optimizer
  steps.  Log batches, optimizer steps, examples, elapsed GPU time, and peak
  VRAM for every trial.

Hardware preflight on 2026-08-08 passed for the largest shared candidate batch
(`192`) on the RTX 5070 Ti Laptop GPU using CUDA/BF16 synthetic training
steps.  The matched ResNet peaked at 586 MiB allocated VRAM and the matched
Transformer at 1,205 MiB; the smaller candidate batches are therefore within
the same hardware envelope.  The machine-readable result is stored locally at
`artifacts/benchmarks/sweep_batch192_preflight.json`.  This preflight did not
read PTB-XL or consult fold-8 outcomes.

### Seed semantics

All 12 candidates use fixed development seed `2026`, so initialization and
data-order variation are not confounded with candidate identity. The selected
configuration is then confirmed with the predeclared paired seeds `[2026,
2027, 2028]`. A failed or interrupted run is repeated with the same candidate
and seed; no seed may be selected, dropped, or replaced after observing fold-8
performance.

### Objective, epoch budget, and stopping

- Encode the objective as: maximize uncalibrated fold-8 macro ROC-AUC over the
  canonical five superclasses, computed from all fold-8 records after each
  epoch.  Require `roc_auc_labels == 5`; a macro mean that silently excludes a
  degenerate label is not a valid trial objective.
- Calibration, thresholds, abstention, subgroup results, fold 9, and fold 10
  are forbidden inputs to trial ranking.
- Freeze deterministic tie-breaking: first use higher objective, then fewer
  completed epochs, then lower trial number.  Treat non-finite objectives as
  failed trials, never as zero.
- The executor must replace the base experiment's 50 epochs with
  `budget.max_epochs == 30` in the resolved trial config.  The scheduler must
  also use 30 as its horizon; merely stopping a 50-epoch schedule at epoch 30
  is a different intervention.
- Early stopping is fixed at patience 10 and minimum delta 0.0001, starting
  with the first fold-8 evaluation.  Record both the selected checkpoint score
  and the literal maximum observed score because the minimum-delta rule can
  leave them slightly different.
- For the primary equal-budget claim, use no Optuna pruning and require 12
  **COMPLETE** trials per architecture.  Early stopping remains the common
  within-trial rule.  An OOM, exception, corrupt artifact, or interrupted run
  does not consume a completed-trial slot; pause and diagnose it.  Do not
  substitute a new candidate for only one side of a pair.

### Leakage and repeated-selection controls

- The sweep executor may construct datasets only through the development
  runner: folds 1–7 for fitting, fold 8 for objective/early stopping, and
  normalization fitted only on folds 1–7.
- Verify the protocol, manifest, normalization, base experiment, candidate
  manifest, source revision, and environment hashes before trial zero and on
  every resume.  Patient-disjointness and exact ECG-ID alignment remain hard
  failures.
- Fold 8 is repeatedly queried by design and is therefore a development set.
  Its winning score is selection-biased and must not appear as a final-test
  estimate.  Fold 9 remains calibration-only and fold 10 remains unopened
  until all downstream choices are frozen.
- Inspecting fold-8 learning curves for debugging is permitted, but changing
  the search space, budget, seed set, objective, or stopping policy in response
  creates a new versioned comparison; the previous study cannot be pooled with
  it.

### Storage and deterministic resume

- Use the shared SQLite file with two distinct study names and a single local
  writer.  On study creation, store immutable user attributes containing the
  comparison ID and hashes for the sweep config, candidate manifest, protocol,
  data manifest, normalization, base experiment, code revision, and lockfile.
  `load_if_exists` must reject any mismatch.
- Trial directories must be architecture/study/trial-specific, created
  atomically, and never reused or overwritten.  A `COMPLETE` database row is
  reportable only if its integrity-bound resolved config, checkpoint, metadata,
  and objective artifact all exist and agree.
- Resume may skip verified `COMPLETE` pairs.  A stale `RUNNING` trial is marked
  failed with its partial directory preserved, then rerun from scratch with
  the identical candidate and seed in a new attempt directory.  The current
  development runner does not resume an interrupted checkpoint, so the sweep
  layer must not imply mid-trial resume.
- Select a winner only after both studies have 12 verified complete paired
  trials.  Database state alone is insufficient; reconcile it against the
  filesystem artifacts first.

## Freeze package before multi-seed development and refit

Write one immutable, integrity-bound selection artifact per architecture and a
comparison-level freeze artifact.  They must freeze at least:

- candidate-manifest hash, winning trial ID, complete resolved hyperparameters,
  model preset/config and parameter count;
- objective/direction/tie-break, selected fold-8 checkpoint epoch and score,
  early-stopping rule, and the rule for converting selected epochs into each
  folds-1–8 refit budget;
- preprocessing/label construction/normalization hashes, input resolution and
  lead order, fold roles, protocol hash, manifest hash, and exact ECG-ID sets;
- confirmatory seed list and whether results are individual-seed, aggregate,
  or an ensemble; never choose the best seed after observing fold 8;
- optimizer, scheduler, batch size, precision/determinism policy, code and
  dependency versions, and source checkpoint hashes;
- the two architectures to carry forward and the complete downstream analysis
  plan: calibration method, threshold objective, abstention score and target
  coverages, subgroup definitions, robustness perturbations, bootstrap unit,
  replicate count, confidence level, and all reported metrics.

After this artifact is signed, multi-seed runs may estimate seed sensitivity
but may not trigger new tuning.  Each refit must use folds 1–8 for the frozen
epoch rule, retain folds-1–7 normalization, use no validation/early stopping,
and preserve one-to-one lineage back to its selected development run.

## Sweep-launch acceptance gates

- [x] Sampler, candidate manifest, objective, direction, tie-break, pruning,
      failure accounting, and seed semantics are explicit typed fields.
- [x] The same 12 candidate rows are feasible and paired across architectures.
- [x] Dry-run tests prove each resolved trial uses 30 scheduler epochs and only
      folds 1–8 in their development roles.
- [x] Resume tests reject hash drift, preserve partial failures, and reproduce
      candidate/seed assignment exactly.
- [x] Completion requires 12 integrity-verified `COMPLETE` pairs; no best trial
      is exposed earlier.
- [ ] A selection/freeze artifact can be generated without accessing folds 9
      or 10 and is sufficient to construct all multi-seed/refit configs.
