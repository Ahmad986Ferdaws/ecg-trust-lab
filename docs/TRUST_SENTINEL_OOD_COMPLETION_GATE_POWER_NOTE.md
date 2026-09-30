# Source-support gate: design-level pass probability

**Status:** post-hoc engineering commentary, written after the one-shot run.
**Does not change:** the sealed [completion result](TRUST_SENTINEL_OOD_COMPLETION_RESULT.md),
its research ineligibility, the frozen [protocol](TRUST_SENTINEL_OOD_COMPLETION_PROTOCOL.md),
or any artifact. Nothing here reruns, retunes, or reinterprets the observed outcome as
a pass.

## Question

The completion run missed its preregistered support gate: 25 of 465
source-validation ECGs were rejected (5.38%), and the one-sided 95%
patient-cluster upper bound (7.30%) exceeded the 5% maximum. How likely was that
gate to pass **before** the run, if the detector behaved exactly as designed?

## Design facts (from the frozen protocol)

- The rejection threshold is the 95% inlier order statistic of 834 calibration
  scores: rank `k = ceil(835 × 0.95) = 794`.
- For exchangeable, tie-free scores, a new inlier exceeds that order statistic
  with probability `(834 + 1 − 794) / (834 + 1) = 41/835 ≈ 4.91%`. That is the
  *designed* false-rejection rate, already almost equal to the 5% maximum.
- The gate passes only if the one-sided 95% upper bound on false rejection
  across 465 records is at most 5%.

## Approximate pass probability

`scripts/plan_gate.py` (library: `ecg_trust.gate_power`) approximates the gate
with an exact record-level count cutoff and models the shared random threshold,
so the rejection count is beta-binomial `(41, 794)`:

```bash
uv run --no-sync python scripts/plan_gate.py --sample-size 465 --maximum-rate 0.05 --conformal 834 794
```

| Threshold rank of 834 | Design rejection rate | Largest passing count (of 465) | Approx. pass probability |
|---|---|---|---|
| 794 (the frozen 95% rule) | 4.91% | 15 | **≈ 0.10** |
| 805 | 3.59% | 15 | ≈ 0.43 |
| 815 (≈ 97.5%) | 2.40% | 15 | ≈ 0.86 |
| 822 | 1.56% | 15 | ≈ 0.98 |

The count cutoff approximates, but is not identical to, the protocol's
patient-cluster percentile bootstrap: the same number of rejections can pass
or fail depending on how records fall across the 409 patients. It is a planning
approximation, not the exact probability of the frozen rule.

## Reading

- Under its own design assumptions the gate would pass roughly one time in ten.
  The observed 25/465 sits close to the designed rate (about 23 expected). The
  miss is therefore weak evidence about detector quality either way; it mostly
  reflects a threshold and a gate that were set almost on top of each other.
- The result remains exactly as sealed: the run missed its gate and the bundle
  is not research-eligible. This note adds context only.

## Recommendation for any successor protocol

Before freezing, report the gate's approximate pass probability with
`scripts/plan_gate.py`, and choose the threshold rank and sample size together.
A stricter threshold (about the 97.5% inlier quantile) gives about 0.86 at the
same n, at the cost of rejecting fewer true out-of-distribution inputs. That
trade-off should be decided explicitly and written into the protocol. Cohort C
must not be reused (see issue #1).
