# Follow-up completeness beside observed-only risk metrics

`evaluate_binary_risk_with_follow_up` adds an aggregate view of target availability
across predicted-risk bins. It embeds the existing observed-only evaluation without
changing its values or public format.

```python
from ecg_trust.longitudinal import (
    RiskEvaluationConfig,
    evaluate_binary_risk_with_follow_up,
)

config = RiskEvaluationConfig.create(
    horizon_days=90,
    calibration_bin_count=10,
    minimum_calibration_bin_count=5,
)
report = evaluate_binary_risk_with_follow_up("future_event", observations, config)
public = report.to_public_dict()
```

Use the same validated `RiskObservation` values as the existing evaluator. Excluded
targets must retain `outcome=None`; they are never converted to negative outcomes.
The iterable is consumed once, and the report retains only aggregates.

The new `ecg_trust.longitudinal_risk_follow_up.v1` envelope contains the unchanged
`observed_only_evaluation` plus `follow_up_completeness`. Its fixed-width bins use
the configured calibration-bin count, include their lower edge, exclude their upper
edge, and include risk 1 in the final bin. For each bin, counts and within-bin
fractions distinguish observed, right-censored, and insufficient-follow-up targets.
These are target statuses; an observed positive target does not by itself imply
complete observation of every day in the prediction horizon.

An empty bin has zero counts and null fractions. Every nonzero status cell must meet
`minimum_calibration_bin_count`. If any cell is smaller, the entire bin's counts,
total, and fractions are withheld. This prevents a rare status count from being
recovered directly from that bin's total and other status counts. Existing global
counts remain in the embedded report, so this suppression is not a formal privacy
guarantee against differencing across aggregates or repeated reports. Public bin
totals cannot be summed when some bins are suppressed.

The profile makes a descriptive selection issue visible: two cohorts with identical
observed outcomes and identical numbers of excluded rows can have identical current
metrics even when exclusions concentrate at opposite ends of predicted risk. This
report shows that difference; it does not correct the metrics, estimate inverse
censoring weights, establish unbiasedness, or provide a new performance claim.
Existing minimum-evidence rules still govern all observed-only metrics. This remains
retrospective research, not clinical decision support or causal evidence.
