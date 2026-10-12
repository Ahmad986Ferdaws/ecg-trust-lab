# Estimate research pilot delivery economics

Use `scripts/estimate_research_pilot.py` to test whether an estimated pilot fee
covers delivery and support effort. Every input is explicit. The output is a
scenario, not a pricing recommendation, market benchmark or revenue forecast.
The script uses only Python's standard library and no scientific artifacts.

```bash
python scripts/estimate_research_pilot.py \
  --fee-usd 5000 --delivery-hours 24 --support-hours 6 \
  --loaded-hourly-cost-usd 100 --compute-usd 200 --other-direct-usd 100 \
  --monthly-fixed-usd 3000 --monthly-available-hours 120
```

These **hypothetical** inputs produce $3,300 direct cost, $1,700 contribution
per pilot and a 34% contribution margin. Two identical pilots cover $3,000
monthly fixed cost. The specified team can deliver four, producing a modeled
$3,800 monthly result at full utilization if sufficient demand exists.

Use a fully loaded labor rate that reflects the delivery team's cost. Include
review, handover and expected follow-up in the hours. Count customer-specific
licensing, storage and similar expenses in other direct costs. Fixed overhead
must exclude anything already in the labor rate or direct cost fields.

`contribution = fee − labor − compute − other direct cost`

`capacity = floor(available hours / (delivery hours + support hours))`

`break-even volume = ceil(monthly fixed cost / contribution)`

Break-even is `null` for zero or negative contribution. A profitable unit can
still fail to cover overhead when required volume exceeds team capacity.
The monthly result is contribution at that capacity minus fixed cost; it is
not net profit after taxes, financing, acquisition spending or omitted costs.
It assumes identical scopes and no other work competing for those hours.

Before quoting, run a pessimistic scenario with twice the delivery effort and
half the available hours, then replace assumptions with actual pilot records.
Do not use the illustrative fee as evidence of willingness to pay. Retain the
JSON inputs with the estimate so another person can reproduce the calculation.
Values are USD, not foreign-exchange conversions. Monetary display rounds to
cents using Decimal's default half-even rule; break-even uses unrounded values.
Inputs are bounded to one billion with at most four decimal places.

After work starts, use the [delivery review](PILOT_DELIVERY_REVIEW.md) to compare
this saved JSON estimate with recorded effort and costs. Preserve the original
plan and keep incomplete support windows marked provisional.
