# Compare pilot delivery with its estimate

The [pilot estimator](PILOT_ECONOMICS.md) models a proposed scope. After work
starts, `scripts.review_pilot_delivery` compares that saved estimate with a
local record of work and direct costs. This helps the delivery owner see where
an engagement exceeded its assumptions before quoting another similar scope.
It does not verify invoices, payments, customer acceptance or repeat demand.

## Try the synthetic example

From the repository root, with Python 3.12, save a plan to a private scratch
location and compare it with the explicitly synthetic example:

```bash
python scripts/estimate_research_pilot.py \
  --fee-usd 5000 --delivery-hours 24 --support-hours 6 \
  --loaded-hourly-cost-usd 100 --compute-usd 200 --other-direct-usd 100 \
  --monthly-fixed-usd 3000 --monthly-available-hours 120 > /tmp/pilot-plan.json

python -m scripts.review_pilot_delivery \
  --plan /tmp/pilot-plan.json \
  --actuals docs/examples/pilot-actuals.synthetic.json
```

The example records 40 hours and $4,600 direct cost against 30 hours and $3,300
planned. It reports a 10-hour overrun, $1,300 extra cost and $400 contribution
against $1,700 planned. Its support window is still open, so the output is
`provisional`; these invented numbers establish no actual customer outcome.
The command writes JSON only to stdout. Redirect it to an appropriate private
location if you need a retained report.

## Record a real delivery window

Copy the [example schema](examples/pilot-actuals.synthetic.json) into private
storage. Change `basis` to `operator_record` only for a record grounded in actual
work logs and cost records. Keep those supporting records separately; this
calculator does not inspect or independently verify them.

- `work` accepts 1–250 entries with `category`, `hours` and
  `loaded_hourly_cost_usd`. Categories are `setup`, `engineering`, `review`,
  `handover` and `support`. Multiple entries in one category allow different
  rates. Include rework and review; exclude waiting time without labor.
- `fee_usd` is the fee used for this scope's arithmetic. It does not mean cash
  was collected or revenue was recognized. A free pilot may use zero; its
  contribution margin percentage is then `null`.
- `compute_usd` and `other_direct_usd` are recorded direct costs. Include the
  same cost categories as the plan, without counting labor or expenses twice.
- `support_window_closed` stays `false` while agreed follow-up is outstanding.
  Set it to `true` only after the defined delivery/support window ends and its
  costs are entered. The resulting `recorded_window_closed` status is not an
  acceptance, audit or commercial-success designation.

Decimal values are strings, in USD, within [0, 1,000,000,000] and at most four
decimal places. Inputs are at most 64 KiB each. Unknown fields, duplicate JSON
keys, non-finite values and unsupported schemas are refused. There are no
customer, contact, patient or free-text fields. Real rates, fees and operational
records can still be commercially sensitive; keep them out of public commits.
These limits are input contracts, not a privacy audit.

## Interpret the variance

`variance = recorded − planned`. Positive hours and direct-cost variances are
overruns. A negative contribution variance means the recorded contribution is
below the plan. Percent variance is `null` when its planned denominator is zero.
Sums retain Decimal precision until monetary/percentage display rounds to two
places using half-even rounding; rounded category amounts may not add exactly
to rounded totals. The report retains the bounded input values for reproduction.

The complete plan export is checked against a fresh calculation of its inputs;
stale or manually edited derived figures are refused. To model an agreed scope
change, generate a separate revised estimate and retain the original. Comparing
with a revised scope answers a different question from measuring the initial
overrun. Preserve both rather than hiding the original assumptions.

Contribution excludes monthly fixed overhead, taxes and other omitted costs;
it is not net profit. This review deliberately produces no extrapolated sales,
capacity forecast or automatic renewal recommendation. Repeat use, independently
reproduced deliverables and customer decisions need their own evidence.
