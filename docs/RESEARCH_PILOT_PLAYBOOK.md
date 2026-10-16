# Research customer discovery and pilot playbook

This proposed offer helps an ECG machine-learning team reproduce a model
comparison and review failure behavior. It is a research evaluation engagement;
patient-care deployment is outside its scope. The template is ready for owner
review, but no customer, contract, price or delivery date is committed.

## Qualify the problem before building integrations

Start with research leads at academic ECG groups and small ECG model developers.
The daily user is likely an ML engineer; the payer may be a principal
investigator or engineering lead. These are hypotheses to test, not an existing
customer list. No outreach has been performed by writing this playbook.

For each conversation, ask for a recent concrete example:

1. Describe the last model evaluation that another person could not reproduce.
   What broke, who repaired it, and how much time did it take?
2. Which calibration, abstention, subgroup and data-shift checks do you run now?
   Which decisions change when those checks fail?
3. Who approves a model release or a paper's quantitative claims? What evidence
   do they request, and what do existing scripts already do well?
4. Which environment, formats and data-access restrictions must a tool support?
   Can evaluation remain inside your controlled environment?
5. Who could approve a bounded evaluation purchase? What would justify spending
   money instead of improving your existing scripts?
6. What would make the pilot a failure even if the report were technically correct?

Record consented notes privately: role, problem recency, current workaround,
time/cost estimate and its basis, budget owner, constraints, contradictory
evidence, next step and date. Publish only permissioned aggregate findings.
Do not store contacts, health records or commercial terms in this public repo.

## Proposed discovery decision rule

Run ten qualified interviews across at least three independent organizations.
Proceed to a pilot proposal only if three describe a recurring evaluation problem,
two identify a plausible budget owner, and one agrees to review a concrete paid
scope. These counts are managerial test criteria, not statistically representative
market evidence. Record the recruitment method and do not replace negative
responses just to reach the threshold.

If the gate fails, narrow the job or stop. If the only demand is for diagnostic
interpretation, treat that as a different product requiring different evidence.

## Bounded pilot scope

| Item | Proposed agreement field |
|---|---|
| Purpose | Reproduce and audit one research comparison; name the decision it informs |
| Inputs | One agreed task and label ontology, up to two frozen models, a specified authorized dataset or predictions |
| Work location | Customer-controlled environment by default; any transfer needs a separate agreed data arrangement |
| Protocol | Freeze cohort roles, estimands, metrics, seeds and acceptance criteria before outcome inspection |
| Deliverables | Reproduction instructions, version/hash manifest, agreed aggregate metrics, failure analysis and limitations, handover session |
| Explicit exclusions | Training sweep, new clinical validation, clinical diagnosis, 24/7 hosting, unbounded adapters, access to spent one-shot cohorts |
| Duration and fee | Fill after access/setup discovery and cost estimation; all figures in a proposal must be marked estimates until agreed |
| Change control | Record a scope change, cost impact and decision before adding another model, task, dataset or deployment |
| Owners | Named customer data custodian and technical reviewer; named delivery owner |

Confirm provenance, input permissions and protocol roles before processing
customer inputs. The repository's [reproducibility guide](REPRODUCIBILITY.md)
and [scientific boundary](../CONTRIBUTING.md#respect-the-scientific-boundary)
remain applicable. Do not reinterpret an earlier sealed cohort as new pilot data.

## Acceptance rubric

Agree tolerances before running the comparison. Pilot completion does not depend
on a favorable model result.

| Acceptance question | Evidence to retain |
|---|---|
| Can the customer repeat the report? | A second operator's run, exact environment, input hashes, commands and expected tolerances |
| Are scientific roles respected? | Recorded split manifest, patient-isolation checks and no held-out tuning |
| Are claims traceable? | Each conclusion linked to an aggregate table and the stated analysis scope |
| Are failures understandable? | Examples using synthetic or approved material, refused inputs and reason codes |
| Is the work operationally useful? | Customer reviewer identifies whether it changed a concrete decision; a negative answer is retained |
| Is delivery economically repeatable? | Actual setup, engineering, review, support and compute costs compared with the estimate |

Use an unsigned acceptance worksheet first. Actual acceptance requires the
customer reviewer, date and supporting evidence. No auto-generated sign-off.

## Measure value without inventing savings

Measure time from an authorized, complete input handoff to an accepted report.
Separate waiting for customer access from engineering work. Compare with the
customer's documented prior workflow on a comparable task. Record hours by role,
manual corrections, repeat runs, and reviewer questions. Do not count repository
test totals, PR totals or GitHub stars as customer value.

A usable value estimate is:

`documented prior effort − pilot effort on a comparable scope`

It remains an estimate if prior effort was recalled rather than measured.
Do not turn engineering hours saved into healthcare savings or patient benefit.

## Renewal and stop criteria

Offer a second engagement only after the first is reproducible and both sides
can name a repeat use case. A subscription requires recurring demand and a
support capacity plan; it should not be assumed from a one-off report.

Stop or renegotiate if input rights are unresolved, the requested claims exceed
the evidence, delivery effort exceeds the agreed scope, or the customer needs an
unsupported clinical workflow. Preserve what was learned in an internal pilot
retrospective, including rejected assumptions and actual delivery costs.
