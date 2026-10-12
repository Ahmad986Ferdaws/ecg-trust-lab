# Technical diligence and handover

This evidence index helps a research partner or prospective acquirer assess the
software they would receive. It is a technical review aid, not an assurance
certificate, purchase agreement, license grant or supported-service offer.
Repository observations below were checked against `4884f750` on 2026-10-11;
recheck them at the proposed handover commit.

## What a reviewer can establish now

| Buyer question | Inspect or exercise | What the evidence does not establish |
|---|---|---|
| Can another engineer install and test the code? | [CPU workflow and wheel check](../CONTRIBUTING.md), [CI definition](../.github/workflows/cpu-quality.yml) | Historical passing CI is not a test of the buyer's environment; private-data and platform skips remain visible |
| What is included in the source delivery? | [Repository contents](../README.md#repository-contents), tracked files, wheel contents | A clone contains no private waveforms, checkpoints or generated evaluation trees |
| Can inference be reproduced? | [Reproducibility guide](REPRODUCIBILITY.md), source and checkpoint identities | Public aggregate artifacts alone cannot reproduce patient-level inference |
| Are dependencies and reuse rights cleared? | [Python manifest](../pyproject.toml), [UI manifest](../results-experience/package.json), actual installed distribution metadata | There is no root code license at the reviewed commit; dependency and dataset rights need separate review |
| What can the model claims support? | [Model card](MODEL_CARD_PTBXL_SUPERCLASS_R3.md), [known limitations](KNOWN_LIMITATIONS.md), [protocol deviations](../reports/PROTOCOL_DEVIATIONS.md) | Retrospective results do not establish clinical validity, unknown-disease detection or prospective benefit |
| Is the service ready for external users? | [Security policy](../SECURITY.md), [demo startup](../scripts/demo_server.py), [Sentinel service](../src/ecg_trust/service/sentinel_service.py) | Network service support and application authentication are not supplied; a service factory is not a complete hosted product |
| What happens under load or restart? | [Analysis capacity](SENTINEL_ANALYSIS_CAPACITY.md), [idempotency concurrency tests](../tests/unit/test_sentinel_idempotency_concurrency.py) | Active-analysis limits and replay state are per process; no cross-worker or restart-persistent replay guarantee |
| Are artifact integrity and logging covered? | [Runtime binding](../src/ecg_trust/runtime_binding.py), [audit ledger](../src/ecg_trust/audit_log/audit_ledger.py), [offline passport](OFFLINE_MODEL_PASSPORT.md) | Hash checks are not an independent assessment of source trust, clinical validity, tenant isolation, backup recovery or retention operations |
| What support exists? | [Security handling](../SECURITY.md#handling), [contribution process](../CONTRIBUTING.md) | No response-time SLA, uptime commitment, staffed on-call rota or paid support contract is established |

Use the existing research surfaces in their documented scope. The local demo
defaults to loopback and has no authentication. The results notebook presents
committed aggregate evidence and synthetic teaching material; it is separate
from an inference deployment. A network-hosted or multi-customer service would
need a separately designed and reviewed operational boundary.

## A concrete handover exercise

Choose a recipient engineer who did not prepare the environment. Agree the
exercise scope and maximum support effort before starting. The recipient should:

1. Obtain the exact agreed commit, record its identity, and install the CPU
   environment from the contribution guide. Retain the actual package inventory.
2. Run relevant tests and the isolated wheel smoke check. Preserve failures,
   skips and warnings; record any assistance needed to complete the instructions.
3. Run the public evidence parity command in the contribution guide and explain
   which committed values it checks. It verifies displayed aggregates, not the
   patient-level study computation.
4. Demonstrate one synthetic quality fault and its refusal using
   `scripts/quality_gate_stress_report.py`. Explain the difference between a
   passing software check and a validated research release.
5. Identify the missing private artifacts for any agreed real inference step.
   Record their expected identities, authorized custodian and transfer status
   privately. An unavailable artifact is an incomplete handover item.
6. Make a small disposable change, run its checks, and restore the agreed commit
   in that disposable checkout. Do not rerun or rewrite sealed scientific work.

Keep an acceptance record containing the commit, recipient, environment,
commands, evidence locations, blocked items, assistance time and reviewer
decision. Leave reviewer and acceptance fields unset until the real review.
No script or document in this repository signs off a transaction automatically.

## Assign operational responsibilities before promising a pilot

| Responsibility to assign | Evidence to agree and retain privately |
|---|---|
| Technical delivery owner | Supported workflow, named backup, scope-change process and escalation contact |
| Customer data custodian | Authorized inputs, processing location, access list, approved outputs and transfer arrangement |
| Environment owner | Capacity measurement, dependency inventory, patch process and rollback procedure tested on synthetic inputs |
| Evidence reviewer | Acceptance tolerances, scientific exclusions, interpretation of negative results and independently checked identities |
| Storage owner | Retention/deletion plan, restore drill for authorized artifacts, access revocation at handover end |
| Commercial owner | Fee, support hours, exclusions and response expectations explicitly agreed; no implied 24/7 service |

These are proposed roles, not claims that personnel or controls already exist.
Document unanswered items with an owner and a next check; avoid replacing gaps
with a generic security or compliance label. The signed agreement and actual
operational controls would govern any future engagement.

## Research basis

The [Software Evaluation Guide](https://www.software.ac.uk/guide/software-evaluation-guide)
describes complementary criteria-based review and tutorial-based evaluation,
including recording the evaluator's experience and agreeing available support.
The handover exercise above applies that approach to this repository. Its
outcome has not yet been measured with an independent adopter.

CISA's [Secure by Demand guide](https://www.cisa.gov/sites/default/files/2024-08/SecureByDemandGuide_080624_508c.pdf)
distinguishes product-security diligence from a supplier's enterprise security
claims. The evidence table follows that distinction: implemented controls and
tests are named, while unsupported deployment properties remain open. This is
an application of the guidance, not a CISA assessment or endorsement.
