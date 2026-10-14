# Commercial strategy and evidence

Reviewed against repository commit `4884f750` on 2026-10-11.
This is a proposed business direction for owner review, not a sales commitment.

## Recommended first product

Offer a reproducible ECG model evaluation toolkit and a bounded research
evaluation service to teams developing ECG machine learning. Help a research
lead answer: **Can another engineer reproduce our model comparison, identify
when predictions should be withheld, and trace the evidence behind a release?**

The repository already contains calibration, abstention, subgroup analysis,
controlled-corruption probes, provenance checks, and publication artifacts.
Those are a credible starting point for that job. Demand, willingness to pay,
delivery cost, and repeatability across customers remain untested hypotheses.
No customers, revenue, patents, clinical partnerships, or market share are
established by the repository.

## Compare the possible businesses

| Direction | Buyer and job | Available foundation | Missing evidence | Decision |
|---|---|---|---|---|
| Research evaluation toolkit and services | ECG ML lead who needs an auditable comparison or release report | Evaluation modules, integrity gates, test suite, aggregate reports | Independent installation, supported input adapters, time saved, paid demand | Test first |
| Clinical interpretation product | Clinical service buying a patient-care workflow | Retrospective superclass benchmark | Intended-use validation, clinical partners, workflow study, regulatory and quality-system work, support operations | Separate future program |
| Training and reproducibility workshops | Research group learning robust ECG evaluation | Protocols, public aggregates, synthetic tests | Teaching outcomes and budget owner | Small parallel discovery experiment |
| Sale of project assets | Acquirer evaluating code and know-how | Source, documentation, retained research history | Rights chain, transferable assets, reproducible handover, commercial traction | Prepare diligence; no valuation yet |

The preferred direction is an inference from the current capabilities and gaps,
not a market-size estimate. Stop pursuing it if qualified teams already have an
adequate workflow and cannot identify budget for the proposed improvement.

## Competitive context

Official product pages reviewed on 2026-10-11 establish adjacent offerings,
not an independent comparison of performance.

| Alternative | What its provider offers | Implication for ECG Trust Lab |
|---|---|---|
| [PMcardio](https://www.powerfulmedical.com/pmcardio-individuals/) | ECG interpretation for clinical users; availability and device authorization vary by module and region | Do not compete using a superclass AUROC claim. Test whether model developers need evaluation infrastructure |
| [Philips Cardiologs](https://cardiologs.com/) | AI-assisted ECG/Holter analysis and clinician workflow | Workflow integration and service delivery matter beyond a classifier |
| [PTB-XL benchmark code and research](https://arxiv.org/abs/2004.13701) | Public benchmark methodology for ECG algorithms | Public data and a model comparison alone are unlikely to create exclusive differentiation |
| A team's own scripts | Customer-specific evaluation under existing infrastructure | The relevant comparator in discovery is measured setup/review effort and error detection, not a claimed accuracy advantage |

Potential differentiation is traceable evidence and reproducible assurance
workflows. It becomes defensible only if adopters repeatedly use it, integrations
are maintainable, and evaluation quality is independently demonstrated. The
repository currently supplies no evidence of a commercial moat.

## What is actually evidenced

| Claim usable in a research discussion | Evidence and necessary qualifier |
|---|---|
| Capacity-matched models were compared using patient-isolated folds | [Model card](MODEL_CARD_PTBXL_SUPERCLASS_R3.md); [DEV-001](../reports/PROTOCOL_DEVIATIONS.md) prevents a completely blind-test claim |
| ResNet had higher sealed macro AUROC in this experiment | [Public results](../reports/FINAL_RESULTS_PUBLIC.md); AUROC is not percentage accuracy or clinical utility |
| A separate retrospective transport stress test exists | [SPH result](../publication/external_transport_sph_r2/FINAL_RESULTS.md); ontology mapping was not clinically adjudicated |
| The project records adverse findings | [Known limitations](KNOWN_LIMITATIONS.md); OOD source-support gate missed, older-age retention unequal |
| Engineering controls are implemented and tested | [Contribution guide](../CONTRIBUTING.md); report the actual revision, environment and skips for each run |

Do not translate those claims into diagnostic safety, prospective benefit,
OOD-detection validity, regulatory authorization, cost savings, or superiority
to a commercial device. Those require their own evidence.

## Diligence before licensing or selling assets

At the reviewed commit there is no root code license. A public repository does
not by itself grant general reuse rights; GitHub distinguishes public viewing
and forking from licensing. The owner must establish contributor and third-party
rights before selecting a code license or negotiating transfer.
[GitHub licensing guidance](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository).

| Asset | Current position | Evidence needed for a buyer |
|---|---|---|
| Original code and documentation | Public source; no root license at reviewed commit | Authorship/contribution inventory, assignments or permissions where applicable, explicit license decision |
| Dataset material | PTB-XL attribution is documented separately | [Data card](DATA_CARD.md), source terms, attribution inventory for derived figures |
| Trained weights and row-level artifacts | Excluded from a source-only clone | Owner's transfer rights, provenance, reproducible generation and approved transfer channel |
| Dependencies and bundled assets | Declared in Python and UI manifests | Versioned dependency inventory and license review for the actual distributed package |
| Customer/clinical evidence | Not established here | Signed agreements and completed study evidence if later obtained |

This document changes no license and promises no right to distribute data or
weights. Before a transaction, obtain qualified legal review of the actual rights
and proposed agreement. Do not assign a valuation from commit or PR counts.

## Research and clinical product scope

FDA's [January 2026 CDS guidance](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/clinical-decision-support-software)
and [signal-analysis navigator](https://www.fda.gov/medical-devices/digital-health-center-excellence/step-6-software-function-intended-provide-clinical-decision-support)
make intended use and signal analysis relevant to device status. Calling software
research-only is not a substitute for assessing its actual claims and use.

A future US clinical product needs its own classification and evidence plan.
The [QMSR](https://www.fda.gov/medical-devices/postmarket-requirements-devices/quality-management-system-regulation-qmsr)
took effect on February 2, 2026 and incorporates ISO 13485:2016 by reference.
This repository does not establish compliance, certification or authorization.
Other jurisdictions require separate assessment; this is planning context,
not a legal determination for a proposed product.

## Investment milestones

1. **Reproducibility:** an independent engineer installs the toolkit and produces
   a synthetic report without private weights. Record setup time and failures.
2. **Problem evidence:** qualified research teams describe a recent costly
   evaluation failure and identify the budget owner. Preserve contrary evidence.
3. **Pilot evidence:** a scoped pilot delivers a reproducible report under an
   agreed acceptance rubric, with measured engineering hours and support effort.
4. **Repeatability:** a second team completes the same workflow with limited
   customization. Measure reuse and contribution margin before projecting growth.
5. **Expansion:** decide whether to remain a research tooling business or fund
   a separately governed clinical program with suitable partners.

Suggested owners are the maintainer for technical delivery, a designated
commercial lead for discovery, and qualified advisors for legal/regulatory work.
These are roles to assign, not claims that a team or advisory board exists.
