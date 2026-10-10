# Start here

ECG Trust Lab contains three related surfaces: published study evidence, an
interactive results notebook, and Python research infrastructure. Choose the
path that matches what you want to do.

| Your goal | Start with | Needs private data or model weights? |
| --- | --- | --- |
| Understand the project and findings | [Portfolio case study](PORTFOLIO_CASE_STUDY.md) | No |
| Run the interactive results notebook | [Website guide](../results-experience/README.md) | No |
| Check the numbers displayed on the website | Public evidence command below | No |
| Review the completed classifier study | [Public PTB-XL results](../reports/FINAL_RESULTS_PUBLIC.md) and [model card](MODEL_CARD_PTBXL_SUPERCLASS_R3.md) | No |
| Review external transport | [SPH results](../publication/external_transport_sph_r2/FINAL_RESULTS.md) and [frozen protocol](EXTERNAL_TRANSPORT_SPH_R2.md) | No |
| Understand the support gate that missed its target | [Source-support completion result](TRUST_SENTINEL_OOD_COMPLETION_RESULT.md) | No |
| Improve or test the Python code | [Contribution workflow](../CONTRIBUTING.md) | Most CPU tests run without private artifacts |
| Reconstruct the scientific environment | [Reproducibility guide](REPRODUCIBILITY.md) and [environment record](ENVIRONMENT.md) | Yes, for waveform/model execution |

## Open the results notebook

With Node.js 22.13+ and pnpm 11.19.0, run from the repository root:

```bash
cd results-experience
pnpm install --frozen-lockfile
pnpm dev
```

Open the local address printed by the server (normally
`http://localhost:3000`). This notebook includes aggregate study results and
synthetic teaching waveforms. It runs independently of the Python inference
service. Its setup does not require a CUDA GPU, patient records, or checkpoints.

## Check public evidence without installing packages

From the repository root, with Node.js 22.13+:

```bash
node --experimental-strip-types --test results-experience/tests/evidence-parity.test.mjs
```

This compares displayed values with the committed public aggregate artifacts
and verifies their checksum inventories. It does not recompute patient-level
results. The [website guide](../results-experience/README.md) explains the
additional production build and rendered-HTML checks.

## Choose the right Python environment

For code contributions, use the disposable CPU environment documented in
[CONTRIBUTING.md](../CONTRIBUTING.md). It bypasses the scientific CUDA package
source without rewriting `pyproject.toml` or `uv.lock`. Subsequent test commands
use `uv run --no-sync` so they keep that environment intact.

For historical scientific reconstruction, follow
[REPRODUCIBILITY.md](REPRODUCIBILITY.md). `ecg-verify` deliberately requires a
working CUDA training stack; it is not a source-only installation check. A
clone does not include raw data, trained weights, or private evaluation trees.
Expected test skips identify unavailable private evidence or platform-specific
contracts and must remain visible in verification reports.

## Understand the research status

| Research line | Recorded state | Interpretation |
| --- | --- | --- |
| PTB-XL r3 | Completed, sealed evaluation and descriptive audits | Retrospective five-superclass evidence; [DEV-001](../reports/PROTOCOL_DEVIATIONS.md) records a label-blindness deviation |
| SPH r2 | Completed, frozen external transport | Exploratory transport without target-domain adaptation; the ontology mapping was not clinically adjudicated |
| Sentinel source calibration | Development preparation completed | Its original unfamiliar-input component remains pending in that immutable artifact |
| Separate OOD completion v1 | One-shot source-support evaluation completed; target missed | The bundle is not research-eligible; no OOD-positive cohort was evaluated |
| Sentinel vNext | Research infrastructure under development | [Five-state policy and optional methods](TRUST_SENTINEL_VNEXT.md) require their own release and study contracts |

The recorded Plotly walkthrough is the historical entropy-gated demo. It is
separate from both the results notebook and the Sentinel service. Completed
engineering tests establish code behavior, not clinical validity. The project
is for retrospective research and education, not patient care.

## Find the code

| Area | Entry point |
| --- | --- |
| Dataset and normalization | [`src/ecg_trust/data/`](../src/ecg_trust/data/) |
| Training, calibration, evaluation | [`training.py`](../src/ecg_trust/training.py), [`evaluation.py`](../src/ecg_trust/evaluation.py) |
| Input quality and uncertainty | [`quality/`](../src/ecg_trust/quality/), [`conformal/`](../src/ecg_trust/conformal/), [`open_world/`](../src/ecg_trust/open_world/) |
| Sentinel decision and service | [`trust_policy.py`](../src/ecg_trust/trust_policy.py), [`service/`](../src/ecg_trust/service/) |
| Experiment and operational commands | [`scripts/`](../scripts/) |
| Website and public metric model | [`results-experience/`](../results-experience/), [`lib/results.ts`](../results-experience/lib/results.ts) |
| Current engineering gates | [CPU CI](../.github/workflows/cpu-quality.yml), [website CI](../.github/workflows/results-experience-quality.yml), [public evidence CI](../.github/workflows/public-evidence.yml) |

Test totals in historical reports describe the revision and environment that
produced those reports. Use the checks on the commit you are reviewing for its
current engineering status.
