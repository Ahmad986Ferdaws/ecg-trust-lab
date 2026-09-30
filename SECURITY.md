# Security policy

ECG Trust Lab is a research repository. The local demo and Trust Sentinel code
are **not** medical devices, are not authorized for patient care, and are not
supported as network services.

## Supported versions

Only the latest commit on `main` receives fixes. Sealed scientific artifacts
(fold-10 results, the SPH transport run, and completed one-shot protocols) are
never regenerated to resolve a security report; a fix lands as new code with
its own evidence.

## Reporting a vulnerability

Please do **not** put exploit details, credentials, patient data, waveforms, or
private artifacts in a public issue, pull request, or discussion.

1. Open a short public issue titled "Security contact request" that names the
   affected area (for example "demo upload handling") and nothing more.
2. The maintainer will reply with a private channel for the details.

Useful details, once a private channel exists: the affected commit, component,
and entry point; a minimal synthetic reproduction; the expected versus observed
behavior; and the impact you believe it has. Synthetic signals are strongly
preferred over real recordings.

## Scope

In scope:

- The local research demo (`scripts/demo_server.py`, `ecg_trust.demo_app`),
  including upload parsing, HTTP headers, and error disclosure.
- Artifact, checkpoint, manifest, and evidence parsing (path handling, bounded
  reads, schema and hash verification).
- Fail-open behavior in the Trust Sentinel gates, such as malformed input that
  reaches a prediction when it should be refused.
- CI workflows and the results-experience build (token permissions, pinned
  actions, dependency integrity).

Out of scope:

- Deploying the demo on a public network. It binds to `127.0.0.1` by default
  and has no authentication; exposing it is unsupported.
- Clinical accuracy or diagnostic performance. These are scientific
  limitations, documented in the model cards and protocols, not security issues.
- Vulnerabilities in third-party packages without a demonstrated path through
  this project; report those upstream.

## Handling

Reports are acknowledged as capacity allows. Confirmed issues are fixed on
`main` with a regression test and described publicly only after a fix is
available. Private data received in a report is used only to reproduce the
issue and is not committed.
