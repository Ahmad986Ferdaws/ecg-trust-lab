# Verify and export an existing model passport

Use the installed Python environment from [Contributing](../CONTRIBUTING.md).
This command verifies an existing aggregate passport offline, without loading
waveforms, model weights, or running inference. It does not create a passport or
change the status of any release.

Obtain the expected release, TrustBundle, and protocol SHA-256 digests from
independent trusted release records. Each value has the form `sha256:` followed
by 64 lowercase hexadecimal digits. Do not copy expectations from the passport
being checked: matching a document to itself does not establish its identity.

After setting `RELEASE_SHA256`, `BUNDLE_SHA256`, and `PROTOCOL_SHA256` to those
values, run from the repository root:

```bash
uv run --no-sync python -m ecg_trust.passport.cli verify passport.json \
  --expected-release-sha256 "$RELEASE_SHA256" \
  --expected-bundle-sha256 "$BUNDLE_SHA256" \
  --expected-protocol-sha256 "$PROTOCOL_SHA256"
```

Add `--expected-release-id` to check an independently known release identifier.
Success prints a small JSON receipt with `verified: true` and the passport's
self-hash. Exit status is zero on success, one on verification or export failure,
and two for invalid command arguments. Failures do not echo passport contents.

Export repeats verification before writing:

```bash
uv run --no-sync python -m ecg_trust.passport.cli export passport.json \
  --expected-release-sha256 "$RELEASE_SHA256" \
  --expected-bundle-sha256 "$BUNDLE_SHA256" \
  --expected-protocol-sha256 "$PROTOCOL_SHA256" \
  --format markdown --output verified-passport.md
```

Use `--format json` for the canonical JSON representation, byte-for-byte equal
to the accepted input. JSON must already match the existing canonical format,
including its final newline; this command does not repair or reseal input.
Markdown uses the existing escaped renderer and preserves the research-only
notice, identity, limitations, and evidence status.

Input must be a regular file no larger than the schema's 2 MiB limit. Malformed,
noncanonical, tampered, privacy-bearing, or identity-incompatible input fails
before export. The output directory must already exist. Publication is atomic
and refuses any existing destination, including symlinks; there is no overwrite
flag. Filesystems without hard-link support fail closed. Temporary output is
removed on normal write/publication failure.

Verification establishes consistency with the supplied identities and the
[existing passport contract](../src/ecg_trust/passport/models.py). It does not
independently audit the underlying evidence, prove a release is currently
promoted, or establish clinical validation.

The Python JSON serializer and Markdown renderer also revalidate the complete
current object, nested contracts, privacy boundary and self-hash before export.
A frozen Pydantic object is not sufficient proof of validity: updated
[`model_copy` values are not validated](https://pydantic.dev/docs/validation/latest/concepts/models/#model-copy).
Copied or constructed objects that violate the passport contract are refused;
export does not silently reseal them. JSON exports obey the same 2 MiB limit as
the reader, including the final newline. Valid JSON and Markdown output retain
their existing bytes. This content check does not establish independent release
identity: continue supplying trusted expectations to the verification API or CLI.
