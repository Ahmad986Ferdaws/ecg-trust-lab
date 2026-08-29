# Trust Sentinel private evidence backup

**Status:** encrypted off-device backup verified on 2026-08-29

**Scope:** sealed `trust-sentinel-ood-completion-v1` evidence only

**Access:** private project owner and explicitly authorized repository collaborators

The permanent one-shot claim and the complete sealed v1 evidence bundle have a
disaster-recovery copy outside this workstation. The archive contains private
embedding and identity-alignment artifacts, so it is deliberately absent from
Git history, the public evidence snapshot, and the visual results site.

## Off-device object

- Private GitHub release:
  [`private-evidence-backup-v1-2026-08-29`](https://github.com/Ahmad986Ferdaws/ecg-trust-lab/releases/tag/private-evidence-backup-v1-2026-08-29)
- Asset: `ood_completion_v1-2026-08-29-encrypted-backup.7z`
- Size: `36,397,121` bytes
- Archive SHA-256:
  `44849987cf92e364ae6ccddfaf236018087d4f2f5697212bb1a19cbfb8fb3da3`
- Protection: 7z AES-256 with encrypted filenames and headers
- Recovery-key policy: the key is not stored in Git, GitHub Releases, project
  documentation, the site, or the archive itself; the project owner must keep
  the separately delivered key in a password manager.

GitHub's remote asset metadata reports the same SHA-256 and byte length as the
local ciphertext. Only encrypted bytes were uploaded.

## Restore verification

A fresh copy was downloaded from the private release, hashed, extracted into a
new disposable directory, and passed the repository's whole-bundle verifier.
The restored logical identities were:

- result:
  `sha256:dd76258b30c95a3ac8f865da54973a42a93d0135caba09da0c6412267f041b53`;
- success manifest:
  `sha256:6f97e0697d661372e62f4aee9245f26014312e6a1d681615314bc9fcb77c5732`;
- terminal status: `SOURCE_SUPPORT_GATE_TARGET_MISSED`.

The downloaded verification copy and extracted plaintext were removed after
verification. The original immutable evidence and its local encrypted backup
remain under the ignored `artifacts/trust_sentinel/` tree.

## Recovery rule

Download the asset only from the private release, compare its SHA-256 before
decryption, extract it into a new private directory, and run
`verify_ood_completion_bundle(...)` against the restored
`ood_completion_v1` directory. A restore is invalid if the archive hash, result
identity, manifest identity, adjacent one-shot claim, or exact bundle tree
differs from the values above.

Never publish the recovery key, decrypted archive, embeddings, identifiers, or
row-level artifacts.
