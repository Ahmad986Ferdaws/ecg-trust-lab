# Trust Sentinel external OOD v2 pre-inference termination

**Protocol:** `trust-sentinel-ood-external-v2-parent`  
**Status:** `PRE_INFERENCE_PROTOCOL_INFEASIBLE`  
**Stopped:** 2026-08-29 18:22:31 UTC, before waveform decoding or model access

The original v2 parent remains an immutable, useful scientific record, but it
cannot be executed against its selected ZZU source as written. The frozen
canonicalization rule requires the exact names `aVR`, `aVL`, and `aVF` and
forbids relabeling. Metadata-only WFDB header inspection established that the
selected ZZU records instead use the exact spellings `AVR`, `AVL`, and `AVF`.
Treating those names as equivalent would add a post-freeze preprocessing rule.

The run was therefore stopped before its permanent one-shot claim, output
directory, or armed marker was created. No waveform sample was decoded; no
quality policy, model, embedding, distribution score, probability, endpoint,
or subgroup analysis was run. The only observed information was explicitly
preclaim-allowed metadata: release and license information, archive and file
checksums, WFDB headers, lead and duration fields, Challenge quality lists,
ZZU patient-role metadata, and ordered file identities.

The metadata inventory contained 1,000 Challenge Set A records and 12,328
eligible ZZU records from 10,350 patients. Its private file SHA-256 is
`01b33b992c3e9a777eb253571f35907e8aa99e3d36b34a19a3c374e7732aef13`;
the aggregate-only public projection file SHA-256 is
`8de32fce76e73fd00958878e74250c9117cdae7e3f4d9f453f2231fcf16a5814`.
These artifacts are feasibility evidence, not model results.

The original parent, claim path, and output path will never be reused. A
separately versioned v2.1 successor may proceed only after it transparently
freezes the exact dataset-specific mapping `AVR→aVR`, `AVL→aVL`, and
`AVF→aVF`, preserves both raw and canonical names, uses a fresh inventory and
claim namespace, and passes a new pre-inference audit. This correction changes
names only; it must never alter, infer, swap, sign-flip, or rescale a signal.

Machine-readable termination details are in
[`configs/trust_sentinel_ood_external_v2_termination.yaml`](../configs/trust_sentinel_ood_external_v2_termination.yaml).
