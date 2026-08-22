# SPH external transport R2: publication-safe aggregate snapshot

This directory is a deliberately restricted, publication-safe snapshot of the
completed SPH exploratory external transport stress test. It contains only
aggregate reports copied byte-for-byte from the audited public output of
`runs/external_transport/sph_figshare_v1__attempt-r2/public/`.

The frozen PTB-XL models were evaluated on SPH without tuning or recalibration
on SPH. These results are research only, are not clinical validation, and must
not be used for diagnosis, treatment, triage, or other clinical decisions.

## Provenance bindings

- Frozen R2 protocol file SHA-256:
  `840acb758d50dbf1a04bf704b16a58d4d29d668370ce7ef91ef7a44860bf311b`
- Source-inventory manifest file SHA-256:
  `91fc22007f312ee1f37b7e20f8cd9f501287a788470b80739f67659982065bbe`
- Source-inventory canonical artifact SHA-256:
  `fef1bb20c24de2764117792cc5ed261bfb67c79b0d75e112c87950e6c228ba64`
- Excluded run public-manifest file SHA-256:
  `bf0eee05a49de54119bd920ca41c60feab562330efba9381ee78869d34f6b476`
- Excluded run public-manifest canonical artifact SHA-256:
  `eb333e255f41beece3cd5fa413e9a605c017bf63b0c977207c28e5ac1373fc0f`

The source-inventory manifest binds the SPH metadata
(`c6f8dc5197758a3970aae554f8b6ee884d96656b121879bc0eb1adf367ea6d86`),
diagnostic dictionary
(`4d22759fd19d37133c78d439a5726f9e0bc412cc17483c424926ce9abb33c8fc`),
translation rules
(`3991b608fa1f9ab470a934a6437352ebd0080c9dd066dc5085cb155208dfec68`),
records archive
(`cebad89d7d25663272eeb45545712d76302d334bb107cdc23fd15390c9194e55`),
and extracted waveform content tree
(`b4382ef49d0ee9cccb7f8a372023815ad31932cb9497152ba8115263b8f5159d`).

## Included boundary

The 35 source artifacts in this snapshot are exactly:

- `FINAL_RESULTS.md` and `cohort_summary.json`;
- six JSON architecture summaries;
- eighteen JSON member-level aggregate metric reports; and
- nine JSON paired patient-cluster bootstrap aggregate reports.

`SHA256SUMS.txt` covers those 35 copied artifacts plus this README. Paths are
sorted and use forward slashes; the checksum file itself is intentionally not
self-listed.

## Excluded boundary

This snapshot excludes the run public manifest, signal-QC files, all member
prediction files, all NumPy archives and aligned arrays, the entire private run
directory, all row-level identifiers, alignment tables, inference
checkpoints, raw ECG waveforms, source data, and model weights. Nothing outside
the explicit 35-file aggregate whitelist was copied from the run.
