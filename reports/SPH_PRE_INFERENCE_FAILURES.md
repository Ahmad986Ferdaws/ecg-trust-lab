# SPH pre-inference operational launch record

This tracked report preserves the sanitized provenance of two operational
launches of the original `sph-external-transport-v1` runner. Neither launch
accessed an SPH waveform through a model, produced an SPH prediction, computed
an SPH metric, or produced an SPH result.

## Launch 1: elevated user stopped by Git ownership protection

- Execution Git revision: `724a510b03eb539eda2add6de359855f3ffaf2b5`
- Execution context: elevated user
- Failure phase: clean-Git preflight, at `git rev-parse HEAD`
- Failure: Git returned exit 128 because repository ownership was considered
  dubious.
- Work reached: no metadata validation, archive audit, runtime loading, output
  reservation, SPH data access, or SPH inference.
- Output root: no official attempt root was acquired.

## Launch 2: sandbox owner stopped by attempt-marker serialization

- Execution Git revision: `724a510b03eb539eda2add6de359855f3ffaf2b5`
- Execution context: repository sandbox owner
- Completed preflight work: metadata/cohort validation, complete archive versus
  extracted-file byte audit, sealed exact-six runtime loading, and PTB fold-10
  clean-equivalence inference.
- Failure phase: pre-SPH-inference attempt reservation payload serialization.
- Failure: the top-level source inventory was an immutable mapping; the generic
  canonical JSON hasher did not recursively convert that mapping to a plain
  JSON object.
- SPH state: no attempt-start marker, no SPH model inference, no SPH
  predictions, and no SPH metrics or results.
- Failed root initial state: only an empty `private/` directory.

The operator appended a local-private post-failure receipt after the exception
because the v1 runner had not emitted one. Its initially broad inference wording
was corrected before the r2 freeze to distinguish the PTB clean-equivalence
inference that did run from SPH inference that did not run. No runner-generated
file was overwritten or deleted. The authoritative receipt is:

- Local path: `runs/external_transport/sph_figshare_v1/private/PRE_INFERENCE_FAILURE.md`
- Size: 874 bytes
- SHA-256: `26cbd6f8a1679faf1ec362712aa63334dd81904666ca4c9de8d8aa5013840a74`

The original v1 YAML remains unchanged at
`configs/external_transport_sph_frozen.yaml`, SHA-256
`6beebf9fc95c90591d5f63b3985ff14fbff403a75862b5690201b1c7d6ff2669`.
The corrected runner is governed by the distinct r2 protocol and output root;
the r2 scientific-projection hash proves that no source, label, signal, cohort,
model, runtime, evaluation, bootstrap, privacy, or reporting rule changed.
