# Counterfactual evaluation under one verified release

`VerifiedCounterfactualEvaluator` connects the existing proposal evaluator to the
complete `TrustSentinelEngine`. It requires a separately retained, unprefixed
TrustBundle manifest SHA-256. The same verified artifact identities must hold
before and after analysis of both the original and candidate waveforms.

```python
from datetime import UTC, datetime

from ecg_trust.counterfactual.verified_release import VerifiedCounterfactualEvaluator

# active_engine was constructed with TrustSentinelEngine.from_verified_bundle.
evaluator = VerifiedCounterfactualEvaluator(
    lambda: active_engine,
    expected_manifest_sha256=approved_manifest_digest,
)
result = evaluator.evaluate(
    proposal,
    original_signal_mv,
    candidate_signal_mv,
    evaluated_at=datetime.now(UTC),
)
public_json = result.canonical_public_json()
```

The provider can resolve the application's active engine. If it switches releases
between the two calls, the evaluation becomes `evidence_unavailable`. Changes to
checkpoint identities, decision-policy/calibration parents, conformal calibration,
normalization, and the other bound parents are covered by the comparison. Parent
files are rehashed at each boundary, and the loaded model's advertised manifest
and checkpoint bindings are checked as well.

Every inference still runs the existing physical-input, quality, distribution,
entropy, and conformal gates. The adapter uses canonical signal metadata and the
engine's quality pass as its artifact-quality gate. It does not add an independent
artifact detector, alter any threshold, or fit any component on the proposal.

The new `counterfactual-release-validation-summary-v1` envelope contains path-free
release and parent digests, a provenance status, closed failure reasons, and the
unchanged v1 counterfactual public summary. Failed provenance never exposes
target scores or a sensitivity effect. `verified` describes release provenance;
the nested scientific evaluation may still reject the proposal. `not_evaluated`
means input validation prevented both full analyses from running.

The existing evaluator and serialized scientific artifacts are unchanged.
Trusted component loaders remain responsible for loading the advertised files
correctly and keeping their in-memory model objects immutable. These checks do
not attest arbitrary model RAM changes that leave every advertised identity
unchanged. Results remain model sensitivity evidence: not physiological truth,
not causal evidence, and not treatment advice.
