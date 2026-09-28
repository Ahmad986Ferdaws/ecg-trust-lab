from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from ecg_trust.conformal import LabelwiseBinaryConformal
from ecg_trust.constants import SUPERCLASSES
from ecg_trust.contracts import TrustDecision
from ecg_trust.counterfactual.counterfactual_review import (
    CounterfactualEvaluator,
    CounterfactualProposal,
    CounterfactualStatus,
    CounterfactualValidationError,
    FloatArray,
    QualityGateStatus,
    SentinelAnalysis,
    TargetDirection,
    TargetSuperclass,
    canonical_waveform_sha256,
)
from ecg_trust.counterfactual.verified_release import (
    ReleaseProvenanceReason,
    ReleaseProvenanceStatus,
    VerifiedCounterfactualEvaluator,
)
from ecg_trust.quality.signal_quality import DEFAULT_SIGNAL_QUALITY_CONFIG, SignalMetadata
from ecg_trust.registry import (
    ArtifactRole,
    TrustBundleCompatibility,
    bind_parent_file,
    seal_trust_bundle,
    verify_trust_bundle,
)
from ecg_trust.sentinel_engine import (
    LoadedDistributionPolicy,
    SentinelModelArtifactInputs,
    SentinelModelEvidence,
    SentinelRuntimeLoaders,
    TrustSentinelEngine,
)
from ecg_trust.trust_policy import DEFAULT_TRUST_POLICY_CONFIG

NOW = datetime(2026, 8, 24, 12, tzinfo=UTC)


def _signals() -> tuple[FloatArray, FloatArray]:
    phase = np.mod(np.arange(1000, dtype=np.float64) / 100.0, 1.0)
    beat = np.zeros(1000)
    for center, width, amplitude in (
        (0.18, 0.035, 0.10),
        (0.375, 0.014, -0.12),
        (0.4, 0.016, 1.1),
        (0.43, 0.018, -0.24),
        (0.66, 0.075, 0.28),
    ):
        beat += amplitude * np.exp(-0.5 * np.square((phase - center) / width))
    lead_i, lead_ii = 0.75 * beat, beat
    original = np.stack(
        (
            lead_i,
            lead_ii,
            lead_ii - lead_i,
            -(lead_i + lead_ii) / 2,
            lead_i - lead_ii / 2,
            lead_ii - lead_i / 2,
            -0.4 * beat,
            -0.15 * beat,
            0.3 * beat,
            0.65 * beat,
            0.9 * beat,
            0.8 * beat,
        )
    )
    candidate = original.copy()
    candidate[6, 100:120] += 0.05
    return original, candidate


def _proposal(original: FloatArray, candidate: FloatArray) -> CounterfactualProposal:
    return CounterfactualProposal(
        original_waveform_sha256=canonical_waveform_sha256(original),
        candidate_waveform_sha256=canonical_waveform_sha256(candidate),
        target_superclass=TargetSuperclass.MI,
        target_direction=TargetDirection.INCREASE,
        method_version="synthetic-test-v1",
        method_artifact_sha256="b" * 64,
        seed=42,
    )


class _Runner:
    def __init__(self, release_id: str, *, base_score: float = 0.1) -> None:
        self.release_id = release_id
        self.base_score = base_score
        self.bound_manifest_sha256 = ""
        self.bound_checkpoint_sha256s: tuple[str, ...] = ()
        self.calls = 0
        self.after_infer: Callable[[], None] = lambda: None

    def bind(self, artifacts: SentinelModelArtifactInputs) -> _Runner:
        self.bound_manifest_sha256 = artifacts.manifest_sha256
        self.bound_checkpoint_sha256s = tuple(
            item.identity.unprefixed_sha256 for item in artifacts.checkpoints
        )
        return self

    def infer(self, signal_mv: FloatArray) -> SentinelModelEvidence:
        score = self.base_score + 0.04 * self.calls
        self.calls += 1
        self.after_infer()
        return SentinelModelEvidence(
            release_id=self.release_id,
            label_order=SUPERCLASSES,
            calibrated_probabilities=(0.9, score, 0.1, 0.1, 0.1),
            embedding=(0.1, -0.2),
            legacy_entropy_gate_accepted=True,
        )


class _Detector:
    threshold = 1.0

    def score(self, embeddings: object) -> FloatArray:
        return np.array([0.5])


def _engine(
    root: Path,
    *,
    changed_role: ArtifactRole | None = None,
    release_id: str = "release-vnext",
    base_score: float = 0.1,
) -> tuple[TrustSentinelEngine, _Runner, dict[ArtifactRole, Path]]:
    root.mkdir()
    paths = {role: root / f"{role.value.lower()}.bin" for role in ArtifactRole}
    for role, path in paths.items():
        path.write_bytes(
            f"{role.value}:{'changed' if role is changed_role else 'original'}".encode()
        )
    parents = tuple(
        sorted(
            (
                bind_parent_file(
                    root,
                    artifact_id=role.value.lower(),
                    role=role,
                    relative_path=path.name,
                    media_type="application/octet-stream",
                )
                for role, path in paths.items()
            ),
            key=lambda parent: (parent.role.value, parent.artifact_id),
        )
    )
    by_role = {parent.role: parent for parent in parents}
    compatibility = TrustBundleCompatibility.canonical()
    bundle = seal_trust_bundle(
        release_id=release_id,
        created_at=NOW,
        code_commit="a" * 40,
        protocol_sha256=by_role[ArtifactRole.PROTOCOL].file_sha256,
        dataset_manifest_sha256=by_role[ArtifactRole.DATASET_MANIFEST].file_sha256,
        environment_lock_sha256=by_role[ArtifactRole.ENVIRONMENT_LOCK].file_sha256,
        compatibility=compatibility,
        parents=parents,
    )
    verified = verify_trust_bundle(bundle, root, expected_compatibility=compatibility)
    conformal = LabelwiseBinaryConformal.from_dict(
        {
            "schema_version": 1,
            "artifact_type": "ecg_trust.labelwise_binary_conformal",
            "label_names": list(SUPERCLASSES),
            "alpha": 0.1,
            "thresholds": [0.25] * 5,
            "n_calibration_samples": 20,
            "quantile_rank": 19,
            "quantile_level": 0.95,
            "coverage_scope": "labelwise_marginal_under_exchangeability",
        }
    )
    runner = _Runner(release_id, base_score=base_score)
    engine = TrustSentinelEngine.from_verified_bundle(
        verified,
        loaders=SentinelRuntimeLoaders(
            model_runner=runner.bind,
            quality_policy=lambda _: DEFAULT_SIGNAL_QUALITY_CONFIG,
            decision_policy=lambda _: DEFAULT_TRUST_POLICY_CONFIG,
            distribution_policy=lambda _: LoadedDistributionPolicy(_Detector(), "test-ood-v1", 1),
            conformal_policy=lambda _: conformal,
        ),
    )
    return engine, runner, paths


def test_verified_release_runs_real_sentinel_and_preserves_legacy_result(tmp_path: Path) -> None:
    engine, runner, _ = _engine(tmp_path / "release")
    original, candidate = _signals()
    evaluator = VerifiedCounterfactualEvaluator(
        lambda: engine, expected_manifest_sha256=engine.bound_manifest_sha256
    )
    result = evaluator.evaluate(
        _proposal(original, candidate), original, candidate, evaluated_at=NOW
    )
    assert runner.calls == 2
    assert result.provenance_status is ReleaseProvenanceStatus.VERIFIED
    assert result.evaluation.status is CounterfactualStatus.ACCEPTED_MODEL_SENSITIVITY
    assert result.evaluation.model_sensitivity_effect == pytest.approx(0.04)
    public = result.to_public_summary()
    assert public["evaluation"] == result.evaluation.to_public_summary()
    serialized = result.canonical_public_json()
    assert "checkpoint_sha256s" in serialized and "conformal_policy_sha256" in serialized
    assert str(tmp_path) not in serialized and "waveform_sha256" not in serialized


@pytest.mark.parametrize(
    "changed_role",
    [
        ArtifactRole.CHECKPOINT,
        ArtifactRole.CONFORMAL_POLICY,
        ArtifactRole.DECISION_POLICY,
        ArtifactRole.NORMALIZATION,
        ArtifactRole.QUALITY_POLICY,
    ],
)
def test_switching_verified_artifacts_between_calls_fails_closed(
    tmp_path: Path, changed_role: ArtifactRole
) -> None:
    first, _, _ = _engine(tmp_path / "first")
    second, _, _ = _engine(tmp_path / "second", changed_role=changed_role, base_score=0.14)
    calls = 0

    def provider() -> TrustSentinelEngine:
        nonlocal calls
        calls += 1
        return first if calls <= 2 else second

    original, candidate = _signals()
    evaluator = VerifiedCounterfactualEvaluator(
        provider, expected_manifest_sha256=first.bound_manifest_sha256
    )
    result = evaluator.evaluate(
        _proposal(original, candidate), original, candidate, evaluated_at=NOW
    )
    assert result.provenance_status is ReleaseProvenanceStatus.UNAVAILABLE
    assert result.provenance_reasons == (ReleaseProvenanceReason.RELEASE_IDENTITY_CHANGED,)
    assert result.evaluation.status is CounterfactualStatus.EVIDENCE_UNAVAILABLE
    assert "model_sensitivity_effect" not in result.canonical_public_json()


def test_unbound_evaluator_can_accept_effect_across_different_releases(tmp_path: Path) -> None:
    first, _, _ = _engine(tmp_path / "first")
    second, _, _ = _engine(tmp_path / "second", release_id="release-other", base_score=0.14)
    original, candidate = _signals()
    outputs = [
        engine.analyze(
            signal_id="synthetic",
            signal_mv=signal,
            metadata=SignalMetadata.canonical(),
            evaluated_at=NOW,
            release_integrity_verified=True,
        )
        for engine, signal in ((first, original), (second, candidate))
    ]
    assert all(output.decision is TrustDecision.PREDICTION_ALLOWED for output in outputs)

    class UnboundAnalyzer:
        def analyze(self, signal_mv: FloatArray) -> SentinelAnalysis:
            output = outputs.pop(0)
            assert output.calibrated_probabilities is not None
            return SentinelAnalysis(
                decision=output.decision,
                reason_codes=output.policy.public_reason_codes,
                quality_status=QualityGateStatus.PASS,
                artifact_free=True,
                ood_supported=True,
                uncertainty_passed=True,
                superclass_scores=tuple(
                    zip(TargetSuperclass, output.calibrated_probabilities, strict=True)
                ),
            )

    legacy = CounterfactualEvaluator(UnboundAnalyzer()).evaluate(
        _proposal(original, candidate), original, candidate
    )
    assert legacy.status is CounterfactualStatus.ACCEPTED_MODEL_SENSITIVITY
    sequence = iter((first, first, second))
    verified = VerifiedCounterfactualEvaluator(
        lambda: next(sequence), expected_manifest_sha256=first.bound_manifest_sha256
    ).evaluate(_proposal(original, candidate), original, candidate, evaluated_at=NOW)
    assert verified.evaluation.status is CounterfactualStatus.EVIDENCE_UNAVAILABLE
    assert verified.provenance_reasons == (ReleaseProvenanceReason.RELEASE_IDENTITY_CHANGED,)


@pytest.mark.parametrize("role", [ArtifactRole.CHECKPOINT, ArtifactRole.CONFORMAL_POLICY])
def test_parent_tampering_during_inference_cannot_publish_effect(
    tmp_path: Path, role: ArtifactRole
) -> None:
    engine, runner, paths = _engine(tmp_path / "release")
    evaluator = VerifiedCounterfactualEvaluator(
        lambda: engine, expected_manifest_sha256=engine.bound_manifest_sha256
    )
    def tamper() -> None:
        paths[role].write_bytes(b"tampered")

    runner.after_infer = tamper
    original, candidate = _signals()
    result = evaluator.evaluate(
        _proposal(original, candidate), original, candidate, evaluated_at=NOW
    )
    assert result.evaluation.status is CounterfactualStatus.EVIDENCE_UNAVAILABLE
    assert ReleaseProvenanceReason.RUNTIME_IDENTITY_UNAVAILABLE in result.provenance_reasons
    assert "model_sensitivity_effect" not in result.canonical_public_json()


def test_loaded_checkpoint_identity_drift_is_rejected(tmp_path: Path) -> None:
    engine, runner, _ = _engine(tmp_path / "release")
    evaluator = VerifiedCounterfactualEvaluator(
        lambda: engine, expected_manifest_sha256=engine.bound_manifest_sha256
    )
    runner.bound_checkpoint_sha256s = ("0" * 64,)
    original, candidate = _signals()
    result = evaluator.evaluate(
        _proposal(original, candidate), original, candidate, evaluated_at=NOW
    )
    assert runner.calls == 0
    assert result.evaluation.status is CounterfactualStatus.EVIDENCE_UNAVAILABLE
    assert result.provenance_status is ReleaseProvenanceStatus.UNAVAILABLE


def test_invalid_signal_and_quality_failure_keep_gates_and_disclosure(tmp_path: Path) -> None:
    engine, runner, _ = _engine(tmp_path / "release")
    evaluator = VerifiedCounterfactualEvaluator(
        lambda: engine, expected_manifest_sha256=engine.bound_manifest_sha256
    )
    original, candidate = _signals()
    proposal = _proposal(original, candidate)
    result = evaluator.evaluate(proposal, original, candidate[:, :10], evaluated_at=NOW)
    assert result.provenance_status is ReleaseProvenanceStatus.NOT_EVALUATED
    assert runner.calls == 0
    flat = np.zeros_like(original)
    result = evaluator.evaluate(_proposal(flat, candidate), flat, candidate, evaluated_at=NOW)
    assert result.evaluation.status is CounterfactualStatus.EVIDENCE_UNAVAILABLE
    assert result.provenance_status is ReleaseProvenanceStatus.VERIFIED
    assert runner.calls == 1
    assert "model_sensitivity_effect" not in result.canonical_public_json()


def test_constructor_rejects_wrong_expected_release(tmp_path: Path) -> None:
    engine, _, _ = _engine(tmp_path / "release")
    with pytest.raises(CounterfactualValidationError, match="does not match"):
        VerifiedCounterfactualEvaluator(lambda: engine, expected_manifest_sha256="0" * 64)
