from __future__ import annotations

import tempfile
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from ecg_trust.conformal import LabelwiseBinaryConformal
from ecg_trust.constants import SUPERCLASSES
from ecg_trust.contracts import TrustDecision
from ecg_trust.quality.signal_quality import (
    DEFAULT_SIGNAL_QUALITY_CONFIG,
    SignalMetadata,
    SignalQualityConfig,
)
from ecg_trust.registry import (
    ArtifactRole,
    TrustBundleCompatibility,
    TrustBundleParent,
    bind_parent_file,
    seal_trust_bundle,
    verify_trust_bundle,
)
from ecg_trust.sentinel_engine import (
    LoadedDistributionPolicy,
    SentinelComponentUnavailable,
    SentinelModelArtifactInputs,
    SentinelModelEvidence,
    SentinelRuntimeLoaders,
    SentinelValidationError,
    TrustSentinelEngine,
)
from ecg_trust.trust_policy import DEFAULT_TRUST_POLICY_CONFIG

FloatArray = NDArray[np.float64]
NOW = datetime(2026, 8, 24, 12, 0, tzinfo=UTC)


def _beat(time: FloatArray) -> FloatArray:
    phase = np.mod(time, 1.0)

    def pulse(center: float, width: float, amplitude: float) -> FloatArray:
        return amplitude * np.exp(-0.5 * np.square((phase - center) / width))

    return (
        pulse(0.18, 0.035, 0.10)
        + pulse(0.375, 0.014, -0.12)
        + pulse(0.400, 0.016, 1.10)
        + pulse(0.430, 0.018, -0.24)
        + pulse(0.660, 0.075, 0.28)
    )


def _clean_signal() -> FloatArray:
    time = np.arange(1_000, dtype=np.float64) / 100.0
    beat = _beat(time)
    lead_i = 0.75 * beat
    lead_ii = beat
    return np.stack(
        (
            lead_i,
            lead_ii,
            lead_ii - lead_i,
            -(lead_i + lead_ii) / 2.0,
            lead_i - lead_ii / 2.0,
            lead_ii - lead_i / 2.0,
            -0.40 * beat,
            -0.15 * beat,
            0.30 * beat,
            0.65 * beat,
            0.90 * beat,
            0.80 * beat,
        )
    )


class FakeRunner:
    def __init__(
        self,
        *,
        probabilities: tuple[float, ...] = (0.9, 0.1, 0.1, 0.1, 0.1),
        entropy_accepted: bool = True,
        release_id: str = "release-vnext",
        fail: bool = False,
    ) -> None:
        self.probabilities = probabilities
        self.entropy_accepted = entropy_accepted
        self.release_id = release_id
        self.fail = fail
        self.calls = 0
        self.bound_manifest_sha256 = ""
        self.bound_checkpoint_sha256s: tuple[str, ...] = ()

    def bind(self, artifacts: SentinelModelArtifactInputs) -> FakeRunner:
        self.bound_manifest_sha256 = artifacts.manifest_sha256
        self.bound_checkpoint_sha256s = tuple(
            item.identity.unprefixed_sha256 for item in artifacts.checkpoints
        )
        return self

    def infer(self, signal_mv: FloatArray) -> SentinelModelEvidence:
        assert signal_mv.shape == (12, 1_000)
        self.calls += 1
        if self.fail:
            raise RuntimeError("C:\\private\\model.ckpt api_key=never-leak")
        return SentinelModelEvidence(
            release_id=self.release_id,
            label_order=SUPERCLASSES,
            calibrated_probabilities=self.probabilities,
            embedding=(0.1, -0.2),
            legacy_entropy_gate_accepted=self.entropy_accepted,
        )


class FakeDetector:
    def __init__(self, score: float, *, fail: bool = False) -> None:
        self.value = score
        self.threshold = 1.0
        self.fail = fail

    def score(self, embeddings: object) -> FloatArray:
        assert np.asarray(embeddings).shape == (1, 2)
        if self.fail:
            raise RuntimeError("detector failure")
        return np.asarray([self.value], dtype=np.float64)


def _conformal(threshold: float = 0.2) -> LabelwiseBinaryConformal:
    return LabelwiseBinaryConformal.from_dict(
        {
            "schema_version": 1,
            "artifact_type": "ecg_trust.labelwise_binary_conformal",
            "label_names": list(SUPERCLASSES),
            "alpha": 0.1,
            "thresholds": [threshold] * len(SUPERCLASSES),
            "n_calibration_samples": 20,
            "quantile_rank": 19,
            "quantile_level": 0.95,
            "coverage_scope": "labelwise_marginal_under_exchangeability",
        }
    )


def _engine(
    runner: FakeRunner,
    *,
    detector: FakeDetector | None = None,
    conformal: LabelwiseBinaryConformal | None = None,
    quality_config: SignalQualityConfig = DEFAULT_SIGNAL_QUALITY_CONFIG,
) -> TrustSentinelEngine:
    temporary = tempfile.TemporaryDirectory(prefix="sentinel-engine-test-")
    root = Path(temporary.name)
    declarations: list[tuple[str, ArtifactRole, str]] = []
    for role in ArtifactRole:
        artifact_id = "checkpoint-0" if role is ArtifactRole.CHECKPOINT else role.value.lower()
        filename = f"{artifact_id}.bin"
        (root / filename).write_bytes(f"{role.value}:{artifact_id}\n".encode())
        declarations.append((artifact_id, role, filename))
    parents = tuple(
        sorted(
            (
                bind_parent_file(
                    root,
                    artifact_id=artifact_id,
                    role=role,
                    relative_path=filename,
                    media_type="application/octet-stream",
                )
                for artifact_id, role, filename in declarations
            ),
            key=lambda parent: (parent.role.value, parent.artifact_id),
        )
    )
    singleton: dict[ArtifactRole, TrustBundleParent] = {
        parent.role: parent for parent in parents if parent.role is not ArtifactRole.CHECKPOINT
    }
    compatibility = TrustBundleCompatibility.canonical()
    bundle = seal_trust_bundle(
        release_id="release-vnext",
        created_at=NOW,
        code_commit="a" * 40,
        protocol_sha256=singleton[ArtifactRole.PROTOCOL].file_sha256,
        dataset_manifest_sha256=singleton[ArtifactRole.DATASET_MANIFEST].file_sha256,
        environment_lock_sha256=singleton[ArtifactRole.ENVIRONMENT_LOCK].file_sha256,
        compatibility=compatibility,
        parents=parents,
    )
    verified = verify_trust_bundle(bundle, root, expected_compatibility=compatibility)
    engine = TrustSentinelEngine.from_verified_bundle(
        verified,
        loaders=SentinelRuntimeLoaders(
            model_runner=runner.bind,
            quality_policy=lambda _: quality_config,
            decision_policy=lambda _: DEFAULT_TRUST_POLICY_CONFIG,
            distribution_policy=lambda _: LoadedDistributionPolicy(
                detector=detector,
                method="shrinkage-mahalanobis-v1",
                schema_version=1,
            ),
            conformal_policy=lambda _: conformal,
        ),
    )
    engine._test_temporary_directory = temporary  # type: ignore[attr-defined]
    return engine


def _analyze(engine: TrustSentinelEngine, **overrides: object):
    arguments: dict[str, object] = {
        "signal_id": "case-001",
        "signal_mv": _clean_signal(),
        "metadata": SignalMetadata.canonical(),
        "evaluated_at": NOW,
        "release_integrity_verified": True,
    }
    arguments.update(overrides)
    return engine.analyze(**arguments)  # type: ignore[arg-type]


def test_all_gates_pass_and_expose_only_calibrated_results() -> None:
    runner = FakeRunner()
    result = _analyze(_engine(runner, detector=FakeDetector(0.5), conformal=_conformal()))

    assert result.decision is TrustDecision.PREDICTION_ALLOWED
    assert result.predictions_exposed
    assert runner.calls == 1
    public = result.to_public_dict()
    assert public["probabilities"] == {
        "NORM": 0.9,
        "MI": 0.1,
        "STTC": 0.1,
        "CD": 0.1,
        "HYP": 0.1,
    }
    assert len(public["prediction_sets"]) == 5  # type: ignore[arg-type]
    assert "embedding" not in repr(public).lower()


def test_engine_uses_the_injected_release_quality_policy() -> None:
    runner = FakeRunner()
    stricter_quality = replace(
        DEFAULT_SIGNAL_QUALITY_CONFIG,
        version="strict-amplitude-v1",
        amplitude_warn_mv=0.1,
        amplitude_reacquire_mv=0.2,
    )

    result = _analyze(
        _engine(
            runner,
            detector=FakeDetector(0.5),
            conformal=_conformal(),
            quality_config=stricter_quality,
        )
    )

    assert result.decision is TrustDecision.REACQUIRE
    assert runner.calls == 0


def test_engine_rejects_model_identity_substitution_during_assembly() -> None:
    class SubstitutedRunner(FakeRunner):
        def bind(self, artifacts: SentinelModelArtifactInputs) -> FakeRunner:
            super().bind(artifacts)
            self.bound_manifest_sha256 = "f" * 64
            return self

    with pytest.raises(SentinelValidationError, match="assembly"):
        _engine(
            SubstitutedRunner(),
            detector=FakeDetector(0.5),
            conformal=_conformal(),
        )


@pytest.mark.parametrize(
    ("signal", "metadata", "expected"),
    [
        (np.zeros((11, 1_000)), SignalMetadata.canonical(), TrustDecision.INVALID_INPUT),
        (
            _clean_signal(),
            replace(SignalMetadata.canonical(), sample_rate_hz=500.0),
            TrustDecision.INVALID_INPUT,
        ),
        (np.zeros((12, 1_000)), SignalMetadata.canonical(), TrustDecision.REACQUIRE),
    ],
)
def test_input_and_quality_failures_skip_model_and_hide_results(
    signal: FloatArray,
    metadata: SignalMetadata,
    expected: TrustDecision,
) -> None:
    runner = FakeRunner()
    result = _analyze(
        _engine(runner, detector=FakeDetector(0.5), conformal=_conformal()),
        signal_mv=signal,
        metadata=metadata,
    )

    assert result.decision is expected
    assert runner.calls == 0
    assert result.calibrated_probabilities is None
    assert "probabilities" not in result.to_public_dict()


def test_unverified_release_stops_before_model() -> None:
    runner = FakeRunner()
    result = _analyze(
        _engine(runner, detector=FakeDetector(0.5), conformal=_conformal()),
        release_integrity_verified=False,
    )

    assert result.decision is TrustDecision.INVALID_INPUT
    assert runner.calls == 0
    assert "RELEASE_INTEGRITY_UNVERIFIED" in repr(result.to_public_dict())


@pytest.mark.parametrize(
    "filename",
    [
        "checkpoint-0.bin",
        "quality_policy.bin",
        "decision_policy.bin",
        "distribution_policy.bin",
        "conformal_policy.bin",
    ],
)
def test_bound_parent_tampering_revokes_readiness_and_prediction(
    filename: str,
) -> None:
    runner = FakeRunner()
    engine = _engine(runner, detector=FakeDetector(0.5), conformal=_conformal())
    temporary = engine._test_temporary_directory  # type: ignore[attr-defined]
    parent = Path(temporary.name) / filename
    original = parent.read_bytes()

    assert engine.is_ready()
    parent.write_bytes(b"X" * len(original))

    assert not engine.is_ready()
    result = _analyze(engine)
    assert result.decision is TrustDecision.INVALID_INPUT
    assert result.calibrated_probabilities is None
    assert runner.calls == 0
    assert str(parent) not in repr(result.to_public_dict())


@pytest.mark.parametrize("detector", [FakeDetector(2.0), None, FakeDetector(0.0, fail=True)])
def test_ood_or_unavailable_detector_is_unsupported_and_hides_predictions(
    detector: FakeDetector | None,
) -> None:
    result = _analyze(_engine(FakeRunner(), detector=detector, conformal=_conformal()))

    assert result.decision is TrustDecision.UNSUPPORTED_INPUT
    assert not result.predictions_exposed
    assert "probabilities" not in result.to_public_dict()


def test_legacy_entropy_rejection_precedes_conformal() -> None:
    result = _analyze(
        _engine(
            FakeRunner(entropy_accepted=False),
            detector=FakeDetector(0.5),
            conformal=_conformal(),
        )
    )

    assert result.decision is TrustDecision.ABSTAIN
    assert result.label_prediction_sets is None
    assert "probabilities" not in result.to_public_dict()


def test_conformal_uncertainty_or_missing_artifact_abstains_without_label_leakage() -> None:
    uncertain = _analyze(
        _engine(FakeRunner(), detector=FakeDetector(0.5), conformal=_conformal(0.95))
    )
    missing = _analyze(_engine(FakeRunner(), detector=FakeDetector(0.5), conformal=None))

    for result in (uncertain, missing):
        assert result.decision is TrustDecision.ABSTAIN
        assert result.label_prediction_sets is None
        public = result.to_public_dict()
        assert "probabilities" not in public
        assert "prediction_sets" not in public
        assert all(label not in repr(public) for label in SUPERCLASSES)


def test_model_failure_and_release_mismatch_fail_without_private_details() -> None:
    for runner in (FakeRunner(fail=True), FakeRunner(release_id="wrong-release")):
        with pytest.raises(SentinelComponentUnavailable) as caught:
            _analyze(_engine(runner, detector=FakeDetector(0.5), conformal=_conformal()))
        assert "private" not in str(caught.value).lower()
        assert "api_key" not in str(caught.value).lower()


def test_engine_readiness_requires_distribution_and_conformal_components() -> None:
    assert _engine(FakeRunner(), detector=FakeDetector(0.5), conformal=_conformal()).is_ready()
    assert not _engine(FakeRunner(), detector=None, conformal=_conformal()).is_ready()
    assert not _engine(FakeRunner(), detector=FakeDetector(0.5), conformal=None).is_ready()
