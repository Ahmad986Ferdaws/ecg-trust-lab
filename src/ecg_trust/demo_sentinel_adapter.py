"""Adapter from a verified demo release to Sentinel model evidence."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ecg_trust.demo_backend import DemoInferenceBackend
from ecg_trust.sentinel_engine import SentinelModelArtifactInputs, SentinelModelEvidence


@dataclass(frozen=True, slots=True)
class DemoSentinelModelRunner:
    """Run frozen calibrated prediction and representation extraction together."""

    backend: DemoInferenceBackend
    release_id: str
    bound_manifest_sha256: str
    bound_checkpoint_sha256s: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.release_id.strip():
            raise ValueError("release_id must be non-empty")
        if len(self.bound_manifest_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.bound_manifest_sha256
        ):
            raise ValueError("bound_manifest_sha256 must be a lowercase SHA-256 digest")
        if not self.bound_checkpoint_sha256s or any(
            len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest)
            for digest in self.bound_checkpoint_sha256s
        ):
            raise ValueError("bound checkpoint identities must be lowercase SHA-256 digests")

    @classmethod
    def load_from_verified_artifacts(
        cls,
        artifacts: SentinelModelArtifactInputs,
    ) -> DemoSentinelModelRunner:
        """Load the legacy frozen backend only from the bound runtime artifact set."""

        if len(artifacts.checkpoints) != 1:
            raise ValueError("the demo Sentinel runner requires exactly one checkpoint")
        checkpoint = artifacts.checkpoints[0]
        try:
            backend = DemoInferenceBackend.load(
                checkpoint_path=checkpoint.private_path,
                resolved_config_path=artifacts.resolved_config.private_path,
                normalization_path=artifacts.normalization.private_path,
                decision_policy_path=artifacts.decision_policy.private_path,
            )
        except Exception:
            raise ValueError("verified demo model loading failed") from None
        provenance = backend.artifact_provenance
        observed = {
            "checkpoint_sha256": checkpoint.identity.unprefixed_sha256,
            "resolved_config_sha256": artifacts.resolved_config.identity.unprefixed_sha256,
            "normalization_sha256": artifacts.normalization.identity.unprefixed_sha256,
            "decision_policy_sha256": artifacts.decision_policy.identity.unprefixed_sha256,
        }
        if any(provenance.get(name) != digest for name, digest in observed.items()):
            raise ValueError("loaded demo artifacts do not match the verified runtime parents")
        return cls(
            backend=backend,
            release_id=artifacts.release_id,
            bound_manifest_sha256=artifacts.manifest_sha256,
            bound_checkpoint_sha256s=(observed["checkpoint_sha256"],),
        )

    def infer(self, signal_mv: NDArray[np.float64]) -> SentinelModelEvidence:
        prediction = self.backend.predict_signal(signal_mv)
        embedding = self.backend.extract_embedding_signal(signal_mv)
        return SentinelModelEvidence(
            release_id=self.release_id,
            label_order=prediction.label_order,
            calibrated_probabilities=tuple(
                float(value) for value in prediction.calibrated_probabilities.tolist()
            ),
            embedding=tuple(float(value) for value in embedding.tolist()),
            legacy_entropy_gate_accepted=prediction.decision == "accept",
        )


__all__ = ["DemoSentinelModelRunner"]
